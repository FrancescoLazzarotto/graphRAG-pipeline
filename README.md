
# GraphRAG Pipeline

A simple GraphRAG framework that combines Neo4j graph retrieval with an LLM-based answer pipeline.

## What this project does

- Connects to a Neo4j knowledge graph.
- Retrieves nodes, triples, neighbors, subgraphs, and shortest paths.
- Builds context from graph results.
- Generates answers with a local Hugging Face model.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Create a local `.env` file manually (not committed) with:

```env
NEO4J_URL=neo4j+s://<your-instance>
NEO4J_USERNAME=<your-username>
NEO4J_PASSWORD=<your-password>
NEO4J_DATABASE=<your-database>
```

For cluster jobs, you can export the same variables from the scheduler/job script.

Then run:

```powershell
graphrag-demo --seed-movie-dataset
```

Enable local LLM generation:

```powershell
graphrag-demo --llm --model-id Qwen/Qwen2.5-7B-Instruct
```

Production-oriented tuning for larger open models:

```bash
graphrag-demo \
	--llm \
	--model-id Qwen/Qwen2.5-14B-Instruct \
	--max-new-tokens 128 \
	--gpu-memory-fraction 0.90
```

Notes:

- `--max-new-tokens` reduces generation cost without changing model size/capability.
- `--gpu-memory-fraction` reserves headroom to reduce OOM/termination during loading.
- For large models (>=30B), fp16 fallback is disabled by default if 4-bit loading fails (safer for production). Enable only if needed with `--allow-large-model-fp16-fallback`.
- Optional env toggle for batch jobs: `GRAPHRAG_ALLOW_LARGE_MODEL_FP16_FALLBACK=1`.

If you use gated Hugging Face models (for example `meta-llama/*`), request access on the model page and authenticate first:

```bash
conda activate graphllm
export HF_TOKEN="<your-hf-token>"
# Run directly with env token (no login required):
python -m graphrag.cli --llm --model-id meta-llama/Llama-3.1-70B

# Optional: persist token in Hugging Face cache for this user:
$CONDA_PREFIX/bin/python -m huggingface_hub.commands.huggingface_cli login --token "$HF_TOKEN"
```

You can also provide a token via environment variable:

```bash
export HF_TOKEN="<your-hf-token>"
```

Note: in some environments, `hf` and `huggingface-cli` can point to stale user-local shims. Prefer the env-pinned `python -m ...` command above.

## Cluster setup

- CPU nodes: install [requirements-cpu.txt](requirements-cpu.txt)
- GPU nodes: install [requirements-gpu.txt](requirements-gpu.txt)
- GPU job template: [scripts/run_graphrag.sbatch](scripts/run_graphrag.sbatch)
- CPU job template: [scripts/run_graphrag_cpu.sbatch](scripts/run_graphrag_cpu.sbatch)
- Cluster guide: [docs/cluster.md](docs/cluster.md)

Quick smoke check after installation:

```bash
python scripts/smoke_check.py
```

Local preflight (no cluster access yet):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/preflight.ps1
```

With Neo4j connectivity check:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/preflight.ps1 -CheckNeo4j
```

Immediate cluster run examples:

```bash
export NEO4J_URL="neo4j+s://<your-instance>"
export NEO4J_USERNAME="<your-username>"
export NEO4J_PASSWORD="<your-password>"
export NEO4J_DATABASE="<your-database>"

# GPU
sbatch -p <gpu_partition> scripts/run_graphrag.sbatch

# CPU
sbatch -p <cpu_partition> scripts/run_graphrag_cpu.sbatch
```

## Standard Text RAG Pipeline (documents)

The repository includes a base text-only retrieval pipeline for document RAG:

- PDF ingestion via PyMuPDF
- Text chunking with overlap
- In-memory retrieval scoring

Quick smoke test on local documents:

```bash
python scripts/smoke_text_rag.py docs/ --query "Summarize the cluster setup" --top-k 4
```

Smoke test on a specific PDF + markdown:

```bash
python scripts/smoke_text_rag.py /path/to/file.pdf README.md --query "What are the prerequisites?"
```

Programmatic entrypoint:

- `graphrag.text_rag.pipeline.StandardTextRAGPipeline`

### Combined Standard RAG + GraphRAG Matrix

You can run a single matrix that includes:

- standard text RAG strategies
- GraphRAG strategies (`default`, `text_only`, `text_plus_triples`, `neighbors_focus`, `subgraph_2hop`, `shortest_path`)

Smoke-first run (small test):

```bash
python scripts/run_retrieval_matrix.py \
	--smoke \
	--questions-file artifacts/experiments/questions_smoke.txt \
	--documents docs/ README.md \
	--runs-per-strategy 1 \
	--output-dir artifacts/experiments \
	--experiment-tag retrieval_matrix_smoke
```

Larger run (all strategies):

```bash
python scripts/run_retrieval_matrix.py \
	--questions-file questions.txt \
	--documents /path/to/docs /path/to/pdfs \
	--runs-per-strategy 3 \
	--output-dir artifacts/experiments \
	--experiment-tag retrieval_matrix_full
```

Resource telemetry is enabled by default in `run_retrieval_matrix.py` and records CPU/RAM/GPU samples.

Useful flags:

- `--resource-sample-interval 1.0` to control sampling period (seconds)
- `--no-monitor-resources` to disable telemetry

## Main entrypoint

- CLI: `graphrag-demo`

## Batch experiments

You can run multiple questions across multiple strategy presets and persist all results.

Create a question file (one question per line):

```bash
cat > questions.txt << 'EOF'
Chi ha diretto The Matrix?
Chi ha recitato in The Matrix?
EOF
```

Run experiments:

```bash
graphrag-demo \
	--experiment \
	--questions-file questions.txt \
	--strategies default,text_only,text_plus_triples,neighbors_focus,subgraph_2hop,shortest_path \
	--runs-per-strategy 3 \
	--output-dir artifacts/experiments \
	--experiment-tag matrix_benchmark \
	--llm --model-id Qwen/Qwen2.5-3B-Instruct
```

Available strategy presets:

- `default`: all retrieval sources enabled
- `text_only`: baseline without graph retrieval (LLM-only context)
- `text_plus_triples`: text + matched triples
- `neighbors_focus`: text + local neighborhood
- `subgraph_2hop`: text + 2-hop subgraph
- `shortest_path`: text + shortest-path context

Produced artifacts:

- `results.jsonl`: one record per execution
- `results.csv`: tabular format for spreadsheet/pandas analysis
- `summary.txt`: quick human-readable overview
- `summary.json`: structured aggregate metrics per strategy
- `resource_samples.jsonl`: time-series samples for CPU/RAM/GPU usage
- `resource_summary.json`: aggregate peaks/averages for sizing comparisons

Analyze a run directory:

```bash
python scripts/analyze_experiments.py artifacts/experiments/<run_folder>
```

Save aggregated metrics to JSON:

```bash
python scripts/analyze_experiments.py artifacts/experiments/<run_folder> --save-json artifacts/experiments/<run_folder>/analysis.json
```

Aggregate resource telemetry across many runs (for multi-LLM sizing):

```bash
python scripts/analyze_resource_usage.py artifacts/experiments \
	--tag-contains retrieval_matrix \
	--save-json artifacts/experiments/resource_usage_summary.json \
	--save-csv artifacts/experiments/resource_usage_summary.csv
```

## Long Cluster Matrix Benchmark (2x A40)

Use the dedicated matrix script to sweep multiple models and strategies.

Set environment first:

```bash
export NEO4J_URL="neo4j+s://<your-instance>"
export NEO4J_USERNAME="<your-username>"
export NEO4J_PASSWORD="<your-password>"
export NEO4J_DATABASE="<your-database>"
```

Submit as SLURM array with max concurrency 2 (to use two A40 in parallel):

```bash
MODELS_CSV="Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct" \
STRATEGIES="default,text_only,text_plus_triples,neighbors_focus,subgraph_2hop,shortest_path" \
RUNS_PER_STRATEGY=3 \
QUESTIONS_FILE=questions.txt \
CONDA_ENV=graphllm \
sbatch -p <gpu_partition> --array=0-2%2 scripts/run_experiment_matrix_gpu.sbatch
```

Monitor jobs:

```bash
squeue -u $USER
tail -f logs/graphrag-matrix-<jobid>_<taskid>.out
```

Aggregate multiple runs (all folders in `artifacts/experiments`):

```bash
python scripts/analyze_matrix.py artifacts/experiments \
	--tag-contains matrix_long \
	--save-json artifacts/experiments/matrix_summary.json \
	--save-csv artifacts/experiments/matrix_summary.csv
```

# GraphRAG Pipeline

A simple GraphRAG project that combines Neo4j graph retrieval with an LLM-based answer pipeline.

## What this project does

- Connects to a Neo4j knowledge graph.
- Retrieves nodes, triples, neighbors, subgraphs, and shortest paths.
- Builds context from graph results.
- Optionally generates answers with a local Hugging Face model.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Create a local `.env` file manually (not committed) with:

```env
NEO4J_URL=neo4j+s://<your-instance>
NEO4J_USERNAME=<your-username>
NEO4J_PASSWORD=<your-password>
NEO4J_DATABASE=<your-database>
```

For cluster jobs, you can export the same variables from the scheduler/job script.

Then run:

```powershell
graphrag-demo --seed-movie-dataset
```

Enable local LLM generation:

```powershell
graphrag-demo --llm --model-id Qwen/Qwen2.5-7B-Instruct
```

## Cluster setup

- CPU nodes: install [requirements-cpu.txt](requirements-cpu.txt)
- GPU nodes: install [requirements-gpu.txt](requirements-gpu.txt)
- GPU job template: [scripts/run_graphrag.sbatch](scripts/run_graphrag.sbatch)
- CPU job template: [scripts/run_graphrag_cpu.sbatch](scripts/run_graphrag_cpu.sbatch)
- Cluster guide: [docs/cluster.md](docs/cluster.md)

Quick smoke check after installation:

```bash
python scripts/smoke_check.py
```

Local preflight (no cluster access yet):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/preflight.ps1
```

With Neo4j connectivity check:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/preflight.ps1 -CheckNeo4j
```

Immediate cluster run examples:

```bash
export NEO4J_URL="neo4j+s://<your-instance>"
export NEO4J_USERNAME="<your-username>"
export NEO4J_PASSWORD="<your-password>"
export NEO4J_DATABASE="<your-database>"

# GPU
sbatch -p <gpu_partition> scripts/run_graphrag.sbatch

# CPU
sbatch -p <cpu_partition> scripts/run_graphrag_cpu.sbatch
```

## Main entrypoint

- CLI: `graphrag-demo`

