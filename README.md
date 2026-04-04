
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

Analyze a run directory:

```bash
python scripts/analyze_experiments.py artifacts/experiments/<run_folder>
```

Save aggregated metrics to JSON:

```bash
python scripts/analyze_experiments.py artifacts/experiments/<run_folder> --save-json artifacts/experiments/<run_folder>/analysis.json
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

