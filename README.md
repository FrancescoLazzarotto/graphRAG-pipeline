
# GraphRAG Pipeline

An experiment-oriented RAG (Retrieval-Augmented Generation) pipeline combining graph retrieval (Neo4j) with LLM-based generation.

[![CI](https://github.com/FrancescoLazzarotto/graphRAG-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/FrancescoLazzarotto/graphRAG-pipeline/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Conda env: graphllm](https://img.shields.io/badge/conda-graphllm-44A833.svg?logo=anaconda)](https://docs.conda.io/)

Status: active development (May 2026)

- Core functionality: knowledge graph construction from documents, hybrid GraphRAG + text-RAG retrieval strategies, LLM-backed answer generation (local or vLLM server), and experiment/matrix orchestration.
- Primary entry points: CLI `graphrag-demo` and KG pipeline runner `python -m kg_pipeline.main`.
- Recommended environment: Conda environment named `graphllm` (see `requirements-*.txt`).

Table of Contents

- Project overview
- Quick start (Conda)
- Configuration (Neo4j, HF token, vLLM)
- Usage examples (CLI, vLLM, model tuning)
- Knowledge Graph pipeline (kg_pipeline)
- Experiments & matrices
- Analysis & telemetry
- Smoke tests & preflight
- Repository structure
- Contributing & license

Project overview

This repository implements a pipeline that:

- Ingests documents (PDF/Markdown/text), chunks and extracts entities and triples (NER + LLM extraction).
- Resolves and links entities/triples to build a Knowledge Graph and ingest into Neo4j.
- Provides a GraphRAG retriever that can return nodes, triples, local neighborhoods, 2-hop subgraphs, and shortest paths.
- Constructs LLM prompt context from retrieved graph/text and can generate answers using either local HF models or an OpenAI-compatible vLLM server.
- Runs reproducible experiment matrices to compare retrieval strategies and LLMs, and records resource telemetry for sizing studies.

End-to-end flow

1. Ingest documents from `docs/`, PDFs, or your own folder.
2. Chunk the text and extract entities, relations, and candidate triples.
3. Resolve, link, and ingest the graph into Neo4j.
4. Retrieve graph and text evidence with a chosen strategy.
5. Build prompt context and generate an answer with a local model or a vLLM server.
6. Save results, summaries, and resource telemetry for later analysis.

Quick start (Conda)

1. Create and activate the Conda environment (recommended name `graphllm`):

```bash
conda create -n graphllm python=3.10 -y
conda activate graphllm
```

2. Install dependencies (pick the right set for your node):

```bash
pip install -r requirements.txt        # base deps
pip install -r requirements-cpu.txt    # CPU-only nodes
pip install -r requirements-gpu.txt    # GPU nodes (if available)
pip install -e .
```

Configuration

Set required Neo4j environment variables (either export or place in `.env` files):

```bash
export NEO4J_URL="neo4j+s://<your-instance>"
export NEO4J_USERNAME="<user>"
export NEO4J_PASSWORD="<pass>"
export NEO4J_DATABASE="<db>"  # optional
```

If you use gated Hugging Face models, set `HF_TOKEN` in the environment:

```bash
export HF_TOKEN="<your-hf-token>"
```

vLLM / OpenAI-compatible server options:

- `VLLM_BASE_URL` (default: `http://localhost:8000/v1`)
- `VLLM_MODEL_NAME` and `VLLM_API_KEY` / `OPENAI_API_KEY` where applicable

Usage examples

- Run single-question demo (GraphRAG retrieval + generation):

```bash
graphrag-demo --question "What are the relations between Entity A and Entity B?" --entity "Entity A"
```

- Use local LLM generation:

```bash
graphrag-demo --llm --model-id Qwen/Qwen2.5-7B-Instruct
```

- Use server-backed (vLLM/OpenAI-compatible) generation:

```bash
graphrag-demo --llm --vllm --vllm-base-url http://localhost:8000/v1 --model-id Qwen/Qwen2.5-7B-Instruct
```

Model tuning examples (reduce generation cost or control GPU memory usage):

```bash
graphrag-demo --llm --model-id Qwen/Qwen2.5-14B-Instruct --max-new-tokens 128 --gpu-memory-fraction 0.90
```

Notes:

- `--max-new-tokens` reduces generation length/cost.
- `--gpu-memory-fraction` reserves headroom when loading large local models to reduce OOMs.
- For very large models (>= 30B) fp16 fallback is disabled by default; enable `--allow-large-model-fp16-fallback` only if you understand the memory/precision tradeoffs.

Knowledge Graph pipeline

The KG pipeline lives in `kg_pipeline/` and outputs checkpointed stage artifacts into a run directory. Defaults are controlled by `kg_pipeline/config.yaml`.

Run the full pipeline (logs to stdout):

```bash
conda activate graphllm
PYTHONUNBUFFERED=1 python -m kg_pipeline.main --config kg_pipeline/config.yaml --env-file kg_pipeline/.env --log-level INFO
```

You can resume an existing run by reusing the same `--run-dir`. To run a single stage set `--stage` to one of: `ingestion|chunking|ner|llm|resolution|linking|neo4j`.

To skip Neo4j ingestion (dry run) use `--dry-run`.

Typical pipeline outputs include:

- `pipeline.log` for the full execution trace.
- Stage artifacts under the selected run directory.
- Neo4j ingest logs when graph loading is enabled.

Example run directory shape:

```text
kg_pipeline/artifacts/run_YYYYMMDD_HHMMSS/
├── pipeline.log
├── chunking/
├── ner/
├── llm/
├── resolution/
├── linking/
└── neo4j/
```

Experiments & matrices

The repository includes scripts to run retrieval matrices that compare multiple strategies and LLMs. Strategies include:

- `default` (all sources), `text_only`, `text_plus_triples`, `neighbors_focus`, `subgraph_2hop`, `shortest_path`.

Smoke example (small experiment):

```bash
python scripts/run_retrieval_matrix.py --smoke --questions-file artifacts/experiments/questions_smoke.txt --documents docs/ README.md --runs-per-strategy 1 --output-dir artifacts/experiments --experiment-tag retrieval_matrix_smoke
```

vLLM-backed matrix example:

```bash
python scripts/run_retrieval_matrix.py --llm --vllm --vllm-base-url http://localhost:8000/v1 --model-id Qwen/Qwen2.5-32B-Instruct --questions-file questions_matrix_long.txt --graph-strategies default --runs-per-strategy 1
```

Analysis & telemetry

- Outputs per run include `results.jsonl`, `results.csv`, `summary.txt`, `summary.json`, `resource_samples.jsonl`, and `resource_summary.json`.
- Use `scripts/analyze_experiments.py` to analyze a single run directory.
- Use `scripts/analyze_matrix.py` and `scripts/analyze_resource_usage.py` to aggregate multiple runs and produce CSV/JSON summaries for sizing and comparison.

Expected experiment artifacts:

```text
artifacts/experiments/<run_name>/
├── results.jsonl
├── results.csv
├── summary.txt
├── summary.json
├── resource_samples.jsonl
└── resource_summary.json
```

What to expect in practice:

- `results.jsonl` contains one record per question/strategy/run combination.
- `summary.json` stores structured statistics per strategy and configuration.
- `summary.txt` is the fast human-readable check for a run.
- `resource_summary.json` reports peak and average resource usage when telemetry is enabled.

Concrete verification example from the repository’s recent runs:

- A verification report in [artifacts/experiments/20260514_170536_test_strategies_verification/REPORT.txt](artifacts/experiments/20260514_170536_test_strategies_verification/REPORT.txt) documents 60 runs, with `results.jsonl` as the raw trace and `summary.json` as the structured summary.
- That same report includes sample answer inspection commands such as `python3 show_samples.py results.jsonl default 2`.

Evaluation workflows

The repository also includes a separate evaluation workspace under [evaluation/README.md](evaluation/README.md) for paper-oriented comparisons.

Use it when you want to:

- build a gold QA dataset from run outputs and manual labels,
- compute retrieval-oriented metrics such as entity coverage and rank-based scores,
- optionally run RAGAS with a local judge model,
- generate summary tables that can be copied into a paper or internal report.

Typical evaluation sequence:

1. Prepare `evaluation/gold_questions_template.csv` with your labels.
2. Join a run output with the gold set using `evaluation/build_eval_dataset.py`.
3. Compute retrieval metrics with `evaluation/retrieval_metrics.py`.
4. Optionally run `evaluation/run_ragas_eval.py` for answer-quality metrics.

Smoke tests & preflight

- Quick smoke check after install: `python scripts/smoke_check.py`
- Local preflight helper (PowerShell): `powershell -ExecutionPolicy Bypass -File scripts/preflight.ps1`
- If you only want a fast sanity check before a long run, start with the smoke matrix command above and verify that `summary.json` and `results.jsonl` are created in the output directory.

Cluster & batch jobs

- CPU nodes: install `requirements-cpu.txt`
- GPU nodes: install `requirements-gpu.txt`
- GPU job template: `scripts/run_graphrag.sbatch`
- CPU job template: `scripts/run_graphrag_cpu.sbatch`

Examples (SLURM):

```bash
export NEO4J_URL="neo4j+s://<your-instance>"
export NEO4J_USERNAME="<user>"
export NEO4J_PASSWORD="<pass>"
export NEO4J_DATABASE="<db>"

# submit GPU job
sbatch -p <gpu_partition> scripts/run_graphrag.sbatch

# submit CPU job
sbatch -p <cpu_partition> scripts/run_graphrag_cpu.sbatch
```

Repository structure (short)

- `kg_pipeline/` — knowledge graph pipeline
- `src/graphrag/` — main package, CLI and agent code
- `scripts/` — utilities, smoke tests, batch templates
- `docs/` — additional documentation (eg. cluster.md)
- `artifacts/` — experiment outputs, logs and reports

Contributing

- Open issues or pull requests for bugs and improvements.
- Run smoke tests and minimal experiments locally before submitting PRs.
- Keep changes small and reproducible; if you touch retrieval logic, include a short note about the affected strategy or output artifact.

Troubleshooting

- If the CLI cannot connect to Neo4j, verify `NEO4J_URL`/`NEO4J_USERNAME`/`NEO4J_PASSWORD` and make sure the database name is correct.
- If local model loading fails, try a smaller model first or reduce `--max-new-tokens` and review GPU memory settings.
- If a vLLM run does not generate answers, confirm the server URL and model name match the server process.
- If experiment runs complete but produce no useful context, inspect the generated `summary.json` and `results.jsonl` before changing the pipeline.



