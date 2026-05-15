<div align="center">

#  GraphRAG Pipeline

**An experiment-oriented Retrieval-Augmented Generation pipeline combining Knowledge Graph retrieval with LLM-based answer generation.**

[![CI](https://github.com/FrancescoLazzarotto/graphRAG-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/FrancescoLazzarotto/graphRAG-pipeline/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-active%20development-brightgreen)](https://github.com/FrancescoLazzarotto/graphRAG-pipeline)
[![Neo4j](https://img.shields.io/badge/Neo4j-Knowledge%20Graph-008CC1?logo=neo4j)](https://neo4j.com/)

*Active development — May 2026*

</div>

---

##  Table of Contents

- [Overview](#-overview)
- [End-to-End Flow](#-end-to-end-flow)
- [Quick Start](#-quick-start-conda)
- [Configuration](#-configuration)
- [Usage Examples](#-usage-examples)
- [Knowledge Graph Pipeline](#-knowledge-graph-pipeline)
- [Experiments & Matrices](#-experiments--matrices)
- [Analysis & Telemetry](#-analysis--telemetry)
- [Evaluation Workflow](#-evaluation-workflow)
- [Smoke Tests & Preflight](#-smoke-tests--preflight)
- [Cluster & Batch Jobs](#-cluster--batch-jobs)
- [Repository Structure](#-repository-structure)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

---

##  Overview

This repository implements a full GraphRAG pipeline that:

-  **Ingests** documents (PDF, Markdown, plain text), chunks them, and extracts entities and triples via NER + LLM-based extraction.
-  **Builds a Knowledge Graph** by resolving and linking entities/triples, then ingesting into Neo4j.
-  **Retrieves** using a GraphRAG retriever supporting nodes, triples, local neighborhoods, 2-hop subgraphs, and shortest paths.
-  **Generates answers** by constructing LLM prompt context from retrieved graph/text evidence, using either local HuggingFace models or an OpenAI-compatible vLLM server.
-  **Runs reproducible experiment matrices** to compare retrieval strategies and LLMs, with full resource telemetry for sizing studies.

Primary entry points:
- CLI: `graphrag-demo`
- KG pipeline runner: `python -m kg_pipeline.main`

---

##  End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  1. INGEST      →  docs/, PDFs, or custom folder                   │
│  2. CHUNK       →  split text into manageable segments             │
│  3. EXTRACT     →  NER + LLM → entities, relations, triples        │
│  4. BUILD KG    →  resolve, link, ingest into Neo4j                │
│  5. RETRIEVE    →  graph + text evidence (chosen strategy)         │
│  6. GENERATE    →  build prompt context → local model or vLLM      │
│  7. SAVE        →  results, summaries, resource telemetry          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

##  Quick Start (Conda)

> **Recommended environment:** Conda, named `graphllm`, Python 3.10+.

**1. Create and activate the environment:**

```bash
conda create -n graphllm python=3.10 -y
conda activate graphllm
```

**2. Install dependencies** — pick the right set for your node:

```bash
pip install -r requirements.txt        # base dependencies
pip install -r requirements-cpu.txt    # CPU-only nodes
pip install -r requirements-gpu.txt    # GPU nodes (if available)
pip install -e .
```

---

##  Configuration

### Neo4j

Set the required connection variables (via `export` or an `.env` file):

```bash
export NEO4J_URL="neo4j+s://<your-instance>"
export NEO4J_USERNAME="<user>"
export NEO4J_PASSWORD="<pass>"
export NEO4J_DATABASE="<db>"        # optional
```

### HuggingFace (gated models)

```bash
export HF_TOKEN="<your-hf-token>"
```

### vLLM / OpenAI-compatible server

| Variable | Default | Description |
|---|---|---|
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM server endpoint |
| `VLLM_MODEL_NAME` | — | Model name served by vLLM |
| `VLLM_API_KEY` / `OPENAI_API_KEY` | — | API key, if required |

---

##  Usage Examples

### Single-question demo (GraphRAG retrieval + generation)

```bash
graphrag-demo \
  --question "What are the relations between Entity A and Entity B?" \
  --entity "Entity A"
```

### Local LLM generation

```bash
graphrag-demo --llm --model-id Qwen/Qwen2.5-7B-Instruct
```

### Server-backed generation (vLLM / OpenAI-compatible)

```bash
graphrag-demo \
  --llm --vllm \
  --vllm-base-url http://localhost:8000/v1 \
  --model-id Qwen/Qwen2.5-7B-Instruct
```

### Model tuning — control cost and GPU memory

```bash
graphrag-demo \
  --llm \
  --model-id Qwen/Qwen2.5-14B-Instruct \
  --max-new-tokens 128 \
  --gpu-memory-fraction 0.90
```

> **Tips:**
> - `--max-new-tokens` — reduces generation length and cost.
> - `--gpu-memory-fraction` — reserves headroom when loading large local models to reduce OOMs.
> - For models ≥ 30B, fp16 fallback is **disabled** by default. Enable `--allow-large-model-fp16-fallback` only if you understand the memory/precision tradeoffs.

---

##  Knowledge Graph Pipeline

The KG pipeline lives in `kg_pipeline/` and outputs checkpointed stage artifacts to a run directory. Defaults are controlled by `kg_pipeline/config.yaml`.

### Run the full pipeline

```bash
conda activate graphllm
PYTHONUNBUFFERED=1 python -m kg_pipeline.main \
  --config kg_pipeline/config.yaml \
  --env-file kg_pipeline/.env \
  --log-level INFO
```

### Pipeline stages

Reuse the same `--run-dir` to resume an existing run. To run a single stage, pass `--stage` with one of:

| Stage | Description |
|---|---|
| `ingestion` | Load raw documents |
| `chunking` | Split documents into chunks |
| `ner` | Named Entity Recognition |
| `llm` | LLM-based triple extraction |
| `resolution` | Entity resolution |
| `linking` | Triple linking |
| `neo4j` | Graph ingestion into Neo4j |

> Use `--dry-run` to skip Neo4j ingestion (useful for testing the extraction steps).

### Run directory structure

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

---

##  Experiments & Matrices

The repository includes scripts to run retrieval matrices comparing multiple strategies and LLMs.

### Available retrieval strategies

| Strategy | Description |
|---|---|
| `default` | All sources combined |
| `text_only` | Plain text retrieval only |
| `text_plus_triples` | Text + extracted triples |
| `neighbors_focus` | Local neighborhood of entities |
| `subgraph_2hop` | 2-hop subgraph expansion |
| `shortest_path` | Shortest path between entities |

### Smoke experiment (fast sanity check)

```bash
python scripts/run_retrieval_matrix.py \
  --smoke \
  --questions-file artifacts/experiments/questions_smoke.txt \
  --documents docs/ README.md \
  --runs-per-strategy 1 \
  --output-dir artifacts/experiments \
  --experiment-tag retrieval_matrix_smoke
```

### Full vLLM-backed matrix

```bash
python scripts/run_retrieval_matrix.py \
  --llm --vllm \
  --vllm-base-url http://localhost:8000/v1 \
  --model-id Qwen/Qwen2.5-32B-Instruct \
  --questions-file questions_matrix_long.txt \
  --graph-strategies default \
  --runs-per-strategy 1
```

---

##  Analysis & Telemetry

Each run produces a structured set of output artifacts:

### Output artifacts per run

```text
artifacts/experiments/<run_name>/
├── results.jsonl           # one record per question/strategy/run
├── results.csv             # tabular version of results
├── summary.txt             # fast human-readable check
├── summary.json            # structured statistics per strategy
├── resource_samples.jsonl  # raw resource telemetry samples
└── resource_summary.json   # peak and average resource usage
```

### Analysis scripts

| Script | Purpose |
|---|---|
| `scripts/analyze_experiments.py` | Analyze a single run directory |
| `scripts/analyze_matrix.py` | Aggregate multiple runs into CSV/JSON summaries |
| `scripts/analyze_resource_usage.py` | Sizing and resource comparison across runs |

> **Verification example:** A report at `artifacts/experiments/20260514_170536_test_strategies_verification/REPORT.txt` documents 60 runs, with `results.jsonl` as the raw trace and `summary.json` as the structured summary. Sample answer inspection: `python3 show_samples.py results.jsonl default 2`.

---

##  Evaluation Workflow

A dedicated evaluation workspace under [`evaluation/README.md`](evaluation/README.md) supports paper-oriented comparisons.

**Use it when you want to:**

- Build a gold QA dataset from run outputs and manual labels.
- Compute retrieval-oriented metrics such as entity coverage and rank-based scores.
- Optionally run RAGAS with a local judge model.
- Generate summary tables for papers or internal reports.

### Typical evaluation sequence

```bash
# 1. Prepare gold labels
#    → edit evaluation/gold_questions_template.csv

# 2. Join run output with gold set
python evaluation/build_eval_dataset.py

# 3. Compute retrieval metrics
python evaluation/retrieval_metrics.py

# 4. (Optional) Answer-quality metrics via RAGAS
python evaluation/run_ragas_eval.py
```

---

## Smoke Tests & Preflight

```bash
# Quick smoke check after install
python scripts/smoke_check.py

# Local preflight helper (PowerShell)
powershell -ExecutionPolicy Bypass -File scripts/preflight.ps1
```

> Before any long run, start with the smoke matrix command and verify that `summary.json` and `results.jsonl` are created in the output directory.

---

## Cluster & Batch Jobs

### Requirements by node type

| Node type | Requirements file |
|---|---|
| CPU | `requirements-cpu.txt` |
| GPU | `requirements-gpu.txt` |

### SLURM templates

```bash
export NEO4J_URL="neo4j+s://<your-instance>"
export NEO4J_USERNAME="<user>"
export NEO4J_PASSWORD="<pass>"
export NEO4J_DATABASE="<db>"

# Submit GPU job
sbatch -p <gpu_partition> scripts/run_graphrag.sbatch

# Submit CPU job
sbatch -p <cpu_partition> scripts/run_graphrag_cpu.sbatch
```

---

## Repository Structure

```text
.
├── kg_pipeline/          # Knowledge Graph construction pipeline
│   ├── config.yaml       # Pipeline configuration
│   ├── main.py           # Pipeline entry point
│   └── .env              # Environment variables (not committed)
├── src/graphrag/         # Main package — CLI and agent code
├── scripts/              # Utilities, smoke tests, batch templates
│   ├── run_retrieval_matrix.py
│   ├── analyze_experiments.py
│   ├── analyze_matrix.py
│   ├── analyze_resource_usage.py
│   ├── smoke_check.py
│   ├── preflight.ps1
│   ├── run_graphrag.sbatch
│   └── run_graphrag_cpu.sbatch
├── evaluation/           # Paper-oriented evaluation workspace
├── docs/                 # Additional documentation (e.g. cluster.md)
├── artifacts/            # Experiment outputs, logs, and reports
├── requirements.txt
├── requirements-cpu.txt
└── requirements-gpu.txt
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| CLI cannot connect to Neo4j | Wrong credentials or DB name | Verify `NEO4J_URL`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` |
| Local model loading fails | Insufficient GPU memory | Try a smaller model, reduce `--max-new-tokens`, review GPU memory settings |
| vLLM run produces no answers | Server URL or model name mismatch | Confirm `VLLM_BASE_URL` and model name match the running server process |
| Runs complete but context is empty | Retrieval or extraction issue | Inspect `summary.json` and `results.jsonl` before modifying the pipeline |



<div align="center">


</div>
