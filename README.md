<div align="center">

# GraphRAG Pipeline

**An experiment-oriented Retrieval-Augmented Generation pipeline combining Knowledge Graph retrieval with LLM-based answer generation.**

[![CI](https://github.com/FrancescoLazzarotto/graphRAG-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/FrancescoLazzarotto/graphRAG-pipeline/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Neo4j](https://img.shields.io/badge/Neo4j-Knowledge%20Graph-008CC1?logo=neo4j)](https://neo4j.com/)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Knowledge Graph Pipeline](#knowledge-graph-pipeline)
- [Experiments & Retrieval Matrices](#experiments--retrieval-matrices)
- [Analysis & Telemetry](#analysis--telemetry)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Cluster & Batch Jobs](#cluster--batch-jobs)
- [Repository Structure](#repository-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

This repository implements a full GraphRAG pipeline that:

- **Ingests** documents (PDF, Markdown, plain text), chunks them, and extracts entities and triples via NER (GLiNER) and LLM-based extraction.
- **Builds a Knowledge Graph** by resolving and linking entities/triples, then ingesting them into Neo4j.
- **Retrieves** graph and text evidence through eight configurable strategies — from pure text retrieval to 2-hop subgraph expansion and shortest-path traversal.
- **Generates answers** with a LangGraph agent (retrieve → grade → generate, with a bounded rewrite loop), backed by either local Hugging Face models or an OpenAI-compatible vLLM server.
- **Runs reproducible experiment matrices** comparing retrieval strategies and LLMs, with full resource telemetry for sizing studies.
- **Evaluates** results with a dedicated toolkit (`evalkit`): retrieval metrics, text-similarity metrics, LLM-as-a-Judge scoring, and optional RAGAS.

### Entry points

| Entry point | Purpose |
|---|---|
| `graphrag-demo` (or `python -m graphrag.cli`) | Single-question retrieval/generation and batch experiments |
| `python -m kg_pipeline.main` | Knowledge Graph construction pipeline |
| `python scripts/run_retrieval_matrix.py` | Retrieval-strategy experiment matrices |
| `python -m evalkit.cli` (with `PYTHONPATH=evaluation`) | Evaluation toolkit |

A complete command reference is available in [COMMANDS.md](COMMANDS.md).

---

## Architecture

```
Documents (PDF / Markdown / text)
        │
        ▼
KG Pipeline — 7 checkpointed stages
(ingest → chunk → NER → LLM triples → resolution → linking → Neo4j)
        │
        ▼
Neo4j Knowledge Graph
        │
        ▼
KGRetriever — 8 retrieval strategies
        │
        ▼
LangGraph agent: retrieve → grade → generate
        │
        ▼
LLMManager (local Hugging Face or vLLM server)
        │
        ▼
Answer + provenance + resource telemetry
```

---

## Installation

Recommended environment: Conda, named `graphllm`, Python 3.10+.

**1. Create and activate the environment:**

```bash
conda create -n graphllm python=3.10 -y
conda activate graphllm
```

**2. Install dependencies** — pick **one** requirements file for your target:

```bash
pip install -r requirements.txt        # development (loose bounds)
pip install -r requirements-cpu.txt    # CPU-only nodes (bounded versions)
pip install -r requirements-gpu.txt    # GPU nodes, CUDA 12.4 (pinned torch/torchvision + vLLM)
pip install -e .
```

The three files form a hierarchy, not a sequence: `requirements.txt` is for local
development, `-cpu`/`-gpu` are the reproducible cluster installs. Evaluation
extras (RAGAS, ROUGE, plotting) live in `evaluation/requirements.txt`.

---

## Configuration

Copy the template and fill in your credentials:

```bash
cp .env.example .env
```

### Neo4j

| Variable | Required | Description |
|---|---|---|
| `NEO4J_URL` | yes | Connection URI, e.g. `bolt://localhost:7687` or `neo4j+s://<instance>` |
| `NEO4J_USERNAME` | yes | Database user |
| `NEO4J_PASSWORD` | yes | Database password |
| `NEO4J_DATABASE` | no | Target database name |
| `NEO4J_URI` | no | Same value as `NEO4J_URL` — read by the `scripts/kg_repair3/4/5.py` post-processing passes |

### Hugging Face (gated models)

```bash
export HF_TOKEN="<your-hf-token>"
```

### vLLM / OpenAI-compatible server

| Variable | Default | Description |
|---|---|---|
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM server endpoint |
| `VLLM_MODEL_NAME` | — | Model name served by vLLM |
| `VLLM_API_KEY` / `OPENAI_API_KEY` | — | API key, if required |

> **Note:** `scripts/smoke_check.py` reads exported environment variables only — it does **not** auto-load `.env`.

---

## Usage

### Single-question demo (retrieval only)

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

### Model tuning — cost and GPU memory

```bash
graphrag-demo \
  --llm \
  --model-id Qwen/Qwen2.5-14B-Instruct \
  --max-new-tokens 128 \
  --gpu-memory-fraction 0.90
```

| Flag | Effect |
|---|---|
| `--max-new-tokens` | Caps generation length (and cost) |
| `--max-context-tokens` | Caps the compressed prompt context (default 1000) |
| `--gpu-memory-fraction` | Reserves headroom when loading large local models to reduce OOMs |
| `--allow-large-model-fp16-fallback` | For models ≥ 30B, fp16 fallback is disabled by default; enable only if you understand the memory/precision trade-offs |

### Test-suite generation

Generate a JSON question suite from the latest KG pipeline run (uses the local vLLM endpoint):

```bash
conda run -n graphllm python scripts/generate_questions.py generate
conda run -n graphllm python scripts/generate_questions.py generate --question-language en
conda run -n graphllm python scripts/generate_questions.py stats --input artifacts/tmp/graphrag_test_suite.json
```

The generator defaults to the most recent `kg_pipeline/artifacts/run_*` directory and writes to `artifacts/tmp/graphrag_test_suite.json` unless `--output` is provided. Use `--matrix-output` to export one-question-per-line text for matrix runs.

---

## Knowledge Graph Pipeline

The KG pipeline lives in `kg_pipeline/` and writes checkpointed stage artifacts to a run directory. Defaults are controlled by `kg_pipeline/config.yaml`.

### Run the full pipeline

```bash
conda activate graphllm
PYTHONUNBUFFERED=1 python -m kg_pipeline.main \
  --config kg_pipeline/config.yaml \
  --env-file kg_pipeline/.env \
  --log-level INFO
```

### Pipeline stages

Stages run sequentially with JSON checkpoint recovery — each stage reads the artifacts of the previous one. Reuse the same `--run-dir` to resume an existing run; pass `--stage` to run a single stage.

| `--stage` | Description | Main artifact |
|---|---|---|
| `ingestion` | Load raw documents (PDF → markdown) | `stage0_documents.json` |
| `chunking` | Token-windowed paragraph chunks | `stage1_chunks.json` |
| `ner` | Named Entity Recognition (GLiNER) | `stage2_ner.json` |
| `llm` | LLM-based triple extraction | `stage3_triples_raw.json`, `stage3_acronyms.json` |
| `resolution` | Entity resolution (embeddings + Jaccard) | `stage4_triples_resolved.json`, `stage4_registry.json` |
| `linking` | Triple linking | `stage5_triples_linked.json` |
| `neo4j` | Graph ingestion into Neo4j | `stage6_neo4j_summary.json` |

Useful flags:

- `--dry-run` — skip Neo4j ingestion (test the extraction stages only).
- `--single-doc <name>` — process a single document.
- Stage 3 checkpoints periodically to `stage3_checkpoint.json` (atomic writes); re-running without clearing it resumes from the last saved chunk.

### Run directory layout

```text
kg_pipeline/artifacts/run_<tag>/
├── pipeline.log
├── failed_chunks.jsonl      # malformed LLM outputs (logged, pipeline continues)
├── new_labels.log
├── stage0_documents.json
├── stage1_chunks.json
├── stage2_ner.json
├── stage3_triples_raw.json
├── stage3_acronyms.json
├── stage3_checkpoint.json
├── stage4_triples_resolved.json
├── stage4_registry.json
├── stage5_triples_linked.json
└── stage6_neo4j_summary.json
```

### Post-processing

After Neo4j ingestion, run the graph repair passes:

```bash
python scripts/kg_postprocess.py --passes 1,2,3,4
```

---

## Experiments & Retrieval Matrices

`scripts/run_retrieval_matrix.py` runs matrices comparing retrieval strategies and LLMs.

### Retrieval strategies

| Strategy | Evidence used |
|---|---|
| `default` | All KG channels: nodes, triples, neighborhoods, 2-hop subgraph, shortest paths |
| `hybrid` | All KG channels plus raw-text retrieval |
| `text_only` | Text retrieval only (no KG) |
| `no_retrieval` | No context — LLM-only baseline |
| `text_plus_triples` | Entity nodes and triples only (no graph traversal) |
| `neighbors_focus` | Triples plus local entity neighborhoods |
| `subgraph_2hop` | Triples plus 2-hop subgraph expansion |
| `shortest_path` | Triples plus shortest paths between entities |

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
  --questions-file evaluation/fixtures/questions_matrix_long.txt \
  --graph-strategies default \
  --runs-per-strategy 1
```

`--questions-file` accepts both plain text (one question per line) and JSON suites produced by `scripts/generate_questions.py`. Before any long run, start with the smoke matrix and verify that `summary.json` and `results.jsonl` appear in the output directory.

---

## Analysis & Telemetry

Each experiment run produces a structured set of artifacts:

```text
artifacts/experiments/<run_name>/
├── results.jsonl           # one record per question/strategy/run
├── results.csv             # tabular version of results
├── summary.txt             # fast human-readable check
├── summary.json            # structured statistics per strategy
├── config.json             # CLI args + fully resolved AgentConfig per strategy
├── resource_samples.jsonl  # raw resource telemetry samples
└── resource_summary.json   # peak and average resource usage
```

`config.json` makes every metric traceable to its exact configuration.

| Script | Purpose |
|---|---|
| `scripts/analyze_experiments.py` | Analyze a single run directory |
| `scripts/analyze_matrix.py` | Aggregate multiple runs into CSV/JSON summaries |
| `scripts/analyze_resource_usage.py` | Sizing and resource comparison across runs |

---

## Evaluation

The evaluation workspace under [`evaluation/`](evaluation/README.md) supports paper-oriented comparisons through the `evalkit` toolkit:

- Build a gold QA dataset from run outputs and manual labels (templates and schema in `evaluation/gold/`).
- Compute retrieval metrics (entity coverage, precision/recall@k, MRR) with bootstrap confidence intervals.
- Score answers with an LLM-as-a-Judge (Anthropic API, local vLLM/HF, or Claude Code backends) and compare judge models.
- Optionally run RAGAS, and generate experiment- or project-level reports.

### Typical sequence

```bash
# 1. Prepare gold labels — fill a copy of evaluation/gold/gold_questions_template.csv

# 2. Join run output with the gold set
PYTHONPATH=evaluation python -m evalkit.cli build-dataset \
  --input artifacts/experiments/<run_dir> \
  --gold-file evaluation/gold/<your_gold>.csv \
  --output artifacts/evaluation/eval_dataset.csv

# 3. Retrieval metrics
PYTHONPATH=evaluation python -m evalkit.cli retrieval \
  --input artifacts/evaluation/eval_dataset.csv \
  --save-json artifacts/evaluation/retrieval_summary.json

# 4. (Optional) LLM-as-a-Judge and RAGAS
PYTHONPATH=evaluation python -m evalkit.cli judge --input artifacts/evaluation/eval_dataset.csv ...
PYTHONPATH=evaluation python -m evalkit.cli ragas --input artifacts/evaluation/eval_dataset.csv ...
```

Available subcommands: `build-dataset`, `retrieval`, `text`, `judge`, `judge-compare`, `ragas`, `kg`, `report-experiment`, `report-project`, `baseline-update`. See [`evaluation/README.md`](evaluation/README.md) for backends, judge configuration, and the recommended paper table schema.

---

## Testing

### Unit tests

```bash
pytest tests/ kg_pipeline/tests/ evaluation/tests/ -q
```

CI (GitHub Actions) runs a syntax check (`python -m compileall src scripts`) and the full test suite on every push and pull request, using the CPU requirements.

### Smoke tests

```bash
python scripts/smoke_check.py            # health check: Neo4j + LLM connectivity
python scripts/smoke_kg_retriever.py     # KG retriever
python scripts/smoke_text_rag.py docs/ --query "Summarize the cluster setup" --top-k 4
python scripts/run_pipeline_smoke_full.py
```

On Windows, a preflight helper is available: `powershell -ExecutionPolicy Bypass -File scripts/preflight.ps1`.

---

## Cluster & Batch Jobs

Install with `requirements-cpu.txt` on CPU nodes and `requirements-gpu.txt` on GPU nodes. Export the Neo4j variables before submission, then use the SLURM templates:

| Script | Purpose |
|---|---|
| `scripts/run_kg_pipeline.sbatch` | Detached KG pipeline run |
| `scripts/run_graphrag.sbatch` | GraphRAG job on a GPU node |
| `scripts/run_graphrag_cpu.sbatch` | GraphRAG job on a CPU node |
| `scripts/run_experiment_matrix_gpu.sbatch` | Experiment matrix on a GPU node |
| `scripts/start_vllm.sh` | Start a local vLLM server |
| `scripts/submit_matrix_from_env.sh` | Submit a matrix parameterized via environment variables |

```bash
export NEO4J_URL="neo4j+s://<your-instance>"
export NEO4J_USERNAME="<user>"
export NEO4J_PASSWORD="<pass>"
export NEO4J_DATABASE="<db>"

sbatch -p <gpu_partition> scripts/run_graphrag.sbatch
sbatch -p <cpu_partition> scripts/run_graphrag_cpu.sbatch
```

See [docs/cluster.md](docs/cluster.md) for the full deployment guide.

---

## Repository Structure

```text
.
├── src/graphrag/         # Main package: CLI, LangGraph agent, retriever, LLM backends
├── kg_pipeline/          # Knowledge Graph construction pipeline (config.yaml, main.py, stages/)
├── scripts/              # Experiment runners, analyzers, smoke tests, SLURM templates
├── evaluation/           # Evaluation workspace
│   ├── evalkit/          #   metrics, LLM judge, reports (CLI: python -m evalkit.cli)
│   ├── gold/             #   gold QA templates and schema
│   ├── fixtures/         #   question sets for matrix runs
│   └── tests/            #   evaluation unit tests
├── tests/                # Core unit tests
├── documents/            # Source corpus (PDFs)
├── docs/                 # Additional documentation (cluster.md, plans, reports)
├── artifacts/            # Experiment outputs, logs, and reports (not committed)
├── COMMANDS.md           # Full command reference
├── pyproject.toml
├── requirements.txt      # + requirements-cpu.txt / requirements-gpu.txt
└── .env.example          # Configuration template
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `graphrag-demo` exits with code 126 | Stale console-script shim | Use `conda run -n graphllm python -m graphrag.cli` |
| CLI cannot connect to Neo4j | Wrong credentials or DB name | Verify `NEO4J_URL`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` |
| `smoke_check.py` reports missing variables | `.env` not loaded | The script reads exported variables only — `export` them or source your `.env` |
| Local model loading fails | Insufficient GPU memory | Try a smaller model, reduce `--max-new-tokens`, tune `--gpu-memory-fraction` |
| torch/torchvision mismatch on GPU nodes | Unpinned installs | Use `requirements-gpu.txt` (pins `torch==2.5.1+cu124`, `torchvision==0.20.1+cu124`) |
| vLLM run produces no answers | Server URL or model name mismatch | Confirm `VLLM_BASE_URL` and model name match the running server |
| Runs complete but context is empty | Retrieval or extraction issue | Inspect `summary.json` and `results.jsonl` before modifying the pipeline |
| KG stage 3 crashes on malformed LLM output | Expected behavior | Failures are logged to `failed_chunks.jsonl`; the pipeline continues |

---

## License

This project is licensed under the [MIT License](LICENSE).
