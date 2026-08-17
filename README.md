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
- [Retrieval Channels](#retrieval-channels)
- [Experiments & Retrieval Matrices](#experiments--retrieval-matrices)
- [Analysis & Telemetry](#analysis--telemetry)
- [Evaluation](#evaluation)
- [Interactive Demos](#interactive-demos)
- [Testing](#testing)
- [Cluster & Batch Jobs](#cluster--batch-jobs)
- [Repository Structure](#repository-structure)
- [Troubleshooting](#troubleshooting)
- [Known Limitations](#known-limitations)
- [License](#license)

---

## Overview

This repository implements a full GraphRAG pipeline that:

- **Ingests** documents (PDF, Markdown, plain text), chunks them, and extracts entities and triples via NER (GLiNER) and LLM-based extraction.
- **Builds a Knowledge Graph** by resolving and linking entities/triples, then ingesting them into Neo4j.
- **Retrieves** graph and text evidence through eight configurable strategies — from pure text retrieval to 2-hop subgraph expansion and shortest-path traversal — over a **lexical** full-text channel and an optional **multilingual vector** channel.
- **Generates answers** with a LangGraph agent (scope → retrieve → grade → generate, with a bounded rewrite loop), backed by either local Hugging Face models or an OpenAI-compatible vLLM server.
- **Grounds and verifies** answers: numbered evidence with document and page, a citation gate that checks every `[S1]`/`[T3]` tag against the index, a quote gate that checks every «quoted» passage against the retrieved text, and an optional verbatim-definition opener.
- **Runs reproducible experiment matrices** comparing retrieval strategies and LLMs, with full resource telemetry for sizing studies.
- **Evaluates** results with a dedicated toolkit (`evalkit`): retrieval metrics, text-similarity metrics, LLM-as-a-Judge scoring, optional RAGAS, and a two-channel / two-level gold scorer.

### Entry points

| Entry point | Purpose |
|---|---|
| `graphrag-demo` (or `python -m graphrag.cli`) | Single-question retrieval/generation and batch experiments — the full option surface |
| `python -m kg_pipeline.main` | Knowledge Graph construction pipeline |
| `python scripts/run_retrieval_matrix.py` | Standard-RAG vs GraphRAG matrices with resource telemetry |
| `python -m evalkit.cli` (with `PYTHONPATH=evaluation`) | Evaluation toolkit |
| `python evaluation/scripts/score_gold_run.py` | Gold scoring for the paper (two channels, two levels) |
| `streamlit run scripts/demo_app.py` | Expert demo console |

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
Neo4j Knowledge Graph  ──  full-text index (node_search)
        │                  vector index (node_embedding, :NodeVec carriers)
        ▼
KGRetriever — 8 strategies over lexical + vector + text channels
        │
        ▼
LangGraph agent: scope → decompose → route → retrieve → grade → generate
        │           (rewrite loop, max 3)
        ▼
LLMManager (local Hugging Face or vLLM server)
        │
        ▼
Answer + citations + provenance + resource telemetry
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

**3. The KG pipeline needs packages none of those files declare yet:**

```bash
pip install pymupdf4llm gliner openai pyyaml requests
```

`kg_pipeline/stages/ingestion.py` imports `pymupdf4llm`, `ner.py` imports `gliner`,
`llm_extraction.py` / `resolution.py` import `openai`, `kg_pipeline/main.py` imports `yaml`,
and `graphrag/embeddings.py` imports `requests`. Until they are added to the requirements
files this step is mandatory — see [Known Limitations](#known-limitations).

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

The Neo4j instance must have **APOC** available: every node and triple projection goes through
`apoc.map.removeKey` to strip the embedding vector from the returned properties.

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

### Embedding endpoint (multilingual vector channel)

| Variable | Default | Description |
|---|---|---|
| `GRAPHRAG_EMBED_BASE_URL` | `http://localhost:8002/v1` | OpenAI-compatible `/embeddings` endpoint |
| `GRAPHRAG_EMBED_MODEL` | `intfloat/multilingual-e5-base` | Encoder id; must match the one the index was built with |

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve intfloat/multilingual-e5-base \
    --runner pooling --port 8002 --gpu-memory-utilization 0.12 --max-model-len 512
```

### Optional runtime knobs

| Variable | Default | Effect |
|---|---|---|
| `GRAPHRAG_FULLTEXT_INDEX` | `node_search` | Full-text index name |
| `GRAPHRAG_VECTOR_PROPERTY` | `embedding` | Property stripped from node projections |
| `GRAPHRAG_NEO4J_QUERY_RETRIES` | `3` | Transient-error retries per Cypher query |
| `GRAPHRAG_LLM_GENERATE_RETRIES` | `2` | Transient-error retries per LLM call |
| `GRAPHRAG_LLM_CONCURRENT_REQUESTS` | `8` | Stage-3 extraction concurrency |
| `KG_EXTRACTION_MAX_TOKENS` | `4096` | Output cap per extraction call |
| `VLLM_HTTP_TIMEOUT` | `900` | OpenAI-client timeout in the KG pipeline |

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

### Grounded, cited answers

```bash
graphrag-demo --llm --vllm \
  --strategies hybrid \
  --cite-evidence --citation-display label \
  --prefer-verbatim-definitions \
  --enforce-language --complexity high
```

| Flag | Effect |
|---|---|
| `--cite-evidence` | Numbers the retrieved evidence, asks for `[S1]`/`[T1]` tags, and verifies every tag against the index |
| `--citation-policy` | `mark` flags an invented tag in place, `strip` deletes it |
| `--citation-display` | `id` keeps `[S1]`, `label` rewrites it as `[Document, p. 12]` after the gate |
| `--prefer-verbatim-definitions` | Ranks the defining passage first and opens the answer with it, quoted and checked |
| `--enforce-language` | Pins the answer to the question's language, with one retry on a mismatch |
| `--focused-answer` | Answer only what was asked, not every related concept in the evidence |
| `--complexity` | `low` / `medium` / `high`; `high` drops the two-paragraph cap and adds the specificity rule |

### Cross-lingual retrieval

The graph is largely Italian; English questions cannot reach Italian node names lexically.
The vector channel bridges that gap and is added to the lexical one, never replacing it.

```bash
python scripts/kg_vector_index.py    # once, after the KG is built

graphrag-demo --llm --vllm \
  --vector-retrieval --vector-nodes-limit 10 --vector-triples-limit 10 \
  --seed-from-retrieved --subgraph-seed-count 3 \
  --drop-predicates "RELATED_TO,PUBLISHED,AUTHORED_BY"
```

### Abstention on out-of-domain questions

```bash
graphrag-demo --llm --vllm --enable-domain-gate
```

One classification call before retrieval. Without it the agent has no path to abstain: the
dense retriever has no score floor, so `grade` always sees evidence and every question reaches
`generate`.

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
| `--max-context-tokens` | Caps the compressed prompt context (default **6000**) |
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

Stages run sequentially with JSON checkpoint recovery — each stage reads the artifacts of the
previous one. Reuse the same `--run-dir` to resume an existing run. `--stage <name>` runs
everything **up to and including** that stage, reusing earlier artifacts where they exist; it
does not run one stage in isolation.

| `--stage` | Description | Main artifact |
|---|---|---|
| `ingestion` | Load raw documents (PDF → markdown, page chunks, sections) | `stage0_documents.json` |
| `chunking` | Token-windowed paragraph chunks (three size profiles by page count) | `stage1_chunks.json` |
| `ner` | Named Entity Recognition (GLiNER, multilingual) | `stage2_ner.json` |
| `llm` | LLM-based triple extraction (async, batched, checkpointed) | `stage3_triples_raw.json`, `stage3_acronyms.json` |
| `resolution` | Entity resolution (embeddings + predicate Jaccard + LLM confirmation) | `stage4_triples_resolved.json`, `stage4_registry.json`, `stage4_merge_approved.json` |
| `linking` | `SAME_AS` alias edges, optional `MENTIONED_IN` | `stage5_triples_linked.json` |
| `neo4j` | Graph ingestion into Neo4j (UNWIND + MERGE per label/predicate signature) | `stage6_neo4j_summary.json` |

Useful flags:

- `--dry-run` — skip Neo4j ingestion (test the extraction stages only).
- `--single-doc <name>` — process a single document.
- `--run-dir <path>` — resume an existing run directory.
- Stage 3 checkpoints every `llm.checkpoint_every` chunks (atomic writes); re-running without clearing it resumes from the last saved chunk.

### Run directory layout

```text
kg_pipeline/artifacts/run_<tag>/
├── pipeline.log
├── run_metadata.json           # seed, models, git commit, vLLM endpoint
├── config.yaml                 # snapshot of the config used
├── relation_vocab_*.json       # snapshot of the predicate vocabulary
├── failed_chunks.jsonl         # malformed LLM outputs (logged, pipeline continues)
├── new_labels.log
├── stage0_documents.json
├── stage1_chunks.json
├── stage2_ner.json
├── stage3_triples_raw.json
├── stage3_acronyms.json
├── stage3_checkpoint.json
├── stage3_checkpoint_info.json
├── stage4_triples_resolved.json
├── stage4_registry.json
├── stage4_merge_approved.json
├── stage5_triples_linked.json
└── stage6_neo4j_summary.json
```

### Post-processing and indexes

After Neo4j ingestion, in this order:

```bash
python scripts/kg_postprocess.py --passes 1,2,3,4,5   # repair passes (kg_repair.py .. kg_repair5.py)
python scripts/kg_search_index.py                     # full-text index used by lexical retrieval
python scripts/kg_vector_index.py                     # :NodeVec carriers + vector index (cross-lingual)
```

The passes are distinct, ordered repair rounds, not versions of one script. Retrieval quality
depends on the last two commands having been run against the live graph.

Other graph utilities: `kg_backup.py` / `kg_restore.py`, `kg_densify.py`,
`kg_ontology_align.py`, `kg_translate_names.py`, `kg_collapse_aliases.py`,
`kg_evaluator.py`, `compare_kg_variants.py`, `kg_wipe.py`.

---

## Retrieval Channels

For each retrieval query the `KGRetriever`:

1. extracts entity candidates (quoted spans, capitalised phrases, numeric terms) and content keywords, optionally weighted by node-name document frequency (`lexical_specificity`);
2. runs one **lexical** full-text query (a Lucene OR-query with per-term boosts) for nodes and for triples;
3. optionally runs a **vector** query against the `:NodeVec` carriers, which is the only channel that can cross the IT/EN gap;
4. picks anchors — by default only names retrieval actually returned (`verify_anchor_exists`), which avoids a full graph scan on an anchor that matches nothing;
5. expands neighbours, the 2-hop subgraph (from `subgraph_seed_count` anchors) and the shortest path;
6. drops uninformative predicates, ranks triples (lexical overlap · mention count · confidence, with a penalty for system links);
7. optionally retrieves raw text (TF-IDF or dense FAISS), capped per document and re-ranked for definitional questions.

Missing infrastructure degrades rather than fails: no vector index or no embedding endpoint →
lexical only, with a WARNING; no full-text index → a per-term `CONTAINS` scan.

---

## Experiments & Retrieval Matrices

### Retrieval strategies

| Strategy | Evidence used |
|---|---|
| `default` | All KG channels: nodes, triples, neighborhoods, 2-hop subgraph, shortest paths |
| `hybrid` | All KG channels plus raw-text retrieval |
| `text_only` | Text retrieval only (no KG) |
| `no_retrieval` | No retrieval channel — the LLM-only baseline |
| `text_plus_triples` | Entity nodes and triples only (no graph traversal) |
| `neighbors_focus` | Triples plus local entity neighborhoods |
| `subgraph_2hop` | Triples plus 2-hop subgraph expansion |
| `shortest_path` | Triples plus shortest paths between entities |

### Which runner to use

| | `python -m graphrag.cli --experiment` | `scripts/run_retrieval_matrix.py` |
|---|---|---|
| GraphRAG strategies | ✅ | ✅ |
| Standard-RAG baselines (tfidf / dense presets) | ❌ | ✅ |
| Resource telemetry (CPU/RAM/GPU) | ❌ | ✅ |
| `query_id` carried into `results.jsonl` | ✅ | ❌ |
| Vector channel, citations, domain gate, complexity, … | ✅ | ❌ |

Use the CLI for anything the gold evaluation will score; use the matrix runner for
Standard-RAG comparisons and sizing studies.

### Batch run via the CLI

```bash
conda run -n graphllm python -m graphrag.cli --experiment \
  --questions-file gold.json \
  --strategies "default,hybrid,text_only,no_retrieval,text_plus_triples,neighbors_focus,subgraph_2hop,shortest_path" \
  --llm --vllm --vllm-base-url http://localhost:8000/v1 \
  --model-id Qwen/Qwen2.5-32B-Instruct \
  --vector-retrieval --seed-from-retrieved \
  --cite-evidence --complexity medium --max-new-tokens 1024 \
  --output-dir exp_results --experiment-tag thesis_qwen25_32b
```

Passing the gold `.json` straight to `--questions-file` guarantees the run emits `query_id`
and joins to the gold by id rather than by question text.

### Smoke matrix (fast sanity check)

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

`--questions-file` accepts both plain text (one question per line) and JSON suites produced by
`scripts/generate_questions.py`. Before any long run, start with the smoke matrix and verify
that `summary.json` and `results.jsonl` appear in the output directory.

---

## Analysis & Telemetry

Each experiment run produces a structured set of artifacts:

```text
<output-dir>/<timestamp>_<tag>/
├── results.jsonl           # one record per question/strategy/run
├── results.csv             # tabular version (does not carry insufficient_answer / text sources)
├── summary.txt             # fast human-readable check
├── summary.json            # structured statistics per strategy
├── config.json             # CLI args + fully resolved AgentConfig per strategy
├── resource_samples.jsonl  # raw resource telemetry samples (matrix runner)
└── resource_summary.json   # peak and average resource usage (matrix runner)
```

`config.json` makes every metric traceable to its exact configuration.

| Script | Purpose |
|---|---|
| `scripts/analyze_experiments.py` | Analyze a single run directory |
| `scripts/analyze_matrix.py` | Aggregate multiple runs into CSV/JSON summaries |
| `scripts/analyze_resource_usage.py` | Sizing and resource comparison across runs |
| `scripts/answer_diff.py` | Side-by-side answer comparison between runs |
| `scripts/provenance_precision.py` | Attribute retrieved text back to its origin documents |
| `scripts/kg_variant_significance.py` | Significance testing across KG variants |

---

## Evaluation

The evaluation workspace under [`evaluation/`](evaluation/README.md) supports paper-oriented comparisons through the `evalkit` toolkit:

- Build a gold QA dataset from run outputs and manual labels (templates and schema in `evaluation/gold/`).
- Compute retrieval metrics (entity coverage, precision/recall@k, MRR, nDCG, MAP) with bootstrap confidence intervals.
- Score answers with an LLM-as-a-Judge (Anthropic API, local vLLM/HF, or Claude Code backends), batched and resumable, and compare judge models.
- Optionally run RAGAS, and generate experiment- or project-level reports.

### Typical sequence

```bash
# 1. Join run output with the gold set
PYTHONPATH=evaluation python -m evalkit.cli build-dataset \
  --input exp_results/<run_dir> \
  --gold-file gold.json \
  --output artifacts/evaluation/eval_dataset.csv

# 2. Retrieval metrics
PYTHONPATH=evaluation python -m evalkit.cli retrieval \
  --input artifacts/evaluation/eval_dataset.csv \
  --save-json artifacts/evaluation/retrieval_summary.json

# 3. (Optional) LLM-as-a-Judge and RAGAS
PYTHONPATH=evaluation python -m evalkit.cli judge --input artifacts/evaluation/eval_dataset.csv ...
PYTHONPATH=evaluation python -m evalkit.cli ragas --input artifacts/evaluation/eval_dataset.csv ...
```

Available subcommands: `build-dataset`, `retrieval`, `text`, `judge`, `judge-compare`, `ragas`,
`kg`, `gold-triples`, `report-experiment`, `report-project`, `baseline-update`. See
[`evaluation/README.md`](evaluation/README.md) for backends, judge configuration, and the
recommended paper table schema.

### Gold scoring (the paper path)

```bash
python evaluation/scripts/score_gold_run.py \
  --run-dir exp_results/<run_dir>/ \
  --gold gold.json \
  --out-prefix artifacts/evaluation/<name>
```

Scores one run on two channels — the entities the retriever surfaced, and the gold surface
forms the answer text actually contains (deterministic gazetteer) — at two levels:

- **concept level**: normalised surface forms against the gold's `surface_forms`, over all expected entities. The pipeline-agnostic retrieval measure.
- **grounding level**: resolved canonical URIs, over `mapping_status == exact` entities only. The interoperability measure.

The two levels are reported side by side and never averaged into one number; the gap between
them is itself a result.

---

## Interactive Demos

```bash
# Streamlit console: multi-chat, intra-session memory, citations, domain gate
conda run -n graphllm streamlit run scripts/demo_app.py

# Same stack, terminal only
conda run -n graphllm python scripts/expert_demo.py --strategy hybrid --max-context-tokens 6000
```

The demos build their own `AgentConfig` inline (citations on, verbatim definitions on, language
enforcement on, domain gate on). They do not read the CLI flags, and they do not currently
enable the vector channel.

---

## Testing

### Unit tests

```bash
pytest tests/ kg_pipeline/tests/ evaluation/tests/ -q     # 448 tests
```

CI (GitHub Actions) runs a syntax check (`python -m compileall src scripts`) and the full test
suite on every push and pull request, using the CPU requirements.

### Smoke tests

```bash
python scripts/smoke_check.py            # health check: Neo4j + LLM connectivity
python scripts/smoke_kg_retriever.py     # KG retriever
python scripts/smoke_text_rag.py docs/ --query "Summarize the cluster setup" --top-k 4
python scripts/smoke_dense_rag.py        # dense text backend
python scripts/check_vector_index.py     # vector index presence and shape
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
| `scripts/start_vllm.sh` and `start_vllm_qwen25_7b.sh` / `_qwen25_72b.sh` / `_qwen3*.sh` | Start a local vLLM server for a specific model |
| `scripts/start_neo4j_staging.sh` / `promote_staging_to_aura.sh` | Local staging graph and promotion |
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
├── src/graphrag/         # Main package
│   ├── cli.py            #   CLI + experiment orchestration
│   ├── config.py         #   AgentConfig / KGConfig
│   ├── strategies.py     #   the 8 retrieval presets (single source of truth)
│   ├── agent/            #   LangGraph state machine, evidence/citations, memory, cache
│   ├── kg/               #   Neo4j manager and retriever
│   ├── llm/              #   backends, prompt library, refusal markers
│   ├── text_rag/         #   TF-IDF and dense (FAISS) text channels
│   ├── embeddings.py     #   shared multilingual encoder client
│   └── experiments/      #   experiment runner and resource monitor
├── kg_pipeline/          # KG construction pipeline (config.yaml, main.py, stages/)
├── scripts/              # Runners, analyzers, KG repair/index utilities, demos, SLURM templates
├── evaluation/           # Evaluation workspace
│   ├── evalkit/          #   metrics, LLM judge, reports (CLI: python -m evalkit.cli)
│   ├── gold/             #   gold QA datasets, templates and schema
│   ├── scripts/          #   score_gold_run.py and companions
│   ├── fixtures/         #   question sets for matrix runs
│   └── tests/            #   evaluation unit tests
├── tests/                # Core unit tests
├── documents/            # Source corpus (PDFs)
├── docs/                 # Cluster guide, plans, audits, worklogs
├── artifacts/            # Experiment and evaluation outputs (not committed)
├── exp_results*/         # Thesis campaign outputs (not committed)
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
| `ModuleNotFoundError: pymupdf4llm` / `gliner` / `openai` | Undeclared KG-pipeline dependency | `pip install pymupdf4llm gliner openai pyyaml requests` |
| CLI cannot connect to Neo4j | Wrong credentials or DB name | Verify `NEO4J_URL`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` |
| `Unknown function 'apoc.map.removeKey'` | APOC not installed on the instance | Install APOC; there is no fallback projection |
| `smoke_check.py` reports missing variables | `.env` not loaded | The script reads exported variables only — `export` them or source your `.env` |
| Local model loading fails | Insufficient GPU memory | Try a smaller model, reduce `--max-new-tokens`, tune `--gpu-memory-fraction` |
| torch/torchvision mismatch on GPU nodes | Unpinned installs | Use `requirements-gpu.txt` (pins `torch==2.5.1+cu124`, `torchvision==0.20.1+cu124`) |
| `import vllm` fails inside `graphllm` | Broken vLLM install in that env | Serve models from the `vllm-serve` virtualenv (`scripts/start_vllm*.sh`) |
| vLLM run produces no answers | Server URL or model name mismatch | Confirm `VLLM_BASE_URL` and model name match the running server |
| "vector channel skipped" warnings | Embedding endpoint down, or index missing | Start the encoder on `GRAPHRAG_EMBED_BASE_URL`, rerun `scripts/kg_vector_index.py` |
| "Full-text index unavailable" warning, then slow queries | The index is missing — or was disabled for the session by one bad query | Run `scripts/kg_search_index.py`; if the index exists, restart the process (see audit §2.1) |
| Evaluation warns "GOLD JOIN FALLBACK" | The run emitted no `query_id` | Use `python -m graphrag.cli --experiment` with a `.json`/`.csv` gold as `--questions-file` |
| Runs complete but context is empty | Retrieval or extraction issue | Inspect `summary.json` and `results.jsonl` before modifying the pipeline |
| KG stage 3 crashes on malformed LLM output | Expected behavior | Failures are logged to `failed_chunks.jsonl`; the pipeline continues |

---

## Known Limitations

Documented so results are read correctly. The full catalogue, with file and line references,
is in [`docs/code_audit_2026-08-15.md`](docs/code_audit_2026-08-15.md).

- **The retrieved context always opens with `Query: <question>`.** Because of this the context
  is never empty, so the agent's zero-evidence guard never fires and the `no_retrieval` arm
  receives the question itself as its context.
- **`run_retrieval_matrix.py` cannot express the newer options.** Vector retrieval, citations,
  the domain gate, `--complexity`, `--drop-predicates` and the rest are CLI-only. Matrix runs
  measure stock defaults.
- **Matrix runs carry no `query_id`,** so the evaluator joins them to the gold by question text.
- **`entity_coverage` in `evalkit.cli retrieval` reads the wrong field** for node entities
  (Neo4j elementId instead of the name). Use `evaluation/scripts/score_gold_run.py` for entity
  numbers.
- **A metric with zero observations is reported as `0.0`, not as "not measured"** — relevant
  for triple metrics under a JSON gold, where `gold_triples` is never populated.
- **Edge provenance is single-valued.** A triple attested in several documents keeps the
  `source_doc` / `page_range` of the last ingestion.
- **`stage6_neo4j_summary.json:relationships_written` counts triples sent, not edges written.**
- **MMR flags are inert on the default TF-IDF text backend**; they apply to `--text-retriever-backend dense` only.
- **The KG pipeline does not control `PYTHONHASHSEED`** despite setting it at runtime; export it
  before launching if you need it fixed.

---

## License

This project is licensed under the [MIT License](LICENSE).
