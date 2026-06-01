# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

`graphRAGPipelineExp1` is an experiment-oriented GraphRAG codebase with three real execution paths:

1. Knowledge graph construction from documents into Neo4j (`kg_pipeline/`)
2. Retrieval + answer generation via `graphrag-demo` CLI (`src/graphrag/`)
3. Experiment/evaluation workflows comparing strategies, models, and resource use (`scripts/`, `evaluation/`)

Many commands write artifacts that are later analyzed or included in paper-style reports.

## Environment

```bash
conda create -n graphllm python=3.10 -y
conda activate graphllm

# CPU
pip install -r requirements-cpu.txt && pip install -e .

# GPU (CUDA 12.4)
pip install -r requirements-gpu.txt && pip install -e .
```

Always prefer the `graphllm` Conda environment. Use `conda run -n graphllm ...` for reproducible script invocation.

Required env vars (`.env` or exported):
```bash
NEO4J_URL="bolt://localhost:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="..."
NEO4J_DATABASE="..."         # optional
HF_TOKEN="..."               # for gated HuggingFace models
VLLM_BASE_URL="http://localhost:8000/v1"
VLLM_MODEL_NAME="Qwen/Qwen2.5-32B-Instruct"
VLLM_API_KEY="..."           # or OPENAI_API_KEY
```

> `scripts/smoke_check.py` reads exported env vars — it does **not** auto-load `.env`.

## Common commands

```bash
# Health check (Neo4j + LLM)
python scripts/smoke_check.py

# Single question
conda run -n graphllm python -m graphrag.cli --question "What is X?" --entity "Y"
# graphrag-demo may point to a stale shim; use the above if exit code 126 occurs

# Build KG from documents
conda run -n graphllm python -m kg_pipeline.main \
  --config kg_pipeline/config.yaml \
  --env-file kg_pipeline/.env \
  --log-level INFO

# Run experiment matrix
python scripts/run_retrieval_matrix.py \
  --questions-file questions.txt \
  --models "7b,32b" \
  --strategies "default,text_plus_triples" \
  --output-dir artifacts/experiments

# Analyze a run
python scripts/analyze_experiments.py --results-dir artifacts/experiments --output-csv results_ranked.csv

# Smoke tests
python scripts/smoke_text_rag.py docs/ --query "Summarize the cluster setup" --top-k 4
python scripts/smoke_kg_retriever.py
python scripts/run_pipeline_smoke_full.py
```

## Running tests

```bash
pytest kg_pipeline/tests/test_pipeline.py -v
pytest evaluation/tests/test_metrics.py -v

# Single test
pytest kg_pipeline/tests/test_pipeline.py::test_schema_validation_accepts_valid_triple -v
```

## Architecture

### High-level data flow

```
Documents → [KG Pipeline: 7 stages] → Neo4j Knowledge Graph
                                            ↓
                            KGRetriever (multi-strategy)
                                            ↓
                           KGRAGAgent (LangGraph state machine)
                   decompose → route → retrieve → grade → generate
                                            ↓
                              LLMManager (local HF or vLLM)
                                            ↓
                             Answer + Provenance + Telemetry
```

### KG Pipeline stages (`kg_pipeline/main.py`)

Stages run sequentially with JSON checkpoint recovery — downstream stages read artifacts from earlier ones. Stage order matters.

| Stage | Output artifact |
|-------|----------------|
| 0: Ingest (PDFs → markdown) | `stage0_documents.json` |
| 1: Chunk (token-windowed paragraphs) | `stage1_chunks.json` |
| 2: NER (GLiNER) | `stage2_ner.json` |
| 3: LLM triple extraction | `stage3_triples_raw.json`, `stage3_acronyms.json` |
| 4: Entity resolution (embeddings + Jaccard) | `stage4_triples_resolved.json`, `stage4_registry.json` |
| 5: Triple linking | `stage5_triples_linked.json` |
| 6: Neo4j ingestion | `stage6_neo4j_summary.json` |

Also produced: `failed_chunks.jsonl`, `new_labels.log`, `pipeline.log`.

Stage 3 checkpoints every N chunks to `stage3_checkpoint.json` — re-running without clearing this file resumes from the last saved chunk.

### GraphRAG agent (`src/graphrag/agent/core.py`)

LangGraph state machine over `RAGState` (TypedDict). Nodes:

1. **decompose** — optional; breaks question into sub-questions; adds latency; disabled by default
2. **route** — optional adaptive routing; disabled by default
3. **retrieve** — executes one of 6 strategies via `KGRetriever`
4. **grade** — relevance check; triggers rewrite loop (max 3)
5. **generate** — builds prompt context (compressed to ≤1000 tokens) and calls LLM
6. **reflect** — optional output reflection

Do not reintroduce decomposition or routing steps unless the task explicitly requires them.

### Retrieval strategies (`src/graphrag/kg/retriever.py`)

`default` · `text_only` · `text_plus_triples` · `neighbors_focus` · `subgraph_2hop` · `shortest_path`

### Key CLI flags

| Flag | Effect |
|------|--------|
| `--llm` | Enable generation; without it, retrieval-only |
| `--vllm` | Use OpenAI-compatible vLLM endpoint instead of local HF model |
| `--enable-decomposition-step` | Extra LLM call before retrieval (higher latency) |
| `--enable-adaptive-routing-step` | Extra LLM call to choose retrieval mode |
| `--experiment` | Batch run over questions/strategies; exports structured artifacts |
| `--performance-profile` | `auto` / `default` / `production_fast` |

### Experiment outputs (`artifacts/experiments/<timestamp>_<tag>/`)

`results.jsonl` · `results.csv` · `summary.txt` · `summary.json` · `resource_samples.jsonl` · `resource_summary.json`

If you touch experiment code, confirm output names still match the analysis scripts.

## Data models

Core Pydantic models live in `kg_pipeline/models/types.py`. `RAGState`, `KGNode`, `Triple`, and `ProvenanceRecord` are TypedDicts in `src/graphrag/types.py`.

KGTriple predicates must be `SCREAMING_SNAKE_CASE` (validated by regex). Entity names are **not** unique before stage 4 resolution — use `CanonicalEntityRecord` after stage 4.

## Conventions

- **Type hints**: always; union types with `|` (Python 3.10+)
- **Docstrings**: Google-style — one-liner + Args/Returns/Raises
- **Logging**: module-level `logger = logging.getLogger("graphrag")` or `"kg_pipeline"`; INFO for milestones, DEBUG for traces, WARNING for recoverable issues
- **Imports**: stdlib → third-party → local, separated by blank lines
- **Pydantic**: `ConfigDict(extra="forbid")`; use `field_validator` for normalization
- **Cypher**: always parameterized — never f-string user input into Cypher
- **Neo4j writes**: UNWIND + MERGE for batches; never loop with individual queries

## Anti-patterns to avoid

- Bare `except:` or silent `except Exception:` — catch specific exceptions
- Querying Neo4j inside loops
- Assuming entity names are unique before resolution (stage 4)
- Calling LLM without retry logic (`LLMManager` handles this internally)
- Hardcoding model paths or credentials
- Skipping `validate_triples()` after LLM JSON parsing
- Mixing async/sync without coordinating `LLMManager._load_lock`
- Appending experiment results without `run_id` / timestamp
- Assuming vLLM is available — check health first
- Ignoring checkpoint files (re-running stage 3 without clearing resumes from checkpoint)

## Validation after edits

- Documentation only: `git diff --check -- README.md AGENTS.md CLAUDE.md`
- Python logic: run the smallest relevant smoke script
- CLI or pipeline changes: smallest command that exercises the touched path
- Experiment code: inspect a recent artifact folder and confirm output names match analyzer scripts

## Known issues & workarounds

| Issue | Workaround |
|-------|-----------|
| Exit code 126 on `graphrag-demo` | Use `conda run -n graphllm python -m graphrag.cli` |
| torch/torchvision version mismatch | Pin `torch==2.5.1+cu124` + `torchvision==0.20.1+cu124` |
| Neo4j UnknownPropertyKey warnings | Use `properties(node)['key']` accessor in Cypher |
| KG stage 3 crash on malformed LLM output | Expected — caught and logged to `failed_chunks.jsonl`; pipeline continues |
| Entity resolution too aggressive | Increase `resolution.similarity_threshold` in `kg_pipeline/config.yaml` |
| KG pipeline hangs on notebook disconnect | Use `sbatch scripts/run_kg_pipeline.sbatch` for detached execution |
