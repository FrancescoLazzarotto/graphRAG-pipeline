# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

`graphRAGPipelineExp1` is an experiment-oriented GraphRAG codebase with four real execution paths:

1. Knowledge graph construction from documents into Neo4j (`kg_pipeline/`)
2. Retrieval + answer generation via the `graphrag-demo` CLI (`src/graphrag/`)
3. Experiment/evaluation workflows comparing strategies, models, and resource use (`scripts/`, `evaluation/`)
4. Two interactive demos for domain-expert sessions (`scripts/demo_app.py` Streamlit, `scripts/expert_demo.py` console)

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

> **Known packaging gap.** The requirements files and `pyproject.toml` do not list
> `pymupdf4llm`, `gliner`, `openai`, `pyyaml` or `requests`, all of which are imported at
> module import time by `kg_pipeline/` and `graphrag.embeddings`. The `graphllm` env has
> them; a clean install does not. See `docs/code_audit_2026-08-15.md` §5.2.

> `import vllm` is broken inside `graphllm`. Every (re)start of a served model must go
> through the `vllm-serve` virtualenv — see the `scripts/start_vllm*.sh` wrappers.

Required env vars (`.env` or exported):
```bash
NEO4J_URL="bolt://localhost:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="..."
NEO4J_DATABASE="..."         # optional
NEO4J_URI="..."              # same value as NEO4J_URL; read by scripts/kg_repair3/4/5.py
HF_TOKEN="..."               # for gated HuggingFace models
VLLM_BASE_URL="http://localhost:8000/v1"
VLLM_MODEL_NAME="Qwen/Qwen2.5-32B-Instruct"
VLLM_API_KEY="..."           # or OPENAI_API_KEY
GRAPHRAG_EMBED_BASE_URL="http://localhost:8002/v1"   # multilingual encoder (vector channel)
GRAPHRAG_EMBED_MODEL="intfloat/multilingual-e5-base"
```

Optional runtime knobs, all read from the environment:

| Variable | Default | Effect |
|---|---|---|
| `GRAPHRAG_FULLTEXT_INDEX` | `node_search` | Neo4j full-text index name |
| `GRAPHRAG_VECTOR_PROPERTY` | `embedding` | Property stripped from node projections |
| `GRAPHRAG_NEO4J_QUERY_RETRIES` | `3` | Transient-error retries per Cypher query |
| `GRAPHRAG_NEO4J_QUERY_RETRY_BACKOFF_SEC` | `1.0` | Linear backoff base |
| `GRAPHRAG_LLM_GENERATE_RETRIES` | `2` | Transient-error retries per LLM call |
| `GRAPHRAG_LLM_CONCURRENT_REQUESTS` | `8` | Stage-3 extraction concurrency |
| `GRAPHRAG_TEXT_STAGE0_RUNS` | — | Default for `--text-stage0-runs` |
| `KG_EXTRACTION_MAX_TOKENS` | `4096` | Output cap per extraction call |
| `VLLM_HTTP_TIMEOUT` | `900` | OpenAI-client timeout in the KG pipeline |

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

# Experiment batch via the CLI (the only runner that exposes the full AgentConfig)
conda run -n graphllm python -m graphrag.cli --experiment \
  --questions-file gold.json \
  --strategies "default,hybrid,text_only,no_retrieval" \
  --llm --vllm --model-id Qwen/Qwen2.5-32B-Instruct \
  --output-dir exp_results/<tag>

# Experiment matrix (Standard RAG vs GraphRAG, resource telemetry)
python scripts/run_retrieval_matrix.py \
  --questions-file evaluation/fixtures/questions.txt \
  --graph-strategies "default,text_plus_triples" \
  --output-dir artifacts/experiments

# Analyze a run
python scripts/analyze_experiments.py --results-dir artifacts/experiments --output-csv results_ranked.csv

# Score a run against the frozen gold (two channels, two levels)
python evaluation/scripts/score_gold_run.py \
  --run-dir exp_results/<tag>/ --gold gold.json \
  --out-prefix artifacts/evaluation/<name>

# Interactive demos
conda run -n graphllm streamlit run scripts/demo_app.py
conda run -n graphllm python scripts/expert_demo.py --strategy hybrid

# Smoke tests
python scripts/smoke_text_rag.py docs/ --query "Summarize the cluster setup" --top-k 4
python scripts/smoke_kg_retriever.py
python scripts/run_pipeline_smoke_full.py
```

## Running tests

```bash
pytest tests/ kg_pipeline/tests/ evaluation/tests/ -q     # 448 tests

# Single file / single test
pytest evaluation/tests/test_metrics.py -v
pytest kg_pipeline/tests/test_pipeline.py::test_schema_validation_accepts_valid_triple -v
```

## Architecture

### High-level data flow

```
Documents → [KG Pipeline: 7 stages] → Neo4j Knowledge Graph
                                            ↓
                            KGRetriever (lexical + vector channels)
                                            ↓
                           KGRAGAgent (LangGraph state machine)
                scope → decompose → route → retrieve → grade → generate
                                            ↓
                              LLMManager (local HF or vLLM)
                                            ↓
                    Answer + citations + provenance + telemetry
```

### KG Pipeline stages (`kg_pipeline/main.py`)

Stages run sequentially with JSON checkpoint recovery — downstream stages read artifacts from
earlier ones. Stage order matters. `--stage <name>` runs everything **up to and including**
that stage (earlier stages are reused from their artifacts if present), it does not run one
stage in isolation.

| Stage | Output artifact |
|-------|----------------|
| 0: Ingest (PDFs → markdown, page chunks, sections) | `stage0_documents.json` |
| 1: Chunk (token-windowed paragraphs, 3 size profiles) | `stage1_chunks.json` |
| 2: NER (GLiNER) | `stage2_ner.json` |
| 3: LLM triple extraction (async, batched) | `stage3_triples_raw.json`, `stage3_acronyms.json` |
| 4: Entity resolution (embeddings + Jaccard + LLM confirm) | `stage4_triples_resolved.json`, `stage4_registry.json`, `stage4_merge_approved.json` |
| 5: Triple linking (SAME_AS, optional MENTIONED_IN) | `stage5_triples_linked.json` |
| 6: Neo4j ingestion (UNWIND + MERGE, grouped by signature) | `stage6_neo4j_summary.json` |

Also produced: `failed_chunks.jsonl`, `new_labels.log`, `pipeline.log`, `stage3_checkpoint.json`,
`stage3_checkpoint_info.json`, and a reproducibility bundle: `run_metadata.json` (seed, models,
git commit) plus snapshots of `config.yaml` and the relation vocab.

Stage 3 checkpoints every `llm.checkpoint_every` chunks (atomic writes) — re-running without
clearing this file resumes from the last saved chunk; triples past the last completed checkpoint
are dropped on resume to prevent duplicates.

After Stage 6, Neo4j post-processing passes are run via `scripts/kg_postprocess.py`
(`--passes 1,2,3,4,5`), which wraps `kg_repair.py`..`kg_repair5.py` — distinct repair rounds,
not script versions. Then the search and vector indexes:

```bash
python scripts/kg_search_index.py    # full-text index (node_search)
python scripts/kg_vector_index.py    # :NodeVec carriers + node_embedding vector index
```

### GraphRAG agent (`src/graphrag/agent/core.py`)

LangGraph state machine over `RAGState` (TypedDict). Nodes, in graph order:

1. **scope** — optional out-of-domain gate (one LLM call); follow-ups and questions of ≤3 words are exempt
2. **refuse** — terminal state for a rejected question; the only path to END without generating
3. **decompose** — optional; breaks the question into sub-questions; disabled by default
4. **route** — optional adaptive routing (`TEXT`/`KG`/`HYBRID`/`MULTIHOP`); disabled by default, defaults to `HYBRID`
5. **retrieve** — merges the channels of one or more retrieval queries, builds the numbered evidence index, compresses the context
6. **grade** — relevance check; routes to **rewrite** (max 3) or to generate
7. **generate** — calls the LLM, then runs the citation gate, the quote gate and the verbatim-definition prepend

There is **no reflect node**. `PromptLibrary.reflection_prompt` and
`AgentConfig.reflection_prompt` exist but nothing calls them.

Do not reintroduce decomposition or routing steps unless the task explicitly requires them.

### Retrieval strategies (`src/graphrag/strategies.py`)

Eight presets, the single source of truth shared by the CLI and the matrix runner:

`default` · `hybrid` · `text_only` · `no_retrieval` · `text_plus_triples` ·
`neighbors_focus` · `subgraph_2hop` · `shortest_path`

Presets only toggle the retrieval channels (`include_*`, `use_text_retriever`, `hops`).
Cardinality limits and ranking options come from the base `AgentConfig`.

### Retrieval channels (`src/graphrag/kg/retriever.py`)

Per query: entity-candidate extraction → search terms (+ optional DF-based specificity
weighting) → **lexical** full-text lookup (Lucene OR-query with boosts) **plus** an optional
**vector** lookup against `:NodeVec` carriers → seeds/anchors → neighbours, 2-hop subgraph,
shortest path → triple ranking → optional text channel (TF-IDF or dense FAISS).

The vector channel is the cross-lingual half: the graph is largely Italian and the gold
questions are English. It requires `scripts/kg_vector_index.py` and a running embedding
endpoint (`scripts/start_vllm_encoder.sh`). If the encoder fails after its retries the
retrieval **raises** rather than degrading, because a silent fallback to lexical-only
produced a model-asymmetric campaign; set `GRAPHRAG_VECTOR_ALLOW_DEGRADED=1` for
interactive use, where a lexical-only answer beats no answer.

The lexical text channel ranks with Okapi BM25 and honours `--text-retriever-mmr`.

### Key CLI flags (`python -m graphrag.cli`)

| Flag | Effect |
|------|--------|
| `--llm` | Enable generation; without it, retrieval-only |
| `--vllm` / `--vllm-base-url` | Use an OpenAI-compatible vLLM endpoint instead of a local HF model |
| `--strategies` | Comma-separated presets; in single-question mode the first one is applied |
| `--experiment` | Batch run over questions/strategies; exports structured artifacts |
| `--questions-file` | `.txt` (optionally `Q01<TAB>question`), `.json` (gold shape), `.jsonl`, `.csv` |
| `--cite-evidence` / `--citation-policy` / `--citation-display` | Numbered evidence, tag verification, reader-facing labels |
| `--prefer-verbatim-definitions` | Promote the defining passage and open the answer with it, quoted |
| `--vector-retrieval` / `--vector-index` / `--vector-*-limit` | Multilingual vector channel |
| `--seed-from-retrieved` | Anchor neighbours/subgraph/shortest path on retrieved node names, not question words |
| `--subgraph-seed-count` / `--subgraph-limit` | Breadth and cap of the subgraph channel |
| `--drop-predicates` | Remove uninformative predicates (e.g. `RELATED_TO,PUBLISHED`) from retrieved triples |
| `--enable-domain-gate` | Refuse out-of-domain questions instead of answering them |
| `--allow-parametric-fallback` | Permit marked parametric answers where retrieval missed |
| `--focused-answer` / `--complexity` | Answer scope and depth |
| `--enforce-language` | Pin the answer language to the question's, with one retry |
| `--legacy-insufficiency-wording` | Restore the pre-repair closing line, for reproducing campaigns E1–E8 |
| `--text-retriever-backend` / `--text-retriever-mmr*` / `--text-retriever-max-per-doc` | Text channel backend and diversification |
| `--text-docs-dir` / `--text-stage0-runs` | Which corpus feeds the text index |
| `--enable-decomposition-step` / `--enable-adaptive-routing-step` | Extra LLM calls before retrieval |
| `--max-new-tokens` / `--max-context-tokens` / `--recursion-limit` | Budgets (context default **6000**) |

`scripts/run_retrieval_matrix.py` exposes `--performance-profile`
(`auto` / `default` / `production_fast`) and resource telemetry, but **not** the
retrieval/answer flags above — see the warning below.

> **The two experiment runners are not equivalent.** `run_retrieval_matrix.py` builds its
> `AgentConfig` from eight fields only, so `vector_retrieval`, `seed_from_retrieved`,
> `cite_evidence`, `drop_predicates`, `enable_domain_gate`, `complexity`,
> `max_content_tokens` and the rest stay at their defaults there. Use
> `python -m graphrag.cli --experiment` for anything that depends on those.
> (`docs/code_audit_2026-08-15.md` §5.1)

### Experiment outputs (`<output-dir>/<timestamp>_<tag>/`)

`results.jsonl` · `results.csv` · `summary.txt` · `summary.json` · `config.json`
(+ `resource_samples.jsonl` · `resource_summary.json` from the matrix runner)

`config.json` records the CLI args and the fully resolved `AgentConfig` per strategy.
If you touch experiment code, confirm output names still match the analysis scripts.

### Evaluation (`evaluation/`)

`evalkit` is the toolkit; run it as `PYTHONPATH=evaluation python -m evalkit.cli <subcommand>`.
Subcommands: `build-dataset`, `retrieval`, `text`, `judge`, `judge-compare`, `ragas`, `kg`,
`gold-triples`, `report-experiment`, `report-project`, `baseline-update`.

The paper path is `evaluation/scripts/score_gold_run.py`: it scores one run against the frozen
gold on **two channels** (retrieval entities, answer-text gazetteer) at **two levels**
(concept, grounding), which are reported side by side and never averaged together.

### Reproducibility

- The KG pipeline seeds `random`, `numpy` and `torch` (`_set_seed` in `kg_pipeline/main.py`).
  It also writes `PYTHONHASHSEED` into `os.environ`, which **has no effect** at that point —
  CPython reads it only at interpreter startup. Export it before launching if you need it.
- Generation is deterministic by construction: local HF uses `do_sample=False`, vLLM uses `temperature=0`.
- Both LLM backends render prompts from `PromptLibrary` (`src/graphrag/llm/prompts.py`) — the
  single source of truth; never hardcode prompt strings in backends.
- Evaluation bootstrap CIs are seeded (`seed=42`).
- The vector index and the query encoder must use the same model and prefixes
  (`src/graphrag/embeddings.py` enforces the `query:` / `passage:` prefixes).

## Data models

Core Pydantic models live in `kg_pipeline/models/types.py`. `RAGState`, `KGNode`, `Triple` and
`ProvenanceRecord` are TypedDicts in `src/graphrag/types.py`.

`RAGState` is the LangGraph channel schema: **a key a node returns but does not declare there is
silently dropped**. `retrieved_neighbors` and `visible_evidence_refs` are declared
there now; check the schema before returning any new key.

KGTriple predicates must be `SCREAMING_SNAKE_CASE` (validated by regex). Entity names are **not**
unique before stage 4 resolution — use `CanonicalEntityRecord` after stage 4.

## Conventions

- **Type hints**: always; union types with `|` (Python 3.10+)
- **Docstrings**: Google-style — one-liner + Args/Returns/Raises
- **Logging**: module-level `logger = logging.getLogger("graphrag")` or `"kg_pipeline"`; INFO for milestones, DEBUG for traces, WARNING for recoverable issues
- **Imports**: stdlib → third-party → local, separated by blank lines
- **Pydantic**: `ConfigDict(extra="forbid")`; use `field_validator` for normalization
- **Cypher**: always parameterized — never f-string user input into Cypher
- **Neo4j writes**: UNWIND + MERGE for batches; never loop with individual queries
- **Comments explain why, not what** — this codebase's comments record measurements and
  rejected alternatives. Keep that register.

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
- Adding a generic phrase to `_REFUSAL_MARKERS` or `_INSUFFICIENT_MARKERS`
  (`src/graphrag/llm/refusal.py`) — they are substring-matched over the whole answer
- Returning a new key from an agent node without declaring it in `RAGState`
- Changing `PromptLibrary.DEFAULT_DOMAIN_SCOPE` or the domain-gate wording without rerunning
  `scripts/eval_domain_gate_llm.py` and `scripts/eval_domain_gate_heldout.py`

## Validation after edits

- Documentation only: `git diff --check -- README.md AGENTS.md CLAUDE.md`
- Python logic: run the smallest relevant smoke script
- CLI or pipeline changes: smallest command that exercises the touched path
- Experiment code: inspect a recent artifact folder and confirm output names match analyzer scripts
- Anything touching retrieval or scoring: `pytest tests/ evaluation/tests/ -q`

## Known issues & workarounds

| Issue | Workaround |
|-------|-----------|
| Exit code 126 on `graphrag-demo` | Use `conda run -n graphllm python -m graphrag.cli` |
| `import vllm` fails inside `graphllm` | Serve models from the `vllm-serve` virtualenv (`scripts/start_vllm*.sh`) |
| torch/torchvision version mismatch | Pin `torch==2.5.1+cu124` + `torchvision==0.20.1+cu124` |
| Neo4j UnknownPropertyKey warnings | Use `properties(node)['key']` accessor in Cypher |
| Retrieval requires APOC (`apoc.map.removeKey`) | No fallback is implemented; the instance must have APOC |
| KG stage 3 crash on malformed LLM output | Expected — caught and logged to `failed_chunks.jsonl`; pipeline continues |
| Entity resolution too aggressive | Increase `resolution.similarity_threshold` in `kg_pipeline/config.yaml` |
| KG pipeline hangs on notebook disconnect | Use `sbatch scripts/run_kg_pipeline.sbatch` for detached execution |
| `EmbeddingUnavailable` raised mid-run | The encoder is down: `scripts/start_vllm_encoder.sh`. It no longer degrades silently (audit §1.1 follow-up) |
| Retrieval suddenly slow mid-run | Was the full-text index disabled by one bad query? The markers are exact since 2026-08-17, so this should no longer happen (audit §2.1) |

## Open defects

`docs/code_audit_2026-08-15.md` is the current catalogue of known logic defects, with
locations and severities. Read it before changing the agent, the Neo4j layer, the
resolution stage or the evaluation metrics — several of the behaviours that look wrong
are already documented there with their failure mode.
