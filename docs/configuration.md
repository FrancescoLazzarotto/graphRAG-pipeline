# Configuration

Every environment variable the project reads, with the default the code actually
carries. Start from the template:

```bash
cp .env.example .env && $EDITOR .env
```

> Defaults below were read from the `os.getenv` call sites in `src/graphrag/`,
> `kg_pipeline/` and `product/`. A dash means the variable has no default and the
> feature that needs it is off or fails without it.

---

## Neo4j

| Variable | Required | Default | Description |
|---|:---:|---|---|
| `NEO4J_URL` | ✅ | — | Connection URI, e.g. `bolt://localhost:7687` or `neo4j+s://<instance>` |
| `NEO4J_USERNAME` | ✅ | — | Database user. The KG pipeline's ingestion stage accepts `NEO4J_USER` as an alias |
| `NEO4J_PASSWORD` | ✅ | — | Database password |
| `NEO4J_DATABASE` | — | `""` | Target database name |
| `NEO4J_URI` | — | — | Same value as `NEO4J_URL`. Read by the `scripts/kg/kg_repair3.py`, `kg_repair4.py` and `kg_repair5.py` post-processing passes, which predate the `NEO4J_URL` convention |

> **APOC is a hard dependency.** Every node and triple projection goes through
> `apoc.map.removeKey` to strip the embedding vector from the returned
> properties. There is no fallback projection — without APOC, retrieval raises.

---

## Generation endpoint

| Variable | Default | Description |
|---|---|---|
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM or other OpenAI-compatible endpoint |
| `VLLM_MODEL_NAME` | `""` | Model name served there |
| `VLLM_API_KEY` | falls back to `OPENAI_API_KEY`, then `EMPTY` | API key, where the endpoint wants one |
| `HF_TOKEN` | — | Hugging Face token for gated models. `HUGGINGFACE_HUB_TOKEN` is accepted as an alias |

---

## Embedding endpoint

The multilingual encoder behind the vector channel — the only retrieval channel
that crosses the Italian/English gap.

| Variable | Default | Description |
|---|---|---|
| `GRAPHRAG_EMBED_BASE_URL` | `http://localhost:8002/v1` | OpenAI-compatible `/embeddings` endpoint |
| `GRAPHRAG_EMBED_MODEL` | `intfloat/multilingual-e5-base` | Encoder id. **Must match the one the index was built with** |

```bash
bash scripts/serving/start_vllm_encoder.sh        # GPU 1, port 8002, pooling runner
```

The wrapper script exists because this command used to live only inside an abort
message, and a mistyped restart cost a campaign its vector channel on three of
six models. Changing `GRAPHRAG_EMBED_MODEL` means rebuilding the index with
`scripts/kg/kg_vector_index.py`.

---

## Retrieval and runtime knobs

All optional. Each is read from the environment at call time.

### Indexes and projections

| Variable | Default | Effect |
|---|---|---|
| `GRAPHRAG_FULLTEXT_INDEX` | `node_search` | Full-text index name |
| `GRAPHRAG_VECTOR_PROPERTY` | `embedding` | Property stripped from node projections |
| `GRAPHRAG_VECTOR_ALLOW_DEGRADED` | `""` (off) for the CLI, `1` for the demo | `1` lets a failed encoder degrade to lexical-only instead of raising. `product/config.py` sets it with `setdefault`, so both demos degrade and say so on the affected answer; the CLI keeps raising, because a campaign scored under two retrieval methods is not recoverable — see [Reproducibility](../README.md#reproducibility-notes) |
| `GRAPHRAG_TEXT_STAGE0_RUNS` | `""` | Default for `--text-stage0-runs` |

### Retries and timeouts

| Variable | Default | Effect |
|---|---|---|
| `GRAPHRAG_EMBED_RETRIES` | `3` | Encoder retries before the channel gives up |
| `GRAPHRAG_EMBED_RETRY_BACKOFF_SEC` | `0.5` | Backoff between encoder retries |
| `GRAPHRAG_EMBED_MAX_CHARS` | `1700` | Truncation applied before the encoder's context window |
| `GRAPHRAG_NEO4J_QUERY_RETRIES` | `3` | Transient-error retries per Cypher query |
| `GRAPHRAG_NEO4J_QUERY_RETRY_BACKOFF_SEC` | `1.0` | Backoff between Cypher retries |
| `GRAPHRAG_NEO4J_QUERY_TIMEOUT_SEC` | `45` | Cap on one Cypher query. Measured on the live graph, 34 of 36 queries in a retrieval finish under 0.23 s and the two slow ones are the unindexed `CONTAINS` scan at ~24 s, so this clears the slowest observed query with room to spare |
| `GRAPHRAG_NEO4J_MAX_RETRY_TIME_SEC` | `8` | Driver retry window per query (its own default is 30). At 30, one unreachable graph cost **301 s** of waiting in a measured demo session, because every query in a retrieval burned the window independently; at 8 the same failure took 119 s |
| `GRAPHRAG_NEO4J_CONNECTION_TIMEOUT_SEC` | `5` | TCP connect timeout |
| `GRAPHRAG_NEO4J_ACQUISITION_TIMEOUT_SEC` | `10` | Wait for a pooled connection |
| `GRAPHRAG_LLM_GENERATE_RETRIES` | `2` | Transient-error retries per LLM call |
| `GRAPHRAG_LLM_GENERATE_RETRY_BACKOFF_SEC` | `1.0` | Backoff between LLM retries |
| `GRAPHRAG_LLM_HTTP_TIMEOUT_SEC` | `300` | Client timeout on the interactive path, where someone is waiting |
| `GRAPHRAG_VLLM_HEALTHCHECK_TIMEOUT_SEC` | `5` | Endpoint health-check timeout |
| `VLLM_HTTP_TIMEOUT` | `900` | OpenAI-client timeout inside the KG pipeline, where nobody is |

### Local model placement

| Variable | Default | Effect |
|---|---|---|
| `GRAPHRAG_OFFLOAD_DIR` | `/tmp/graphrag-offload` | Offload target for local models |
| `GRAPHRAG_CPU_OFFLOAD_GIB` | `64` | CPU offload budget |
| `GRAPHRAG_TORCH_COMPILE` | `""` (off) | Opt into `torch.compile` for local models |
| `GRAPHRAG_ALLOW_LARGE_MODEL_FP16_FALLBACK` | `""` (off) | Environment equivalent of `--allow-large-model-fp16-fallback` |

### KG pipeline

| Variable | Default | Effect |
|---|---|---|
| `GRAPHRAG_LLM_CONCURRENT_REQUESTS` | `8` | Concurrency in stage-3 extraction and stage-4 merge confirmation |
| `KG_EXTRACTION_MAX_TOKENS` | `4096` | Output cap per extraction call |
| `KG_NER_BATCH_SIZE` | `16` | Chunks per GLiNER forward pass. `gliner.batch_size` in `config.yaml` wins over it |
| `KG_NER_DEVICE` | `""` | Device placement for GLiNER |
| `KG_EMBED_DEVICE` | — | Device placement for the resolution encoder |
| `KG_PIPELINE_DEBUG_OPENAI` | `""` (off) | Log raw extraction requests and responses |
| `PYTHONHASHSEED` | — | Export **before** launching if set-iteration order must be reproducible. CPython reads it at interpreter startup, so the pipeline cannot set it from inside; it warns when it is unset |

---

## Demo settings

The two demos in `product/` build their own config from `product/config.py`.
Every setting there is an environment variable with the value the demo ships
with, so nothing needs editing to try something else.

| Variable | Default | Effect |
|---|---|---|
| `DEMO_STRATEGY` | `hybrid` | Retrieval strategy |
| `DEMO_COMPLEXITY` | `high` | Answer depth |
| `DEMO_MAX_NEW_TOKENS` | `2048` | Generation cap |
| `DEMO_MAX_CONTEXT_TOKENS` | `6000` | Compressed-context cap |
| `DEMO_CITATION_POLICY` | `mark` | Invented-tag handling |
| `DEMO_CITATION_DISPLAY` | `label` | `[Document, p. 12]` instead of `[S1]` |
| `DEMO_TEXT_RETRIEVER_BACKEND` | `dense` | Text channel backend |
| `DEMO_TEXT_TOP_K` | `8` | Text chunks retrieved |
| `DEMO_TEXT_MAX_PER_DOC` | `2` | Cap on chunks from one document |
| `DEMO_TEXT_MMR_LAMBDA` | `0.7` | MMR relevance/diversity balance |
| `DEMO_NEO4J_FALLBACK_URL` | `""` | Graph used when the primary one does not answer |
| `DEMO_ENV_FILE` | `kg_pipeline/.env` | Where the demo reads credentials |
| `DEMO_LOG_DIR` | `artifacts/demo_sessions` | Session transcripts |
| `DEMO_PRODUCT_NAME` | `Assistente CEFF` | Name on the page and in the browser tab |
| `DEMO_PRODUCT_TAGLINE` | (Italian line) | The sentence under the name |
| `DEMO_PRODUCT_TAGLINE_EN` | (English line) | Same, when the interface is in English |
| `DEMO_PRODUCT_ICON` | `🌾` | Browser-tab icon |
| `DEMO_UI_LANGUAGE` | `it` | Interface language at startup; the reader can switch it |
| `DEMO_DEBUG` | `0` | Show the strategy, the model id and the graph URL on the page |
| `DEMO_EXAMPLE_QUESTIONS` | 3 questions, separated by `\|` | Offered when a question is refused as out of domain |

```bash
DEMO_STRATEGY=default DEMO_COMPLEXITY=medium \
  conda run -n graphllm streamlit run product/app.py
```

Change demo behaviour here, never in `graphrag.config` or `graphrag.strategies`:
those are what the campaigns were measured with, and editing them makes future
runs incomparable with the ones already reported.

---

## How `.env` is loaded

Not uniform across entry points, and it has caused confusion before:

| Entry point | `.env` handling |
|---|---|
| `scripts/smoke/smoke_check.py` | Loads `--env-file` (default `kg_pipeline/.env`), then a local `.env`, both with `override=False` — anything already exported wins |
| `python -m kg_pipeline.main` | Loads the file given by `--env-file`. Pass it explicitly |
| `python -m graphrag.cli` | Reads exported variables. Source your `.env` or run under a wrapper that does |
| `product/app.py`, `product/console.py` | Load `DEMO_ENV_FILE` (default `kg_pipeline/.env`) |

---

## Verify the configuration

```bash
python scripts/smoke/smoke_check.py
```

Checks imports, the graph (node count plus **both** indexes `ONLINE`), the
generator and the encoder. Every check runs by default and a failure is a
non-zero exit; waive one with `--skip-neo4j`, `--skip-llm` or `--skip-encoder`.

A carrier count alone cannot tell a live vector index from one whose identifiers
went stale under a store reload. Check that carriers still resolve:

```bash
python scripts/kg/check_vector_index.py --min-resolving 1000
```
