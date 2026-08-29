# Troubleshooting

Symptoms seen in this project, with the cause that actually produced them.

---

## Installation and entry points

| Symptom | Likely cause | Fix |
|---|---|---|
| `graphrag-demo` exits with code 126 | Stale console-script shim from an older install | `pip install -e .`, or use `conda run -n graphllm python -m graphrag.cli`, which never depends on the shim |
| `ModuleNotFoundError` on a KG-pipeline import | Environment built before the dependency declarations landed | `pip install -r requirements.txt && pip install -e .` |
| torch/torchvision mismatch on GPU nodes | Unpinned installs | Use `requirements-gpu.txt`, which pins `torch==2.5.1+cu124` and its torchvision |
| `import vllm` fails inside `graphllm` | Broken vLLM install in that environment | Serve models from the `vllm-serve` virtualenv via `scripts/serving/start_vllm*.sh`. Do not try to repair `graphllm` |

---

## Graph

| Symptom | Likely cause | Fix |
|---|---|---|
| Cannot connect to Neo4j | Wrong credentials or database name | Check `NEO4J_URL`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` |
| `Unknown function 'apoc.map.removeKey'` | APOC not installed on the instance | Install APOC. There is no fallback projection |
| "Full-text index unavailable" warning, then slow queries | Index missing | `python scripts/kg/kg_search_index.py` |
| "vector channel skipped" warnings | Vector index not built | `python scripts/kg/kg_vector_index.py` |
| Vector channel looks healthy but recall drops | Carriers went stale after a store reload — the count still passes | `python scripts/kg/check_vector_index.py --min-resolving 1000`, then rebuild the index |
| Hosted instance stops resolving (NXDOMAIN) | A free-tier instance suspended itself after idle days | Resume it, or point `DEMO_NEO4J_FALLBACK_URL` at the staging graph |
| `UnknownPropertyKey` warnings from Cypher | Direct `node.key` access on a property that does not exist on every node | Use `properties(node)['key']` |
| Retrieval suddenly slow mid-run | One bad query disabled the full-text index | Since 2026-08-22 a disabled index is re-probed on a 30 s → 15 min backoff instead of staying off for the life of the process. If it persists, rebuild with `scripts/kg/kg_search_index.py` |

---

## Models and serving

| Symptom | Likely cause | Fix |
|---|---|---|
| Local model loading fails | Not enough GPU memory | Smaller model, lower `--max-new-tokens`, tune `--gpu-memory-fraction` |
| vLLM run produces no answers | Server URL or model name mismatch | Confirm `VLLM_BASE_URL` and the served model name match what the CLI was given |
| Run aborts on `EmbeddingUnavailable` | Encoder down or overloaded | Start it with `scripts/serving/start_vllm_encoder.sh`. This is a stop, not a degradation, by design — see below |

**Why the encoder failure stops the run.** Every other channel degrades with a
warning. The encoder does not, because a silent fallback to lexical-only changes
the retrieval method mid-comparison and does it asymmetrically: measured once, it
dropped the channel on 3 queries for 3 of 6 compared models and 0 for the other
3. A stopped run is recoverable; a campaign scored under two different retrieval
methods is not. `GRAPHRAG_VECTOR_ALLOW_DEGRADED=1` restores degradation for
interactive use only.

---

## Runs and evaluation

| Symptom | Likely cause | Fix |
|---|---|---|
| `smoke_check.py` reports missing variables | Neither the exported environment nor `--env-file` supplies them | Export them, or point `--env-file` at the file that has them. Exported values always win |
| Evaluation warns `GOLD JOIN FALLBACK` | The run emitted no `query_id`, so the join fell back to matching question text | Re-run through `python -m graphrag.cli --experiment` with a `.json`/`.csv` gold as `--questions-file`. The matrix runner cannot carry ids |
| Scores differ from a colleague's on the same run | Different gold file. `gold.json` and `gold_v3.json` differ on 7 of 30 questions | Pass `--gold evaluation/gold/gold_v3.json` explicitly |
| Runs complete but the context is empty | Retrieval or extraction problem, not generation | Inspect `summary.json` and `results.jsonl` before changing the pipeline |
| A metric reads `None` instead of `0.0` | Zero observations — the metric was never measured | Intentional. A printed zero reads as "the system scored zero" when it means "never measured" |

---

## KG pipeline

| Symptom | Likely cause | Fix |
|---|---|---|
| Stage 3 hits malformed LLM output | Expected behaviour | Failures are logged to `failed_chunks.jsonl` and the pipeline continues |
| A stage re-runs work you thought was done | `--stage <name>` runs everything **up to and including** that stage | Reuse the same `--run-dir` so earlier artifacts are found and reused |
| Stage 3 restarts from the beginning | Checkpoint cleared or a different run directory | It checkpoints every `llm.checkpoint_every` chunks with atomic writes; resume with the same `--run-dir` |
| Resolution results shift between runs without a code change | Set-iteration order | Export `PYTHONHASHSEED` **before** launching. CPython reads it at interpreter startup, so the pipeline cannot set it from inside |
| Entity resolution exhausts memory | It materialises a full similarity matrix (~576 MB at 12k groups) | A scaling limit on larger corpora. Split the corpus or raise the machine |
| Entity resolution merges things it should not | Threshold too permissive for this corpus | Raise `resolution.similarity_threshold` in `kg_pipeline/config.yaml`, then re-run resolution alone with `scripts/kg/remerge_entities.py` |
| The pipeline dies when a notebook or SSH session drops | It is running in the foreground of that session | `sbatch scripts/cluster/run_kg_pipeline.sbatch` for detached execution |

---

## Health checks

```bash
python scripts/smoke/smoke_check.py                 # imports, graph + both indexes, generator, encoder
python scripts/kg/check_vector_index.py --min-resolving 1000
python scripts/smoke/smoke_kg_retriever.py
python scripts/smoke/smoke_text_rag.py docs/ --query "Summarize the cluster setup" --top-k 4
python scripts/smoke/smoke_dense_rag.py
python scripts/smoke/run_pipeline_smoke_full.py
```

On Windows: `powershell -ExecutionPolicy Bypass -File scripts/cluster/preflight.ps1`.
