# scripts/

Operational entrypoints, grouped by job. Every script is run from the repo root
(`python scripts/<group>/<name>.py`, `bash scripts/<group>/<name>.sh`); each one
resolves the repo root from its own location, so the working directory only
matters for relative `--output` paths.

| Group | Contains |
|---|---|
| `kg/` | Graph lifecycle against Neo4j: backup/restore/wipe, the `kg_repair*` passes and their `kg_postprocess.py` driver, alias collapse, translation, densification, ontology alignment, search and vector indexes. |
| `kg/quality/` | Standalone graph cleanup passes and structural metrics (`pass1_cleanup`, `pass3_rename_merge`, `merge_same_as`, `kg_metrics`). |
| `gold/` | Gold-set construction: question generation, Italian gold build, annotation backfill, AGROVOC lexicon. |
| `domain_gate/` | Domain-scope threshold calibration and its held-out evaluation. Rerun both after any change to `PromptLibrary.DEFAULT_DOMAIN_SCOPE`. |
| `runners/` | Experiment drivers: retrieval matrix, A/B fast profile, gold variant, Italian and abstention arms. |
| `smoke/` | Reachability and end-to-end checks — the fastest way to tell whether Neo4j and the LLM endpoint are alive. |
| `analysis/` | Post-run analysis: result aggregation, answer diffs, provenance precision, KG variant comparison and significance, slot ceiling, visualisation. |
| `serving/` | vLLM, Neo4j staging and demo start/stop wrappers, plus the per-model `chat_templates/`. vLLM wrappers use the `vllm-serve` virtualenv, never `graphllm`. |
| `cluster/` | SLURM job templates and submission helpers. |

## Order matters in two places

After a KG build, run these three against the live graph, in this order:

```bash
python scripts/kg/kg_postprocess.py --passes 1,2,3,4,5   # repair rounds, not versions of one script
python scripts/kg/kg_search_index.py                     # full-text index — lexical retrieval
python scripts/kg/kg_vector_index.py                     # :NodeVec carriers + vector index — cross-lingual
```

Retrieval quality depends on the last two having been run. `--passes` defaults to
`1,2,3,4`; pass 5 exists and is opt-in, so name it explicitly.

The `kg_repair*.py` passes are driven through `kg_postprocess.py`, never called
directly, and each loads `kg_pipeline/.env` for its `NEO4J_*` and `VLLM_*`
variables.

## Two things the runners do not share

- `runners/run_retrieval_matrix.py` takes **`--graph-strategies`** and
  **`--standard-strategies`**. It has no `--strategies` and no `--models`; pass a
  single `--model-id`.
- Matrix runs carry no `query_id`, so the evaluator joins them to the gold by
  question text. Use `python -m graphrag.cli --experiment` for anything the gold
  scorer will read.

Fuller descriptions: [../docs/experiments.md](../docs/experiments.md) and
[../COMMANDS.md](../COMMANDS.md).
