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
