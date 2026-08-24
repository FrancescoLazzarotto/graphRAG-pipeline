# AGENTS.md

This file is the repository guide for coding agents. Read it first, then work from the nearest implementation file.

## What this repository is

`graphRAGPipelineExp1` is an experiment-oriented GraphRAG codebase. It has four real execution paths:

1. Knowledge graph construction from documents into Neo4j
2. Retrieval + answer generation through `graphrag-demo` / `python -m graphrag.cli`
3. Experiment and evaluation workflows that compare strategies, models, and resource use
4. Two interactive demos used for domain-expert sessions

The repository is used for practical experiments, not only for demos. Many commands write
artifacts that are later analyzed or included in paper-style reports, so output names and
schemas are part of the contract.

## How to read the project

Start from the owning file for the behavior you want to change:

- [README.md](README.md): user-facing documentation and canonical run examples
- [CLAUDE.md](CLAUDE.md): environment, architecture, conventions, known issues
- [docs/code_audit_2026-08-15.md](docs/code_audit_2026-08-15.md): current catalogue of known logic defects
- [pyproject.toml](pyproject.toml): package metadata, dependencies, CLI entry point
- [src/graphrag/cli.py](src/graphrag/cli.py): public command-line interface and experiment orchestration
- [src/graphrag/config.py](src/graphrag/config.py): `AgentConfig` / `KGConfig` — every tunable, each with the measurement that motivated its default
- [src/graphrag/strategies.py](src/graphrag/strategies.py): the eight retrieval presets, single source of truth
- [src/graphrag/agent/core.py](src/graphrag/agent/core.py): the LangGraph state machine
- [src/graphrag/agent/evidence.py](src/graphrag/agent/evidence.py): numbered evidence, citation gate, quote gate, reference lists
- [src/graphrag/agent/memory.py](src/graphrag/agent/memory.py): intra-session conversational memory (demo only)
- [src/graphrag/kg/retriever.py](src/graphrag/kg/retriever.py): search terms, channels, anchors, triple ranking
- [src/graphrag/kg/manager.py](src/graphrag/kg/manager.py): Cypher, full-text and vector lookups, retries
- [src/graphrag/llm/manager.py](src/graphrag/llm/manager.py): local HF / vLLM backends, language detection, domain gate
- [src/graphrag/llm/prompts.py](src/graphrag/llm/prompts.py): every prompt in the system
- [src/graphrag/text_rag/](src/graphrag/text_rag/): TF-IDF and dense (FAISS) text channels
- [src/graphrag/embeddings.py](src/graphrag/embeddings.py): the shared multilingual encoder client
- [src/graphrag/experiments/](src/graphrag/experiments/): experiment runner and resource monitoring
- [kg_pipeline/main.py](kg_pipeline/main.py): KG pipeline entry point with staged checkpointing
- [kg_pipeline/config.yaml](kg_pipeline/config.yaml): pipeline defaults, stage settings, paths
- [scripts/run_retrieval_matrix.py](scripts/run_retrieval_matrix.py): Standard-RAG vs GraphRAG matrix with telemetry
- [scripts/analyze_experiments.py](scripts/analyze_experiments.py): per-run analysis
- [scripts/analyze_resource_usage.py](scripts/analyze_resource_usage.py): telemetry aggregation across runs
- [evaluation/README.md](evaluation/README.md): gold dataset building, retrieval metrics, judge, optional RAGAS
- [evaluation/scripts/score_gold_run.py](evaluation/scripts/score_gold_run.py): the paper scorer (two channels, two levels)

## Repository structure

- `src/graphrag/`: main package (`cli`, `agent/`, `kg/`, `llm/`, `text_rag/`, `experiments/`)
- `kg_pipeline/`: document-to-KG pipeline with checkpointed stages
- `scripts/`: smoke checks, experiment runners, KG repair passes, index builders, demos, SLURM templates
- `evaluation/`: `evalkit` toolkit, gold datasets, fixtures, evaluation tests
- `tests/`: core unit tests for the agent, evidence, memory, domain gate, definitions
- `docs/`: cluster guidance, plans, audits, worklogs
- `artifacts/`, `exp_results*/`: generated experiment and evaluation outputs
- `kg_pipeline/artifacts/`: pipeline run directories, stage outputs, logs
- `logs/`: ad hoc run logs and matrix logs

## Execution model

### 1) GraphRAG CLI

`python -m graphrag.cli` (or `graphrag-demo`) for single-question or batched runs. It is the
**only** runner that exposes the full `AgentConfig` surface.

Important behavior from `src/graphrag/cli.py`:

- `--llm` enables generation; without it the agent runs retrieval-only paths.
- `--vllm` switches from local HF loading to an OpenAI-compatible vLLM endpoint.
- `--strategies` selects presets; in single-question mode only the first is applied.
- `--experiment` runs a batch over questions and strategies and exports structured artifacts.
- `--questions-file` accepts `.txt` (optionally `Q01<TAB>question`), `.json` (the gold's
  `{"queries": [...]}` shape), `.jsonl` and `.csv`. **Declaring `query_id` is what lets the
  evaluator join by id instead of by question text** — the text join is fragile and warns loudly.
- Retrieval and answer behaviour flags: `--vector-retrieval`, `--seed-from-retrieved`,
  `--subgraph-seed-count`, `--subgraph-limit`, `--drop-predicates`, `--cite-evidence`,
  `--prefer-verbatim-definitions`, `--enable-domain-gate`, `--allow-parametric-fallback`,
  `--focused-answer`, `--complexity`, `--enforce-language`, `--legacy-insufficiency-wording`.
- `--enable-decomposition-step` and `--enable-adaptive-routing-step` add extra LLM calls before
  retrieval; enable only when the experiment needs them.

### 2) KG pipeline

`kg_pipeline/main.py` runs stages 0–6 and checkpoints outputs so it can resume.
`--stage <name>` runs everything **up to and including** that stage, reusing earlier artifacts.

Stage outputs inside the run directory:

- `stage0_documents.json`
- `stage1_chunks.json`
- `stage2_ner.json`
- `stage3_triples_raw.json`, `stage3_acronyms.json`
- `stage3_checkpoint.json`, `stage3_checkpoint_info.json`
- `stage4_triples_resolved.json`, `stage4_registry.json`, `stage4_merge_approved.json`
- `stage5_triples_linked.json`
- `stage6_neo4j_summary.json`
- `failed_chunks.jsonl`, `new_labels.log`, `pipeline.log`
- `run_metadata.json` plus snapshots of `config.yaml` and the relation vocab

Post-ingestion, in order: `scripts/kg_postprocess.py --passes 1,2,3,4,5`, then
`scripts/kg_search_index.py` (full-text), then `scripts/kg_vector_index.py` (vector carriers).
Retrieval quality depends on the last two having been run against the live graph.

### 3) Experiments and analysis

Two runners write the same artifact shape but do **not** offer the same configuration surface:

| | `python -m graphrag.cli --experiment` | `scripts/run_retrieval_matrix.py` |
|---|---|---|
| GraphRAG strategies | yes | yes |
| Standard-RAG baselines | no | yes |
| Resource telemetry | no | yes |
| `query_id` in results | yes | **no** |
| Full `AgentConfig` (vector, citations, gate, …) | yes | **no** — 8 fields only |

Expected run outputs:

- `results.jsonl`, `results.csv`, `summary.txt`, `summary.json`, `config.json`
- `resource_samples.jsonl`, `resource_summary.json` (matrix runner only)

### 4) Evaluation

`PYTHONPATH=evaluation python -m evalkit.cli <subcommand>` — `build-dataset`, `retrieval`,
`text`, `judge`, `judge-compare`, `ragas`, `kg`, `gold-triples`, `report-experiment`,
`report-project`, `baseline-update`.

For paper numbers use `evaluation/scripts/score_gold_run.py`, which scores both the retrieval
channel and the answer-text channel at concept and grounding level. The two levels are never
merged into one number — that separation is the finding, not an implementation detail.

### 5) Demos

- `streamlit run product/app.py` — multi-chat expert console with intra-session memory,
  citations, verbatim definitions and the domain gate.
- `python product/console.py --strategy hybrid` — the same stack, console-only.

Both build their own `AgentConfig` inline; they do not read the CLI flags.

## Working conventions

- Prefer the Conda environment `graphllm` for local work; `conda run -n graphllm ...` for automation.
- The CLI entry point is `graphrag-demo`; `python -m graphrag.cli` is also valid and more reliable.
- Do not edit generated artifacts unless the task explicitly asks for it.
- Preserve existing outputs and logs unless you are fixing those files directly.
- Keep changes local to the controlling code path; avoid broad refactors during a bug fix.
- Prefer root-cause fixes over surface patches.
- Prompts live only in `PromptLibrary`. A prompt string in a backend breaks the invariant that
  vLLM and local HF are comparable.
- Retrieval presets live only in `strategies.py`. Both runners import them; keeping a second
  list anywhere is how they drift.

## Known repository details

- `scripts/smoke_check.py` reads exported environment variables; it does not auto-load `.env`.
- `graphrag-demo` may point to a stale interpreter if a user-local shim is on PATH; if that
  happens, use `conda run -n graphllm python -m graphrag.cli`.
- `import vllm` is broken inside `graphllm`; serve models from the `vllm-serve` virtualenv.
- Retrieval needs APOC on the Neo4j instance (`apoc.map.removeKey` in every node projection).
- `RAGState` is a LangGraph channel schema: a key returned by a node but not declared there is
  silently dropped from the final state.
- `requirements*.txt` and `pyproject.toml` are missing `pymupdf4llm`, `gliner`, `openai`,
  `pyyaml` and `requests`; the `graphllm` env has them, a clean install does not.
- CI runs `python -m compileall src scripts` (not `kg_pipeline` / `evaluation`) plus the full
  test suite on `requirements-cpu.txt`.
- Real sample outputs exist under `artifacts/experiments/`, `exp_results*/` and
  `kg_pipeline/artifacts/`; use them when you need concrete examples of a schema.

## Validation habits

After an edit, use the smallest check that can falsify the change:

- Documentation only: `git diff --check -- README.md AGENTS.md CLAUDE.md`
- Python logic changes: targeted smoke script or a narrow module run
- CLI or pipeline changes: the smallest relevant command that exercises the touched path
- Retrieval or metric changes: the unit tests, which are fast and cover the tricky parts

```bash
pytest tests/ kg_pipeline/tests/ evaluation/tests/ -q
python scripts/smoke_check.py
python scripts/smoke_text_rag.py docs/ --query "Summarize the cluster setup" --top-k 4
python scripts/smoke_kg_retriever.py
conda run -n graphllm python -m graphrag.cli --help
conda run -n graphllm python -m kg_pipeline.main --config kg_pipeline/config.yaml --env-file kg_pipeline/.env --log-level INFO
```

If you touch experiment code, also inspect a recent artifact folder and confirm the output
names still match the analysis scripts.

## Repository-specific implementation notes

- Retrieval and experiment code is performance-sensitive; avoid adding extra LLM calls or broad
  abstractions unless required.
- In `scripts/run_retrieval_matrix.py`, the runner may checkpoint or finalize outputs during long
  runs. Do not assume the output directory is empty until the process finishes. Note also that it
  cannot express the newer `AgentConfig` options.
- In `src/graphrag/agent/core.py`, do not reintroduce decomposition or routing steps unless the
  task explicitly requires them; they add latency and are often not needed.
- In `src/graphrag/llm/refusal.py`, the two marker lists are substring-matched over the whole
  answer and feed both an answer-replacement path and a published metric. Adding a generic phrase
  there silently rewrites correct answers.
- In `kg_pipeline/main.py`, the stage order matters because downstream stages read stage artifacts
  from earlier stages.
- In `kg_pipeline/stages/resolution.py`, the merge cache stores raw group indices; it is only valid
  for an unchanged stage-3 output.
- In `evaluation/`, the evaluation flow assumes run outputs already exist and a gold file has been
  prepared. A join that silently degrades to matching on question text has produced wrong numbers
  in this project before — the join report at WARNING level is there for that reason.

## Before you change behaviour

Read [docs/code_audit_2026-08-15.md](docs/code_audit_2026-08-15.md). It lists the known logic
defects with file, line and failure mode. Several behaviours that look like bugs on first reading
are already catalogued there; several that look correct are not.

## If you are unsure

- Re-read the nearest implementation file, not the whole repo.
- Check the latest artifact folder or the corresponding analyzer before changing experiment code.
- If the change touches evaluation, consult [evaluation/README.md](evaluation/README.md) first.
