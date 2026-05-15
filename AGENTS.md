# AGENTS.md

This file is the repository guide for coding agents. Read it first, then work from the nearest implementation file.

## What this repository is

`graphRAGPipelineExp1` is an experiment-oriented GraphRAG codebase. It has three real execution paths:

1. Knowledge graph construction from documents into Neo4j
2. Retrieval + answer generation through `graphrag-demo`
3. Experiment/evaluation workflows that compare strategies, models, and resource use

The repository is used for practical experiments, not only for demos. Many commands write artifacts that are later analyzed or included in paper-style reports.

## How to read the project

Start from the owning file for the behavior you want to change:

- [README.md](README.md): current user-facing documentation and canonical run examples
- [pyproject.toml](pyproject.toml): package metadata, dependencies, and the CLI entry point
- [src/graphrag/cli.py](src/graphrag/cli.py): public command-line interface and experiment orchestration
- [src/graphrag/agent/core.py](src/graphrag/agent/core.py): GraphRAG agent routing and retrieval behavior
- [src/graphrag/experiments/](src/graphrag/experiments/): experiment runner and resource monitoring helpers
- [src/graphrag/kg/](src/graphrag/kg/): Neo4j manager and graph retrieval logic
- [src/graphrag/llm/](src/graphrag/llm/): local model / vLLM model management
- [kg_pipeline/main.py](kg_pipeline/main.py): KG pipeline entry point with staged checkpointing
- [kg_pipeline/config.yaml](kg_pipeline/config.yaml): pipeline defaults, stage settings, paths
- [scripts/run_retrieval_matrix.py](scripts/run_retrieval_matrix.py): batch experiment runner for standard RAG and GraphRAG
- [scripts/analyze_experiments.py](scripts/analyze_experiments.py): per-run analysis of outputs and metrics
- [scripts/analyze_resource_usage.py](scripts/analyze_resource_usage.py): aggregate telemetry across runs
- [evaluation/README.md](evaluation/README.md): gold dataset building, retrieval metrics, optional RAGAS evaluation

## Repository structure

- `src/graphrag/`: main package
- `kg_pipeline/`: document-to-KG pipeline with checkpointed stages
- `scripts/`: smoke checks, experiment runners, analyzers, SLURM templates
- `evaluation/`: evaluation dataset creation and metrics
- `docs/`: supplemental documentation such as cluster guidance
- `artifacts/`: generated experiment and evaluation outputs
- `kg_pipeline/artifacts/`: pipeline run directories, stage outputs, logs
- `logs/`: ad hoc run logs and matrix logs

## Execution model

### 1) GraphRAG CLI

Use `graphrag-demo` for single-question or batched retrieval/generation runs.

Important behavior from `src/graphrag/cli.py`:

- `--llm` enables generation; without it the agent can run retrieval-only paths.
- `--vllm` switches from local HF model loading to an OpenAI-compatible vLLM endpoint.
- `--enable-decomposition-step` and `--enable-adaptive-routing-step` add extra LLM calls before retrieval; they increase latency and should only be enabled when the experiment needs them.
- `--experiment` runs a batch over questions and strategies and exports structured artifacts.
- `--performance-profile` in `scripts/run_retrieval_matrix.py` can choose `auto`, `default`, or `production_fast` behavior.

### 2) KG pipeline

`kg_pipeline/main.py` is the document-to-graph pipeline. It runs in stages and checkpoints outputs so it can resume.

Stage outputs are written inside the selected run directory and typically include:

- `stage0_documents.json`
- `stage1_chunks.json`
- `stage2_ner.json`
- `stage3_triples_raw.json`
- `stage3_acronyms.json`
- `stage4_triples_resolved.json`
- `stage4_registry.json`
- `stage5_triples_linked.json`
- `stage6_neo4j_summary.json`
- `failed_chunks.jsonl`
- `new_labels.log`
- `pipeline.log`

### 3) Experiments and analysis

`scripts/run_retrieval_matrix.py` compares retrieval strategies and optional LLM configurations. It writes its final artifacts to an output directory named like `artifacts/experiments/<timestamp>_<tag>/`.

Expected run outputs include:

- `results.jsonl`
- `results.csv`
- `summary.txt`
- `summary.json`
- `resource_samples.jsonl`
- `resource_summary.json`

## Working conventions

- Prefer the Conda environment `graphllm` for local work.
- Use `conda run -n graphllm ...` for reproducible script or automation invocation.
- The CLI entry point is `graphrag-demo`; `python -m graphrag.cli` is also valid.
- Do not edit generated artifacts unless the task explicitly asks for it.
- Preserve existing outputs and logs unless you are fixing those files directly.
- Keep changes local to the controlling code path; avoid broad refactors during a bug fix.
- Prefer root-cause fixes over surface patches.

## Known repository details

- `scripts/smoke_check.py` reads exported environment variables; it does not auto-load `.env`.
- `graphrag-demo` may point to a stale interpreter if a user-local shim is on PATH; if that happens, use `conda run -n graphllm python -m graphrag.cli`.
- Graph retrieval experiments often produce the six files listed above plus per-run metadata inside the run directory.
- KG pipeline runs create checkpointed stage artifacts and `pipeline.log`.
- Real sample outputs already exist under `artifacts/experiments/` and `kg_pipeline/artifacts/`; use them when you need concrete examples.

## Validation habits

After an edit, use the smallest check that can falsify the change:

- Documentation only: `git diff --check -- README.md AGENTS.md`
- Python logic changes: targeted smoke script or a narrow module run
- CLI or pipeline changes: the smallest relevant command that exercises the touched path

Useful project checks:

```bash
python scripts/smoke_check.py
python scripts/smoke_text_rag.py docs/ --query "Summarize the cluster setup" --top-k 4
conda run -n graphllm python -m graphrag.cli --help
conda run -n graphllm python -m kg_pipeline.main --config kg_pipeline/config.yaml --env-file kg_pipeline/.env --log-level INFO
```

If you touch experiment code, also inspect a recent artifact folder and confirm the output names still match the analysis scripts.

## Repository-specific implementation notes

- Retrieval and experiment code is performance-sensitive; avoid adding extra LLM calls or broad abstractions unless required.
- In `scripts/run_retrieval_matrix.py`, the runner may checkpoint or finalize outputs during long runs. Do not assume the output directory is empty until the process finishes.
- In `src/graphrag/agent/core.py`, do not reintroduce decomposition or routing steps unless the task explicitly requires them; they add latency and are often not needed for retrieval behavior.
- In `kg_pipeline/main.py`, the stage order matters because downstream stages read stage artifacts from earlier stages.
- In `evaluation/README.md`, the evaluation flow assumes that run outputs already exist and that a gold file has been prepared.

## If you are unsure

- Re-read the nearest implementation file, not the whole repo.
- Check the latest artifact folder or the corresponding analyzer before changing experiment code.
- If the change touches evaluation, consult [evaluation/README.md](evaluation/README.md) first.
