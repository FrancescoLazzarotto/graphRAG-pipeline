# COMMANDS.md — Project Command Reference

Quick reference for all executable commands in this repository, organised by functional area.

---

## 1. GraphRAG CLI — `graphrag.cli`

Main entry point for single-question retrieval/generation and batch experiments.

> If `graphrag-demo` returns exit code 126 (stale shim), use `conda run -n graphllm python -m graphrag.cli` instead.

### Single question — retrieval only

```bash
conda run -n graphllm python -m graphrag.cli \
  --question "What is the relationship between X and Y?" \
  --entity "X" \
  --strategies default
```

### Single question with local LLM

```bash
conda run -n graphllm python -m graphrag.cli \
  --question "What is the relationship between X and Y?" \
  --entity "X" \
  --llm \
  --model-id meta-llama/Meta-Llama-3-8B-Instruct
```

### Single question with vLLM (OpenAI-compatible endpoint)

```bash
conda run -n graphllm python -m graphrag.cli \
  --question "What is the relationship between X and Y?" \
  --llm \
  --vllm \
  --vllm-base-url http://localhost:8000/v1 \
  --model-id "Qwen/Qwen2.5-32B-Instruct-AWQ"
```

### Batch experiment over a question file

```bash
conda run -n graphllm python -m graphrag.cli \
  --experiment \
  --questions-file artifacts/tmp/graphrag_test_suite_questions.txt \
  --strategies "default,text_only,no_retrieval,text_plus_triples,neighbors_focus,subgraph_2hop,shortest_path" \
  --llm \
  --vllm \
  --vllm-base-url http://localhost:8000/v1 \
  --model-id "Qwen/Qwen2.5-32B-Instruct-AWQ" \
  --output-dir artifacts/experiments \
  --experiment-tag my_run
```

To monitor progress while running in background:

```bash
conda run -n graphllm python -m graphrag.cli ... \
  --experiment-tag my_run 2>&1 | tee artifacts/experiments/run_my_run.log &

tail -f artifacts/experiments/run_my_run.log
```

> **Note on log visibility**: stdout is buffered until the first LLM call completes (typically 5–10 s). The log will appear empty at first — this is normal.

### CLI flags reference

| Flag | Effect |
|------|--------|
| `--question` | Question to answer (default: Italian placeholder) |
| `--entity` | Seed entity for graph traversal (empty = auto-detected) |
| `--llm` | Enable LLM generation; without it, only retrieval runs |
| `--vllm` | Use a vLLM OpenAI-compatible endpoint instead of loading a local HF model |
| `--vllm-base-url` | vLLM API base URL (default: `http://localhost:8000/v1`) |
| `--model-id` | HuggingFace or vLLM model identifier |
| `--llm-warmup` | Pre-load model at startup |
| `--strategies` | Comma-separated retrieval strategy presets (see §14) |
| `--text-docs-dir` | Document directory for `text_only` BM25 indexing (PDF/txt/md). If omitted, auto-discovers from the latest KG pipeline `stage0` artifacts |
| `--enable-decomposition-step` | Add an LLM call before retrieval (higher latency) |
| `--enable-adaptive-routing-step` | Add adaptive routing before retrieval (higher latency) |
| `--max-new-tokens` | Maximum generated tokens per response (default: 256) |
| `--max-context-tokens` | Maximum tokens for compressed context before generation (default: 1000) |
| `--gpu-memory-fraction` | Fraction of GPU memory reserved for the model (default: 0.92) |
| `--allow-large-model-fp16-fallback` | Allow fp16 fallback for large models when 4-bit loading fails |
| `--experiment` | Batch mode: writes structured output artifacts |
| `--questions-file` | UTF-8 text file with one question per line |
| `--runs-per-strategy` | Repetitions per strategy (default: 1) |
| `--output-dir` | Experiment output directory (default: `artifacts/experiments`) |
| `--experiment-tag` | Identifier tag for the run |
| `--recursion-limit` | Maximum LangGraph recursion steps before aborting (default: 50) |

---

## 1b. Streamlit Demo — `scripts/demo_app.py`

Browser UI over the GraphRAG agent (text box, spinner, answer + sources). Logs every exchange to `artifacts/demo_sessions/`.

```bash
conda run -n graphllm streamlit run scripts/demo_app.py --server.address 0.0.0.0 --server.port 8501
```

From your local machine, tunnel then browse `http://localhost:8501`:

```bash
ssh -L 8501:localhost:8501 <user>@<server>
```

---

## 2. KG Pipeline — `kg_pipeline/main.py`

Builds the knowledge graph from source documents and ingests it into Neo4j. Runs across 7 sequential stages with JSON checkpoint recovery — the pipeline can be resumed from any stage.

### Full pipeline

```bash
conda run -n graphllm python -m kg_pipeline.main \
  --config kg_pipeline/config.yaml \
  --env-file kg_pipeline/.env \
  --log-level INFO
```

### Single stage

```bash
conda run -n graphllm python -m kg_pipeline.main \
  --config kg_pipeline/config.yaml \
  --env-file kg_pipeline/.env \
  --stage ner
```

### Single document

```bash
conda run -n graphllm python -m kg_pipeline.main \
  --config kg_pipeline/config.yaml \
  --env-file kg_pipeline/.env \
  --single-doc documento.pdf
```

### Dry-run (no writes)

```bash
conda run -n graphllm python -m kg_pipeline.main \
  --config kg_pipeline/config.yaml \
  --dry-run
```

### Parameters

| Flag | Effect |
|------|--------|
| `--config` | Path to configuration file (default: `kg_pipeline/config.yaml`) |
| `--env-file` | `.env` file with Neo4j credentials and other secrets |
| `--run-dir` | Specific run directory; if empty, a timestamped one is created |
| `--single-doc` | Process a single document (filename or doc_id) |
| `--stage` | Run only the specified stage: `all` `ingestion` `chunking` `ner` `llm` `resolution` `linking` `neo4j` |
| `--dry-run` | Simulate execution without writing output |
| `--log-level` | Log verbosity: `DEBUG` `INFO` `WARNING` (default: `INFO`) |

### Stages and produced artifacts

| Stage | Description | Artifact |
|-------|-------------|----------|
| 0 — ingestion | PDF → markdown | `stage0_documents.json` |
| 1 — chunking | Token-windowed paragraph splitting | `stage1_chunks.json` |
| 2 — ner | Named entity recognition (GLiNER) | `stage2_ner.json` |
| 3 — llm | Triple extraction with LLM | `stage3_triples_raw.json`, `stage3_acronyms.json` |
| 4 — resolution | Entity resolution (embedding + Jaccard) | `stage4_triples_resolved.json`, `stage4_registry.json` |
| 5 — linking | Triple linking | `stage5_triples_linked.json` |
| 6 — neo4j | Neo4j ingestion | `stage6_neo4j_summary.json` |

Additional outputs: `failed_chunks.jsonl`, `new_labels.log`, `pipeline.log`,
`run_metadata.json` (seed, models, git commit, timestamps) plus snapshots of
`config.yaml` and the relation vocab — every run directory is self-describing
and reproducible.

> **Stage 3 checkpoint**: progress is saved every N chunks to `stage3_checkpoint.json` (written atomically). Re-running without deleting this file resumes from the last saved chunk; triples from chunks past the last completed checkpoint are dropped on resume so recovery never duplicates.

---

## 3. Experiment Matrix — `scripts/run_retrieval_matrix.py`

Compares retrieval strategies and LLM configurations across a question set. Results are written to `artifacts/experiments/<timestamp>_<tag>/`.

```bash
python scripts/run_retrieval_matrix.py \
  --questions-file questions.txt \
  --models "7b,32b" \
  --strategies "default,text_plus_triples" \
  --output-dir artifacts/experiments
```

### With vLLM and tag

```bash
python scripts/run_retrieval_matrix.py \
  --questions-file questions.txt \
  --strategies "default,text_plus_triples,subgraph_2hop" \
  --llm \
  --vllm \
  --model-id Qwen/Qwen2.5-32B-Instruct \
  --output-dir artifacts/experiments \
  --experiment-tag strategy_comparison
```

### vLLM demo
```bash
conda run -n graphllm python -m graphrag.cli \
  --question "According to the Regulation, what is the definition of 'food business operator'?" \
  --strategies "hybrid" \
  --llm \
  --vllm \
  --vllm-base-url http://localhost:8000/v1 \
  --model-id "Qwen/Qwen2.5-32B-Instruct-AWQ"
```

### Graph-only (skip standard RAG)

```bash
python scripts/run_retrieval_matrix.py \
  --questions-file questions.txt \
  --skip-standard \
  --strategies "neighbors_focus,subgraph_2hop,shortest_path"
```

### Parameters

| Flag | Effect |
|------|--------|
| `--questions-file` | File with one question per line |
| `--strategies` | Comma-separated strategy presets |
| `--models` | Model presets (`7b`, `32b`, etc.) |
| `--llm` | Enable LLM generation |
| `--vllm` | Use vLLM endpoint |
| `--model-id` | Specific model override |
| `--runs-per-strategy` | Repetitions per strategy |
| `--output-dir` | Output directory |
| `--experiment-tag` | Identifier tag |
| `--skip-standard` | Skip standard RAG, run GraphRAG only |
| `--skip-graph` | Skip GraphRAG, run standard RAG only |
| `--performance-profile` | `auto` / `default` / `production_fast` |
| `--enable-decomposition-step` | Add LLM decomposition step |
| `--enable-adaptive-routing-step` | Add adaptive routing step |
| `--max-new-tokens` | Maximum tokens per response |
| `--gpu-memory-fraction` | GPU memory fraction reserved |
| `--allow-large-model-fp16-fallback` | fp16 fallback for large models |
| `--llm-warmup` | Pre-load model before runs |
| `--smoke` | Quick test on a reduced question set |
| `--smoke-questions` | Number of questions in smoke mode (default: 2) |

### Output artifacts

```
artifacts/experiments/<timestamp>_<tag>/
  results.jsonl
  results.csv
  summary.txt
  summary.json
  config.json            # CLI args + fully resolved AgentConfig per strategy
  resource_samples.jsonl
  resource_summary.json
```

> **Traceability**: `config.json` records the exact resolved configuration for
> each strategy, so any metric in `summary.json` can be traced back to the
> settings that produced it.

---

## 4. A/B Fast Profile — `scripts/run_ab_fast_profile.py`

Compares `default` vs `production_fast` profile for the 32B model and reports latency and quality delta.

```bash
python scripts/run_ab_fast_profile.py \
  --model-id Qwen/Qwen2.5-32B-Instruct \
  --questions-file questions_matrix_long.txt \
  --questions-count 10 \
  --graph-strategies default \
  --vllm \
  --output-dir artifacts/experiments \
  --report-dir artifacts/evaluation
```

---

## 5. Results Analysis — `scripts/analyze_experiments.py`

Analyses run artifacts and produces a ranked metric report.

```bash
python scripts/analyze_experiments.py \
  --results-dir artifacts/experiments \
  --output-csv results_ranked.csv
```

```bash
python scripts/analyze_experiments.py artifacts/experiments/20240601_120000_my_run
```

---

## 6. Resource Usage Analysis — `scripts/analyze_resource_usage.py`

Aggregates GPU/CPU telemetry from multiple runs.

```bash
python scripts/analyze_resource_usage.py artifacts/experiments \
  --tag-contains confronto \
  --output-csv resource_report.csv
```

---

## 7. Matrix Analysis — `scripts/analyze_matrix.py`

Aggregated comparative analysis across multiple experiment runs/folders.

```bash
python scripts/analyze_matrix.py \
  --root artifacts/experiments \
  --tag-contains strategy_comparison \
  --output-csv matrix_summary.csv
```

---

## 8. Entity Re-merge — `scripts/remerge_entities.py`

Re-runs entity resolution and linking on existing stage 3 output without redoing NER or LLM extraction.  
Useful for tuning similarity thresholds without re-running the full pipeline.

```bash
python scripts/remerge_entities.py \
  --run-dir kg_pipeline/artifacts/run_20240601_120000 \
  --similarity-threshold 0.90 \
  --context-jaccard-floor 0.15
```

| Flag | Effect |
|------|--------|
| `--run-dir` | Run directory containing stage 3 output (required) |
| `--output-dir` | Alternative output directory for stages 4/5 (optional) |
| `--embedding-model` | SentenceTransformer model for resolution |
| `--similarity-threshold` | Cosine similarity threshold (default: 0.88) |
| `--context-jaccard-floor` | Minimum context Jaccard threshold (default: 0.15) |
| `--base-url` | vLLM endpoint URL |
| `--model-name` | vLLM model name |

---

## 9. Question Generation — `scripts/generate_questions.py`

Automatically generates a test question suite from pipeline chunks/documents.

### Generate suite from an existing run

```bash
python scripts/generate_questions.py generate \
  --run-dir kg_pipeline/artifacts/run_20240601_120000 \
  --output artifacts/tmp/graphrag_test_suite.json
```

### Extract questions to plain text (one per line)

```python
import json
from pathlib import Path

data = json.loads(Path("artifacts/tmp/graphrag_test_suite.json").read_text())
questions = [q["question"] for q in data if q.get("question", "").strip()]
Path("artifacts/tmp/graphrag_test_suite_questions.txt").write_text(
    "\n".join(questions), encoding="utf-8"
)
print(f"{len(questions)} questions written")
```

### For a specific document

```bash
python scripts/generate_questions.py generate \
  --run-dir kg_pipeline/artifacts/run_20240601_120000 \
  --doc my_document.pdf \
  --output artifacts/tmp/suite_doc.json
```

### Statistics on an existing suite

```bash
python scripts/generate_questions.py stats \
  --input artifacts/tmp/graphrag_test_suite.json
```

---

## 10. KG Visualisation — `scripts/visualize_kg.py`

Generates an interactive HTML view of the knowledge graph from Neo4j.

```bash
python scripts/visualize_kg.py \
  --output artifacts/tmp/kg_viz.html
```

Requires `NEO4J_URL`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` to be set.

---

## 11. Evaluation — `evaluation/`

### Build evaluation dataset

Joins experiment results with gold-standard labels.

```bash
python evaluation/build_eval_dataset.py \
  --input artifacts/experiments \
  --gold-file evaluation/gold_questions_template.csv \
  --output artifacts/evaluation/eval_dataset.csv
```

| Flag | Effect |
|------|--------|
| `--input` | Experiments root, single run folder, or `results.csv` |
| `--gold-file` | Gold CSV with at least `question` and `ground_truth` columns |
| `--tag-contains` | Optional filter on run folder name |
| `--output` | Output CSV path |
| `--smoke` | Smoke mode on local fixtures |

### Retrieval metrics

```bash
python evaluation/retrieval_metrics.py \
  --input artifacts/evaluation/eval_dataset.csv \
  --k 5 \
  --save-csv artifacts/evaluation/metrics.csv \
  --save-json artifacts/evaluation/metrics.json
```

| Flag | Effect |
|------|--------|
| `--input` | CSV produced by `build_eval_dataset.py` |
| `--k` | Top-k for precision/recall/hit/NDCG |
| `--n-bootstrap` | Bootstrap resamples (default: 1000) |
| `--ci` | Confidence interval level (default: 0.95) |
| `--save-csv` / `--save-json` | Aggregate metrics output path |
| `--save-row-csv` | Per-row metrics output path |

### RAGAS evaluation (optional)

```bash
python evaluation/run_ragas_eval.py \
  --input artifacts/evaluation/eval_dataset.csv \
  --save-summary-json artifacts/evaluation/ragas_summary.json \
  --save-row-csv artifacts/evaluation/ragas_rows.csv
```

### LLM-as-a-Judge — `evalkit.cli judge`

Scores generated answers against the gold set on `answer_correctness`,
`groundedness`, `relevance` (score 0–1 + rationale per row, with bootstrap CIs).
The judge model is pluggable via `--backend`:

| `--backend` | Auth | Cost |
|---|---|---|---|---|
| `claude_code` | Claude **subscription** (Pro/Max, OAuth) | $0 extra |  
| `api` | `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | ~$3–15 / full run | 
| `vllm` / `local_hf` | local model | $0 |

> **Run from the `evaluation/` directory** (`cd evaluation`) so `evalkit` is
> importable, otherwise `python -m evalkit.cli` fails on import.

#### Prerequisite for `claude_code`: make `claude` reachable

The `claude` binary is **not on `PATH`** — it ships bundled inside the VSCode
extension. Create a stable symlink **once** (auto-picks the newest extension
version, survives updates):

```bash
ln -sf "$(ls -d /home/flazzarotto/.vscode-server/extensions/anthropic.claude-code-*/resources/native-binary/claude | sort -V | tail -1)" ~/.local/bin/claude
claude --version    # → e.g. "2.1.193 (Claude Code)"
```

Alternative (no symlink): pass `--claude-bin <path>` per run, or
`export CLAUDE_CODE_BIN=<path>`. The CLI must be logged in to your claude.ai
account (it reuses `~/.claude/.credentials.json`).

#### Smoke run (5 rows → 1 call, ~$0 extra)

```bash
cd evaluation
conda run -n graphllm python -m evalkit.cli judge \
  --input ../artifacts/evaluation/smoke_eval_dataset.csv \
  --backend claude_code --model haiku --batch-size 8 \
  --out ../artifacts/evaluation/judge_smoke
```

Output in `judge_smoke/`: `judge_summary.json` (per-rubric means + CIs) and
`judge_rows.jsonl` (one scored row per question).

#### Full run + judge agreement (paper)

```bash
cd evaluation
# Haiku and Sonnet as two independent judges (batched + resumable):
conda run -n graphllm python -m evalkit.cli judge \
  --input ../artifacts/evaluation/eval_dataset_gold23q_v2.csv \
  --backend claude_code --model haiku --batch-size 8 --resume \
  --out ../artifacts/evaluation/judge_haiku
conda run -n graphllm python -m evalkit.cli judge \
  --input ../artifacts/evaluation/eval_dataset_gold23q_v2.csv \
  --backend claude_code --model sonnet --batch-size 8 --resume \
  --out ../artifacts/evaluation/judge_sonnet

# Inter-judge agreement table (robustness):
conda run -n graphllm python -m evalkit.cli judge-compare \
  --a ../artifacts/evaluation/judge_haiku --b ../artifacts/evaluation/judge_sonnet \
  --label-a haiku --label-b sonnet \
  --out ../artifacts/evaluation/judge_compare
```

> **Reproducible paper numbers**: regenerate the final table with
> `--backend api --provider anthropic --model claude-sonnet-4-6` (needs
> `ANTHROPIC_API_KEY`). The subscription path is account-bound and **not**
> reproducible by reviewers (grey area in Anthropic's usage policy).

#### Flags

| Flag | Effect |
|------|--------|
| `--input` | Eval dataset CSV (from `build_eval_dataset.py`) |
| `--backend` | `vllm` / `local_hf` / `api` / `claude_code` (default `vllm`) |
| `--model` | Judge model. `claude_code`: `haiku` / `sonnet`. `api`: full id, e.g. `claude-sonnet-4-6` |
| `--provider` | `anthropic` / `openai` (only for `--backend api`) |
| `--claude-bin` | Path to the `claude` binary (or set `CLAUDE_CODE_BIN`) |
| `--rubrics` | Comma list (default `answer_correctness,groundedness,relevance`) |
| `--batch-size` | Rows per call; always batched for `claude_code` (rate-limit friendly) |
| `--resume` | Skip rows already in `<out>/judge_rows.jsonl` (recover an interrupted run) |
| `--out` | Output directory |

> `claude_code` env overrides: `CLAUDE_CODE_BIN` (binary path),
> `CLAUDE_CODE_TIMEOUT` (seconds, default 300), `CLAUDE_CODE_EXTRA_ARGS`.

---

## 11b. KG Post-processing — `scripts/kg_postprocess.py`

Unified entrypoint for the four Neo4j repair passes (hub cleanup, relation
consolidation, RELATED_TO reclassification via LLM, property enrichment, ...).
The passes are distinct post-processing rounds implemented in
`kg_repair.py`..`kg_repair4.py`; run them through this script, not directly.

```bash
conda run -n graphllm python scripts/kg_postprocess.py            # all passes
conda run -n graphllm python scripts/kg_postprocess.py --passes 3,4
```

Requires `NEO4J_*` and `VLLM_*` env vars (each pass loads `kg_pipeline/.env`).

Related: `scripts/kg_evaluator.py` queries the live Neo4j graph and writes a
structural quality report to `artifacts/kg_reports/` (used for manual/LLM
review of graph quality after ingestion or repair).

---

## 12. Smoke tests and health checks

| Command | What it checks |
|---------|----------------|
| `python scripts/smoke_check.py` | Neo4j + LLM connectivity (reads exported env vars, does not auto-load `.env`) |
| `python scripts/smoke_text_rag.py docs/ --query "..." --top-k 4` | BM25 text retrieval on a document directory |
| `python scripts/smoke_kg_retriever.py` | KG retrieval against Neo4j |
| `python scripts/smoke_test_pipeline.py` | Quick smoke run of the KG pipeline |
| `python scripts/run_pipeline_smoke_full.py` | Full end-to-end pipeline smoke |
| `pytest kg_pipeline/tests/test_pipeline.py -v` | KG pipeline unit/integration tests |
| `pytest evaluation/tests/test_metrics.py -v` | Evaluation metric tests |

---

## 13. SLURM — cluster execution

| Script | What it launches |
|--------|-----------------|
| `sbatch scripts/run_kg_pipeline.sbatch` | KG pipeline in background (avoids hang on notebook disconnect) |
| `sbatch scripts/run_graphrag.sbatch` | GraphRAG demo on GPU cluster |
| `sbatch scripts/run_graphrag_cpu.sbatch` | GraphRAG demo on CPU cluster |
| `sbatch scripts/run_experiment_matrix_gpu.sbatch` | Experiment matrix on GPU cluster |
| `bash scripts/submit_matrix_from_env.sh` | Submit matrix reading parameters from env vars |

---

## 14. Retrieval strategies

Pass to `--strategies` as a single value or comma-separated list:

| Strategy | Description | KG used | Text used |
|----------|-------------|---------|-----------|
| `default` | Full GraphRAG — nodes, triples, neighbors, subgraph (1-hop), shortest path | yes | no |
| `text_only` | Standard RAG — BM25/TF-IDF over indexed documents (no KG) | no | yes |
| `no_retrieval` | Zero-shot LLM — no retrieval at all (baseline) | no | no |
| `text_plus_triples` | KG direct triples only (nodes + triples, no subgraph/neighbors/path) | triples only | no |
| `neighbors_focus` | KG triples + direct entity neighbors (no subgraph/path) | yes | no |
| `subgraph_2hop` | KG triples + 2-hop subgraph (larger context than default) | yes | no |
| `shortest_path` | KG triples + shortest path between entity pairs | yes | no |

> **Paper note**: `text_only` is the BM25 sparse-retrieval baseline; `no_retrieval` is the zero-shot LLM baseline. All other strategies are GraphRAG variants.

---

## 15. Required environment variables

```bash
NEO4J_URL="bolt://localhost:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="..."
NEO4J_DATABASE="..."          # optional
HF_TOKEN="..."                # for gated HuggingFace models
VLLM_BASE_URL="http://localhost:8000/v1"
VLLM_MODEL_NAME="Qwen/Qwen2.5-32B-Instruct-AWQ"
VLLM_API_KEY="..."            # or OPENAI_API_KEY
```

> `scripts/smoke_check.py` reads exported shell env vars — it does **not** auto-load `.env`.  
> Always pass `--env-file kg_pipeline/.env` when running the KG pipeline.

---

## 16. Typical end-to-end workflow

```bash
# 1. Build KG from documents
conda run -n graphllm python -m kg_pipeline.main \
  --config kg_pipeline/config.yaml \
  --env-file kg_pipeline/.env

# 2. Generate test questions from the KG artifacts
python scripts/generate_questions.py generate \
  --run-dir kg_pipeline/artifacts/run_<timestamp> \
  --output artifacts/tmp/graphrag_test_suite.json

# 3. Extract to plain text
python -c "
import json; from pathlib import Path
d = json.loads(Path('artifacts/tmp/graphrag_test_suite.json').read_text())
qs = [q['question'] for q in d if q.get('question','').strip()]
Path('artifacts/tmp/graphrag_test_suite_questions.txt').write_text('\n'.join(qs))
print(len(qs), 'questions')
"

# 4. Check vLLM is running
curl -s http://localhost:8000/v1/models | python -m json.tool

# 5. Run the full strategy comparison
conda run -n graphllm python -m graphrag.cli \
  --experiment \
  --questions-file artifacts/tmp/graphrag_test_suite_questions.txt \
  --strategies "text_only,no_retrieval,default,text_plus_triples,neighbors_focus,subgraph_2hop,shortest_path" \
  --llm --vllm \
  --vllm-base-url http://localhost:8000/v1 \
  --model-id "Qwen/Qwen2.5-32B-Instruct-AWQ" \
  --output-dir artifacts/experiments \
  --experiment-tag strategy_comparison_v1

# 6. Inspect results
cat artifacts/experiments/<timestamp>_strategy_comparison_v1/summary.txt
```
