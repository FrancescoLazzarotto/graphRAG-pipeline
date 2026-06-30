# Evaluation Workspace

This folder contains paper-oriented evaluation utilities for GraphRAG runs.

## Goals

The pipeline is organized in three layers:

1. Retrieval quality (component-level)
2. End-to-end answer quality (RAGAS, optional)
3. Ablation-ready summaries by strategy/model

## Files

- `gold_questions_template.csv`: template for a gold QA set
- `build_eval_dataset.py`: join run outputs (`results.csv`) with gold answers
- `retrieval_metrics.py`: compute retrieval-oriented metrics from joined data
- `run_ragas_eval.py`: run RAGAS metrics on joined data (optional local judge)
- `requirements.txt`: optional dependencies for this folder

## 1) Prepare gold dataset

Create and fill a copy of `gold_questions_template.csv` with your true labels.

Minimum required columns:

- `question`
- `ground_truth`

Optional but recommended columns:

- `expected_entities_json` (JSON list)
- `gold_triples_json` (JSON list)

## 2) Build joined evaluation dataset

```bash
conda run -n graphllm python evaluation/build_eval_dataset.py \
  --input artifacts/experiments/20260418_133106_retrieval_matrix_full_llm_qwen25_7b \
  --gold-file evaluation/gold_questions_template.csv \
  --output artifacts/evaluation/qwen25_7b_eval_dataset.csv
```

This creates a row-level dataset ready for retrieval metrics and RAGAS.

## 3) Retrieval metrics

```bash
conda run -n graphllm python evaluation/retrieval_metrics.py \
  --input artifacts/evaluation/qwen25_7b_eval_dataset.csv \
  --save-csv artifacts/evaluation/qwen25_7b_retrieval_summary.csv \
  --save-json artifacts/evaluation/qwen25_7b_retrieval_summary.json
```

Metrics include:

- `entity_coverage`
- `precision_at_k`, `recall_at_k`, `mrr` (when triple labels are available)

## 3b) LLM-as-a-Judge

Scores generated answers against the gold set on `answer_correctness`,
`groundedness`, `relevance` (JSON score + rationale per row, with bootstrap CIs).
The judge model is pluggable via `--backend` — pick the one that fits your wallet:

| `--backend` | Auth | Cost | Reproducible | When |
|---|---|---|---|---|
| `claude_code` | Claude Code subscription (Pro/Max, OAuth) | $0 extra | ✗ (tied to your account) | Day-to-day iteration on your own machine |
| `api` | `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`) | ~$3–15 / full run | ✓ (pin model + run) | Final paper numbers; anyone reusing the project with their own key |
| `vllm` / `local_hf` | local model | $0 | ✓ | No Claude access; uses your own model as judge |

**Subscription backend (`claude_code`).** Requires the `claude` CLI installed and
logged in to your claude.ai account (`claude login`). It drives `claude -p`
headless; the judge prompt is fully self-contained. Overrides: `CLAUDE_CODE_BIN`,
`CLAUDE_CODE_TIMEOUT`, `CLAUDE_CODE_EXTRA_ARGS`. Note: this uses the coding
subscription as a batch judge (grey area in Anthropic's usage policy) and is
**not reproducible by reviewers** — regenerate final numbers with `--backend api`.

```bash
# Pro subscription, both models, batched + resumable:
conda run -n graphllm python -m evalkit.cli judge \
  --input artifacts/evaluation/eval_dataset_gold23q_v2.csv \
  --backend claude_code --model haiku --batch-size 8 --resume \
  --out artifacts/evaluation/judge_haiku
conda run -n graphllm python -m evalkit.cli judge \
  --input artifacts/evaluation/eval_dataset_gold23q_v2.csv \
  --backend claude_code --model sonnet --batch-size 8 --resume \
  --out artifacts/evaluation/judge_sonnet

# Judge-model agreement (robustness table for the paper):
conda run -n graphllm python -m evalkit.cli judge-compare \
  --a artifacts/evaluation/judge_haiku --b artifacts/evaluation/judge_sonnet \
  --label-a haiku --label-b sonnet \
  --out artifacts/evaluation/judge_compare

# Same numbers via API (needs ANTHROPIC_API_KEY), pinned + reproducible:
conda run -n graphllm python -m pip install anthropic
conda run -n graphllm python -m evalkit.cli judge \
  --input artifacts/evaluation/eval_dataset_gold23q_v2.csv \
  --backend api --provider anthropic --model claude-sonnet-4-6 \
  --batch-size 8 --out artifacts/evaluation/judge_api
```

`--batch-size > 1` (always on for `claude_code`) scores N rows per call —
keeps subscription rate/weekly limits a non-issue and lets `--resume` recover an
interrupted run from `<out>/judge_rows.jsonl` without re-spending quota.

## 4) RAGAS (optional)

Install evaluation dependencies:

```bash
conda run -n graphllm python -m pip install -r evaluation/requirements.txt
```

Run RAGAS with local judge model:

```bash
conda run -n graphllm python evaluation/run_ragas_eval.py \
  --input artifacts/evaluation/qwen25_7b_eval_dataset.csv \
  --metrics faithfulness,answer_relevancy,answer_correctness,context_precision,context_recall \
  --judge-model Qwen/Qwen2.5-14B-Instruct \
  --embed-model sentence-transformers/all-MiniLM-L6-v2 \
  --save-row-csv artifacts/evaluation/qwen25_7b_ragas_rows.csv \
  --save-summary-json artifacts/evaluation/qwen25_7b_ragas_summary.json
```

Notes:

- If run outputs do not contain explicit contexts/triples, context metrics are skipped.
- The script will report skipped metrics and why.

## Recommended paper table schema

Rows:

- `(model_id, framework, strategy)`

Columns:

- retrieval: `entity_coverage`, `precision_at_k`, `recall_at_k`, `mrr`
- answer quality: `ragas_faithfulness`, `ragas_answer_relevancy`, `ragas_answer_correctness`
- runtime: latency/resource summaries from your existing analyzers
