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
