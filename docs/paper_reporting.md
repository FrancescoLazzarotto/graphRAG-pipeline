# Paper-Ready Experiment Reporting (GraphRAG)

This guide turns raw run artifacts into report-quality evidence for a paper submission.

## 1) Aggregate Performance Across Runs

Use the matrix analyzer to aggregate latency and retrieval usage by `(model_id, strategy)`.

```bash
python scripts/analyze_matrix.py artifacts/experiments \
  --tag-contains retrieval_matrix \
  --save-json artifacts/experiments/paper_matrix_summary.json \
  --save-csv artifacts/experiments/paper_matrix_summary.csv
```

Output:

- `paper_matrix_summary.csv`: table-ready metrics (`avg_latency_ms`, `p95_latency_ms`, retrieval counters)
- `paper_matrix_summary.json`: machine-readable aggregate

## 2) Aggregate Resource/Cost Evidence

Use resource summaries to report memory and utilization constraints.

```bash
python scripts/analyze_resource_usage.py artifacts/experiments \
  --tag-contains retrieval_matrix \
  --save-json artifacts/experiments/paper_resource_summary.json \
  --save-csv artifacts/experiments/paper_resource_summary.csv
```

Output includes peak/average process RAM, system RAM, and GPU memory/utilization.

## 3) Compute Factual Quality (Automatic Baseline)

`evaluate_matrix_quality.py` adds factual checks over experiment answers for the Matrix benchmark question family.

```bash
python scripts/evaluate_matrix_quality.py artifacts/experiments \
  --tag-contains retrieval_matrix \
  --save-summary-json artifacts/experiments/paper_quality_auto_summary.json \
  --save-summary-csv artifacts/experiments/paper_quality_auto_summary.csv \
  --save-row-csv artifacts/experiments/paper_quality_auto_rows.csv
```

Key fields:

- `strict_accuracy`: answer satisfies expected facts for evaluable questions
- `partial_accuracy`: weaker correctness signal
- `contradiction_rate`: known entity-role contradictions
- `refusal_rate`: generic/non-grounded refusal-style answers
- `evaluable_coverage`: percentage of rows included in factual auto-eval

## 4) Build Human Evaluation Sheet

Create a stratified sample for manual annotation.

```bash
python scripts/prepare_human_eval_sample.py artifacts/experiments \
  --tag-contains retrieval_matrix \
  --per-group 15 \
  --seed 42 \
  --output artifacts/experiments/paper_human_eval_sample.csv
```

Annotate these columns:

- `correctness_score`: `0` incorrect, `1` partial, `2` correct
- `grounded_score`: `0` not grounded, `1` grounded

## 5) Aggregate Human Evaluation

```bash
python scripts/analyze_human_eval.py artifacts/experiments/paper_human_eval_sample.csv \
  --save-json artifacts/experiments/paper_human_eval_summary.json \
  --save-csv artifacts/experiments/paper_human_eval_summary.csv
```

Outputs include rates and Wilson 95% confidence intervals.

## 6) Recommended Tables/Figures for Paper

Main table:

- Rows: strategies
- Columns: model, strict accuracy (human), p95 latency, peak GPU memory, contradiction rate

Figures:

- Pareto plot: `strict_accuracy` vs `avg_latency_ms`
- Resource plot: `max_peak_gpu_memory_used_mb` by model
- Error bars: human-eval CI per strategy

## 7) Reporting Caveats

For current pipeline state, do not use `reflection_pass_rate` and `confidence` as primary quality metrics.
Use automatic factual checks plus human annotation as quality evidence.
