#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONDA_ENV="${CONDA_ENV:-graphllm}"
QUESTIONS_FILE="${QUESTIONS_FILE:-questions_matrix_long.txt}"
RUNS_PER_STRATEGY="${RUNS_PER_STRATEGY:-3}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-retrieval_matrix_full_llm_qwen25_7b}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/experiments}"
RESOURCE_SAMPLE_INTERVAL="${RESOURCE_SAMPLE_INTERVAL:-1.0}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-25}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found. Load conda and retry." >&2
  exit 1
fi

if [ ! -f "$QUESTIONS_FILE" ]; then
  echo "Questions file not found: $QUESTIONS_FILE" >&2
  exit 1
fi

mkdir -p logs "$OUTPUT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/${EXPERIMENT_TAG}_${TS}.log"

echo "Starting large retrieval matrix run"
echo "- conda env: $CONDA_ENV"
echo "- questions file: $QUESTIONS_FILE"
echo "- runs/strategy: $RUNS_PER_STRATEGY"
echo "- model: $MODEL_ID"
echo "- experiment tag: $EXPERIMENT_TAG"
echo "- output dir: $OUTPUT_DIR"
echo "- checkpoint every: $CHECKPOINT_EVERY"
if [ -n "$RESUME_RUN_DIR" ]; then
  echo "- resume run dir: $RESUME_RUN_DIR"
fi
echo "- log file: $LOG_FILE"

echo "Tip: in another terminal run: tail -f $LOG_FILE"

echo "Launching..."
resume_args=()
if [ -n "$RESUME_RUN_DIR" ]; then
  resume_args=(--resume-run-dir "$RESUME_RUN_DIR")
fi

# --no-capture-output avoids conda buffering, so tee receives live logs.
PYTHONUNBUFFERED=1 conda run --no-capture-output -n "$CONDA_ENV" python -u scripts/run_retrieval_matrix.py \
  --questions-file "$QUESTIONS_FILE" \
  --runs-per-strategy "$RUNS_PER_STRATEGY" \
  --output-dir "$OUTPUT_DIR" \
  --experiment-tag "$EXPERIMENT_TAG" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --llm \
  --model-id "$MODEL_ID" \
  --monitor-resources \
  --resource-sample-interval "$RESOURCE_SAMPLE_INTERVAL" \
  "${resume_args[@]}" \
  2>&1 | tee -a "$LOG_FILE"
