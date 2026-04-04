#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  echo ".env not found in project root" >&2
  exit 1
fi

# Export variables from .env for sbatch inheritance.
set -a
# shellcheck disable=SC1091
source .env
set +a

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not available in this shell. Run this script on a SLURM login node." >&2
  exit 1
fi

: "${NEO4J_URL:?NEO4J_URL is required in .env}"
: "${NEO4J_USERNAME:?NEO4J_USERNAME is required in .env}"
: "${NEO4J_PASSWORD:?NEO4J_PASSWORD is required in .env}"
export NEO4J_DATABASE="${NEO4J_DATABASE:-neo4j}"

export CONDA_ENV="${CONDA_ENV:-graphllm}"
export QUESTIONS_FILE="${QUESTIONS_FILE:-questions_matrix_long.txt}"
export STRATEGIES="${STRATEGIES:-default,text_only,text_plus_triples,neighbors_focus,subgraph_2hop,shortest_path}"
export RUNS_PER_STRATEGY="${RUNS_PER_STRATEGY:-5}"
export EXPERIMENT_TAG_PREFIX="${EXPERIMENT_TAG_PREFIX:-matrix_long}"
export MODELS_CSV="${MODELS_CSV:-Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct}"

IFS=',' read -r -a MODELS <<< "$MODELS_CSV"
if [ "${#MODELS[@]}" -eq 0 ]; then
  echo "MODELS_CSV is empty" >&2
  exit 1
fi

MAX_INDEX=$(( ${#MODELS[@]} - 1 ))
ARRAY_SPEC="${ARRAY_SPEC:-0-${MAX_INDEX}%2}"
PARTITION="${PARTITION:-}"

CMD=(sbatch)
if [ -n "$PARTITION" ]; then
  CMD+=(-p "$PARTITION")
fi
CMD+=(--array="$ARRAY_SPEC" scripts/run_experiment_matrix_gpu.sbatch)

echo "Submitting matrix benchmark"
echo "- models: ${#MODELS[@]}"
echo "- array: $ARRAY_SPEC"
echo "- questions: $QUESTIONS_FILE"
echo "- runs/strategy: $RUNS_PER_STRATEGY"
echo "- strategies: $STRATEGIES"
if [ -n "$PARTITION" ]; then
  echo "- partition: $PARTITION"
fi

"${CMD[@]}"
