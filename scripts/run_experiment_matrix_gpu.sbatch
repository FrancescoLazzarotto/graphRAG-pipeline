#!/bin/bash
# Submit all models sequentially in one job:
#   sbatch -p <gpu_partition> scripts/run_experiment_matrix_gpu.sbatch
#
# Submit as job array (recommended for 2x A40):
#   MODELS_CSV="Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct" \
#   sbatch -p <gpu_partition> --array=0-2%2 scripts/run_experiment_matrix_gpu.sbatch
#SBATCH --job-name=graphrag-matrix
#SBATCH --output=logs/graphrag-matrix-%A_%a.out
#SBATCH --error=logs/graphrag-matrix-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
mkdir -p artifacts/experiments

: "${NEO4J_URL:?NEO4J_URL is required}"
: "${NEO4J_USERNAME:?NEO4J_USERNAME is required}"
: "${NEO4J_PASSWORD:?NEO4J_PASSWORD is required}"
export NEO4J_DATABASE="${NEO4J_DATABASE:-neo4j}"

CONDA_ENV="${CONDA_ENV:-graphllm}"
QUESTIONS_FILE="${QUESTIONS_FILE:-questions.txt}"
STRATEGIES="${STRATEGIES:-default,text_only,text_plus_triples,neighbors_focus,subgraph_2hop,shortest_path}"
RUNS_PER_STRATEGY="${RUNS_PER_STRATEGY:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/experiments}"
EXPERIMENT_TAG_PREFIX="${EXPERIMENT_TAG_PREFIX:-matrix_long}"
MODELS_CSV="${MODELS_CSV:-Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found. Load your conda module before submitting." >&2
  exit 1
fi

if [ ! -f "$QUESTIONS_FILE" ]; then
  echo "Questions file not found: $QUESTIONS_FILE" >&2
  exit 1
fi

IFS=',' read -r -a MODELS <<< "$MODELS_CSV"
if [ "${#MODELS[@]}" -eq 0 ]; then
  echo "MODELS_CSV is empty." >&2
  exit 1
fi

if [ "${RUN_SMOKE_CHECK:-1}" = "1" ]; then
  conda run -n "$CONDA_ENV" python scripts/smoke_check.py --check-neo4j
fi

if [ "${CHECK_CUDA:-1}" = "1" ]; then
  conda run -n "$CONDA_ENV" python -c "import torch,sys;sys.exit(0) if torch.cuda.is_available() else sys.exit('CUDA is not available on this node')"
fi

run_for_model() {
  local model_id="$1"
  local safe_model
  safe_model="$(echo "$model_id" | tr '/: @' '____' | tr -cd 'A-Za-z0-9._-')"
  local tag="${EXPERIMENT_TAG_PREFIX}_${safe_model}"

  echo "Running model: $model_id"
  echo "Strategies: $STRATEGIES"
  echo "Runs per strategy: $RUNS_PER_STRATEGY"
  echo "Questions file: $QUESTIONS_FILE"
  echo "Tag: $tag"

  conda run -n "$CONDA_ENV" graphrag-demo \
    --experiment \
    --questions-file "$QUESTIONS_FILE" \
    --strategies "$STRATEGIES" \
    --runs-per-strategy "$RUNS_PER_STRATEGY" \
    --output-dir "$OUTPUT_DIR" \
    --experiment-tag "$tag" \
    --llm \
    --model-id "$model_id"
}

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  index="${SLURM_ARRAY_TASK_ID}"
  if [ "$index" -lt 0 ] || [ "$index" -ge "${#MODELS[@]}" ]; then
    echo "SLURM_ARRAY_TASK_ID=$index out of bounds for ${#MODELS[@]} models" >&2
    exit 1
  fi
  run_for_model "${MODELS[$index]}"
else
  for model_id in "${MODELS[@]}"; do
    run_for_model "$model_id"
  done
fi

echo "Matrix benchmark completed. Output root: $OUTPUT_DIR"
