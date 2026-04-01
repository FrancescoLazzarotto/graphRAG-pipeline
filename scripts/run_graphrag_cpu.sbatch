#!/bin/bash
# Submit example: sbatch -p <cpu_partition> scripts/run_graphrag_cpu.sbatch
#SBATCH --job-name=graphrag-cpu
#SBATCH --output=logs/graphrag-cpu-%j.out
#SBATCH --error=logs/graphrag-cpu-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

VENV_PATH="${VENV_PATH:-.venv}"
if [ ! -d "$VENV_PATH" ]; then
  python3 -m venv "$VENV_PATH"
fi
source "$VENV_PATH/bin/activate"

if [ "${INSTALL_DEPS:-1}" = "1" ]; then
  python -m pip install --upgrade pip
  python -m pip install -r requirements-cpu.txt
  python -m pip install -e . --no-deps
fi

: "${NEO4J_URL:?NEO4J_URL is required}"
: "${NEO4J_USERNAME:?NEO4J_USERNAME is required}"
: "${NEO4J_PASSWORD:?NEO4J_PASSWORD is required}"
export NEO4J_DATABASE="${NEO4J_DATABASE:-neo4j}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-3B-Instruct}"
export QUESTION="${QUESTION:-Who directed and acted in The Matrix?}"
export ENTITY="${ENTITY:-The Matrix}"

python scripts/smoke_check.py --check-neo4j

CMD=(
  graphrag-demo
  --question "$QUESTION"
  --entity "$ENTITY"
)


if [ "${RUN_LLM_ON_CPU:-0}" = "1" ]; then
  CMD+=(--llm --model-id "$MODEL_ID")
  if [ "${LLM_WARMUP:-0}" = "1" ]; then
    CMD+=(--llm-warmup)
  fi
fi

"${CMD[@]}"
