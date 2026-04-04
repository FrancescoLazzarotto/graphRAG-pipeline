#!/bin/bash
# Submit example: sbatch -p <gpu_partition> scripts/run_graphrag.sbatch
#SBATCH --job-name=graphrag-gpu
#SBATCH --output=logs/graphrag-gpu-%j.out
#SBATCH --error=logs/graphrag-gpu-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

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
  python -m pip install -r requirements-gpu.txt
  python -m pip install -e . --no-deps
fi

: "${NEO4J_URL:?NEO4J_URL is required}"
: "${NEO4J_USERNAME:?NEO4J_USERNAME is required}"
: "${NEO4J_PASSWORD:?NEO4J_PASSWORD is required}"
export NEO4J_DATABASE="${NEO4J_DATABASE:-neo4j}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
export QUESTION="${QUESTION:-Who directed and acted in The Matrix?}"
export ENTITY="${ENTITY:-The Matrix}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

python scripts/smoke_check.py --check-neo4j

# Hard fail if this GPU job cannot actually use CUDA.
python -c "import torch;\
import sys;\
(_ for _ in ()).throw(SystemExit('CUDA is not available. Aborting GPU job to avoid CPU fallback.')) if not torch.cuda.is_available() else None;\
print(f'CUDA detected: {torch.cuda.device_count()} device(s)');\
[print(f'- GPU {idx}: {torch.cuda.get_device_name(idx)}') for idx in range(torch.cuda.device_count())]"

CMD=(
  graphrag-demo
  --question "$QUESTION"
  --entity "$ENTITY"
  --model-id "$MODEL_ID"
  --llm
)

if [ "${LLM_WARMUP:-1}" = "1" ]; then
  CMD+=(--llm-warmup)
fi

"${CMD[@]}"
