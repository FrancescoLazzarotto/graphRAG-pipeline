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
export QUESTION="${QUESTION:-What is the relationship between Entity A and Entity B?}"
export ENTITY="${ENTITY:-Entity A}"
export USE_VLLM="${USE_VLLM:-0}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
export VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
export VLLM_STARTUP_TIMEOUT_SEC="${VLLM_STARTUP_TIMEOUT_SEC:-180}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

VLLM_PID=""

cleanup() {
  if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "Stopping vLLM server (pid=$VLLM_PID)"
    kill "$VLLM_PID" || true
    wait "$VLLM_PID" || true
  fi
}

check_vllm_ready() {
  python - "$VLLM_BASE_URL" <<'PY'
import sys
import urllib.request

base_url = sys.argv[1].rstrip("/")
url = base_url + "/models"
request = urllib.request.Request(url, method="GET")
with urllib.request.urlopen(request, timeout=2):
    pass
PY
}

trap cleanup EXIT INT TERM

python scripts/smoke_check.py --check-neo4j

# Hard fail if this GPU job cannot actually use CUDA.
python -c "import torch;\
import sys;\
(_ for _ in ()).throw(SystemExit('CUDA is not available. Aborting GPU job to avoid CPU fallback.')) if not torch.cuda.is_available() else None;\
print(f'CUDA detected: {torch.cuda.device_count()} device(s)');\
[print(f'- GPU {idx}: {torch.cuda.get_device_name(idx)}') for idx in range(torch.cuda.device_count())]"

if [ "$USE_VLLM" = "1" ]; then
  VLLM_LOG="logs/vllm-${SLURM_JOB_ID:-local}.log"
  echo "Starting vLLM server (log: $VLLM_LOG)"
  python -m vllm.entrypoints.openai.api_server \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --model "$MODEL_ID" \
    --served-model-name "$MODEL_ID" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    >"$VLLM_LOG" 2>&1 &
  VLLM_PID=$!

  ready=0
  for ((attempt=1; attempt<=VLLM_STARTUP_TIMEOUT_SEC; attempt++)); do
    if check_vllm_ready >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "vLLM server exited unexpectedly while starting."
      tail -n 200 "$VLLM_LOG" || true
      exit 1
    fi
    if (( attempt % 10 == 0 )); then
      echo "Waiting for vLLM readiness... elapsed=${attempt}s"
    fi
    sleep 1
  done

  if [ "$ready" != "1" ]; then
    echo "Timed out waiting for vLLM readiness after ${VLLM_STARTUP_TIMEOUT_SEC}s"
    tail -n 200 "$VLLM_LOG" || true
    exit 1
  fi

  echo "vLLM ready at ${VLLM_BASE_URL}"
fi

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

if [ "$USE_VLLM" = "1" ]; then
  CMD+=(--vllm --vllm-base-url "$VLLM_BASE_URL")
fi

"${CMD[@]}"
