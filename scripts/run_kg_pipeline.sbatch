#!/bin/bash
# Submit example:
#   sbatch -p <partition> scripts/run_kg_pipeline.sbatch

#SBATCH --job-name=kg-pipeline
#SBATCH --output=logs/kg-pipeline-%j.out
#SBATCH --error=logs/kg-pipeline-%j.err
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-graphllm}"
else
  echo "conda not available in this environment" >&2
  exit 1
fi

: "${KG_CONFIG:=kg_pipeline/config.yaml}"
: "${KG_ENV_FILE:=kg_pipeline/.env}"
: "${KG_STAGE:=all}"
: "${KG_LOG_LEVEL:=INFO}"
: "${KG_CUDA_VISIBLE_DEVICES:=1}"
: "${KG_START_VLLM:=0}"
: "${VLLM_CUDA_VISIBLE_DEVICES:=0}"

export PYTHONUNBUFFERED=1
export PYTHONPATH=".:${PYTHONPATH:-}"

if [ -f "$KG_ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$KG_ENV_FILE"
  set +a
fi

: "${VLLM_HOST:=127.0.0.1}"
: "${VLLM_PORT:=8000}"
: "${VLLM_BASE_URL:=http://${VLLM_HOST}:${VLLM_PORT}/v1}"
: "${VLLM_GPU_MEMORY_UTILIZATION:=0.90}"
: "${VLLM_STARTUP_TIMEOUT_SEC:=180}"

timestamp="$(date +%Y%m%d_%H%M%S)"
run_dir="${KG_RUN_DIR:-kg_pipeline/artifacts/run_${timestamp}_slurm_${SLURM_JOB_ID:-local}}"
mkdir -p "$run_dir"

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

if [ "$KG_START_VLLM" = "1" ]; then
  if [ -z "${VLLM_MODEL_NAME:-}" ]; then
    echo "VLLM_MODEL_NAME is required when KG_START_VLLM=1" >&2
    exit 1
  fi

  VLLM_LOG="${VLLM_LOG:-$run_dir/vllm.log}"
  echo "Starting vLLM server (log: $VLLM_LOG)"
  CUDA_VISIBLE_DEVICES="$VLLM_CUDA_VISIBLE_DEVICES" \
    python -m vllm.entrypoints.openai.api_server \
      --host "$VLLM_HOST" \
      --port "$VLLM_PORT" \
      --model "$VLLM_MODEL_NAME" \
      --served-model-name "$VLLM_MODEL_NAME" \
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

CUDA_VISIBLE_DEVICES="$KG_CUDA_VISIBLE_DEVICES" \
  python -m kg_pipeline.main \
  --config "$KG_CONFIG" \
  --env-file "$KG_ENV_FILE" \
  --run-dir "$run_dir" \
  --stage "$KG_STAGE" \
  --log-level "$KG_LOG_LEVEL"