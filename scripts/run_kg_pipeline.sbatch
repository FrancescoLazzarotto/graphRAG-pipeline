#!/bin/bash
# Submit example:
#   sbatch -p <partition> scripts/run_kg_pipeline.sbatch

#SBATCH --job-name=kg-pipeline
#SBATCH --output=logs/kg-pipeline-%j.out
#SBATCH --error=logs/kg-pipeline-%j.err
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

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

timestamp="$(date +%Y%m%d_%H%M%S)"
run_dir="${KG_RUN_DIR:-kg_pipeline/artifacts/run_${timestamp}_slurm_${SLURM_JOB_ID:-local}}"

export PYTHONUNBUFFERED=1
export PYTHONPATH=".:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$KG_CUDA_VISIBLE_DEVICES"

if [ -f "$KG_ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$KG_ENV_FILE"
  set +a
fi

exec python -m kg_pipeline.main \
  --config "$KG_CONFIG" \
  --env-file "$KG_ENV_FILE" \
  --run-dir "$run_dir" \
  --stage "$KG_STAGE" \
  --log-level "$KG_LOG_LEVEL"