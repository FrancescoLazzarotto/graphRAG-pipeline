# Cluster Deployment Guide

## 1. Prerequisites

- Python 3.10+
- Access to Neo4j endpoint
- Network access to Hugging Face model hub or pre-cached model files

## 2. Immediate Run (Recommended)

The provided SLURM scripts can bootstrap the environment automatically (create venv + install dependencies + run smoke checks).

Export Neo4j variables before submission:

```bash
export NEO4J_URL="neo4j+s://<your-instance>"
export NEO4J_USERNAME="<your-username>"
export NEO4J_PASSWORD="<your-password>"
export NEO4J_DATABASE="<your-database>"
```

GPU run:

```bash
sbatch -p <gpu_partition> scripts/run_graphrag.sbatch
```

CPU run:

```bash
sbatch -p <cpu_partition> scripts/run_graphrag_cpu.sbatch
```

Before submission, export required Neo4j variables in your shell or directly in the script.

## 2.1 Local Validation Before Cluster Access

If you do not have cluster access yet, run the local preflight helper from Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/preflight.ps1
```

Optional Neo4j connectivity validation:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/preflight.ps1 -CheckNeo4j
```

## 3. Environment Variables

Required:

- NEO4J_URL
- NEO4J_USERNAME
- NEO4J_PASSWORD

Optional:

- NEO4J_DATABASE (default: neo4j)
- MODEL_ID (default: Qwen/Qwen2.5-7B-Instruct)
- HF_HOME / TRANSFORMERS_CACHE for model cache location

## 4. Manual Install (Optional)

Use this only if you prefer pre-installing dependencies yourself.

CPU nodes:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-cpu.txt
pip install -e . --no-deps
```

GPU nodes:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-gpu.txt
pip install -e . --no-deps
```

## 5. Smoke Check

```bash
python scripts/smoke_check.py
```

With Neo4j connectivity preflight:

```bash
python scripts/smoke_check.py --check-neo4j
```

## 6. Job Script Behavior

Both scripts support these runtime overrides:

- INSTALL_DEPS=0: skip dependency installation
- VENV_PATH=/path/to/venv: custom virtual environment path
- QUESTION / ENTITY: override default prompt inputs
- MODEL_ID: set custom model
- LLM_WARMUP=0 or 1: disable/enable warmup (GPU script default is 1)
- RUN_LLM_ON_CPU=1: enable local LLM on CPU script (default is 0)

Example:

```bash
INSTALL_DEPS=0 QUESTION="Who directed The Matrix?" sbatch -p <gpu_partition> scripts/run_graphrag.sbatch
```

## 7. Submit Job (SLURM)

```bash
sbatch scripts/run_graphrag.sbatch
```

To inspect live logs:

```bash
squeue -u $USER
tail -f logs/graphrag-gpu-<job_id>.out
tail -f logs/graphrag-gpu-<job_id>.err
```

## 7.1 Long Matrix Benchmark (2x A40)

For long-running comparisons across retrieval strategies and LLMs, use:

- `scripts/run_experiment_matrix_gpu.sbatch`

Recommended submission pattern for two GPUs:

```bash
MODELS_CSV="Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct" \
STRATEGIES="default,text_only,text_plus_triples,neighbors_focus,subgraph_2hop,shortest_path" \
RUNS_PER_STRATEGY=3 \
CONDA_ENV=graphllm \
sbatch -p <gpu_partition> --array=0-2%2 scripts/run_experiment_matrix_gpu.sbatch
```

Notes:

- `--array=0-2%2` runs up to 2 tasks concurrently (one model per task).
- Each task requests 1 GPU, 12 CPU, 64G RAM, 24h walltime by default.
- Override `QUESTIONS_FILE`, `OUTPUT_DIR`, and `EXPERIMENT_TAG_PREFIX` as needed.

Aggregate all produced run folders with:

```bash
python scripts/analyze_matrix.py artifacts/experiments --tag-contains matrix_long
```

## 8. Troubleshooting

- If CUDA mismatch occurs, install a compatible PyTorch wheel for your cluster.
- If model download fails on compute nodes, pre-populate HF_HOME cache on shared storage.
- If Neo4j connection fails, verify firewall/egress rules and credentials.
