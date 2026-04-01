# GraphRAG Pipeline

A simple GraphRAG framework that combines Neo4j graph retrieval with an LLM-based answer pipeline.

## What this project does

- Connects to a Neo4j knowledge graph.
- Retrieves nodes, triples, neighbors, subgraphs, and shortest paths.
- Builds context from graph results.
- Generates answers with a local Hugging Face model.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Create a local `.env` file manually (not committed) with:

```env
NEO4J_URL=neo4j+s://<your-instance>
NEO4J_USERNAME=<your-username>
NEO4J_PASSWORD=<your-password>
NEO4J_DATABASE=<your-database>
```

For cluster jobs, you can export the same variables from the scheduler/job script.

Then run:

```powershell
graphrag-demo --seed-movie-dataset
```

Enable local LLM generation:

```powershell
graphrag-demo --llm --model-id Qwen/Qwen2.5-3B-Instruct
```

## Cluster setup

- CPU nodes: install [requirements-cpu.txt](requirements-cpu.txt)
- GPU nodes: install [requirements-gpu.txt](requirements-gpu.txt)
- GPU job template: [scripts/run_graphrag.sbatch](scripts/run_graphrag.sbatch)
- CPU job template: [scripts/run_graphrag_cpu.sbatch](scripts/run_graphrag_cpu.sbatch)
- Cluster guide: [docs/cluster.md](docs/cluster.md)

Quick smoke check after installation:

```bash
python scripts/smoke_check.py
```

Local preflight (no cluster access yet):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/preflight.ps1
```

With Neo4j connectivity check:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/preflight.ps1 -CheckNeo4j
```

Immediate cluster run examples:

```bash
export NEO4J_URL="neo4j+s://<your-instance>"
export NEO4J_USERNAME="<your-username>"
export NEO4J_PASSWORD="<your-password>"
export NEO4J_DATABASE="<your-database>"

# GPU
sbatch -p <gpu_partition> scripts/run_graphrag.sbatch

# CPU
sbatch -p <cpu_partition> scripts/run_graphrag_cpu.sbatch
```

## Main entrypoint

- CLI: `graphrag-demo`
