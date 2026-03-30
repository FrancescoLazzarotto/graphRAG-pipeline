# GraphRAG Pipeline

A simple GraphRAG project that combines Neo4j graph retrieval with an LLM-based answer pipeline.

## What this project does

- Connects to a Neo4j knowledge graph.
- Retrieves nodes, triples, neighbors, subgraphs, and shortest paths.
- Builds context from graph results.
- Optionally generates answers with a local Hugging Face model.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
Copy-Item .env.example .env
```

Set your Neo4j credentials in `.env`, then run:

```powershell
graphrag-demo --seed-movie-dataset
```

Enable local LLM generation:

```powershell
graphrag-demo --llm --model-id Qwen/Qwen2.5-3B-Instruct
```

## Main entrypoint

- CLI: `graphrag-demo`
