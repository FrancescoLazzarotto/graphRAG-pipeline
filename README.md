
# GraphRAG Pipeline

An experiment-oriented RAG (Retrieval-Augmented Generation) pipeline combining graph retrieval (Neo4j) with LLM-based generation.

[![CI](https://github.com/FrancescoLazzarotto/graphRAG-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/FrancescoLazzarotto/graphRAG-pipeline/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Conda env: graphllm](https://img.shields.io/badge/conda-graphllm-44A833.svg?logo=anaconda)](https://docs.conda.io/)

Status: active development (May 2026)

- Core functionality: knowledge graph construction from documents, hybrid GraphRAG + text-RAG retrieval strategies, LLM-backed answer generation (local or vLLM server), and experiment/matrix orchestration.
- Primary entry points: CLI `graphrag-demo` and KG pipeline runner `python -m kg_pipeline.main`.
- Recommended environment: Conda environment named `graphllm` (see `requirements-*.txt`).

Table of Contents

- Project overview
- Quick start (Conda)
- Configuration (Neo4j, HF token, vLLM)
- Usage examples (CLI, vLLM, model tuning)
- Knowledge Graph pipeline (kg_pipeline)
- Experiments & matrices
- Analysis & telemetry
- Smoke tests & preflight
- Repository structure
- Contributing & license

Project overview

This repository implements a pipeline that:

- Ingests documents (PDF/Markdown/text), chunks and extracts entities and triples (NER + LLM extraction).
- Resolves and links entities/triples to build a Knowledge Graph and ingest into Neo4j.
- Provides a GraphRAG retriever that can return nodes, triples, local neighborhoods, 2-hop subgraphs, and shortest paths.
- Constructs LLM prompt context from retrieved graph/text and can generate answers using either local HF models or an OpenAI-compatible vLLM server.
- Runs reproducible experiment matrices to compare retrieval strategies and LLMs, and records resource telemetry for sizing studies.

End-to-end flow

1. Ingest documents from `docs/`, PDFs, or your own folder.
2. Chunk the text and extract entities, relations, and candidate triples.
3. Resolve, link, and ingest the graph into Neo4j.
4. Retrieve graph and text evidence with a chosen strategy.
5. Build prompt context and generate an answer with a local model or a vLLM server.
6. Save results, summaries, and resource telemetry for later analysis.

Quick start (Conda)

1. Create and activate the Conda environment (recommended name `graphllm`):

```bash
conda create -n graphllm python=3.10 -y
conda activate graphllm
```

2. Install dependencies (pick the right set for your node):

```bash
pip install -r requirements.txt        # base deps
pip install -r requirements-cpu.txt    # CPU-only nodes
pip install -r requirements-gpu.txt    # GPU nodes (if available)
pip install -e .
```

Configuration

Set required Neo4j environment variables (either export or place in `.env` files):

```bash
export NEO4J_URL="neo4j+s://<your-instance>"
export NEO4J_USERNAME="<user>"
export NEO4J_PASSWORD="<pass>"
export NEO4J_DATABASE="<db>"  # optional
```

If you use gated Hugging Face models, set `HF_TOKEN` in the environment:

```bash
export HF_TOKEN="<your-hf-token>"
```

vLLM / OpenAI-compatible server options:

- `VLLM_BASE_URL` (default: `http://localhost:8000/v1`)
- `VLLM_MODEL_NAME` and `VLLM_API_KEY` / `OPENAI_API_KEY` where applicable

Usage examples

- Run single-question demo (GraphRAG retrieval + generation):

```bash
graphrag-demo --question "What are the relations between Entity A and Entity B?" --entity "Entity A"
```

- Use local LLM generation:

```bash
graphrag-demo --llm --model-id Qwen/Qwen2.5-7B-Instruct
```

- Use server-backed (vLLM/OpenAI-compatible) generation:

```bash
graphrag-demo --llm --vllm --vllm-base-url http://localhost:8000/v1 --model-id Qwen/Qwen2.5-7B-Instruct
```

Model tuning examples (reduce generation cost or control GPU memory usage):

```bash
graphrag-demo --llm --model-id Qwen/Qwen2.5-14B-Instruct --max-new-tokens 128 --gpu-memory-fraction 0.90
```

Notes:

- `--max-new-tokens` reduces generation length/cost.
- `--gpu-memory-fraction` reserves headroom when loading large local models to reduce OOMs.
- For very large models (>= 30B) fp16 fallback is disabled by default; enable `--allow-large-model-fp16-fallback` only if you understand the memory/precision tradeoffs.

Knowledge Graph pipeline

The KG pipeline lives in `kg_pipeline/` and outputs checkpointed stage artifacts into a run directory. Defaults are controlled by `kg_pipeline/config.yaml`.

Run the full pipeline (logs to stdout):

```bash
conda activate graphllm
PYTHONUNBUFFERED=1 python -m kg_pipeline.main --config kg_pipeline/config.yaml --env-file kg_pipeline/.env --log-level INFO
```

You can resume an existing run by reusing the same `--run-dir`. To run a single stage set `--stage` to one of: `ingestion|chunking|ner|llm|resolution|linking|neo4j`.

To skip Neo4j ingestion (dry run) use `--dry-run`.

Typical pipeline outputs include:

- `pipeline.log` for the full execution trace.
- Stage artifacts under the selected run directory.
- Neo4j ingest logs when graph loading is enabled.

Example run directory shape:

```text
kg_pipeline/artifacts/run_YYYYMMDD_HHMMSS/
├── pipeline.log
├── chunking/
├── ner/
├── llm/
├── resolution/
├── linking/
└── neo4j/
```

Experiments & matrices

The repository includes scripts to run retrieval matrices that compare multiple strategies and LLMs. Strategies include:

- `default` (all sources), `text_only`, `text_plus_triples`, `neighbors_focus`, `subgraph_2hop`, `shortest_path`.

Smoke example (small experiment):

```bash
python scripts/run_retrieval_matrix.py --smoke --questions-file artifacts/experiments/questions_smoke.txt --documents docs/ README.md --runs-per-strategy 1 --output-dir artifacts/experiments --experiment-tag retrieval_matrix_smoke
```

vLLM-backed matrix example:

```bash
python scripts/run_retrieval_matrix.py --llm --vllm --vllm-base-url http://localhost:8000/v1 --model-id Qwen/Qwen2.5-32B-Instruct --questions-file questions_matrix_long.txt --graph-strategies default --runs-per-strategy 1
```

Analysis & telemetry

- Outputs per run include `results.jsonl`, `results.csv`, `summary.txt`, `summary.json`, `resource_samples.jsonl`, and `resource_summary.json`.
- Use `scripts/analyze_experiments.py` to analyze a single run directory.
- Use `scripts/analyze_matrix.py` and `scripts/analyze_resource_usage.py` to aggregate multiple runs and produce CSV/JSON summaries for sizing and comparison.

Expected experiment artifacts:

```text
artifacts/experiments/<run_name>/
├── results.jsonl
├── results.csv
├── summary.txt
├── summary.json
├── resource_samples.jsonl
└── resource_summary.json
```

What to expect in practice:

- `results.jsonl` contains one record per question/strategy/run combination.
- `summary.json` stores structured statistics per strategy and configuration.
- `summary.txt` is the fast human-readable check for a run.
- `resource_summary.json` reports peak and average resource usage when telemetry is enabled.

Concrete verification example from the repository’s recent runs:

- A verification report in [artifacts/experiments/20260514_170536_test_strategies_verification/REPORT.txt](artifacts/experiments/20260514_170536_test_strategies_verification/REPORT.txt) documents 60 runs, with `results.jsonl` as the raw trace and `summary.json` as the structured summary.
- That same report includes sample answer inspection commands such as `python3 show_samples.py results.jsonl default 2`.

Evaluation workflows

The repository also includes a separate evaluation workspace under [evaluation/README.md](evaluation/README.md) for paper-oriented comparisons.

Use it when you want to:

- build a gold QA dataset from run outputs and manual labels,
- compute retrieval-oriented metrics such as entity coverage and rank-based scores,
- optionally run RAGAS with a local judge model,
- generate summary tables that can be copied into a paper or internal report.

Typical evaluation sequence:

1. Prepare `evaluation/gold_questions_template.csv` with your labels.
2. Join a run output with the gold set using `evaluation/build_eval_dataset.py`.
3. Compute retrieval metrics with `evaluation/retrieval_metrics.py`.
4. Optionally run `evaluation/run_ragas_eval.py` for answer-quality metrics.

Smoke tests & preflight

- Quick smoke check after install: `python scripts/smoke_check.py`
- Local preflight helper (PowerShell): `powershell -ExecutionPolicy Bypass -File scripts/preflight.ps1`
- If you only want a fast sanity check before a long run, start with the smoke matrix command above and verify that `summary.json` and `results.jsonl` are created in the output directory.

Cluster & batch jobs

- CPU nodes: install `requirements-cpu.txt`
- GPU nodes: install `requirements-gpu.txt`
- GPU job template: `scripts/run_graphrag.sbatch`
- CPU job template: `scripts/run_graphrag_cpu.sbatch`

Examples (SLURM):

```bash
export NEO4J_URL="neo4j+s://<your-instance>"
export NEO4J_USERNAME="<user>"
export NEO4J_PASSWORD="<pass>"
export NEO4J_DATABASE="<db>"

# submit GPU job
sbatch -p <gpu_partition> scripts/run_graphrag.sbatch

# submit CPU job
sbatch -p <cpu_partition> scripts/run_graphrag_cpu.sbatch
```

Repository structure (short)

- `kg_pipeline/` — knowledge graph pipeline
- `src/graphrag/` — main package, CLI and agent code
- `scripts/` — utilities, smoke tests, batch templates
- `docs/` — additional documentation (eg. cluster.md)
- `artifacts/` — experiment outputs, logs and reports

Contributing

- Open issues or pull requests for bugs and improvements.
- Run smoke tests and minimal experiments locally before submitting PRs.
- Keep changes small and reproducible; if you touch retrieval logic, include a short note about the affected strategy or output artifact.

Troubleshooting

- If the CLI cannot connect to Neo4j, verify `NEO4J_URL`/`NEO4J_USERNAME`/`NEO4J_PASSWORD` and make sure the database name is correct.
- If local model loading fails, try a smaller model first or reduce `--max-new-tokens` and review GPU memory settings.
- If a vLLM run does not generate answers, confirm the server URL and model name match the server process.
- If experiment runs complete but produce no useful context, inspect the generated `summary.json` and `results.jsonl` before changing the pipeline.



<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GraphRAG Pipeline</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:       #080d14;
    --surface:  #0d1520;
    --border:   #1a2a3a;
    --accent:   #00e5ff;
    --accent2:  #7b2fff;
    --accent3:  #00ff88;
    --muted:    #4a6278;
    --text:     #c8dce8;
    --heading:  #eaf4ff;
    --code-bg:  #060c14;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  html { scroll-behavior: smooth; }

  body {
    font-family: 'JetBrains Mono', monospace;
    background: var(--bg);
    color: var(--text);
    line-height: 1.75;
    font-size: 14px;
    overflow-x: hidden;
  }

  /* ── CANVAS BG ── */
  #bg-canvas {
    position: fixed; inset: 0; z-index: 0;
    pointer-events: none; opacity: .35;
  }

  /* ── LAYOUT ── */
  .wrapper {
    position: relative; z-index: 1;
    max-width: 960px; margin: 0 auto;
    padding: 0 2rem 6rem;
  }

  /* ── HERO ── */
  .hero {
    padding: 7rem 0 5rem;
    border-bottom: 1px solid var(--border);
    position: relative;
  }

  .hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; letter-spacing: .22em;
    text-transform: uppercase;
    color: var(--accent); opacity: .8;
    margin-bottom: 1.2rem;
  }

  .hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(3rem, 7vw, 5.5rem);
    font-weight: 800;
    line-height: 1;
    color: var(--heading);
    letter-spacing: -.03em;
    margin-bottom: .6rem;
  }

  .hero h1 span {
    background: linear-gradient(120deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .hero-sub {
    font-size: 15px; max-width: 540px;
    color: var(--muted); margin: 1.4rem 0 2rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 300; line-height: 1.8;
  }

  .badges { display: flex; gap: .6rem; flex-wrap: wrap; margin-bottom: 2.5rem; }

  .badge {
    display: inline-flex; align-items: center; gap: .4rem;
    font-size: 11px; font-family: 'JetBrains Mono', monospace;
    padding: .28rem .7rem;
    border-radius: 3px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text);
    text-decoration: none;
    transition: border-color .2s, color .2s;
  }
  .badge:hover { border-color: var(--accent); color: var(--accent); }
  .badge .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent3); }

  .status-pill {
    display: inline-flex; align-items: center; gap: .5rem;
    font-size: 11px; padding: .3rem 1rem;
    border: 1px solid var(--accent3);
    border-radius: 100px;
    color: var(--accent3);
    background: rgba(0,255,136,.06);
    margin-bottom: 2rem;
  }
  .status-pill::before {
    content: ''; width: 6px; height: 6px; border-radius: 50%;
    background: var(--accent3);
    animation: pulse 1.6s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(.7)} }

  /* ── TOC ── */
  .toc {
    margin: 3.5rem 0;
    padding: 1.8rem 2rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent2);
    border-radius: 4px;
    animation: fadeUp .5s ease both;
  }

  .toc-title {
    font-family: 'Syne', sans-serif;
    font-size: 11px; letter-spacing: .2em;
    text-transform: uppercase; color: var(--muted);
    margin-bottom: 1rem;
  }

  .toc-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: .3rem .5rem;
  }

  .toc-grid a {
    color: var(--text); text-decoration: none;
    font-size: 12.5px; padding: .2rem 0;
    display: flex; align-items: center; gap: .5rem;
    transition: color .2s;
  }
  .toc-grid a::before { content: '→'; color: var(--accent2); font-size: 11px; }
  .toc-grid a:hover { color: var(--accent); }

  /* ── SECTIONS ── */
  section { margin: 4.5rem 0; animation: fadeUp .5s ease both; }

  @keyframes fadeUp { from{opacity:0;transform:translateY(18px)} to{opacity:1;transform:translateY(0)} }

  .section-label {
    font-size: 10px; letter-spacing: .25em; text-transform: uppercase;
    color: var(--accent); margin-bottom: .6rem; display: block;
  }

  h2 {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem; font-weight: 700;
    color: var(--heading); letter-spacing: -.02em;
    margin-bottom: 1.4rem;
    padding-bottom: .7rem;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: .7rem;
  }

  h2 .h2-num {
    font-size: 11px; color: var(--accent2);
    font-family: 'JetBrains Mono', monospace;
    font-weight: 400; letter-spacing: .1em;
  }

  h3 {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem; font-weight: 600;
    color: var(--heading); margin: 2rem 0 .8rem;
    letter-spacing: -.01em;
  }

  p { margin-bottom: 1rem; }

  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }

  /* ── CODE ── */
  pre {
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 4px;
    padding: 1.2rem 1.4rem;
    overflow-x: auto;
    margin: 1rem 0 1.6rem;
    position: relative;
  }

  pre code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12.5px; color: #8ecfea; line-height: 1.7;
  }

  .pre-label {
    font-size: 10px; letter-spacing: .15em; text-transform: uppercase;
    color: var(--muted); margin-bottom: .4rem; display: block;
  }

  code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    background: rgba(0,229,255,.07);
    color: var(--accent);
    padding: .1rem .35rem;
    border-radius: 3px;
  }

  pre code { background: none; color: #8ecfea; padding: 0; }

  /* ── FLOW ── */
  .flow {
    display: flex; flex-direction: column; gap: 0;
    margin: 1.5rem 0;
    border: 1px solid var(--border);
    border-radius: 6px; overflow: hidden;
  }

  .flow-step {
    display: flex; align-items: flex-start; gap: 1rem;
    padding: 1rem 1.4rem;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
    transition: background .2s;
  }
  .flow-step:last-child { border-bottom: none; }
  .flow-step:hover { background: #111d2a; }

  .flow-num {
    font-size: 11px; color: var(--accent2);
    font-weight: 500; min-width: 1.8rem;
    padding-top: .1rem;
  }

  .flow-text strong {
    font-family: 'Syne', sans-serif;
    font-size: .95rem; color: var(--heading);
    display: block; margin-bottom: .15rem; font-weight: 600;
  }

  .flow-text span { font-size: 12.5px; color: var(--muted); }

  /* ── STRATEGY GRID ── */
  .strategy-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: .8rem; margin: 1.4rem 0;
  }

  .strategy-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 4px; padding: 1rem 1.2rem;
    transition: border-color .2s, transform .2s;
  }
  .strategy-card:hover { border-color: var(--accent2); transform: translateY(-2px); }

  .strategy-card .sc-tag {
    font-size: 10px; letter-spacing: .15em; text-transform: uppercase;
    color: var(--accent2); display: block; margin-bottom: .4rem;
  }

  .strategy-card .sc-name {
    font-family: 'Syne', sans-serif;
    font-size: 1rem; font-weight: 600; color: var(--heading);
  }

  /* ── ARTIFACT TREE ── */
  .tree {
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 4px; padding: 1.2rem 1.4rem;
    font-size: 12.5px; line-height: 1.9;
  }

  .tree .dir { color: var(--accent); }
  .tree .file { color: var(--muted); }
  .tree .ann { color: var(--accent3); font-size: 11px; }

  /* ── INFO CARDS ── */
  .cards {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1rem; margin: 1.5rem 0;
  }

  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; padding: 1.4rem;
    transition: border-color .2s;
  }
  .card:hover { border-color: var(--accent); }

  .card-icon {
    font-size: 1.5rem; margin-bottom: .7rem; display: block;
    filter: grayscale(.3);
  }

  .card h4 {
    font-family: 'Syne', sans-serif;
    font-size: 1rem; font-weight: 600;
    color: var(--heading); margin-bottom: .5rem;
  }

  .card p { font-size: 12.5px; color: var(--muted); margin: 0; }

  /* ── ENV TABLE ── */
  .env-table {
    width: 100%; border-collapse: collapse; margin: 1.2rem 0;
    font-size: 12.5px;
  }
  .env-table th {
    text-align: left; font-size: 10px; letter-spacing: .15em;
    text-transform: uppercase; color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding: .5rem .8rem;
    font-weight: 400;
  }
  .env-table td {
    padding: .6rem .8rem;
    border-bottom: 1px solid rgba(26,42,58,.6);
    vertical-align: top;
  }
  .env-table tr:last-child td { border-bottom: none; }
  .env-table .key { color: var(--accent); font-weight: 500; }
  .env-table .req {
    font-size: 10px; color: #ff6b6b; letter-spacing: .1em;
    text-transform: uppercase;
  }
  .env-table .opt {
    font-size: 10px; color: var(--muted); letter-spacing: .1em;
    text-transform: uppercase;
  }
  .env-table .desc { color: var(--muted); }

  /* ── STAGE PIPELINE ── */
  .stages {
    display: flex; gap: .4rem; flex-wrap: wrap; margin: 1.4rem 0;
  }

  .stage {
    font-size: 11.5px;
    padding: .35rem .9rem;
    border: 1px solid var(--border);
    border-radius: 3px;
    background: var(--surface);
    color: var(--text);
    display: flex; align-items: center; gap: .5rem;
  }
  .stage::before { content: ''; width: 5px; height: 5px; border-radius: 50%; background: var(--accent2); }
  .stage:last-child::after { display: none; }

  /* ── ALERT ── */
  .alert {
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent2);
    background: rgba(123,47,255,.06);
    padding: .9rem 1.2rem; border-radius: 4px;
    font-size: 12.5px; margin: 1rem 0;
    color: var(--text);
  }
  .alert strong { color: var(--accent2); }

  .alert.warn { border-left-color: #ffaa00; background: rgba(255,170,0,.05); }
  .alert.warn strong { color: #ffaa00; }

  /* ── TROUBLESHOOT LIST ── */
  .trouble-list { list-style: none; padding: 0; }

  .trouble-list li {
    display: flex; gap: 1rem; align-items: flex-start;
    padding: .9rem 0;
    border-bottom: 1px solid rgba(26,42,58,.5);
    font-size: 13px;
  }
  .trouble-list li:last-child { border-bottom: none; }

  .trouble-list .ti { color: #ff6b6b; font-size: 1rem; min-width: 1.5rem; }

  /* ── FOOTER ── */
  .footer {
    margin-top: 6rem; padding-top: 2rem;
    border-top: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center;
    flex-wrap: wrap; gap: 1rem;
    font-size: 12px; color: var(--muted);
  }

  .footer-brand {
    font-family: 'Syne', sans-serif;
    font-weight: 700; font-size: 1rem; color: var(--heading);
  }

  /* ── SCROLLBAR ── */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--muted); }

  /* Stagger entrance */
  section:nth-child(1) { animation-delay: .05s }
  section:nth-child(2) { animation-delay: .10s }
  section:nth-child(3) { animation-delay: .15s }
  section:nth-child(4) { animation-delay: .20s }
  section:nth-child(5) { animation-delay: .25s }
  section:nth-child(6) { animation-delay: .30s }
</style>
</head>
<body>

<!-- Animated graph background -->
<canvas id="bg-canvas"></canvas>

<div class="wrapper">

  <!-- ── HERO ── -->
  <header class="hero">
    <p class="hero-eyebrow">// experiment-oriented RAG pipeline</p>
    <h1>Graph<span>RAG</span><br>Pipeline</h1>
    <div class="status-pill">Active development — May 2026</div>
    <p class="hero-sub">Knowledge graph construction · Hybrid GraphRAG + text-RAG retrieval · LLM-backed generation · Experiment orchestration over Neo4j</p>
    <div class="badges">
      <span class="badge"><span class="dot"></span> CI passing</span>
      <span class="badge">Python 3.10+</span>
      <span class="badge">Conda: graphllm</span>
      <span class="badge">Neo4j</span>
      <span class="badge">vLLM / HF</span>
    </div>
  </header>

  <!-- ── TOC ── -->
  <nav class="toc">
    <p class="toc-title">// Table of Contents</p>
    <div class="toc-grid">
      <a href="#overview">Project Overview</a>
      <a href="#quickstart">Quick Start</a>
      <a href="#config">Configuration</a>
      <a href="#usage">Usage Examples</a>
      <a href="#kg">KG Pipeline</a>
      <a href="#experiments">Experiments & Matrices</a>
      <a href="#analysis">Analysis & Telemetry</a>
      <a href="#evaluation">Evaluation Workflows</a>
      <a href="#smoke">Smoke Tests</a>
      <a href="#cluster">Cluster & Batch Jobs</a>
      <a href="#structure">Repository Structure</a>
      <a href="#contributing">Contributing</a>
      <a href="#troubleshooting">Troubleshooting</a>
    </div>
  </nav>

  <!-- ── OVERVIEW ── -->
  <section id="overview">
    <span class="section-label">01 — overview</span>
    <h2><span class="h2-num">01</span> Project Overview</h2>
    <p>GraphRAG Pipeline is a research-oriented system that combines knowledge graph retrieval with LLM-based generation. It ingests unstructured documents, builds a Neo4j knowledge graph via NER and triple extraction, and answers questions by retrieving structured graph evidence alongside traditional text chunks.</p>

    <div class="cards">
      <div class="card">
        <span class="card-icon">🏗️</span>
        <h4>Ingest &amp; Build</h4>
        <p>Parses PDF, Markdown and plain-text sources; extracts entities, relations and triples; resolves and links them into Neo4j.</p>
      </div>
      <div class="card">
        <span class="card-icon">🔍</span>
        <h4>Hybrid Retrieval</h4>
        <p>Six configurable strategies — from plain text to 2-hop subgraphs and shortest-path traversal.</p>
      </div>
      <div class="card">
        <span class="card-icon">🤖</span>
        <h4>Flexible Generation</h4>
        <p>Run answers locally with Hugging Face models or route to any OpenAI-compatible vLLM server.</p>
      </div>
      <div class="card">
        <span class="card-icon">📊</span>
        <h4>Reproducible Experiments</h4>
        <p>Matrix orchestration compares strategies &amp; models; resource telemetry logs CPU/GPU usage for sizing studies.</p>
      </div>
    </div>

    <h3>End-to-end flow</h3>
    <div class="flow">
      <div class="flow-step">
        <span class="flow-num">01</span>
        <div class="flow-text">
          <strong>Ingest</strong>
          <span>Load documents from <code>docs/</code>, PDFs, or a custom folder into the pipeline.</span>
        </div>
      </div>
      <div class="flow-step">
        <span class="flow-num">02</span>
        <div class="flow-text">
          <strong>Chunk &amp; Extract</strong>
          <span>Split text into chunks; run NER + LLM extraction to surface entities, relations, and candidate triples.</span>
        </div>
      </div>
      <div class="flow-step">
        <span class="flow-num">03</span>
        <div class="flow-text">
          <strong>Resolve, Link &amp; Ingest</strong>
          <span>Deduplicate and co-reference entities; load the resulting graph into Neo4j.</span>
        </div>
      </div>
      <div class="flow-step">
        <span class="flow-num">04</span>
        <div class="flow-text">
          <strong>Retrieve</strong>
          <span>Query graph and text evidence using the chosen retrieval strategy.</span>
        </div>
      </div>
      <div class="flow-step">
        <span class="flow-num">05</span>
        <div class="flow-text">
          <strong>Generate</strong>
          <span>Build the LLM prompt from retrieved context and produce an answer — locally or via vLLM server.</span>
        </div>
      </div>
      <div class="flow-step">
        <span class="flow-num">06</span>
        <div class="flow-text">
          <strong>Record</strong>
          <span>Save results, summaries, and resource telemetry for downstream analysis.</span>
        </div>
      </div>
    </div>
  </section>

  <!-- ── QUICK START ── -->
  <section id="quickstart">
    <span class="section-label">02 — setup</span>
    <h2><span class="h2-num">02</span> Quick Start (Conda)</h2>
    <p>The recommended environment is a Conda environment named <code>graphllm</code> running Python 3.10+.</p>

    <h3>1 — Create and activate the environment</h3>
    <span class="pre-label">bash</span>
    <pre><code>conda create -n graphllm python=3.10 -y
conda activate graphllm</code></pre>

    <h3>2 — Install dependencies</h3>
    <p>Select the requirement set that matches your hardware:</p>
    <span class="pre-label">bash</span>
    <pre><code># Base dependencies (always required)
pip install -r requirements.txt

# CPU-only nodes
pip install -r requirements-cpu.txt

# GPU nodes
pip install -r requirements-gpu.txt

# Install the package in editable mode
pip install -e .</code></pre>

    <div class="alert">
      <strong>Note:</strong> Install either <code>requirements-cpu.txt</code> or <code>requirements-gpu.txt</code> — not both. Install on top of the base <code>requirements.txt</code>.
    </div>
  </section>

  <!-- ── CONFIG ── -->
  <section id="config">
    <span class="section-label">03 — configuration</span>
    <h2><span class="h2-num">03</span> Configuration</h2>
    <p>Set required environment variables via export or place them in a <code>.env</code> file at the project root.</p>

    <h3>Neo4j</h3>
    <span class="pre-label">bash</span>
    <pre><code>export NEO4J_URL="neo4j+s://&lt;your-instance&gt;"
export NEO4J_USERNAME="&lt;user&gt;"
export NEO4J_PASSWORD="&lt;pass&gt;"
export NEO4J_DATABASE="&lt;db&gt;"   # optional</code></pre>

    <h3>Hugging Face token</h3>
    <p>Required only for gated/private models:</p>
    <span class="pre-label">bash</span>
    <pre><code>export HF_TOKEN="&lt;your-hf-token&gt;"</code></pre>

    <h3>All environment variables</h3>
    <table class="env-table">
      <thead>
        <tr>
          <th>Variable</th>
          <th>Default</th>
          <th>Required</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="key">NEO4J_URL</td>
          <td>—</td>
          <td><span class="req">required</span></td>
          <td class="desc">Bolt/Neo4j connection URL</td>
        </tr>
        <tr>
          <td class="key">NEO4J_USERNAME</td>
          <td>—</td>
          <td><span class="req">required</span></td>
          <td class="desc">Database user</td>
        </tr>
        <tr>
          <td class="key">NEO4J_PASSWORD</td>
          <td>—</td>
          <td><span class="req">required</span></td>
          <td class="desc">Database password</td>
        </tr>
        <tr>
          <td class="key">NEO4J_DATABASE</td>
          <td><code>neo4j</code></td>
          <td><span class="opt">optional</span></td>
          <td class="desc">Target database name</td>
        </tr>
        <tr>
          <td class="key">HF_TOKEN</td>
          <td>—</td>
          <td><span class="opt">optional</span></td>
          <td class="desc">Hugging Face token for gated models</td>
        </tr>
        <tr>
          <td class="key">VLLM_BASE_URL</td>
          <td><code>http://localhost:8000/v1</code></td>
          <td><span class="opt">optional</span></td>
          <td class="desc">vLLM / OpenAI-compatible server endpoint</td>
        </tr>
        <tr>
          <td class="key">VLLM_MODEL_NAME</td>
          <td>—</td>
          <td><span class="opt">optional</span></td>
          <td class="desc">Model name as reported by the server</td>
        </tr>
        <tr>
          <td class="key">VLLM_API_KEY / OPENAI_API_KEY</td>
          <td>—</td>
          <td><span class="opt">optional</span></td>
          <td class="desc">Auth key for the inference server</td>
        </tr>
      </tbody>
    </table>
  </section>

  <!-- ── USAGE ── -->
  <section id="usage">
    <span class="section-label">04 — usage</span>
    <h2><span class="h2-num">04</span> Usage Examples</h2>
    <p>The primary CLI entry point is <code>graphrag-demo</code>.</p>

    <h3>Single-question demo (GraphRAG retrieval + generation)</h3>
    <span class="pre-label">bash</span>
    <pre><code>graphrag-demo \
  --question "What are the relations between Entity A and Entity B?" \
  --entity "Entity A"</code></pre>

    <h3>Local LLM generation (Hugging Face)</h3>
    <span class="pre-label">bash</span>
    <pre><code>graphrag-demo --llm --model-id Qwen/Qwen2.5-7B-Instruct</code></pre>

    <h3>Server-backed generation (vLLM / OpenAI-compatible)</h3>
    <span class="pre-label">bash</span>
    <pre><code>graphrag-demo \
  --llm --vllm \
  --vllm-base-url http://localhost:8000/v1 \
  --model-id Qwen/Qwen2.5-7B-Instruct</code></pre>

    <h3>Model tuning — GPU memory &amp; generation length</h3>
    <span class="pre-label">bash</span>
    <pre><code>graphrag-demo \
  --llm \
  --model-id Qwen/Qwen2.5-14B-Instruct \
  --max-new-tokens 128 \
  --gpu-memory-fraction 0.90</code></pre>

    <div class="cards" style="margin-top:1.2rem">
      <div class="card">
        <h4>--max-new-tokens</h4>
        <p>Caps generation length. Reduces cost and latency on large models.</p>
      </div>
      <div class="card">
        <h4>--gpu-memory-fraction</h4>
        <p>Reserves a headroom fraction to avoid OOM errors when loading large local models.</p>
      </div>
      <div class="card">
        <h4>--allow-large-model-fp16-fallback</h4>
        <p>Enables FP16 fallback for models ≥ 30B. Disabled by default. Only use if you understand the memory/precision tradeoffs.</p>
      </div>
    </div>
  </section>

  <!-- ── KG PIPELINE ── -->
  <section id="kg">
    <span class="section-label">05 — kg pipeline</span>
    <h2><span class="h2-num">05</span> Knowledge Graph Pipeline</h2>
    <p>The KG pipeline lives in <code>kg_pipeline/</code>. It runs as a series of checkpointed stages, outputting artifacts into a timestamped run directory. Defaults are controlled by <code>kg_pipeline/config.yaml</code>.</p>

    <h3>Run the full pipeline</h3>
    <span class="pre-label">bash</span>
    <pre><code>conda activate graphllm
PYTHONUNBUFFERED=1 python -m kg_pipeline.main \
  --config kg_pipeline/config.yaml \
  --env-file kg_pipeline/.env \
  --log-level INFO</code></pre>

    <h3>Pipeline stages</h3>
    <p>Stages can be run individually via <code>--stage &lt;name&gt;</code>. Resume an existing run by reusing the same <code>--run-dir</code>.</p>
    <div class="stages">
      <span class="stage">ingestion</span>
      <span class="stage">chunking</span>
      <span class="stage">ner</span>
      <span class="stage">llm</span>
      <span class="stage">resolution</span>
      <span class="stage">linking</span>
      <span class="stage">neo4j</span>
    </div>

    <div class="alert">
      <strong>Dry run:</strong> use <code>--dry-run</code> to skip Neo4j ingestion and validate the pipeline locally.
    </div>

    <h3>Run directory structure</h3>
    <div class="tree">
      <div><span class="dir">kg_pipeline/artifacts/run_YYYYMMDD_HHMMSS/</span></div>
      <div>&nbsp;&nbsp;├── <span class="file">pipeline.log</span> <span class="ann">← full execution trace</span></div>
      <div>&nbsp;&nbsp;├── <span class="dir">chunking/</span></div>
      <div>&nbsp;&nbsp;├── <span class="dir">ner/</span></div>
      <div>&nbsp;&nbsp;├── <span class="dir">llm/</span></div>
      <div>&nbsp;&nbsp;├── <span class="dir">resolution/</span></div>
      <div>&nbsp;&nbsp;├── <span class="dir">linking/</span></div>
      <div>&nbsp;&nbsp;└── <span class="dir">neo4j/</span></div>
    </div>
  </section>

  <!-- ── EXPERIMENTS ── -->
  <section id="experiments">
    <span class="section-label">06 — experiments</span>
    <h2><span class="h2-num">06</span> Experiments &amp; Matrices</h2>
    <p>The matrix runner in <code>scripts/run_retrieval_matrix.py</code> sweeps over retrieval strategies and LLMs, producing structured output for comparison.</p>

    <h3>Retrieval strategies</h3>
    <div class="strategy-grid">
      <div class="strategy-card">
        <span class="sc-tag">strategy</span>
        <span class="sc-name">default</span>
      </div>
      <div class="strategy-card">
        <span class="sc-tag">strategy</span>
        <span class="sc-name">text_only</span>
      </div>
      <div class="strategy-card">
        <span class="sc-tag">strategy</span>
        <span class="sc-name">text_plus_triples</span>
      </div>
      <div class="strategy-card">
        <span class="sc-tag">strategy</span>
        <span class="sc-name">neighbors_focus</span>
      </div>
      <div class="strategy-card">
        <span class="sc-tag">strategy</span>
        <span class="sc-name">subgraph_2hop</span>
      </div>
      <div class="strategy-card">
        <span class="sc-tag">strategy</span>
        <span class="sc-name">shortest_path</span>
      </div>
    </div>

    <h3>Smoke run (quick validation)</h3>
    <span class="pre-label">bash</span>
    <pre><code>python scripts/run_retrieval_matrix.py \
  --smoke \
  --questions-file artifacts/experiments/questions_smoke.txt \
  --documents docs/ README.md \
  --runs-per-strategy 1 \
  --output-dir artifacts/experiments \
  --experiment-tag retrieval_matrix_smoke</code></pre>

    <h3>Full vLLM-backed matrix</h3>
    <span class="pre-label">bash</span>
    <pre><code>python scripts/run_retrieval_matrix.py \
  --llm --vllm \
  --vllm-base-url http://localhost:8000/v1 \
  --model-id Qwen/Qwen2.5-32B-Instruct \
  --questions-file questions_matrix_long.txt \
  --graph-strategies default \
  --runs-per-strategy 1</code></pre>
  </section>

  <!-- ── ANALYSIS ── -->
  <section id="analysis">
    <span class="section-label">07 — analysis</span>
    <h2><span class="h2-num">07</span> Analysis &amp; Telemetry</h2>
    <p>Each experiment run produces a self-contained output directory. Use the analysis scripts to aggregate results across runs.</p>

    <h3>Per-run artifacts</h3>
    <div class="tree">
      <div><span class="dir">artifacts/experiments/&lt;run_name&gt;/</span></div>
      <div>&nbsp;&nbsp;├── <span class="file">results.jsonl</span>     <span class="ann">← one record per question/strategy/run</span></div>
      <div>&nbsp;&nbsp;├── <span class="file">results.csv</span></div>
      <div>&nbsp;&nbsp;├── <span class="file">summary.txt</span>       <span class="ann">← fast human-readable check</span></div>
      <div>&nbsp;&nbsp;├── <span class="file">summary.json</span>      <span class="ann">← structured stats per strategy</span></div>
      <div>&nbsp;&nbsp;├── <span class="file">resource_samples.jsonl</span></div>
      <div>&nbsp;&nbsp;└── <span class="file">resource_summary.json</span> <span class="ann">← peak &amp; avg resource usage</span></div>
    </div>

    <h3>Analysis scripts</h3>
    <table class="env-table">
      <thead>
        <tr><th>Script</th><th>Purpose</th></tr>
      </thead>
      <tbody>
        <tr>
          <td class="key">scripts/analyze_experiments.py</td>
          <td class="desc">Analyze a single run directory</td>
        </tr>
        <tr>
          <td class="key">scripts/analyze_matrix.py</td>
          <td class="desc">Aggregate results across multiple runs into CSV/JSON summaries</td>
        </tr>
        <tr>
          <td class="key">scripts/analyze_resource_usage.py</td>
          <td class="desc">Aggregate resource telemetry for sizing and comparison studies</td>
        </tr>
      </tbody>
    </table>

    <div class="alert">
      <strong>Concrete example:</strong> A verification report at <code>artifacts/experiments/20260514_170536_test_strategies_verification/REPORT.txt</code> documents 60 runs. Sample answer inspection: <code>python3 show_samples.py results.jsonl default 2</code>
    </div>
  </section>

  <!-- ── EVALUATION ── -->
  <section id="evaluation">
    <span class="section-label">08 — evaluation</span>
    <h2><span class="h2-num">08</span> Evaluation Workflows</h2>
    <p>A separate evaluation workspace under <code>evaluation/</code> enables paper-oriented metric computation. Use it when you need gold QA datasets, retrieval metrics, or RAGAS-based answer quality scores.</p>

    <div class="flow">
      <div class="flow-step">
        <span class="flow-num">01</span>
        <div class="flow-text">
          <strong>Prepare labels</strong>
          <span>Fill <code>evaluation/gold_questions_template.csv</code> with your gold question/answer pairs.</span>
        </div>
      </div>
      <div class="flow-step">
        <span class="flow-num">02</span>
        <div class="flow-text">
          <strong>Build eval dataset</strong>
          <span>Join a pipeline run output with the gold set: <code>evaluation/build_eval_dataset.py</code>.</span>
        </div>
      </div>
      <div class="flow-step">
        <span class="flow-num">03</span>
        <div class="flow-text">
          <strong>Retrieval metrics</strong>
          <span>Compute entity coverage and rank-based scores: <code>evaluation/retrieval_metrics.py</code>.</span>
        </div>
      </div>
      <div class="flow-step">
        <span class="flow-num">04</span>
        <div class="flow-text">
          <strong>RAGAS evaluation (optional)</strong>
          <span>Run LLM-judge answer quality metrics: <code>evaluation/run_ragas_eval.py</code>.</span>
        </div>
      </div>
    </div>
  </section>

  <!-- ── SMOKE TESTS ── -->
  <section id="smoke">
    <span class="section-label">09 — testing</span>
    <h2><span class="h2-num">09</span> Smoke Tests &amp; Preflight</h2>

    <div class="cards">
      <div class="card">
        <h4>Quick smoke check</h4>
        <p>Run after install to verify the package is healthy.</p>
        <pre style="margin-top:.8rem"><code>python scripts/smoke_check.py</code></pre>
      </div>
      <div class="card">
        <h4>Local preflight (PowerShell)</h4>
        <p>Validates the local environment before a long run on Windows.</p>
        <pre style="margin-top:.8rem"><code>powershell -ExecutionPolicy Bypass \
  -File scripts/preflight.ps1</code></pre>
      </div>
    </div>

    <div class="alert" style="margin-top:1.2rem">
      <strong>Recommended workflow:</strong> always run the smoke matrix first and verify that both <code>summary.json</code> and <code>results.jsonl</code> are created in the output directory before launching a full experiment.
    </div>
  </section>

  <!-- ── CLUSTER ── -->
  <section id="cluster">
    <span class="section-label">10 — cluster</span>
    <h2><span class="h2-num">10</span> Cluster &amp; Batch Jobs</h2>
    <p>SLURM job templates are provided for both CPU and GPU nodes.</p>

    <table class="env-table">
      <thead>
        <tr><th>Template</th><th>Target</th></tr>
      </thead>
      <tbody>
        <tr>
          <td class="key">scripts/run_graphrag.sbatch</td>
          <td class="desc">GPU nodes</td>
        </tr>
        <tr>
          <td class="key">scripts/run_graphrag_cpu.sbatch</td>
          <td class="desc">CPU-only nodes</td>
        </tr>
      </tbody>
    </table>

    <h3>Submit a job</h3>
    <span class="pre-label">bash</span>
    <pre><code>export NEO4J_URL="neo4j+s://&lt;your-instance&gt;"
export NEO4J_USERNAME="&lt;user&gt;"
export NEO4J_PASSWORD="&lt;pass&gt;"
export NEO4J_DATABASE="&lt;db&gt;"

# GPU job
sbatch -p &lt;gpu_partition&gt; scripts/run_graphrag.sbatch

# CPU job
sbatch -p &lt;cpu_partition&gt; scripts/run_graphrag_cpu.sbatch</code></pre>

    <p>See <code>docs/cluster.md</code> for additional cluster-specific guidance.</p>
  </section>

  <!-- ── STRUCTURE ── -->
  <section id="structure">
    <span class="section-label">11 — repository</span>
    <h2><span class="h2-num">11</span> Repository Structure</h2>
    <div class="tree">
      <div><span class="dir">graphRAG-pipeline/</span></div>
      <div>&nbsp;&nbsp;├── <span class="dir">kg_pipeline/</span>     <span class="ann">← knowledge graph pipeline + config.yaml</span></div>
      <div>&nbsp;&nbsp;├── <span class="dir">src/graphrag/</span>    <span class="ann">← main package, CLI entry point, agent code</span></div>
      <div>&nbsp;&nbsp;├── <span class="dir">scripts/</span>         <span class="ann">← smoke tests, matrix runner, batch templates, analysis</span></div>
      <div>&nbsp;&nbsp;├── <span class="dir">docs/</span>            <span class="ann">← additional docs (cluster.md, …)</span></div>
      <div>&nbsp;&nbsp;├── <span class="dir">evaluation/</span>      <span class="ann">← evaluation workspace (metrics, RAGAS)</span></div>
      <div>&nbsp;&nbsp;├── <span class="dir">artifacts/</span>       <span class="ann">← experiment outputs, logs, reports</span></div>
      <div>&nbsp;&nbsp;├── <span class="file">requirements.txt</span></div>
      <div>&nbsp;&nbsp;├── <span class="file">requirements-cpu.txt</span></div>
      <div>&nbsp;&nbsp;└── <span class="file">requirements-gpu.txt</span></div>
    </div>
  </section>

  <!-- ── CONTRIBUTING ── -->
  <section id="contributing">
    <span class="section-label">12 — contributing</span>
    <h2><span class="h2-num">12</span> Contributing</h2>
    <ul style="list-style:none;padding:0;display:flex;flex-direction:column;gap:.7rem">
      <li style="display:flex;gap:.8rem;align-items:flex-start"><span style="color:var(--accent2)">→</span> Open issues or pull requests for bugs and feature improvements.</li>
      <li style="display:flex;gap:.8rem;align-items:flex-start"><span style="color:var(--accent2)">→</span> Run smoke tests and a minimal experiment locally before submitting a PR.</li>
      <li style="display:flex;gap:.8rem;align-items:flex-start"><span style="color:var(--accent2)">→</span> Keep changes small and reproducible. If you modify retrieval logic, include a note about the affected strategy or output artifact.</li>
    </ul>
  </section>

  <!-- ── TROUBLESHOOTING ── -->
  <section id="troubleshooting">
    <span class="section-label">13 — troubleshooting</span>
    <h2><span class="h2-num">13</span> Troubleshooting</h2>
    <ul class="trouble-list">
      <li>
        <span class="ti">⚠</span>
        <div>
          <strong style="color:var(--heading);display:block;margin-bottom:.3rem">Cannot connect to Neo4j</strong>
          Verify <code>NEO4J_URL</code>, <code>NEO4J_USERNAME</code>, and <code>NEO4J_PASSWORD</code> are set correctly and that the database name matches the target instance.
        </div>
      </li>
      <li>
        <span class="ti">⚠</span>
        <div>
          <strong style="color:var(--heading);display:block;margin-bottom:.3rem">Local model loading fails</strong>
          Try a smaller model first, or reduce <code>--max-new-tokens</code> and review GPU memory settings with <code>--gpu-memory-fraction</code>.
        </div>
      </li>
      <li>
        <span class="ti">⚠</span>
        <div>
          <strong style="color:var(--heading);display:block;margin-bottom:.3rem">vLLM run produces no answers</strong>
          Confirm that <code>--vllm-base-url</code> and <code>--model-id</code> exactly match what the server reports. Check the server process is running.
        </div>
      </li>
      <li>
        <span class="ti">⚠</span>
        <div>
          <strong style="color:var(--heading);display:block;margin-bottom:.3rem">Experiments complete but context is empty</strong>
          Inspect <code>summary.json</code> and <code>results.jsonl</code> before making pipeline changes. The issue is often in retrieval configuration rather than the LLM.
        </div>
      </li>
    </ul>
  </section>

  <!-- ── FOOTER ── -->
  <footer class="footer">
    <div>
      <span class="footer-brand">GraphRAG Pipeline</span>
      <p style="margin-top:.3rem;font-size:11px">Active development — May 2026</p>
    </div>
    <div style="text-align:right;font-size:11px;color:var(--muted)">
      <p>Neo4j · Hugging Face · vLLM</p>
      <p>Python 3.10+ · Conda graphllm</p>
    </div>
  </footer>

</div>

<!-- Graph canvas animation -->
<script>
(function(){
  const canvas = document.getElementById('bg-canvas');
  const ctx = canvas.getContext('2d');
  let W, H, nodes, edges;
  const N = 55;

  function resize(){
    W = canvas.width = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function init(){
    resize();
    nodes = Array.from({length:N}, () => ({
      x: Math.random() * W,
      y: Math.random() * H,
      vx: (Math.random()-.5) * .35,
      vy: (Math.random()-.5) * .35,
      r: Math.random() * 2 + 1,
      pulse: Math.random() * Math.PI * 2
    }));
  }

  function draw(){
    ctx.clearRect(0, 0, W, H);
    const t = Date.now() * .001;

    // edges
    for(let i=0; i<nodes.length; i++){
      for(let j=i+1; j<nodes.length; j++){
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
        const d = Math.sqrt(dx*dx+dy*dy);
        if(d < 130){
          const alpha = (1 - d/130) * .18;
          ctx.beginPath();
          ctx.strokeStyle = `rgba(0,229,255,${alpha})`;
          ctx.lineWidth = .5;
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);
          ctx.stroke();
        }
      }
    }

    // nodes
    nodes.forEach(n => {
      n.pulse += .018;
      const glow = .5 + .5 * Math.sin(n.pulse);
      const radius = n.r + glow * .8;
      const alpha = .3 + glow * .35;

      ctx.beginPath();
      ctx.arc(n.x, n.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0,229,255,${alpha})`;
      ctx.fill();

      n.x += n.vx; n.y += n.vy;
      if(n.x < 0 || n.x > W) n.vx *= -1;
      if(n.y < 0 || n.y > H) n.vy *= -1;
    });

    requestAnimationFrame(draw);
  }

  window.addEventListener('resize', resize);
  init();
  draw();
})();
</script>

</body>
</html>
