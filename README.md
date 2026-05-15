
# GraphRAG Pipeline

Una pipeline per esperimenti RAG che combina retrieval su grafo (Neo4j) e generazione con LLM locali o server vLLM.

**Stato del codice (maggio 2026)**

- Sviluppo attivo: il repository contiene una pipeline di Knowledge Graph (`kg_pipeline/`), un motore di retrieval ibrido GraphRAG + text-RAG, e script per eseguire matrici di esperimenti.
- Entrypoints principali funzionanti: CLI `graphrag-demo` e runner della pipeline `python -m kg_pipeline.main`.
- Ambito di uso: ricerca sperimentale e confronto tra strategie di retrieval e modelli LLM (locale o server vLLM).
- Requisiti: ambiente Conda suggerito `graphllm` (vedi `requirements-*.txt`).

## Contenuto principale

- `kg_pipeline/`: costruzione del knowledge graph da documenti (stages: chunking, NER, estrazione triple, risoluzione, linking, ingest in Neo4j).
- `src/` e package `graphrag`: implementazione del CLI e dei runner per esperimenti e demo.
- `scripts/`: comandi di utilità per smoke tests, run cluster, analisi e visualizzazione.
- `artifacts/`: output di esperimenti, log, report.

## Installazione rapida (locale, Conda)

1. Creare e attivare l'ambiente Conda (si raccomanda `graphllm`):

```bash
conda create -n graphllm python=3.10 -y
conda activate graphllm
```

2. Installare dipendenze (usa la versione adatta al tuo nodo):

```bash
pip install -r requirements.txt        # base
pip install -r requirements-cpu.txt    # CPU-only
pip install -r requirements-gpu.txt    # GPU (se disponibile)
pip install -e .
```

3. Configurare Neo4j e variabili (file `.env` o export):

```bash
export NEO4J_URL="neo4j+s://<your-instance>"
export NEO4J_USERNAME="<user>"
export NEO4J_PASSWORD="<pass>"
export NEO4J_DATABASE="<db>"  # opzionale
```

Per modelli gated su Hugging Face: impostare `HF_TOKEN` nell'ambiente.

## Esecuzione rapida

- Demo CLI (domanda singola):

```bash
graphrag-demo --question "Quali sono le relazioni tra Entita A e Entita B?" --entity "Entita A"
```

- Abilitare LLM locale:

```bash
graphrag-demo --llm --model-id Qwen/Qwen2.5-7B-Instruct
```

- Usare server vLLM (API compatibile OpenAI):

```bash
graphrag-demo --llm --vllm --vllm-base-url http://localhost:8000/v1 --model-id Qwen/Qwen2.5-7B-Instruct
```

## Eseguire la Knowledge Graph pipeline

Esempio di esecuzione (log su stdout):

```bash
conda activate graphllm
PYTHONUNBUFFERED=1 python -m kg_pipeline.main --config kg_pipeline/config.yaml --env-file kg_pipeline/.env --log-level INFO
```

Per eseguire in batch su cluster, usare gli script in `scripts/` (es. `run_kg_pipeline.sbatch`). I run vengono salvati sotto `kg_pipeline/artifacts/`.

## Esperimenti e matrici

Gli script di esperimento consentono di valutare strategie di retrieval (text-only vs GraphRAG) e di confrontare modelli LLM.

Esempio smoke matrix:

```bash
python scripts/run_retrieval_matrix.py --smoke --questions-file artifacts/experiments/questions_smoke.txt --documents docs/ README.md --runs-per-strategy 1 --output-dir artifacts/experiments --experiment-tag retrieval_matrix_smoke
```

Per run più lunghe e tuning su GPU consultare `scripts/run_experiment_matrix_gpu.sbatch`.

## Analisi dei risultati

- `scripts/analyze_experiments.py` per analisi singolo-run.
- `scripts/analyze_matrix.py` e `scripts/analyze_resource_usage.py` per aggregazioni di più run e profiling risorse.

## Smoke tests e preflight

- Controllo rapido dopo installazione: `python scripts/smoke_check.py`
- Preflight locale (PowerShell helper): `powershell -ExecutionPolicy Bypass -File scripts/preflight.ps1`

## Struttura del repository (riassunto)

- `kg_pipeline/` – pipeline KG
- `src/graphrag/` – implementazione principale e CLI
- `scripts/` – utilità e job templates
- `docs/` – documentazione aggiuntiva (es. cluster.md)
- `artifacts/` – dati di output, report e risultati sperimentali

## Contribuire

- Aprire issue o pull request per bug e miglioramenti.
- Seguire le istruzioni di style e testing del progetto (eseguire smoke tests prima di PR).

## Licenza

Verificare il file di licenza nel repository (se presente) prima di riutilizzare codice o dataset.

---

Aggiornamenti: README aggiornato per riflettere lo stato corrente del codice e comandi principali. Se vuoi, posso:

- Aggiungere badge di CI/coverage/conda.
- Tradurre in inglese o mantenere versione bilingue.
- Inserire esempi più specifici (es. run effettivo con modelli e output di esempio).

File aggiornato: [README.md](README.md)

