# GraphRAG Pipeline

Progetto GraphRAG con retrieval su Neo4j, orchestrazione a grafo e generazione opzionale con modello locale Hugging Face.

## Cosa e cambiato

- Refactor da monolite a package modulare in src.
- Rimozione credenziali hardcoded: ora tutto passa da variabili ambiente.
- CLI pronta per demo locale e sviluppo incrementale.
- Struttura pronta per versionamento e pubblicazione su GitHub.

## Struttura

- src/graphrag/config.py: configurazione agente e KG.
- src/graphrag/types.py: stati e tipi condivisi.
- src/graphrag/kg/: manager Neo4j, retriever, seed dataset.
- src/graphrag/agent/: cache, compressione contesto, agente LangGraph.
- src/graphrag/llm/: prompt library e manager modello locale.
- src/graphrag/experiments/: runner esperimenti e export JSONL.
- src/graphrag/cli.py: entrypoint da linea di comando.

## Setup rapido

1. Crea ambiente virtuale e installa dipendenze.
2. Copia .env.example in .env e imposta le credenziali Neo4j.
3. Avvia la demo.

Comandi consigliati:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
Copy-Item .env.example .env
graphrag-demo --seed-movie-dataset
```

Per abilitare la generazione con LLM locale:

```powershell
graphrag-demo --llm --model-id Qwen/Qwen2.5-3B-Instruct
```

## Miglioramenti consigliati

- Aggiungere test unitari su prompt routing e retrieval strategy.
- Aggiungere benchmark automatici su diversi dataset.
- Versionare prompt e config per esperimenti riproducibili.

## Pubblicazione su GitHub

Se non hai ancora creato il repository remoto, crealo su GitHub e poi esegui:

```powershell
git add .
git commit -m "Initial GraphRAG modular refactor"
git branch -M main
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```
