# COMMANDS.md — Guida ai comandi del progetto

Riferimento rapido a tutti i comandi eseguibili nella repo. Organizzato per area funzionale.

---

## 1. GraphRAG CLI — `graphrag-demo`

Entry point principale per retrieval e generazione su singola domanda o batch.

> Se `graphrag-demo` restituisce exit code 126 (shim stantio), sostituire con  
> `conda run -n graphllm python -m graphrag.cli`.

### Domanda singola — solo retrieval

```bash
conda run -n graphllm python -m graphrag.cli \
  --question "Qual è la relazione tra X e Y?" \
  --entity "X" \
  --strategies default
```

### Domanda singola con generazione LLM locale

```bash
conda run -n graphllm python -m graphrag.cli \
  --question "Qual è la relazione tra X e Y?" \
  --entity "X" \
  --llm \
  --model-id meta-llama/Meta-Llama-3-8B-Instruct
```

### Domanda singola con vLLM (endpoint OpenAI-compatible)

```bash
conda run -n graphllm python -m graphrag.cli \
  --question "Qual è la relazione tra X e Y?" \
  --llm \
  --vllm \
  --vllm-base-url http://localhost:8000/v1
```

### Batch experiment su file di domande

```bash
conda run -n graphllm python -m graphrag.cli \
  --experiment \
  --questions-file questions.txt \
  --strategies "default,text_plus_triples" \
  --llm \
  --vllm \
  --output-dir artifacts/experiments \
  --experiment-tag mio_test
```

### Parametri principali

| Flag | Effetto |
|------|---------|
| `--question` | Domanda da porre (default: placeholder italiano) |
| `--entity` | Entità seed per la traversal del grafo (vuoto = auto) |
| `--llm` | Abilita la generazione LLM; senza di esso gira solo il retrieval |
| `--vllm` | Usa un endpoint vLLM invece di caricare il modello HF localmente |
| `--vllm-base-url` | URL base dell'API vLLM (default: `http://localhost:8000/v1`) |
| `--model-id` | ID modello HuggingFace o vLLM |
| `--llm-warmup` | Pre-carica il modello all'avvio |
| `--strategies` | Strategie di retrieval separate da virgola (v. §7) |
| `--enable-decomposition-step` | Aggiunge un LLM call prima del retrieval (più latenza) |
| `--enable-adaptive-routing-step` | Aggiunge routing adattivo prima del retrieval (più latenza) |
| `--max-new-tokens` | Token massimi generati (default: 256) |
| `--max-context-tokens` | Token massimi del contesto compresso (default: 1000) |
| `--gpu-memory-fraction` | Frazione GPU riservata al modello (default: 0.92) |
| `--allow-large-model-fp16-fallback` | Fallback fp16 per modelli grandi se il caricamento 4-bit fallisce |
| `--experiment` | Modalità batch; scrive artifact strutturati |
| `--questions-file` | File UTF-8 con una domanda per riga |
| `--runs-per-strategy` | Ripetizioni per strategia (default: 1) |
| `--output-dir` | Directory output esperimenti (default: `artifacts/experiments`) |
| `--experiment-tag` | Tag identificativo del run |
| `--recursion-limit` | Passi massimi LangGraph prima di abortire (default: 50) |

---

## 2. KG Pipeline — `kg_pipeline/main.py`

Costruisce il knowledge graph da documenti e lo inserisce in Neo4j. Gira in 7 stage sequenziali con checkpoint JSON (si può riprendere da dove si è interrotto).

### Pipeline completa

```bash
conda run -n graphllm python -m kg_pipeline.main \
  --config kg_pipeline/config.yaml \
  --env-file kg_pipeline/.env \
  --log-level INFO
```

### Singolo stage

```bash
conda run -n graphllm python -m kg_pipeline.main \
  --config kg_pipeline/config.yaml \
  --env-file kg_pipeline/.env \
  --stage ner
```

### Solo un documento

```bash
conda run -n graphllm python -m kg_pipeline.main \
  --config kg_pipeline/config.yaml \
  --env-file kg_pipeline/.env \
  --single-doc documento.pdf
```

### Dry-run (nessuna scrittura)

```bash
conda run -n graphllm python -m kg_pipeline.main \
  --config kg_pipeline/config.yaml \
  --dry-run
```

### Parametri principali

| Flag | Effetto |
|------|---------|
| `--config` | Path al file di configurazione (default: `kg_pipeline/config.yaml`) |
| `--env-file` | File `.env` con credenziali Neo4j e altri segreti |
| `--run-dir` | Directory run specifica; se vuota ne crea una con timestamp |
| `--single-doc` | Processa un solo documento (filename o doc_id) |
| `--stage` | Esegue solo lo stage indicato: `all` `ingestion` `chunking` `ner` `llm` `resolution` `linking` `neo4j` |
| `--dry-run` | Simula l'esecuzione senza scrivere output |
| `--log-level` | Livello di log: `DEBUG` `INFO` `WARNING` (default: `INFO`) |

### Stage e artifact prodotti

| Stage | Cosa fa | Artifact |
|-------|---------|----------|
| 0 — ingestion | PDF → markdown | `stage0_documents.json` |
| 1 — chunking | Suddivisione in chunk token-windowed | `stage1_chunks.json` |
| 2 — ner | Named entity recognition con GLiNER | `stage2_ner.json` |
| 3 — llm | Estrazione triple con LLM | `stage3_triples_raw.json`, `stage3_acronyms.json` |
| 4 — resolution | Entity resolution (embedding + Jaccard) | `stage4_triples_resolved.json`, `stage4_registry.json` |
| 5 — linking | Collegamento triple | `stage5_triples_linked.json` |
| 6 — neo4j | Inserimento in Neo4j | `stage6_neo4j_summary.json` |

Prodotti aggiuntivi: `failed_chunks.jsonl`, `new_labels.log`, `pipeline.log`.  
**Nota**: lo stage 3 fa checkpoint ogni N chunk in `stage3_checkpoint.json`; rilanciare senza cancellarlo riprende dall'ultimo chunk salvato.

---

## 3. Experiment matrix — `scripts/run_retrieval_matrix.py`

Confronta strategie di retrieval e configurazioni LLM su un set di domande. Scrive i risultati in `artifacts/experiments/<timestamp>_<tag>/`.

```bash
python scripts/run_retrieval_matrix.py \
  --questions-file questions.txt \
  --models "7b,32b" \
  --strategies "default,text_plus_triples" \
  --output-dir artifacts/experiments
```

### Con vLLM e tag

```bash
python scripts/run_retrieval_matrix.py \
  --questions-file questions.txt \
  --strategies "default,text_plus_triples,subgraph_2hop" \
  --llm \
  --vllm \
  --model-id Qwen/Qwen2.5-32B-Instruct \
  --output-dir artifacts/experiments \
  --experiment-tag confronto_strategie
```

### Solo retrieval grafico (skip standard RAG)

```bash
python scripts/run_retrieval_matrix.py \
  --questions-file questions.txt \
  --skip-standard \
  --strategies "neighbors_focus,subgraph_2hop,shortest_path"
```

### Parametri principali

| Flag | Effetto |
|------|---------|
| `--questions-file` | File con una domanda per riga |
| `--strategies` | Strategie separate da virgola |
| `--models` | Preset di modelli (`7b`, `32b`, ecc.) |
| `--llm` | Abilita generazione LLM |
| `--vllm` | Usa endpoint vLLM |
| `--model-id` | Modello specifico (override) |
| `--runs-per-strategy` | Ripetizioni per strategia |
| `--output-dir` | Directory output |
| `--experiment-tag` | Tag identificativo |
| `--skip-standard` | Salta standard RAG, gira solo GraphRAG |
| `--skip-graph` | Salta GraphRAG, gira solo standard RAG |
| `--performance-profile` | `auto` / `default` / `production_fast` |
| `--enable-decomposition-step` | Aggiunge decomposizione LLM |
| `--enable-adaptive-routing-step` | Aggiunge routing adattivo |
| `--max-new-tokens` | Token massimi per risposta |
| `--gpu-memory-fraction` | Frazione GPU riservata |
| `--allow-large-model-fp16-fallback` | Fallback fp16 per modelli grandi |
| `--llm-warmup` | Pre-carica il modello prima dei run |
| `--smoke` | Test rapido su N domande ridotte |
| `--smoke-questions` | Numero domande in smoke mode (default: 2) |

### Output prodotti

```
artifacts/experiments/<timestamp>_<tag>/
  results.jsonl
  results.csv
  summary.txt
  summary.json
  resource_samples.jsonl
  resource_summary.json
```

---

## 4. A/B test fast profile — `scripts/run_ab_fast_profile.py`

Confronta il profilo `default` vs `production_fast` per il modello 32B e riporta delta di latenza e qualità.

```bash
python scripts/run_ab_fast_profile.py \
  --model-id Qwen/Qwen2.5-32B-Instruct \
  --questions-file questions_matrix_long.txt \
  --questions-count 10 \
  --graph-strategies default \
  --vllm \
  --output-dir artifacts/experiments \
  --report-dir artifacts/evaluation
```

---

## 5. Analisi risultati — `scripts/analyze_experiments.py`

Analizza gli artifact di un run e produce una classifica per metrica.

```bash
python scripts/analyze_experiments.py \
  --results-dir artifacts/experiments \
  --output-csv results_ranked.csv
```

```bash
python scripts/analyze_experiments.py artifacts/experiments/20240601_120000_mio_test
```

---

## 6. Analisi resource usage — `scripts/analyze_resource_usage.py`

Aggrega telemetria GPU/CPU da più run.

```bash
python scripts/analyze_resource_usage.py artifacts/experiments \
  --tag-contains confronto \
  --output-csv resource_report.csv
```

---

## 7. Analisi matrix — `scripts/analyze_matrix.py`

Analisi comparativa aggregata tra più run/folder di esperimenti.

```bash
python scripts/analyze_matrix.py \
  --root artifacts/experiments \
  --tag-contains confronto_strategie \
  --output-csv matrix_summary.csv
```

---

## 8. Re-merge entità — `scripts/remerge_entities.py`

Ri-esegue entity resolution e linking su output stage 3 esistenti, senza rifare NER o estrazione LLM.  
Utile per regolare soglie di similarità senza rigirare tutta la pipeline.

```bash
python scripts/remerge_entities.py \
  --run-dir kg_pipeline/artifacts/run_20240601_120000 \
  --similarity-threshold 0.90 \
  --context-jaccard-floor 0.15
```

| Flag | Effetto |
|------|---------|
| `--run-dir` | Directory del run con output stage 3 (obbligatorio) |
| `--output-dir` | Directory alternativa per stage 4/5 (opzionale) |
| `--embedding-model` | Modello SentenceTransformer per resolution |
| `--similarity-threshold` | Soglia similarità coseno (default: 0.88) |
| `--context-jaccard-floor` | Soglia minima Jaccard di contesto (default: 0.15) |
| `--base-url` | URL endpoint vLLM |
| `--model-name` | Nome modello vLLM |

---

## 9. Generazione domande test — `scripts/generate_questions.py`

Genera automaticamente un test suite di domande dai chunk/documenti della pipeline.

### Genera suite da run esistente

```bash
python scripts/generate_questions.py generate \
  --run-dir kg_pipeline/artifacts/run_20240601_120000 \
  --output artifacts/tmp/graphrag_test_suite.json
```

### Solo per un documento specifico

```bash
python scripts/generate_questions.py generate \
  --run-dir kg_pipeline/artifacts/run_20240601_120000 \
  --doc mio_documento.pdf \
  --output artifacts/tmp/suite_doc.json
```

### Statistiche su suite esistente

```bash
python scripts/generate_questions.py stats \
  --input artifacts/tmp/graphrag_test_suite.json
```

---

## 10. Visualizzazione KG — `scripts/visualize_kg.py`

Genera un HTML interattivo del knowledge graph da Neo4j.

```bash
python scripts/visualize_kg.py \
  --output artifacts/tmp/kg_viz.html
```

Richiede le variabili d'ambiente `NEO4J_URL`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`.

---

## 11. Evaluation — `evaluation/`

### Build dataset di valutazione

Unisce i risultati dei run con le label gold.

```bash
python evaluation/build_eval_dataset.py \
  --input artifacts/experiments \
  --gold-file evaluation/gold_questions_template.csv \
  --output artifacts/evaluation/eval_dataset.csv
```

| Flag | Effetto |
|------|---------|
| `--input` | Root esperimenti, singolo run folder, o `results.csv` |
| `--gold-file` | CSV gold con almeno colonne `question` e `ground_truth` |
| `--tag-contains` | Filtro opzionale sul nome del run folder |
| `--output` | Output CSV del dataset unificato |
| `--smoke` | Modalità smoke su fixture locali |

### Metriche di retrieval

```bash
python evaluation/retrieval_metrics.py \
  --input artifacts/evaluation/eval_dataset.csv \
  --k 5 \
  --save-csv artifacts/evaluation/metrics.csv \
  --save-json artifacts/evaluation/metrics.json
```

| Flag | Effetto |
|------|---------|
| `--input` | CSV prodotto da `build_eval_dataset.py` |
| `--k` | Top-k per precision/recall/hit/NDCG |
| `--n-bootstrap` | Numero di resample bootstrap (default: 1000) |
| `--ci` | Livello confidence interval (default: 0.95) |
| `--save-csv` / `--save-json` | Path output metriche aggregate |
| `--save-row-csv` | Path output metriche riga per riga |

### Valutazione RAGAS (opzionale)

```bash
python evaluation/run_ragas_eval.py \
  --input artifacts/evaluation/eval_dataset.csv \
  --save-summary-json artifacts/evaluation/ragas_summary.json \
  --save-row-csv artifacts/evaluation/ragas_rows.csv
```

---

## 12. Smoke test e health check

| Comando | Cosa verifica |
|---------|---------------|
| `python scripts/smoke_check.py` | Connettività Neo4j + LLM (legge env vars esportate, non `.env`) |
| `python scripts/smoke_text_rag.py docs/ --query "..." --top-k 4` | Retrieval testuale su directory di documenti |
| `python scripts/smoke_kg_retriever.py` | Retrieval sul knowledge graph in Neo4j |
| `python scripts/smoke_test_pipeline.py` | Smoke rapido sulla pipeline KG |
| `python scripts/run_pipeline_smoke_full.py` | Smoke end-to-end completo della pipeline |
| `pytest kg_pipeline/tests/test_pipeline.py -v` | Test unitari/integrazione pipeline |
| `pytest evaluation/tests/test_metrics.py -v` | Test delle metriche di valutazione |

---

## 13. SLURM — esecuzione su cluster

| Script | Cosa lancia |
|--------|-------------|
| `sbatch scripts/run_kg_pipeline.sbatch` | Pipeline KG in background (evita hang su disconnessione notebook) |
| `sbatch scripts/run_graphrag.sbatch` | GraphRAG demo su cluster GPU |
| `sbatch scripts/run_graphrag_cpu.sbatch` | GraphRAG demo su cluster CPU |
| `sbatch scripts/run_experiment_matrix_gpu.sbatch` | Experiment matrix su GPU cluster |
| `bash scripts/submit_matrix_from_env.sh` | Submette la matrix leggendo parametri da env vars |

---

## 14. Strategie di retrieval disponibili

Da passare a `--strategies` (singola o lista separata da virgola):

| Strategia | Descrizione |
|-----------|-------------|
| `default` | Bilanciamento testo + grafo |
| `text_only` | Solo retrieval testuale (standard RAG) |
| `text_plus_triples` | Testo + triple del KG |
| `neighbors_focus` | Vicini diretti dell'entità seed |
| `subgraph_2hop` | Sottografo a 2 hop dall'entità seed |
| `shortest_path` | Cammino minimo tra entità |

---

## 15. Variabili d'ambiente richieste

```bash
NEO4J_URL="bolt://localhost:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="..."
NEO4J_DATABASE="..."          # opzionale
HF_TOKEN="..."                # per modelli gated su HuggingFace
VLLM_BASE_URL="http://localhost:8000/v1"
VLLM_MODEL_NAME="Qwen/Qwen2.5-32B-Instruct"
VLLM_API_KEY="..."            # oppure OPENAI_API_KEY
```

> `scripts/smoke_check.py` legge le env var esportate nella shell — **non** carica `.env` automaticamente.  
> Per la pipeline KG usare sempre `--env-file kg_pipeline/.env`.
