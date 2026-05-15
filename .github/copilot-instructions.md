# GraphRAG Pipeline Codebase Instructions

> **Purpose**: This document provides comprehensive context for GitHub Copilot to work efficiently on this codebase, reducing token waste and improving suggestion accuracy.
>
> **Last Updated**: May 2026

---

## Project Overview

GraphRAG Pipeline is an **experiment-oriented Retrieval-Augmented Generation system** combining knowledge graph extraction with LLM-based answer generation. The pipeline:

1. **Ingests** documents (PDF, Markdown) and chunks them by semantic/size boundaries
2. **Extracts** entities (NER via GLiNER) and triples (LLM-based structured extraction)
3. **Resolves** entities across chunks and builds a Neo4j knowledge graph
4. **Retrieves** evidence using multi-strategy graph traversal (nodes, triples, neighbors, 2-hop subgraphs, shortest paths)
5. **Generates** answers by constructing LLM prompt context from retrieved evidence

Primary entry points: `graphrag-demo` (CLI) and `python -m kg_pipeline.main` (KG builder).

---

## Architecture

### High-Level Flow

```
┌────────────────────────────────────────────────────────────────────┐
│ KG PIPELINE (kg_pipeline/)                                         │
│ ─────────────────────────────────────────────────────────────────  │
│  Stage 0: Ingest (PDFs → markdown documents)                      │
│    ↓                                                                │
│  Stage 1: Chunk (paragraph windowing with overlap)                │
│    ↓                                                                │
│  Stage 2: NER (GLiNER → entity candidates)                        │
│    ↓                                                                │
│  Stage 3: LLM Extraction (structured triples + acronym tracking)  │
│    ↓                                                                │
│  Stage 4: Resolution (entity canonicalization + dedup)            │
│    ↓                                                                │
│  Stage 5: Linking (add document links, optional MENTIONED_IN)     │
│    ↓                                                                │
│  Stage 6: Neo4j Ingestion (create nodes/relationships)            │
│    ↓                                                                │
│  [Neo4j Knowledge Graph]                                           │
└────────────────────────────────────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────────────┐
│ RETRIEVAL & GENERATION (src/graphrag/)                             │
│ ─────────────────────────────────────────────────────────────────  │
│  KGRetriever → multi-strategy retrieval from Neo4j                │
│    • Nodes + Triples (search term based)                          │
│    • Neighbors (direct 1-hop)                                     │
│    • Subgraph (N-hop exploration)                                 │
│    • Shortest Path (entity A → entity B)                          │
│    ↓                                                                │
│  KGRAGAgent (LangGraph state machine) →                           │
│    1. Decompose (optional: break question into sub-questions)     │
│    2. Adaptive Route (optional: choose retrieval strategy)        │
│    3. Retrieve (execute retrieval plan)                           │
│    4. Grade (check relevance; rewrite if needed)                  │
│    5. Generate (build prompt context + call LLM)                  │
│    ↓                                                                │
│  [LLM Manager] → local HuggingFace OR vLLM server                 │
│    ↓                                                                │
│  [Answer + Provenance + Latency Telemetry]                        │
└────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Module | Role |
|-----------|--------|------|
| **KG Pipeline** | `kg_pipeline/` | Orchestrates document→KG flow with checkpoint recovery |
| **KG Retriever** | `src/graphrag/kg/retriever.py` | Multi-strategy evidence extraction from Neo4j |
| **KGRAGAgent** | `src/graphrag/agent/core.py` | LangGraph-based state machine for adaptive retrieval+gen |
| **LLMManager** | `src/graphrag/llm/manager.py` | Lazy load local models (HuggingFace) or vLLM clients |
| **KnowledgeGraphManager** | `src/graphrag/kg/manager.py` | Neo4j query builder + retry logic |
| **Experiment Runner** | `src/graphrag/experiments/runner.py` | Multi-question, multi-strategy matrix executor |

---

## Project Structure

```
graphRAGPipelineExp1/
├── .github/
│   ├── workflows/
│   │   └── ci.yml                          # GitHub Actions CI/CD
│   └── copilot-instructions.md             # This file
├── pyproject.toml                          # Package metadata + entry point (graphrag-demo)
├── requirements.txt / -cpu.txt / -gpu.txt  # Conda deps per node type
├── README.md                               # End-user guide
│
├── kg_pipeline/                            # Knowledge Graph Extraction Pipeline
│   ├── main.py                             # Orchestrator + checkpoint recovery
│   ├── config.yaml                         # Ontology, chunking, NER, resolution params
│   ├── models/
│   │   └── types.py                        # Pydantic models: DocumentRecord, ChunkRecord, KGTriple, etc.
│   ├── stages/
│   │   ├── ingestion.py                    # PDFs → DocumentRecord via pymupdf4llm
│   │   ├── chunking.py                     # Paragraph windowing with token-based overlap
│   │   ├── ner.py                          # GLiNER entity extraction
│   │   ├── llm_extraction.py               # Structured JSON triple extraction (with retries + checkpoints)
│   │   ├── resolution.py                   # Entity canonicalization (embeddings + similarity threshold)
│   │   ├── linking.py                      # Assign labels to triples, link doc sources
│   │   ├── neo4j_ingestion.py              # Create nodes/relationships in Neo4j with error recovery
│   │   └── neo4j_postprocess.py            # Post-ingestion cleanup/labeling
│   ├── prompts/
│   │   └── extraction_prompt.py            # LLM few-shot prompt for triple extraction
│   ├── utils/
│   │   ├── validation.py                   # JSON parsing, triple validation, failed chunk logging
│   │   └── acronym_map.py                  # Track acronyms found during extraction
│   └── tests/                              # Unit tests for stages
│
├── src/graphrag/                           # Main GraphRAG module
│   ├── cli.py                              # Entry point (graphrag-demo CLI)
│   ├── config.py                           # AgentConfig & KGConfig dataclasses
│   ├── types.py                            # TypedDict for RAGState, KGNode, KGTriple, ProvenanceRecord
│   ├── agent/
│   │   ├── core.py                         # KGRAGAgent: LangGraph orchestrator (6-node state machine)
│   │   ├── cache.py                        # LRUCache for query result caching
│   │   └── compression.py                  # ContextCompressor: token budget enforcement
│   ├── kg/
│   │   ├── manager.py                      # KnowledgeGraphManager: Neo4j query builder + retry logic
│   │   └── retriever.py                    # KGRetriever: multi-strategy search term extraction + collection
│   ├── llm/
│   │   ├── manager.py                      # LLMManager: lazy-load local/vLLM models + perf knobs
│   │   └── prompts.py                      # PromptLibrary: decomposition, routing, generation prompts
│   ├── text_rag/
│   │   ├── agent.py                        # StandardRAGAgent: fallback text-only RAG
│   │   ├── manager.py                      # TextRAGManager: BM25 retrieval (LangChain)
│   │   └── pipeline.py                     # StandardTextRAGPipeline: orchestrator
│   └── experiments/
│       ├── runner.py                       # ExperimentRunner: multi-question, multi-strategy execution
│       ├── resource_monitor.py             # ResourceMonitor: GPU/CPU/memory/latency tracking
│       └── telemetry.py                    # Serialize run results to CSV + JSON
│
├── scripts/
│   ├── run_retrieval_matrix.py             # Main experiment matrix launcher (A/B, performance profiles)
│   ├── run_ab_fast_profile.py              # A/B test: default vs production_fast with quality delta
│   ├── analyze_experiments.py              # Aggregate results.csv, rank by latency/pass-rate
│   ├── smoke_check.py                      # Smoke test: Neo4j health + LLM connectivity
│   ├── smoke_test_pipeline.py              # End-to-end pipeline smoke test
│   └── run_*.sbatch                        # SLURM batch job submission scripts
│
├── evaluation/
│   ├── run_ragas_eval.py                   # RAGAS framework evaluation (faithfulness, relevance, etc.)
│   ├── retrieval_metrics.py                # Compute recall/precision/MAP
│   └── build_eval_dataset.py               # Create gold standard eval set
│
└── docs/
    └── cluster.md                          # Cluster resource allocation guide
```

---

## Data Contracts

### KG Pipeline Data Flow

```python
# Stage 0: Document Ingestion
DocumentRecord:
  doc_id: str                      # unique per document
  filename: str
  page_count: int
  markdown_text: str               # full markdown from pymupdf4llm
  sections: list[SectionRecord]    # (title, level, page range)
  page_chunks: list[PageChunkRecord]
  title: str | None
  publication_year: int | None

# Stage 1: Chunking (via paragraph windowing)
ChunkRecord:
  doc_id: str
  filename: str
  chunk_id: str                    # e.g. "doc_id::chunk_001"
  page_range: str                  # "10-15"
  section_title: str               # inherited from document structure
  chunk_index: int
  text: str                        # joined paragraphs within token budget

# Stage 2: NER
NEREntityCandidate:
  text_span: str                   # actual entity text
  entity_label: str                # GLiNER class (Person, Location, etc.)
  start_char: int
  end_char: int
  confidence_score: float          # [0, 1]

# Stage 3: LLM Extraction (CORE DATA MODEL)
KGTriple:
  subject: str                     # trimmed, non-empty
  predicate: str                   # SCREAMING_SNAKE_CASE (validated via regex)
  object: str                      # trimmed, non-empty
  subject_labels: list[str]        # ontology classes (falls back to ["Concept"])
  object_labels: list[str]         # ditto
  subject_properties: dict[str, Any]    # always includes "name": subject
  object_properties: dict[str, Any]     # always includes "name": object
  relationship_properties: dict[str, Any]  # includes "source_doc", "extraction_method"
  properties: dict[str, Any]       # optional top-level metadata

  # Validators:
  # - subject/object: stripped, non-empty
  # - predicate: SCREAMING_SNAKE_CASE (A-Z, 0-9, _)
  # - labels: normalized to ["Concept"] if empty

# Stage 4: Entity Resolution
CanonicalEntityRecord:
  canonical_name: str              # chosen representative
  aliases: list[str]               # variant names merged
  labels: list[str]                # ontology labels
  merged_properties: dict[str, Any]  # union of all properties
  alias_sources: dict[str, list[str]]  # which chunks contributed each alias

# Final Output (Neo4j)
# Nodes: (Entity:Label {name, property1, property2, ...})
# Rels:  (Entity)-[:PREDICATE {source_doc, extraction_method, ...}]->(Entity)
```

### RAG State (LangGraph)

```python
RAGState (TypedDict):
  # Input
  question: str
  run_id: str
  strategy: str                    # e.g. "default", "text_plus_triples"
  
  # Intermediate steps
  sub_questions: list[str]         # from decomposition
  rewritten_question: str          # if grade failed
  rewrite_count: int
  chosen_retrieval_mode: str       # e.g. "TEXT", "KG_NODES"
  
  # Retrieved context
  text_context: str
  kg_triples: list[Triple]         # {subject, predicate, object}
  kg_context: str
  merged_context: str
  retrieved_nodes_count: int
  retrieved_neighbors_count: int
  retrieved_subgraph_count: int
  retrieved_shortest_path_count: int
  
  # Grading & rewriting
  relevance: Literal["relevant", "not_relevant"]
  confidence: float                # [0, 1]
  confidence_retries: int
  
  # Output
  answer: str
  provenance: list[ProvenanceRecord]  # claim, source_type, source_id, content
  reflection_passed: bool
  reflection_feedback: str
  
  # Telemetry
  latency_ms: float
  node_timings: dict[str, float]   # per LangGraph node
```

### TypedDict Models in `src/graphrag/types.py`

```python
Triple(TypedDict):
  subject: str
  predicate: str
  object: str

KGNode(TypedDict, total=False):
  node_id: str
  labels: list[str]
  properties: dict[str, Any]
  text: str

ProvenanceRecord(TypedDict, total=False):
  claim: str
  source_type: Literal["text_chunk", "kg_triple"]
  source_id: str
  content: str
```

---

## Stack & Libraries

### Core Dependencies

| Package | Version Range | Role | Notes |
|---------|---------------|------|-------|
| **torch** | 2.5.1+ | GPU/CPU backend | Must match `torchvision` version exactly |
| **transformers** | latest | HuggingFace pipelines | Used for local LLM inference |
| **langchain-core** | ^0.1 | LLM abstraction layer | Chain/LCEL composability |
| **langchain-huggingface** | ^0.0.8 | HF pipeline wrapper | For local inference |
| **langchain-community** | ^0.0.x | Neo4j graph abstraction | Neo4jGraph class |
| **langchain-neo4j** | ^0.1 | Improved Neo4j bindings | Replaces community version |
| **langgraph** | ^0.0.x | State machine orchestration | Stateful workflows + graph compilation |
| **pymupdf** / **pymupdf4llm** | ^4.x | PDF → markdown extraction | Preserves structure (headings, tables) |
| **gliner** | ^2.1+ | Named entity recognition | via `urchade/gliner_large-v2.1` |
| **sentence-transformers** | latest | Entity embeddings | For similarity-based resolution |
| **neo4j** | ^5.x | Neo4j driver | Direct Cypher execution |
| **pydantic** | ^2.0 | Data validation | BaseModel + field validators |
| **python-dotenv** | latest | Environment loading | Load `.env` for NEO4J_* creds |
| **openai** | ^1.x | vLLM OpenAI-compatible client | For vLLM server communication |
| **tqdm** | latest | Progress bars | Used in stage pipelines |
| **psutil** | latest | System monitoring | CPU/memory telemetry |
| **bitsandbytes** | (optional, GPU) | 4-bit quantization | For large model inference |
| **accelerate** | latest | Distributed inference | Used by transformers |

### Optional Dependencies

- **bitsandbytes** (GPU nodes): 4-bit quantized model loading
- **requirements-cpu.txt**: torch-cpu variants
- **requirements-gpu.txt**: CUDA-enabled torch + bitsandbytes

### Usage Patterns

```python
# LLM Loading (managed by LLMManager)
from transformers import AutoTokenizer, pipeline
from langchain_huggingface import ChatHuggingFace

# Pydantic models (data validation)
from pydantic import BaseModel, Field, field_validator
kg_triple = KGTriple.model_validate(raw_dict)

# LangGraph (state machine)
from langgraph.graph import StateGraph, START, END
builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve_func)
graph = builder.compile()

# Neo4j queries (retry logic)
from graphrag.kg.manager import KnowledgeGraphManager
results = kg_manager.run_query(cypher, params)

# Environment config
from dotenv import load_dotenv
load_dotenv()
url = os.getenv("NEO4J_URL")
```

---

## Conventions

### Naming

| Category | Pattern | Example | Notes |
|----------|---------|---------|-------|
| **Modules** | `snake_case` | `llm_extraction.py` | Single responsibility |
| **Classes** | `PascalCase` | `KGRAGAgent`, `DocumentRecord` | Data models use `-Record` suffix |
| **Functions** | `snake_case` | `_build_search_terms`, `retrieve` | Private methods start with `_` |
| **Variables** | `snake_case` | `entity_count`, `max_tokens` | Use full descriptive names |
| **Constants** | `SCREAMING_SNAKE_CASE` | `DEFAULT_MODEL_ID`, `MAX_DEPTH` | Module-level config |
| **Pydantic Fields** | `snake_case` | `subject_labels`, `node_id` | Use Field() for validation |
| **Predicates (KG)** | `SCREAMING_SNAKE_CASE` | `HAS_AUTHOR`, `MENTIONS_CONCEPT` | Cypher relationships |
| **Cypher Labels** | `PascalCase` | `:Entity`, `:Document`, `:Region` | Neo4j node types |

### Type Hints

**Rule: Always use type hints for function signatures and complex variables.**

```python
# ✅ Good
def retrieve(self, query: str | None = None) -> dict[str, Any]:
    search_terms: list[str] = self._build_search_terms(query)
    nodes: list[KGNode] = self._collect_nodes(search_terms)
    return {"nodes": nodes, "query": query}

# ❌ Avoid
def retrieve(query=None):
    search_terms = self._build_search_terms(query)
    return search_terms

# Union types (Python 3.10+)
value: str | int | None = None

# Generics
results: list[dict[str, Any]]
mapping: dict[str, list[str]]
```

### Docstrings

**Pattern: Google-style docstrings; brief one-liner + Args/Returns/Raises.**

```python
def _collect_triples(
    self,
    search_terms: list[str],
    limit: int = 20
) -> list[KGTriple]:
    """Retrieve triples matching any search term from Neo4j.
    
    Uses full-text search index; prioritizes exact matches over fuzzy.
    
    Args:
        search_terms: Keywords or entity names to search for.
        limit: Max triples per term (default 20).
        
    Returns:
        Ranked list of KGTriple; may be shorter than limit if few matches.
        
    Raises:
        ValueError: If search_terms is empty.
    """
    if not search_terms:
        raise ValueError("search_terms cannot be empty")
    # implementation...
```

### Error Handling

**Pattern: Catch specific exceptions; log with context; gracefully degrade or fail fast.**

```python
# ❌ Avoid bare except
try:
    result = extract_triples(chunk)
except:
    pass

# ✅ Good: specific exception + logging
try:
    result = extract_triples(chunk)
except ValueError as e:
    logger.warning("Malformed triple in chunk %s: %s", chunk.chunk_id, e)
    result = None
except Exception as e:
    logger.exception("Unexpected error extracting triples from %s: %s", chunk.chunk_id, e)
    write_failed_chunk(failed_path, chunk_metadata, attempt, str(e), raw_response)
    result = None

# Retry logic with backoff (used in KnowledgeGraphManager)
for attempt in range(1, max_attempts + 1):
    try:
        return self.graph.query(cypher, params)
    except Exception as exc:
        if not self._is_retryable_query_error(exc):
            raise
        if attempt < max_attempts:
            time.sleep(self.query_retry_backoff_sec)
        else:
            raise
```

### Logging

**Pattern: Use module-level logger; INFO for milestones; DEBUG for traces; WARNING for recoverable issues.**

```python
import logging

logger = logging.getLogger("graphrag")  # or "kg_pipeline"

# Milestone
logger.info("Starting stage 3: LLM extraction (%d chunks)", len(chunks))

# Progress
logger.debug("Extracting triples from chunk %s (tokens=%d)", chunk_id, token_count)

# Warning (recoverable)
logger.warning("Failed to resolve entity '%s'; using original name", entity_name)

# Error (will be retried)
logger.exception("Query failed for entity %s (attempt %d): %s", entity, attempt, exc)
```

### Imports

**Pattern: Standard library → third-party → local modules; grouped with blank line separators.**

```python
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from pydantic import BaseModel, Field

from kg_pipeline.models.types import ChunkRecord, KGTriple
from kg_pipeline.stages import chunking
```

### Validation (Pydantic)

```python
from pydantic import BaseModel, ConfigDict, Field, field_validator

class MyRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")  # Reject unknown fields
    
    name: str = Field(..., min_length=1)      # Required, non-empty
    count: int = Field(default=0, ge=0)       # Optional, >= 0
    labels: list[str] = Field(default_factory=list)
    
    @field_validator("name")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        return v.strip()
```

---

## Token Efficiency Rules

### Batching & Parallelization

1. **Parallel LLM calls**: Use `asyncio` or thread pools for independent extractions (not yet implemented; TODO).
2. **Batch Neo4j queries**: Collect multiple Cypher statements; execute with batch create operations:
   ```python
   # ✅ Efficient: one Cypher with MERGE/CREATE multi-node
   cypher = """
   UNWIND $items as item
   MERGE (s:Entity {name: item.subject})
   MERGE (o:Entity {name: item.object})
   MERGE (s)-[r:PREDICATE]->(o)
   SET r += item.rel_props
   """
   manager.run_query(cypher, {"items": triples})
   
   # ❌ Inefficient: loop + individual queries
   for triple in triples:
       manager.run_query(f"CREATE (s:Entity {{name: '{triple.subject}'}})")
       # ...
   ```

3. **Checkpoint recovery**: KG pipeline saves stage outputs (JSON) to resume from failures:
   ```python
   # Loads stage N output if it exists; skips stages 0..(N-1)
   if paths["triples_resolved"].exists():
       triples = load_json(paths["triples_resolved"])
       skip_to_stage_5()
   ```

### Chunking Strategy

- **Document chunks**: config.yaml defines `small_max_tokens: 400`, `large_max_tokens: 1024` with overlap to maintain context continuity
- **Default in production**: `medium_window_tokens: 512, medium_overlap_tokens: 128`
- **Adjust if**: Extraction fails on tokens > 1200 (OOM); use smaller window

### Caching

- **Query result cache** (KGRAGAgent): LRUCache(128) for identical retrieve() calls during multi-hop reasoning
  - Cache hit if `(question, chosen_mode)` matches exactly
  - No staleness—cache is per-session only
- **Model cache** (LLMManager): Lazy-load once; reuse across queries
  - Use `enable_cache=False` in AgentConfig to disable

### Compression & Context Windows

- **ContextCompressor** enforces `max_content_tokens` (default 1000):
  ```python
  # If retrieved context exceeds budget, summarize/truncate per section
  merged_context = self.compressor.compress(
      nodes_text + "\n" + triples_text,
      budget=1000
  )
  ```
- **Token ratio estimator**: `token_estimator_ratio=0.75` (empirical ~75% of actual)

### Summarization Thresholds

- **LLM prompt budget exceeded**: Log warning + truncate tail sections (keep head evidence first)
- **Decomposition skipped**: If `enable_decomposition_step=False` (default post-perf-patch)
- **Routing skipped**: If `enable_adaptive_routing_step=False` (default post-perf-patch)

### Defaults for Performance Profiles

```python
# see run_retrieval_matrix.py
_PRODUCTION_FAST_LARGE_MODEL_MAX_NEW_TOKENS = 160  # vs default 256

# Use profile auto + large model → max_new_tokens: 160
if profile == "production_fast" and model_size_gb >= 30:
    max_new_tokens = 160
```

---

## Do NOT

### Anti-Patterns in This Codebase

1. **❌ Do NOT use bare `except:` or `except Exception:`**
   - Always catch specific exceptions (e.g., `ValueError`, `ServiceUnavailable`)
   - Bare catch hides bugs and breaks debugging
   - **Exception**: OK in experiment runners for graceful failure collection

2. **❌ Do NOT query Neo4j inside loops**
   - Use UNWIND + MERGE for batch operations (see Batching above)
   - Example: Don't loop `for triple in triples: create_node(triple)` —use one Cypher

3. **❌ Do NOT assume entity names are unique**
   - Resolution stage merges aliases → canonical names
   - Use `CanonicalEntityRecord` after stage 4; don't query raw stage 3 triples

4. **❌ Do NOT call LLM without retry logic**
   - LLMManager handles retries internally
   - If calling OpenAI/vLLM directly, wrap in try-except with exponential backoff

5. **❌ Do NOT hardcode model paths or API keys**
   - Use env vars (`NEO4J_URL`, `VLLM_MODEL_NAME`, `OPENAI_API_KEY`)
   - Load via `dotenv.load_dotenv()` or pass to constructors

6. **❌ Do NOT skip validation of LLM output**
   - Always call `validate_triples()` after JSON parsing
   - Pydantic validators will reject malformed predicates (not SCREAMING_SNAKE_CASE)

7. **❌ Do NOT mix synchronous + asynchronous code without careful threading**
   - LLMManager uses `threading.Lock` for model caching (already handled)
   - Don't add async code without coordinating cache access

8. **❌ Do NOT ignore checkpoint files**
   - Stage 3 (LLM extraction) saves `stage3_checkpoint.json` every N chunks
   - Resume from checkpoint: `load_checkpoint()` in main.py
   - Re-running stage 3 without clearing checkpoint will skip already-extracted chunks

9. **❌ Do NOT append to experiment results without timestamp/run_id**
   - Each run gets `run_id = str(uuid.uuid4())`
   - Results exported to `artifacts/experiments/{timestamp}_{tag}/results.jsonl`
   - Prevents accidental overwrites

10. **❌ Do NOT use string formatting for Cypher with user input**
    - Always use parameterized queries:
      ```python
      # ✅ Safe
      results = manager.run_query("MATCH (n:Entity {name: $name}) RETURN n", {"name": entity})
      
      # ❌ SQL injection risk
      results = manager.run_query(f"MATCH (n:Entity {{name: '{entity}'}}) RETURN n")
      ```

11. **❌ Do NOT rely on GLiNER confidence scores alone**
    - Threshold 0.45 is default (config.yaml)
    - Recommended: post-filter by entity length + context + second NER pass for low-confidence

12. **❌ Do NOT assume vLLM is always running**
    - Check health first: `smoke_check.py --check-llm`
    - Graceful fallback to local model if vLLM unavailable (not yet implemented; TODO)

13. **❌ Do NOT modify neo4j_manager's connection mid-query**
    - Thread-safe operations already handled by `_load_lock` in LLMManager
    - Connection retry (`_reconnect()`) is automatic on transient errors

14. **❌ Do NOT assume resolved entities are error-free**
    - Entity resolution is heuristic-based (embeddings + Jaccard similarity)
    - False merges can occur; manual review recommended for small KGs
    - Use `alias_sources` in CanonicalEntityRecord to trace merges

15. **❌ Do NOT ignore logging configuration**
    - Always call `logging.basicConfig()` or configure handlers in main entry points
    - Stage outputs should log progress milestones (not just debug traces)
    - See `kg_pipeline/main.py` for example setup

---

## Fast Reference

### Common Tasks

**Run a single retrieval query:**
```bash
cd graphRAGPipelineExp1
conda activate graphllm
python -m graphrag.cli --question "What is X?" --entity "Y"
```

**Build KG from documents:**
```bash
python -m kg_pipeline.main --input-dir ./documents --output-dir ./kg_pipeline/artifacts
```

**Run experiment matrix:**
```bash
python scripts/run_retrieval_matrix.py \
  --questions-file questions.txt \
  --models "7b,32b" \
  --strategies "default,text_plus_triples" \
  --output-dir artifacts/experiments
```

**Check system health:**
```bash
python scripts/smoke_check.py --check-neo4j --check-llm
```

**Analyze results:**
```bash
python scripts/analyze_experiments.py \
  --results-dir artifacts/experiments \
  --output-csv results_ranked.csv
```

### Key Files to Read First

1. **Understanding the full flow**: [README.md](../README.md)
2. **Data models**: [kg_pipeline/models/types.py](../kg_pipeline/models/types.py)
3. **KG extraction pipeline**: [kg_pipeline/main.py](../kg_pipeline/main.py)
4. **Retrieval & generation**: [src/graphrag/agent/core.py](../src/graphrag/agent/core.py)
5. **Configuration**: [kg_pipeline/config.yaml](../kg_pipeline/config.yaml)

### Environment Setup

```bash
# Create conda env (required)
conda create -n graphllm python=3.10 -y
conda activate graphllm

# Install for CPU
pip install -r requirements-cpu.txt
pip install -e .

# Install for GPU
pip install -r requirements-gpu.txt
pip install -e .

# Load env vars
export NEO4J_URL="neo4j+s://..."
export NEO4J_USERNAME="..."
export NEO4J_PASSWORD="..."
export VLLM_MODEL_NAME="Qwen/Qwen2.5-32B-Instruct"
export VLLM_BASE_URL="http://localhost:8000/v1"
```

---

## Known Issues & Workarounds

| Issue | Cause | Workaround |
|-------|-------|-----------|
| Exit code 126 on `graphrag-demo` | Stale shebang in `~/.local/bin/graphrag-demo` | Use `conda run -n graphllm python -m graphrag.cli` |
| Torch/torchvision version mismatch | Incompatible CUDA versions | Ensure `torch==2.5.1+cu124` + `torchvision==0.20.1+cu124` |
| Neo4j UnknownPropertyKey warnings | Using string concatenation in Cypher | Use `properties(node)['key']` accessor in Cypher |
| KG stage 3 crashes on malformed LLM output | JSON parse error or invalid triple schema | Exception caught; failed chunk logged to `failed_chunks.jsonl`; continues to next chunk |
| Entity resolution too aggressive | Similarity threshold too low (default 0.92) | Increase threshold in config.yaml `resolution.similarity_threshold: 0.95` |
| Long KG pipeline hangs on notebook disconnect | Terminal killed; async process orphaned | Use `sbatch scripts/run_kg_pipeline.sbatch` for detached execution |

---

**Generated**: May 2026
**Scope**: src/, kg_pipeline/, scripts/, evaluation/
**Version**: 0.1.0
