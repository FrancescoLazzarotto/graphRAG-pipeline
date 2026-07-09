"""Standard (text-only) RAG baseline presets shared by experiment drivers.

Single source of truth for the baseline retrieval configurations used by
``scripts/run_retrieval_matrix.py``; keep preset names stable because they end
up in experiment artifacts (results.jsonl ``strategy`` field).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StandardStrategyPreset:
    top_k: int
    chunk_size: int
    chunk_overlap: int
    min_chunk_chars: int = 80
    include_sources: bool = True
    backend: str = "tfidf"  # "tfidf" | "dense"
    embedding_model: str | None = None  # None = factory default


STANDARD_STRATEGY_PRESETS: dict[str, StandardStrategyPreset] = {
    # --- TF-IDF (lexical) baselines ---
    "std_topk3": StandardStrategyPreset(top_k=3, chunk_size=1200, chunk_overlap=180),
    "std_topk5": StandardStrategyPreset(top_k=5, chunk_size=1200, chunk_overlap=180),
    "std_wide_context": StandardStrategyPreset(
        top_k=6, chunk_size=1800, chunk_overlap=180
    ),
    "std_fine_chunks": StandardStrategyPreset(
        top_k=5, chunk_size=800, chunk_overlap=140
    ),
    # --- Dense (cosine similarity) variants ---
    "std_dense_topk3": StandardStrategyPreset(
        top_k=3, chunk_size=1200, chunk_overlap=180, backend="dense"
    ),
    "std_dense_topk5": StandardStrategyPreset(
        top_k=5, chunk_size=1200, chunk_overlap=180, backend="dense"
    ),
    "std_dense_wide": StandardStrategyPreset(
        top_k=6, chunk_size=1800, chunk_overlap=180, backend="dense"
    ),
}

STANDARD_STRATEGIES_DEFAULT = tuple(STANDARD_STRATEGY_PRESETS.keys())
STANDARD_STRATEGIES_SMOKE = ("std_topk3", "std_topk5")
