from __future__ import annotations

import copy

from graphrag.config import AgentConfig

# Canonical retrieval-strategy presets shared by the CLI (`graphrag.cli`) and the
# experiment matrix driver (`scripts/run_retrieval_matrix.py`). This module is the
# single source of truth: keep STRATEGY_PRESETS and apply_strategy() in sync.
STRATEGY_PRESETS: tuple[str, ...] = (
    "default",
    "text_only",
    "no_retrieval",
    "text_plus_triples",
    "neighbors_focus",
    "subgraph_2hop",
    "shortest_path",
)


def apply_strategy(base: AgentConfig, label: str) -> AgentConfig:
    """Return a deep copy of ``base`` configured for the named retrieval strategy.

    Args:
        base: Baseline agent configuration to clone.
        label: One of :data:`STRATEGY_PRESETS`.

    Returns:
        A new ``AgentConfig`` with the strategy's ``include_*`` flags applied.

    Raises:
        ValueError: If ``label`` is not a known strategy.
    """
    config = copy.deepcopy(base)

    if label == "default":
        return config

    if label == "text_only":
        config.include_nodes = False
        config.include_triples = False
        config.include_neighbors = False
        config.include_subgraph = False
        config.include_shortest_path = False
        config.use_text_retriever = True
        return config

    if label == "no_retrieval":
        config.include_nodes = False
        config.include_triples = False
        config.include_neighbors = False
        config.include_subgraph = False
        config.include_shortest_path = False
        config.use_text_retriever = False
        return config

    if label == "text_plus_triples":
        config.include_neighbors = False
        config.include_subgraph = False
        config.include_shortest_path = False
        return config

    if label == "neighbors_focus":
        config.include_nodes = False
        config.include_subgraph = False
        config.include_shortest_path = False
        return config

    if label == "subgraph_2hop":
        config.hops = max(2, int(config.hops))
        config.include_nodes = False
        config.include_neighbors = False
        config.include_shortest_path = False
        return config

    if label == "shortest_path":
        config.include_nodes = False
        config.include_neighbors = False
        config.include_subgraph = False
        return config

    allowed = ",".join(STRATEGY_PRESETS)
    raise ValueError(f"Unknown strategy '{label}'. Allowed: {allowed}")
