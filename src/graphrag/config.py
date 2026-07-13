from __future__ import annotations

import enum
import logging
import os
from dataclasses import dataclass


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

logger = logging.getLogger("graphrag")


class OUTPUT_TONE(enum.Enum):
    TECHNICAL = "technical"
    SIMPLIFIED = "simplified"
    FORMAL = "formal"


class OUTPUT_COMPLEXITY(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(slots=True)
class AgentConfig:
    query: str | None = None
    entity: str | None = None
    entity_a: str | None = None
    entity_b: str | None = None
    hops: int = 1
    max_depth: int = 6
    nodes_limit: int = 10
    triples_limit: int = 20
    neighbors_limit: int = 25
    subgraph_limit: int = 200
    labels: tuple[str, ...] = ()
    relationship_types: tuple[str, ...] = ()
    include_nodes: bool = True
    include_triples: bool = True
    include_neighbors: bool = True
    include_subgraph: bool = True
    include_shortest_path: bool = True
    answer_prompt: str = ""
    rewrite_prompt: str = ""
    kg_reasoning_prompt: str = ""
    decomposition_prompt: str = ""
    reflection_prompt: str = ""
    adaptive_router_prompt: str = ""
    llm_warmup: bool = False
    enable_decomposition_step: bool = False
    enable_adaptive_routing_step: bool = False
    enable_cache: bool = True
    cache_maxsize: int = 128
    recursion_limit: int = 50
    # 6000 fits comfortably in Qwen2.5-32B's 32k window with prompt + answer;
    # 1000 truncated most multi-channel retrievals (head/tail compression cut
    # the mid-context evidence and inflated "insufficient context" answers).
    max_content_tokens: int = 6000
    token_estimator_ratio: float = 0.25  # tokens-per-char (~4 chars/token)
    tone: OUTPUT_TONE = OUTPUT_TONE.TECHNICAL
    complexity: OUTPUT_COMPLEXITY = OUTPUT_COMPLEXITY.MEDIUM
    target_audience: str = "domain_expert"
    use_structured_response: bool = False
    rank_triples: bool = True
    # When decomposition produces multiple retrieval queries, merged results
    # preserve arrival order by default; enable to re-rank them globally by
    # score. Off by default to keep existing baselines unchanged.
    rerank_merged_results: bool = False
    # The answer prompt asks for a 'Limits and confidence' section only when
    # context is sparse; enable to request it on every answer (demo UX). Off
    # by default to keep existing baselines unchanged.
    always_include_limits: bool = False
    # Triples carry no per-edge confidence yet (see KG-side item B8), so the
    # confidence weight is 0.0 and lexical/mention absorb it. Keeping the field
    # lets a future confidence signal be re-enabled without code changes.
    ranker_weight_lexical: float = 0.70
    ranker_weight_mention: float = 0.30
    ranker_weight_confidence: float = 0.0
    ranker_system_link_penalty: float = 0.5
    adaptive_hops: bool = True
    min_subgraph_triples: int = 10
    max_hops: int = 4
    include_triple_metadata: bool = True
    use_text_retriever: bool = False
    text_retriever_top_k: int = 5
    text_retriever_backend: str = "tfidf"  # "tfidf" | "dense"
    dense_embedding_model: str = "intfloat/multilingual-e5-base"
    dense_query_prefix: str = "query: "
    dense_passage_prefix: str = "passage: "
    dense_normalize: bool = True
    dense_device: str = "auto"  # "auto" | "cpu" | "cuda"
    vector_index_dir: str = "artifacts/vector_index"

    def __post_init__(self) -> None:
        if self.rank_triples:
            weight_sum = (
                self.ranker_weight_lexical
                + self.ranker_weight_mention
                + self.ranker_weight_confidence
            )
            if abs(weight_sum - 1.0) > 0.01:
                logger.warning(
                    "Ranker weights sum to %.3f instead of 1.0 "
                    "(lexical=%.2f mention=%.2f confidence=%.2f): triple scores "
                    "will not be comparable across configurations",
                    weight_sum,
                    self.ranker_weight_lexical,
                    self.ranker_weight_mention,
                    self.ranker_weight_confidence,
                )


@dataclass(slots=True)
class KGConfig:
    url: str
    username: str
    password: str
    database: str | None = None
    node_name_properties: tuple[str, ...] = (
        "name",
        "title",
        "label",
        "id",
        "uuid",
        "entity",
    )
    default_limit: int = 50


def build_kg_config_from_env(
    url_env: str = "NEO4J_URL",
    username_env: str = "NEO4J_USERNAME",
    password_env: str = "NEO4J_PASSWORD",
    database_env: str = "NEO4J_DATABASE",
) -> KGConfig:
    url = os.getenv(url_env)
    username = os.getenv(username_env)
    password = os.getenv(password_env)
    database = os.getenv(database_env)

    missing = [
        key
        for key, value in (
            (url_env, url),
            (username_env, username),
            (password_env, password),
        )
        if not value
    ]
    if missing:
        missing_csv = ", ".join(missing)
        raise ValueError(f"Missing required environment variables: {missing_csv}")

    return KGConfig(
        url=url,
        username=username,
        password=password,
        database=database,
    )
