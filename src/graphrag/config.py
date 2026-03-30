from __future__ import annotations

import enum
import os
from dataclasses import dataclass


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


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
    enable_cache: bool = True
    cache_maxsize: int = 128
    max_content_tokens: int = 200
    token_estimator_ratio: float = 0.75
    tone: OUTPUT_TONE = OUTPUT_TONE.TECHNICAL
    complexity: OUTPUT_COMPLEXITY = OUTPUT_COMPLEXITY.MEDIUM
    target_audience: str = "domain_expert"
    use_structured_response: bool = False


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
