from __future__ import annotations

from typing import Any, Literal, TypedDict


class Triple(TypedDict):
    subject: str
    predicate: str
    object: str


class ProvenanceRecord(TypedDict, total=False):
    claim: str
    source_type: Literal["text_chunk", "kg_triple"]
    source_id: str
    content: str


class RAGState(TypedDict, total=False):
    question: str
    run_id: str
    sub_questions: list[str]
    rewritten_question: str
    rewrite_count: int
    text_context: str
    kg_triples: list[Triple]
    retrieved_neighbors_count: int
    retrieved_subgraph_count: int
    retrieved_shortest_path_count: int
    kg_context: str
    merged_context: str
    chosen_retrieval_mode: str
    hop_history: list[dict[str, Any]]
    current_hop: int
    relevance: Literal["relevant", "not_relevant"]
    confidence: float
    confidence_retries: int
    answer: str
    provenance: list[ProvenanceRecord]
    reflection_passed: bool
    reflection_feedback: str
    strategy: str
    latency_ms: float
    node_timings: dict[str, float]


class KGNode(TypedDict, total=False):
    node_id: str
    labels: list[str]
    properties: dict[str, Any]
    text: str


class KGTriple(TypedDict, total=False):
    subject_id: str
    subject: str
    predicate: str
    object_id: str
    object: str
    subject_labels: list[str]
    object_labels: list[str]
    subject_properties: dict[str, Any]
    object_properties: dict[str, Any]
    relationship_properties: dict[str, Any]
