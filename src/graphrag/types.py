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
    # Retrieved evidence lists. Declared as state channels so LangGraph
    # propagates them from the retrieve node to the final state (undeclared keys
    # are dropped); the experiment runner serialises these for provenance/answer
    # analysis, and the *_count fields below mirror their lengths.
    retrieved_nodes: list[dict[str, Any]]
    retrieved_subgraph: list[dict[str, Any]]
    retrieved_shortest_path: list[dict[str, Any]]
    retrieved_text_sources: list[dict[str, Any]]
    retrieved_nodes_count: int
    retrieved_neighbors_count: int
    retrieved_subgraph_count: int
    retrieved_shortest_path_count: int
    kg_context: str
    merged_context: str
    chosen_retrieval_mode: str
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
