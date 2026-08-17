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
    # Declared so `neighbors_focus` runs are auditable at the evidence level: the
    # retrieve node has always returned this key, but without a declaration
    # LangGraph dropped it and only the count reached the artifacts (audit §1.7).
    retrieved_neighbors: list[dict[str, Any]]
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
    # Numbered, citable evidence for this turn (WP1). Serialised as plain dicts
    # so the state stays JSON-dumpable for the experiment runner; rebuild the
    # dataclasses with graphrag.agent.evidence.evidence_from_dicts.
    evidence_index: list[dict[str, Any]]
    # Reference ids that survived context compression, i.e. the blocks the model
    # was actually shown. The citation gate validates against these.
    visible_evidence_refs: list[str]
    citation_report: dict[str, Any]
    quote_report: dict[str, Any]
    answer: str
    # The answer before the refusal-rescue retry, and whether that retry fired.
    # Abstention must be measured on the pre-retry text (audit §1.5).
    pre_retry_answer: str
    refusal_retry_applied: bool
    # Domain gate. `in_domain` is False only when the gate ran and rejected the
    # question; `out_of_scope` marks the answer as the fixed refusal, so callers
    # can tell an abstention from a generated answer without parsing prose.
    # `follow_up` exempts a question that continues an already-admitted topic.
    in_domain: bool
    out_of_scope: bool
    follow_up: bool
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
