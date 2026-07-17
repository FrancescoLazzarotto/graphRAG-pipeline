from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evalkit.normalisation import match_key

# Mapping status values from gold_entity_eval_protocol.md §4.
MAPPING_EXACT = "exact"
MAPPING_LOCAL = "benchmark_local_extension"


@dataclass(frozen=True)
class GoldEntity:
    """One expected entity of a gold query (protocol §4).

    Attributes:
        label: Human-readable label as written in the gold.
        normalised_label: Canonical form the shared resolver targets.
        alt_labels: Accepted concept-level variants (synonyms, plurals, IT forms).
        uri: Canonical vocabulary IRI, or a ``urn:ceff:`` local identifier, or None.
        mapping_status: ``exact`` (counts at grounding-level) or
            ``benchmark_local_extension`` (concept-level only).
        vocabulary: Source vocabulary name, informational.
        aligned_to: Broader CEON/AGROVOC class a local concept is aligned to.
            A proposed alignment, never asserted equivalence — never scored.
    """

    label: str
    normalised_label: str
    alt_labels: tuple[str, ...]
    uri: str | None
    mapping_status: str
    vocabulary: str = ""
    aligned_to: str | None = None

    @property
    def counts_at_grounding_level(self) -> bool:
        """True when the entity has a real vocabulary URI to be anchored to (§2b)."""
        return self.mapping_status == MAPPING_EXACT

    @property
    def surface_forms(self) -> frozenset[str]:
        """Every form accepted as a concept-level hit, as comparison keys (§2a)."""
        forms = [self.normalised_label, *self.alt_labels]
        return frozenset(match_key(f) for f in forms if match_key(f))


@dataclass(frozen=True)
class GoldQuery:
    """One gold query with its expected answer and entities."""

    query_id: str
    query_type: str
    query: str
    expected_answer: str
    expected_entities: tuple[GoldEntity, ...]
    expected_relations: tuple[str, ...] = ()
    distractor_expected: bool = False
    source_verified: tuple[dict[str, Any], ...] = ()

    @property
    def grounding_entities(self) -> tuple[GoldEntity, ...]:
        """Entities in scope for grounding-level scoring (§2b: exact only)."""
        return tuple(e for e in self.expected_entities if e.counts_at_grounding_level)


@dataclass
class EvalRow:
    """One (run, strategy, question) tuple with all evaluation data."""

    run_dir: str
    strategy: str
    framework: str
    model_id: str
    run_index: str
    question_id: str
    question_type: str
    difficulty: str
    notes: str
    question: str
    answer: str
    ground_truth: str
    answer_variants: list[str]
    contexts: list[str]
    retrieved_triples: list[dict[str, Any]]
    retrieved_entities: list[Any]
    expected_entities: list[Any]
    gold_triples: list[dict[str, Any]]
    latency_ms: float
    kg_triples_used: int
    kg_neighbors_used: int
    kg_subgraph_triples_used: int
    kg_shortest_path_triples_used: int
    sub_questions: int
    insufficient: bool
    skip_reason: str
    gold_query: GoldQuery | None = None
    pipeline: str = ""

    @property
    def has_gold(self) -> bool:
        return bool(self.ground_truth)

    @property
    def is_skipped(self) -> bool:
        return bool(self.skip_reason)

    @property
    def is_distractor(self) -> bool:
        """True when correct behaviour is abstention (protocol §4, gold `scoring`)."""
        return bool(self.gold_query and self.gold_query.distractor_expected)


@dataclass
class MetricResult:
    """Uniform output of a single metric over a dataset."""

    name: str
    per_row: list[float | None] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def value(self) -> float | None:
        values = [v for v in self.per_row if v is not None]
        if not values:
            return None
        return sum(values) / len(values)


@dataclass
class GroupSummary:
    """Aggregated metric stats for one (model_id, framework, strategy, segment)."""

    keys: dict[str, str]
    n_rows: int
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class KGQualityResult:
    """Structural quality metrics of a knowledge graph."""

    n_entities: int = 0
    n_triples: int = 0
    n_predicates: int = 0
    n_documents: int = 0
    density: float = 0.0
    avg_degree: float = 0.0
    median_degree: float = 0.0
    n_components: int = 0
    isolated_ratio: float = 0.0
    predicate_entropy: float = 0.0
    failed_chunks: int = 0
    failed_chunks_ratio: float = 0.0
    resolution_collapse_ratio: float = 0.0
    entity_gold_coverage: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegressionResult:
    """Comparison of current metric vs baseline."""

    metric: str
    baseline: float
    current: float
    delta: float
    status: str  # "improved" | "stable" | "regressed"


@dataclass
class ReportModel:
    """Single input to all report renderers."""

    scope: str  # "experiment" | "project"
    runs: list[str]
    groups: list[GroupSummary]
    kg: KGQualityResult | None = None
    regression: list[RegressionResult] | None = None
    meta: dict[str, Any] = field(default_factory=dict)
