"""The canonical relationship vocabulary, in one place.

Every pass that renames or reclassifies a relationship type has to agree on
what the allowed types are. They used to agree by copy: `neo4j_postprocess`
held this list and the three `kg_repair*` scripts held their own, marked
"must match neo4j_postprocess.py". They had already drifted — the repair
copies carried `HAS_DEFINITION`, a type the pipeline never produces and which
has zero instances in the graph, while `DEFINED_AS` is the canonical type that
actually holds the definition edges.

That kind of drift is invisible: a repair pass writes a type no other pass
considers canonical, and nothing reports it as unknown.

Order is preserved from the original list because it is serialised into the
LLM reclassification prompts, where changing the order changes the prompt.
"""

from __future__ import annotations

CANONICAL_RELATION_TYPES: list[str] = [
    "RELATED_TO",
    "AFFECTS",
    "IMPACTS",
    "INFLUENCES",
    "CAUSES",
    "CAUSED_BY",
    "CONTRIBUTES_TO",
    "LEADS_TO",
    "DRIVEN_BY",
    "DEPENDS_ON",
    "ASSOCIATED_WITH",
    "BASED_ON",
    "DERIVED_FROM",
    "PART_OF",
    "HAS_PART",
    "HAS_COMPONENT",
    "COMPOSED_OF",
    "INCLUDES",
    "CONTAINS_DATA",
    "IS_TYPE_OF",
    "DEFINED_AS",
    "HAS_MAXIMUM_LEVEL",
    "HAS_MINIMUM_LEVEL",
    "HAS_VALUE",
    "HAS_UNIT",
    "VALUE_OF",
    "MEASURES",
    "INDICATES",
    "APPLIES_TO",
    "TARGETS",
    "TARGET_OF",
    "REQUIRES",
    "REQUIRED_BY",
    "USES",
    "USED_BY",
    "USES_METHOD",
    "HAS_METHOD",
    "MANAGES",
    "MANAGED_BY",
    "REGULATES",
    "REGULATED_BY",
    "GOVERNS",
    "GOVERNED_BY",
    "COMPLIES_WITH",
    "SHOULD_BE_MANAGED_BY",
    "ENSURES",
    "AIMS_TO_ACHIEVE",
    "NEEDED_FOR",
    "PUBLISHED",
    "WORKED_WITH",
    "EXCHANGES_INFO_WITH",
    "TAKE_INTO_ACCOUNT",
    "PRODUCES",
    "LOCATED_IN",
    "OCCURS_IN",
    "BELONGS_TO",
    "HAS_MEMBER",
    "MEMBER_OF",
    "ANALYZES",
    "ESTABLISHES",
    "ESTABLISHED_BY",
]

CANONICAL_RELATION_SET: frozenset[str] = frozenset(CANONICAL_RELATION_TYPES)
