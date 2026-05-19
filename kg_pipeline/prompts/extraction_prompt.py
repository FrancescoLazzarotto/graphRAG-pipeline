from __future__ import annotations

import json

from kg_pipeline.models.types import ChunkRecord


def build_extraction_prompt(
    chunk: ChunkRecord,
    candidate_entities: list[dict],
    allowed_labels: list[str],
) -> str:
    few_shot = [
        {
            "subject": "Europe",
            "predicate": "HAS_TEMPERATURE_ANOMALY",
            "object": "2.7 C",
            "subject_labels": ["Region"],
            "object_labels": ["DataValue"],
            "subject_properties": {"name": "Europe"},
            "object_properties": {"name": "2.7 C", "value": 2.7, "unit": "C"},
            "relationship_properties": {
                "source_doc": "example_report.pdf",
                "extraction_method": "llm",
                "value": 2.7,
                "unit": "C",
                "year": 2025,
            },
        },
        {
            "subject": "wheat",
            "predicate": "PRODUCED_IN",
            "object": "France",
            "subject_labels": ["Commodity"],
            "object_labels": ["Region"],
            "subject_properties": {"name": "wheat"},
            "object_properties": {"name": "France"},
            "relationship_properties": {
                "source_doc": "example_report.pdf",
                "extraction_method": "llm",
            },
        },
    ]

    prompt = f"""
You are an information extraction system for FAO food-domain documents.

Task:
1) Validate and correct candidate entities from GLiNER.
2) Extract all semantic and quantitative relationships from the chunk.
3) Add entities that GLiNER missed when necessary.
4) Return only a JSON array of KGTriple dictionaries.

CRITICAL Validation rules (DO NOT VIOLATE):
- SUBJECT and OBJECT must NEVER be empty strings - always have a value (1+ chars).
- SUBJECT and OBJECT must be stripped of leading/trailing whitespace.
- PREDICATE must be ALL UPPERCASE letters, numbers, and underscores (e.g., HAS_VALUE, PRODUCED_IN).
- PREDICATE must start with a letter (not a number).
- PREDICATE must be 1-4 words in SCREAMING_SNAKE_CASE format (max 50 chars total).
- If a relationship cannot be fully extracted, SKIP IT - do not emit triples with empty fields.
- If you cannot find a valid OBJECT, SKIP the relationship - do not use empty string.

Additional guidelines:
- Allowed labels: {json.dumps(allowed_labels)}
- You may introduce a new label only if none of the allowed labels fits.
- Do not use meta labels like Email, PageRange, SectionTitle, Chunk, Grant, Identifier.
- Document label is only for whole documents (filename or full document title), not section headings.
- Avoid generic predicates like RELATED_TO and do not emit MENTIONED_IN (added later by the system).
- Output must be pure JSON array only. No prose.
- Include numeric relationship attributes in relationship_properties when present
    (for example value, unit, year).
- If possible, add relationship_properties.confidence as a float between 0 and 1.
- relationship_properties must always include source_doc and extraction_method.
- extraction_method must be "llm".

You MUST use ONLY the following predicate types. Never invent new predicates.
If the relationship does not fit any of these, either skip it or use the 
closest match.

ALLOWED PREDICATES (with usage guidance):
- GOVERNS            → a policy/regulation governs a commodity/concept
- ESTABLISHES        → an org or document establishes another org or policy
- ESTABLISHED_BY     → inverse of ESTABLISHES
- HAS_COMPONENT      → a concept/method is composed of sub-components
- BASED_ON           → a method or policy is based on a concept/principle
- AFFECTS            → an event or entity affects another entity
- CONTRIBUTES_TO     → an entity contributes to a goal or concept
- APPLIES_TO         → a rule/document applies to an entity
- DEFINED_AS         → canonical definition relationship
- INCLUDES           → a set includes members
- IS_TYPE_OF         → instance-of or subtype relationship
- HAS_MAXIMUM_LEVEL  → a commodity/indicator has a maximum regulatory level
                                             Use ONLY this form. Never use HAS_MAX_LEVEL, 
                                             HAS_MAXIMUM_LEVEL_AFLATOXIN_B1, or any variant.
                                             Instead, add a property "contaminant" to the triple.
- PUBLISHED          → an org published a document. Use for all publish 
                                             variants (PUBLISHED_REPORT, PUBLISHER_OF, etc.)
- WORKED_WITH        → collaboration between organizations
- EXCHANGES_INFO_WITH → information exchange between organizations
- TAKE_INTO_ACCOUNT  → a method considers a factor
- ENSURES            → an entity ensures a quality or outcome. 
                                             Use for ALL variants of ENSURES_HIGH_LEVEL_OF_*.
- SHOULD_BE_MANAGED_BY → governance/management responsibility
- AIMS_TO_ACHIEVE    → goal of a policy or process
- NEEDED_FOR         → dependency relationship
- CONTAINS_DATA      → a document contains a dataset
- COMPLIES_WITH      → regulatory compliance
- ANALYZES           → an org analyzes a concept

PROPERTY RULES — the following must NEVER become relationships. 
Encode them as properties of the subject node instead:
- Titles, full names, acronym expansions → add property "full_name" or "title" 
    to the subject node
- Trade role descriptors (e.g. "major global trader") → add property "role"
- Membership of a region → add property "region"

Output schema note: each KGTriple object may include an optional top-level
"properties" dictionary for additional key/value pairs extracted from the
sentence context. Example triple schema:
{{
    "subject": "...",
    "subject_label": "...",
    "predicate": "...",
    "object": "...",
    "object_label": "...",
    "properties": {{}},   // optional key-value pairs extracted from context
    "subject_properties": {{...}},
    "object_properties": {{...}},
    "relationship_properties": {{...}}
}}

Chunk metadata:
- filename: {chunk.filename}
- section_title: {chunk.section_title}
- chunk_id: {chunk.chunk_id}
- page_range: {chunk.page_range}

Few-shot examples (these are CORRECT format - follow them):
{json.dumps(few_shot, ensure_ascii=False, indent=2)}

Candidate entities from GLiNER:
{json.dumps(candidate_entities, ensure_ascii=False, indent=2)}

Chunk markdown:
{chunk.text}

Remember: Do not use empty strings for subject or object. If you cannot identify a complete relationship, skip it.
"""
    return prompt.strip()
