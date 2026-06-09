from __future__ import annotations

import json

from kg_pipeline.models.types import ChunkRecord


_DEFAULT_RELATION_VOCAB = [
    "WORKED_WITH",
    "HAS_COMPONENT",
    "CONTRIBUTES_TO",
    "COMPLIES_WITH",
    "REQUIRES",
    "ESTABLISHES",
    "DEFINED_AS",
    "APPLIES_TO",
    "AIMS_TO_ACHIEVE",
    "INCLUDES",
    "ENSURES",
    "IS_TYPE_OF",
    "HAS_MAXIMUM_LEVEL",
    "BASED_ON",
    "PRODUCES",
    "AFFECTS",
    "CAUSES",
    "MEASURES",
    "HAS_VALUE",
    "LOCATED_IN",
    "NEEDED_FOR",
    "USES",
    "PUBLISHED",
    "CONTAINS_DATA",
    "REGULATED_BY",
    "HAS_MEMBER",
    "ANALYZES",
    "IMPACTS",
    "GOVERNED_BY",
    "RELATED_TO",
    "GOVERNS",
    "EXCHANGES_INFO_WITH",
    "DECREASED_FROM",
    "INCREASED_FROM",
]


def build_extraction_prompt(
    chunk: ChunkRecord,
    candidate_entities: list[dict],
    allowed_labels: list[str],
    relation_vocab: list[str] | None = None,
) -> str:
    canonical_vocab = relation_vocab or _DEFAULT_RELATION_VOCAB
    canonical_vocab = [
        str(item).strip().upper() for item in canonical_vocab if str(item).strip()
    ]
    few_shot = [
        {
            "subject": "Europe",
            "predicate": "HAS_VALUE",
            "object": "2.7 C temperature anomaly",
            "subject_labels": ["Region"],
            "object_labels": ["DataValue"],
            "subject_properties": {"name": "Europe"},
            "object_properties": {"name": "2.7 C temperature anomaly", "value": 2.7, "unit": "C"},
            "relationship_properties": {
                "source_doc": "example_report.pdf",
                "extraction_method": "llm",
                "value": 2.7,
                "unit": "C",
                "year": 2025,
                "confidence": 0.95,
            },
        },
        {
            "subject": "wheat",
            "predicate": "LOCATED_IN",
            "object": "France",
            "subject_labels": ["Commodity"],
            "object_labels": ["Region"],
            "subject_properties": {"name": "wheat"},
            "object_properties": {"name": "France"},
            "relationship_properties": {
                "source_doc": "example_report.pdf",
                "extraction_method": "llm",
                "confidence": 0.9,
            },
        },
        # Temporal comparison: text says "GHG fell from 5.6 Gt in 1990 to 4.5 Gt in 2017"
        # → emit two HAS_VALUE triples + one DECREASED_FROM comparison triple
        {
            "subject": "EU GHG emissions 1990",
            "predicate": "HAS_VALUE",
            "object": "5.6 Gt CO2eq",
            "subject_labels": ["Indicator"],
            "object_labels": ["DataValue"],
            "subject_properties": {"name": "EU GHG emissions 1990"},
            "object_properties": {"name": "5.6 Gt CO2eq", "value": 5.6, "unit": "Gt CO2eq", "year": 1990},
            "relationship_properties": {
                "source_doc": "example_report.pdf",
                "extraction_method": "llm",
                "value": 5.6,
                "unit": "Gt CO2eq",
                "year": 1990,
                "confidence": 0.9,
            },
        },
        {
            "subject": "EU GHG emissions 2017",
            "predicate": "HAS_VALUE",
            "object": "4.5 Gt CO2eq",
            "subject_labels": ["Indicator"],
            "object_labels": ["DataValue"],
            "subject_properties": {"name": "EU GHG emissions 2017"},
            "object_properties": {"name": "4.5 Gt CO2eq", "value": 4.5, "unit": "Gt CO2eq", "year": 2017},
            "relationship_properties": {
                "source_doc": "example_report.pdf",
                "extraction_method": "llm",
                "value": 4.5,
                "unit": "Gt CO2eq",
                "year": 2017,
                "confidence": 0.9,
            },
        },
        {
            "subject": "EU GHG emissions 2017",
            "predicate": "DECREASED_FROM",
            "object": "EU GHG emissions 1990",
            "subject_labels": ["Indicator"],
            "object_labels": ["Indicator"],
            "subject_properties": {"name": "EU GHG emissions 2017"},
            "object_properties": {"name": "EU GHG emissions 1990"},
            "relationship_properties": {
                "source_doc": "example_report.pdf",
                "extraction_method": "llm",
                "absolute_change": -1.1,
                "unit": "Gt CO2eq",
                "confidence": 0.9,
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
- Canonical relation vocabulary (use ONLY these predicate types):
{json.dumps(canonical_vocab, ensure_ascii=False, indent=2)}
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

Predicate guidance:
- Use HAS_MAXIMUM_LEVEL for max regulatory limits. Put the contaminant in relationship_properties.contaminant.
- Use PUBLISHED for all publish variants (PUBLISHED_REPORT, PUBLISHER_OF, etc.).
- Use ENSURES for all ENSURES_HIGH_LEVEL_OF_* variants.
- Use RELATED_TO only if no other canonical predicate fits.

Temporal and quantitative facts — CRITICAL:
- Use HAS_VALUE ONLY for numeric, measurable quantities: numbers, percentages, counts, physical
  units (kg, Mt, Gt, %, °C, million tonnes, etc.). Do NOT use HAS_VALUE for qualitative
  statements, descriptions, goals, or trends without a concrete number.
  BAD: (pesticides, HAS_VALUE, dependency needs to be reduced)
  GOOD: (pesticide use 2030, HAS_VALUE, 50% reduction)
- When the text states a NUMERIC value for a specific year or time period, create a DISTINCT
  subject node that includes the year: e.g. subject="EU GHG emissions 2017",
  object_labels=["DataValue"], subject_labels=["Indicator"], predicate="HAS_VALUE".
- Always set object_properties.year, object_properties.value (numeric), object_properties.unit
  on DataValue nodes. Skip HAS_VALUE if you cannot populate a numeric value field.
- Also set relationship_properties.year, relationship_properties.value, relationship_properties.unit.
- For temporal comparisons ("decreased from X in 1990 to Y in 2017"), emit TWO HAS_VALUE triples
  (one per year) AND one comparison triple using DECREASED_FROM or INCREASED_FROM:
    ("GHG emissions 2017", "DECREASED_FROM", "GHG emissions 1990") with
    relationship_properties.percent_change and relationship_properties.absolute_change when available.
- For counts and statistics ("227 million households"), always capture value and unit as both
  object text and object_properties fields.

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
    "subject_labels": ["..."],
    "predicate": "...",
    "object": "...",
    "object_labels": ["..."],
    "properties": {{}},
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
