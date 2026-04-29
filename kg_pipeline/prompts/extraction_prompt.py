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
        }
    ]

    prompt = f"""
You are an information extraction system for FAO food-domain documents.

Task:
1) Validate and correct candidate entities from GLiNER.
2) Extract all semantic and quantitative relationships from the chunk.
3) Add entities that GLiNER missed when necessary.
4) Return only a JSON array of KGTriple dictionaries.

Strict rules:
- Allowed labels: {json.dumps(allowed_labels)}
- You may introduce a new label only if none of the allowed labels fits.
- Do not use meta labels like Email, PageRange, SectionTitle, Chunk, Grant, Identifier.
- Document label is only for whole documents (filename or full document title), not section headings.
- Predicates must be SCREAMING_SNAKE_CASE, short (1-3 words), and reusable across documents.
- Avoid generic predicates like RELATED_TO and do not emit MENTIONED_IN (added later by the system).
- Output must be pure JSON array only. No prose.
- Include numeric relationship attributes in relationship_properties when present
  (for example value, unit, year).
- relationship_properties must always include source_doc and extraction_method.
- extraction_method must be "llm".

Chunk metadata:
- filename: {chunk.filename}
- section_title: {chunk.section_title}
- chunk_id: {chunk.chunk_id}
- page_range: {chunk.page_range}

Few-shot example:
{json.dumps(few_shot, ensure_ascii=False, indent=2)}

Candidate entities from GLiNER:
{json.dumps(candidate_entities, ensure_ascii=False, indent=2)}

Chunk markdown:
{chunk.text}
"""
    return prompt.strip()
