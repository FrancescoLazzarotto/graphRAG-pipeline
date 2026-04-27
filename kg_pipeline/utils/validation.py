from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kg_pipeline.models.types import KGTriple


def normalize_json_text(raw_text: str) -> str:
    cleaned = (raw_text or "").strip()
    fence = chr(96) * 3
    if cleaned.startswith(fence):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith(fence):
            lines = lines[1:]
        if lines and lines[-1].strip() == fence:
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def parse_json_array(raw_text: str) -> list[dict[str, Any]]:
    cleaned = normalize_json_text(raw_text)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, list):
        raise ValueError("LLM output is not a JSON array")
    return parsed


def validate_triples(raw_items: list[dict[str, Any]]) -> list[KGTriple]:
    triples: list[KGTriple] = []
    for item in raw_items:
        triple = KGTriple.model_validate(item)
        triples.append(triple)
    return triples


def write_failed_chunk(
    failed_path: Path,
    chunk_metadata: dict[str, Any],
    attempt: int,
    error: str,
    raw_response: str,
) -> None:
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "chunk_metadata": chunk_metadata,
        "attempt": attempt,
        "error": error,
        "raw_response": raw_response,
    }
    with failed_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
