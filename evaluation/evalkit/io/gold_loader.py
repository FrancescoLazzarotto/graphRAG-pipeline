from __future__ import annotations

import csv
import json
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

logger = logging.getLogger("graphrag")


def _remove_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_question(text: str) -> str:
    """Normalize question text for join-key matching.

    Lowercases, collapses whitespace, removes trailing punctuation.
    Accent removal is on by default; disable with EVAL_NORMALIZE_REMOVE_ACCENTS=0.
    """
    normalized = text.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^\w\s]+$", "", normalized)
    remove_accents = (
        os.getenv("EVAL_NORMALIZE_REMOVE_ACCENTS", "1").strip().lower()
        not in {"0", "false", "no"}
    )
    if remove_accents:
        normalized = _remove_accents(normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _parse_gold_list(raw: str) -> tuple[list[Any], bool]:
    """Parse a gold annotation field that may be JSON or pipe-separated."""
    if not raw or not raw.strip():
        return ([], False)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return (parsed, False)
    except json.JSONDecodeError:
        pass

    # Pipe-separated fallback (no valid JSON)
    fallback = [item.strip() for item in raw.split("|") if item.strip()]
    if fallback:
        return (fallback, False)

    return ([], True)


def load_gold(gold_path: Path) -> dict[str, dict[str, str]]:
    """Load a gold CSV and return a dict keyed by normalised question text.

    Args:
        gold_path: Path to gold CSV with at least ``question`` and one of
            ``ground_truth``, ``canonical_answer``, or ``gold_answer`` columns.

    Returns:
        Mapping of normalised question → raw row dict.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    if not gold_path.exists() or not gold_path.is_file():
        raise FileNotFoundError(f"Gold file not found: {gold_path}")

    gold_by_question: dict[str, dict[str, str]] = {}
    duplicates = 0

    with gold_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            question = (row.get("question", "") or "").strip()
            if not question:
                continue
            key = normalize_question(question)
            if key in gold_by_question:
                duplicates += 1
                continue
            gold_by_question[key] = dict(row)

    if duplicates:
        logger.warning("Ignored %d duplicate questions in gold file: %s", duplicates, gold_path)

    return gold_by_question


def pick_ground_truth(gold_row: dict[str, str]) -> str:
    """Return the canonical answer from a gold row, trying multiple field names."""
    for key in ("ground_truth", "canonical_answer", "gold_answer", "answer"):
        value = (gold_row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def extract_gold_annotations(
    gold_row: dict[str, str],
) -> tuple[list[Any], list[Any], list[Any]]:
    """Parse expected_entities, gold_triples, and answer_variants from a gold row.

    Returns:
        (expected_entities, gold_triples, answer_variants)
    """
    expected_raw = str(
        gold_row.get("expected_entities_json") or gold_row.get("expected_entities") or ""
    )
    triples_raw = str(
        gold_row.get("gold_triples_json") or gold_row.get("gold_triples") or ""
    )
    variants_raw = str(
        gold_row.get("answer_variants_json") or gold_row.get("answer_variants") or ""
    )

    expected_entities, _ = _parse_gold_list(expected_raw)
    gold_triples, _ = _parse_gold_list(triples_raw)
    answer_variants, _ = _parse_gold_list(variants_raw)
    return expected_entities, gold_triples, answer_variants
