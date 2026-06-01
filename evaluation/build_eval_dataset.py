from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger("graphrag")

REQUIRED_OUTPUT_COLUMNS = [
    "run_dir",
    "strategy",
    "framework",
    "model_id",
    "question",
    "answer",
    "ground_truth",
    "contexts_json",
    "retrieved_triples_json",
    "retrieved_entities_json",
    "expected_entities_json",
    "gold_triples_json",
    "has_gold_match",
    "skip_reason",
]


def _remove_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_question(text: str) -> str:
    """Normalize question text used as join key.

    Args:
        text: Raw question text.

    Returns:
        Lowercased question with collapsed whitespace, removed trailing
        punctuation, and optional accent removal. Accent removal can be
        disabled with ``EVAL_NORMALIZE_REMOVE_ACCENTS=0``.
    """
    normalized = text.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^\w\s]+$", "", normalized)
    remove_accents = os.getenv("EVAL_NORMALIZE_REMOVE_ACCENTS", "1").strip().lower() not in {"0", "false", "no"}
    if remove_accents:
        normalized = _remove_accents(normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _parse_json(raw: str) -> tuple[Any, bool]:
    if not raw or not raw.strip():
        return (None, False)
    try:
        return (json.loads(raw), False)
    except json.JSONDecodeError:
        return (None, True)


def _parse_json_list(raw: str) -> tuple[list[Any], bool]:
    parsed, malformed = _parse_json(raw)
    if isinstance(parsed, list):
        return (parsed, malformed)
    return ([], malformed)


def _parse_metadata(raw: str) -> tuple[dict[str, Any], bool]:
    parsed, malformed = _parse_json(raw)
    if isinstance(parsed, dict):
        return (parsed, malformed)
    return ({}, malformed)


def _parse_gold_list(raw: str) -> tuple[list[Any], bool]:
    parsed, malformed = _parse_json_list(raw)
    if parsed:
        return (parsed, malformed)

    if raw and raw.strip() and not malformed:
        fallback = [item.strip() for item in raw.split("|") if item.strip()]
        if fallback:
            return (fallback, False)

    return ([], malformed)


def _iter_results_csv(input_path: Path, tag_contains: str) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() == ".csv":
            return [input_path]
        raise FileNotFoundError(f"Input file is not a CSV: {input_path}")

    if input_path.is_dir() and (input_path / "results.csv").exists():
        return [input_path / "results.csv"]

    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    files: list[Path] = []
    for candidate in sorted(input_path.rglob("*")):
        if not candidate.is_file() or candidate.name != "results.csv":
            continue
        run_dir_name = candidate.parent.name
        if tag_contains and tag_contains not in run_dir_name:
            continue
        files.append(candidate)

    if not files:
        raise FileNotFoundError(f"No results.csv files found under: {input_path}")

    return files


def _extract_list_json(
    row: dict[str, str],
    metadata: dict[str, Any],
    row_key_candidates: list[str],
    metadata_key_candidates: list[str],
) -> tuple[list[Any], bool]:
    malformed = False

    for key in row_key_candidates:
        value, parse_malformed = _parse_json_list(row.get(key, ""))
        malformed = malformed or parse_malformed
        if value:
            return value, malformed

    for key in metadata_key_candidates:
        value = metadata.get(key)
        if isinstance(value, list):
            return value, malformed
        if isinstance(value, str):
            parsed, parse_malformed = _parse_json_list(value)
            malformed = malformed or parse_malformed
            if parsed:
                return parsed, malformed

    return [], malformed


def _load_gold(gold_path: Path) -> tuple[dict[str, dict[str, str]], int]:
    if not gold_path.exists() or not gold_path.is_file():
        raise FileNotFoundError(f"Gold file not found: {gold_path}")

    gold_by_question: dict[str, dict[str, str]] = {}
    duplicate_keys = 0

    with gold_path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            question = (row.get("question", "") or "").strip()
            if not question:
                continue

            key = normalize_question(question)
            if key in gold_by_question:
                duplicate_keys += 1
                continue
            gold_by_question[key] = row

    return gold_by_question, duplicate_keys


def _pick_ground_truth(gold_row: dict[str, str]) -> str:
    for key in ("ground_truth", "canonical_answer", "gold_answer", "answer"):
        value = (gold_row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _extract_gold_annotations(gold_row: dict[str, str]) -> tuple[list[Any], list[Any], list[Any], bool]:
    expected_raw = str(gold_row.get("expected_entities_json") or gold_row.get("expected_entities") or "")
    triples_raw = str(gold_row.get("gold_triples_json") or gold_row.get("gold_triples") or "")
    variants_raw = str(gold_row.get("answer_variants_json") or gold_row.get("answer_variants") or "")

    expected_entities, expected_malformed = _parse_gold_list(expected_raw)
    gold_triples, triples_malformed = _parse_gold_list(triples_raw)
    answer_variants, _ = _parse_gold_list(variants_raw)
    return expected_entities, gold_triples, answer_variants, (expected_malformed or triples_malformed)


def _determine_skip_reason(question_key: str, has_gold: bool, malformed_json: bool, contexts: list[Any]) -> str:
    if not question_key:
        return "question_not_found"
    if malformed_json:
        return "malformed_json"
    if not contexts:
        return "empty_context"
    if not has_gold:
        return "no_gold"
    return ""


def _validate_output_schema(rows: list[dict[str, str]], min_coverage: float = 0.95) -> None:
    """Validate required columns and non-empty coverage.

    Args:
        rows: Output rows.
        min_coverage: Minimum non-empty ratio required per required column.
    """
    if not rows:
        logger.warning("Schema validation skipped because output has no rows.")
        return

    total = len(rows)
    warning_lines: list[str] = []
    for column in REQUIRED_OUTPUT_COLUMNS:
        present = sum(1 for row in rows if column in row)
        non_null = sum(1 for row in rows if row.get(column, None) is not None)
        coverage = non_null / total
        if present < total or coverage < min_coverage:
            warning_lines.append(
                f"- {column}: present={present}/{total} non_null={non_null}/{total} coverage={coverage:.3f}"
            )

    if warning_lines:
        logger.warning(
            "Output schema coverage below threshold %.0f%% for one or more columns:\n%s",
            min_coverage * 100,
            "\n".join(warning_lines),
        )


def _default_smoke_input_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "smoke_results.csv"


def _default_smoke_gold_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "smoke_gold.csv"


def _default_smoke_output_path() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "artifacts" / "evaluation" / "smoke_eval_dataset.csv"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Join experiment results with gold labels for evaluation")
    parser.add_argument("--input", default="", help="Path to experiments root, run folder, or results.csv")
    parser.add_argument("--gold-file", default="", help="CSV with at least question and ground_truth")
    parser.add_argument("--tag-contains", default="", help="Optional run-folder filter when input is experiments root")
    parser.add_argument("--output", default="", help="Output CSV path for joined evaluation dataset")
    parser.add_argument("--smoke", action="store_true", help="Run with local fixture data and default output path")
    parser.add_argument("--smoke-size", type=int, default=5, help="Maximum number of rows kept in smoke mode")
    parser.add_argument(
        "--min-schema-coverage",
        type=float,
        default=0.95,
        help="Minimum non-empty ratio required per required column",
    )
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s - %(message)s")
    args = _build_parser().parse_args()

    if args.min_schema_coverage <= 0 or args.min_schema_coverage > 1:
        raise SystemExit("--min-schema-coverage must be in (0, 1]")

    if args.smoke:
        input_path = Path(args.input) if args.input else _default_smoke_input_path()
        gold_path = Path(args.gold_file) if args.gold_file else _default_smoke_gold_path()
        output_path = Path(args.output) if args.output else _default_smoke_output_path()
    else:
        if not args.input or not args.gold_file or not args.output:
            raise SystemExit("--input, --gold-file and --output are required unless --smoke is used")
        input_path = Path(args.input)
        gold_path = Path(args.gold_file)
        output_path = Path(args.output)

    results_files = _iter_results_csv(input_path, tag_contains=args.tag_contains.strip())
    gold_map, duplicates = _load_gold(gold_path)

    out_rows: list[dict[str, str]] = []
    total_rows = 0
    matched_rows = 0
    skip_reasons = Counter()

    for csv_path in results_files:
        run_dir_name = csv_path.parent.name
        with csv_path.open("r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            for row in reader:
                total_rows += 1

                question = row.get("question", "") or ""
                question_key = normalize_question(question)
                gold_row = gold_map.get(question_key, {})
                has_gold = bool(gold_row)
                if has_gold:
                    matched_rows += 1

                metadata, malformed_metadata = _parse_metadata(row.get("metadata_json", ""))

                contexts, malformed_contexts = _extract_list_json(
                    row=row,
                    metadata=metadata,
                    row_key_candidates=["contexts_json", "retrieved_contexts_json"],
                    metadata_key_candidates=["contexts", "retrieved_contexts", "retrieval_contexts"],
                )
                retrieved_triples, malformed_retrieved_triples = _extract_list_json(
                    row=row,
                    metadata=metadata,
                    row_key_candidates=["retrieved_triples_json", "kg_triples_json"],
                    metadata_key_candidates=["retrieved_triples", "kg_triples"],
                )
                retrieved_entities, malformed_retrieved_entities = _extract_list_json(
                    row=row,
                    metadata=metadata,
                    row_key_candidates=["retrieved_entities_json"],
                    metadata_key_candidates=["retrieved_entities", "kg_entities"],
                )

                expected_entities: list[Any] = []
                gold_triples: list[Any] = []
                answer_variants: list[Any] = []
                malformed_gold = False
                if has_gold:
                    expected_entities, gold_triples, answer_variants, malformed_gold = _extract_gold_annotations(gold_row)

                malformed_json = (
                    malformed_metadata
                    or malformed_contexts
                    or malformed_retrieved_triples
                    or malformed_retrieved_entities
                    or malformed_gold
                )
                skip_reason = _determine_skip_reason(
                    question_key=question_key,
                    has_gold=has_gold,
                    malformed_json=malformed_json,
                    contexts=contexts,
                )
                if skip_reason:
                    skip_reasons[skip_reason] += 1

                out_rows.append(
                    {
                        "run_dir": run_dir_name,
                        "strategy": row.get("strategy", "") or "",
                        "framework": str(metadata.get("framework") or row.get("framework") or "unknown"),
                        "model_id": str(metadata.get("model_id") or row.get("model_id") or "unknown"),
                        "run_index": str(metadata.get("run_index") or row.get("run_index") or "0"),
                        "question_id": (gold_row.get("question_id", "") if has_gold else "") or "",
                        "question_type": (gold_row.get("question_type", "") if has_gold else "") or "",
                        "difficulty": (gold_row.get("difficulty", "") if has_gold else "") or "",
                        "notes": (gold_row.get("notes", "") if has_gold else "") or "",
                        "question": question,
                        "answer": row.get("answer", "") or "",
                        "ground_truth": _pick_ground_truth(gold_row) if has_gold else "",
                        "answer_variants_json": json.dumps(answer_variants, ensure_ascii=False),
                        "latency_ms": row.get("latency_ms", "") or "",
                        "kg_triples_used": row.get("kg_triples_used", "") or "",
                        "kg_neighbors_used": row.get("kg_neighbors_used", "") or "",
                        "kg_subgraph_triples_used": row.get("kg_subgraph_triples_used", "") or "",
                        "kg_shortest_path_triples_used": row.get("kg_shortest_path_triples_used", "") or "",
                        "sub_questions": row.get("sub_questions", "") or "",
                        "contexts_json": json.dumps(contexts, ensure_ascii=False),
                        "retrieved_triples_json": json.dumps(retrieved_triples, ensure_ascii=False),
                        "retrieved_entities_json": json.dumps(retrieved_entities, ensure_ascii=False),
                        "expected_entities_json": json.dumps(expected_entities, ensure_ascii=False),
                        "gold_triples_json": json.dumps(gold_triples, ensure_ascii=False),
                        "has_gold_match": "1" if has_gold else "0",
                        "skip_reason": skip_reason,
                    }
                )

    if args.smoke and args.smoke_size > 0:
        out_rows = out_rows[: args.smoke_size]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_dir",
        "strategy",
        "framework",
        "model_id",
        "run_index",
        "question_id",
        "question_type",
        "difficulty",
        "notes",
        "question",
        "answer",
        "ground_truth",
        "answer_variants_json",
        "latency_ms",
        "kg_triples_used",
        "kg_neighbors_used",
        "kg_subgraph_triples_used",
        "kg_shortest_path_triples_used",
        "sub_questions",
        "contexts_json",
        "retrieved_triples_json",
        "retrieved_entities_json",
        "expected_entities_json",
        "gold_triples_json",
        "has_gold_match",
        "skip_reason",
    ]

    _validate_output_schema(rows=out_rows, min_coverage=args.min_schema_coverage)

    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for item in out_rows:
            writer.writerow(item)

    logger.info("results_files=%d", len(results_files))
    logger.info("rows_total=%d", total_rows)
    logger.info("rows_output=%d", len(out_rows))
    logger.info("rows_with_gold=%d", matched_rows)
    logger.info("rows_without_gold=%d", max(total_rows - matched_rows, 0))
    logger.info("gold_unique_questions=%d", len(gold_map))
    logger.info("gold_duplicate_questions_ignored=%d", duplicates)
    logger.info("skip_reasons=%s", dict(skip_reasons))
    logger.info("saved=%s", output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())