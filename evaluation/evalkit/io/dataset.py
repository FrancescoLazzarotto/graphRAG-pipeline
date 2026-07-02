from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from evalkit.io.gold_loader import (
    extract_gold_annotations,
    load_gold,
    normalize_question,
    pick_ground_truth,
)
from evalkit.io.run_loader import _is_insufficient, load_raw_rows, raw_row_to_partial
from evalkit.models import EvalRow

logger = logging.getLogger("graphrag")

REQUIRED_EVAL_COLUMNS = [
    "run_dir", "strategy", "framework", "model_id",
    "question", "answer", "ground_truth",
    "contexts_json", "retrieved_triples_json", "retrieved_entities_json",
    "expected_entities_json", "gold_triples_json",
    "has_gold_match", "skip_reason",
]


def _determine_skip_reason(
    question_key: str,
    has_gold: bool,
    malformed_json: bool,
    contexts: list[Any],
) -> str:
    if not question_key:
        return "question_not_found"
    if malformed_json:
        return "malformed_json"
    if not contexts:
        return "empty_context"
    if not has_gold:
        return "no_gold"
    return ""


def _validate_schema(rows: list[EvalRow], min_coverage: float = 0.95) -> None:
    if not rows:
        return
    total = len(rows)
    issues: list[str] = []
    for col in ("question", "answer", "ground_truth"):
        non_empty = sum(1 for r in rows if getattr(r, col, ""))
        if non_empty / total < min_coverage:
            issues.append(f"  {col}: {non_empty}/{total} non-empty")
    if issues:
        logger.warning(
            "Schema coverage below %.0f%% for:\n%s", min_coverage * 100, "\n".join(issues)
        )


def build_dataset(
    run_dirs: list[Path],
    gold_path: Path | None = None,
    tag_contains: str = "",
    min_schema_coverage: float = 0.95,
) -> list[EvalRow]:
    """Join run results with optional gold labels into a list of EvalRow.

    Args:
        run_dirs: One or more experiment run directories OR a single parent
            directory (artifacts/experiments) — in the latter case all
            subdirectories matching *tag_contains* are walked.
        gold_path: Optional gold CSV for ground-truth join.
        tag_contains: Filter string applied to run-directory names.
        min_schema_coverage: Minimum non-empty ratio for key columns.

    Returns:
        List of EvalRow, one per (run, strategy, question).
    """
    resolved_dirs = _resolve_run_dirs(run_dirs, tag_contains)
    gold_map: dict[str, dict[str, str]] = {}
    if gold_path:
        gold_map = load_gold(gold_path)

    out: list[EvalRow] = []
    skip_counter: Counter[str] = Counter()

    for run_dir in resolved_dirs:
        raw_rows = load_raw_rows(run_dir)
        run_dir_name = run_dir.name

        for raw in raw_rows:
            partial = raw_row_to_partial(raw, run_dir_name)
            question = partial["question"]
            question_key = normalize_question(question)

            gold_row = gold_map.get(question_key, {})
            has_gold = bool(gold_row)

            expected_entities: list[Any] = []
            gold_triples: list[Any] = []
            answer_variants: list[str] = []
            ground_truth = ""

            if has_gold:
                ground_truth = pick_ground_truth(gold_row)
                expected_entities, gold_triples, answer_variants = extract_gold_annotations(gold_row)

            skip_reason = _determine_skip_reason(
                question_key=question_key,
                has_gold=has_gold,
                malformed_json=False,
                contexts=partial["contexts"],
            )
            if skip_reason:
                skip_counter[skip_reason] += 1

            out.append(
                EvalRow(
                    run_dir=run_dir_name,
                    strategy=partial["strategy"],
                    framework=partial["framework"],
                    model_id=partial["model_id"],
                    run_index=partial["run_index"],
                    question_id=gold_row.get("question_id", "") if has_gold else "",
                    question_type=gold_row.get("question_type", "") if has_gold else "",
                    difficulty=gold_row.get("difficulty", "") if has_gold else "",
                    notes=gold_row.get("notes", "") if has_gold else "",
                    question=question,
                    answer=partial["answer"],
                    ground_truth=ground_truth,
                    answer_variants=[str(v) for v in answer_variants],
                    contexts=partial["contexts"],
                    retrieved_triples=partial["retrieved_triples"],
                    retrieved_entities=partial["retrieved_entities"],
                    expected_entities=expected_entities,
                    gold_triples=gold_triples,
                    latency_ms=partial["latency_ms"],
                    kg_triples_used=partial["kg_triples_used"],
                    kg_neighbors_used=partial["kg_neighbors_used"],
                    kg_subgraph_triples_used=partial["kg_subgraph_triples_used"],
                    kg_shortest_path_triples_used=partial["kg_shortest_path_triples_used"],
                    sub_questions=partial["sub_questions"],
                    insufficient=partial["insufficient"],
                    skip_reason=skip_reason,
                )
            )

    if skip_counter:
        logger.info("skip_reasons=%s", dict(skip_counter))

    _validate_schema(out, min_schema_coverage)
    return out


def _resolve_run_dirs(run_dirs: list[Path], tag_contains: str) -> list[Path]:
    """Resolve a list of paths to concrete run directories.

    A single directory with results.jsonl/results.csv is used as-is.
    A parent directory without results files is walked to find subdirs.
    """
    resolved: list[Path] = []

    for path in run_dirs:
        if not path.exists():
            logger.warning("Run path not found, skipping: %s", path)
            continue

        if (path / "results.jsonl").exists() or (path / "results.csv").exists():
            resolved.append(path)
            continue

        # Parent directory — walk subdirs
        for child in sorted(path.iterdir()):
            if not child.is_dir():
                continue
            if tag_contains and tag_contains not in child.name:
                continue
            if (child / "results.jsonl").exists() or (child / "results.csv").exists():
                resolved.append(child)

    return resolved


# ─── CSV round-trip helpers ──────────────────────────────────────────────────

def rows_to_csv(rows: list[EvalRow], output_path: Path) -> None:
    """Write EvalRow list to a CSV compatible with the legacy build_eval_dataset format."""
    fieldnames = [
        "run_dir", "strategy", "framework", "model_id", "run_index",
        "question_id", "question_type", "difficulty", "notes",
        "question", "answer", "ground_truth", "answer_variants_json",
        "latency_ms", "kg_triples_used", "kg_neighbors_used",
        "kg_subgraph_triples_used", "kg_shortest_path_triples_used", "sub_questions",
        "contexts_json", "retrieved_triples_json", "retrieved_entities_json",
        "expected_entities_json", "gold_triples_json",
        "has_gold_match", "skip_reason",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_row_to_dict(row))


def _row_to_dict(row: EvalRow) -> dict[str, str]:
    return {
        "run_dir": row.run_dir,
        "strategy": row.strategy,
        "framework": row.framework,
        "model_id": row.model_id,
        "run_index": row.run_index,
        "question_id": row.question_id,
        "question_type": row.question_type,
        "difficulty": row.difficulty,
        "notes": row.notes,
        "question": row.question,
        "answer": row.answer,
        "ground_truth": row.ground_truth,
        "answer_variants_json": json.dumps(row.answer_variants, ensure_ascii=False),
        "latency_ms": str(row.latency_ms),
        "kg_triples_used": str(row.kg_triples_used),
        "kg_neighbors_used": str(row.kg_neighbors_used),
        "kg_subgraph_triples_used": str(row.kg_subgraph_triples_used),
        "kg_shortest_path_triples_used": str(row.kg_shortest_path_triples_used),
        "sub_questions": str(row.sub_questions),
        "contexts_json": json.dumps(row.contexts, ensure_ascii=False),
        "retrieved_triples_json": json.dumps(row.retrieved_triples, ensure_ascii=False),
        "retrieved_entities_json": json.dumps(row.retrieved_entities, ensure_ascii=False),
        "expected_entities_json": json.dumps(row.expected_entities, ensure_ascii=False),
        "gold_triples_json": json.dumps(row.gold_triples, ensure_ascii=False),
        "has_gold_match": "1" if row.has_gold else "0",
        "skip_reason": row.skip_reason,
    }


def rows_from_csv(input_path: Path) -> list[EvalRow]:
    """Load EvalRow list from a CSV produced by rows_to_csv or build_eval_dataset."""
    rows: list[EvalRow] = []
    with input_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            rows.append(_dict_to_row(raw))
    return rows


def _dict_to_row(d: dict[str, str]) -> EvalRow:
    def _jlist(key: str) -> list[Any]:
        raw = d.get(key, "")
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []

    def _f(key: str) -> float:
        try:
            return float(d.get(key, 0) or 0)
        except (ValueError, TypeError):
            return 0.0

    def _i(key: str) -> int:
        try:
            return int(d.get(key, 0) or 0)
        except (ValueError, TypeError):
            return 0

    return EvalRow(
        run_dir=d.get("run_dir", ""),
        strategy=d.get("strategy", ""),
        framework=d.get("framework", ""),
        model_id=d.get("model_id", ""),
        run_index=d.get("run_index", "0"),
        question_id=d.get("question_id", ""),
        question_type=d.get("question_type", ""),
        difficulty=d.get("difficulty", ""),
        notes=d.get("notes", ""),
        question=d.get("question", ""),
        answer=d.get("answer", ""),
        ground_truth=d.get("ground_truth", ""),
        answer_variants=_jlist("answer_variants_json"),
        contexts=_jlist("contexts_json"),
        retrieved_triples=_jlist("retrieved_triples_json"),
        retrieved_entities=_jlist("retrieved_entities_json"),
        expected_entities=_jlist("expected_entities_json"),
        gold_triples=_jlist("gold_triples_json"),
        latency_ms=_f("latency_ms"),
        kg_triples_used=_i("kg_triples_used"),
        kg_neighbors_used=_i("kg_neighbors_used"),
        kg_subgraph_triples_used=_i("kg_subgraph_triples_used"),
        kg_shortest_path_triples_used=_i("kg_shortest_path_triples_used"),
        sub_questions=_i("sub_questions"),
        # Recompute from the answer text (same rule as raw_row_to_partial):
        # hardcoding False here made the CSV round-trip silently lose the
        # insufficiency signal.
        insufficient=_is_insufficient(d.get("answer", "")),
        skip_reason=d.get("skip_reason", ""),
    )
