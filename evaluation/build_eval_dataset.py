from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def _normalize_question(question: str) -> str:
    text = question.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _parse_json(raw: str) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _parse_json_list(raw: str) -> list[Any]:
    parsed = _parse_json(raw)
    if isinstance(parsed, list):
        return parsed
    return []


def _parse_metadata(raw: str) -> dict[str, Any]:
    parsed = _parse_json(raw)
    if isinstance(parsed, dict):
        return parsed
    return {}


def _iter_results_csv(input_path: Path, tag_contains: str) -> list[Path]:
    if input_path.is_file() and input_path.name == "results.csv":
        return [input_path]

    if input_path.is_dir() and (input_path / "results.csv").exists():
        return [input_path / "results.csv"]

    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    files: list[Path] = []
    for csv_path in sorted(input_path.glob("*/results.csv")):
        run_dir_name = csv_path.parent.name
        if tag_contains and tag_contains not in run_dir_name:
            continue
        files.append(csv_path)

    if not files:
        raise FileNotFoundError(f"No results.csv files found under: {input_path}")

    return files


def _extract_list_json(row: dict[str, str], metadata: dict[str, Any], row_key_candidates: list[str], metadata_key_candidates: list[str]) -> list[Any]:
    for key in row_key_candidates:
        value = _parse_json_list(row.get(key, ""))
        if value:
            return value

    for key in metadata_key_candidates:
        value = metadata.get(key)
        if isinstance(value, list):
            return value

    return []


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

            key = _normalize_question(question)
            if key in gold_by_question:
                duplicate_keys += 1
                continue
            gold_by_question[key] = row

    return gold_by_question, duplicate_keys


def _pick_ground_truth(gold_row: dict[str, str]) -> str:
    for key in ("ground_truth", "gold_answer", "answer"):
        value = (gold_row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Join experiment results with gold labels for evaluation")
    parser.add_argument("--input", required=True, help="Path to experiments root, run folder, or results.csv")
    parser.add_argument("--gold-file", required=True, help="CSV with at least question and ground_truth")
    parser.add_argument("--tag-contains", default="", help="Optional run-folder filter when input is experiments root")
    parser.add_argument("--output", required=True, help="Output CSV path for joined evaluation dataset")
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    results_files = _iter_results_csv(Path(args.input), tag_contains=args.tag_contains.strip())
    gold_map, duplicates = _load_gold(Path(args.gold_file))

    out_rows: list[dict[str, str]] = []
    total_rows = 0
    matched_rows = 0

    for csv_path in results_files:
        run_dir_name = csv_path.parent.name
        with csv_path.open("r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            for row in reader:
                total_rows += 1

                question = row.get("question", "") or ""
                question_key = _normalize_question(question)
                gold_row = gold_map.get(question_key, {})
                has_gold = bool(gold_row)
                if has_gold:
                    matched_rows += 1

                metadata = _parse_metadata(row.get("metadata_json", ""))

                contexts = _extract_list_json(
                    row=row,
                    metadata=metadata,
                    row_key_candidates=["contexts_json", "retrieved_contexts_json"],
                    metadata_key_candidates=["contexts", "retrieved_contexts", "retrieval_contexts"],
                )
                retrieved_triples = _extract_list_json(
                    row=row,
                    metadata=metadata,
                    row_key_candidates=["retrieved_triples_json", "kg_triples_json"],
                    metadata_key_candidates=["retrieved_triples", "kg_triples"],
                )
                retrieved_entities = _extract_list_json(
                    row=row,
                    metadata=metadata,
                    row_key_candidates=["retrieved_entities_json"],
                    metadata_key_candidates=["retrieved_entities", "kg_entities"],
                )

                expected_entities = _parse_json_list(gold_row.get("expected_entities_json", "") if has_gold else "")
                gold_triples = _parse_json_list(gold_row.get("gold_triples_json", "") if has_gold else "")

                out_rows.append(
                    {
                        "run_dir": run_dir_name,
                        "strategy": row.get("strategy", "") or "",
                        "framework": str(metadata.get("framework", "unknown")),
                        "model_id": str(metadata.get("model_id", "unknown")),
                        "run_index": str(metadata.get("run_index", "0")),
                        "question": question,
                        "answer": row.get("answer", "") or "",
                        "ground_truth": _pick_ground_truth(gold_row) if has_gold else "",
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
                    }
                )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_dir",
        "strategy",
        "framework",
        "model_id",
        "run_index",
        "question",
        "answer",
        "ground_truth",
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
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for item in out_rows:
            writer.writerow(item)

    print(f"results_files={len(results_files)}")
    print(f"rows_total={total_rows}")
    print(f"rows_with_gold={matched_rows}")
    print(f"rows_without_gold={max(total_rows - matched_rows, 0)}")
    print(f"gold_unique_questions={len(gold_map)}")
    print(f"gold_duplicate_questions_ignored={duplicates}")
    print(f"saved={output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
