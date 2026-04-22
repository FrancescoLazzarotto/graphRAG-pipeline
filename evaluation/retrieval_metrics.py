from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _normalize_text(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _parse_json_list(raw: str) -> list[Any]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        return []
    return []


def _normalize_entity(item: Any) -> str:
    if isinstance(item, str):
        return _normalize_text(item)
    if isinstance(item, dict):
        for key in ("id", "name", "label", "entity"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return _normalize_text(value)
    return ""


def _normalize_triple(item: Any) -> str:
    if isinstance(item, dict):
        subject = str(item.get("subject") or item.get("s") or "").strip()
        predicate = str(item.get("predicate") or item.get("p") or "").strip()
        obj = str(item.get("object") or item.get("o") or "").strip()
        if subject or predicate or obj:
            return "|".join([_normalize_text(subject), _normalize_text(predicate), _normalize_text(obj)])

    if isinstance(item, list) and len(item) >= 3:
        return "|".join(_normalize_text(str(part)) for part in item[:3])

    if isinstance(item, str):
        value = item.strip()
        if "|" in value:
            parts = [part.strip() for part in value.split("|", 2)]
            while len(parts) < 3:
                parts.append("")
            return "|".join(_normalize_text(part) for part in parts[:3])
        return _normalize_text(value)

    return ""


def _entity_coverage(expected_entities: list[Any], retrieved_entities: list[Any]) -> float | None:
    expected = {_normalize_entity(item) for item in expected_entities}
    expected.discard("")
    if not expected:
        return None

    retrieved = {_normalize_entity(item) for item in retrieved_entities}
    retrieved.discard("")

    return len(expected & retrieved) / len(expected)


def _triple_metrics(gold_triples: list[Any], retrieved_triples: list[Any]) -> tuple[float | None, float | None, float | None]:
    ranked = [_normalize_triple(item) for item in retrieved_triples]
    ranked = [item for item in ranked if item]

    gold_set = {_normalize_triple(item) for item in gold_triples}
    gold_set.discard("")

    if not gold_set or not ranked:
        return (None, None, None)

    relevant_positions = [index + 1 for index, triple in enumerate(ranked) if triple in gold_set]
    relevant_count = len(relevant_positions)

    precision_at_k = relevant_count / len(ranked)
    recall_at_k = relevant_count / len(gold_set)
    mrr = 1.0 / relevant_positions[0] if relevant_positions else 0.0
    return (precision_at_k, recall_at_k, mrr)


def _safe_float(raw: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["model_id"]), str(row["framework"]), str(row["strategy"]))
        grouped[key].append(row)

    summary: list[dict[str, Any]] = []

    for (model_id, framework, strategy), items in sorted(grouped.items(), key=lambda item: item[0]):
        cov_values = [float(item["entity_coverage"]) for item in items if item["entity_coverage"] is not None]
        p_values = [float(item["precision_at_k"]) for item in items if item["precision_at_k"] is not None]
        r_values = [float(item["recall_at_k"]) for item in items if item["recall_at_k"] is not None]
        mrr_values = [float(item["mrr"]) for item in items if item["mrr"] is not None]

        summary.append(
            {
                "model_id": model_id,
                "framework": framework,
                "strategy": strategy,
                "rows": len(items),
                "rows_with_gold": sum(1 for item in items if item["has_gold_match"]),
                "rows_with_entity_labels": len(cov_values),
                "rows_with_triple_labels": len(p_values),
                "avg_entity_coverage": mean(cov_values) if cov_values else 0.0,
                "avg_precision_at_k": mean(p_values) if p_values else 0.0,
                "avg_recall_at_k": mean(r_values) if r_values else 0.0,
                "avg_mrr": mean(mrr_values) if mrr_values else 0.0,
                "avg_latency_ms": mean(_safe_float(str(item.get("latency_ms", "0"))) for item in items),
            }
        )

    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute retrieval-oriented metrics from evaluation dataset")
    parser.add_argument("--input", required=True, help="CSV produced by evaluation/build_eval_dataset.py")
    parser.add_argument("--save-csv", default="", help="Optional output CSV path")
    parser.add_argument("--save-json", default="", help="Optional output JSON path")
    parser.add_argument("--save-row-csv", default="", help="Optional row-level metrics CSV path")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    input_path = Path(args.input)

    if not input_path.exists() or not input_path.is_file():
        print(f"Input CSV not found: {input_path}")
        return 1

    row_metrics: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            expected_entities = _parse_json_list(row.get("expected_entities_json", ""))
            retrieved_entities = _parse_json_list(row.get("retrieved_entities_json", ""))
            retrieved_triples = _parse_json_list(row.get("retrieved_triples_json", ""))
            gold_triples = _parse_json_list(row.get("gold_triples_json", ""))

            entity_coverage = _entity_coverage(expected_entities=expected_entities, retrieved_entities=retrieved_entities)
            precision_at_k, recall_at_k, mrr = _triple_metrics(gold_triples=gold_triples, retrieved_triples=retrieved_triples)

            row_metrics.append(
                {
                    "run_dir": row.get("run_dir", ""),
                    "model_id": row.get("model_id", ""),
                    "framework": row.get("framework", ""),
                    "strategy": row.get("strategy", ""),
                    "question": row.get("question", ""),
                    "latency_ms": row.get("latency_ms", ""),
                    "has_gold_match": str(row.get("has_gold_match", "0")).strip() == "1",
                    "entity_coverage": entity_coverage,
                    "precision_at_k": precision_at_k,
                    "recall_at_k": recall_at_k,
                    "mrr": mrr,
                }
            )

    summary = _aggregate(row_metrics)

    print(f"rows_total={len(row_metrics)}")
    print(f"groups={len(summary)}")
    print("\nSummary:")
    header = (
        f"{'model_id':<28} {'framework':<13} {'strategy':<18} {'rows':>6} {'gold':>6} "
        f"{'entity_cov':>10} {'p@k':>8} {'r@k':>8} {'mrr':>8}"
    )
    print(header)
    print("-" * len(header))
    for item in summary:
        print(
            f"{str(item['model_id'])[:28]:<28} {str(item['framework'])[:13]:<13} {str(item['strategy'])[:18]:<18}"
            f" {int(item['rows']):>6d} {int(item['rows_with_gold']):>6d}"
            f" {float(item['avg_entity_coverage']):>10.3f} {float(item['avg_precision_at_k']):>8.3f}"
            f" {float(item['avg_recall_at_k']):>8.3f} {float(item['avg_mrr']):>8.3f}"
        )

    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"saved_json={output_path}")

    if args.save_csv:
        output_path = Path(args.save_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "model_id",
            "framework",
            "strategy",
            "rows",
            "rows_with_gold",
            "rows_with_entity_labels",
            "rows_with_triple_labels",
            "avg_entity_coverage",
            "avg_precision_at_k",
            "avg_recall_at_k",
            "avg_mrr",
            "avg_latency_ms",
        ]
        with output_path.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            for item in summary:
                writer.writerow(item)
        print(f"saved_csv={output_path}")

    if args.save_row_csv:
        output_path = Path(args.save_row_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "run_dir",
            "model_id",
            "framework",
            "strategy",
            "question",
            "latency_ms",
            "has_gold_match",
            "entity_coverage",
            "precision_at_k",
            "recall_at_k",
            "mrr",
        ]
        with output_path.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            for item in row_metrics:
                writer.writerow({key: item.get(key, "") for key in fieldnames})
        print(f"saved_row_csv={output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
