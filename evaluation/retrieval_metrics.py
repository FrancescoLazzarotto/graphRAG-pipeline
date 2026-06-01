from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

logger = logging.getLogger("graphrag")

EXPECTED_QUESTION_TYPES = {"factoid", "relation", "path", "multi_hop"}
METRICS_TO_AGGREGATE = [
    "entity_coverage",
    "precision_at_k",
    "recall_at_k",
    "hit_at_k",
    "ndcg_at_k",
    "mrr",
    "map",
    "latency_ms",
]


def _normalize_text(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _parse_json_list(raw: str) -> list[Any]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return parsed
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


def _unique_non_empty(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def precision_at_k(relevant: list[str], retrieved: list[str], k: int | None = None) -> float:
    """Compute precision with optional top-k truncation.

    Args:
        relevant: Relevant item ids.
        retrieved: Retrieved item ids in rank order.
        k: Top-k cutoff. When None, uses all retrieved items.

    Returns:
        Precision in [0, 1].
    """
    if k is not None and k <= 0:
        return 0.0
    ranked = retrieved if k is None else retrieved[:k]
    if not ranked:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for item in ranked if item in relevant_set)
    return hits / len(ranked)


def recall_at_k(relevant: list[str], retrieved: list[str], k: int | None = None) -> float:
    """Compute recall with optional top-k truncation.

    Args:
        relevant: Relevant item ids.
        retrieved: Retrieved item ids in rank order.
        k: Top-k cutoff. When None, uses all retrieved items.

    Returns:
        Recall in [0, 1].
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    if k is not None and k <= 0:
        return 0.0
    ranked = retrieved if k is None else retrieved[:k]
    hits = {item for item in ranked if item in relevant_set}
    return len(hits) / len(relevant_set)


def hit_at_k(relevant: list[str], retrieved: list[str], k: int) -> float:
    """Return 1.0 if any relevant item appears in top-k, else 0.0.

    Args:
        relevant: Relevant item ids.
        retrieved: Retrieved item ids in rank order.
        k: Top-k cutoff.

    Returns:
        Hit@k as 0.0 or 1.0.
    """
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    return 1.0 if any(item in relevant_set for item in retrieved[:k]) else 0.0


def ndcg_at_k(relevant: list[str], retrieved: list[str], k: int) -> float:
    """Compute binary nDCG@k.

    Args:
        relevant: Relevant item ids.
        retrieved: Retrieved item ids in rank order.
        k: Top-k cutoff.

    Returns:
        nDCG@k in [0, 1].
    """
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0

    ranked = retrieved[:k]
    if not ranked:
        return 0.0

    dcg = 0.0
    for index, item in enumerate(ranked):
        if item in relevant_set:
            dcg += 1.0 / math.log2(index + 2)

    ideal_hits = min(len(relevant_set), len(ranked))
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(relevant: list[str], retrieved: list[str]) -> float:
    """Compute reciprocal rank for a single query.

    Args:
        relevant: Relevant item ids.
        retrieved: Retrieved item ids in rank order.

    Returns:
        Reciprocal rank in [0, 1].
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    for index, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            return 1.0 / index
    return 0.0


def mean_average_precision(relevant: list[str], retrieved: list[str]) -> float:
    """Compute average precision for a single query.

    Args:
        relevant: Relevant item ids.
        retrieved: Retrieved item ids in rank order.

    Returns:
        Average precision in [0, 1].
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0

    hit_count = 0
    precision_sum = 0.0
    seen_relevant: set[str] = set()
    for index, item in enumerate(retrieved, start=1):
        if item in relevant_set and item not in seen_relevant:
            seen_relevant.add(item)
            hit_count += 1
            precision_sum += hit_count / index
    return precision_sum / len(relevant_set)


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    if quantile <= 0:
        return sorted_values[0]
    if quantile >= 1:
        return sorted_values[-1]
    position = quantile * (len(sorted_values) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return sorted_values[low]
    fraction = position - low
    return sorted_values[low] + fraction * (sorted_values[high] - sorted_values[low])


def bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval over mean scores.

    Args:
        scores: Observed metric values.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level in (0, 1).
        seed: Random seed.

    Returns:
        Tuple with lower and upper confidence bounds.
    """
    if not scores:
        return (0.0, 0.0)
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be > 0")
    if not 0 < ci < 1:
        raise ValueError("ci must be in (0, 1)")
    if len(scores) == 1:
        return (scores[0], scores[0])

    rng = random.Random(seed)
    n = len(scores)
    means: list[float] = []
    for _ in range(n_bootstrap):
        sample = [scores[rng.randrange(n)] for _ in range(n)]
        means.append(mean(sample))
    means.sort()

    alpha = (1.0 - ci) / 2.0
    return (_percentile(means, alpha), _percentile(means, 1.0 - alpha))


def _entity_coverage(expected_entities: list[Any], retrieved_entities: list[Any]) -> float | None:
    expected = {_normalize_entity(item) for item in expected_entities}
    expected.discard("")
    if not expected:
        return None
    retrieved = {_normalize_entity(item) for item in retrieved_entities}
    retrieved.discard("")
    return len(expected & retrieved) / len(expected)


def _safe_float(raw: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _compute_row_metrics(row: dict[str, str], k: int | None) -> dict[str, Any]:
    expected_entities = _parse_json_list(row.get("expected_entities_json", ""))
    retrieved_entities = _parse_json_list(row.get("retrieved_entities_json", ""))
    gold_triples_raw = _parse_json_list(row.get("gold_triples_json", ""))
    retrieved_triples_raw = _parse_json_list(row.get("retrieved_triples_json", ""))

    relevant_triples = _unique_non_empty([_normalize_triple(item) for item in gold_triples_raw])
    retrieved_triples = _unique_non_empty([_normalize_triple(item) for item in retrieved_triples_raw])

    skip_reason = (row.get("skip_reason", "") or "").strip()

    precision: float | None = None
    recall: float | None = None
    hit: float | None = None
    ndcg: float | None = None
    reciprocal_rank: float | None = None
    average_precision: float | None = None

    if not skip_reason and relevant_triples:
        cutoff = k if k is not None else len(retrieved_triples)
        effective_k = max(cutoff, 1)
        precision = precision_at_k(relevant_triples, retrieved_triples, k=k)
        recall = recall_at_k(relevant_triples, retrieved_triples, k=k)
        hit = hit_at_k(relevant_triples, retrieved_triples, k=effective_k)
        ndcg = ndcg_at_k(relevant_triples, retrieved_triples, k=effective_k)
        reciprocal_rank = mrr(relevant_triples, retrieved_triples)
        average_precision = mean_average_precision(relevant_triples, retrieved_triples)

    question_type = _normalize_text(row.get("question_type", "")) if row.get("question_type") else ""
    return {
        "run_dir": row.get("run_dir", ""),
        "model_id": row.get("model_id", ""),
        "framework": row.get("framework", ""),
        "strategy": row.get("strategy", ""),
        "question": row.get("question", ""),
        "question_type": question_type,
        "latency_ms": _safe_float(row.get("latency_ms", "")),
        "has_gold_match": str(row.get("has_gold_match", "0")).strip() == "1",
        "skip_reason": skip_reason,
        "entity_coverage": _entity_coverage(expected_entities, retrieved_entities),
        "precision_at_k": precision,
        "recall_at_k": recall,
        "hit_at_k": hit,
        "ndcg_at_k": ndcg,
        "mrr": reciprocal_rank,
        "map": average_precision,
    }


def _metric_summary(values: list[float], n_bootstrap: int, ci: float, seed: int) -> dict[str, float | int]:
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "ci_lower": 0.0,
            "ci_95_upper": 0.0,
            "n_samples": 0,
        }
    ci_lower, ci_upper = bootstrap_ci(values, n_bootstrap=n_bootstrap, ci=ci, seed=seed)
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "ci_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "n_samples": len(values),
    }


def _summarize_group(
    items: list[dict[str, Any]],
    model_id: str,
    framework: str,
    strategy: str,
    segment: str,
    n_bootstrap: int,
    ci: float,
    seed: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "model_id": model_id,
        "framework": framework,
        "strategy": strategy,
        "segment": segment,
        "rows": len(items),
        "rows_with_gold": sum(1 for item in items if item["has_gold_match"]),
        "rows_with_entity_labels": sum(1 for item in items if item["entity_coverage"] is not None),
        "rows_with_triple_labels": sum(1 for item in items if item["precision_at_k"] is not None),
    }

    for metric_name in METRICS_TO_AGGREGATE:
        values = [float(item[metric_name]) for item in items if item[metric_name] is not None]
        stats = _metric_summary(values, n_bootstrap=n_bootstrap, ci=ci, seed=seed)
        summary[f"{metric_name}_mean"] = stats["mean"]
        summary[f"{metric_name}_std"] = stats["std"]
        summary[f"{metric_name}_ci_lower"] = stats["ci_lower"]
        summary[f"{metric_name}_ci_95_upper"] = stats["ci_95_upper"]
        summary[f"{metric_name}_n_samples"] = stats["n_samples"]

    # Backward-compatible aliases.
    summary["avg_entity_coverage"] = summary["entity_coverage_mean"]
    summary["avg_precision_at_k"] = summary["precision_at_k_mean"]
    summary["avg_recall_at_k"] = summary["recall_at_k_mean"]
    summary["avg_mrr"] = summary["mrr_mean"]
    summary["avg_latency_ms"] = summary["latency_ms_mean"]
    return summary


def _aggregate(
    rows: list[dict[str, Any]],
    n_bootstrap: int,
    ci: float,
    seed: int,
    question_type_available: bool,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["model_id"]), str(row["framework"]), str(row["strategy"]))
        grouped[key].append(row)

    summary: list[dict[str, Any]] = []
    for (model_id, framework, strategy), items in sorted(grouped.items(), key=lambda value: value[0]):
        summary.append(
            _summarize_group(
                items=items,
                model_id=model_id,
                framework=framework,
                strategy=strategy,
                segment="global",
                n_bootstrap=n_bootstrap,
                ci=ci,
                seed=seed,
            )
        )

        if question_type_available:
            by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for item in items:
                segment = str(item.get("question_type", "")).strip()
                if segment:
                    by_segment[segment].append(item)

            for segment, segment_items in sorted(by_segment.items(), key=lambda value: value[0]):
                summary.append(
                    _summarize_group(
                        items=segment_items,
                        model_id=model_id,
                        framework=framework,
                        strategy=strategy,
                        segment=segment,
                        n_bootstrap=n_bootstrap,
                        ci=ci,
                        seed=seed,
                    )
                )

    return summary


def _default_smoke_input_path() -> Path:
    return Path(__file__).resolve().parent.parent / "artifacts" / "evaluation" / "smoke_eval_dataset.csv"


def _build_smoke_rows() -> list[dict[str, str]]:
    return [
        {
            "run_dir": "smoke_run",
            "model_id": "smoke-model",
            "framework": "graphrag",
            "strategy": "default",
            "question": "What ingredient replaces egg in vegan mayo?",
            "question_type": "factoid",
            "latency_ms": "25",
            "has_gold_match": "1",
            "skip_reason": "",
            "expected_entities_json": json.dumps(["aquafaba", "egg", "vegan mayo"]),
            "retrieved_entities_json": json.dumps(["aquafaba", "vegan mayo"]),
            "gold_triples_json": json.dumps(
                [{"subject": "aquafaba", "predicate": "SUBSTITUTES", "object": "egg"}]
            ),
            "retrieved_triples_json": json.dumps(
                [{"subject": "aquafaba", "predicate": "SUBSTITUTES", "object": "egg"}]
            ),
        },
        {
            "run_dir": "smoke_run",
            "model_id": "smoke-model",
            "framework": "graphrag",
            "strategy": "default",
            "question": "Which method keeps oven potatoes crispy?",
            "question_type": "relation",
            "latency_ms": "35",
            "has_gold_match": "1",
            "skip_reason": "",
            "expected_entities_json": json.dumps(["potato", "double cook", "oven"]),
            "retrieved_entities_json": json.dumps(["potato", "oven"]),
            "gold_triples_json": json.dumps(
                [{"subject": "double cook", "predicate": "IMPROVES_TEXTURE", "object": "potato"}]
            ),
            "retrieved_triples_json": json.dumps(
                [
                    {"subject": "salt", "predicate": "SEASONS", "object": "potato"},
                    {"subject": "double cook", "predicate": "IMPROVES_TEXTURE", "object": "potato"},
                ]
            ),
        },
    ]


def _load_csv_rows(input_path: Path) -> tuple[list[dict[str, str]], bool]:
    with input_path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        question_type_available = "question_type" in (reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return rows, question_type_available


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute retrieval-oriented metrics from evaluation dataset")
    parser.add_argument("--input", default="", help="CSV produced by evaluation/build_eval_dataset.py")
    parser.add_argument("--k", type=int, default=None, help="Optional top-k for precision/recall/hit/ndcg")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Number of bootstrap resamples")
    parser.add_argument("--ci", type=float, default=0.95, help="Confidence interval level")
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap seed")
    parser.add_argument("--save-csv", default="", help="Optional output CSV path")
    parser.add_argument("--save-json", default="", help="Optional output JSON path")
    parser.add_argument("--save-row-csv", default="", help="Optional row-level metrics CSV path")
    parser.add_argument("--smoke", action="store_true", help="Run smoke evaluation on fixture/mock dataset")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s - %(message)s")
    args = _build_parser().parse_args()

    if args.k is not None and args.k <= 0:
        logger.error("--k must be > 0 when provided")
        return 1
    if args.n_bootstrap <= 0:
        logger.error("--n-bootstrap must be > 0")
        return 1
    if not 0 < args.ci < 1:
        logger.error("--ci must be in (0, 1)")
        return 1

    if args.smoke:
        input_path = Path(args.input) if args.input else _default_smoke_input_path()
        if input_path.exists() and input_path.is_file():
            raw_rows, question_type_available = _load_csv_rows(input_path)
            logger.info("Loaded smoke dataset from %s", input_path)
        else:
            raw_rows = _build_smoke_rows()
            question_type_available = True
            logger.warning("Smoke input file not found, using in-memory mock rows.")
    else:
        if not args.input:
            logger.error("--input is required unless --smoke is used")
            return 1
        input_path = Path(args.input)
        if not input_path.exists() or not input_path.is_file():
            logger.error("Input CSV not found: %s", input_path)
            return 1
        raw_rows, question_type_available = _load_csv_rows(input_path)

    if not raw_rows:
        logger.error("No rows found in input dataset")
        return 1

    row_metrics = [_compute_row_metrics(row=row, k=args.k) for row in raw_rows]

    if not question_type_available:
        logger.warning("Column question_type not found in input; segmented metrics are skipped.")
    else:
        observed_types = {
            str(item.get("question_type", "")).strip()
            for item in row_metrics
            if str(item.get("question_type", "")).strip()
        }
        unknown_types = sorted(observed_types - EXPECTED_QUESTION_TYPES)
        if unknown_types:
            logger.warning("Unexpected question_type values found: %s", ", ".join(unknown_types))

    summary = _aggregate(
        rows=row_metrics,
        n_bootstrap=args.n_bootstrap,
        ci=args.ci,
        seed=args.seed,
        question_type_available=question_type_available,
    )

    logger.info("rows_total=%d", len(row_metrics))
    logger.info("groups=%d", len(summary))
    for item in summary:
        logger.info(
            "group model=%s framework=%s strategy=%s segment=%s rows=%d p=%.3f r=%.3f mrr=%.3f",
            item["model_id"],
            item["framework"],
            item["strategy"],
            item["segment"],
            int(item["rows"]),
            float(item["precision_at_k_mean"]),
            float(item["recall_at_k_mean"]),
            float(item["mrr_mean"]),
        )

    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info("saved_json=%s", output_path)

    if args.save_csv:
        output_path = Path(args.save_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in summary for key in row.keys()})
        with output_path.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            for item in summary:
                writer.writerow(item)
        logger.info("saved_csv=%s", output_path)

    if args.save_row_csv:
        output_path = Path(args.save_row_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "run_dir",
            "model_id",
            "framework",
            "strategy",
            "question",
            "question_type",
            "latency_ms",
            "has_gold_match",
            "skip_reason",
            "entity_coverage",
            "precision_at_k",
            "recall_at_k",
            "hit_at_k",
            "ndcg_at_k",
            "mrr",
            "map",
        ]
        with output_path.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            for item in row_metrics:
                writer.writerow({key: item.get(key, "") for key in fieldnames})
        logger.info("saved_row_csv=%s", output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
