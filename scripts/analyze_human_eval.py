from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _parse_correctness(row: dict[str, str]) -> int | None:
    raw_score = (row.get("correctness_score", "") or "").strip()
    if raw_score:
        try:
            score = int(raw_score)
            if score in {0, 1, 2}:
                return score
        except ValueError:
            return None

    label = (row.get("correctness_label", "") or "").strip().lower()
    mapping = {
        "incorrect": 0,
        "wrong": 0,
        "partial": 1,
        "partially_correct": 1,
        "correct": 2,
        "fully_correct": 2,
    }
    return mapping.get(label)


def _parse_grounded(row: dict[str, str]) -> int | None:
    raw = (row.get("grounded_score", "") or "").strip().lower()
    if raw in {"1", "true", "yes", "y"}:
        return 1
    if raw in {"0", "false", "no", "n"}:
        return 0
    return None


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)

    phat = successes / total
    z2 = z * z
    denominator = 1.0 + (z2 / total)
    center = (phat + (z2 / (2.0 * total))) / denominator
    margin = (z * math.sqrt((phat * (1.0 - phat) + (z2 / (4.0 * total))) / total)) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["model_id"]), str(row["framework"]), str(row["strategy"]))
        grouped[key].append(row)

    output: list[dict[str, Any]] = []
    for (model_id, framework, strategy), items in sorted(grouped.items(), key=lambda x: x[0]):
        n = len(items)
        correct = sum(1 for item in items if int(item["correctness"]) == 2)
        partial_or_better = sum(1 for item in items if int(item["correctness"]) >= 1)

        grounded_values = [int(item["grounded"]) for item in items if item["grounded"] is not None]
        grounded_n = len(grounded_values)
        grounded_yes = sum(grounded_values)

        correct_ci_low, correct_ci_high = _wilson_interval(correct, n)
        pob_ci_low, pob_ci_high = _wilson_interval(partial_or_better, n)
        grounded_ci_low, grounded_ci_high = _wilson_interval(grounded_yes, grounded_n) if grounded_n > 0 else (0.0, 0.0)

        output.append(
            {
                "model_id": model_id,
                "framework": framework,
                "strategy": strategy,
                "n": n,
                "full_correct_rate": correct / n if n else 0.0,
                "full_correct_ci_low": correct_ci_low,
                "full_correct_ci_high": correct_ci_high,
                "partial_or_better_rate": partial_or_better / n if n else 0.0,
                "partial_or_better_ci_low": pob_ci_low,
                "partial_or_better_ci_high": pob_ci_high,
                "grounded_n": grounded_n,
                "grounded_rate": grounded_yes / grounded_n if grounded_n else 0.0,
                "grounded_ci_low": grounded_ci_low,
                "grounded_ci_high": grounded_ci_high,
            }
        )

    return output


def _print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No annotated rows found.")
        return

    header = (
        f"{'model_id':<30} {'framework':<13} {'strategy':<18} {'n':>4} "
        f"{'full_acc':>9} {'partial+':>9} {'grounded':>9}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{str(row['model_id'])[:30]:<30} "
            f"{str(row['framework'])[:13]:<13} "
            f"{str(row['strategy'])[:18]:<18} "
            f"{int(row['n']):>4d} "
            f"{float(row['full_correct_rate']) * 100:>8.2f}% "
            f"{float(row['partial_or_better_rate']) * 100:>8.2f}% "
            f"{float(row['grounded_rate']) * 100:>8.2f}%"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate annotated human evaluation sheet")
    parser.add_argument("input", help="Path to annotated CSV generated from prepare_human_eval_sample.py")
    parser.add_argument("--save-json", default="", help="Optional path for aggregated JSON")
    parser.add_argument("--save-csv", default="", help="Optional path for aggregated CSV")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    input_path = Path(args.input)
    if not input_path.exists() or not input_path.is_file():
        print(f"Input CSV not found: {input_path}")
        return 1

    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            correctness = _parse_correctness(row)
            if correctness is None:
                continue

            rows.append(
                {
                    "model_id": (row.get("model_id", "") or "unknown").strip(),
                    "framework": (row.get("framework", "") or "unknown").strip(),
                    "strategy": (row.get("strategy", "") or "unknown").strip(),
                    "correctness": correctness,
                    "grounded": _parse_grounded(row),
                }
            )

    aggregated = _aggregate(rows)
    print(f"Annotated rows considered: {len(rows)}")
    _print_table(aggregated)

    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(aggregated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved JSON: {output_path}")

    if args.save_csv:
        output_path = Path(args.save_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "model_id",
            "framework",
            "strategy",
            "n",
            "full_correct_rate",
            "full_correct_ci_low",
            "full_correct_ci_high",
            "partial_or_better_rate",
            "partial_or_better_ci_low",
            "partial_or_better_ci_high",
            "grounded_n",
            "grounded_rate",
            "grounded_ci_low",
            "grounded_ci_high",
        ]
        with output_path.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            for row in aggregated:
                writer.writerow(row)
        print(f"Saved CSV: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
