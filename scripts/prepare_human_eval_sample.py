from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_metadata(raw_value: str) -> dict[str, Any]:
    if not raw_value:
        return {}
    try:
        parsed = json.loads(raw_value)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _iter_result_files(root: Path, tag_contains: str) -> list[Path]:
    if root.is_file() and root.name == "results.csv":
        return [root]

    if root.is_dir() and (root / "results.csv").exists():
        return [root / "results.csv"]

    if not root.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {root}")

    files: list[Path] = []
    for candidate in sorted(root.glob("*/results.csv")):
        run_dir = candidate.parent.name
        if tag_contains and tag_contains not in run_dir:
            continue
        files.append(candidate)

    if not files:
        raise FileNotFoundError(f"No results.csv files found under: {root}")

    return files


def _load_rows(csv_files: list[Path], framework_filter: str, model_filter: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            for row in reader:
                metadata = _load_metadata(row.get("metadata_json", ""))

                framework = str(metadata.get("framework", "unknown"))
                model_id = str(metadata.get("model_id", "unknown"))
                if framework_filter and framework != framework_filter:
                    continue
                if model_filter and model_id != model_filter:
                    continue

                rows.append(
                    {
                        "run_dir": csv_path.parent.name,
                        "framework": framework,
                        "model_id": model_id,
                        "strategy": row.get("strategy", ""),
                        "question": row.get("question", ""),
                        "answer": row.get("answer", ""),
                        "latency_ms": float(row.get("latency_ms", "0") or 0.0),
                    }
                )

    return rows


def _sample_rows(rows: list[dict[str, Any]], sample_size: int, per_group: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if not rows:
        return []

    if per_group > 0:
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = (str(row["model_id"]), str(row["framework"]), str(row["strategy"]))
            grouped[key].append(row)

        sampled: list[dict[str, Any]] = []
        for key in sorted(grouped.keys()):
            group_rows = list(grouped[key])
            rng.shuffle(group_rows)
            sampled.extend(group_rows[: min(per_group, len(group_rows))])
        return sampled

    if sample_size <= 0 or sample_size >= len(rows):
        sampled = list(rows)
        rng.shuffle(sampled)
        return sampled

    return rng.sample(rows, sample_size)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a human-evaluation sample from experiment results")
    parser.add_argument(
        "input",
        nargs="?",
        default="artifacts/experiments",
        help="Path to experiments root, run folder, or results.csv",
    )
    parser.add_argument("--tag-contains", default="", help="Optional run folder substring filter")
    parser.add_argument("--framework", default="", help="Optional framework filter (e.g. graph_rag)")
    parser.add_argument("--model-id", default="", help="Optional model filter")
    parser.add_argument("--sample-size", type=int, default=200, help="Global sample size when --per-group=0")
    parser.add_argument("--per-group", type=int, default=0, help="Rows sampled for each (model,framework,strategy)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="artifacts/experiments/human_eval_sample.csv",
        help="Output CSV for manual annotation",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    try:
        csv_files = _iter_result_files(Path(args.input), tag_contains=args.tag_contains.strip())
    except FileNotFoundError as exc:
        print(str(exc))
        return 1

    rows = _load_rows(
        csv_files=csv_files,
        framework_filter=args.framework.strip(),
        model_filter=args.model_id.strip(),
    )

    sampled_rows = _sample_rows(
        rows=rows,
        sample_size=max(0, int(args.sample_size)),
        per_group=max(0, int(args.per_group)),
        seed=int(args.seed),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        fieldnames = [
            "sample_id",
            "run_dir",
            "model_id",
            "framework",
            "strategy",
            "question",
            "answer",
            "latency_ms",
            "correctness_score",
            "correctness_label",
            "grounded_score",
            "notes",
            "reviewer",
            "reviewed_at",
        ]
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()

        for index, row in enumerate(sampled_rows, start=1):
            writer.writerow(
                {
                    "sample_id": index,
                    "run_dir": row["run_dir"],
                    "model_id": row["model_id"],
                    "framework": row["framework"],
                    "strategy": row["strategy"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "latency_ms": f"{float(row['latency_ms']):.6f}",
                    "correctness_score": "",
                    "correctness_label": "",
                    "grounded_score": "",
                    "notes": "",
                    "reviewer": "",
                    "reviewed_at": "",
                }
            )

    print(f"Loaded rows: {len(rows)}")
    print(f"Sampled rows: {len(sampled_rows)}")
    print(f"Saved annotation sheet: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
