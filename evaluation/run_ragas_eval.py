from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from pathlib import Path
from statistics import mean, stdev
from typing import Any

logger = logging.getLogger("graphrag")


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


def _load_rows(input_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for index, row in enumerate(reader, start=1):
            rows.append(
                {
                    "row_id": row.get("question_id", "") or f"row_{index}",
                    "run_dir": row.get("run_dir", ""),
                    "strategy": row.get("strategy", ""),
                    "framework": row.get("framework", ""),
                    "model_id": row.get("model_id", ""),
                    "question": row.get("question", ""),
                    "answer": row.get("answer", ""),
                    "ground_truth": row.get("ground_truth", ""),
                    "contexts": _parse_json_list(row.get("contexts_json", "")),
                    "skip_reason": row.get("skip_reason", "") or "",
                }
            )
    return rows


def _resolve_metric_objects(metric_names: list[str]) -> tuple[list[Any], list[str]]:
    import ragas.metrics as ragas_metrics  # type: ignore

    candidates: dict[str, list[str]] = {
        "faithfulness": ["faithfulness", "Faithfulness"],
        "answer_relevancy": ["answer_relevancy", "answer_relevance", "AnswerRelevancy", "ResponseRelevancy"],
        "answer_correctness": ["answer_correctness", "AnswerCorrectness"],
        "context_precision": ["context_precision", "ContextPrecision"],
        "context_recall": ["context_recall", "ContextRecall"],
    }

    resolved: list[Any] = []
    missing: list[str] = []

    for metric_name in metric_names:
        found = None
        for candidate in candidates.get(metric_name, [metric_name]):
            if hasattr(ragas_metrics, candidate):
                found = getattr(ragas_metrics, candidate)
                break

        if found is None:
            missing.append(metric_name)
            continue

        if isinstance(found, type):
            try:
                resolved.append(found())
            except TypeError:
                missing.append(metric_name)
        else:
            resolved.append(found)

    return resolved, missing


def _build_llm_and_embeddings(judge_model: str, embed_model: str) -> tuple[Any | None, Any | None]:
    if not judge_model:
        return (None, None)

    from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline  # type: ignore
    from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore
    from ragas.llms import LangchainLLMWrapper  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(judge_model)
    base_model = AutoModelForCausalLM.from_pretrained(judge_model, device_map="auto")

    generation = pipeline(
        "text-generation",
        model=base_model,
        tokenizer=tokenizer,
        max_new_tokens=192,
        do_sample=False,
        return_full_text=False,
    )

    lc_llm = HuggingFacePipeline(pipeline=generation)
    ragas_llm = LangchainLLMWrapper(lc_llm)

    lc_embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    ragas_embeddings = LangchainEmbeddingsWrapper(lc_embeddings)

    return ragas_llm, ragas_embeddings


def _to_ragas_sample(row: dict[str, Any]) -> dict[str, Any]:
    # Keep contexts structured as list[Any] during loading; RAGAS requires list[str]
    # in the final Dataset object, so cast happens only at this final conversion step.
    contexts = [str(item).strip() for item in row.get("contexts", []) if str(item).strip()]
    return {
        "question": str(row.get("question", "")),
        "answer": str(row.get("answer", "")),
        "ground_truth": str(row.get("ground_truth", "")),
        "contexts": contexts,
    }


def _coerce_score(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _extract_scores(result: Any) -> dict[str, float]:
    if result is None:
        return {}

    if hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        if frame is None or frame.empty:
            return {}
        row = frame.iloc[0].to_dict()
        scores: dict[str, float] = {}
        for key, value in row.items():
            if key in {"question", "answer", "ground_truth", "contexts"}:
                continue
            numeric = _coerce_score(value)
            if numeric is not None:
                scores[str(key)] = numeric
        return scores

    if isinstance(result, dict):
        maybe_scores = result.get("scores")
        if isinstance(maybe_scores, dict):
            scores: dict[str, float] = {}
            for key, value in maybe_scores.items():
                numeric = _coerce_score(value)
                if numeric is not None:
                    scores[str(key)] = numeric
            return scores

    return {}


def _call_ragas(
    evaluate_fn: Any,
    dataset_cls: Any,
    sample: dict[str, Any],
    metric_objects: list[Any],
    ragas_llm: Any | None,
    ragas_embeddings: Any | None,
) -> dict[str, Any]:
    dataset = dataset_cls.from_list([sample])
    raw_result = evaluate_fn(
        dataset=dataset,
        metrics=metric_objects,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    return {"scores": _extract_scores(raw_result)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RAGAS metrics on a joined evaluation dataset")
    parser.add_argument("--input", required=True, help="CSV produced by evaluation/build_eval_dataset.py")
    parser.add_argument(
        "--metrics",
        default="faithfulness,answer_relevancy,answer_correctness,context_precision,context_recall",
        help="Comma-separated RAGAS metric names",
    )
    parser.add_argument("--judge-model", default="", help="Optional local judge model id for RAGAS LLM metrics")
    parser.add_argument(
        "--embed-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model used with RAGAS",
    )
    parser.add_argument("--save-row-csv", default="", help="Optional row-level output CSV")
    parser.add_argument("--save-summary-json", default="", help="Optional summary JSON output")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s - %(message)s")
    args = _build_parser().parse_args()
    input_path = Path(args.input)

    if not input_path.exists() or not input_path.is_file():
        logger.error("Input CSV not found: %s", input_path)
        return 1

    rows = _load_rows(input_path)
    if not rows:
        logger.error("No rows found in input CSV.")
        return 1

    try:
        from datasets import Dataset  # type: ignore
        from ragas import evaluate  # type: ignore
    except Exception as exc:
        logger.error("RAGAS dependencies are missing.")
        logger.error("Install with: conda run -n graphllm python -m pip install -r evaluation/requirements.txt")
        logger.error("Import error: %s", exc)
        return 1

    metric_names = [item.strip() for item in args.metrics.split(",") if item.strip()]
    metric_objects, missing_metrics = _resolve_metric_objects(metric_names)
    if missing_metrics:
        logger.warning("Skipped unsupported metrics: %s", ", ".join(missing_metrics))
    if not metric_objects:
        logger.error("No valid RAGAS metrics resolved. Nothing to evaluate.")
        return 1

    ragas_llm = None
    ragas_embeddings = None
    judge_model = args.judge_model.strip()

    if not judge_model and not os.getenv("OPENAI_API_KEY"):
        logger.error("No local judge model provided and OPENAI_API_KEY is not set.")
        logger.error("Use --judge-model for local reproducible evaluation (example: Qwen/Qwen2.5-14B-Instruct).")
        return 1

    if judge_model:
        ragas_llm, ragas_embeddings = _build_llm_and_embeddings(
            judge_model=judge_model,
            embed_model=args.embed_model.strip(),
        )

    scored_rows: list[dict[str, Any]] = []
    skipped_rows = 0
    skipped_by_input_skip_reason = 0

    for row in rows:
        row_id = str(row.get("row_id", "unknown"))
        if str(row.get("skip_reason", "")).strip():
            skipped_rows += 1
            skipped_by_input_skip_reason += 1
            continue

        sample = _to_ragas_sample(row)

        try:
            result = _call_ragas(
                evaluate_fn=evaluate,
                dataset_cls=Dataset,
                sample=sample,
                metric_objects=metric_objects,
                ragas_llm=ragas_llm,
                ragas_embeddings=ragas_embeddings,
            )
        except Exception as exc:
            logger.warning("RAGAS evaluation failed for row %s: %s", row_id, exc)
            skipped_rows += 1
            continue

        if not result or not result.get("scores"):
            logger.warning("RAGAS returned empty scores for row %s; skipping", row_id)
            skipped_rows += 1
            continue

        merged_row: dict[str, Any] = {
            "row_id": row_id,
            "run_dir": row.get("run_dir", ""),
            "strategy": row.get("strategy", ""),
            "framework": row.get("framework", ""),
            "model_id": row.get("model_id", ""),
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "ground_truth": row.get("ground_truth", ""),
            "contexts_json": json.dumps(sample["contexts"], ensure_ascii=False),
        }
        merged_row.update(result["scores"])
        scored_rows.append(merged_row)

    metric_columns = sorted(
        {
            key
            for row in scored_rows
            for key in row.keys()
            if key
            not in {
                "row_id",
                "run_dir",
                "strategy",
                "framework",
                "model_id",
                "question",
                "answer",
                "ground_truth",
                "contexts_json",
            }
        }
    )

    summary: dict[str, Any] = {
        "rows_input": len(rows),
        "rows_evaluated": len(scored_rows),
        "rows_skipped": skipped_rows,
        "rows_skipped_input_skip_reason": skipped_by_input_skip_reason,
        "metrics": {},
    }

    for metric_name in metric_columns:
        values = [float(row[metric_name]) for row in scored_rows if metric_name in row and _coerce_score(row[metric_name]) is not None]
        if values:
            summary["metrics"][metric_name] = {
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
                "count": len(values),
            }

    logger.info("rows_input=%d", summary["rows_input"])
    logger.info("rows_evaluated=%d", summary["rows_evaluated"])
    logger.info("rows_skipped=%d", summary["rows_skipped"])

    if summary["metrics"]:
        logger.info("metric_means:")
        for metric_name, payload in summary["metrics"].items():
            logger.info(
                "- %s: mean=%.4f std=%.4f n=%d",
                metric_name,
                float(payload["mean"]),
                float(payload["std"]),
                int(payload["count"]),
            )
    else:
        logger.warning("No metric scores produced by RAGAS.")

    if args.save_row_csv:
        output_path = Path(args.save_row_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if scored_rows:
            fieldnames = sorted({key for row in scored_rows for key in row.keys()})
            with output_path.open("w", encoding="utf-8", newline="") as file_obj:
                writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
                writer.writeheader()
                for row in scored_rows:
                    writer.writerow(row)
        else:
            output_path.write_text("", encoding="utf-8")
        logger.info("saved_row_csv=%s", output_path)

    if args.save_summary_json:
        output_path = Path(args.save_summary_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info("saved_summary_json=%s", output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())