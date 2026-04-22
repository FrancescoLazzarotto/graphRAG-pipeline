from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any


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
        for row in reader:
            contexts = [str(item) for item in _parse_json_list(row.get("contexts_json", "")) if str(item).strip()]
            rows.append(
                {
                    "run_dir": row.get("run_dir", ""),
                    "strategy": row.get("strategy", ""),
                    "framework": row.get("framework", ""),
                    "model_id": row.get("model_id", ""),
                    "question": row.get("question", ""),
                    "answer": row.get("answer", ""),
                    "ground_truth": row.get("ground_truth", ""),
                    "contexts": contexts,
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
    args = _build_parser().parse_args()
    input_path = Path(args.input)

    if not input_path.exists() or not input_path.is_file():
        print(f"Input CSV not found: {input_path}")
        return 1

    rows = _load_rows(input_path)
    if not rows:
        print("No rows found in input CSV.")
        return 1

    try:
        from datasets import Dataset  # type: ignore
        from ragas import evaluate  # type: ignore
    except Exception as exc:
        print("RAGAS dependencies are missing.")
        print("Install with: conda run -n graphllm python -m pip install -r evaluation/requirements.txt")
        print(f"Import error: {exc}")
        return 1

    metric_names = [item.strip() for item in args.metrics.split(",") if item.strip()]
    metric_objects, missing_metrics = _resolve_metric_objects(metric_names)

    if missing_metrics:
        print("Skipped unsupported metrics: " + ", ".join(missing_metrics))

    if not metric_objects:
        print("No valid RAGAS metrics resolved. Nothing to evaluate.")
        return 1

    ragas_rows: list[dict[str, Any]] = []
    for row in rows:
        ragas_rows.append(
            {
                "question": row["question"],
                "answer": row["answer"],
                "ground_truth": row["ground_truth"],
                "contexts": row["contexts"],
                "run_dir": row["run_dir"],
                "strategy": row["strategy"],
                "framework": row["framework"],
                "model_id": row["model_id"],
            }
        )

    dataset = Dataset.from_list(ragas_rows)

    ragas_llm = None
    ragas_embeddings = None
    judge_model = args.judge_model.strip()

    if not judge_model and not os.getenv("OPENAI_API_KEY"):
        print("No local judge model provided and OPENAI_API_KEY is not set.")
        print("Use --judge-model for local reproducible evaluation, for example:")
        print("  --judge-model Qwen/Qwen2.5-14B-Instruct")
        return 1

    if judge_model:
        ragas_llm, ragas_embeddings = _build_llm_and_embeddings(
            judge_model=judge_model,
            embed_model=args.embed_model.strip(),
        )

    try:
        result = evaluate(
            dataset=dataset,
            metrics=metric_objects,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
    except Exception as exc:
        print("RAGAS evaluation failed.")
        print(f"Reason: {exc}")
        return 1

    if hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        metric_columns = [col for col in frame.columns if col not in {"question", "answer", "ground_truth", "contexts"}]
        row_output = frame.to_dict(orient="records")
    else:
        # Fallback for API variations.
        metric_columns = []
        row_output = []

    summary: dict[str, Any] = {
        "rows": len(row_output),
        "metrics": {},
    }

    for metric in metric_columns:
        values = [float(item[metric]) for item in row_output if item.get(metric) is not None]
        if values:
            summary["metrics"][metric] = {
                "mean": mean(values),
                "count": len(values),
            }

    print(f"rows_evaluated={summary['rows']}")
    if summary["metrics"]:
        print("metric_means:")
        for metric_name, payload in summary["metrics"].items():
            print(f"- {metric_name}: mean={float(payload['mean']):.4f} n={int(payload['count'])}")
    else:
        print("No metric columns found in RAGAS result output.")

    if args.save_row_csv:
        output_path = Path(args.save_row_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if row_output:
            fieldnames = sorted({key for row in row_output for key in row.keys()})
            with output_path.open("w", encoding="utf-8", newline="") as file_obj:
                writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
                writer.writeheader()
                for row in row_output:
                    writer.writerow(row)
        else:
            with output_path.open("w", encoding="utf-8", newline="") as file_obj:
                file_obj.write("")
        print(f"saved_row_csv={output_path}")

    if args.save_summary_json:
        output_path = Path(args.save_summary_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"saved_summary_json={output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
