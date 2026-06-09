from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from evalkit.models import EvalRow

logger = logging.getLogger("graphrag")

DEFAULT_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "context_precision",
    "context_recall",
]


def _coerce_score(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _to_ragas_sample(row: EvalRow) -> dict[str, Any]:
    contexts = [str(c).strip() for c in row.contexts if str(c).strip()]
    return {
        "question": row.question,
        "answer": row.answer,
        "ground_truth": row.ground_truth,
        "contexts": contexts,
    }


def _resolve_metrics(metric_names: list[str]) -> tuple[list[Any], list[str]]:
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

    for name in metric_names:
        found = None
        for candidate in candidates.get(name, [name]):
            if hasattr(ragas_metrics, candidate):
                found = getattr(ragas_metrics, candidate)
                break

        if found is None:
            missing.append(name)
            continue

        if isinstance(found, type):
            try:
                resolved.append(found())
            except TypeError:
                missing.append(name)
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


def run_ragas(
    rows: list[EvalRow],
    metric_names: list[str] | None = None,
    judge_model: str = "",
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict[str, Any]:
    """Run RAGAS metrics on a list of EvalRow.

    Args:
        rows: EvalRows to evaluate; rows with skip_reason are skipped.
        metric_names: RAGAS metric names to compute.
        judge_model: Local HuggingFace model ID for the judge (or empty → uses OPENAI_API_KEY).
        embed_model: Embedding model ID for RAGAS.

    Returns:
        Dict with rows_evaluated, rows_skipped, metrics (mean/std/count), row_scores.
    """
    if metric_names is None:
        metric_names = DEFAULT_METRICS

    try:
        from datasets import Dataset  # type: ignore
        from ragas import evaluate  # type: ignore
    except Exception as exc:
        raise ImportError(
            "RAGAS dependencies missing. Install with:\n"
            "  pip install -r evaluation/requirements.txt"
        ) from exc

    metric_objects, missing = _resolve_metrics(metric_names)
    if missing:
        logger.warning("Skipped unsupported RAGAS metrics: %s", ", ".join(missing))
    if not metric_objects:
        raise ValueError("No valid RAGAS metrics resolved.")

    judge_model = judge_model.strip()
    if not judge_model and not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "Provide --judge-model for local evaluation or set OPENAI_API_KEY for OpenAI."
        )

    ragas_llm, ragas_embeddings = _build_llm_and_embeddings(judge_model, embed_model)

    scored_rows: list[dict[str, Any]] = []
    skipped = 0

    for row in rows:
        if row.skip_reason:
            skipped += 1
            continue

        sample = _to_ragas_sample(row)
        try:
            dataset = Dataset.from_list([sample])
            result = evaluate(
                dataset=dataset,
                metrics=metric_objects,
                llm=ragas_llm,
                embeddings=ragas_embeddings,
            )
        except Exception as exc:
            logger.warning("RAGAS failed for question %r: %s", row.question[:60], exc)
            skipped += 1
            continue

        # Extract scores
        scores: dict[str, float] = {}
        if hasattr(result, "to_pandas"):
            frame = result.to_pandas()
            if not frame.empty:
                row_data = frame.iloc[0].to_dict()
                for key, value in row_data.items():
                    if key in {"question", "answer", "ground_truth", "contexts"}:
                        continue
                    s = _coerce_score(value)
                    if s is not None:
                        scores[str(key)] = s
        elif isinstance(result, dict):
            for key, value in result.get("scores", {}).items():
                s = _coerce_score(value)
                if s is not None:
                    scores[str(key)] = s

        if not scores:
            logger.warning("RAGAS returned empty scores for %r; skipping", row.question[:60])
            skipped += 1
            continue

        entry = {
            "run_dir": row.run_dir,
            "model_id": row.model_id,
            "framework": row.framework,
            "strategy": row.strategy,
            "question": row.question,
            "question_type": row.question_type,
        }
        entry.update(scores)
        scored_rows.append(entry)

    metric_cols = sorted(
        {
            key
            for row_data in scored_rows
            for key in row_data
            if key not in {"run_dir", "model_id", "framework", "strategy", "question", "question_type"}
        }
    )

    summaries: dict[str, Any] = {}
    for col in metric_cols:
        values = [float(r[col]) for r in scored_rows if _coerce_score(r.get(col)) is not None]
        if values:
            summaries[col] = {
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
                "count": len(values),
            }

    return {
        "rows_evaluated": len(scored_rows),
        "rows_skipped": skipped,
        "metrics": summaries,
        "row_scores": scored_rows,
    }
