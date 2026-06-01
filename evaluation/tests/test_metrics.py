from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import build_eval_dataset as build
from evaluation import retrieval_metrics as metrics
from evaluation import run_ragas_eval as ragas_eval


def test_normalize_question_applies_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_NORMALIZE_REMOVE_ACCENTS", "1")
    assert build.normalize_question("  Caff\u00e8   latte???  ") == "caffe latte"

    monkeypatch.setenv("EVAL_NORMALIZE_REMOVE_ACCENTS", "0")
    assert build.normalize_question("  Caff\u00e8   latte???  ") == "caff\u00e8 latte"


def test_determine_skip_reason_priority() -> None:
    assert build._determine_skip_reason("", has_gold=True, malformed_json=False, contexts=["ctx"]) == "question_not_found"
    assert build._determine_skip_reason("q", has_gold=True, malformed_json=True, contexts=["ctx"]) == "malformed_json"
    assert build._determine_skip_reason("q", has_gold=True, malformed_json=False, contexts=[]) == "empty_context"
    assert build._determine_skip_reason("q", has_gold=False, malformed_json=False, contexts=["ctx"]) == "no_gold"
    assert build._determine_skip_reason("q", has_gold=True, malformed_json=False, contexts=["ctx"]) == ""


def test_validate_output_schema_warns(caplog: pytest.LogCaptureFixture) -> None:
    row = {
        "run_dir": "r1",
        "strategy": "default",
        "framework": "graphrag",
        "model_id": "m1",
        "question": "q",
        "answer": "a",
        "contexts_json": "[]",
        "retrieved_triples_json": "[]",
        "retrieved_entities_json": "[]",
        "expected_entities_json": "[]",
        "gold_triples_json": "[]",
        "has_gold_match": "1",
        "skip_reason": "",
    }

    caplog.set_level(logging.WARNING, logger="graphrag")
    build._validate_output_schema([row], min_coverage=0.95)

    assert "coverage below threshold" in caplog.text


def test_precision_and_recall_with_k() -> None:
    relevant = ["a", "b"]
    retrieved = ["x", "a", "b"]

    assert metrics.precision_at_k(relevant, retrieved, k=2) == pytest.approx(0.5)
    assert metrics.recall_at_k(relevant, retrieved, k=2) == pytest.approx(0.5)
    assert metrics.precision_at_k(relevant, retrieved, k=None) == pytest.approx(2.0 / 3.0)
    assert metrics.recall_at_k(relevant, retrieved, k=None) == pytest.approx(1.0)


def test_rank_aware_metrics() -> None:
    relevant = ["a", "b"]
    retrieved = ["x", "a", "b"]

    assert metrics.hit_at_k(relevant, retrieved, k=1) == 0.0
    assert metrics.hit_at_k(relevant, retrieved, k=2) == 1.0
    assert metrics.ndcg_at_k(relevant, retrieved, k=3) == pytest.approx(0.6934, rel=1e-3)
    assert metrics.mrr(relevant, retrieved) == pytest.approx(0.5)
    assert metrics.mean_average_precision(relevant, retrieved) == pytest.approx((0.5 + (2.0 / 3.0)) / 2.0)


def test_bootstrap_ci_is_reproducible() -> None:
    scores = [0.1, 0.2, 0.4, 0.8]

    first = metrics.bootstrap_ci(scores, n_bootstrap=250, ci=0.95, seed=7)
    second = metrics.bootstrap_ci(scores, n_bootstrap=250, ci=0.95, seed=7)

    assert first == second
    assert first[0] <= first[1]
    assert 0.1 <= first[0] <= 0.8
    assert 0.1 <= first[1] <= 0.8


def test_aggregate_includes_segment_rows() -> None:
    raw_rows = [
        {
            "run_dir": "r1",
            "model_id": "m1",
            "framework": "graphrag",
            "strategy": "default",
            "question": "q1",
            "question_type": "factoid",
            "latency_ms": "10",
            "has_gold_match": "1",
            "skip_reason": "",
            "expected_entities_json": "[\"a\", \"b\"]",
            "retrieved_entities_json": "[\"a\", \"b\"]",
            "gold_triples_json": "[{\"subject\":\"a\",\"predicate\":\"R\",\"object\":\"b\"}]",
            "retrieved_triples_json": "[{\"subject\":\"a\",\"predicate\":\"R\",\"object\":\"b\"}]",
        },
        {
            "run_dir": "r1",
            "model_id": "m1",
            "framework": "graphrag",
            "strategy": "default",
            "question": "q2",
            "question_type": "relation",
            "latency_ms": "20",
            "has_gold_match": "1",
            "skip_reason": "",
            "expected_entities_json": "[\"x\", \"y\"]",
            "retrieved_entities_json": "[\"x\"]",
            "gold_triples_json": "[{\"subject\":\"x\",\"predicate\":\"R\",\"object\":\"y\"}]",
            "retrieved_triples_json": "[{\"subject\":\"x\",\"predicate\":\"R\",\"object\":\"y\"}]",
        },
    ]

    row_metrics = [metrics._compute_row_metrics(row, k=None) for row in raw_rows]
    summary = metrics._aggregate(
        rows=row_metrics,
        n_bootstrap=50,
        ci=0.95,
        seed=42,
        question_type_available=True,
    )

    segments = {item["segment"] for item in summary}
    assert "global" in segments
    assert "factoid" in segments
    assert "relation" in segments

    first = summary[0]
    assert "precision_at_k_mean" in first
    assert "precision_at_k_ci_lower" in first
    assert "precision_at_k_ci_95_upper" in first
    assert "precision_at_k_n_samples" in first


def test_to_ragas_sample_casts_contexts_to_str_list() -> None:
    row = {
        "question": "Q",
        "answer": "A",
        "ground_truth": "G",
        "contexts": [" context one ", 42, "   "],
    }

    sample = ragas_eval._to_ragas_sample(row)
    assert sample["contexts"] == ["context one", "42"]


def test_extract_scores_from_dict_payload() -> None:
    result = {"scores": {"faithfulness": "0.75", "answer_relevancy": 0.5, "bad": "nan"}}

    scores = ragas_eval._extract_scores(result)
    assert scores["faithfulness"] == pytest.approx(0.75)
    assert scores["answer_relevancy"] == pytest.approx(0.5)
    assert "bad" not in scores


def test_coerce_score_filters_non_finite() -> None:
    assert ragas_eval._coerce_score("0.2") == pytest.approx(0.2)
    assert ragas_eval._coerce_score("nan") is None
    assert ragas_eval._coerce_score("inf") is None
