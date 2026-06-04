"""Unit tests for the GraphRAG inference path (retrieval + generation glue).

These cover the behaviour changed by the inference-focused fixes:
- A1: context compression keeps a realistic token budget (no over-trimming)
- A5: shared refusal detection
- A2: agent fallback only fires on genuine refusal / ungrounded sparse answers
- A4: triple ranker weights reflect the signals actually available
- A3: experiment runner no longer emits always-constant metric columns
- A8: a single shared strategy-preset applier
"""

from __future__ import annotations

import pytest

from graphrag.agent.compression import ContextCompressor
from graphrag.config import AgentConfig
from graphrag.experiments.runner import ExperimentResult, ExperimentRunner
from graphrag.kg.retriever import KGRetriever
from graphrag.llm.refusal import looks_like_refusal
from graphrag.strategies import STRATEGY_PRESETS, apply_strategy


# --------------------------------------------------------------------------- #
# A1 - ContextCompressor
# --------------------------------------------------------------------------- #
def test_compressor_keeps_short_context_untrimmed():
    comp = ContextCompressor(max_tokens=1000)
    text = "x" * 3000  # ~750 estimated tokens (< 1000) -> untouched
    assert comp.compress(text) == text


def test_compressor_trims_only_when_over_budget_and_keeps_more_than_old_ratio():
    comp = ContextCompressor(max_tokens=1000)
    text = "x" * 8000  # ~2000 estimated tokens -> trimmed
    out = comp.compress(text)
    assert "[... context trimmed ...]" in out
    # With the corrected ratio the budget is ~4000 chars, far more than the
    # ~1333 chars the inverted ratio used to keep.
    assert len(out) > 3000


def test_compressor_ratio_not_inverted_regression():
    # Guard against re-introducing the inverted tokens-per-char ratio.
    assert ContextCompressor(max_tokens=1000).ratio <= 0.3


# --------------------------------------------------------------------------- #
# A5 - refusal detection
# --------------------------------------------------------------------------- #
def test_refusal_empty_is_refusal():
    assert looks_like_refusal("") is True
    assert looks_like_refusal("   ") is True


def test_refusal_phrase_detected_en_and_it():
    assert looks_like_refusal("The context is insufficient to answer.") is True
    assert looks_like_refusal("Mi dispiace, contesto insufficiente.") is True


def test_refusal_substantive_answer_not_flagged():
    answer = "Regulation 178/2002 establishes the European Food Safety Authority."
    assert looks_like_refusal(answer) is False


def test_refusal_common_words_not_flagged():
    # The old per-component heuristic mis-handled common words like
    # "context"/"information"/"analysis"; the phrase detector must not.
    answer = (
        "The available information and analysis in this context show that the "
        "policy affects food security indicators."
    )
    assert looks_like_refusal(answer) is False


# --------------------------------------------------------------------------- #
# A8 - shared strategy presets
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label", STRATEGY_PRESETS)
def test_apply_strategy_known_labels_return_config(label):
    assert isinstance(apply_strategy(AgentConfig(), label), AgentConfig)


def test_apply_strategy_text_only_disables_kg_channels():
    cfg = apply_strategy(AgentConfig(), "text_only")
    assert cfg.include_nodes is False
    assert cfg.include_triples is False
    assert cfg.include_subgraph is False
    assert cfg.include_shortest_path is False


def test_apply_strategy_subgraph_2hop_sets_min_two_hops():
    assert apply_strategy(AgentConfig(hops=1), "subgraph_2hop").hops >= 2


def test_apply_strategy_does_not_mutate_base():
    base = AgentConfig()
    apply_strategy(base, "text_only")
    assert base.include_nodes is True  # deep-copied, base untouched


def test_apply_strategy_unknown_raises():
    with pytest.raises(ValueError):
        apply_strategy(AgentConfig(), "does_not_exist")


# --------------------------------------------------------------------------- #
# A3 - experiment runner: no dead metric columns
# --------------------------------------------------------------------------- #
class _StubAgent:
    def invoke(self, question: str) -> dict:
        return {"answer": "ok", "latency_ms": 1.0, "kg_triples": [], "run_id": "r1"}


def test_experiment_result_has_no_dead_fields():
    fields = set(ExperimentResult.__dataclass_fields__)
    assert "confidence" not in fields
    assert "reflection_passed" not in fields


def test_runner_csv_has_no_dead_metric_columns(tmp_path):
    runner = ExperimentRunner(questions=["q1"])
    runner.run_agent(agent=_StubAgent(), label="default")
    csv_path = tmp_path / "results.csv"
    runner.export_csv(str(csv_path))
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert "confidence" not in header
    assert "reflection_passed" not in header


def test_runner_summary_stats_has_no_dead_metrics():
    runner = ExperimentRunner(questions=["q1"])
    runner.run_agent(agent=_StubAgent(), label="default")
    stats = runner.summary_stats()["default"]
    assert "avg_confidence" not in stats
    assert "reflection_pass_rate" not in stats


# --------------------------------------------------------------------------- #
# A2 - agent fallback gating + localization
# --------------------------------------------------------------------------- #
def _make_agent() -> "object":
    from graphrag.agent.core import KGRAGAgent

    return KGRAGAgent(config=AgentConfig())


def test_agent_keeps_grounded_answer_under_sparse_context():
    agent = _make_agent()
    query = "What does Regulation establish about EFSA?"
    context = "Regulation 178/2002 established EFSA."
    answer = (
        "The context and information here, by analysis, show that Regulation "
        "178/2002 established EFSA as the food safety authority."
    )
    # Old heuristic (>=3 meta words) would have replaced this; it must not now,
    # because the answer references salient terms.
    assert (
        agent._should_replace_with_fallback(
            answer=answer,
            query=query,
            context=context,
            triples=[],
            sparse_context=True,
        )
        is False
    )


def test_agent_replaces_genuine_refusal():
    agent = _make_agent()
    assert (
        agent._should_replace_with_fallback(
            answer="The context is insufficient to answer.",
            query="q",
            context="c",
            triples=[],
            sparse_context=True,
        )
        is True
    )


def test_agent_sparse_fallback_text_is_localized():
    from graphrag.agent.core import KGRAGAgent

    en = KGRAGAgent._build_sparse_fallback_answer(
        query="What is EFSA?", context="", triples=[], language="en"
    )
    assert "too sparse" in en
    assert "Limiti e fiducia" not in en

    it = KGRAGAgent._build_sparse_fallback_answer(
        query="Cos'e EFSA?", context="", triples=[], language="it"
    )
    assert "troppo scarno" in it


# --------------------------------------------------------------------------- #
# A4 - triple ranker
# --------------------------------------------------------------------------- #
def test_ranker_weights_reflect_available_signals():
    cfg = AgentConfig()
    assert cfg.ranker_weight_confidence == 0.0
    assert abs((cfg.ranker_weight_lexical + cfg.ranker_weight_mention) - 1.0) < 1e-9


def test_rank_triples_prefers_query_overlap():
    retriever = KGRetriever(kg_store=None, config=AgentConfig())
    triples = [
        {"subject": "Banana", "predicate": "GROWN_IN", "object": "Ecuador"},
        {"subject": "EFSA", "predicate": "ESTABLISHED", "object": "Authority"},
    ]
    ranked = retriever._rank_triples(triples, "What established EFSA?")
    assert ranked[0]["subject"] == "EFSA"
