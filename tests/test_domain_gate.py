"""Unit tests for the out-of-domain gate.

The demo answered "scrivi una funzione python che costruisca una rete neurale"
with a Keras function carrying three circular-economy PDFs as sources. The
cause was structural, not a bad model: the dense retriever has no score floor,
so `_grade` always saw evidence, and `grade_condition` routed every question to
`generate` after three rewrites. There was no terminal state that abstains.

These tests pin the pieces that must not silently regress: the refusal is
reachable and terminal, it is off by default so thesis baselines are untouched,
the gate reads the typed question rather than the memory-rewritten one, and a
gate failure fails open.

The scope wording itself is measured, not asserted here — see
`scripts/domain_gate/eval_domain_gate_llm.py` and `scripts/domain_gate/eval_domain_gate_heldout.py`.
"""

from __future__ import annotations

from graphrag.agent.core import KGRAGAgent
from graphrag.config import AgentConfig
from graphrag.llm.prompts import PromptLibrary


class _StubLLM:
    """Records what the gate was asked and answers with a fixed verdict."""

    def __init__(self, verdict: bool = True, explode: bool = False) -> None:
        self.verdict = verdict
        self.explode = explode
        self.seen: list[str] = []

    def classify_in_domain(self, question: str, config) -> bool:
        self.seen.append(question)
        if self.explode:
            raise RuntimeError("gate backend down")
        return self.verdict

    def warmup(self) -> None:  # pragma: no cover - never called, llm_warmup is off
        pass


def _agent(llm: _StubLLM | None, **config_kwargs) -> KGRAGAgent:
    config = AgentConfig(llm_warmup=False, **config_kwargs)
    return KGRAGAgent(config=config, kg_retriever=None, llm=llm)


def test_gate_refuses_out_of_domain_question_without_generating():
    llm = _StubLLM(verdict=False)
    agent = _agent(llm, enable_domain_gate=True)

    state = agent._scope_gate({"question": "scrivi una rete neurale in python"})
    assert state["in_domain"] is False

    refusal = agent._refuse_out_of_scope({"question": "scrivi una rete neurale in python"})
    assert refusal["out_of_scope"] is True
    assert refusal["answer"]
    # No evidence index exists on this path, so no source list can be attached.
    assert "Fonti" not in refusal["answer"]
    assert "Sources" not in refusal["answer"]


def test_refusal_answers_in_the_language_of_the_question():
    agent = _agent(_StubLLM(verdict=False), enable_domain_gate=True)

    italian = agent._refuse_out_of_scope({"question": "Qual è la capitale dell'Australia?"})
    english = agent._refuse_out_of_scope({"question": "What is the capital of Australia?"})

    assert "non rispondo" in italian["answer"]
    assert "not answering" in english["answer"]


def test_refusal_names_what_the_collection_covers():
    """A bare refusal gives the expert nothing to rephrase towards."""
    agent = _agent(_StubLLM(verdict=False), enable_domain_gate=True)

    answer = agent._refuse_out_of_scope({"question": "Chi ha scritto la Divina Commedia?"})["answer"]
    assert "La raccolta copre" in answer
    assert "circular economy" in answer


def test_custom_scope_replaces_the_default_in_prompt_and_refusal():
    agent = _agent(_StubLLM(verdict=False), enable_domain_gate=True, domain_scope="marine biology")

    answer = agent._refuse_out_of_scope({"question": "Chi ha scritto la Divina Commedia?"})["answer"]
    assert "marine biology" in answer
    assert "circular economy principles" not in answer

    rendered = str(PromptLibrary.domain_gate_prompt("marine biology"))
    assert "marine biology" in rendered


def test_gate_is_off_by_default_so_baselines_are_unchanged():
    llm = _StubLLM(verdict=False)
    agent = _agent(llm)

    assert agent._scope_gate({"question": "anything at all"})["in_domain"] is True
    assert llm.seen == []


def test_gate_failure_fails_open():
    """A broken classifier must not silence a working demo."""
    from graphrag.llm.manager import LLMManager

    manager = LLMManager.__new__(LLMManager)

    def _boom():
        raise RuntimeError("vLLM unreachable")

    manager.load_llm = _boom  # type: ignore[method-assign]
    assert manager.classify_in_domain("qualsiasi cosa", AgentConfig()) is True


def test_gate_reads_the_typed_question_not_the_memory_rewrite():
    """A follow-up inherits entities from the previous answer.

    Gating the rewritten question would let "and write me the code for it"
    through on the strength of the subject carried over from the turn before.
    """
    llm = _StubLLM(verdict=True)
    agent = _agent(llm, enable_domain_gate=True)

    agent._scope_gate(
        {
            "question": "e scrivimi il codice",
            "rewritten_question": "e scrivimi il codice sulla ciclicità del sistema alimentare",
        }
    )
    assert llm.seen == ["e scrivimi il codice"]


def test_follow_ups_skip_the_gate():
    """A terse follow-up carries no domain of its own.

    Measured: of ten realistic follow-ups, "e quindi?", "in che senso?" and
    "perché?" were classified out of domain. Refusing those breaks the
    conversation, and the topic they continue was gated when it was introduced.
    """
    llm = _StubLLM(verdict=False)
    agent = _agent(llm, enable_domain_gate=True)

    state = agent._scope_gate({"question": "e quindi?", "follow_up": True})

    assert state["in_domain"] is True
    assert llm.seen == []


def test_the_barest_continuations_skip_the_gate_without_memory():
    """Memory fills only from KG entities, so the flag cannot be the only test.

    On an answer built entirely from the text channel — the common case — no
    entity is observed, `follow_up` stays False, and these would be refused.
    """
    llm = _StubLLM(verdict=False)
    agent = _agent(llm, enable_domain_gate=True)

    for question in ("in che senso?", "perché?", "non ho capito", "fammi un esempio"):
        assert agent._scope_gate({"question": question})["in_domain"] is True
    assert llm.seen == []


def test_the_floor_does_not_exempt_a_short_out_of_domain_question():
    """Four words is a topic, three is a continuation.

    "Spiegami la relatività generale" is exactly four and must still be gated.
    """
    llm = _StubLLM(verdict=False)
    agent = _agent(llm, enable_domain_gate=True)

    assert agent._scope_gate({"question": "Spiegami la relatività generale"})["in_domain"] is False
    assert llm.seen == ["Spiegami la relatività generale"]


def test_a_first_question_is_still_gated():
    llm = _StubLLM(verdict=False)
    agent = _agent(llm, enable_domain_gate=True)

    state = agent._scope_gate({"question": "scrivi una rete neurale", "follow_up": False})

    assert state["in_domain"] is False
    assert llm.seen == ["scrivi una rete neurale"]


def test_empty_question_is_not_refused():
    llm = _StubLLM(verdict=False)
    agent = _agent(llm, enable_domain_gate=True)

    assert agent._scope_gate({"question": "   "})["in_domain"] is True
    assert llm.seen == []


def test_closing_prompt_line_matches_the_grounding_rule():
    """The last line is the one models follow; it must not contradict the top.

    The old wording told the model to call the context insufficient only when
    it was empty, which is exactly wrong when an out-of-domain question arrives
    with a full context of unrelated-but-factual chunks.
    """
    strict = str(PromptLibrary.answer_prompt(AgentConfig(allow_parametric_fallback=False)))
    assert "insufficient whenever it does not cover what was asked" in strict
    assert "insufficient only when context is empty" not in strict

    marked = str(PromptLibrary.answer_prompt(AgentConfig(allow_parametric_fallback=True)))
    assert "not in the retrieved evidence" in marked
