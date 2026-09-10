"""The first thing typed into the demo is not a gold question.

"ciao", "chi sei?", "prova, sistema operativo?" — none of them has anything to
retrieve, and before this each one opened the session with a failure: an empty
context answered "non so", or a refusal that described documents without ever
saying what the assistant is. What is pinned here is both directions: the ones
answered without retrieval, and — the expensive error — the domain questions
that must still reach it.
"""

from __future__ import annotations

from typing import Any

import pytest

from graphrag.agent.core import KGRAGAgent
from graphrag.agent.smalltalk import detect_meta_question
from graphrag.config import AgentConfig
from graphrag.llm.prompts import PromptLibrary


def _agent(**overrides: Any) -> KGRAGAgent:
    base: dict[str, Any] = {"llm_warmup": False, "enable_cache": False}
    base.update(overrides)
    return KGRAGAgent(config=AgentConfig(**base), kg_retriever=None, llm=None)


@pytest.mark.parametrize(
    "question, language",
    [
        ("ciao", "it"),
        ("Ciao!", "it"),
        ("Buongiorno", "it"),
        ("chi sei?", "it"),
        ("Che cosa sai fare?", "it"),
        ("A cosa servi?", "it"),
        ("Che domande posso farti?", "it"),
        ("Prova, sistema operativo?", "it"),
        ("Che modello sei?", "it"),
        ("hello", "en"),
        ("who are you?", "en"),
        ("what can you do?", "en"),
        ("How do you work?", "en"),
        ("What model are you using?", "en"),
    ],
)
def test_a_question_about_the_assistant_is_recognised(question, language):
    assert detect_meta_question(question) == language


@pytest.mark.parametrize(
    "question",
    [
        # Every one of these has a subject the collection can be searched for.
        "Che cos'e' l'economia circolare applicata al cibo?",
        "Come funziona la simbiosi industriale?",
        "Ciao, cos'e' l'economia circolare del cibo?",
        "Quali sottoprodotti agroalimentari possono essere valorizzati, e come?",
        "Che modello di economia circolare descrivono i documenti?",
        "Quali sono i sistemi di packaging descritti?",
        "What do you know about rice bran?",
        "What can you do with grape pomace?",
        "Which model of circular economy do the documents describe?",
    ],
)
def test_a_domain_question_still_reaches_retrieval(question):
    assert detect_meta_question(question) is None


def test_the_greeting_never_reaches_the_gate_when_the_flag_is_off():
    """Default off: a measurement run must reach retrieval for every question."""
    agent = _agent(answer_meta_questions=False)

    assert agent._scope_gate({"question": "ciao"}) == {"in_domain": True}


def test_the_greeting_short_circuits_before_retrieval():
    agent = _agent(answer_meta_questions=True)

    verdict = agent._scope_gate({"question": "ciao"})

    assert verdict["in_domain"] is False
    assert verdict["meta_question"] is True
    assert verdict["meta_language"] == "it"


def test_the_introduction_answers_in_the_language_of_the_greeting():
    """Not through `_detect_query_language`, which reads "ciao" as English."""
    agent = _agent(
        answer_meta_questions=True,
        example_questions=("Che cos'e' l'economia circolare applicata al cibo?",),
    )

    out = agent._refuse_out_of_scope(
        {"question": "ciao", "meta_question": True, "meta_language": "it"}
    )

    assert out["out_of_scope"] is True
    assert out["meta_question"] is True
    assert "economia circolare applicata al cibo" in out["answer"]
    assert "Puoi chiedermi per esempio:" in out["answer"]


def test_the_introduction_offers_the_operator_s_questions():
    examples = ("Prima domanda?", "Seconda domanda?")

    text = PromptLibrary.identity_message("it", examples)

    for example in examples:
        assert f"- {example}" in text


def test_an_out_of_scope_question_still_gets_the_refusal():
    """The meta path must not swallow the gate's own terminal state."""
    agent = _agent(answer_meta_questions=True)

    out = agent._refuse_out_of_scope({"question": "Scrivimi una funzione python"})

    assert out["out_of_scope"] is True
    assert "meta_question" not in out
    assert "non rispondo" in out["answer"]
