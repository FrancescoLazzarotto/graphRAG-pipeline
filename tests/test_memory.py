"""Unit tests for intra-session conversational memory (WP7).

Covers `docs/demo_quality_plan_2026-07.md` §9. Two properties matter more than
the feature itself:

* **No regression when memory is off.** `invoke(question)` without a memory must
  reach the graph with exactly the state it reached before WP7 — gold runs and
  experiment baselines depend on it.
* **No false positives.** The follow-up detector must not fire on a
  self-contained question; the acceptance criterion is 0 hits on the 30 gold
  queries, which are all self-contained by construction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from graphrag.agent.core import KGRAGAgent
from graphrag.agent.memory import ConversationMemory, is_follow_up
from graphrag.config import AgentConfig

GOLD_PATH = Path(__file__).resolve().parents[1] / "evaluation/gold/gold_circular_v1.json"

# The real chain from the 2026-07-20 expert session: Q3 is self-contained, Q5
# borrows its subject ("vino") from the answer to Q4.
Q3 = (
    "Quali sono le 5 filiere della regione Piemonte in cui l'economia circolare "
    "per il cibo può trovare una buona espressione?"
)
Q5 = "Mi indichi le strategie nel settore vino individuate dalla ricerca?"


class _FakeOutput:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeModel:
    def __init__(self, *answers: str) -> None:
        self._answers = list(answers)
        self.prompts: list[Any] = []

    def invoke(self, payload: Any) -> _FakeOutput:
        self.prompts.append(payload)
        return _FakeOutput(self._answers.pop(0) if self._answers else "")


class _FakeLLM:
    def __init__(self, model: Any) -> None:
        self._model = model

    def load_llm(self) -> Any:
        return self._model


class _RecordingAgent(KGRAGAgent):
    """Agent whose graph records the initial state instead of running."""

    def _build_graph(self):  # type: ignore[override]
        agent = self

        class _Graph:
            def invoke(self, state: dict, config: dict | None = None) -> dict:
                agent.seen_state = dict(state)
                return {"answer": "risposta", "retrieved_nodes": [], "kg_triples": []}

        return _Graph()


def _agent(model: Any = None) -> _RecordingAgent:
    llm = _FakeLLM(model) if model is not None else None
    return _RecordingAgent(config=AgentConfig(), kg_retriever=None, llm=llm)


def _memory_with(*entities: str) -> ConversationMemory:
    memory = ConversationMemory()
    memory.observe(
        question="domanda precedente",
        answer=" ".join(entities),
        triples=[{"subject": name, "predicate": "REL", "object": "x"} for name in entities],
    )
    return memory


# --- follow-up detection --------------------------------------------------


def test_no_follow_up_on_the_first_turn():
    """Nothing to resolve a reference against yet."""
    assert is_follow_up("approfondisci", has_context=False) is False


def test_elliptical_request_is_a_follow_up():
    """The Q5 case: short imperative, no subject of its own."""
    assert is_follow_up(Q5) is True


def test_continuity_markers_fire_regardless_of_length():
    for question in (
        "approfondisci",
        "dimmi di più",
        "E invece per il latte?",
        "puoi approfondire il punto sulle filiere?",
        "tell me more about it",
    ):
        assert is_follow_up(question) is True, question


def test_self_contained_questions_are_left_alone():
    for question in (
        Q3,
        "Che cos'è SEeD?",
        "Dammi la definizione delle 3C della circular economy for food",
        "Mi dai un approfondimento sulle 5 filiere della regione Piemonte in cui "
        "l'economia circolare per il cibo può trovare una buona espressione?",
        "Cosa diceva Petrini su materia rinnovabile n. 33?",
    ):
        assert is_follow_up(question) is False, question


@pytest.mark.skipif(not GOLD_PATH.exists(), reason="gold set not available")
def test_no_false_positives_on_the_gold_set():
    """Acceptance criterion of §9.6: 0 hits on 30 self-contained questions."""
    queries = json.loads(GOLD_PATH.read_text(encoding="utf-8"))["queries"]
    fired = [q["query"] for q in queries if is_follow_up(q["query"])]

    assert fired == []


# --- memory state ---------------------------------------------------------


def test_entities_come_from_retrieval_not_from_the_answer_text():
    memory = ConversationMemory()
    memory.observe(
        question="Quali filiere?",
        answer="Le filiere sono vino e riso, e anche il turismo.",
        nodes=[{"text": "Vino"}],
        triples=[{"subject": "Riso", "predicate": "PART_OF", "object": "Filiere"}],
    )

    names = {item.name for item in memory.active_entities}
    assert names == {"Vino", "Riso", "Filiere"}
    # "turismo" is in the answer but was never retrieved: memory carries only
    # what the graph actually returned.
    assert "turismo" not in {name.lower() for name in names}


def test_last_answer_entities_are_the_ones_the_model_talked_about():
    memory = ConversationMemory()
    memory.observe(
        question="Quali filiere?",
        answer="La filiera del vino è la più matura.",
        nodes=[{"text": "Vino"}, {"text": "Acqua minerale"}],
    )

    assert memory.last_answer_entities == ["Vino"]


def test_a_name_inside_a_longer_word_is_not_a_mention():
    """The dominant false positive on the 2026-07 demo logs: "riso" in "risorse".

    A name promoted this way leads the seed ranking, so the follow-up gets
    rewritten around a topic the answer never discussed.
    """
    memory = ConversationMemory()
    memory.observe(
        question="q",
        answer="Le risorse idriche del sistema sono limitate.",
        nodes=[{"text": "Riso"}, {"text": "Tema"}],
    )

    assert memory.last_answer_entities == []


def test_an_elided_article_does_not_hide_a_mention():
    memory = ConversationMemory()
    memory.observe(
        question="q",
        answer="L'economia circolare parte dai sotto-prodotti.",
        nodes=[{"text": "Economia circolare"}, {"text": "Prodotti"}],
    )

    assert memory.last_answer_entities == ["Economia circolare", "Prodotti"]


def test_seed_entities_put_the_last_answer_first():
    memory = ConversationMemory()
    memory.observe(
        question="q1",
        answer="parla di Riso",
        nodes=[{"text": "Carne bovina"}, {"text": "Riso"}],
    )

    assert memory.seed_entities()[0] == "Riso"


def test_a_broader_name_absorbs_its_own_substring():
    """"Regione" next to "Regione Piemonte" wastes a slot and vaguens the rewrite."""
    memory = ConversationMemory()
    memory.observe(
        question="q",
        answer="",
        nodes=[{"text": "Regione Piemonte"}, {"text": "Regione"}, {"text": "Vino"}],
    )

    assert memory.seed_entities() == ["Regione Piemonte", "Vino"]


def test_the_specific_name_wins_even_when_the_broad_one_ranks_first():
    """Ranking decides the order, so absorption cannot depend on it."""
    memory = ConversationMemory()
    memory.observe(
        question="q",
        answer="",
        nodes=[{"text": "Regione"}, {"text": "Regione Piemonte"}, {"text": "Vino"}],
    )

    assert memory.seed_entities() == ["Regione Piemonte", "Vino"]


def test_absorption_needs_whole_words_not_a_substring():
    """"Riso" is not a broader "Risorsa": dropping it would lose a real topic."""
    memory = ConversationMemory()
    memory.observe(
        question="q",
        answer="",
        nodes=[{"text": "Risorsa"}, {"text": "Riso"}],
    )

    assert memory.seed_entities() == ["Risorsa", "Riso"]


def test_entities_decay_out_of_the_window():
    memory = ConversationMemory(window=2)
    memory.observe(question="q1", answer="", nodes=[{"text": "Vecchio argomento"}])
    for turn in range(2, 5):
        memory.observe(question=f"q{turn}", answer="", nodes=[{"text": "Nuovo"}])

    assert [item.name for item in memory.active_entities] == ["Nuovo"]


def test_reset_clears_the_topic():
    memory = _memory_with("Vino")
    memory.reset()

    assert memory.has_context() is False
    assert memory.seed_entities() == []
    assert memory.last_question == ""


def test_noise_is_kept_out_of_the_seed_list():
    memory = ConversationMemory()
    memory.observe(
        question="q",
        answer="",
        nodes=[{"text": "ab"}, {"text": "123"}, {"text": "x" * 80}, {"text": "Vino"}],
    )

    assert [item.name for item in memory.active_entities] == ["Vino"]


# --- wiring into the agent ------------------------------------------------


def test_without_memory_the_initial_state_is_unchanged():
    agent = _agent()
    agent.invoke(Q5)

    assert set(agent.seen_state) == {"question", "run_id", "rewrite_count"}
    assert agent.seen_state["question"] == Q5


def test_a_follow_up_reaches_retrieval_rewritten_and_generation_intact():
    """`_retrieve`/`_grade` read `rewritten_question`, `_generate` reads `question`."""
    model = _FakeModel(
        "Quali strategie di economia circolare sono individuate per la filiera "
        "del vino in Piemonte?"
    )
    agent = _agent(model)
    agent.invoke(Q5, memory=_memory_with("Vino", "Piemonte"))

    assert agent.seen_state["question"] == Q5
    assert agent.seen_state["rewritten_question"].startswith("Quali strategie")
    assert "Vino" in str(model.prompts[0])


def test_a_self_contained_question_never_calls_the_rewriter():
    model = _FakeModel("non dovrebbe servire")
    agent = _agent(model)
    agent.invoke(Q3, memory=_memory_with("Vino"))

    assert model.prompts == []
    assert "rewritten_question" not in agent.seen_state


def test_an_implausible_rewrite_is_discarded():
    """A model that answers instead of rewriting must not poison retrieval."""
    model = _FakeModel("Le strategie individuate sono molte. " * 30)
    agent = _agent(model)
    agent.invoke(Q5, memory=_memory_with("Vino"))

    assert "rewritten_question" not in agent.seen_state


def test_a_failing_rewrite_keeps_the_question():
    class _BrokenModel:
        def invoke(self, payload: Any) -> Any:
            raise ValueError("backend down")

    agent = _agent(_BrokenModel())
    agent.invoke(Q5, memory=_memory_with("Vino"))

    assert "rewritten_question" not in agent.seen_state


def test_the_rewrite_is_logged_next_to_the_original():
    model = _FakeModel("Quali strategie per la filiera del vino in Piemonte?")
    agent = _agent(model)
    result = agent.invoke(Q5, memory=_memory_with("Vino"))

    assert result["original_question"] == Q5
    assert result["retrieval_question"].startswith("Quali strategie")
    assert result["follow_up"] is True
    assert result["memory_entities"] == ["Vino"]


def test_the_turn_is_recorded_in_memory():
    memory = _memory_with("Vino")
    agent = _agent()
    agent.invoke(Q3, memory=memory)

    assert memory.turn == 2
    assert memory.last_question == Q3
