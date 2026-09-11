"""The demo's own logic, underneath the widgets.

`product/app.py` is the interface the expert actually uses and nothing in it
had ever run under a test. Most of the file paints; what is pinned here is the
part that decides — which conversation a question belongs to, what gets
written to the session log, when a failure is the graph rather than the
question, and what the streaming worker does with an exception raised off the
script thread.

Streamlit is imported but never rendered: `st.session_state` accepts writes
outside a script run, so the state functions are exercised directly.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import pytest

# `product/app.py` is a Streamlit script, and importing it runs its body —
# including `_configure_logging`, which calls `logging.basicConfig`, pins the
# "graphrag" and "expert_demo" loggers to WARNING and attaches a file handler,
# for the whole process. Right for a demo, wrong for a test session: every
# test after this one that reads a log would see nothing. The state is taken
# before the import and put back after it.
_ROOT = logging.getLogger()
_SAVED_HANDLERS = list(_ROOT.handlers)
_SAVED_ROOT_LEVEL = _ROOT.level
_SAVED_LEVELS = {
    name: logging.getLogger(name).level for name in ("graphrag", "expert_demo")
}

from product import app  # noqa: E402

_ROOT.handlers[:] = _SAVED_HANDLERS
_ROOT.setLevel(_SAVED_ROOT_LEVEL)
for _name, _level in _SAVED_LEVELS.items():
    logging.getLogger(_name).setLevel(_level)


@pytest.fixture(autouse=True)
def clean_state(monkeypatch, tmp_path):
    app.st.session_state.clear()
    monkeypatch.setattr(app, "LOG_DIR", tmp_path)
    yield
    app.st.session_state.clear()


def _rows(path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- conversations ---------------------------------------------------------


def test_a_new_chat_becomes_the_current_one():
    app._init_state()

    chat_id = app._new_chat()

    assert app.st.session_state.current_chat == chat_id
    assert app._current_chat()["messages"] == []


def test_each_chat_keeps_its_own_transcript():
    # Two threads of questions must not resolve each other's follow-ups.
    app._init_state()
    first = app.st.session_state.current_chat
    app.st.session_state.chats[first]["messages"].append({"question": "prima"})

    second = app._new_chat()

    assert app.st.session_state.chats[second]["messages"] == []
    assert len(app.st.session_state.chats[first]["messages"]) == 1


def test_chats_keep_the_order_they_were_opened_in():
    app._init_state()
    first = app.st.session_state.current_chat
    second = app._new_chat()

    assert app.st.session_state.chat_order == [first, second]


def test_deleting_the_current_chat_falls_back_to_the_most_recent():
    app._init_state()
    first = app.st.session_state.current_chat
    second = app._new_chat()

    app._delete_chat(second)

    assert app.st.session_state.current_chat == first
    assert second not in app.st.session_state.chats


def test_deleting_another_chat_leaves_the_current_one_alone():
    app._init_state()
    first = app.st.session_state.current_chat
    second = app._new_chat()

    app._delete_chat(first)

    assert app.st.session_state.current_chat == second


def test_deleting_the_last_chat_opens_a_fresh_one():
    # The page has nothing to render without a current chat.
    app._init_state()
    only = app.st.session_state.current_chat

    app._delete_chat(only)

    assert app.st.session_state.chat_order
    assert app.st.session_state.current_chat != only


def test_initialising_twice_does_not_discard_the_conversation():
    app._init_state()
    chat_id = app.st.session_state.current_chat

    app._init_state()

    assert app.st.session_state.current_chat == chat_id


def test_initialising_sets_the_defaults_the_page_reads():
    app._init_state()

    assert app.st.session_state.feedback == {}
    assert app.st.session_state.ui_lang in app.ui.LANGUAGES
    assert app.st.session_state.confirm_delete == ""


# --- naming a conversation -------------------------------------------------


def test_a_short_question_names_the_chat_as_it_is():
    assert app._chat_label("Cos'e' la scotta?") == "Cos'e' la scotta?"


def test_a_long_question_is_cut_at_a_word_boundary():
    label = app._chat_label(
        "Quali sono le tre componenti principali del capitale secondo il framework"
    )

    assert label.endswith("…")
    assert len(label) <= 31
    assert not label.rstrip("…").endswith(" ")


def test_whitespace_is_collapsed_before_naming():
    assert app._chat_label("  due   parole  ") == "due parole"


@pytest.mark.parametrize("value", ["", None, "   "])
def test_a_chat_with_no_question_yet_has_an_empty_name(value):
    assert app._chat_label(value) == ""


# --- the session log -------------------------------------------------------


def test_one_log_file_per_browser_session():
    first = app._session_log_path()
    second = app._session_log_path()

    assert first == second
    assert first.name.startswith("session_")


def test_a_vote_is_one_line_naming_the_turn_it_rates():
    app._record_feedback(turn_id="t1", chat_id="c1", verdict="up")

    row = _rows(app._session_log_path())[0]
    assert row["kind"] == "feedback"
    assert row["surface"] == "streamlit"
    assert (row["turn_id"], row["chat_id"], row["feedback"]) == ("t1", "c1", "up")


def test_a_note_is_a_separate_line_and_is_not_counted_as_a_vote():
    # The log is append-only: a note repeating the vote would be read as a
    # second vote.
    app._record_feedback(turn_id="t1", chat_id="c1", verdict="down")
    app._record_feedback(turn_id="t1", chat_id="c1", note="la risposta è generica")

    rows = _rows(app._session_log_path())
    assert len(rows) == 2
    assert "feedback" not in rows[1]
    assert rows[1]["note"] == "la risposta è generica"


def test_a_fixed_reason_is_recorded_alongside_the_free_text():
    # Free text says what went wrong for one reader; the fixed reason is the
    # only part that can be counted across readers.
    app._record_feedback(turn_id="t1", chat_id="c1", reason="generic", note="troppo vaga")

    row = _rows(app._session_log_path())[0]
    assert row["reason"] == "generic"
    assert row["note"] == "troppo vaga"
    assert "feedback" not in row


def test_a_reason_without_a_note_is_still_not_a_vote():
    app._record_feedback(turn_id="t1", chat_id="c1", reason="wrong")

    row = _rows(app._session_log_path())[0]
    assert row["reason"] == "wrong"
    assert "feedback" not in row


def test_the_last_vote_on_a_turn_is_the_one_that_counts():
    app._record_feedback(turn_id="t1", chat_id="c1", verdict="up")
    app._record_feedback(turn_id="t1", chat_id="c1", verdict="down")

    votes = [r for r in _rows(app._session_log_path()) if "feedback" in r]
    assert [v["feedback"] for v in votes] == ["up", "down"]


def test_every_line_says_which_surface_wrote_it():
    app._record_feedback(turn_id="t1", chat_id="c1", verdict="up")

    assert _rows(app._session_log_path())[0]["surface"] == "streamlit"


# --- telling a broken graph from a broken question -------------------------


def test_an_ordinary_failure_is_not_a_graph_outage():
    # An unreachable graph is nothing the person typing can fix; telling them
    # to rephrase sends them chasing their own question.
    assert app._is_graph_outage(ValueError("bad question")) is False


def test_a_driver_outage_wrapped_by_another_exception_is_still_found():
    if not app._GRAPH_OUTAGE_EXCEPTIONS:
        pytest.skip("neo4j exception classes unavailable")
    inner = app._GRAPH_OUTAGE_EXCEPTIONS[0]("unreachable")

    try:
        try:
            raise inner
        except Exception as exc:
            raise RuntimeError("answering failed") from exc
    except RuntimeError as wrapped:
        assert app._is_graph_outage(wrapped) is True


def test_a_cycle_in_the_exception_chain_does_not_hang():
    first = RuntimeError("a")
    second = RuntimeError("b")
    first.__cause__ = second
    second.__cause__ = first

    assert app._is_graph_outage(first) is False


# --- the degraded-channel counter ------------------------------------------


def test_skips_from_both_causes_reach_the_same_notice():
    # The encoder being unreachable and the vector index being unqueryable are
    # different causes; the reader cannot tell them apart and should not have to.
    class _Store:
        vector_skips = 2

    class _Retriever:
        vector_skips = 3
        kg_store = _Store()

    class _Agent:
        kg_retriever = _Retriever()

    assert app._vector_skips(_Agent()) == 5


def test_an_agent_that_counts_nothing_reports_zero():
    class _Agent:
        pass

    assert app._vector_skips(_Agent()) == 0


# --- streaming off the script thread ---------------------------------------


def _placeholder():
    class _P:
        def __init__(self) -> None:
            self.painted: list[str] = []
            self.captions: list[str] = []

        def caption(self, text: str) -> None:
            self.captions.append(text)

        def markdown(self, text: str) -> None:
            self.painted.append(text)

    return _P()


def test_the_answer_is_returned_once_the_worker_finishes():
    app._init_state()
    placeholder = _placeholder()

    def _run(sink):
        sink("La scotta ")
        sink("e' un residuo.")
        return {"answer": "La scotta e' un residuo."}

    assert app._stream(_run, placeholder) == {"answer": "La scotta e' un residuo."}


def test_a_failure_on_the_worker_thread_is_raised_on_the_script_thread():
    # Streamlit only repaints from the script thread, so an exception swallowed
    # in the worker would leave the page spinning with no error.
    app._init_state()

    def _run(sink):
        raise RuntimeError("vLLM down")

    with pytest.raises(RuntimeError, match="vLLM down"):
        app._stream(_run, _placeholder())


def test_a_retry_that_throws_away_its_draft_repaints_from_empty():
    app._init_state()
    placeholder = _placeholder()

    def _run(sink):
        sink("prima meta")
        time.sleep(0.15)  # past the repaint throttle, so the draft is painted
        sink(None)  # the retry discards what it had written
        sink("la risposta buona")
        time.sleep(0.15)
        return {"answer": "la risposta buona"}

    app._stream(_run, placeholder)

    assert any("prima meta" in frame for frame in placeholder.painted)
    assert "prima meta" not in placeholder.painted[-1]
    assert "la risposta buona" in placeholder.painted[-1]


def test_the_reader_is_told_something_is_happening_before_any_text():
    app._init_state()
    placeholder = _placeholder()

    app._stream(lambda sink: {"answer": ""}, placeholder)

    assert placeholder.captions


def test_a_run_that_returns_nothing_yields_an_empty_result():
    app._init_state()

    assert app._stream(lambda sink: None, _placeholder()) == {}
