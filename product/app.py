#!/usr/bin/env python3
"""Streamlit front-end for domain-expert demo sessions.

Browser UI over the same GraphRAG agent used by product/console.py — text box,
Invio/Invia submits, spinner while the agent works, answer with sources.
Every exchange is logged to the same JSONL format under artifacts/demo_sessions/.

Usage (on the server):
    conda run -n graphllm streamlit run product/app.py --server.address 0.0.0.0 --server.port 8501

Then from your machine, open an SSH tunnel and browse to localhost:8501:
    ssh -L 8501:localhost:8501 <user>@<server>
    # apri http://localhost:8501 nel browser locale
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sys
import time
import traceback
import uuid
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# Streamlit and `python product/console.py` both put this file's own directory
# on the path, not the repository root, so `product.config` needs it added.
sys.path.insert(0, str(ROOT))

from graphrag.agent.core import KGRAGAgent  # noqa: E402
from graphrag.agent.memory import ConversationMemory  # noqa: E402

# Answer-quality settings and the agent itself come from the module shared with
# the console demo: the two surfaces are documented as the same product and
# must not drift apart again.
from product.config import (  # noqa: E402
    LOG_DIR,
    MEMORY,
    SHOW_FULL_ANSWER,
    STRATEGY,
    VLLM_ENDPOINTS,
    build_demo_agent,
    probe_vllm_endpoints,
)

logger = logging.getLogger("expert_demo")
# Separates the prose body from the raw evidence block in stored messages;
# the renderer shows what follows inside a monospace expander so triple IDs
# and <doc.pdf> references are not parsed as Markdown links/HTML.
EVIDENCE_MARKER = "\n\n%%EVIDENZE%%\n"
try:  # pragma: no cover - depends on the installed driver
    from neo4j.exceptions import ServiceUnavailable, SessionExpired

    _GRAPH_OUTAGE_EXCEPTIONS: tuple[type[BaseException], ...] = (
        ServiceUnavailable,
        SessionExpired,
    )
except Exception:  # pragma: no cover - driver without these names
    _GRAPH_OUTAGE_EXCEPTIONS = ()

# Rebuilding the agent re-indexes the text corpus, so a burst of questions
# arriving during an outage must not each pay for it. One rebuild per minute is
# enough: after the first, the others find the rebuilt agent in the cache.
_FAILOVER_COOLDOWN_SEC = 60.0
_last_failover_at = 0.0

# Shown on an answer built without the cross-lingual channel, because the
# encoder was unreachable. The product degrades instead of failing (see
# product/config.py); this is the half that keeps the degradation honest.
# Shown when memory rewrote the question before retrieval. The expert reads an
# answer that went somewhere they did not ask about and has no way to see why;
# the rewrite was logged and never displayed. Only when it actually changed:
# on most turns it is the question as typed, and saying so every time would
# train the reader to skip the line that matters.
def _rewrite_notice(question: str, retrieval_question: str) -> str:
    typed = " ".join(str(question or "").split())
    used = " ".join(str(retrieval_question or "").split())
    if not used or used.casefold() == typed.casefold():
        return ""
    return f"\n\n*Cercato nei documenti come: «{used}»*"


DEGRADED_NOTICE = (
    "\n\n---\n*Nota: il canale di ricerca cross-lingua non era disponibile per "
    "questa domanda. La risposta usa solo la ricerca testuale e per parole "
    "chiave, quindi può essere meno completa — soprattutto se la domanda è in "
    "una lingua diversa da quella dei documenti.*"
)


@st.cache_resource(show_spinner=False)
def _available_models() -> dict[str, tuple[str, str]]:
    """Probe the configured vLLM endpoints once per process."""
    return probe_vllm_endpoints()


def _configure_logging() -> None:
    """Send the engine's own warnings to a file that survives the session.

    `graphrag` used to be pinned to ERROR here, which silenced exactly the
    messages that report silent degradation — "embedding endpoint unavailable,
    vector channel skipped", "fulltext index disabled, falling back", "answer
    language mismatch", "discarding an implausible rewrite". A session that
    answered badly was then indistinguishable from a healthy one after the fact.

    The file is separate from streamlit.log because that one is mostly Neo4j
    deprecation notices, and a warning worth reading was lost in it.
    """
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    engine = logging.getLogger("graphrag")
    engine.setLevel(logging.WARNING)

    # Streamlit re-runs this module on every interaction, so attaching the
    # handler unguarded would multiply it by the number of clicks.
    if any(getattr(h, "_demo_handler", False) for h in engine.handlers):
        return
    # Beside streamlit.log, not beside the session transcripts: this is
    # operational output, while LOG_DIR holds what people asked and were told
    # and is meant to become access-restricted.
    run_log_dir = Path(os.environ.get("DEMO_LOG_DIR_RUNTIME", ROOT / "artifacts" / "demo_logs"))
    run_log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(run_log_dir / "graphrag.log", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    handler._demo_handler = True  # type: ignore[attr-defined]
    engine.addHandler(handler)


@st.cache_resource(show_spinner="Avvio in corso (connessione al grafo e indice testi)...")
def _load_agent(base_url: str, model_id: str) -> tuple[KGRAGAgent, str, str]:
    _configure_logging()
    os.chdir(ROOT)
    agent, graph_label = build_demo_agent(base_url, model_id)
    return agent, model_id, graph_label


def _session_log_path() -> Path:
    if "session_log" not in st.session_state:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        st.session_state.session_log = LOG_DIR / f"session_{dt.datetime.now():%Y%m%d_%H%M%S}.jsonl"
    return st.session_state.session_log


def _record_feedback(
    turn_id: str, chat_id: str, verdict: str = "", note: str = ""
) -> None:
    """Append one rating, or one note, to the log the turn was written to.

    The demo is the instrument the expert was given to judge the answers, and it
    collected nothing: searching product/ for feedback, rating or voto found
    nothing at all, so the only quality signal was whatever they remembered to
    say out loud. Ratings live in the same file as the turns they rate, so the
    two are read together and nothing new has to be kept in sync.

    A vote and a note are separate lines and a note carries no `feedback` key,
    because the log is append-only and a note that repeated the vote would be
    counted as a second vote. Reading it: lines with `feedback` are votes, last
    one per `turn_id` wins; lines with `note` are comments on that turn.
    """
    row: dict[str, object] = {
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "chat_id": chat_id,
        "turn_id": turn_id,
    }
    if note:
        row["note"] = note
    else:
        row["feedback"] = verdict
    with _session_log_path().open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# A chat is named after its first question; until then it needs a placeholder
# that cannot be confused with the button that creates one.
NEW_CHAT_TITLE = "(vuota)"


def _new_chat() -> str:
    """Start an empty conversation and make it current.

    Each chat owns its transcript and its own memory: two threads of questions
    must not resolve each other's follow-ups. Streamlit keeps this per browser
    session, so a second tab is a second independent set of chats — and the
    only way to have two questions running at the same time, since one session
    processes one question at a time.
    """
    chat_id = uuid.uuid4().hex[:8]
    st.session_state.chats[chat_id] = {
        "title": NEW_CHAT_TITLE,
        "messages": [],
        "memory": ConversationMemory() if MEMORY else None,
    }
    st.session_state.chat_order.append(chat_id)
    st.session_state.current_chat = chat_id
    return chat_id


def _init_chats() -> None:
    if "chats" not in st.session_state:
        st.session_state.chats = {}
        st.session_state.chat_order = []
        _new_chat()
    # Ratings already given in this browser session, so the buttons can show
    # what was recorded. The record itself is the JSONL line, not this: a
    # reload loses the highlight, never the feedback.
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}


def _current_chat() -> dict:
    return st.session_state.chats[st.session_state.current_chat]


def _clear_chat(chat: dict) -> None:
    """Empty a conversation without deleting it (the 'Azzera' button)."""
    chat["messages"].clear()
    chat["title"] = NEW_CHAT_TITLE
    if chat["memory"] is not None:
        chat["memory"].reset()


def _delete_chat(chat_id: str) -> None:
    st.session_state.chats.pop(chat_id, None)
    st.session_state.chat_order.remove(chat_id)
    if not st.session_state.chat_order:
        _new_chat()
    elif st.session_state.current_chat == chat_id:
        st.session_state.current_chat = st.session_state.chat_order[-1]


def _chat_label(text: str, max_chars: int = 30) -> str:
    """Name a conversation after its first question."""
    label = " ".join(str(text or "").split())
    if len(label) <= max_chars:
        return label
    return label[:max_chars].rsplit(" ", 1)[0] + "…"


def _is_graph_outage(exc: BaseException) -> bool:
    """True when this failure is the graph becoming unreachable, not a bug.

    The whole chain is inspected because the driver's exception reaches here
    wrapped by whatever re-raised it, and a failover decision taken on the
    outermost type alone would miss the case it exists for.
    """
    if not _GRAPH_OUTAGE_EXCEPTIONS:
        return False
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, _GRAPH_OUTAGE_EXCEPTIONS):
            return True
        current = current.__cause__ or current.__context__
    return False


def _rebuild_agent(base_url: str, model_id: str) -> tuple[KGRAGAgent, str, str] | None:
    """Drop the cached agent and build a new one, or None if that failed too.

    `build_kg_manager` already falls back from the suspended Aura instance to
    the local mirror — but it only runs while the agent is being built, and the
    agent is a cached resource built once per process. An instance that
    suspended *after* the demo started therefore killed every later question,
    with a working mirror one rebuild away and nothing to trigger it.
    """
    global _last_failover_at
    now = time.monotonic()
    if now - _last_failover_at < _FAILOVER_COOLDOWN_SEC:
        # Another question already rebuilt; use what it left in the cache.
        try:
            return _load_agent(base_url, model_id)
        except Exception:  # noqa: BLE001 - the caller reports the original failure
            return None
    _last_failover_at = now
    logger.warning("Graph unreachable mid-session: rebuilding the agent once.")
    try:
        _load_agent.clear()
        return _load_agent(base_url, model_id)
    except Exception as exc:  # noqa: BLE001 - no graph at all is the caller's problem
        logger.error("Rebuild after the graph outage failed too: %s", exc)
        return None


def _vector_skips(agent: KGRAGAgent) -> int:
    """How many times the cross-lingual channel has been skipped so far.

    Two independent causes, and the reader of an answer cannot tell them
    apart nor should have to: the encoder is unreachable (counted by the
    retriever) or the vector index cannot be queried (counted by the graph
    manager). Either way the answer in front of them was built without that
    channel, so both feed the same notice.
    """
    retriever = getattr(agent, "kg_retriever", None)
    store = getattr(retriever, "kg_store", None)
    return int(getattr(retriever, "vector_skips", 0) or 0) + int(
        getattr(store, "vector_skips", 0) or 0
    )


def _ask(
    agent: KGRAGAgent,
    model_id: str,
    question: str,
    turn_id: str = "",
    base_url: str = "",
    memory: ConversationMemory | None = None,
    chat_id: str = "",
    graph_label: str = "",
) -> str:
    started = time.perf_counter()
    record: dict[str, object] = {
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "strategy": STRATEGY,
        "model_id": model_id,
        # Which graph answered. Without it a session served by the local mirror
        # during an Aura outage reads exactly like a healthy one.
        "graph_label": graph_label,
        # One log per browser session, several conversations inside it: the id
        # is what separates two parallel threads of questions after the fact.
        "chat_id": chat_id,
        # What a feedback line points at. Without it a rating could only be
        # matched to a turn by timestamp and question text, which stops working
        # the moment the same question is asked twice.
        "turn_id": turn_id,
    }
    skips_before = _vector_skips(agent)
    try:
        try:
            result = agent.invoke(question, memory=memory)
        except Exception as exc:  # noqa: BLE001 - only a graph outage is handled here
            if not (base_url and _is_graph_outage(exc)):
                raise
            rebuilt = _rebuild_agent(base_url, model_id)
            if rebuilt is None:
                raise
            # The caption above still names the old graph; the st.rerun() at the
            # end of the question redraws it from the rebuilt agent.
            agent, _, record["graph_label"] = rebuilt
            record["graph_failover"] = True
            # Counters belong to the agent, and this one is new.
            skips_before = _vector_skips(agent)
            # memory.observe() runs after a successful graph.invoke, so the
            # failed attempt left no turn behind and this is not a double count.
            result = agent.invoke(question, memory=memory)
        answer = str(result.get("answer", "")).strip()
        elapsed = time.perf_counter() - started
        record["answer"] = answer
        record["latency_s"] = round(elapsed, 2)
        # What the answer was actually built from. A thin answer has two very
        # different causes — the gate refused, or retrieval came back empty —
        # and without these counts the log cannot tell them apart.
        record["out_of_scope"] = bool(result.get("out_of_scope"))
        record["insufficient"] = bool(result.get("insufficient_answer"))
        record["n_triples"] = len(result.get("kg_triples") or [])
        record["n_nodes"] = len(result.get("retrieved_nodes") or [])
        record["n_text_sources"] = len(result.get("retrieved_text_sources") or [])
        # WP7: the question actually sent to retrieval, and what resolved it.
        # Logged separately from `question` so a rewrite that hurt the answer
        # can be recognised as such after the session.
        if memory is not None:
            record["follow_up"] = bool(result.get("follow_up"))
            record["retrieval_question"] = result.get("retrieval_question", question)
            record["memory_entities"] = result.get("memory_entities", [])
        # Phantom-reference rate per model: the WP1 acceptance metric, and the
        # number that will compare Qwen2.5-32B with Qwen3-30B on hallucination.
        citation_report = result.get("citation_report")
        if isinstance(citation_report, dict):
            record["citation_report"] = citation_report
        # With cite_evidence the source list is part of the answer body and stays
        # inline; only the legacy triple dump goes into the collapsed expander.
        body, sep, evidence = answer.partition("\nVerifica nel grafo:")
        shown = body.strip() + f"\n\n*[{elapsed:.0f}s]*"
        # A degraded answer has to say so. Per-question, not per-session: the
        # encoder can come back, and an answer given while it was down is worth
        # less than the one before it and the one after it.
        # Before the degradation notice: this one explains the answer above it,
        # the other one qualifies it.
        shown += _rewrite_notice(question, str(result.get("retrieval_question") or ""))
        record["vector_degraded"] = _vector_skips(agent) > skips_before
        if record["vector_degraded"]:
            shown += DEGRADED_NOTICE
        if sep and SHOW_FULL_ANSWER:
            shown += EVIDENCE_MARKER + evidence.strip()
    except Exception as exc:  # noqa: BLE001 - UI must survive any failure
        elapsed = time.perf_counter() - started
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["latency_s"] = round(elapsed, 2)
        logger.error("Question failed: %s\n%s", exc, traceback.format_exc())
        shown = "Si è verificato un problema tecnico con questa domanda. Riprova, magari riformulandola."
    with _session_log_path().open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return shown


st.set_page_config(page_title="AI — Economia Circolare del Cibo", page_icon="")
st.title("AI - Circular Food Economy")
st.caption("Scrivi una domanda e premi Invio. Le risposte citano le fonti quando disponibili.")

models = _available_models()
if not models:
    st.error("Nessun server vLLM raggiungibile (DEMO_VLLM_ENDPOINTS) e VLLM_MODEL_NAME/VLLM_BASE_URL mancanti.")
    st.stop()

env_base_url = os.environ.get("VLLM_BASE_URL", "").rstrip("/")
labels = list(models)
default_index = next(
    (i for i, lbl in enumerate(labels) if models[lbl][0] == env_base_url), 0
)
_init_chats()

with st.sidebar:
    choice = st.selectbox("Modello", labels, index=default_index)

    st.divider()
    st.markdown("**Conversazioni**")
    if st.button("+ Nuova chat", use_container_width=True):
        _new_chat()
        st.rerun()
    for chat_id in list(st.session_state.chat_order):
        entry = st.session_state.chats[chat_id]
        is_current = chat_id == st.session_state.current_chat
        if st.button(
            entry["title"],
            key=f"select_{chat_id}",
            use_container_width=True,
            type="primary" if is_current else "secondary",
        ):
            st.session_state.current_chat = chat_id
            st.rerun()

    chat = _current_chat()
    st.divider()
    clear_col, delete_col = st.columns(2)
    if clear_col.button("Azzera", use_container_width=True):
        _clear_chat(chat)
        st.rerun()
    if delete_col.button(
        "Elimina",
        use_container_width=True,
        disabled=len(st.session_state.chat_order) == 1,
    ):
        _delete_chat(st.session_state.current_chat)
        st.rerun()

    if MEMORY and chat["memory"] is not None:
        active = chat["memory"].seed_entities()
        if active:
            st.caption("Argomenti in memoria: " + ", ".join(active))
        else:
            st.caption("Nessun argomento attivo.")

base_url, model_id = models[choice]

try:
    agent, model_id, graph_label = _load_agent(base_url, model_id)
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()
st.caption(f"strategia: {STRATEGY} | modello: {model_id} | grafo: {graph_label}")

def _render(content: str) -> None:
    body, sep, evidence = content.partition(EVIDENCE_MARKER)
    st.markdown(body)
    if sep:
        with st.expander("Verifica nel grafo (evidenze)"):
            st.code(evidence, language=None)


def _feedback_row(turn_id: str, chat_id: str) -> None:
    """Two buttons and an optional note, under one answer.

    Rendered from the history loop rather than beside the fresh answer, so a
    reader who changes their mind three questions later can still say so.
    """
    given = st.session_state.feedback.get(turn_id)
    up, down, said = st.columns([1, 1, 10])
    if up.button("👍", key=f"up_{turn_id}", help="Risposta utile"):
        st.session_state.feedback[turn_id] = "up"
        _record_feedback(turn_id, chat_id, "up")
        st.rerun()
    if down.button("👎", key=f"down_{turn_id}", help="Risposta sbagliata o inutile"):
        st.session_state.feedback[turn_id] = "down"
        _record_feedback(turn_id, chat_id, "down")
        st.rerun()
    if given:
        said.caption("Grazie, registrato." if given == "up" else "Registrato: cosa non andava?")
        with st.expander("Aggiungi una nota", expanded=False):
            note = st.text_area(
                "Nota", key=f"note_{turn_id}", label_visibility="collapsed",
                placeholder="Che cosa mancava, o che cosa era sbagliato?",
            )
            if st.button("Salva nota", key=f"savenote_{turn_id}") and note.strip():
                _record_feedback(turn_id, chat_id, note=note.strip())
                st.session_state.feedback[turn_id] = f"{given}+nota"
                st.rerun()


for message in chat["messages"]:
    # Three items since feedback was added; older entries in a session that was
    # already open when the app reloaded still have two.
    role, content = message[0], message[1]
    turn_id = message[2] if len(message) > 2 else ""
    with st.chat_message(role):
        _render(content)
        if role == "assistant" and turn_id:
            _feedback_row(turn_id, st.session_state.current_chat)

question = st.chat_input("Scrivi qui la tua domanda...")
if question:
    if not chat["messages"]:
        chat["title"] = _chat_label(question)
    chat["messages"].append(("user", question, ""))
    turn_id = uuid.uuid4().hex[:12]
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Sto pensando..."):
            answer = _ask(
                agent,
                model_id,
                question,
                turn_id=turn_id,
                base_url=base_url,
                memory=chat["memory"],
                chat_id=st.session_state.current_chat,
                graph_label=graph_label,
            )
        # Appended before rendering: a rerun raised inside _render (a sidebar
        # click during the spinner, a browser reconnect) used to drop the answer
        # from the transcript while the JSONL row was already written and
        # memory.observe() had already run, leaving the three disagreeing.
        chat["messages"].append(("assistant", answer, turn_id))
        _render(answer)
    # The sidebar rendered before the answer existed: rerun so the chat list
    # shows the new title and the updated topics.
    st.rerun()
