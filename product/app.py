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


@st.cache_resource(show_spinner=False)
def _available_models() -> dict[str, tuple[str, str]]:
    """Probe the configured vLLM endpoints once per process."""
    return probe_vllm_endpoints()


@st.cache_resource(show_spinner="Avvio in corso (connessione al grafo e indice testi)...")
def _load_agent(base_url: str, model_id: str) -> tuple[KGRAGAgent, str, str]:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("graphrag").setLevel(logging.ERROR)
    os.chdir(ROOT)
    agent, graph_label = build_demo_agent(base_url, model_id)
    return agent, model_id, graph_label


def _session_log_path() -> Path:
    if "session_log" not in st.session_state:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        st.session_state.session_log = LOG_DIR / f"session_{dt.datetime.now():%Y%m%d_%H%M%S}.jsonl"
    return st.session_state.session_log


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


def _ask(
    agent: KGRAGAgent,
    model_id: str,
    question: str,
    memory: ConversationMemory | None = None,
    chat_id: str = "",
) -> str:
    started = time.perf_counter()
    record: dict[str, object] = {
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "strategy": STRATEGY,
        "model_id": model_id,
        # One log per browser session, several conversations inside it: the id
        # is what separates two parallel threads of questions after the fact.
        "chat_id": chat_id,
    }
    try:
        result = agent.invoke(question, memory=memory)
        answer = str(result.get("answer", "")).strip()
        elapsed = time.perf_counter() - started
        record["answer"] = answer
        record["latency_s"] = round(elapsed, 2)
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
st.title("Demo GraphRAG")
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
            st.caption("In conversazione su: " + ", ".join(active))
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


for role, content in chat["messages"]:
    with st.chat_message(role):
        _render(content)

question = st.chat_input("Scrivi qui la tua domanda...")
if question:
    if not chat["messages"]:
        chat["title"] = _chat_label(question)
    chat["messages"].append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Sto pensando..."):
            answer = _ask(
                agent,
                model_id,
                question,
                memory=chat["memory"],
                chat_id=st.session_state.current_chat,
            )
        _render(answer)
    chat["messages"].append(("assistant", answer))
    # The sidebar rendered before the answer existed: rerun so the chat list
    # shows the new title and the updated topics.
    st.rerun()
