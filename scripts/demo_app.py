#!/usr/bin/env python3
"""Streamlit front-end for domain-expert demo sessions.

Browser UI over the same GraphRAG agent used by expert_demo.py — text box,
Invio/Invia submits, spinner while the agent works, answer with sources.
Every exchange is logged to the same JSONL format under artifacts/demo_sessions/.

Usage (on the server):
    conda run -n graphllm streamlit run scripts/demo_app.py --server.address 0.0.0.0 --server.port 8501

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

import urllib.error
import urllib.parse
import urllib.request

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphrag import cli as graphrag_cli  # noqa: E402
from graphrag.agent.core import KGRAGAgent  # noqa: E402
from graphrag.agent.memory import ConversationMemory  # noqa: E402
from graphrag.config import (  # noqa: E402
    AgentConfig,
    OUTPUT_COMPLEXITY,
    build_kg_config_from_env,
)
from graphrag.kg.manager import KnowledgeGraphManager  # noqa: E402
from graphrag.kg.retriever import KGRetriever  # noqa: E402
from graphrag.llm.manager import LLMManager  # noqa: E402
from graphrag.strategies import STRATEGY_PRESETS, apply_strategy  # noqa: E402

logger = logging.getLogger("expert_demo")

STRATEGY = os.environ.get("DEMO_STRATEGY", "hybrid")
MAX_CONTEXT_TOKENS = int(os.environ.get("DEMO_MAX_CONTEXT_TOKENS", "6000"))
# WP2: 512 tokens fit a summary, not a detailed answer with citations; the
# expert's recurring complaint was genericity, and the previous cap left no room
# for figures, names and per-claim references.
MAX_NEW_TOKENS = int(os.environ.get("DEMO_MAX_NEW_TOKENS", "2048"))
# WP2: HIGH drops the "1-2 short paragraphs" instruction and adds the
# specificity rule. WP5: the answer language is pinned to the question language.
COMPLEXITY = OUTPUT_COMPLEXITY(os.environ.get("DEMO_COMPLEXITY", "high"))
ENFORCE_LANGUAGE = os.environ.get("DEMO_ENFORCE_LANGUAGE", "1") == "1"
# Show the full model answer (including 'Verifica nel grafo'); ask the prompt
# for a 'Limits and confidence' section on every answer, not only sparse ones.
SHOW_FULL_ANSWER = os.environ.get("DEMO_SHOW_FULL_ANSWER", "1") == "1"
ALWAYS_LIMITS = os.environ.get("DEMO_ALWAYS_LIMITS", "1") == "1"
# WP1: numbered evidence in the context, [S1]/[T1] tags on specific claims, and a
# source list built from what the model actually cited. Replaces the old
# 'Verifica nel grafo' block, which listed the top-4 triples regardless of use.
CITE_EVIDENCE = os.environ.get("DEMO_CITE_EVIDENCE", "1") == "1"
CITATION_POLICY = os.environ.get("DEMO_CITATION_POLICY", "mark")
# "label" shows "[SEeD for Change, p. 3]" instead of "[S1]": the reader asked
# what S and T meant, which is the answer to whether the ids belong on screen.
CITATION_DISPLAY = os.environ.get("DEMO_CITATION_DISPLAY", "label")
# WP7: intra-session conversational memory. The expert reads an answer and asks
# a follow-up ("mi indichi le strategie nel settore vino") whose subject came
# from that answer; without memory the question reaches retrieval isolated.
# Steers retrieval only — never a source of facts. Demo-only: every other entry
# point passes no memory and behaves exactly as before.
MEMORY = os.environ.get("DEMO_MEMORY", "1") == "1"
# WP3: on a definitional question the chunk carrying the verbatim definition is
# ranked first and the answer opens with it between guillemets. The expert's
# question on SEeD was answered entirely out of triples, which described what
# SEeD does and never said what it is.
VERBATIM_DEFINITIONS = os.environ.get("DEMO_VERBATIM_DEFINITIONS", "1") == "1"
# WP4: MMR plus a per-document cap, so one PDF stops filling the whole context.
# top_k 5 -> 8 pays for the cap: without it, diversification buys breadth by
# giving up depth on the document that actually answers.
TEXT_TOP_K = int(os.environ.get("DEMO_TEXT_TOP_K", "8"))
TEXT_MMR = os.environ.get("DEMO_TEXT_MMR", "1") == "1"
TEXT_MMR_LAMBDA = float(os.environ.get("DEMO_TEXT_MMR_LAMBDA", "0.7"))
TEXT_MAX_PER_DOC = int(os.environ.get("DEMO_TEXT_MAX_PER_DOC", "2"))
# Separates the prose body from the raw evidence block in stored messages;
# the renderer shows what follows inside a monospace expander so triple IDs
# and <doc.pdf> references are not parsed as Markdown links/HTML.
EVIDENCE_MARKER = "\n\n%%EVIDENZE%%\n"
TEXT_RETRIEVER_BACKEND = os.environ.get("DEMO_TEXT_RETRIEVER_BACKEND", "dense")
# Two layers over the same failure, because it has two causes that look alike.
# An out-of-domain question is refused outright by the gate (~0.11 s, no
# retrieval, no answer). An in-domain question whose retrieval came back weak —
# which the recall numbers say is common — is answered, with everything the
# evidence does not support marked '(not in the retrieved evidence)'. A single
# hard gate for both would stonewall legitimate questions, which is the
# expensive error for a demo whose complaint was already genericity.
DOMAIN_GATE = os.environ.get("DEMO_DOMAIN_GATE", "1") == "1"
PARAMETRIC_FALLBACK = os.environ.get("DEMO_PARAMETRIC_FALLBACK", "1") == "1"
# Stage0 runs feeding the text index, most authoritative first. Explicit on
# purpose: auto-discovery picked the newest run, which is the 2-document repair
# run, so the text channel saw 2 of the 22 circular-food documents. Older runs
# in the same artifacts folder hold the previous food-security corpus and must
# stay out.
TEXT_STAGE0_RUNS = os.environ.get(
    "DEMO_TEXT_STAGE0_RUNS",
    "run_fix2docs_20260710,run_full_circular_20260707",
)
ENV_FILE = os.environ.get("DEMO_ENV_FILE", str(ROOT / "kg_pipeline" / ".env"))
LOG_DIR = Path(os.environ.get("DEMO_LOG_DIR", str(ROOT / "artifacts" / "demo_sessions")))
# Comma-separated vLLM endpoints offered in the model selector; each is probed
# at startup and skipped when unreachable, so a stopped server just disappears
# from the list instead of breaking the demo.
VLLM_ENDPOINTS = os.environ.get(
    "DEMO_VLLM_ENDPOINTS",
    "http://localhost:8000/v1,http://localhost:8001/v1",
)


def _build_text_pipeline(backend: str) -> object | None:
    import argparse

    ns = argparse.Namespace(
        text_retriever_backend=backend,
        dense_embedding_model="intfloat/multilingual-e5-base",
        vector_index_dir=str(ROOT / "artifacts" / "vector_index"),
        text_docs_dir="",
        text_stage0_runs=TEXT_STAGE0_RUNS,
    )
    return graphrag_cli._build_text_pipeline(ns)


@st.cache_resource(show_spinner=False)
def _available_models() -> dict[str, tuple[str, str]]:
    """Probe the configured vLLM endpoints and map label -> (base_url, model_id).

    Falls back to VLLM_BASE_URL/VLLM_MODEL_NAME when no endpoint answers, so the
    demo keeps working in single-server setups without the selector env var.
    """
    load_dotenv(ENV_FILE, override=False)
    options: dict[str, tuple[str, str]] = {}
    for base_url in (u.strip().rstrip("/") for u in VLLM_ENDPOINTS.split(",") if u.strip()):
        try:
            with urllib.request.urlopen(f"{base_url}/models", timeout=3) as resp:
                model_id = json.load(resp)["data"][0]["id"]
        except (urllib.error.URLError, OSError, KeyError, IndexError, json.JSONDecodeError):
            continue
        port = urllib.parse.urlparse(base_url).port or "?"
        options[f"{model_id.split('/')[-1]} (:{port})"] = (base_url, model_id)
    if not options:
        model_id = os.environ.get("VLLM_MODEL_NAME", "")
        base_url = os.environ.get("VLLM_BASE_URL", "")
        if model_id and base_url:
            options[model_id.split("/")[-1]] = (base_url, model_id)
    return options


@st.cache_resource(show_spinner="Avvio in corso (connessione al grafo e indice testi)...")
def _load_agent(base_url: str, model_id: str) -> tuple[KGRAGAgent, str]:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("graphrag").setLevel(logging.ERROR)
    load_dotenv(ENV_FILE, override=False)
    os.chdir(ROOT)

    kg_manager = KnowledgeGraphManager(build_kg_config_from_env())
    base = AgentConfig(
        max_content_tokens=MAX_CONTEXT_TOKENS,
        always_include_limits=ALWAYS_LIMITS,
        cite_evidence=CITE_EVIDENCE,
        citation_policy=CITATION_POLICY,
        citation_display=CITATION_DISPLAY,
        complexity=COMPLEXITY,
        enforce_language=ENFORCE_LANGUAGE,
        prefer_verbatim_definitions=VERBATIM_DEFINITIONS,
        text_retriever_top_k=TEXT_TOP_K,
        text_retriever_mmr=TEXT_MMR,
        text_retriever_mmr_lambda=TEXT_MMR_LAMBDA,
        text_retriever_max_per_doc=TEXT_MAX_PER_DOC,
        enable_domain_gate=DOMAIN_GATE,
        allow_parametric_fallback=PARAMETRIC_FALLBACK,
    )
    config = apply_strategy(base, STRATEGY)

    text_pipeline = (
        _build_text_pipeline(TEXT_RETRIEVER_BACKEND) if config.use_text_retriever else None
    )
    retriever = KGRetriever(kg_store=kg_manager, config=config, text_pipeline=text_pipeline)
    llm = LLMManager(
        model_id=model_id,
        warmup=False,
        max_new_tokens=MAX_NEW_TOKENS,
        use_vllm=True,
        vllm_base_url=base_url,
    )
    agent = KGRAGAgent(config=config, kg_retriever=retriever, llm=llm)
    return agent, model_id


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


st.set_page_config(
    page_title="Demo GraphRAG — Economia Circolare del Cibo",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Presentation only — every value below is cosmetic, no functional state lives
# here. Keeps the demo legible for a non-technical expert without touching
# the agent/session logic.
st.markdown(
    """
    <style>
    :root {
        --grag-primary: #3A7D44;
        --grag-primary-dark: #2C5E34;
        --grag-accent: #E2984A;
        --grag-bg-soft: #F3F1EA;
        --grag-border: #E4E0D6;
    }
    .block-container { padding-top: 1.5rem; max-width: 900px; }
    .grag-header {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        padding: 1.1rem 1.4rem;
        margin-bottom: 0.6rem;
        border-radius: 14px;
        background: linear-gradient(135deg, var(--grag-primary) 0%, var(--grag-primary-dark) 100%);
        color: #FFFFFF;
    }
    .grag-header h1 {
        font-size: 1.35rem;
        font-weight: 700;
        margin: 0;
        color: #FFFFFF;
    }
    .grag-header p {
        margin: 0.15rem 0 0 0;
        font-size: 0.88rem;
        color: rgba(255, 255, 255, 0.85);
    }
    .grag-badges { display: flex; gap: 0.5rem; margin-bottom: 1.1rem; flex-wrap: wrap; }
    .grag-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        background: var(--grag-bg-soft);
        border: 1px solid var(--grag-border);
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--grag-primary-dark);
    }
    section[data-testid="stSidebar"] .stButton button { text-align: left; }
    section[data-testid="stSidebar"] h3 {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #6B7280;
        margin-top: 0.4rem;
    }
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        border: 1px solid var(--grag-border);
        padding: 0.2rem 0.4rem;
        margin-bottom: 0.4rem;
    }
    div[data-testid="stExpander"] {
        border-radius: 10px;
        border: 1px solid var(--grag-border);
    }
    </style>
    <div class="grag-header">
        <div>
            <h1>Demo GraphRAG — Economia Circolare del Cibo</h1>
            <p>Scrivi una domanda e premi Invio. Le risposte citano le fonti quando disponibili.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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
    st.markdown("### Modello")
    choice = st.selectbox("Modello", labels, index=default_index, label_visibility="collapsed")

    st.markdown("### Conversazioni")
    if st.button("+ Nuova chat", use_container_width=True, type="primary"):
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
        st.caption(
            "In conversazione su: " + ", ".join(active)
            if active
            else "Nessun argomento attivo."
        )

base_url, model_id = models[choice]

agent, model_id = _load_agent(base_url, model_id)
st.markdown(
    f"""
    <div class="grag-badges">
        <span class="grag-badge">Strategia: {STRATEGY}</span>
        <span class="grag-badge">Modello: {model_id}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

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
        with st.spinner("Sto consultando il grafo e i documenti (10-30 secondi)..."):
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
