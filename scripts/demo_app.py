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
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphrag import cli as graphrag_cli  # noqa: E402
from graphrag.agent.core import KGRAGAgent  # noqa: E402
from graphrag.config import AgentConfig, build_kg_config_from_env  # noqa: E402
from graphrag.kg.manager import KnowledgeGraphManager  # noqa: E402
from graphrag.kg.retriever import KGRetriever  # noqa: E402
from graphrag.llm.manager import LLMManager  # noqa: E402
from graphrag.strategies import STRATEGY_PRESETS, apply_strategy  # noqa: E402

logger = logging.getLogger("expert_demo")

STRATEGY = os.environ.get("DEMO_STRATEGY", "hybrid")
MAX_CONTEXT_TOKENS = int(os.environ.get("DEMO_MAX_CONTEXT_TOKENS", "6000"))
MAX_NEW_TOKENS = int(os.environ.get("DEMO_MAX_NEW_TOKENS", "512"))
# Show the full model answer (including 'Verifica nel grafo'); ask the prompt
# for a 'Limits and confidence' section on every answer, not only sparse ones.
SHOW_FULL_ANSWER = os.environ.get("DEMO_SHOW_FULL_ANSWER", "1") == "1"
ALWAYS_LIMITS = os.environ.get("DEMO_ALWAYS_LIMITS", "1") == "1"
# Separates the prose body from the raw evidence block in stored messages;
# the renderer shows what follows inside a monospace expander so triple IDs
# and <doc.pdf> references are not parsed as Markdown links/HTML.
EVIDENCE_MARKER = "\n\n%%EVIDENZE%%\n"
TEXT_RETRIEVER_BACKEND = os.environ.get("DEMO_TEXT_RETRIEVER_BACKEND", "dense")
ENV_FILE = os.environ.get("DEMO_ENV_FILE", str(ROOT / "kg_pipeline" / ".env"))
LOG_DIR = Path(os.environ.get("DEMO_LOG_DIR", str(ROOT / "artifacts" / "demo_sessions")))


def _build_text_pipeline(backend: str) -> object | None:
    import argparse

    ns = argparse.Namespace(
        text_retriever_backend=backend,
        dense_embedding_model="intfloat/multilingual-e5-base",
        vector_index_dir=str(ROOT / "artifacts" / "vector_index"),
        text_docs_dir="",
    )
    return graphrag_cli._build_text_pipeline(ns)


@st.cache_resource(show_spinner="Avvio in corso (connessione al grafo e indice testi)...")
def _load_agent() -> tuple[KGRAGAgent, str]:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("graphrag").setLevel(logging.ERROR)
    load_dotenv(ENV_FILE, override=False)
    os.chdir(ROOT)

    model_id = os.environ.get("VLLM_MODEL_NAME", "")
    base_url = os.environ.get("VLLM_BASE_URL", "")
    if not model_id or not base_url:
        st.error("VLLM_MODEL_NAME/VLLM_BASE_URL mancanti (env o .env).")
        st.stop()

    kg_manager = KnowledgeGraphManager(build_kg_config_from_env())
    base = AgentConfig(
        max_content_tokens=MAX_CONTEXT_TOKENS,
        always_include_limits=ALWAYS_LIMITS,
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


def _ask(agent: KGRAGAgent, model_id: str, question: str) -> str:
    started = time.perf_counter()
    record: dict[str, object] = {
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "strategy": STRATEGY,
        "model_id": model_id,
    }
    try:
        result = agent.invoke(question)
        answer = str(result.get("answer", "")).strip()
        elapsed = time.perf_counter() - started
        record["answer"] = answer
        record["latency_s"] = round(elapsed, 2)
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


st.set_page_config(page_title="Demo GraphRAG — Economia Circolare del Cibo", page_icon="")
st.title("Demo GraphRAG")
st.caption("Scrivi una domanda e premi Invio. Le risposte citano le fonti quando disponibili.")

agent, model_id = _load_agent()
st.caption(f"strategia: {STRATEGY} | modello: {model_id}")

if "messages" not in st.session_state:
    st.session_state.messages = []

def _render(content: str) -> None:
    body, sep, evidence = content.partition(EVIDENCE_MARKER)
    st.markdown(body)
    if sep:
        with st.expander("Verifica nel grafo (evidenze)"):
            st.code(evidence, language=None)


for role, content in st.session_state.messages:
    with st.chat_message(role):
        _render(content)

question = st.chat_input("Scrivi qui la tua domanda...")
if question:
    st.session_state.messages.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Sto consultando il grafo e i documenti (10-30 secondi)..."):
            answer = _ask(agent, model_id, question)
        _render(answer)
    st.session_state.messages.append(("assistant", answer))
