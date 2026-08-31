"""Shared configuration for the two interactive demos.

``product/app.py`` (Streamlit) and ``product/console.py`` (console) are
documented as the same product, but each used to build its own ``AgentConfig``:
the Streamlit one set fourteen fields, the console one set a single field. The
console therefore answered the same question shorter, without citations, with no
language pin and no domain gate — exactly the defects the WP1-WP7 work removed
from the other surface. This module is the one place that decision is made, so a
future improvement reaches both demos at once.

Every setting is an environment variable with the value the Streamlit demo
already shipped as its default, so the observable behaviour of that demo does
not change; the console demo inherits it.

Nothing here is imported by the CLI, the experiment runners or the evaluation
scripts: campaign configuration stays in ``graphrag.config`` and
``graphrag.strategies``, and runs stay comparable with the ones already
measured. That is the whole point of the split — ``src/graphrag`` is the engine
the thesis measured, ``product/`` is how it is presented.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

from graphrag.config import (
    AgentConfig,
    KGConfig,
    OUTPUT_COMPLEXITY,
    build_kg_config_from_env,
)
from graphrag.kg.manager import KnowledgeGraphManager
from graphrag.kg.retriever import KGRetriever
from graphrag.llm.manager import LLMManager
from graphrag.strategies import apply_strategy

logger = logging.getLogger("graphrag")

ROOT = Path(__file__).resolve().parents[1]


def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip() == "1"


STRATEGY = os.environ.get("DEMO_STRATEGY", "hybrid")
MAX_CONTEXT_TOKENS = int(os.environ.get("DEMO_MAX_CONTEXT_TOKENS", "6000"))
# WP2: 512 tokens fit a summary, not a detailed answer with citations; the
# expert's recurring complaint was genericity, and the previous cap left no room
# for figures, names and per-claim references.
MAX_NEW_TOKENS = int(os.environ.get("DEMO_MAX_NEW_TOKENS", "2048"))
# WP2: HIGH drops the "1-2 short paragraphs" instruction and adds the
# specificity rule. WP5: the answer language is pinned to the question language.
COMPLEXITY = OUTPUT_COMPLEXITY(os.environ.get("DEMO_COMPLEXITY", "high"))
ENFORCE_LANGUAGE = _flag("DEMO_ENFORCE_LANGUAGE")
# Show the full model answer (including 'Verifica nel grafo'); ask the prompt
# for a 'Limits and confidence' section on every answer, not only sparse ones.
SHOW_FULL_ANSWER = _flag("DEMO_SHOW_FULL_ANSWER")
ALWAYS_LIMITS = _flag("DEMO_ALWAYS_LIMITS")
# WP1: numbered evidence in the context, [S1]/[T1] tags on specific claims, and a
# source list built from what the model actually cited. Replaces the old
# 'Verifica nel grafo' block, which listed the top-4 triples regardless of use.
CITE_EVIDENCE = _flag("DEMO_CITE_EVIDENCE")
CITATION_POLICY = os.environ.get("DEMO_CITATION_POLICY", "mark")
# "label" shows "[SEeD for Change, p. 3]" instead of "[S1]": the reader asked
# what S and T meant, which is the answer to whether the ids belong on screen.
CITATION_DISPLAY = os.environ.get("DEMO_CITATION_DISPLAY", "label")
# WP7: intra-session conversational memory. The expert reads an answer and asks
# a follow-up ("mi indichi le strategie nel settore vino") whose subject came
# from that answer; without memory the question reaches retrieval isolated.
# Steers retrieval only — never a source of facts. Demo-only: every other entry
# point passes no memory and behaves exactly as before.
MEMORY = _flag("DEMO_MEMORY")
# WP3: on a definitional question the chunk carrying the verbatim definition is
# ranked first and the answer opens with it between guillemets. The expert's
# question on SEeD was answered entirely out of triples, which described what
# SEeD does and never said what it is.
VERBATIM_DEFINITIONS = _flag("DEMO_VERBATIM_DEFINITIONS")
# WP4: MMR plus a per-document cap, so one PDF stops filling the whole context.
# top_k 5 -> 8 pays for the cap: without it, diversification buys breadth by
# giving up depth on the document that actually answers.
TEXT_TOP_K = int(os.environ.get("DEMO_TEXT_TOP_K", "8"))
TEXT_MMR = _flag("DEMO_TEXT_MMR")
TEXT_MMR_LAMBDA = float(os.environ.get("DEMO_TEXT_MMR_LAMBDA", "0.7"))
TEXT_MAX_PER_DOC = int(os.environ.get("DEMO_TEXT_MAX_PER_DOC", "2"))
TEXT_RETRIEVER_BACKEND = os.environ.get("DEMO_TEXT_RETRIEVER_BACKEND", "dense")
# Two layers over the same failure, because it has two causes that look alike.
# An out-of-domain question is refused outright by the gate (~0.11 s, no
# retrieval, no answer). An in-domain question whose retrieval came back weak —
# which the recall numbers say is common — is answered, with everything the
# evidence does not support marked '(not in the retrieved evidence)'. A single
# hard gate for both would stonewall legitimate questions, which is the
# expensive error for a demo whose complaint was already genericity.
DOMAIN_GATE = _flag("DEMO_DOMAIN_GATE")
PARAMETRIC_FALLBACK = _flag("DEMO_PARAMETRIC_FALLBACK")
# The cross-lingual half of retrieval: the graph is largely Italian, the
# questions arrive in both languages. Measured on the gold set at the end of
# July, it moved context recall 0.386 -> 0.602 on this strategy. Neither demo
# passed it before, so the fix had never reached a live session; both graphs in
# use carry the 14 520 :NodeVec carriers the channel needs.
VECTOR_RETRIEVAL = _flag("DEMO_VECTOR_RETRIEVAL")
# ...and what to do when that encoder is unreachable. The engine raises by
# default, which is right for a campaign: a run that silently changes retrieval
# method halfway is worse than a run that stops. It is the wrong answer here.
# A stopped encoder made every single question in the demo fail with "problema
# tecnico", including the ones the graph could still answer lexically and the
# ones answered mostly from the text channel, which does not use that encoder
# at all. Degrading is only acceptable because the UI says so on the affected
# answer; without that caption this line would trade a loud failure for a
# quiet loss of quality. `setdefault`, so an operator can still export 0.
os.environ.setdefault("GRAPHRAG_VECTOR_ALLOW_DEGRADED", "1")
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
    "http://localhost:8000/v1,http://localhost:8001/v1,http://localhost:8003/v1",
)


# ---------------------------------------------------------------------- #
# graph connection
# ---------------------------------------------------------------------- #


def _probe_kg(config: KGConfig) -> KnowledgeGraphManager:
    """Open a connection and prove it answers, or raise."""
    manager = KnowledgeGraphManager(config)
    # Straight through the driver, not through run_query: that one retries
    # three times with backoff, which is right mid-session and wrong here,
    # where the point is to find out quickly whether to use the other graph.
    manager.graph.query("RETURN 1 AS ok")
    return manager


def build_kg_manager() -> tuple[KnowledgeGraphManager, str]:
    """Connect to the primary graph, fall back to the secondary one.

    The primary graph is an Aura Free instance, which suspends itself after
    three idle days and then resolves to nothing at all; the same graph is also
    mirrored locally. A demo that dies because the hosted copy went to sleep is
    a demo that dies in front of the person it was booked for, so an unreachable
    primary moves to the fallback instead of failing.

    Returns:
        The connected manager and a label naming which graph answered.

    Raises:
        RuntimeError: Neither graph could be reached.
    """
    # Idempotent and never overrides an exported variable, so a caller that
    # already loaded the file (or set NEO4J_URL inline) keeps its choice.
    load_dotenv(ENV_FILE, override=False)

    primary_error: Exception | None = None
    try:
        primary = build_kg_config_from_env()
        return _probe_kg(primary), f"primario ({primary.url})"
    except Exception as exc:  # noqa: BLE001 - any failure means "try the other one"
        primary_error = exc
        logger.warning("Primary graph unreachable (%s); trying the fallback.", exc)

    fallback_url = os.environ.get("DEMO_NEO4J_FALLBACK_URL", "").strip()
    if not fallback_url:
        raise RuntimeError(
            f"Grafo primario non raggiungibile ({primary_error}) e nessun "
            "fallback configurato: imposta DEMO_NEO4J_FALLBACK_URL / "
            "_USERNAME / _PASSWORD / _DATABASE."
        )
    fallback = build_kg_config_from_env(
        url_env="DEMO_NEO4J_FALLBACK_URL",
        username_env="DEMO_NEO4J_FALLBACK_USERNAME",
        password_env="DEMO_NEO4J_FALLBACK_PASSWORD",
        database_env="DEMO_NEO4J_FALLBACK_DATABASE",
    )
    # An unset database is not "no database": the driver then reads NEO4J_DATABASE
    # itself, so the fallback inherited Aura's database name ("588fe1bc") and
    # failed with DatabaseNotFound against a local instance that only has "neo4j".
    if not fallback.database:
        fallback.database = "neo4j"
    try:
        manager = _probe_kg(fallback)
    except Exception as exc:  # noqa: BLE001 - report both failures, not the last
        raise RuntimeError(
            f"Nessun grafo raggiungibile. Primario: {primary_error}. "
            f"Fallback ({fallback_url}): {exc}."
        ) from exc
    logger.warning("Using the fallback graph at %s.", fallback.url)
    return manager, f"fallback ({fallback.url})"


# ---------------------------------------------------------------------- #
# model selection
# ---------------------------------------------------------------------- #


def probe_vllm_endpoints(timeout_sec: float = 3.0) -> dict[str, tuple[str, str]]:
    """Map "model (:port)" -> (base_url, model_id) for every endpoint that answers.

    Falls back to VLLM_BASE_URL/VLLM_MODEL_NAME when no endpoint answers, so the
    demo keeps working in single-server setups without the selector env var.
    """
    options: dict[str, tuple[str, str]] = {}
    for base_url in (u.strip().rstrip("/") for u in VLLM_ENDPOINTS.split(",") if u.strip()):
        try:
            with urllib.request.urlopen(f"{base_url}/models", timeout=timeout_sec) as resp:
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


# ---------------------------------------------------------------------- #
# agent
# ---------------------------------------------------------------------- #


def build_text_pipeline(backend: str = TEXT_RETRIEVER_BACKEND) -> object | None:
    """Index the corpus reusing the CLI's stage0 auto-discovery logic."""
    import argparse

    from graphrag import cli as graphrag_cli

    ns = argparse.Namespace(
        text_retriever_backend=backend,
        dense_embedding_model="intfloat/multilingual-e5-base",
        vector_index_dir=str(ROOT / "artifacts" / "vector_index"),
        text_docs_dir="",
        text_stage0_runs=TEXT_STAGE0_RUNS,
    )
    return graphrag_cli._build_text_pipeline(ns)


def build_agent_config(strategy: str = STRATEGY) -> AgentConfig:
    """The demo's answer-quality settings, before the strategy preset is applied."""
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
        vector_retrieval=VECTOR_RETRIEVAL,
    )
    return apply_strategy(base, strategy)


def build_demo_agent(
    base_url: str,
    model_id: str,
    strategy: str = STRATEGY,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> tuple[object, str]:
    """Build the agent both demos run, and say which graph it is talking to.

    Returns:
        The ``KGRAGAgent`` and the label of the graph it connected to.
    """
    from graphrag.agent.core import KGRAGAgent

    kg_manager, graph_label = build_kg_manager()
    config = build_agent_config(strategy)

    text_pipeline = build_text_pipeline() if config.use_text_retriever else None
    retriever = KGRetriever(
        kg_store=kg_manager, config=config, text_pipeline=text_pipeline
    )
    llm = LLMManager(
        model_id=model_id,
        warmup=False,
        max_new_tokens=max_new_tokens,
        use_vllm=True,
        vllm_base_url=base_url,
    )
    return KGRAGAgent(config=config, kg_retriever=retriever, llm=llm), graph_label
