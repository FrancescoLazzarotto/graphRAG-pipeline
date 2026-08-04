#!/usr/bin/env python3
"""Interactive demo console for domain-expert evaluation sessions.

A minimal REPL over the GraphRAG agent, designed so a non-technical expert can
query the knowledge graph directly: type a question, read the answer with its
graph evidence. Strategy, model and limits are pre-wired; every exchange is
appended to a JSONL session log (questions collected this way seed the gold
set for the new domain).

Usage:
    conda run -n graphllm python scripts/expert_demo.py
    # options: --strategy hybrid --max-context-tokens 6000 --language it

Exit with 'esci', 'exit', 'quit' or Ctrl-D. Errors never kill the session:
they are logged with the stack trace and the expert sees a polite retry
message.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import readline  # noqa: F401 - enables line editing/history in input()
import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphrag import cli as graphrag_cli  # noqa: E402
from graphrag.agent.core import KGRAGAgent  # noqa: E402
from graphrag.config import build_kg_config_from_env  # noqa: E402
from graphrag.config import AgentConfig  # noqa: E402
from graphrag.kg.manager import KnowledgeGraphManager  # noqa: E402
from graphrag.kg.retriever import KGRetriever  # noqa: E402
from graphrag.llm.manager import LLMManager  # noqa: E402
from graphrag.strategies import STRATEGY_PRESETS, apply_strategy  # noqa: E402

logger = logging.getLogger("expert_demo")

BANNER = """
============================================================
  Demo GraphRAG — Economia Circolare del Cibo
  Scrivi una domanda e premi Invio. Comandi: 'esci' per uscire.
  Le risposte citano le fonti (documento | pagine) quando disponibili.
============================================================
"""


def _build_text_pipeline(backend: str) -> object | None:
    """Index the corpus reusing the CLI's stage0 auto-discovery logic."""
    ns = argparse.Namespace(
        text_retriever_backend=backend,
        dense_embedding_model="intfloat/multilingual-e5-base",
        vector_index_dir=str(ROOT / "artifacts" / "vector_index"),
        text_docs_dir="",
    )
    return graphrag_cli._build_text_pipeline(ns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="hybrid", choices=STRATEGY_PRESETS)
    parser.add_argument("--model-id", default=None, help="Defaults to $VLLM_MODEL_NAME")
    parser.add_argument("--vllm-base-url", default=None, help="Defaults to $VLLM_BASE_URL")
    parser.add_argument("--env-file", default=str(ROOT / "kg_pipeline" / ".env"))
    parser.add_argument("--max-context-tokens", type=int, default=6000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--text-retriever-backend", default="dense", choices=("tfidf", "dense"))
    parser.add_argument("--log-dir", default=str(ROOT / "artifacts" / "demo_sessions"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    # Compression/retrieval warnings are routine noise between questions and
    # would interleave with the expert's prompt; failures still surface.
    logging.getLogger("graphrag").setLevel(logging.ERROR)
    load_dotenv(args.env_file, override=False)

    import os

    # The CLI's stage0 auto-discovery resolves kg_pipeline/artifacts relative
    # to the working directory.
    os.chdir(ROOT)

    model_id = args.model_id or os.environ.get("VLLM_MODEL_NAME", "")
    base_url = args.vllm_base_url or os.environ.get("VLLM_BASE_URL", "")
    if not model_id or not base_url:
        sys.exit("VLLM_MODEL_NAME/VLLM_BASE_URL mancanti (env o flag).")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    session_log = log_dir / f"session_{dt.datetime.now():%Y%m%d_%H%M%S}.jsonl"

    print("Avvio in corso (connessione al grafo e indice testi)...")
    kg_manager = KnowledgeGraphManager(build_kg_config_from_env())

    base = AgentConfig(
        max_content_tokens=args.max_context_tokens,
    )
    config = apply_strategy(base, args.strategy)

    text_pipeline = (
        _build_text_pipeline(args.text_retriever_backend)
        if config.use_text_retriever
        else None
    )
    retriever = KGRetriever(
        kg_store=kg_manager, config=config, text_pipeline=text_pipeline
    )
    llm = LLMManager(
        model_id=model_id,
        warmup=False,
        max_new_tokens=args.max_new_tokens,
        use_vllm=True,
        vllm_base_url=base_url,
    )
    agent = KGRAGAgent(config=config, kg_retriever=retriever, llm=llm)

    print(BANNER)
    print(f"[strategia: {args.strategy} | modello: {model_id}]")
    print(f"[log sessione: {session_log}]\n")

    n_questions = 0
    while True:
        try:
            question = input("Domanda> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"esci", "exit", "quit"}:
            break

        started = time.perf_counter()
        record: dict[str, object] = {
            "ts": dt.datetime.now().isoformat(timespec="seconds"),
            "question": question,
            "strategy": args.strategy,
            "model_id": model_id,
        }
        try:
            print("... sto consultando il grafo e i documenti (10-30 secondi) ...")
            result = agent.invoke(question)
            answer = str(result.get("answer", "")).strip()
            elapsed = time.perf_counter() - started
            record["answer"] = answer
            record["latency_s"] = round(elapsed, 2)
            # The trailing "Verifica nel grafo:" block carries internal node
            # ids for debugging: keep it in the JSONL log, hide it on screen.
            shown = answer.split("\nVerifica nel grafo:")[0].strip()
            print(f"\n{shown}\n")
            print(f"[{elapsed:.0f}s]\n")
            n_questions += 1
        except Exception as exc:  # noqa: BLE001 - REPL must survive any failure
            elapsed = time.perf_counter() - started
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["latency_s"] = round(elapsed, 2)
            logger.error("Question failed: %s\n%s", exc, traceback.format_exc())
            print(
                "\nSi è verificato un problema tecnico con questa domanda. "
                "Riprova, magari riformulandola.\n"
            )
        with session_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Sessione terminata: {n_questions} domande. Log: {session_log}")


if __name__ == "__main__":
    main()
