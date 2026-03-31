from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv

from graphrag.agent.core import KGRAGAgent
from graphrag.config import AgentConfig, DEFAULT_MODEL_ID, build_kg_config_from_env
from graphrag.kg.manager import KnowledgeGraphManager
from graphrag.kg.retriever import KGRetriever
from graphrag.kg.seed import inject_movie_dataset
from graphrag.llm.manager import LLMManager


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GraphRAG demo pipeline")
    parser.add_argument("--question", default="Chi ha diretto e recitato nel film The Matrix?")
    parser.add_argument("--entity", default="The Matrix")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--seed-movie-dataset", action="store_true")
    parser.add_argument("--llm", action="store_true", help="Enable Hugging Face local generation")
    parser.add_argument("--llm-warmup", action="store_true", help="Preload model at startup")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load local .env if present while preserving variables already set by system/scheduler.
    load_dotenv(override=False)

    if args.llm_warmup and not args.llm:
        parser.error("--llm-warmup requires --llm")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    kg_config = build_kg_config_from_env()
    kg_manager = KnowledgeGraphManager(kg_config)

    if args.seed_movie_dataset:
        inject_movie_dataset(kg_manager)

    print("Graph Schema:", kg_manager.refresh_schema())

    config = AgentConfig(
        query=args.question,
        entity=args.entity,
        include_nodes=True,
        include_triples=True,
        include_neighbors=True,
        llm_warmup=args.llm_warmup,
    )

    retriever = KGRetriever(kg_store=kg_manager, config=config)
    llm_manager = LLMManager(model_id=args.model_id) if args.llm else None

    agent = KGRAGAgent(config=config, kg_retriever=retriever, llm=llm_manager)
    result = agent.invoke(args.question)

    print("\nAgent answer:")
    print(result.get("answer", ""))
    print("\nLatency (ms):", f"{result.get('latency_ms', 0.0):.2f}")


if __name__ == "__main__":
    main()
