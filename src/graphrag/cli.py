from __future__ import annotations

import argparse
import copy
import json
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from graphrag.agent.core import KGRAGAgent
from graphrag.config import AgentConfig, DEFAULT_MODEL_ID, build_kg_config_from_env
from graphrag.experiments import ExperimentRunner
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
    parser.add_argument("--llm", action="store_true", help="Enable LLM generation")
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="Use a vLLM OpenAI-compatible endpoint instead of local Hugging Face loading",
    )
    parser.add_argument(
        "--vllm-base-url",
        default="http://localhost:8000/v1",
        help="Base URL for the vLLM OpenAI-compatible API",
    )
    parser.add_argument("--llm-warmup", action="store_true", help="Preload model at startup")
    parser.add_argument(
        "--enable-decomposition-step",
        action="store_true",
        help="Enable LLM decomposition step before retrieval",
    )
    parser.add_argument(
        "--enable-adaptive-routing-step",
        action="store_true",
        help="Enable LLM adaptive routing step before retrieval",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum generated tokens per response")
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=0.92,
        help="Fraction of each GPU memory reserved for model placement (0,1]",
    )
    parser.add_argument(
        "--allow-large-model-fp16-fallback",
        action="store_true",
        help="Allow fp16 fallback for large models when 4-bit quantized loading fails",
    )
    parser.add_argument("--experiment", action="store_true", help="Run batch experiments and persist outputs")
    parser.add_argument("--questions-file", help="Path to a UTF-8 text file with one question per line")
    parser.add_argument("--strategies", default="default", help="Comma-separated strategy presets")
    parser.add_argument("--runs-per-strategy", type=int, default=1)
    parser.add_argument("--output-dir", default="artifacts/experiments")
    parser.add_argument("--experiment-tag", default="")
    return parser


def _build_llm_manager(args: argparse.Namespace, warmup: bool) -> LLMManager | None:
    if not args.llm:
        return None

    return LLMManager(
        model_id=args.model_id,
        warmup=warmup,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_fraction=args.gpu_memory_fraction,
        allow_large_model_fp16_fallback=args.allow_large_model_fp16_fallback,
        use_vllm=args.vllm,
        vllm_base_url=args.vllm_base_url,
    )


def _build_base_config(args: argparse.Namespace) -> AgentConfig:
    return AgentConfig(
        query=args.question,
        entity=args.entity,
        include_nodes=True,
        include_triples=True,
        include_neighbors=True,
        include_subgraph=True,
        include_shortest_path=True,
        llm_warmup=args.llm_warmup,
        enable_decomposition_step=args.enable_decomposition_step,
        enable_adaptive_routing_step=args.enable_adaptive_routing_step,
    )


def _strategy_config(base: AgentConfig, label: str) -> AgentConfig:
    config = copy.deepcopy(base)

    if label == "default":
        return config

    if label == "text_only":
        config.include_nodes = False
        config.include_triples = False
        config.include_neighbors = False
        config.include_subgraph = False
        config.include_shortest_path = False
        return config

    if label == "text_plus_triples":
        config.include_neighbors = False
        config.include_subgraph = False
        config.include_shortest_path = False
        return config

    if label == "neighbors_focus":
        config.include_nodes = False
        config.include_subgraph = False
        config.include_shortest_path = False
        return config

    if label == "subgraph_2hop":
        config.hops = max(2, int(config.hops))
        config.include_nodes = False
        config.include_neighbors = False
        config.include_shortest_path = False
        return config

    if label == "shortest_path":
        config.include_nodes = False
        config.include_neighbors = False
        config.include_subgraph = False
        return config

    raise ValueError(
        "Unknown strategy '"
        + label
        + "'. Allowed: default,text_only,text_plus_triples,neighbors_focus,subgraph_2hop,shortest_path"
    )


def _load_questions(args: argparse.Namespace) -> list[str]:
    if not args.questions_file:
        return [args.question]

    questions_path = Path(args.questions_file)
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    questions = [line.strip() for line in questions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not questions:
        raise ValueError(f"Questions file is empty: {questions_path}")
    return questions


def _run_experiments(args: argparse.Namespace, kg_manager: KnowledgeGraphManager) -> None:
    if args.runs_per_strategy < 1:
        raise ValueError("--runs-per-strategy must be >= 1")

    questions = _load_questions(args)
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    if not strategies:
        raise ValueError("--strategies must include at least one strategy")

    base_config = _build_base_config(args)
    llm_manager = _build_llm_manager(args=args, warmup=args.llm_warmup)
    runner = ExperimentRunner(questions=questions)

    for strategy in strategies:
        for run_index in range(1, args.runs_per_strategy + 1):
            config = _strategy_config(base=base_config, label=strategy)
            retriever = KGRetriever(kg_store=kg_manager, config=config)
            agent = KGRAGAgent(config=config, kg_retriever=retriever, llm=llm_manager)
            runner.run_agent(
                agent=agent,
                label=strategy,
                run_metadata={
                    "run_index": run_index,
                    "model_id": args.model_id if args.llm else "none",
                    "llm_enabled": args.llm,
                    "vllm_enabled": args.vllm,
                    "vllm_base_url": args.vllm_base_url if args.llm and args.vllm else "",
                    "max_new_tokens": args.max_new_tokens if args.llm else 0,
                    "gpu_memory_fraction": args.gpu_memory_fraction if args.llm else 0.0,
                    "allow_large_model_fp16_fallback": args.allow_large_model_fp16_fallback,
                    "enable_decomposition_step": args.enable_decomposition_step,
                    "enable_adaptive_routing_step": args.enable_adaptive_routing_step,
                },
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.experiment_tag.strip() or "batch"
    output_dir = Path(args.output_dir) / f"{timestamp}_{tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "results.jsonl"
    csv_path = output_dir / "results.csv"
    summary_txt_path = output_dir / "summary.txt"
    summary_json_path = output_dir / "summary.json"

    runner.export_jsonl(str(jsonl_path))
    runner.export_csv(str(csv_path))
    summary_text = runner.summary()
    summary_txt_path.write_text(summary_text + "\n", encoding="utf-8")
    summary_json_path.write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "tag": tag,
                "questions_count": len(questions),
                "strategies": strategies,
                "runs_per_strategy": args.runs_per_strategy,
                "llm": {
                    "enabled": args.llm,
                    "model_id": args.model_id if args.llm else "none",
                    "vllm_enabled": args.vllm,
                    "vllm_base_url": args.vllm_base_url if args.llm and args.vllm else "",
                    "max_new_tokens": args.max_new_tokens if args.llm else 0,
                    "gpu_memory_fraction": args.gpu_memory_fraction if args.llm else 0.0,
                    "allow_large_model_fp16_fallback": args.allow_large_model_fp16_fallback,
                },
                "agent_pipeline": {
                    "enable_decomposition_step": args.enable_decomposition_step,
                    "enable_adaptive_routing_step": args.enable_adaptive_routing_step,
                },
                "stats": runner.summary_stats(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nExperiment completed.")
    print("Output directory:", output_dir)
    print("-", jsonl_path)
    print("-", csv_path)
    print("-", summary_txt_path)
    print("-", summary_json_path)
    print("\nSummary:")
    print(summary_text)


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load local .env if present while preserving variables already set by system/scheduler.
    load_dotenv(override=False)

    if args.llm_warmup and not args.llm:
        parser.error("--llm-warmup requires --llm")
    if args.vllm and not args.llm:
        parser.error("--vllm requires --llm")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")
    if args.gpu_memory_fraction <= 0 or args.gpu_memory_fraction > 1:
        parser.error("--gpu-memory-fraction must be in (0, 1]")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    kg_config = build_kg_config_from_env()
    kg_manager = KnowledgeGraphManager(kg_config)

    if args.seed_movie_dataset:
        inject_movie_dataset(kg_manager)

    print("Graph Schema:", kg_manager.refresh_schema())

    if args.experiment:
        _run_experiments(args=args, kg_manager=kg_manager)
        return

    config = _build_base_config(args)

    retriever = KGRetriever(kg_store=kg_manager, config=config)
    llm_manager = _build_llm_manager(args=args, warmup=False)

    agent = KGRAGAgent(config=config, kg_retriever=retriever, llm=llm_manager)
    result = agent.invoke(args.question)

    print("\nAgent answer:")
    print(result.get("answer", ""))
    print("\nLatency (ms):", f"{result.get('latency_ms', 0.0):.2f}")


if __name__ == "__main__":
    main()
