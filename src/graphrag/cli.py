from __future__ import annotations

import argparse
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
from graphrag.llm.manager import LLMManager
from graphrag.strategies import apply_strategy
from graphrag.text_rag.pipeline import StandardTextRAGPipeline


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GraphRAG demo pipeline")
    parser.add_argument(
        "--question", default="Quali sono le relazioni tra Entita A e Entita B?"
    )
    parser.add_argument(
        "--entity",
        default="",
        help="Optional entity seed for graph traversal (leave empty for auto-seeding)",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
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
    parser.add_argument(
        "--llm-warmup", action="store_true", help="Preload model at startup"
    )
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
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens per response",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=1000,
        help="Maximum tokens for compressed context before generation",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=50,
        help="Maximum LangGraph recursion steps before aborting",
    )
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
    parser.add_argument(
        "--experiment",
        action="store_true",
        help="Run batch experiments and persist outputs",
    )
    parser.add_argument(
        "--questions-file", help="Path to a UTF-8 text file with one question per line"
    )
    parser.add_argument(
        "--strategies", default="default", help="Comma-separated strategy presets"
    )
    parser.add_argument("--runs-per-strategy", type=int, default=1)
    parser.add_argument("--output-dir", default="artifacts/experiments")
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument(
        "--text-docs-dir",
        default="",
        help="Directory of documents (PDF/txt/md) to index for text_only standard RAG. "
             "If omitted, auto-discovers from the latest KG pipeline stage0 artifacts.",
    )
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
        recursion_limit=args.recursion_limit,
        max_content_tokens=args.max_context_tokens,
    )


def _build_text_pipeline(args: argparse.Namespace) -> StandardTextRAGPipeline | None:
    logger = logging.getLogger("graphrag.cli")
    pipeline = StandardTextRAGPipeline()

    docs_dir = (args.text_docs_dir or "").strip()
    if docs_dir:
        target = Path(docs_dir)
        if not target.exists():
            logger.warning("--text-docs-dir %s not found; text retrieval disabled", docs_dir)
            return None
        n = pipeline.index_directory(target)
        logger.info("Text pipeline: indexed %d chunks from %s", n, docs_dir)
        return pipeline

    # Auto-discover from the latest KG stage0 artifacts.
    kg_artifacts = Path("kg_pipeline/artifacts")
    if kg_artifacts.exists():
        run_dirs = sorted(
            [p for p in kg_artifacts.iterdir() if p.is_dir() and p.name.startswith("run_")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for run_dir in run_dirs:
            stage0 = run_dir / "stage0_documents.json"
            if not stage0.exists():
                continue
            try:
                import json as _json
                from graphrag.text_rag.manager import TextChunk
                docs = _json.loads(stage0.read_text(encoding="utf-8"))
                if not isinstance(docs, list):
                    docs = []
                chunks: list[TextChunk] = []
                for doc_idx, doc in enumerate(docs, start=1):
                    text = str(doc.get("markdown_text", "") or "").strip()
                    filename = str(doc.get("filename", f"doc_{doc_idx}"))
                    if not text:
                        continue
                    # Split into ~1200-char chunks with 180-char overlap.
                    step = 1200 - 180
                    for c_idx, start in enumerate(range(0, len(text), step), start=1):
                        fragment = text[start : start + 1200].strip()
                        if len(fragment) >= 80:
                            chunks.append(TextChunk(
                                chunk_id=f"d{doc_idx:04d}-c{c_idx:04d}",
                                content=fragment,
                                source=filename,
                            ))
                if chunks:
                    pipeline.retriever.add_chunks(chunks)
                    logger.info(
                        "Text pipeline: indexed %d chunks from %s (stage0)",
                        len(chunks), run_dir.name,
                    )
                    return pipeline
            except Exception as exc:
                logger.warning("Failed to load stage0 from %s: %s", run_dir, exc)
                continue

    logger.warning("No text documents found; text_only strategy will have empty context")
    return None


def _load_questions(args: argparse.Namespace) -> list[str]:
    if not args.questions_file:
        return [args.question]

    questions_path = Path(args.questions_file)
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    questions = [
        line.strip()
        for line in questions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not questions:
        raise ValueError(f"Questions file is empty: {questions_path}")
    return questions


def _run_experiments(
    args: argparse.Namespace, kg_manager: KnowledgeGraphManager
) -> None:
    if args.runs_per_strategy < 1:
        raise ValueError("--runs-per-strategy must be >= 1")

    questions = _load_questions(args)
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    if not strategies:
        raise ValueError("--strategies must include at least one strategy")

    base_config = _build_base_config(args)
    llm_manager = _build_llm_manager(args=args, warmup=args.llm_warmup)
    runner = ExperimentRunner(questions=questions)

    needs_text = any(s in ("text_only",) for s in strategies)
    text_pipeline = _build_text_pipeline(args) if needs_text else None

    for strategy in strategies:
        for run_index in range(1, args.runs_per_strategy + 1):
            config = apply_strategy(base_config, strategy)
            retriever = KGRetriever(
                kg_store=kg_manager,
                config=config,
                text_pipeline=text_pipeline if config.use_text_retriever else None,
            )
            agent = KGRAGAgent(config=config, kg_retriever=retriever, llm=llm_manager)
            runner.run_agent(
                agent=agent,
                label=strategy,
                run_metadata={
                    "run_index": run_index,
                    "model_id": args.model_id if args.llm else "none",
                    "llm_enabled": args.llm,
                    "vllm_enabled": args.vllm,
                    "vllm_base_url": args.vllm_base_url
                    if args.llm and args.vllm
                    else "",
                    "max_new_tokens": args.max_new_tokens if args.llm else 0,
                    "gpu_memory_fraction": args.gpu_memory_fraction
                    if args.llm
                    else 0.0,
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
                    "vllm_base_url": args.vllm_base_url
                    if args.llm and args.vllm
                    else "",
                    "max_new_tokens": args.max_new_tokens if args.llm else 0,
                    "gpu_memory_fraction": args.gpu_memory_fraction
                    if args.llm
                    else 0.0,
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
    if args.max_context_tokens < 1:
        parser.error("--max-context-tokens must be >= 1")
    if args.recursion_limit < 1:
        parser.error("--recursion-limit must be >= 1")
    if args.gpu_memory_fraction <= 0 or args.gpu_memory_fraction > 1:
        parser.error("--gpu-memory-fraction must be in (0, 1]")

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    kg_config = build_kg_config_from_env()
    kg_manager = KnowledgeGraphManager(kg_config)

    print("Graph Schema:", kg_manager.refresh_schema())

    if args.experiment:
        _run_experiments(args=args, kg_manager=kg_manager)
        return

    config = _build_base_config(args)

    text_pipeline = _build_text_pipeline(args) if config.use_text_retriever else None
    retriever = KGRetriever(kg_store=kg_manager, config=config, text_pipeline=text_pipeline)
    llm_manager = _build_llm_manager(args=args, warmup=False)

    agent = KGRAGAgent(config=config, kg_retriever=retriever, llm=llm_manager)
    result = agent.invoke(args.question)

    print("\nAgent answer:")
    print(result.get("answer", ""))
    print("\nLatency (ms):", f"{result.get('latency_ms', 0.0):.2f}")


if __name__ == "__main__":
    main()
