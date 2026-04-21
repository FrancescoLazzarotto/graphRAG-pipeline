from __future__ import annotations

import argparse
import copy
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from graphrag.agent.core import KGRAGAgent
from graphrag.config import AgentConfig, DEFAULT_MODEL_ID, build_kg_config_from_env
from graphrag.experiments import ExperimentRunner
from graphrag.experiments.resource_monitor import ResourceMonitor
from graphrag.kg.manager import KnowledgeGraphManager
from graphrag.kg.retriever import KGRetriever
from graphrag.kg.seed import inject_movie_dataset
from graphrag.llm.manager import LLMManager
from graphrag.text_rag.agent import StandardRAGAgent
from graphrag.text_rag.pipeline import StandardTextRAGPipeline

_GRAPH_STRATEGIES_DEFAULT = (
    "default",
    "text_only",
    "text_plus_triples",
    "neighbors_focus",
    "subgraph_2hop",
    "shortest_path",
)
_GRAPH_STRATEGIES_SMOKE = ("default", "text_plus_triples")


@dataclass(frozen=True)
class StandardStrategyPreset:
    top_k: int
    chunk_size: int
    chunk_overlap: int
    min_chunk_chars: int = 80
    include_sources: bool = True


_STANDARD_STRATEGY_PRESETS: dict[str, StandardStrategyPreset] = {
    "std_topk3": StandardStrategyPreset(top_k=3, chunk_size=1200, chunk_overlap=180),
    "std_topk5": StandardStrategyPreset(top_k=5, chunk_size=1200, chunk_overlap=180),
    "std_wide_context": StandardStrategyPreset(top_k=6, chunk_size=1800, chunk_overlap=240),
    "std_fine_chunks": StandardStrategyPreset(top_k=5, chunk_size=800, chunk_overlap=140),
}
_STANDARD_STRATEGIES_DEFAULT = tuple(_STANDARD_STRATEGY_PRESETS.keys())
_STANDARD_STRATEGIES_SMOKE = ("std_topk3", "std_topk5")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run multi-question, multi-strategy matrix over Standard RAG and GraphRAG",
    )

    parser.add_argument("--question", default="Chi ha diretto e recitato nel film The Matrix?")
    parser.add_argument("--questions-file", default="", help="One question per line")
    parser.add_argument("--entity", default="The Matrix")

    parser.add_argument(
        "--documents",
        nargs="+",
        default=["docs", "README.md"],
        help="Files/folders used by standard text RAG",
    )
    parser.add_argument(
        "--doc-patterns",
        default="*.pdf,*.txt,*.md,*.markdown",
        help="Comma-separated file patterns for directory discovery",
    )

    parser.add_argument("--standard-strategies", default=",".join(_STANDARD_STRATEGIES_DEFAULT))
    parser.add_argument("--graph-strategies", default=",".join(_GRAPH_STRATEGIES_DEFAULT))

    parser.add_argument("--runs-per-strategy", type=int, default=1)
    parser.add_argument("--output-dir", default="artifacts/experiments")
    parser.add_argument("--experiment-tag", default="")

    parser.add_argument(
        "--monitor-resources",
        dest="monitor_resources",
        action="store_true",
        help="Enable CPU/RAM/GPU sampling and save telemetry files",
    )
    parser.add_argument(
        "--no-monitor-resources",
        dest="monitor_resources",
        action="store_false",
        help="Disable resource telemetry sampling",
    )
    parser.set_defaults(monitor_resources=True)
    parser.add_argument(
        "--resource-sample-interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds for resource telemetry",
    )

    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--llm-warmup", action="store_true")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)

    parser.add_argument("--seed-movie-dataset", action="store_true")

    parser.add_argument("--skip-standard", action="store_true")
    parser.add_argument("--skip-graph", action="store_true")

    parser.add_argument("--smoke", action="store_true", help="Run a smaller test first")
    parser.add_argument("--smoke-questions", type=int, default=2)
    parser.add_argument("--smoke-standard-strategies", default=",".join(_STANDARD_STRATEGIES_SMOKE))
    parser.add_argument("--smoke-graph-strategies", default=",".join(_GRAPH_STRATEGIES_SMOKE))

    return parser


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _resource_summary_lines(resource_summary: dict[str, Any] | None) -> list[str]:
    if not resource_summary:
        return []

    lines = [
        "Resource Monitor:",
        f"- duration_sec={_format_optional_float(resource_summary.get('monitoring_duration_sec'))}",
        f"- peak_process_rss_mb={_format_optional_float(resource_summary.get('peak_process_rss_mb'))}",
        f"- peak_system_ram_percent={_format_optional_float(resource_summary.get('peak_system_ram_percent'))}",
        f"- peak_system_cpu_percent={_format_optional_float(resource_summary.get('peak_system_cpu_percent'))}",
    ]

    gpus = resource_summary.get("gpus", [])
    if isinstance(gpus, list) and gpus:
        for gpu in gpus:
            if not isinstance(gpu, dict):
                continue
            gpu_index = gpu.get("index", "?")
            gpu_name = gpu.get("name", "unknown")
            lines.append(
                f"- gpu[{gpu_index}] {gpu_name}: peak_mem_mb={_format_optional_float(gpu.get('peak_memory_used_mb'))}, "
                f"peak_util_gpu_percent={_format_optional_float(gpu.get('peak_utilization_gpu_percent'))}"
            )
    else:
        lines.append("- gpu: no samples detected")

    return lines


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


def _base_graph_config(args: argparse.Namespace, default_question: str) -> AgentConfig:
    return AgentConfig(
        query=default_question,
        entity=args.entity,
        include_nodes=True,
        include_triples=True,
        include_neighbors=True,
        include_subgraph=True,
        include_shortest_path=True,
        llm_warmup=args.llm_warmup,
    )


def _graph_strategy_config(base: AgentConfig, label: str) -> AgentConfig:
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

    allowed = ",".join(_GRAPH_STRATEGIES_DEFAULT)
    raise ValueError(f"Unknown graph strategy '{label}'. Allowed: {allowed}")


def _resolve_standard_labels(raw_labels: list[str]) -> list[str]:
    invalid = [label for label in raw_labels if label not in _STANDARD_STRATEGY_PRESETS]
    if invalid:
        allowed = ", ".join(sorted(_STANDARD_STRATEGY_PRESETS.keys()))
        invalid_csv = ", ".join(invalid)
        raise ValueError(f"Unknown standard strategies: {invalid_csv}. Allowed: {allowed}")
    return raw_labels


def _run_standard_matrix(
    runner: ExperimentRunner,
    questions: list[str],
    labels: list[str],
    runs_per_strategy: int,
    llm_manager: LLMManager | None,
    args: argparse.Namespace,
) -> None:
    if not labels:
        return

    discovery_patterns = _parse_csv(args.doc_patterns)

    for label in labels:
        preset = _STANDARD_STRATEGY_PRESETS[label]
        pipeline = StandardTextRAGPipeline(
            chunk_size=preset.chunk_size,
            chunk_overlap=preset.chunk_overlap,
            min_chunk_chars=preset.min_chunk_chars,
        )
        indexed_chunks = pipeline.index_paths(args.documents, discovery_patterns=discovery_patterns)

        agent_config = AgentConfig(
            query=questions[0],
            llm_warmup=args.llm_warmup,
        )

        for run_index in range(1, runs_per_strategy + 1):
            agent = StandardRAGAgent(
                pipeline=pipeline,
                config=agent_config,
                llm=llm_manager,
                top_k=preset.top_k,
                include_sources=preset.include_sources,
            )
            runner.run_agent(
                agent=agent,
                label=label,
                run_metadata={
                    "framework": "standard_rag",
                    "run_index": run_index,
                    "model_id": args.model_id if args.llm else "none",
                    "llm_enabled": args.llm,
                    "indexed_chunks": indexed_chunks,
                    "top_k": preset.top_k,
                    "chunk_size": preset.chunk_size,
                    "chunk_overlap": preset.chunk_overlap,
                    "documents": [str(Path(item).expanduser()) for item in args.documents],
                },
            )


def _run_graph_matrix(
    runner: ExperimentRunner,
    questions: list[str],
    labels: list[str],
    runs_per_strategy: int,
    llm_manager: LLMManager | None,
    args: argparse.Namespace,
) -> None:
    if not labels:
        return

    kg_manager = KnowledgeGraphManager(build_kg_config_from_env())
    if args.seed_movie_dataset:
        inject_movie_dataset(kg_manager)

    print("Graph Schema:", kg_manager.refresh_schema())

    base_config = _base_graph_config(args=args, default_question=questions[0])

    for label in labels:
        for run_index in range(1, runs_per_strategy + 1):
            config = _graph_strategy_config(base=base_config, label=label)
            retriever = KGRetriever(kg_store=kg_manager, config=config)
            agent = KGRAGAgent(config=config, kg_retriever=retriever, llm=llm_manager)
            runner.run_agent(
                agent=agent,
                label=label,
                run_metadata={
                    "framework": "graph_rag",
                    "run_index": run_index,
                    "model_id": args.model_id if args.llm else "none",
                    "llm_enabled": args.llm,
                },
            )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.llm_warmup and not args.llm:
        parser.error("--llm-warmup requires --llm")
    if args.runs_per_strategy < 1:
        parser.error("--runs-per-strategy must be >= 1")
    if args.resource_sample_interval <= 0:
        parser.error("--resource-sample-interval must be > 0")
    if args.skip_standard and args.skip_graph:
        parser.error("Cannot skip both standard and graph matrices")

    load_dotenv(override=False)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    questions = _load_questions(args)
    if args.smoke:
        questions = questions[: max(1, int(args.smoke_questions))]

    if not questions:
        parser.error("No questions available after filtering")

    standard_labels = [] if args.skip_standard else _resolve_standard_labels(
        _parse_csv(args.smoke_standard_strategies if args.smoke else args.standard_strategies)
    )
    graph_labels = [] if args.skip_graph else _parse_csv(
        args.smoke_graph_strategies if args.smoke else args.graph_strategies
    )

    llm_manager = LLMManager(model_id=args.model_id, warmup=args.llm_warmup) if args.llm else None
    runner = ExperimentRunner(questions=questions)

    resource_monitor: ResourceMonitor | None = None
    resource_summary: dict[str, Any] | None = None
    run_error: Exception | None = None

    if args.monitor_resources:
        resource_monitor = ResourceMonitor(sample_interval_sec=args.resource_sample_interval, include_gpu=True)
        resource_monitor.start()

    try:
        _run_standard_matrix(
            runner=runner,
            questions=questions,
            labels=standard_labels,
            runs_per_strategy=args.runs_per_strategy,
            llm_manager=llm_manager,
            args=args,
        )
        _run_graph_matrix(
            runner=runner,
            questions=questions,
            labels=graph_labels,
            runs_per_strategy=args.runs_per_strategy,
            llm_manager=llm_manager,
            args=args,
        )
    except Exception as exc:
        run_error = exc
        logging.exception("Matrix execution failed. Partial outputs will still be exported.")
    finally:
        if resource_monitor is not None:
            resource_summary = resource_monitor.stop()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_tag = "retrieval_matrix_smoke" if args.smoke else "retrieval_matrix"
    tag = args.experiment_tag.strip() or default_tag

    output_dir = Path(args.output_dir) / f"{timestamp}_{tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "results.jsonl"
    csv_path = output_dir / "results.csv"
    summary_txt_path = output_dir / "summary.txt"
    summary_json_path = output_dir / "summary.json"
    resource_samples_path = output_dir / "resource_samples.jsonl"
    resource_summary_path = output_dir / "resource_summary.json"

    runner.export_jsonl(str(jsonl_path))
    runner.export_csv(str(csv_path))

    summary_text = runner.summary()
    resource_lines = _resource_summary_lines(resource_summary)
    if resource_lines:
        summary_text = summary_text + ("\n\n" if summary_text else "") + "\n".join(resource_lines)

    summary_txt_path.write_text(summary_text + "\n", encoding="utf-8")

    summary_json_path.write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "tag": tag,
                "mode": "smoke" if args.smoke else "full",
                "questions_count": len(questions),
                "runs_per_strategy": args.runs_per_strategy,
                "standard_strategies": standard_labels,
                "graph_strategies": graph_labels,
                "documents": [str(Path(item).expanduser()) for item in args.documents],
                "llm_enabled": args.llm,
                "model_id": args.model_id if args.llm else "none",
                "stats": runner.summary_stats(),
                "status": "failed" if run_error else "completed",
                "error": str(run_error) if run_error else "",
                "resource_monitor": {
                    "enabled": args.monitor_resources,
                    "sample_interval_sec": args.resource_sample_interval,
                    "summary": resource_summary,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if resource_monitor is not None:
        resource_monitor.export_samples_jsonl(str(resource_samples_path))
        resource_monitor.export_summary_json(
            str(resource_summary_path),
            extra={
                "timestamp": timestamp,
                "tag": tag,
                "status": "failed" if run_error else "completed",
                "model_id": args.model_id if args.llm else "none",
                "llm_enabled": args.llm,
            },
        )

    if run_error:
        print("\nCombined retrieval matrix failed. Partial outputs exported.")
    else:
        print("\nCombined retrieval matrix completed.")
    print("Output directory:", output_dir)
    print("-", jsonl_path)
    print("-", csv_path)
    print("-", summary_txt_path)
    print("-", summary_json_path)
    if resource_monitor is not None:
        print("-", resource_samples_path)
        print("-", resource_summary_path)
    print("\nSummary:")
    print(summary_text)

    if run_error is not None:
        raise run_error


if __name__ == "__main__":
    main()
