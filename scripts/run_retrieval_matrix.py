from __future__ import annotations

import argparse
import copy
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

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
        "--checkpoint-every",
        type=int,
        default=25,
        help="Checkpoint results.jsonl/results.csv every N new executions",
    )
    parser.add_argument(
        "--resume-run-dir",
        default="",
        help="Resume from an existing run directory containing results.jsonl",
    )

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


def _run_has_pending_questions(
    runner: ExperimentRunner,
    framework: str,
    label: str,
    run_index: int,
    questions: list[str],
) -> bool:
    for question in questions:
        if not runner.has_completion(
            strategy=label,
            question=question,
            framework=framework,
            run_index=run_index,
        ):
            return True
    return False


def _count_completed_for_plan(
    runner: ExperimentRunner,
    questions: list[str],
    standard_labels: list[str],
    graph_labels: list[str],
    runs_per_strategy: int,
) -> int:
    completed = 0

    for label in standard_labels:
        for run_index in range(1, runs_per_strategy + 1):
            for question in questions:
                if runner.has_completion(
                    strategy=label,
                    question=question,
                    framework="standard_rag",
                    run_index=run_index,
                ):
                    completed += 1

    for label in graph_labels:
        for run_index in range(1, runs_per_strategy + 1):
            for question in questions:
                if runner.has_completion(
                    strategy=label,
                    question=question,
                    framework="graph_rag",
                    run_index=run_index,
                ):
                    completed += 1

    return completed


def _infer_timestamp_from_dir_name(name: str) -> str | None:
    parts = name.split("_", 2)
    if len(parts) < 2:
        return None

    day, hms = parts[0], parts[1]
    if len(day) != 8 or len(hms) != 6:
        return None
    if not day.isdigit() or not hms.isdigit():
        return None
    return f"{day}_{hms}"


def _infer_tag_from_dir_name(name: str) -> str | None:
    parts = name.split("_", 2)
    if len(parts) == 3 and _infer_timestamp_from_dir_name(name) is not None:
        return parts[2]
    return None


def _run_standard_matrix(
    runner: ExperimentRunner,
    questions: list[str],
    labels: list[str],
    runs_per_strategy: int,
    llm_manager: LLMManager | None,
    args: argparse.Namespace,
    on_result: Callable[[Any], None] | None = None,
) -> None:
    if not labels:
        return

    discovery_patterns = _parse_csv(args.doc_patterns)

    for label in labels:
        label_has_pending = any(
            _run_has_pending_questions(
                runner=runner,
                framework="standard_rag",
                label=label,
                run_index=run_index,
                questions=questions,
            )
            for run_index in range(1, runs_per_strategy + 1)
        )
        if not label_has_pending:
            logging.info("Standard strategy skip strategy=%s already fully checkpointed", label)
            continue

        logging.info("Standard strategy setup strategy=%s: indexing documents", label)
        preset = _STANDARD_STRATEGY_PRESETS[label]
        pipeline = StandardTextRAGPipeline(
            chunk_size=preset.chunk_size,
            chunk_overlap=preset.chunk_overlap,
            min_chunk_chars=preset.min_chunk_chars,
        )
        indexed_chunks = pipeline.index_paths(args.documents, discovery_patterns=discovery_patterns)
        logging.info(
            "Standard strategy ready strategy=%s indexed_chunks=%d top_k=%d chunk_size=%d chunk_overlap=%d",
            label,
            indexed_chunks,
            preset.top_k,
            preset.chunk_size,
            preset.chunk_overlap,
        )

        agent_config = AgentConfig(
            query=questions[0],
            llm_warmup=args.llm_warmup,
        )

        for run_index in range(1, runs_per_strategy + 1):
            if not _run_has_pending_questions(
                runner=runner,
                framework="standard_rag",
                label=label,
                run_index=run_index,
                questions=questions,
            ):
                logging.info(
                    "Standard run skip strategy=%s run=%d/%d already fully checkpointed",
                    label,
                    run_index,
                    runs_per_strategy,
                )
                continue

            logging.info(
                "Standard run start strategy=%s run=%d/%d questions=%d",
                label,
                run_index,
                runs_per_strategy,
                len(questions),
            )
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
                    "max_new_tokens": args.max_new_tokens if args.llm else 0,
                    "gpu_memory_fraction": args.gpu_memory_fraction if args.llm else 0.0,
                    "allow_large_model_fp16_fallback": args.allow_large_model_fp16_fallback,
                    "indexed_chunks": indexed_chunks,
                    "top_k": preset.top_k,
                    "chunk_size": preset.chunk_size,
                    "chunk_overlap": preset.chunk_overlap,
                    "documents": [str(Path(item).expanduser()) for item in args.documents],
                },
                on_result=on_result,
            )


def _run_graph_matrix(
    runner: ExperimentRunner,
    questions: list[str],
    labels: list[str],
    runs_per_strategy: int,
    llm_manager: LLMManager | None,
    args: argparse.Namespace,
    on_result: Callable[[Any], None] | None = None,
) -> None:
    if not labels:
        return

    has_pending = any(
        _run_has_pending_questions(
            runner=runner,
            framework="graph_rag",
            label=label,
            run_index=run_index,
            questions=questions,
        )
        for label in labels
        for run_index in range(1, runs_per_strategy + 1)
    )
    if not has_pending:
        logging.info("Graph matrix skip: all executions already checkpointed")
        return

    kg_manager = KnowledgeGraphManager(build_kg_config_from_env())
    if args.seed_movie_dataset:
        inject_movie_dataset(kg_manager)

    print("Graph Schema:", kg_manager.refresh_schema())

    base_config = _base_graph_config(args=args, default_question=questions[0])

    for label in labels:
        for run_index in range(1, runs_per_strategy + 1):
            if not _run_has_pending_questions(
                runner=runner,
                framework="graph_rag",
                label=label,
                run_index=run_index,
                questions=questions,
            ):
                logging.info(
                    "Graph run skip strategy=%s run=%d/%d already fully checkpointed",
                    label,
                    run_index,
                    runs_per_strategy,
                )
                continue

            logging.info(
                "Graph run start strategy=%s run=%d/%d questions=%d",
                label,
                run_index,
                runs_per_strategy,
                len(questions),
            )
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
                    "max_new_tokens": args.max_new_tokens if args.llm else 0,
                    "gpu_memory_fraction": args.gpu_memory_fraction if args.llm else 0.0,
                    "allow_large_model_fp16_fallback": args.allow_large_model_fp16_fallback,
                },
                on_result=on_result,
            )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.llm_warmup and not args.llm:
        parser.error("--llm-warmup requires --llm")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")
    if args.gpu_memory_fraction <= 0 or args.gpu_memory_fraction > 1:
        parser.error("--gpu-memory-fraction must be in (0, 1]")
    if args.runs_per_strategy < 1:
        parser.error("--runs-per-strategy must be >= 1")
    if args.resource_sample_interval <= 0:
        parser.error("--resource-sample-interval must be > 0")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be >= 1")
    if args.skip_standard and args.skip_graph:
        parser.error("Cannot skip both standard and graph matrices")

    load_dotenv(override=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

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

    default_tag = "retrieval_matrix_smoke" if args.smoke else "retrieval_matrix"
    run_started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir: Path
    resume_mode = bool(args.resume_run_dir.strip())

    if resume_mode:
        output_dir = Path(args.resume_run_dir).expanduser()
        inferred_timestamp = _infer_timestamp_from_dir_name(output_dir.name)
        run_started_at = inferred_timestamp or run_started_at
        inferred_tag = _infer_tag_from_dir_name(output_dir.name)
        tag = args.experiment_tag.strip() or inferred_tag or default_tag
    else:
        tag = args.experiment_tag.strip() or default_tag
        output_dir = Path(args.output_dir) / f"{run_started_at}_{tag}"

    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "results.jsonl"
    csv_path = output_dir / "results.csv"
    summary_txt_path = output_dir / "summary.txt"
    summary_json_path = output_dir / "summary.json"
    progress_json_path = output_dir / "checkpoint_progress.json"
    resource_samples_path = output_dir / "resource_samples.jsonl"
    resource_summary_path = output_dir / "resource_summary.json"

    existing_results = []
    if jsonl_path.exists():
        existing_results = ExperimentRunner.load_jsonl(str(jsonl_path))
        logging.info("Checkpoint preload rows=%d source=%s", len(existing_results), jsonl_path)
    elif resume_mode:
        logging.info("Resume directory has no checkpoint file yet: %s", jsonl_path)

    llm_manager = (
        LLMManager(
            model_id=args.model_id,
            warmup=args.llm_warmup,
            max_new_tokens=args.max_new_tokens,
            gpu_memory_fraction=args.gpu_memory_fraction,
            allow_large_model_fp16_fallback=args.allow_large_model_fp16_fallback,
        )
        if args.llm
        else None
    )
    runner = ExperimentRunner(questions=questions, existing_results=existing_results)

    total_strategies = len(standard_labels) + len(graph_labels)
    total_executions = len(questions) * total_strategies * args.runs_per_strategy
    completed_for_plan = _count_completed_for_plan(
        runner=runner,
        questions=questions,
        standard_labels=standard_labels,
        graph_labels=graph_labels,
        runs_per_strategy=args.runs_per_strategy,
    )

    logging.info(
        "Matrix plan questions=%d strategies=%d runs_per_strategy=%d total_executions=%d llm=%s model_id=%s",
        len(questions),
        total_strategies,
        args.runs_per_strategy,
        total_executions,
        str(args.llm),
        args.model_id if args.llm else "none",
    )
    logging.info(
        "Checkpoint plan run_dir=%s checkpoint_every=%d resume_mode=%s completed_before_start=%d remaining=%d",
        output_dir,
        args.checkpoint_every,
        str(resume_mode),
        completed_for_plan,
        max(total_executions - completed_for_plan, 0),
    )

    resource_monitor: ResourceMonitor | None = None
    resource_summary: dict[str, Any] | None = None
    run_error: BaseException | None = None
    run_status = "completed"

    def _export_outputs(status: str, error_message: str = "", checkpoint_reason: str = "") -> str:
        resource_snapshot = resource_summary
        if resource_monitor is not None and resource_snapshot is None:
            resource_snapshot = resource_monitor.summary()

        runner.export_jsonl(str(jsonl_path))
        runner.export_csv(str(csv_path))

        summary_text = runner.summary()
        resource_lines = _resource_summary_lines(resource_snapshot)
        if resource_lines:
            summary_text = summary_text + ("\n\n" if summary_text else "") + "\n".join(resource_lines)

        summary_txt_path.write_text(summary_text + "\n", encoding="utf-8")

        completion_ratio = 1.0 if total_executions == 0 else (completed_for_plan / total_executions)
        progress_payload = {
            "updated_at": datetime.now().isoformat(),
            "completed_executions": completed_for_plan,
            "total_executions": total_executions,
            "remaining_executions": max(total_executions - completed_for_plan, 0),
            "completion_ratio": completion_ratio,
            "checkpoint_every": args.checkpoint_every,
            "status": status,
            "checkpoint_reason": checkpoint_reason,
            "run_dir": str(output_dir),
        }

        summary_json_path.write_text(
            json.dumps(
                {
                    "timestamp": run_started_at,
                    "tag": tag,
                    "mode": "smoke" if args.smoke else "full",
                    "questions_count": len(questions),
                    "runs_per_strategy": args.runs_per_strategy,
                    "standard_strategies": standard_labels,
                    "graph_strategies": graph_labels,
                    "documents": [str(Path(item).expanduser()) for item in args.documents],
                    "llm_enabled": args.llm,
                    "model_id": args.model_id if args.llm else "none",
                    "max_new_tokens": args.max_new_tokens if args.llm else 0,
                    "gpu_memory_fraction": args.gpu_memory_fraction if args.llm else 0.0,
                    "allow_large_model_fp16_fallback": args.allow_large_model_fp16_fallback,
                    "stats": runner.summary_stats(),
                    "status": status,
                    "error": error_message,
                    "progress": progress_payload,
                    "checkpoint": {
                        "enabled": True,
                        "checkpoint_every": args.checkpoint_every,
                        "is_checkpoint": status == "running",
                        "reason": checkpoint_reason,
                    },
                    "resource_monitor": {
                        "enabled": args.monitor_resources,
                        "sample_interval_sec": args.resource_sample_interval,
                        "summary": resource_snapshot,
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        progress_json_path.write_text(json.dumps(progress_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        if resource_monitor is not None:
            resource_monitor.export_samples_jsonl(str(resource_samples_path))
            resource_monitor.export_summary_json(
                str(resource_summary_path),
                extra={
                    "timestamp": run_started_at,
                    "tag": tag,
                    "status": status,
                    "model_id": args.model_id if args.llm else "none",
                    "llm_enabled": args.llm,
                    "max_new_tokens": args.max_new_tokens if args.llm else 0,
                    "gpu_memory_fraction": args.gpu_memory_fraction if args.llm else 0.0,
                    "allow_large_model_fp16_fallback": args.allow_large_model_fp16_fallback,
                    "checkpoint_reason": checkpoint_reason,
                    "completed_executions": completed_for_plan,
                    "total_executions": total_executions,
                },
            )

        return summary_text

    new_results_since_checkpoint = 0

    def _on_result(_result: Any) -> None:
        nonlocal completed_for_plan, new_results_since_checkpoint

        completed_for_plan += 1
        new_results_since_checkpoint += 1
        if new_results_since_checkpoint < args.checkpoint_every:
            return

        _export_outputs(status="running", checkpoint_reason=f"periodic_every_{args.checkpoint_every}")
        logging.info(
            "Checkpoint saved completed=%d/%d (%.2f%%) dir=%s",
            completed_for_plan,
            total_executions,
            (100.0 if total_executions == 0 else (completed_for_plan * 100.0 / total_executions)),
            output_dir,
        )
        new_results_since_checkpoint = 0

    if args.monitor_resources:
        resource_monitor = ResourceMonitor(sample_interval_sec=args.resource_sample_interval, include_gpu=True)
        resource_monitor.start()

    try:
        if completed_for_plan >= total_executions:
            logging.info("All planned executions already checkpointed. Skipping execution phase.")
        else:
            _run_standard_matrix(
                runner=runner,
                questions=questions,
                labels=standard_labels,
                runs_per_strategy=args.runs_per_strategy,
                llm_manager=llm_manager,
                args=args,
                on_result=_on_result,
            )
            _run_graph_matrix(
                runner=runner,
                questions=questions,
                labels=graph_labels,
                runs_per_strategy=args.runs_per_strategy,
                llm_manager=llm_manager,
                args=args,
                on_result=_on_result,
            )
    except KeyboardInterrupt as exc:
        run_error = exc
        run_status = "interrupted"
        logging.warning("Matrix execution interrupted by user. Partial outputs will still be exported.")
    except Exception as exc:
        run_error = exc
        run_status = "failed"
        logging.exception("Matrix execution failed. Partial outputs will still be exported.")
    finally:
        if resource_monitor is not None:
            resource_summary = resource_monitor.stop()

    summary_text = _export_outputs(
        status=run_status,
        error_message=str(run_error) if run_error else "",
        checkpoint_reason="final",
    )

    if run_status == "failed":
        print("\nCombined retrieval matrix failed. Partial outputs exported.")
    elif run_status == "interrupted":
        print("\nCombined retrieval matrix interrupted. Partial outputs exported.")
    elif resume_mode and completed_for_plan >= total_executions:
        print("\nCombined retrieval matrix already complete from checkpoint data.")
    else:
        print("\nCombined retrieval matrix completed.")
    print("Output directory:", output_dir)
    print("-", jsonl_path)
    print("-", csv_path)
    print("-", summary_txt_path)
    print("-", summary_json_path)
    print("-", progress_json_path)
    if resource_monitor is not None:
        print("-", resource_samples_path)
        print("-", resource_summary_path)
    print("\nSummary:")
    print(summary_text)

    if run_error is not None:
        raise run_error


if __name__ == "__main__":
    main()
