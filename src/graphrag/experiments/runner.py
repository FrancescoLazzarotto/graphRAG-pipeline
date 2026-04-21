from __future__ import annotations

import csv
import dataclasses
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


class SupportsInvoke(Protocol):
    def invoke(self, question: str) -> dict[str, Any]:
        ...


logger = logging.getLogger("graphrag.experiments")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class ExperimentResult:
    strategy: str
    question: str
    answer: str
    latency_ms: float
    confidence: float = 0.0
    reflection_passed: bool = True
    kg_triples_used: int = 0
    kg_neighbors_used: int = 0
    kg_subgraph_triples_used: int = 0
    kg_shortest_path_triples_used: int = 0
    sub_questions: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class ExperimentRunner:
    def __init__(self, questions: list[str], existing_results: list[ExperimentResult] | None = None) -> None:
        self.questions = questions
        self.results: list[ExperimentResult] = []
        self._completed_keys: set[tuple[str, str, str, int]] = set()

        if existing_results:
            loaded = self.add_existing_results(existing_results)
            logger.info("Loaded %d existing checkpoint rows", loaded)

    @staticmethod
    def completion_key(strategy: str, question: str, framework: str, run_index: int) -> tuple[str, str, str, int]:
        return (strategy, question, framework, int(run_index))

    def _result_completion_key(self, result: ExperimentResult) -> tuple[str, str, str, int]:
        framework = str(result.metadata.get("framework", "unknown"))
        run_index = _safe_int(result.metadata.get("run_index", 1), default=1)
        return self.completion_key(
            strategy=result.strategy,
            question=result.question,
            framework=framework,
            run_index=run_index,
        )

    def has_completion(self, strategy: str, question: str, framework: str, run_index: int) -> bool:
        return self.completion_key(strategy, question, framework, run_index) in self._completed_keys

    def completed_count(self) -> int:
        return len(self._completed_keys)

    def add_existing_results(self, results: list[ExperimentResult]) -> int:
        loaded = 0
        duplicates = 0
        for result in results:
            completion_key = self._result_completion_key(result)
            if completion_key in self._completed_keys:
                duplicates += 1
                continue
            self.results.append(result)
            self._completed_keys.add(completion_key)
            loaded += 1

        if duplicates:
            logger.warning("Skipped %d duplicate checkpoint rows", duplicates)
        return loaded

    @staticmethod
    def load_jsonl(path: str) -> list[ExperimentResult]:
        rows: list[ExperimentResult] = []

        with open(path, "r", encoding="utf-8") as input_file:
            for line_number, raw_line in enumerate(input_file, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON at %s:%d", path, line_number)
                    continue

                if not isinstance(payload, dict):
                    logger.warning("Skipping non-object payload at %s:%d", path, line_number)
                    continue

                metadata = payload.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}

                rows.append(
                    ExperimentResult(
                        strategy=str(payload.get("strategy", "")),
                        question=str(payload.get("question", "")),
                        answer=str(payload.get("answer", "")),
                        latency_ms=_safe_float(payload.get("latency_ms", 0.0), default=0.0),
                        confidence=_safe_float(payload.get("confidence", 0.0), default=0.0),
                        reflection_passed=bool(payload.get("reflection_passed", True)),
                        kg_triples_used=_safe_int(payload.get("kg_triples_used", 0), default=0),
                        kg_neighbors_used=_safe_int(payload.get("kg_neighbors_used", 0), default=0),
                        kg_subgraph_triples_used=_safe_int(payload.get("kg_subgraph_triples_used", 0), default=0),
                        kg_shortest_path_triples_used=_safe_int(payload.get("kg_shortest_path_triples_used", 0), default=0),
                        sub_questions=_safe_int(payload.get("sub_questions", 0), default=0),
                        metadata=metadata,
                    )
                )

        return rows

    def run_agent(
        self,
        agent: SupportsInvoke,
        label: str,
        run_metadata: dict[str, Any] | None = None,
        on_result: Callable[[ExperimentResult], None] | None = None,
    ) -> list[ExperimentResult]:
        batch: list[ExperimentResult] = []
        total_questions = len(self.questions)
        framework = str(run_metadata.get("framework", "unknown")) if run_metadata else "unknown"
        run_index = _safe_int(run_metadata.get("run_index", 1), default=1) if run_metadata else 1

        for question_index, question in enumerate(self.questions, start=1):
            completion_key = self.completion_key(
                strategy=label,
                question=question,
                framework=framework,
                run_index=run_index,
            )
            if completion_key in self._completed_keys:
                logger.info(
                    "Progress skip framework=%s strategy=%s run=%s question=%d/%d (already checkpointed)",
                    framework,
                    label,
                    run_index,
                    question_index,
                    total_questions,
                )
                continue

            logger.info(
                "Progress start framework=%s strategy=%s run=%s question=%d/%d",
                framework,
                label,
                run_index,
                question_index,
                total_questions,
            )

            question_start = time.perf_counter()
            state = agent.invoke(question)
            invoke_latency_ms = (time.perf_counter() - question_start) * 1000.0
            metadata: dict[str, Any] = {"run_id": state.get("run_id", "")}
            if run_metadata:
                metadata.update(run_metadata)
            metadata.setdefault("framework", framework)
            metadata.setdefault("run_index", run_index)
            result = ExperimentResult(
                strategy=label,
                question=question,
                answer=state.get("answer", ""),
                latency_ms=float(state.get("latency_ms", 0.0)),
                confidence=float(state.get("confidence", 0.0)),
                reflection_passed=bool(state.get("reflection_passed", True)),
                kg_triples_used=len(state.get("kg_triples", [])) if isinstance(state.get("kg_triples", []), list) else 0,
                kg_neighbors_used=int(state.get("retrieved_neighbors_count", 0) or 0),
                kg_subgraph_triples_used=int(state.get("retrieved_subgraph_count", 0) or 0),
                kg_shortest_path_triples_used=int(state.get("retrieved_shortest_path_count", 0) or 0),
                sub_questions=len(state.get("sub_questions", [])) if isinstance(state.get("sub_questions", []), list) else 0,
                metadata=metadata,
            )

            logger.info(
                "Progress done framework=%s strategy=%s run=%s question=%d/%d latency_ms=%.2f invoke_ms=%.2f",
                framework,
                label,
                run_index,
                question_index,
                total_questions,
                result.latency_ms,
                invoke_latency_ms,
            )

            self.results.append(result)
            self._completed_keys.add(completion_key)
            if on_result is not None:
                on_result(result)

            batch.append(result)
        return batch

    def compare(self) -> dict[str, list[ExperimentResult]]:
        grouped: dict[str, list[ExperimentResult]] = {}
        for result in self.results:
            grouped.setdefault(result.strategy, []).append(result)
        return grouped

    def export_jsonl(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as output_file:
            for result in self.results:
                output_file.write(json.dumps(dataclasses.asdict(result), ensure_ascii=False) + "\n")

    def export_csv(self, path: str) -> None:
        with open(path, "w", encoding="utf-8", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(
                [
                    "strategy",
                    "question",
                    "answer",
                    "latency_ms",
                    "confidence",
                    "reflection_passed",
                    "kg_triples_used",
                    "kg_neighbors_used",
                    "kg_subgraph_triples_used",
                    "kg_shortest_path_triples_used",
                    "sub_questions",
                    "metadata_json",
                ]
            )
            for result in self.results:
                writer.writerow(
                    [
                        result.strategy,
                        result.question,
                        result.answer,
                        f"{result.latency_ms:.6f}",
                        f"{result.confidence:.6f}",
                        str(result.reflection_passed),
                        result.kg_triples_used,
                        result.kg_neighbors_used,
                        result.kg_subgraph_triples_used,
                        result.kg_shortest_path_triples_used,
                        result.sub_questions,
                        json.dumps(result.metadata, ensure_ascii=False, sort_keys=True),
                    ]
                )

    def summary_stats(self) -> dict[str, dict[str, float | int]]:
        grouped = self.compare()
        summary: dict[str, dict[str, float | int]] = {}
        for strategy, results in grouped.items():
            if not results:
                continue
            count = len(results)
            summary[strategy] = {
                "runs": count,
                "avg_latency_ms": sum(item.latency_ms for item in results) / count,
                "avg_confidence": sum(item.confidence for item in results) / count,
                "reflection_pass_rate": sum(1 for item in results if item.reflection_passed) / count,
                "avg_kg_triples_used": sum(item.kg_triples_used for item in results) / count,
                "avg_kg_neighbors_used": sum(item.kg_neighbors_used for item in results) / count,
                "avg_kg_subgraph_triples_used": sum(item.kg_subgraph_triples_used for item in results) / count,
                "avg_kg_shortest_path_triples_used": sum(item.kg_shortest_path_triples_used for item in results) / count,
                "avg_sub_questions": sum(item.sub_questions for item in results) / count,
            }
        return summary

    def summary(self) -> str:
        lines: list[str] = []
        for strategy, stats in self.summary_stats().items():
            lines.append(
                f"{strategy}: runs={int(stats['runs'])}, "
                f"avg_latency_ms={float(stats['avg_latency_ms']):.2f}, "
                f"reflection_pass_rate={float(stats['reflection_pass_rate']):.2%}, "
                f"avg_kg_triples_used={float(stats['avg_kg_triples_used']):.2f}, "
                f"avg_kg_neighbors_used={float(stats['avg_kg_neighbors_used']):.2f}, "
                f"avg_kg_subgraph_triples_used={float(stats['avg_kg_subgraph_triples_used']):.2f}, "
                f"avg_kg_shortest_path_triples_used={float(stats['avg_kg_shortest_path_triples_used']):.2f}, "
                f"avg_sub_questions={float(stats['avg_sub_questions']):.2f}"
            )
        return "\n".join(lines)
