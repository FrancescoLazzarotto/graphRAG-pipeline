from __future__ import annotations

import csv
import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Protocol


class SupportsInvoke(Protocol):
    def invoke(self, question: str) -> dict[str, Any]:
        ...


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
    def __init__(self, questions: list[str]) -> None:
        self.questions = questions
        self.results: list[ExperimentResult] = []

    def run_agent(
        self,
        agent: SupportsInvoke,
        label: str,
        run_metadata: dict[str, Any] | None = None,
    ) -> list[ExperimentResult]:
        batch: list[ExperimentResult] = []
        for question in self.questions:
            state = agent.invoke(question)
            metadata: dict[str, Any] = {"run_id": state.get("run_id", "")}
            if run_metadata:
                metadata.update(run_metadata)
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
            batch.append(result)

        self.results.extend(batch)
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
