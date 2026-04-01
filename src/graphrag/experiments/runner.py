from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any

from graphrag.agent.core import KGRAGAgent


@dataclass
class ExperimentResult:
    strategy: str
    question: str
    answer: str
    latency_ms: float
    confidence: float = 0.0
    reflection_passed: bool = True
    kg_triples_used: int = 0
    sub_questions: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class ExperimentRunner:
    def __init__(self, questions: list[str]) -> None:
        self.questions = questions
        self.results: list[ExperimentResult] = []

    def run_agent(self, agent: KGRAGAgent, label: str) -> list[ExperimentResult]:
        batch: list[ExperimentResult] = []
        for question in self.questions:
            state = agent.invoke(question)
            result = ExperimentResult(
                strategy=label,
                question=question,
                answer=state.get("answer", ""),
                latency_ms=float(state.get("latency_ms", 0.0)),
                confidence=float(state.get("confidence", 0.0)),
                reflection_passed=bool(state.get("reflection_passed", True)),
                kg_triples_used=len(state.get("kg_triples", [])) if isinstance(state.get("kg_triples", []), list) else 0,
                sub_questions=len(state.get("sub_questions", [])) if isinstance(state.get("sub_questions", []), list) else 0,
                metadata={"run_id": state.get("run_id", "")},
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

    def summary(self) -> str:
        grouped = self.compare()
        lines: list[str] = []
        for strategy, results in grouped.items():
            if not results:
                continue
            avg_latency = sum(item.latency_ms for item in results) / len(results)
            pass_rate = sum(1 for item in results if item.reflection_passed) / len(results)
            lines.append(
                f"{strategy}: runs={len(results)}, avg_latency_ms={avg_latency:.2f}, reflection_pass_rate={pass_rate:.2%}"
            )
        return "\n".join(lines)
