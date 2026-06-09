from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rubric:
    name: str
    description: str
    score_field: str  # JSON key to look for in judge output
    rationale_field: str = "rationale"


RUBRICS: dict[str, Rubric] = {
    "answer_correctness": Rubric(
        name="answer_correctness",
        score_field="correctness_score",
        description=(
            "Does the generated answer correctly answer the question compared to the ground truth? "
            "Evaluate factual accuracy and completeness. "
            "Score 1 = completely correct and complete, 0 = completely wrong or missing."
        ),
    ),
    "groundedness": Rubric(
        name="groundedness",
        score_field="groundedness_score",
        description=(
            "Is every claim in the generated answer grounded in the provided contexts or retrieved triples? "
            "Penalise any claim that cannot be inferred from the provided evidence. "
            "Score 1 = fully grounded, 0 = mostly hallucinated."
        ),
    ),
    "relevance": Rubric(
        name="relevance",
        score_field="relevance_score",
        description=(
            "Does the generated answer directly address the question asked? "
            "Penalise answers that are off-topic or that answer a different question. "
            "Score 1 = perfectly on-topic, 0 = completely irrelevant."
        ),
    ),
    "completeness": Rubric(
        name="completeness",
        score_field="completeness_score",
        description=(
            "Does the generated answer cover all relevant aspects of the expected answer? "
            "Penalise partial answers that omit important information present in the ground truth. "
            "Score 1 = fully complete, 0 = minimal or no useful content."
        ),
    ),
}


def get_rubric(name: str) -> Rubric:
    """Return a Rubric by name.

    Raises:
        ValueError: If the rubric name is unknown.
    """
    if name not in RUBRICS:
        available = ", ".join(sorted(RUBRICS))
        raise ValueError(f"Unknown rubric '{name}'. Available: {available}")
    return RUBRICS[name]
