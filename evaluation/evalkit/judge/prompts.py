from __future__ import annotations

import json

from evalkit.judge.rubrics import Rubric
from evalkit.models import EvalRow


SYSTEM_TEMPLATE = """\
You are an expert evaluator for question-answering systems.
You will be given a question, a generated answer, optional reference information \
(ground truth, retrieved context), and a scoring rubric.
Your task is to evaluate the generated answer according to the rubric and return a \
JSON object with the score and a brief rationale.

Respond ONLY with a JSON object. Do not add any text before or after the JSON.
Example format:
{{
  "{score_field}": <float between 0 and 1>,
  "rationale": "<one or two sentences explaining the score>"
}}
"""

USER_TEMPLATE = """\
## Question
{question}

## Generated Answer
{answer}
{ground_truth_section}
{context_section}

## Rubric
{rubric_description}

Return only the JSON object with keys "{score_field}" and "rationale".
"""


def build_row_content(row: EvalRow) -> str:
    """Assemble the per-row evaluation block (question, answer, references).

    Shared by the single-rubric prompt and the batched multi-row prompt so both
    feed the judge identical evidence.

    Args:
        row: EvalRow with question, answer, ground_truth, contexts, triples.

    Returns:
        A markdown string with the row's question, generated answer, and any
        available ground truth / retrieved contexts / retrieved triples.
    """
    parts = [
        f"## Question\n{row.question}",
        f"## Generated Answer\n{row.answer}",
    ]
    if row.ground_truth.strip():
        parts.append(f"## Ground Truth Answer\n{row.ground_truth}")
    if row.contexts:
        parts.append("## Retrieved Text Contexts\n" + "\n---\n".join(row.contexts[:5]))
    if row.retrieved_triples:
        triples_str = json.dumps(row.retrieved_triples[:20], ensure_ascii=False, indent=2)
        parts.append(f"## Retrieved KG Triples\n{triples_str}")
    return "\n\n".join(parts)


def build_prompt(row: EvalRow, rubric: Rubric) -> tuple[str, str]:
    """Build (system, user) prompt strings for a given row and rubric.

    Args:
        row: EvalRow with question, answer, ground_truth, contexts.
        rubric: Rubric defining what to evaluate.

    Returns:
        (system_prompt, user_prompt)
    """
    system = SYSTEM_TEMPLATE.format(score_field=rubric.score_field)

    gt_section = ""
    if row.ground_truth.strip():
        gt_section = f"\n## Ground Truth Answer\n{row.ground_truth}"

    context_parts: list[str] = []
    if row.contexts:
        context_parts.append("## Retrieved Text Contexts\n" + "\n---\n".join(row.contexts[:5]))
    if row.retrieved_triples:
        triples_str = json.dumps(row.retrieved_triples[:20], ensure_ascii=False, indent=2)
        context_parts.append(f"## Retrieved KG Triples\n{triples_str}")
    context_section = ("\n\n" + "\n\n".join(context_parts)) if context_parts else ""

    user = USER_TEMPLATE.format(
        question=row.question,
        answer=row.answer,
        ground_truth_section=gt_section,
        context_section=context_section,
        rubric_description=rubric.description,
        score_field=rubric.score_field,
    )

    return system, user
