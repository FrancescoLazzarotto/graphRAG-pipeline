from __future__ import annotations

# Phrase-level markers (lowercased substring match) signalling that the model
# declined to answer or asked for more context, in English or Italian.
# These are deliberately multi-word phrases: single common words such as
# "context" or "information" must NOT be added here, otherwise legitimate
# answers that merely mention those words get misclassified as refusals.
_REFUSAL_MARKERS: tuple[str, ...] = (
    # English
    "context is insufficient",
    "provide additional context",
    "specific details regarding the question",
    "specific details regarding the context",
    "without a specific question",
    "without a specific question or detailed context",
    "without these elements",
    "without these elements, crafting",
    "could you specify the question",
    "serve specific details",
    "crucial to first establish",
    "not feasible",
    "challenging to construct",
    "the current context does not provide sufficient information",
    # Italian
    "non ho abbastanza contesto",
    "contesto fornito e insufficiente",
    "contesto insufficiente",
    "ho bisogno di ulteriori informazioni",
)


def looks_like_refusal(text: str) -> bool:
    """Return True when ``text`` is empty or matches a known refusal phrase.

    Args:
        text: Candidate answer produced by the LLM.

    Returns:
        True if the answer is blank or contains a known refusal/insufficient-context
        phrase, otherwise False.
    """
    if not text or not str(text).strip():
        return True
    lowered = str(text).lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


# Markers for the *insufficiency metric* (`insufficient_answer`). This is a
# DISTINCT concept from a model refusal: it captures "no factual evidence /
# cannot find an answer" responses, including the agent's own canonical
# no-evidence fallbacks. It deliberately does NOT include the generic LLM
# hedging in `_REFUSAL_MARKERS` (e.g. "not feasible", "challenging to
# construct"), which would inflate the metric with false positives.
#
# This is the single source of truth — `graphrag.experiments.runner` and
# `evalkit.io.run_loader` import `is_insufficient` from here. Never re-mirror
# this list elsewhere; the previous duplicated copies drifted and silently
# missed the agent's own fallback message.
_INSUFFICIENT_MARKERS: tuple[str, ...] = (
    # LLM-produced "no evidence in context" phrasings
    "the provided context does not contain",
    "the context does not contain",
    "does not contain enough information",
    "does not contain information",
    "i don't have enough information",
    "i cannot find",
    "cannot answer",
    "unable to answer",
    "not enough information",
    "no information available",
    "no relevant information",
    # Italian equivalents
    "non ho informazioni",
    "non posso rispondere",
    "il contesto fornito non contiene",
    "il contesto non contiene",
    # Agent-emitted canonical fallbacks (graphrag.agent.core) — these were
    # missed by the old metric, undercounting true insufficiency.
    "context is insufficient",
    "too sparse to build a reliable answer",
    "troppo scarno per costruire una risposta",
)


def is_insufficient(text: str) -> bool:
    """Return True when ``text`` signals an insufficient / no-evidence answer.

    Used to compute the ``insufficient_answer`` experiment metric. Distinct from
    :func:`looks_like_refusal`: this matches only "no factual evidence" / "cannot
    find an answer" phrasings (plus the agent's own fallback messages), not
    generic refusal hedging.

    Args:
        text: Candidate answer produced by the agent or LLM.

    Returns:
        True if the answer is blank or contains a known insufficiency phrase.
    """
    if not text or not str(text).strip():
        return True
    lowered = str(text).lower()
    return any(marker in lowered for marker in _INSUFFICIENT_MARKERS)
