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
