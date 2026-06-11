from __future__ import annotations

import logging

logger = logging.getLogger("graphrag")


class ContextCompressor:
    """Bounds retrieval context to a token budget by keeping head and tail.

    The budget applies to the context only, not the full prompt (system
    message and question are added on top). When trimming occurs, the middle
    section of the context is dropped entirely — information located there is
    lost. Retrieval orders evidence by relevance, so the head carries the
    strongest signal, but raise ``max_tokens`` if mid-context evidence matters.
    """

    def __init__(self, max_tokens: int, ratio: float = 0.25) -> None:
        # ``ratio`` is tokens-per-character. Real subword tokenizers average
        # ~4 chars/token, so ~0.25 is the correct estimate. A larger value
        # over-estimates tokens and trims the context far too aggressively.
        self.max_tokens = max_tokens
        self.ratio = ratio

    def _estimate_tokens(self, text: str) -> int:
        return int(len(text) * self.ratio)

    def compress(self, text: str) -> str:
        estimated = self._estimate_tokens(text)
        if estimated <= self.max_tokens:
            return text

        char_budget = int(self.max_tokens / self.ratio)
        half = char_budget // 2
        compressed = text[:half] + "\n\n[... context trimmed ...]\n\n" + text[-half:]
        logger.warning(
            "Context compressed: %d to %d estimated tokens (middle section dropped)",
            estimated,
            self.max_tokens,
        )
        return compressed
