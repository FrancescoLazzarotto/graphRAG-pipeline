from __future__ import annotations

import logging

logger = logging.getLogger("graphrag")


class ContextCompressor:
    def __init__(self, max_tokens: int, ratio: float = 0.75) -> None:
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
        logger.info("Context compressed: %d to %d estimated tokens", estimated, self.max_tokens)
        return compressed
