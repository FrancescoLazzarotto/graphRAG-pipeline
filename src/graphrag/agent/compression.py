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

    # Evidence blocks are separated by a blank line; lines are the fallback
    # boundary when a single block is larger than half the budget.
    _BLOCK_SEP = "\n\n"

    def _estimate_tokens(self, text: str) -> int:
        return int(len(text) * self.ratio)

    @staticmethod
    def _snap_head(head: str) -> str:
        """Drop a trailing partial block from ``head``."""
        for sep in (ContextCompressor._BLOCK_SEP, "\n"):
            cut = head.rfind(sep)
            if cut > 0:
                return head[:cut]
        return head

    @staticmethod
    def _snap_tail(tail: str) -> str:
        """Drop a leading partial block from ``tail``."""
        for sep in (ContextCompressor._BLOCK_SEP, "\n"):
            cut = tail.find(sep)
            if 0 <= cut < len(tail) - len(sep):
                return tail[cut + len(sep) :]
        return tail

    def compress(self, text: str) -> str:
        estimated = self._estimate_tokens(text)
        if estimated <= self.max_tokens:
            return text

        char_budget = int(self.max_tokens / self.ratio)
        half = char_budget // 2

        # Cut on block boundaries. A raw character cut landed mid-block and left
        # a half-rendered evidence entry at each seam — a reference id with a
        # truncated passage under it, or a passage with no id at all, which the
        # model then cited or mis-cited. See docs/code_audit_2026-08-15.md §1.3.
        head = self._snap_head(text[:half]) or text[:half]
        tail = self._snap_tail(text[-half:]) or text[-half:]

        compressed = head + "\n\n[... context trimmed ...]\n\n" + tail
        logger.warning(
            "Context compressed: %d to %d estimated tokens (middle section dropped)",
            estimated,
            self.max_tokens,
        )
        return compressed
