from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    content: str
    source: str | None = None


class TextRAGManager:
    """In-memory manager for standard textual RAG retrieval."""

    def __init__(self) -> None:
        self._chunks: list[TextChunk] = []
        self._chunk_tokens: list[list[str]] = []
        self._idf: dict[str, float] = {}

    @property
    def size(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()
        self._chunk_tokens.clear()
        self._idf.clear()

    def add_chunks(self, chunks: Iterable[TextChunk]) -> int:
        added = 0
        for chunk in chunks:
            tokens = _tokenize(chunk.content)
            if not tokens:
                continue
            self._chunks.append(chunk)
            self._chunk_tokens.append(tokens)
            added += 1

        if added:
            self._rebuild_idf()

        return added

    def add_documents(self, documents: Iterable[str], source_prefix: str = "doc") -> int:
        prepared_chunks: list[TextChunk] = []
        for index, content in enumerate(documents, start=1):
            text = content.strip()
            if not text:
                continue
            prepared_chunks.append(
                TextChunk(
                    chunk_id=f"{source_prefix}-{index}",
                    content=text,
                    source=source_prefix,
                )
            )
        return self.add_chunks(prepared_chunks)

    def retrieve_with_scores(self, query: str, top_k: int = 5) -> list[tuple[TextChunk, float]]:
        if top_k <= 0:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        query_tf: Counter[str] = Counter(query_tokens)
        scored_chunks: list[tuple[TextChunk, float]] = []

        for chunk, tokens in zip(self._chunks, self._chunk_tokens):
            score = self._score(query_tf, tokens)
            if score > 0:
                scored_chunks.append((chunk, score))

        scored_chunks.sort(key=lambda item: item[1], reverse=True)
        return scored_chunks[:top_k]

    def retrieve(self, query: str, top_k: int = 5) -> list[TextChunk]:
        return [chunk for chunk, _ in self.retrieve_with_scores(query=query, top_k=top_k)]

    def build_context(self, query: str, top_k: int = 4, separator: str = "\n\n---\n\n") -> str:
        chunks = self.retrieve(query=query, top_k=top_k)
        return separator.join(chunk.content for chunk in chunks)

    def _rebuild_idf(self) -> None:
        doc_count = len(self._chunk_tokens)
        document_frequency: Counter[str] = Counter()

        for tokens in self._chunk_tokens:
            document_frequency.update(set(tokens))

        self._idf = {
            token: math.log((1 + doc_count) / (1 + frequency)) + 1.0
            for token, frequency in document_frequency.items()
        }

    def _score(self, query_tf: Counter[str], document_tokens: list[str]) -> float:
        document_tf: Counter[str] = Counter(document_tokens)
        norm = math.sqrt(sum(freq * freq for freq in document_tf.values())) or 1.0

        score = 0.0
        for token, query_frequency in query_tf.items():
            document_frequency = document_tf.get(token, 0)
            if document_frequency == 0:
                continue

            idf = self._idf.get(token, 1.0)
            score += query_frequency * (1.0 + math.log(document_frequency)) * (idf * idf)

        return score / norm


    def _check_text():
        NotImplementedError
        
    def _score_check():
        NotImplementedError