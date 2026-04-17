from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from graphrag.text_rag.manager import TextChunk, TextRAGManager

try:
    import fitz  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - runtime dependency check
    fitz = None

_WHITESPACE_RE = re.compile(r"\s+")
_SUPPORTED_TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".log",
    ".csv",
}
_DEFAULT_DISCOVERY_PATTERNS = (
    "*.pdf",
    "*.txt",
    "*.md",
    "*.markdown",
)


def _normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


@dataclass(frozen=True)
class RetrievedTextChunk:
    chunk_id: str
    source: str | None
    content: str
    score: float


class StandardTextRAGPipeline:
    """Basic document retrieval pipeline for standard (text-only) RAG."""

    def __init__(
        self,
        retriever: TextRAGManager | None = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 180,
        min_chunk_chars: int = 80,
    ) -> None:
        if chunk_size < 128:
            raise ValueError("chunk_size must be >= 128")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if min_chunk_chars < 1:
            raise ValueError("min_chunk_chars must be >= 1")

        self.retriever = retriever or TextRAGManager()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_chars = min_chunk_chars

    @property
    def indexed_chunks(self) -> int:
        return self.retriever.size

    def clear(self) -> None:
        self.retriever.clear()

    def index_paths(
        self,
        paths: Sequence[str | Path],
        discovery_patterns: Sequence[str] | None = None,
    ) -> int:
        files_to_index = self._resolve_paths(paths, discovery_patterns=discovery_patterns)
        prepared_chunks: list[TextChunk] = []

        for doc_index, file_path in enumerate(files_to_index, start=1):
            sections = self._load_sections_from_path(file_path)
            for section_index, (source_tag, section_text) in enumerate(sections, start=1):
                chunk_texts = self._split_into_chunks(section_text)
                for chunk_index, chunk_text in enumerate(chunk_texts, start=1):
                    chunk_id = f"d{doc_index:04d}-s{section_index:04d}-c{chunk_index:04d}"
                    chunk_source = f"{source_tag}#chunk={chunk_index}"
                    prepared_chunks.append(
                        TextChunk(
                            chunk_id=chunk_id,
                            content=chunk_text,
                            source=chunk_source,
                        )
                    )

        return self.retriever.add_chunks(prepared_chunks)

    def index_directory(
        self,
        root: str | Path,
        discovery_patterns: Sequence[str] | None = None,
    ) -> int:
        return self.index_paths([root], discovery_patterns=discovery_patterns)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedTextChunk]:
        items = self.retriever.retrieve_with_scores(query=query, top_k=top_k)
        return [
            RetrievedTextChunk(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                content=chunk.content,
                score=score,
            )
            for chunk, score in items
        ]

    def build_context(
        self,
        query: str,
        top_k: int = 4,
        include_sources: bool = True,
        separator: str = "\n\n---\n\n",
    ) -> str:
        retrieved = self.retrieve(query=query, top_k=top_k)
        if include_sources:
            rendered = []
            for item in retrieved:
                if item.source:
                    rendered.append(f"Source: {item.source}\n{item.content}")
                else:
                    rendered.append(item.content)
            return separator.join(rendered)

        return separator.join(item.content for item in retrieved)

    def _resolve_paths(
        self,
        paths: Sequence[str | Path],
        discovery_patterns: Sequence[str] | None,
    ) -> list[Path]:
        patterns = tuple(discovery_patterns) if discovery_patterns else _DEFAULT_DISCOVERY_PATTERNS
        resolved: list[Path] = []

        for raw_path in paths:
            current = Path(raw_path).expanduser().resolve()
            if not current.exists():
                raise FileNotFoundError(f"Path does not exist: {current}")

            if current.is_file():
                resolved.append(current)
                continue

            for pattern in patterns:
                resolved.extend(current.rglob(pattern))

        unique_files = sorted({path.resolve() for path in resolved if path.is_file()})
        if not unique_files:
            raise ValueError("No files discovered for indexing")
        return unique_files

    def _load_sections_from_path(self, file_path: Path) -> list[tuple[str, str]]:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self._load_pdf_sections(file_path)

        if suffix in _SUPPORTED_TEXT_SUFFIXES:
            text = _normalize_text(file_path.read_text(encoding="utf-8", errors="ignore"))
            if not text:
                return []
            return [(str(file_path), text)]

        return []

    def _load_pdf_sections(self, file_path: Path) -> list[tuple[str, str]]:
        if fitz is None:
            raise RuntimeError(
                "PyMuPDF is required for PDF ingestion. Install with: pip install pymupdf"
            )

        sections: list[tuple[str, str]] = []
        with fitz.open(file_path) as document:
            for page_number, page in enumerate(document, start=1):
                page_text = _normalize_text(page.get_text("text"))
                if not page_text:
                    continue
                source_tag = f"{file_path}#page={page_number}"
                sections.append((source_tag, page_text))
        return sections

    def _split_into_chunks(self, text: str) -> list[str]:
        normalized = _normalize_text(text)
        if len(normalized) < self.min_chunk_chars:
            return []

        if len(normalized) <= self.chunk_size:
            return [normalized]

        chunks: list[str] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0

        while start < len(normalized):
            end = min(len(normalized), start + self.chunk_size)
            candidate = normalized[start:end].strip()
            if len(candidate) >= self.min_chunk_chars:
                chunks.append(candidate)
            if end >= len(normalized):
                break
            start += step

        return chunks
