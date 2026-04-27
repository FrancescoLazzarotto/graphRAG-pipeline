from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import NamedTuple

from tqdm import tqdm

from kg_pipeline.models.types import ChunkRecord, DocumentRecord


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class ParagraphUnit(NamedTuple):
    text: str
    page_number: int
    section_title: str


def _token_count(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


def _split_paragraphs(text: str) -> list[str]:
    parts = [part.strip() for part in text.split("\n\n")]
    return [part for part in parts if part]


def _paragraphs_for_range(
    doc: DocumentRecord,
    start_page: int,
    end_page: int,
    section_title: str,
) -> list[ParagraphUnit]:
    units: list[ParagraphUnit] = []
    for page in doc.page_chunks:
        if page.page_number < start_page or page.page_number > end_page:
            continue
        for paragraph in _split_paragraphs(page.text):
            units.append(ParagraphUnit(paragraph, page.page_number, section_title))
    return units


def _window_paragraphs(
    paragraphs: list[ParagraphUnit],
    max_tokens: int,
    overlap_tokens: int,
) -> list[list[ParagraphUnit]]:
    windows: list[list[ParagraphUnit]] = []
    if not paragraphs:
        return windows

    idx = 0
    while idx < len(paragraphs):
        current: list[ParagraphUnit] = []
        token_budget = 0
        j = idx

        while j < len(paragraphs):
            p_tokens = _token_count(paragraphs[j].text)
            if current and token_budget + p_tokens > max_tokens:
                break
            current.append(paragraphs[j])
            token_budget += p_tokens
            j += 1

        if not current:
            current = [paragraphs[idx]]
            j = idx + 1

        windows.append(current)
        if j >= len(paragraphs):
            break

        overlap = 0
        back = j - 1
        while back >= idx and overlap < overlap_tokens:
            overlap += _token_count(paragraphs[back].text)
            back -= 1
        idx = max(back + 1, idx + 1)

    return windows


def _build_chunk(
    doc: DocumentRecord,
    chunk_index: int,
    section_title: str,
    paragraphs: list[ParagraphUnit],
) -> ChunkRecord:
    page_numbers = [p.page_number for p in paragraphs]
    start_page = min(page_numbers)
    end_page = max(page_numbers)
    text = "\n\n".join(p.text for p in paragraphs)

    return ChunkRecord(
        doc_id=doc.doc_id,
        filename=doc.filename,
        chunk_id=f"{doc.doc_id}_chunk_{chunk_index:05d}",
        page_range=f"{start_page}-{end_page}",
        section_title=section_title,
        chunk_index=chunk_index,
        text=text,
    )


def chunk_documents(docs: list[DocumentRecord], config: dict) -> list[ChunkRecord]:
    chunk_cfg = config["chunking"]

    small_max_pages = int(chunk_cfg["small_max_pages"])
    medium_max_pages = int(chunk_cfg["medium_max_pages"])
    small_min_tokens = int(chunk_cfg["small_min_tokens"])
    small_max_tokens = int(chunk_cfg["small_max_tokens"])
    medium_window = int(chunk_cfg["medium_window_tokens"])
    medium_overlap = int(chunk_cfg["medium_overlap_tokens"])
    large_window = int(chunk_cfg["large_window_tokens"])
    large_overlap = int(chunk_cfg["large_overlap_tokens"])

    chunks: list[ChunkRecord] = []

    for doc in tqdm(docs, desc="Stage 1 Chunking", unit="doc"):
        next_chunk_idx = 1

        if doc.page_count <= small_max_pages:
            paragraphs = _paragraphs_for_range(
                doc=doc,
                start_page=1,
                end_page=doc.page_count,
                section_title="SmallDoc",
            )
            for p in paragraphs:
                t = _token_count(p.text)
                if t < small_min_tokens:
                    continue
                if t <= small_max_tokens:
                    chunks.append(_build_chunk(doc, next_chunk_idx, p.section_title, [p]))
                    next_chunk_idx += 1
                else:
                    windows = _window_paragraphs([p], max_tokens=small_max_tokens, overlap_tokens=0)
                    for win in windows:
                        chunks.append(_build_chunk(doc, next_chunk_idx, p.section_title, win))
                        next_chunk_idx += 1
            continue

        if doc.page_count <= medium_max_pages:
            for section in doc.sections:
                paragraphs = _paragraphs_for_range(
                    doc=doc,
                    start_page=section.start_page,
                    end_page=section.end_page,
                    section_title=section.title,
                )
                windows = _window_paragraphs(paragraphs, max_tokens=medium_window, overlap_tokens=medium_overlap)
                for win in windows:
                    chunks.append(_build_chunk(doc, next_chunk_idx, section.title, win))
                    next_chunk_idx += 1
            continue

        top_sections = [s for s in doc.sections if s.level == 1] or doc.sections
        for section in top_sections:
            paragraphs = _paragraphs_for_range(
                doc=doc,
                start_page=section.start_page,
                end_page=section.end_page,
                section_title=section.title,
            )
            windows = _window_paragraphs(paragraphs, max_tokens=large_window, overlap_tokens=large_overlap)
            for win in windows:
                chunks.append(_build_chunk(doc, next_chunk_idx, section.title, win))
                next_chunk_idx += 1

    return chunks


def save_chunks(path: Path, chunks: list[ChunkRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [chunk.model_dump() for chunk in chunks]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_chunks(path: Path) -> list[ChunkRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [ChunkRecord.model_validate(item) for item in payload]


def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents-json", required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    docs_payload = json.loads(Path(args.documents_json).read_text(encoding="utf-8"))
    docs = [DocumentRecord.model_validate(item) for item in docs_payload]
    config = json.loads(Path(args.config_json).read_text(encoding="utf-8"))

    chunks = chunk_documents(docs, config)
    save_chunks(Path(args.output_json), chunks)


if __name__ == "__main__":
    _cli()
