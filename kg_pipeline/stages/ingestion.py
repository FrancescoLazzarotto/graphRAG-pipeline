from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import fitz
import pymupdf4llm
from tqdm import tqdm

from kg_pipeline.models.types import DocumentRecord, PageChunkRecord, SectionRecord


_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _doc_id_from_filename(filename: str) -> str:
    stem = Path(filename).stem.lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return cleaned or "document"


def _read_page_chunks(pdf_path: Path) -> list[PageChunkRecord]:
    chunks: list[PageChunkRecord] = []

    try:
        raw = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    except TypeError:
        raw = None

    if isinstance(raw, list) and raw:
        for idx, item in enumerate(raw, start=1):
            if isinstance(item, dict):
                meta = item.get("metadata", {})
                page_num = int(meta.get("page", idx))
                text = str(item.get("text", ""))
            else:
                page_num = idx
                text = str(item)
            chunks.append(PageChunkRecord(page_number=page_num, text=text))
        return chunks

    with fitz.open(pdf_path) as doc:
        for page_no in range(1, len(doc) + 1):
            text = str(pymupdf4llm.to_markdown(str(pdf_path), pages=[page_no - 1]))
            chunks.append(PageChunkRecord(page_number=page_no, text=text))

    return chunks


def _extract_sections(page_chunks: list[PageChunkRecord]) -> list[SectionRecord]:
    starts: list[tuple[int, int, str]] = []

    for page in page_chunks:
        for line in page.text.splitlines():
            match = _HEADER_RE.match(line.strip())
            if not match:
                continue
            level = len(match.group(1))
            title = match.group(2).strip()
            starts.append((page.page_number, level, title))
            break

    if not starts:
        return [
            SectionRecord(
                title="Full Document",
                level=1,
                start_page=1,
                end_page=max(1, page_chunks[-1].page_number if page_chunks else 1),
            )
        ]

    sections: list[SectionRecord] = []
    for idx, (start_page, level, title) in enumerate(starts):
        if idx < len(starts) - 1:
            end_page = max(start_page, starts[idx + 1][0] - 1)
        else:
            end_page = max(start_page, page_chunks[-1].page_number)
        sections.append(
            SectionRecord(
                title=title,
                level=level,
                start_page=start_page,
                end_page=end_page,
            )
        )
    return sections


def _extract_title_and_year(page_chunks: list[PageChunkRecord], fallback_title: str) -> tuple[str, int | None]:
    title = fallback_title
    publication_year: int | None = None

    head_text = "\n".join(chunk.text for chunk in page_chunks[: min(3, len(page_chunks))])

    for line in head_text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = _HEADER_RE.match(line)
        if match and len(match.group(1)) == 1:
            title = match.group(2).strip()
            break
        if title == fallback_title:
            title = line
            break

    year_match = _YEAR_RE.search(head_text)
    if year_match:
        publication_year = int(year_match.group(0))

    return title, publication_year


def ingest_documents(input_dir: Path, single_doc: str | None = None) -> list[DocumentRecord]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if single_doc:
        pdf_paths = [input_dir / single_doc]
    else:
        pdf_paths = sorted(input_dir.glob("*.pdf"))

    if not pdf_paths:
        raise ValueError(f"No PDF files found in {input_dir}")

    docs: list[DocumentRecord] = []

    for pdf_path in tqdm(pdf_paths, desc="Stage 0 Ingestion", unit="doc"):
        if not pdf_path.exists():
            continue

        with fitz.open(pdf_path) as doc:
            page_count = len(doc)

        page_chunks = _read_page_chunks(pdf_path)
        markdown_text = "\n\n".join(chunk.text for chunk in page_chunks)
        sections = _extract_sections(page_chunks)
        title, publication_year = _extract_title_and_year(page_chunks, fallback_title=pdf_path.stem)

        docs.append(
            DocumentRecord(
                doc_id=_doc_id_from_filename(pdf_path.name),
                filename=pdf_path.name,
                page_count=page_count,
                markdown_text=markdown_text,
                sections=sections,
                page_chunks=page_chunks,
                title=title,
                publication_year=publication_year,
            )
        )

    return docs


def save_documents(path: Path, docs: list[DocumentRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [doc.model_dump() for doc in docs]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_documents(path: Path) -> list[DocumentRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [DocumentRecord.model_validate(item) for item in payload]


def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--single-doc", default=None)
    args = parser.parse_args()

    docs = ingest_documents(Path(args.input_dir), single_doc=args.single_doc)
    save_documents(Path(args.output_json), docs)


if __name__ == "__main__":
    _cli()
