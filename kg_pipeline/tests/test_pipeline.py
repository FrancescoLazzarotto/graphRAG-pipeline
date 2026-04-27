from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from kg_pipeline.models.types import DocumentRecord, PageChunkRecord, SectionRecord
from kg_pipeline.stages.chunking import chunk_documents
from kg_pipeline.stages.ingestion import ingest_documents
from kg_pipeline.utils.validation import validate_triples


def test_schema_validation_accepts_valid_triple():
    payload = [
        {
            "subject": "Europe",
            "predicate": "HAS_VALUE",
            "object": "2.7 C",
            "subject_labels": ["Region"],
            "object_labels": ["DataValue"],
            "subject_properties": {"name": "Europe"},
            "object_properties": {"name": "2.7 C"},
            "relationship_properties": {
                "source_doc": "demo.pdf",
                "extraction_method": "llm",
                "value": 2.7,
                "unit": "C",
                "year": 2025,
            },
        }
    ]
    triples = validate_triples(payload)
    assert len(triples) == 1
    assert triples[0].predicate == "HAS_VALUE"


def test_schema_validation_rejects_bad_predicate():
    payload = [
        {
            "subject": "Europe",
            "predicate": "has_value",
            "object": "2.7 C",
            "subject_labels": ["Region"],
            "object_labels": ["DataValue"],
            "subject_properties": {"name": "Europe"},
            "object_properties": {"name": "2.7 C"},
            "relationship_properties": {"source_doc": "demo.pdf", "extraction_method": "llm"},
        }
    ]
    with pytest.raises(Exception):
        validate_triples(payload)


def test_chunking_metadata_fields_present():
    doc = DocumentRecord(
        doc_id="demo_doc",
        filename="demo.pdf",
        page_count=2,
        markdown_text="# Intro\n\nParagraph one.\n\nParagraph two.",
        sections=[SectionRecord(title="Intro", level=1, start_page=1, end_page=2)],
        page_chunks=[
            PageChunkRecord(page_number=1, text="# Intro\n\nParagraph one."),
            PageChunkRecord(page_number=2, text="Paragraph two."),
        ],
        title="Intro",
        publication_year=2025,
    )
    config = {
        "chunking": {
            "small_max_pages": 10,
            "medium_max_pages": 80,
            "small_min_tokens": 1,
            "small_max_tokens": 400,
            "medium_window_tokens": 512,
            "medium_overlap_tokens": 128,
            "large_window_tokens": 1024,
            "large_overlap_tokens": 256,
        }
    }
    chunks = chunk_documents([doc], config)
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.doc_id
        assert chunk.filename
        assert chunk.chunk_id
        assert chunk.page_range
        assert chunk.section_title


def test_ingestion_reads_pdf(tmp_path: Path):
    pytest.importorskip("pymupdf4llm")

    pdf_path = tmp_path / "mini.pdf"
    with fitz.open() as doc:
        page1 = doc.new_page()
        page1.insert_text((72, 72), "# Test Report\n\nThis is page one.")
        page2 = doc.new_page()
        page2.insert_text((72, 72), "This is page two.")
        doc.save(pdf_path)

    docs = ingest_documents(tmp_path)
    assert len(docs) == 1
    assert docs[0].page_count == 2
    assert docs[0].filename == "mini.pdf"
