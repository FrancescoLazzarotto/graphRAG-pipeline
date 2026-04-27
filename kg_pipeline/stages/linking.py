from __future__ import annotations

import argparse
import json
from pathlib import Path

from kg_pipeline.models.types import CanonicalEntityRecord, DocumentRecord, KGTriple


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _document_props(doc: DocumentRecord) -> dict:
    return {
        "name": doc.filename,
        "filename": doc.filename,
        "page_count": doc.page_count,
        "title": doc.title,
        "publication_year": doc.publication_year,
    }


def add_cross_document_links(
    triples: list[KGTriple],
    registry: dict[str, CanonicalEntityRecord],
    documents: list[DocumentRecord],
) -> list[KGTriple]:
    doc_map = {doc.filename: doc for doc in documents}
    linked: list[KGTriple] = list(triples)
    seen_mention_edges: set[tuple[str, str, str, str]] = set()

    for triple in triples:
        source_doc = str(triple.relationship_properties.get("source_doc", "")).strip()
        chunk_id = str(triple.relationship_properties.get("chunk_id", "")).strip()
        page_range = str(triple.relationship_properties.get("page_range", "")).strip()

        if source_doc and source_doc in doc_map:
            doc_props = _document_props(doc_map[source_doc])

            for entity_name, entity_labels, entity_props in (
                (triple.subject, triple.subject_labels, triple.subject_properties),
                (triple.object, triple.object_labels, triple.object_properties),
            ):
                edge_key = (entity_name, source_doc, chunk_id, page_range)
                if edge_key in seen_mention_edges:
                    continue
                seen_mention_edges.add(edge_key)

                linked.append(
                    KGTriple.model_validate(
                        {
                            "subject": entity_name,
                            "predicate": "MENTIONED_IN",
                            "object": source_doc,
                            "subject_labels": entity_labels or ["Concept"],
                            "object_labels": ["Document"],
                            "subject_properties": dict(entity_props),
                            "object_properties": dict(doc_props),
                            "relationship_properties": {
                                "source_doc": source_doc,
                                "extraction_method": "system_linking",
                                "chunk_id": chunk_id,
                                "page_range": page_range,
                            },
                        }
                    )
                )

    for canonical_name, record in registry.items():
        docs_for_concept = set()
        for alias, src_docs in record.alias_sources.items():
            for d in src_docs:
                docs_for_concept.add(d)

        if len(docs_for_concept) < 2:
            continue

        for alias in record.aliases:
            if alias == canonical_name:
                continue

            alias_docs = record.alias_sources.get(alias, [])
            source_doc = alias_docs[0] if alias_docs else "registry"

            linked.append(
                KGTriple.model_validate(
                    {
                        "subject": alias,
                        "predicate": "SAME_AS",
                        "object": canonical_name,
                        "subject_labels": record.labels or ["Concept"],
                        "object_labels": record.labels or ["Concept"],
                        "subject_properties": {"name": alias},
                        "object_properties": dict(record.merged_properties),
                        "relationship_properties": {
                            "source_doc": source_doc,
                            "extraction_method": "system_linking",
                        },
                    }
                )
            )

    return linked


def save_triples(path: Path, triples: list[KGTriple]) -> None:
    _save_json(path, [triple.as_dict() for triple in triples])


def load_triples(path: Path) -> list[KGTriple]:
    payload = _load_json(path)
    return [KGTriple.model_validate(item) for item in payload]


def load_documents(path: Path) -> list[DocumentRecord]:
    payload = _load_json(path)
    return [DocumentRecord.model_validate(item) for item in payload]


def load_registry(path: Path) -> dict[str, CanonicalEntityRecord]:
    payload = _load_json(path)
    return {k: CanonicalEntityRecord.model_validate(v) for k, v in payload.items()}


def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples-json", required=True)
    parser.add_argument("--registry-json", required=True)
    parser.add_argument("--documents-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    triples = load_triples(Path(args.triples_json))
    registry = load_registry(Path(args.registry_json))
    documents = load_documents(Path(args.documents_json))

    output = add_cross_document_links(triples, registry, documents)
    save_triples(Path(args.output_json), output)


if __name__ == "__main__":
    _cli()
