from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from dotenv import load_dotenv

from kg_pipeline.models.types import CanonicalEntityRecord, ChunkRecord, DocumentRecord, KGTriple, NEREntityCandidate
from kg_pipeline.stages import chunking, ingestion, linking, llm_extraction, neo4j_ingestion, ner, resolution


LOGGER = logging.getLogger("kg_pipeline")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _log_versions(config: dict[str, Any]) -> None:
    pkg_names = ["pymupdf4llm", "gliner", "openai", "sentence-transformers", "neo4j", "pydantic", "tqdm"]
    versions = {}
    for pkg in pkg_names:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not_installed"

    LOGGER.info("Dependency versions: %s", versions)
    LOGGER.info(
        "Configured vLLM model=%s base_url=%s",
        os.getenv("VLLM_MODEL_NAME", ""),
        os.getenv("VLLM_BASE_URL", ""),
    )


def _stage_output_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "documents": run_dir / "stage0_documents.json",
        "chunks": run_dir / "stage1_chunks.json",
        "ner": run_dir / "stage2_ner.json",
        "triples_raw": run_dir / "stage3_triples_raw.json",
        "acronyms": run_dir / "stage3_acronyms.json",
        "triples_resolved": run_dir / "stage4_triples_resolved.json",
        "registry": run_dir / "stage4_registry.json",
        "triples_linked": run_dir / "stage5_triples_linked.json",
        "failed_chunks": run_dir / "failed_chunks.jsonl",
        "new_labels_log": run_dir / "new_labels.log",
        "neo4j_summary": run_dir / "stage6_neo4j_summary.json",
    }


def _load_or_run_documents(paths: dict[str, Path], config: dict[str, Any], single_doc: str | None) -> list[DocumentRecord]:
    if paths["documents"].exists():
        return ingestion.load_documents(paths["documents"])
    docs = ingestion.ingest_documents(
        input_dir=Path(config["paths"]["input_dir"]),
        single_doc=single_doc,
    )
    ingestion.save_documents(paths["documents"], docs)
    return docs


def _load_or_run_chunks(paths: dict[str, Path], config: dict[str, Any], docs: list[DocumentRecord]) -> list[ChunkRecord]:
    if paths["chunks"].exists():
        return chunking.load_chunks(paths["chunks"])
    chunks = chunking.chunk_documents(docs, config)
    chunking.save_chunks(paths["chunks"], chunks)
    return chunks


def _load_or_run_ner(paths: dict[str, Path], config: dict[str, Any], chunks: list[ChunkRecord]) -> dict[str, list[NEREntityCandidate]]:
    if paths["ner"].exists():
        return ner.load_ner(paths["ner"])
    ner_map = ner.run_ner(
        chunks=chunks,
        model_name=config["gliner"]["model_name"],
        labels=config["ontology"]["labels"],
        threshold=float(config["gliner"]["threshold"]),
    )
    ner.save_ner(paths["ner"], ner_map)
    return ner_map


def _load_or_run_raw_triples(
    paths: dict[str, Path],
    config: dict[str, Any],
    chunks: list[ChunkRecord],
    ner_map: dict[str, list[NEREntityCandidate]],
    seed: int,
) -> tuple[list[KGTriple], dict[str, str]]:
    if paths["triples_raw"].exists() and paths["acronyms"].exists():
        return llm_extraction.load_triples(paths["triples_raw"]), llm_extraction.load_acronyms(paths["acronyms"])

    base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    model_name = os.getenv("VLLM_MODEL_NAME", "")
    api_key = os.getenv("VLLM_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY"))

    triples, acronym_map = llm_extraction.extract_triples(
        chunks=chunks,
        ner_map=ner_map,
        allowed_labels=config["ontology"]["labels"],
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        max_retries_per_chunk=int(config["llm"]["max_retries_per_chunk"]),
        temperature=float(config["llm"]["temperature"]),
        seed=seed,
        use_structured_output=bool(config["llm"]["use_structured_output"]),
        failed_chunks_path=paths["failed_chunks"],
        new_label_log_path=paths["new_labels_log"],
    )

    llm_extraction.save_triples(paths["triples_raw"], triples)
    llm_extraction.save_acronyms(paths["acronyms"], acronym_map)
    return triples, acronym_map


def _load_or_run_resolution(
    paths: dict[str, Path],
    config: dict[str, Any],
    triples: list[KGTriple],
    acronym_map: dict[str, str],
) -> tuple[list[KGTriple], dict[str, CanonicalEntityRecord]]:
    if paths["triples_resolved"].exists() and paths["registry"].exists():
        return resolution.load_triples(paths["triples_resolved"]), resolution.load_registry(paths["registry"])

    base_url = os.getenv("VLLM_BASE_URL", "")
    model_name = os.getenv("VLLM_MODEL_NAME", "")
    api_key = os.getenv("VLLM_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY"))

    resolved_triples, registry = resolution.resolve_entities(
        triples=triples,
        acronym_map=acronym_map,
        embedding_model=config["resolution"]["embedding_model"],
        similarity_threshold=float(config["resolution"]["similarity_threshold"]),
        context_jaccard_floor=float(config["resolution"]["context_jaccard_floor"]),
        base_url=base_url or None,
        api_key=api_key,
        model_name=model_name or None,
    )

    resolution.save_triples(paths["triples_resolved"], resolved_triples)
    resolution.save_registry(paths["registry"], registry)
    return resolved_triples, registry


def _load_or_run_linking(
    paths: dict[str, Path],
    resolved_triples: list[KGTriple],
    registry: dict[str, CanonicalEntityRecord],
    documents: list[DocumentRecord],
    config: dict[str, Any],
) -> list[KGTriple]:
    if paths["triples_linked"].exists():
        return linking.load_triples(paths["triples_linked"])

    include_mentioned_in = bool(config.get("linking", {}).get("include_mentioned_in", True))
    linked = linking.add_cross_document_links(
        triples=resolved_triples,
        registry=registry,
        documents=documents,
        include_mentioned_in=include_mentioned_in,
    )
    linking.save_triples(paths["triples_linked"], linked)
    return linked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="kg_pipeline/config.yaml")
    parser.add_argument("--env-file", default="kg_pipeline/.env")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--single-doc", default=None)
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "ingestion", "chunking", "ner", "llm", "resolution", "linking", "neo4j"],
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    config = _load_yaml(Path(args.config))
    load_dotenv(args.env_file, override=True)

    seed = int(config.get("seed", 42))
    _set_seed(seed)
    _log_versions(config)

    if args.run_dir.strip():
        run_dir = Path(args.run_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path(config["paths"]["output_dir"]) / f"run_{timestamp}"

    run_dir.mkdir(parents=True, exist_ok=True)
    paths = _stage_output_paths(run_dir)

    documents = _load_or_run_documents(paths, config, args.single_doc)
    if args.stage == "ingestion":
        LOGGER.info("Completed stage=ingestion docs=%d", len(documents))
        return

    chunks = _load_or_run_chunks(paths, config, documents)
    if args.stage == "chunking":
        LOGGER.info("Completed stage=chunking chunks=%d", len(chunks))
        return

    ner_map = _load_or_run_ner(paths, config, chunks)
    if args.stage == "ner":
        entity_count = sum(len(v) for v in ner_map.values())
        LOGGER.info("Completed stage=ner entities=%d", entity_count)
        return

    raw_triples, acronym_map = _load_or_run_raw_triples(paths, config, chunks, ner_map, seed=seed)
    if args.stage == "llm":
        LOGGER.info("Completed stage=llm triples=%d", len(raw_triples))
        return

    resolved_triples, registry = _load_or_run_resolution(paths, config, raw_triples, acronym_map)
    if args.stage == "resolution":
        LOGGER.info("Completed stage=resolution triples=%d canonical_entities=%d", len(resolved_triples), len(registry))
        return

    linked_triples = _load_or_run_linking(paths, resolved_triples, registry, documents, config)
    if args.stage == "linking":
        LOGGER.info("Completed stage=linking triples=%d", len(linked_triples))
        return

    if args.dry_run:
        sample = [triple.as_dict() for triple in linked_triples[:5]]
        LOGGER.info("Dry-run enabled, skipping Neo4j ingestion.")
        LOGGER.info("Total triples after linking: %d", len(linked_triples))
        LOGGER.info("Sample triples: %s", json.dumps(sample, ensure_ascii=False, indent=2))
        return

    uri, user, password, env_db = neo4j_ingestion._resolve_neo4j_env()
    db = config.get("neo4j", {}).get("database") or env_db

    written = neo4j_ingestion.ingest_triples(
        triples=linked_triples,
        uri=uri,
        user=user,
        password=password,
        database=db,
    )
    summary = neo4j_ingestion.summary_counts(uri=uri, user=user, password=password, database=db)

    _save_json(paths["neo4j_summary"], {"relationships_written": written, "summary": summary})
    LOGGER.info("Neo4j ingestion complete, relationships_written=%d", written)


if __name__ == "__main__":
    main()
