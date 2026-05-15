#!/usr/bin/env python3
"""Lightweight smoke tests for the KG pipeline (no external services started).

This script performs safe checks:
- importability of key modules
- prompt build sanity
- KGTriple schema contains "properties"
- KGTriple model validation for a minimal triple
- presence of cross-label merge parameter and log default
- ner threshold change present in source
- neo4j quality-check function exists

Run without network or model downloads.
"""
from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
import sys
# ensure project root is importable for local package imports
sys.path.insert(0, str(ROOT))
SYS_OK = 0
results = {"ok": [], "failed": []}

def ok(msg: str):
    results["ok"].append(msg)

def fail(msg: str):
    results["failed"].append(msg)

def main():
    try:
        # import modules
        mods = [
            "kg_pipeline.prompts.extraction_prompt",
            "kg_pipeline.models.types",
            "kg_pipeline.stages.resolution",
            "kg_pipeline.stages.ner",
            "kg_pipeline.stages.neo4j_ingestion",
        ]
        loaded = {}
        for m in mods:
            try:
                loaded[m] = importlib.import_module(m)
                ok(f"import {m}")
            except Exception as e:
                # treat missing optional runtime deps (openai, gliner, sentence_transformers)
                if isinstance(e, ModuleNotFoundError):
                    missing = getattr(e, "name", None)
                    ok(f"import {m} skipped, missing dependency: {missing}")
                else:
                    fail(f"import {m} failed: {e}")

        # build prompt
        if "kg_pipeline.prompts.extraction_prompt" in loaded:
            mod = loaded["kg_pipeline.prompts.extraction_prompt"]
            from kg_pipeline.models.types import ChunkRecord

            chunk = ChunkRecord.model_validate({
                "doc_id": "d1",
                "filename": "example.pdf",
                "chunk_id": "c1",
                "page_range": "1-1",
                "section_title": "Introduction",
                "chunk_index": 1,
                "text": "This policy governs wheat and ensures high levels of protection.",
            })
            prompt = mod.build_extraction_prompt(chunk, [], ["Concept", "Policy"])
            if "ALLOWED PREDICATES" in prompt or "You MUST use ONLY the following predicate types" in prompt:
                ok("extraction prompt contains allowed-predicates instructions")
            else:
                fail("extraction prompt does not contain allowed-predicates block")

        # schema contains properties
        if "kg_pipeline.models.types" in loaded:
            types_mod = loaded["kg_pipeline.models.types"]
            schema = types_mod.kg_triple_array_schema()
            props = schema.get("items", {}).get("properties", {})
            if "properties" in props:
                ok("kg_triple_array_schema contains top-level 'properties'")
            else:
                fail("kg_triple_array_schema missing 'properties' key")

            # validate a minimal KGTriple
            triple_payload = {
                "subject": "wheat",
                "predicate": "PUBLISHED",
                "object": "report",
                "subject_labels": ["Commodity"],
                "object_labels": ["Document"],
                "subject_properties": {"name": "wheat"},
                "object_properties": {"name": "report"},
                "relationship_properties": {"source_doc": "example.pdf", "extraction_method": "llm"},
            }
            try:
                kt = types_mod.KGTriple.model_validate(triple_payload)
                ok("KGTriple model_validate succeeded for minimal triple")
            except Exception as e:
                fail(f"KGTriple validation failed: {e}")

        # check resolution signature and crosslabel log default present in CLI wiring
        if "kg_pipeline.stages.resolution" in loaded:
            res_mod = loaded["kg_pipeline.stages.resolution"]
            sig = inspect.signature(res_mod.resolve_entities)
            if "crosslabel_log_path" in sig.parameters:
                ok("resolve_entities accepts crosslabel_log_path")
            else:
                fail("resolve_entities missing crosslabel_log_path parameter")

            # CLI file should accept --crosslabel-log (quick source check)
            cli_src = Path(res_mod.__file__).read_text(encoding="utf-8")
            if "--crosslabel-log" in cli_src:
                ok("resolution CLI exposes --crosslabel-log")
            else:
                fail("resolution CLI missing --crosslabel-log flag")

        # check ner threshold default
        if "kg_pipeline.stages.ner" in loaded:
            ner_src = Path(loaded["kg_pipeline.stages.ner"].__file__).read_text(encoding="utf-8")
            if "default=0.55" in ner_src:
                ok("ner default threshold set to 0.55")
            else:
                fail("ner default threshold not set to 0.55 in source")

        # neo4j quality check function exists and contains expected query tokens
        if "kg_pipeline.stages.neo4j_ingestion" in loaded:
            neo_mod = loaded["kg_pipeline.stages.neo4j_ingestion"]
            if hasattr(neo_mod, "run_quality_checks"):
                ok("neo4j_ingestion.run_quality_checks exists")
                neo_src = Path(neo_mod.__file__).read_text(encoding="utf-8")
                if "HAS_MAXIMUM_LEVEL" in neo_src and "outOfVocab" in neo_src:
                    ok("neo4j quality queries include HAS_MAXIMUM_LEVEL and outOfVocab check")
                else:
                    fail("neo4j quality queries appear incomplete")
            else:
                fail("neo4j_ingestion missing run_quality_checks")

    except Exception as e:
        fail(f"unexpected error in smoke test harness: {e}")

    # summary
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if results["failed"]:
        sys.exit(2)

if __name__ == "__main__":
    main()
