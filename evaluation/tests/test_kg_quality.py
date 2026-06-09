from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "evaluation"
for p in (str(PROJECT_ROOT), str(EVAL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from evalkit.kg.kg_quality import compute_from_artifacts


def _make_artifacts(
    tmp_path: Path,
    n_triples: int = 10,
    n_entities: int = 5,
    n_chunks: int = 20,
    n_failed: int = 2,
) -> Path:
    # stage6
    stage6 = {
        "relationships_written": n_triples,
        "summary": {
            "nodes_by_label": [
                {"label": "Concept", "count": n_entities // 2},
                {"label": "Organization", "count": n_entities - n_entities // 2},
            ]
        },
    }
    (tmp_path / "stage6_neo4j_summary.json").write_text(json.dumps(stage6))

    # stage5
    triples = [
        {
            "subject": f"entity_{i % n_entities}",
            "predicate": "IS_RELATED_TO" if i % 2 == 0 else "CAUSES",
            "object": f"entity_{(i + 1) % n_entities}",
        }
        for i in range(n_triples)
    ]
    (tmp_path / "stage5_triples_linked.json").write_text(json.dumps(triples))

    # stage4 registry
    registry = {f"entity_{i}_raw": {"canonical_name": f"entity_{i}"} for i in range(n_entities + 3)}
    (tmp_path / "stage4_registry.json").write_text(json.dumps(registry))

    # stage1 chunks
    chunks = [{"chunk_id": f"chunk_{i}"} for i in range(n_chunks)]
    (tmp_path / "stage1_chunks.json").write_text(json.dumps(chunks))

    # stage0 documents
    docs = [{"doc_id": f"doc_{i}"} for i in range(3)]
    (tmp_path / "stage0_documents.json").write_text(json.dumps(docs))

    # failed_chunks
    with (tmp_path / "failed_chunks.jsonl").open("w") as fh:
        for i in range(n_failed):
            fh.write(json.dumps({"chunk_id": f"chunk_{i}", "error": "LLM error"}) + "\n")

    return tmp_path


def test_basic_metrics(tmp_path: Path) -> None:
    artifacts = _make_artifacts(tmp_path, n_triples=10, n_entities=5, n_chunks=20, n_failed=2)
    result = compute_from_artifacts(artifacts)

    assert result.n_triples == 10
    assert result.n_entities == 5
    assert result.density == pytest.approx(10 / 5)
    assert result.n_predicates == 2  # IS_RELATED_TO, CAUSES
    assert result.predicate_entropy > 0
    assert result.failed_chunks == 2
    assert result.failed_chunks_ratio == pytest.approx(2 / 20)
    assert result.n_documents == 3


def test_resolution_collapse(tmp_path: Path) -> None:
    artifacts = _make_artifacts(tmp_path, n_entities=5)
    result = compute_from_artifacts(artifacts)
    # pre-resolution: 8 entries (5 + 3), post: 5 nodes → collapse > 0
    assert result.resolution_collapse_ratio > 0.0


def test_gold_entity_coverage(tmp_path: Path) -> None:
    artifacts = _make_artifacts(tmp_path, n_entities=5)
    gold = ["entity_0", "entity_1", "entity_99"]  # entity_99 not in registry
    result = compute_from_artifacts(artifacts, gold_entities=gold)
    # entity_0_raw and entity_1_raw are keys in registry (lowercased match)
    assert result.entity_gold_coverage is not None
    assert 0.0 <= result.entity_gold_coverage <= 1.0


def test_missing_artifacts_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        compute_from_artifacts(tmp_path / "nonexistent")


def test_partial_artifacts(tmp_path: Path) -> None:
    # Only stage5 present — should still produce partial results
    triples = [{"subject": "a", "predicate": "IS", "object": "b"}]
    (tmp_path / "stage5_triples_linked.json").write_text(json.dumps(triples))
    result = compute_from_artifacts(tmp_path)
    assert result.n_triples == 1
    assert result.n_predicates == 1
