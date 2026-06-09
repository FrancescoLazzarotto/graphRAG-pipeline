from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evalkit.models import KGQualityResult

logger = logging.getLogger("graphrag")


def _entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load %s: %s", path, exc)
        return None


def _count_failed_chunks(artifacts_dir: Path) -> int:
    path = artifacts_dir / "failed_chunks.jsonl"
    if not path.exists():
        return 0
    count = 0
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    count += 1
    except OSError:
        pass
    return count


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    return sorted_vals[mid]


def compute_from_artifacts(
    artifacts_dir: Path,
    gold_entities: list[str] | None = None,
) -> KGQualityResult:
    """Compute KG quality metrics from KG pipeline stage artifacts.

    Uses stage4_registry.json (entity list), stage5_triples_linked.json (triple list),
    stage6_neo4j_summary.json (node/rel counts), stage1_chunks.json (chunk count),
    and failed_chunks.jsonl (extraction failures).

    All reads are best-effort: missing stages produce partial results.

    Args:
        artifacts_dir: Path to a KG pipeline run directory
            (e.g. ``kg_pipeline/artifacts/run_<timestamp>``).
        gold_entities: Optional list of expected entities for coverage check.

    Returns:
        KGQualityResult with structural and quality metrics.
    """
    if not artifacts_dir.is_dir():
        raise FileNotFoundError(f"KG artifacts directory not found: {artifacts_dir}")

    result = KGQualityResult()
    extra: dict[str, Any] = {}

    # ── Stage 6 summary (node/rel counts, authoritative) ──────────────────
    stage6 = _load_json(artifacts_dir / "stage6_neo4j_summary.json")
    if stage6:
        result.n_triples = int(stage6.get("relationships_written", 0))
        summary = stage6.get("summary", {})
        nodes_by_label = summary.get("nodes_by_label", [])
        result.n_entities = sum(entry.get("count", 0) for entry in nodes_by_label)
        extra["nodes_by_label"] = nodes_by_label

    # ── Stage 5 (triples with predicate distribution) ──────────────────────
    stage5 = _load_json(artifacts_dir / "stage5_triples_linked.json")
    if stage5 and isinstance(stage5, list):
        if result.n_triples == 0:
            result.n_triples = len(stage5)

        predicate_counts: Counter[str] = Counter()
        degree: dict[str, int] = defaultdict(int)

        for triple in stage5:
            if not isinstance(triple, dict):
                continue
            pred = str(triple.get("predicate", "") or "").strip()
            if pred:
                predicate_counts[pred] += 1
            subj = str(triple.get("subject", "") or "").strip().lower()
            obj = str(triple.get("object", "") or "").strip().lower()
            if subj:
                degree[subj] += 1
            if obj:
                degree[obj] += 1

        result.n_predicates = len(predicate_counts)
        result.predicate_entropy = _entropy(list(predicate_counts.values()))
        extra["top_predicates"] = predicate_counts.most_common(20)

        degrees = list(degree.values())
        if degrees:
            result.avg_degree = sum(degrees) / len(degrees)
            result.median_degree = _median([float(d) for d in degrees])
            isolated = sum(1 for d in degrees if d <= 1)
            result.isolated_ratio = isolated / len(degrees)

    # ── Stage 4 registry (entity count pre-resolution, collapse ratio) ─────
    stage4_registry = _load_json(artifacts_dir / "stage4_registry.json")
    if stage4_registry and isinstance(stage4_registry, dict):
        n_pre_resolution = len(stage4_registry)
        if result.n_entities > 0:
            result.resolution_collapse_ratio = max(
                0.0, 1.0 - result.n_entities / n_pre_resolution
            )
        extra["entities_pre_resolution"] = n_pre_resolution

    # ── Density ───────────────────────────────────────────────────────────
    if result.n_entities > 0:
        result.density = result.n_triples / result.n_entities

    # ── Chunk-level extraction quality ────────────────────────────────────
    stage1 = _load_json(artifacts_dir / "stage1_chunks.json")
    n_chunks = len(stage1) if isinstance(stage1, list) else 0
    if n_chunks > 0:
        extra["n_chunks"] = n_chunks

    failed = _count_failed_chunks(artifacts_dir)
    result.failed_chunks = failed
    if n_chunks > 0:
        result.failed_chunks_ratio = failed / n_chunks
    elif failed > 0:
        result.failed_chunks_ratio = 1.0

    # ── Documents ─────────────────────────────────────────────────────────
    stage0 = _load_json(artifacts_dir / "stage0_documents.json")
    if isinstance(stage0, list):
        result.n_documents = len(stage0)

    # ── Gold entity coverage ──────────────────────────────────────────────
    if gold_entities:
        if stage4_registry and isinstance(stage4_registry, dict):
            known = {k.lower() for k in stage4_registry}
            hits = sum(1 for e in gold_entities if e.lower() in known)
            result.entity_gold_coverage = hits / len(gold_entities)
        else:
            logger.warning("Cannot compute entity_gold_coverage without stage4_registry.json")

    result.extra = extra
    return result


def compute_from_neo4j(
    neo4j_url: str,
    neo4j_user: str,
    neo4j_password: str,
    database: str = "neo4j",
    gold_entities: list[str] | None = None,
) -> KGQualityResult:
    """Compute KG quality metrics by querying a live Neo4j instance.

    Requires ``neo4j`` Python driver.

    Args:
        neo4j_url: Bolt URL (e.g. ``bolt://localhost:7687``).
        neo4j_user: Username.
        neo4j_password: Password.
        database: Database name.
        gold_entities: Optional list of expected entities.

    Returns:
        KGQualityResult with live metrics.
    """
    try:
        from neo4j import GraphDatabase  # type: ignore
    except ImportError as exc:
        raise ImportError("Install neo4j: pip install neo4j") from exc

    driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))
    result = KGQualityResult()
    extra: dict[str, Any] = {}

    try:
        with driver.session(database=database) as session:
            # Node count
            r = session.run("MATCH (n) RETURN count(n) AS n").single()
            result.n_entities = r["n"] if r else 0

            # Relationship count
            r = session.run("MATCH ()-[r]->() RETURN count(r) AS n").single()
            result.n_triples = r["n"] if r else 0

            # Predicate distribution
            rows = session.run(
                "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY c DESC LIMIT 50"
            )
            pred_counts: list[tuple[str, int]] = [(row["t"], row["c"]) for row in rows]
            result.n_predicates = len(pred_counts)
            result.predicate_entropy = _entropy([c for _, c in pred_counts])
            extra["top_predicates"] = pred_counts[:20]

            # Degree
            rows = session.run(
                "MATCH (n) RETURN size([(n)-[]-() | 1]) AS deg"
            )
            degrees = [row["deg"] for row in rows]
            if degrees:
                result.avg_degree = sum(degrees) / len(degrees)
                result.median_degree = _median([float(d) for d in degrees])
                isolated = sum(1 for d in degrees if d <= 1)
                result.isolated_ratio = isolated / len(degrees)

            # Gold coverage
            if gold_entities:
                hits = 0
                for entity in gold_entities:
                    r = session.run(
                        "MATCH (n) WHERE toLower(n.name) = toLower($e) RETURN count(n) AS c",
                        e=entity,
                    ).single()
                    if r and r["c"] > 0:
                        hits += 1
                result.entity_gold_coverage = hits / len(gold_entities)

    finally:
        driver.close()

    if result.n_entities > 0:
        result.density = result.n_triples / result.n_entities

    result.extra = extra
    return result
