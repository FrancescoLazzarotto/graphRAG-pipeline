#!/usr/bin/env python3
"""KG repair pass 5 — deterministic structural fixes (no LLM).

Steps:
  1. SAME_AS cluster merge: nodes linked by SAME_AS arcs are fused into a
     single canonical node via APOC mergeNodes. Canonical name = shortest in
     the cluster, preferring forms without a leading "the "/"THE ".
  2. PUBLISHED direction fix: flip (Document)-[PUBLISHED]->(Organization)
     to (Organization)-[PUBLISHED]->(Document).
  3. Micro-type consolidation (6 remaps):
       USED_BY(a→b)      → USES(b→a)            passive→active, reversed
       LEADS_TO(a→b)     → AFFECTS(a→b)          synonym
       PART_OF(a→b)      → HAS_COMPONENT(b→a)    semantic inverse
       DEPENDS_ON(a→b)   → REQUIRES(a→b)         synonym
       REGULATED_BY(a→b) → REGULATES(b→a)        passive→active, reversed
       OCCURS_IN(a→b)    → LOCATED_IN(a→b)       synonym
"""

from __future__ import annotations

import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / "kg_pipeline" / ".env")

NEO4J_URI = os.getenv("NEO4J_URI") or os.getenv("NEO4J_URL", "")
NEO4J_USER = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "").strip() or None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("kg_repair5")

_SESSION_KWARGS: dict = {"database": NEO4J_DATABASE} if NEO4J_DATABASE else {}

# ── 1. SAME_AS cluster merge ──────────────────────────────────────────────────

def _find(parent: dict[str, str], x: str) -> str:
    if parent.setdefault(x, x) != x:
        parent[x] = _find(parent, parent[x])
    return parent[x]


def _union(parent: dict[str, str], x: str, y: str) -> None:
    parent[_find(parent, x)] = _find(parent, y)


def _canonical_name(names: list[str]) -> str:
    return min(names, key=lambda n: (n.lower().startswith("the "), len(n), n))


def merge_same_as_clusters(session) -> int:
    rows = session.run(
        "MATCH (a)-[:SAME_AS]-(b) "
        "RETURN elementId(a) AS ea, elementId(b) AS eb, a.name AS na, b.name AS nb"
    ).data()

    if not rows:
        logger.info("No SAME_AS relationships — skip.")
        return 0

    parent: dict[str, str] = {}
    names: dict[str, str] = {}
    for row in rows:
        ea, eb = row["ea"], row["eb"]
        names.setdefault(ea, row["na"] or "")
        names.setdefault(eb, row["nb"] or "")
        _union(parent, ea, eb)

    components: dict[str, list[str]] = defaultdict(list)
    for eid in names:
        components[_find(parent, eid)].append(eid)

    merged = 0
    for eids in components.values():
        if len(eids) < 2:
            continue
        canonical = _canonical_name([names[e] for e in eids])
        logger.info("Merging %s → '%s'", [names[e] for e in eids], canonical)
        session.run(
            """
            MATCH (n) WHERE elementId(n) IN $eids
            WITH collect(n) AS nodes
            CALL apoc.refactor.mergeNodes(nodes, {properties: 'combine', mergeRels: true})
            YIELD node
            SET node.name = $canonical
            RETURN node
            """,
            eids=eids,
            canonical=canonical,
        )
        merged += 1

    # Remove self-loops left by the merge
    removed = session.run(
        "MATCH (n)-[r:SAME_AS]->(n) DELETE r RETURN count(r) AS n"
    ).single()["n"]
    if removed:
        logger.info("Removed %d SAME_AS self-loops.", removed)

    logger.info("SAME_AS: merged %d cluster(s).", merged)
    return merged


# ── 2. PUBLISHED direction fix ────────────────────────────────────────────────

def fix_published_direction(session) -> int:
    result = session.run(
        """
        MATCH (d:Document)-[r:PUBLISHED]->(o:Organization)
        WITH d, o, r, properties(r) AS props
        MERGE (o)-[nr:PUBLISHED]->(d)
          ON CREATE SET nr = props
        DELETE r
        RETURN count(*) AS fixed
        """
    )
    fixed = result.single()["fixed"]
    logger.info("PUBLISHED direction: flipped %d arcs.", fixed)
    return fixed


# ── 3. Micro-type consolidation ───────────────────────────────────────────────

# (source_type, target_type, reverse_direction)
_MICRO_REMAPS = [
    ("USED_BY",      "USES",          True),
    ("LEADS_TO",     "AFFECTS",       False),
    ("PART_OF",      "HAS_COMPONENT", True),
    ("DEPENDS_ON",   "REQUIRES",      False),
    ("REGULATED_BY", "REGULATES",     True),
    ("OCCURS_IN",    "LOCATED_IN",    False),
]


def consolidate_micro_types(session) -> int:
    total = 0
    for src, dst, reverse in _MICRO_REMAPS:
        if reverse:
            cypher = f"""
                MATCH (a)-[r:{src}]->(b)
                WITH a, b, r, properties(r) AS props
                MERGE (b)-[nr:{dst}]->(a)
                  ON CREATE SET nr = props
                DELETE r
                RETURN count(*) AS n
            """
        else:
            cypher = f"""
                MATCH (a)-[r:{src}]->(b)
                WITH a, b, r, properties(r) AS props
                MERGE (a)-[nr:{dst}]->(b)
                  ON CREATE SET nr = props
                DELETE r
                RETURN count(*) AS n
            """
        n = session.run(cypher).single()["n"]
        direction = "reversed" if reverse else "same dir"
        logger.info("%s → %s (%s): %d arcs.", src, dst, direction, n)
        total += n
    return total


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    if not NEO4J_URI or not NEO4J_PASSWORD:
        logger.error("NEO4J credentials missing. Check kg_pipeline/.env")
        sys.exit(1)

    logger.info("URI=%s DB=%s", NEO4J_URI, NEO4J_DATABASE or "<default>")

    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        with driver.session(**_SESSION_KWARGS) as session:
            logger.info("=== Step 1: SAME_AS cluster merge ===")
            merge_same_as_clusters(session)

            logger.info("=== Step 2: PUBLISHED direction fix ===")
            fix_published_direction(session)

            logger.info("=== Step 3: Micro-type consolidation ===")
            n = consolidate_micro_types(session)
            logger.info("Total arcs remapped: %d", n)

    logger.info("Pass 5 complete.")


if __name__ == "__main__":
    main()
