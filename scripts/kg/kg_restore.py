"""Restore a KG JSON backup (produced by kg_backup.py) into a Neo4j instance.

Reads ``nodes.json``, ``edges.json`` and ``schema.json`` from a backup folder
and rebuilds the graph, then verifies node/relationship counts per label and
type against the backup content.

Intended for the local staging instance (see docs/kg_fix_plan_2026-07.md,
fase 0). Refuses to write into a non-empty database unless ``--wipe`` is given.

Usage:
    python scripts/kg/kg_restore.py \
        --backup-dir artifacts/kg_backups/20260710_114216 \
        --uri bolt://localhost:7688 --user neo4j --password staging-password
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from neo4j import Driver, GraphDatabase

logger = logging.getLogger("kg_pipeline.kg_restore")

BATCH_SIZE = 1000
RESTORE_LABEL = "_Restore"
RESTORE_ID = "_rid"


def _batches(rows: list, size: int = BATCH_SIZE):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def restore_schema(driver: Driver, database: str | None, schema: dict) -> None:
    """Recreate uniqueness constraints and the fulltext index.

    LOOKUP indexes exist by default; RANGE indexes owned by uniqueness
    constraints are created implicitly with the constraint. Property-existence
    constraints are Enterprise-only and are skipped with a warning on
    Community.
    """
    constraint_names = {c.get("name") for c in schema.get("constraints", [])}
    with driver.session(database=database) as session:
        for con in schema.get("constraints", []):
            label = (con.get("labelsOrTypes") or [None])[0]
            prop = (con.get("properties") or [None])[0]
            if not label or not prop:
                continue
            if con["type"] == "NODE_PROPERTY_UNIQUENESS":
                session.run(
                    f"CREATE CONSTRAINT {con['name']} IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
                )
            elif con["type"] == "NODE_PROPERTY_EXISTENCE":
                try:
                    session.run(
                        f"CREATE CONSTRAINT {con['name']} IF NOT EXISTS "
                        f"FOR (n:{label}) REQUIRE n.{prop} IS NOT NULL"
                    )
                except Exception as exc:  # Enterprise-only on Community
                    logger.warning("skipped existence constraint %s: %s",
                                   con["name"], exc)
            else:
                logger.warning("unhandled constraint type %s (%s)",
                               con["type"], con["name"])
        for idx in schema.get("indexes", []):
            if idx.get("type") == "FULLTEXT":
                labels = "|".join(idx["labelsOrTypes"])
                props = ", ".join(f"n.{p}" for p in idx["properties"])
                session.run(
                    f"CREATE FULLTEXT INDEX {idx['name']} IF NOT EXISTS "
                    f"FOR (n:{labels}) ON EACH [{props}]"
                )
            elif idx.get("type") == "RANGE" and idx.get("name") not in constraint_names:
                labels = idx["labelsOrTypes"]
                props = ", ".join(f"n.{p}" for p in idx["properties"])
                session.run(
                    f"CREATE INDEX {idx['name']} IF NOT EXISTS "
                    f"FOR (n:{labels[0]}) ON ({props})"
                )


def restore_nodes(driver: Driver, database: str | None, nodes: list[dict]) -> None:
    """Create all nodes, tagging each with a temporary restore id."""
    by_labels: dict[tuple[str, ...], list[dict]] = {}
    for node in nodes:
        by_labels.setdefault(tuple(sorted(node["labels"])), []).append(node)

    with driver.session(database=database) as session:
        session.run(
            f"CREATE INDEX restore_rid IF NOT EXISTS "
            f"FOR (n:{RESTORE_LABEL}) ON (n.{RESTORE_ID})"
        )
        for labels, group in by_labels.items():
            label_expr = ":".join((RESTORE_LABEL,) + labels)
            for batch in _batches(group):
                session.run(
                    f"UNWIND $rows AS row "
                    f"CREATE (n:{label_expr}) "
                    f"SET n = row.props, n.{RESTORE_ID} = row.id",
                    rows=[{"id": n["id"], "props": n["props"]} for n in batch],
                )


def restore_edges(driver: Driver, database: str | None, edges: list[dict]) -> None:
    by_type: dict[str, list[dict]] = {}
    for edge in edges:
        by_type.setdefault(edge["type"], []).append(edge)

    with driver.session(database=database) as session:
        for rel_type, group in by_type.items():
            for batch in _batches(group):
                session.run(
                    f"UNWIND $rows AS row "
                    f"MATCH (a:{RESTORE_LABEL} {{{RESTORE_ID}: row.src}}) "
                    f"MATCH (b:{RESTORE_LABEL} {{{RESTORE_ID}: row.dst}}) "
                    f"CREATE (a)-[r:{rel_type}]->(b) SET r = row.props",
                    rows=[{"src": e["src"], "dst": e["dst"], "props": e["props"]}
                          for e in batch],
                )


def cleanup_restore_markers(driver: Driver, database: str | None) -> None:
    # ``CALL { } IN TRANSACTIONS`` is core Cypher since 5.0 and does the same
    # batching as ``apoc.periodic.iterate``. APOC is a plugin and is absent from
    # a plain Community tarball, which is what the local staging instance is.
    with driver.session(database=database) as session:
        session.run(
            f"MATCH (n:{RESTORE_LABEL}) "
            f"CALL {{ WITH n REMOVE n:{RESTORE_LABEL} REMOVE n.{RESTORE_ID} }} "
            f"IN TRANSACTIONS OF {BATCH_SIZE} ROWS"
        ).consume()
        session.run("DROP INDEX restore_rid IF EXISTS")


def verify(driver: Driver, database: str | None,
           nodes: list[dict], edges: list[dict]) -> bool:
    expected_labels = Counter(lbl for n in nodes for lbl in n["labels"])
    expected_types = Counter(e["type"] for e in edges)
    with driver.session(database=database) as session:
        got_labels = Counter({
            rec["l"]: rec["c"] for rec in session.run(
                "MATCH (n) UNWIND labels(n) AS l RETURN l, count(*) AS c")
        })
        got_types = Counter({
            rec["t"]: rec["c"] for rec in session.run(
                "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c")
        })
    ok = True
    for name, expected, got in (("labels", expected_labels, got_labels),
                                ("rel types", expected_types, got_types)):
        if expected != got:
            ok = False
            diff = {k: (expected.get(k, 0), got.get(k, 0))
                    for k in expected.keys() | got.keys()
                    if expected.get(k, 0) != got.get(k, 0)}
            logger.error("%s mismatch (expected, got): %s", name, diff)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-dir", required=True, type=Path)
    parser.add_argument("--uri", required=True)
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", required=True)
    parser.add_argument("--database", default=None)
    parser.add_argument("--wipe", action="store_true",
                        help="Delete all existing data in the target first.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    nodes = json.loads((args.backup_dir / "nodes.json").read_text())
    edges = json.loads((args.backup_dir / "edges.json").read_text())
    schema = json.loads((args.backup_dir / "schema.json").read_text())
    logger.info("backup: %d nodes, %d edges", len(nodes), len(edges))

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        with driver.session(database=args.database) as session:
            existing = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            if existing and not args.wipe:
                logger.error("target has %d nodes; pass --wipe to overwrite", existing)
                return 1
            if existing:
                logger.info("wiping %d existing nodes", existing)
                session.run(
                    "CALL apoc.periodic.iterate('MATCH (n) RETURN n', "
                    "'DETACH DELETE n', {batchSize: 1000})"
                )

        restore_schema(driver, args.database, schema)
        logger.info("schema restored")
        restore_nodes(driver, args.database, nodes)
        logger.info("nodes restored")
        restore_edges(driver, args.database, edges)
        logger.info("edges restored")
        cleanup_restore_markers(driver, args.database)

        if not verify(driver, args.database, nodes, edges):
            logger.error("VERIFICATION FAILED")
            return 2
        logger.info("VERIFICATION OK: restore matches backup")
        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    sys.exit(main())
