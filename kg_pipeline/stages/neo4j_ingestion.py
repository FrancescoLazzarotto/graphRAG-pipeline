from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from neo4j import GraphDatabase
from tqdm import tqdm

from kg_pipeline.models.types import KGTriple


_ID_RE = re.compile(r"[^A-Za-z0-9_]+")


def _safe_identifier(value: str, fallback: str) -> str:
    cleaned = _ID_RE.sub("_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = fallback
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned


def _resolve_neo4j_env() -> tuple[str, str, str, str | None]:
    uri = os.getenv("NEO4J_URI") or os.getenv("NEO4J_URL")
    user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE") or None

    missing = []
    if not uri:
        missing.append("NEO4J_URI or NEO4J_URL")
    if not user:
        missing.append("NEO4J_USER or NEO4J_USERNAME")
    if not password:
        missing.append("NEO4J_PASSWORD")

    if missing:
        raise ValueError("Missing Neo4j env vars: " + ", ".join(missing))

    return uri, user, password, database


def _merge_triple(tx, triple: KGTriple) -> None:
    s_labels = triple.subject_labels or ["Concept"]
    o_labels = triple.object_labels or ["Concept"]

    s_primary = _safe_identifier(s_labels[0], "Concept")
    o_primary = _safe_identifier(o_labels[0], "Concept")
    rel_type = _safe_identifier(triple.predicate, "RELATED_TO")

    s_extra = [_safe_identifier(label, "Concept") for label in s_labels[1:]]
    o_extra = [_safe_identifier(label, "Concept") for label in o_labels[1:]]

    set_subject_extra = "\n".join([f"SET s:{label}" for label in s_extra])
    set_object_extra = "\n".join([f"SET o:{label}" for label in o_extra])

    query = f"""
MERGE (s:{s_primary} {{name: $s_name}})
SET s += $s_props
{set_subject_extra}
MERGE (o:{o_primary} {{name: $o_name}})
SET o += $o_props
{set_object_extra}
MERGE (s)-[r:{rel_type} {{subject: $s_name, object: $o_name}}]->(o)
SET r += $r_props
"""

    tx.run(
        query,
        s_name=triple.subject_properties.get("name", triple.subject),
        o_name=triple.object_properties.get("name", triple.object),
        s_props=triple.subject_properties,
        o_props=triple.object_properties,
        r_props=triple.relationship_properties,
    ).consume()


def ingest_triples(
    triples: list[KGTriple],
    uri: str,
    user: str,
    password: str,
    database: str | None = None,
) -> int:
    count = 0
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session(database=database) as session:
            for triple in tqdm(triples, desc="Stage 6 Neo4j Ingestion", unit="triple"):
                session.execute_write(_merge_triple, triple)
                count += 1
    return count


def summary_counts(
    uri: str,
    user: str,
    password: str,
    database: str | None = None,
) -> dict:
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session(database=database) as session:
            node_records = session.run(
                "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS count ORDER BY count DESC"
            ).data()
            rel_records = session.run(
                "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY count DESC"
            ).data()
    return {"nodes_by_label": node_records, "relationships_by_type": rel_records}


def load_triples(path: Path) -> list[KGTriple]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [KGTriple.model_validate(item) for item in payload]


def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples-json", required=True)
    parser.add_argument("--database", default="")
    args = parser.parse_args()

    uri, user, password, env_db = _resolve_neo4j_env()
    database = args.database.strip() or env_db

    triples = load_triples(Path(args.triples_json))
    written = ingest_triples(triples, uri=uri, user=user, password=password, database=database)
    summary = summary_counts(uri=uri, user=user, password=password, database=database)

    print(json.dumps({"relationships_written": written, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
