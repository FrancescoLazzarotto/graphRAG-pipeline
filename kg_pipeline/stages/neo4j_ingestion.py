from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from neo4j import GraphDatabase
from tqdm import tqdm
import logging
from neo4j.exceptions import CypherTypeError

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


def _is_primitive(value: object) -> bool:
    return isinstance(value, (str, bool, int, float))


def _sanitize_value(value: object) -> object:
    if _is_primitive(value):
        return value
    if value is None:
        return ""
    if isinstance(value, (list, dict, set, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return str(value)
    return str(value)


def _sanitize_props(props: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for k, v in (props or {}).items():
        if v is None:
            # skip null values
            continue
        if _is_primitive(v):
            out[str(k)] = v
            continue

        if isinstance(v, list):
            # convert list elements to primitives or strings (no nested maps)
            new_list: list[object] = []
            categories: set[str] = set()
            for item in v:
                if item is None:
                    continue
                if _is_primitive(item):
                    # treat bool separately from numeric to avoid subclassing issues
                    if isinstance(item, bool):
                        categories.add("bool")
                    elif isinstance(item, (int, float)) and not isinstance(item, bool):
                        categories.add("num")
                    else:
                        categories.add("str")
                    new_list.append(item)
                else:
                    try:
                        s = json.dumps(item, ensure_ascii=False, default=str)
                    except Exception:
                        s = str(item)
                    new_list.append(s)
                    categories.add("str")

            # Neo4j requires property arrays to be of a single allowed type.
            # If the list mixes numbers and strings (or other categories), coerce all
            # elements to strings to ensure a homogeneous, supported type.
            if len(categories - {""}) > 1:
                coerced = [str(x) for x in new_list]
                out[str(k)] = coerced
            else:
                out[str(k)] = new_list
            continue

        if isinstance(v, dict):
            try:
                out[str(k)] = json.dumps(v, ensure_ascii=False, default=str)
            except Exception:
                out[str(k)] = str(v)
            continue

        if isinstance(v, (set, tuple)):
            try:
                seq = list(v)
                new_seq: list[object] = []
                categories: set[str] = set()
                for i in seq:
                    if _is_primitive(i):
                        if isinstance(i, bool):
                            categories.add("bool")
                        elif isinstance(i, (int, float)) and not isinstance(i, bool):
                            categories.add("num")
                        else:
                            categories.add("str")
                        new_seq.append(i)
                    else:
                        try:
                            s = json.dumps(i, ensure_ascii=False, default=str)
                        except Exception:
                            s = str(i)
                        new_seq.append(s)
                        categories.add("str")

                if len(categories) > 1:
                    out[str(k)] = [str(x) for x in new_seq]
                else:
                    out[str(k)] = new_seq
            except Exception:
                out[str(k)] = str(v)
            continue

        # fallback to string representation for any other types
        out[str(k)] = str(v)

    return out


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

    s_name = _sanitize_value(triple.subject_properties.get("name", triple.subject))
    o_name = _sanitize_value(triple.object_properties.get("name", triple.object))
    s_props = _sanitize_props(triple.subject_properties)
    o_props = _sanitize_props(triple.object_properties)
    r_props = _sanitize_props(triple.relationship_properties)

    logger = logging.getLogger(__name__)

    try:
        tx.run(
            query,
            s_name=s_name,
            o_name=o_name,
            s_props=s_props,
            o_props=o_props,
            r_props=r_props,
        ).consume()
    except CypherTypeError as e:
        # log details for debugging and skip this triple to allow ingestion to continue
        debug = {
            "subject": triple.subject,
            "predicate": triple.predicate,
            "object": triple.object,
            "subject_labels": triple.subject_labels,
            "object_labels": triple.object_labels,
            "s_name": s_name,
            "o_name": o_name,
            "s_props_sanitized": s_props,
            "o_props_sanitized": o_props,
            "r_props_sanitized": r_props,
        }
        logger.exception("CypherTypeError writing triple: %s", e)
        try:
            logs_dir = Path(__file__).resolve().parents[1] / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            with (logs_dir / "problematic_triples.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(debug, ensure_ascii=False, default=str) + "\n")
        except Exception:
            logger.exception("Failed to write problematic triple to log file")
        return
    except Exception as e:
        logger.exception("Unexpected error writing triple: %s", e)
        # attempt to persist debug info as well
        try:
            logs_dir = Path(__file__).resolve().parents[1] / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            with (logs_dir / "problematic_triples.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "error": str(e),
                    "subject": triple.subject,
                    "predicate": triple.predicate,
                    "object": triple.object,
                    "s_props_sanitized": s_props,
                    "o_props_sanitized": o_props,
                    "r_props_sanitized": r_props,
                }, ensure_ascii=False, default=str) + "\n")
        except Exception:
            logger.exception("Failed to write unexpected error info to log file")
        return


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
