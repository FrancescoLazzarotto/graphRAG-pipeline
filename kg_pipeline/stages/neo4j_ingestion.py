from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from neo4j import GraphDatabase
from tqdm import tqdm
import logging
from neo4j.exceptions import CypherTypeError
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kg_pipeline.models.types import KGTriple


_ID_RE = re.compile(r"[^A-Za-z0-9_]+")


def _setup_logging(log_level: str, log_file: str | None = None) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


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


def _triple_cypher_parts(triple: KGTriple) -> tuple[str, dict[str, object]]:
    """Build the per-signature Cypher (labels/rel type are identifiers, not
    parameters) and the parameter row for a triple.

    Triples sharing the same query text can be ingested together via UNWIND.
    """
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
UNWIND $rows AS row
MERGE (s:{s_primary} {{name: row.s_name}})
SET s += row.s_props
{set_subject_extra}
MERGE (o:{o_primary} {{name: row.o_name}})
SET o += row.o_props
{set_object_extra}
MERGE (s)-[r:{rel_type} {{subject: row.s_name, object: row.o_name}}]->(o)
SET r += row.r_props
"""

    row: dict[str, object] = {
        "s_name": _sanitize_value(triple.subject_properties.get("name", triple.subject)),
        "o_name": _sanitize_value(triple.object_properties.get("name", triple.object)),
        "s_props": _sanitize_props(triple.subject_properties),
        "o_props": _sanitize_props(triple.object_properties),
        "r_props": _sanitize_props(triple.relationship_properties),
    }
    return query, row


def _merge_triple(tx, triple: KGTriple) -> None:
    query, row = _triple_cypher_parts(triple)
    s_props = row["s_props"]
    o_props = row["o_props"]
    r_props = row["r_props"]

    logger = logging.getLogger(__name__)

    try:
        tx.run(query, rows=[row]).consume()
    except CypherTypeError as e:
        # log details for debugging and skip this triple to allow ingestion to continue
        debug = {
            "subject": triple.subject,
            "predicate": triple.predicate,
            "object": triple.object,
            "subject_labels": triple.subject_labels,
            "object_labels": triple.object_labels,
            "s_name": row["s_name"],
            "o_name": row["o_name"],
            "s_props_sanitized": s_props,
            "o_props_sanitized": o_props,
            "r_props_sanitized": r_props,
        }
        logger.exception("CypherTypeError writing triple: %s", e)
        try:
            logs_dir = Path(__file__).resolve().parents[1] / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            with (logs_dir / "problematic_triples.jsonl").open(
                "a", encoding="utf-8"
            ) as fh:
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
            with (logs_dir / "problematic_triples.jsonl").open(
                "a", encoding="utf-8"
            ) as fh:
                fh.write(
                    json.dumps(
                        {
                            "error": str(e),
                            "subject": triple.subject,
                            "predicate": triple.predicate,
                            "object": triple.object,
                            "s_props_sanitized": s_props,
                            "o_props_sanitized": o_props,
                            "r_props_sanitized": r_props,
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )
        except Exception:
            logger.exception("Failed to write unexpected error info to log file")
        return


def _merge_triples_batch(tx, query: str, rows: list[dict[str, object]]) -> None:
    tx.run(query, rows=rows).consume()


def ingest_triples(
    triples: list[KGTriple],
    uri: str,
    user: str,
    password: str,
    database: str | None = None,
    log_every: int = 0,
    batch_size: int = 200,
) -> int:
    """Ingest triples batched with UNWIND, grouped by label/predicate signature.

    Falls back to per-triple ingestion (which logs and skips problematic
    triples) when a whole batch fails.
    """
    count = 0
    total = len(triples)
    logger = logging.getLogger(__name__)

    # Group triples that share the same Cypher text so each group can be sent
    # as one UNWIND statement. MERGE is idempotent, so cross-group reordering
    # does not change the resulting graph.
    grouped: dict[str, tuple[str, list[tuple[KGTriple, dict[str, object]]]]] = {}
    for triple in triples:
        query, row = _triple_cypher_parts(triple)
        if query not in grouped:
            grouped[query] = (query, [])
        grouped[query][1].append((triple, row))

    batch_size = max(1, int(batch_size))
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session(database=database) as session:
            with tqdm(
                total=total, desc="Stage 6 Neo4j Ingestion", unit="triple"
            ) as progress:
                for query, items in grouped.values():
                    for start in range(0, len(items), batch_size):
                        batch = items[start : start + batch_size]
                        rows = [row for _, row in batch]
                        try:
                            session.execute_write(_merge_triples_batch, query, rows)
                        except Exception as exc:
                            logger.warning(
                                "Batch ingestion failed (%d triples), retrying "
                                "one by one: %s",
                                len(batch),
                                exc,
                            )
                            for triple, _row in batch:
                                try:
                                    session.execute_write(_merge_triple, triple)
                                except Exception:
                                    # Errors surfacing at commit time (e.g.
                                    # ConstraintError) bypass _merge_triple's
                                    # internal handling — skip the triple.
                                    logger.exception(
                                        "Skipping triple after per-triple retry "
                                        "failed: %s -[%s]-> %s",
                                        triple.subject,
                                        triple.predicate,
                                        triple.object,
                                    )
                        count += len(batch)
                        progress.update(len(batch))
                        if log_every > 0 and count % log_every < batch_size:
                            logger.info(
                                "ingest_progress count=%d total=%d", count, total
                            )
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


def run_quality_checks(
    uri: str,
    user: str,
    password: str,
    report_path: Path,
    database: str | None = None,
    relation_vocab: list[str] | None = None,
) -> None:
    """Run post-ingestion validation Cypher queries and write a report.

    ``relation_vocab`` drives the out-of-vocab predicate check; when omitted the
    check is skipped (system predicates SAME_AS/MENTIONED_IN are always allowed).
    """
    queries = {}
    if relation_vocab:
        allowed = sorted(
            {str(item).strip().upper() for item in relation_vocab if str(item).strip()}
            | {"SAME_AS", "MENTIONED_IN"}
        )
        queries["predicates_out_of_vocab"] = (
            "MATCH ()-[r]->() WHERE NOT type(r) IN $allowed_predicates "
            "RETURN type(r) AS outOfVocab, count(*) AS n ORDER BY n DESC"
        )
    queries["duplicate_nodes_by_name"] = """
MATCH (n)
WITH n.name AS name, collect(labels(n)) AS labelSets, count(*) AS c
WHERE c > 1
RETURN name, labelSets, c ORDER BY c DESC
        """.strip()
    queries["sparsely_connected_nodes"] = """
MATCH (n)
WHERE size((n)--()) <= 1
RETURN labels(n) AS labels, n.name AS name LIMIT 20
        """.strip()

    report: dict[str, object] = {"queries": {}}
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session(database=database) as session:
            for key, q in queries.items():
                params = (
                    {"allowed_predicates": allowed}
                    if key == "predicates_out_of_vocab"
                    else {}
                )
                try:
                    data = session.run(q, **params).data()
                except Exception as e:
                    data = {"error": str(e)}
                report["queries"][key] = data

    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        # best-effort: try simple write to current working dir
        try:
            with open("kg_quality_report.txt", "w", encoding="utf-8") as fh:
                fh.write(json.dumps(report, ensure_ascii=False, indent=2))
        except Exception:
            pass


def load_triples(path: Path) -> list[KGTriple]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [KGTriple.model_validate(item) for item in payload]


def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples-json", required=True)
    parser.add_argument("--database", default="")
    parser.add_argument(
        "--env-file",
        default="kg_pipeline/.env",
        help="Optional .env file to load Neo4j credentials from",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--log-file",
        default="",
        help="Optional log file path for ingest progress",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Log progress every N triples (0 disables)",
    )
    parser.add_argument(
        "--relation-vocab-json",
        default="",
        help="Relation vocab JSON for the out-of-vocab quality check (skipped if empty)",
    )
    args = parser.parse_args()

    env_file = args.env_file.strip()
    if env_file:
        env_path = Path(env_file)
        if env_path.exists():
            load_dotenv(env_path, override=False)

    _setup_logging(args.log_level, args.log_file.strip() or None)

    uri, user, password, env_db = _resolve_neo4j_env()
    database = args.database.strip() or env_db

    triples = load_triples(Path(args.triples_json))
    written = ingest_triples(
        triples,
        uri=uri,
        user=user,
        password=password,
        database=database,
        log_every=int(args.log_every or 0),
    )
    summary = summary_counts(uri=uri, user=user, password=password, database=database)
    print(
        json.dumps(
            {"relationships_written": written, "summary": summary},
            ensure_ascii=False,
            indent=2,
        )
    )

    # Run validation queries and write kg_quality_report.txt next to triples JSON
    try:
        relation_vocab = None
        if args.relation_vocab_json.strip():
            relation_vocab = json.loads(
                Path(args.relation_vocab_json).read_text(encoding="utf-8")
            )
        report_path = Path(args.triples_json).resolve().parent / "kg_quality_report.txt"
        run_quality_checks(
            uri=uri,
            user=user,
            password=password,
            report_path=report_path,
            database=database,
            relation_vocab=relation_vocab,
        )
    except Exception:
        pass


if __name__ == "__main__":
    _cli()
