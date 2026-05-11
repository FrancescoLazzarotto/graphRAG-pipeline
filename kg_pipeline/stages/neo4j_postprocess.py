from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable

import yaml
from neo4j import GraphDatabase
from openai import OpenAI

from kg_pipeline.stages.neo4j_ingestion import _resolve_neo4j_env
from kg_pipeline.utils.validation import parse_json_array

LOGGER = logging.getLogger("kg_pipeline.neo4j_postprocess")

_CANONICAL_RELATION_TYPES = [
    "RELATED_TO",
    "AFFECTS",
    "IMPACTS",
    "INFLUENCES",
    "CAUSES",
    "CAUSED_BY",
    "CONTRIBUTES_TO",
    "LEADS_TO",
    "DRIVEN_BY",
    "DEPENDS_ON",
    "ASSOCIATED_WITH",
    "BASED_ON",
    "DERIVED_FROM",
    "PART_OF",
    "HAS_PART",
    "HAS_COMPONENT",
    "COMPOSED_OF",
    "INCLUDES",
    "CONTAINS_DATA",
    "IS_TYPE_OF",
    "DEFINED_AS",
    "HAS_MAXIMUM_LEVEL",
    "HAS_MINIMUM_LEVEL",
    "HAS_VALUE",
    "HAS_UNIT",
    "VALUE_OF",
    "MEASURES",
    "INDICATES",
    "APPLIES_TO",
    "TARGETS",
    "TARGET_OF",
    "REQUIRES",
    "REQUIRED_BY",
    "USES",
    "USED_BY",
    "USES_METHOD",
    "HAS_METHOD",
    "MANAGES",
    "MANAGED_BY",
    "REGULATES",
    "REGULATED_BY",
    "GOVERNS",
    "COMPLIES_WITH",
    "SHOULD_BE_MANAGED_BY",
    "ENSURES",
    "AIMS_TO_ACHIEVE",
    "NEEDED_FOR",
    "PUBLISHED",
    "WORKED_WITH",
    "EXCHANGES_INFO_WITH",
    "TAKE_INTO_ACCOUNT",
    "PRODUCES",
    "LOCATED_IN",
    "OCCURS_IN",
    "BELONGS_TO",
    "HAS_MEMBER",
    "MEMBER_OF",
    "ANALYZES",
    "ESTABLISHES",
    "ESTABLISHED_BY",
]

_DEFAULT_PROPERTY_SCHEMA: dict[str, dict[str, str]] = {
    "Organization": {
        "description": "Short description of the organization",
        "organization_type": "Type such as company, agency, NGO, ministry",
        "country": "Primary country or region",
    },
    "Region": {
        "description": "Short description of the region",
        "region_type": "Type such as country, continent, basin",
        "country": "Country if applicable",
    },
    "Event": {
        "description": "Short description of the event",
        "date": "ISO date or year if known",
        "location": "Location or region",
    },
    "Indicator": {
        "description": "Short definition of the indicator",
        "unit": "Unit of measure if applicable",
        "category": "Category such as climate, nutrition, economy",
    },
}

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_REL_CLEAN = re.compile(r"[^A-Z0-9_]+")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _chunked(items: list[Any], size: int) -> Iterable[list[Any]]:
    if size <= 0:
        size = len(items) or 1
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def _normalize_name(value: str) -> str:
    cleaned = " ".join(value.strip().split()).lower()
    cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned)
    cleaned = _NON_ALNUM.sub(" ", cleaned)
    return " ".join(cleaned.split())


def _normalize_rel_type(value: str) -> str:
    cleaned = _REL_CLEAN.sub("_", value.strip().upper())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def _sanitize_label(value: str) -> str:
    return value.replace("`", "").strip()


def _build_llm_client(base_url: str, api_key: str) -> OpenAI:
    http_timeout = float(os.getenv("VLLM_HTTP_TIMEOUT", "900"))
    return OpenAI(base_url=base_url.rstrip("/"), api_key=api_key or "EMPTY", timeout=http_timeout)


def _setup_logging(log_file: Path | None) -> None:
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    if log_file is not None:
        LOGGER.info("Logging to %s", log_file)


def _confirm_db_changes(uri: str, database: str | None, dry_run: bool, assume_yes: bool) -> None:
    if dry_run or assume_yes:
        return
    db_name = database or "<default>"
    prompt = (
        "This will modify the Neo4j database "
        f"'{db_name}' at {uri}. Type YES to continue: "
    )
    if input(prompt).strip() != "YES":
        raise SystemExit("Aborted by user.")


def _resolve_llm_env() -> tuple[str, str, str]:
    base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1").strip()
    model_name = os.getenv("VLLM_MODEL_NAME", "").strip()
    api_key = os.getenv("VLLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    if not model_name:
        raise ValueError("Missing VLLM_MODEL_NAME for LLM mapping")
    return base_url, model_name, api_key


def _llm_json_array(client: OpenAI, model_name: str, prompt: str) -> list[dict[str, Any]]:
    response = client.chat.completions.create(
        model=model_name,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content or "[]"
    return parse_json_array(content)


def _has_apoc(session) -> bool:
    try:
        session.run("RETURN apoc.version() AS version").single()
        return True
    except Exception:
        return False


def _fetch_relation_types(session) -> list[dict[str, Any]]:
    return session.run(
        "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY count DESC"
    ).data()


def _relation_mapping_prompt(canonical: list[str], items: list[dict[str, Any]]) -> str:
    return (
        "You map Neo4j relationship types to a fixed canonical list.\n"
        "Rules:\n"
        "- Use only the canonical list.\n"
        "- Preserve direction and semantics when possible.\n"
        "- If no good match, use RELATED_TO.\n"
        "Return JSON array of objects: {\"source\": str, \"target\": str}.\n\n"
        "Canonical list:\n"
        f"{json.dumps(canonical, indent=2)}\n\n"
        "Items:\n"
        f"{json.dumps(items, indent=2)}"
    )


def _classify_concepts_prompt(labels: list[str], nodes: list[dict[str, Any]]) -> str:
    return (
        "You assign a single label to each Concept node.\n"
        "Rules:\n"
        "- Use only the allowed labels list.\n"
        "- If unsure, return Concept.\n"
        "Return JSON array of objects: {\"id\": int, \"label\": str}.\n\n"
        "Allowed labels:\n"
        f"{json.dumps(labels, indent=2)}\n\n"
        "Nodes:\n"
        f"{json.dumps(nodes, indent=2)}"
    )


def _enrichment_prompt(schema: dict[str, dict[str, str]], nodes: list[dict[str, Any]]) -> str:
    return (
        "You enrich node properties for a knowledge graph.\n"
        "Rules:\n"
        "- Use only the properties defined in the schema per label.\n"
        "- Return only properties that are missing for the node.\n"
        "- If nothing to add, return an empty properties object.\n"
        "Return JSON array of objects: {\"id\": int, \"properties\": {..}}.\n\n"
        "Schema:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "Nodes:\n"
        f"{json.dumps(nodes, indent=2)}"
    )


def _fallback_relation_target(source: str, canonical_set: set[str]) -> str:
    normalized = _normalize_rel_type(source)
    if normalized in canonical_set:
        return normalized

    for suffix in ("S", "ES", "ED", "ING"):
        if normalized.endswith(suffix):
            candidate = normalized[: -len(suffix)]
            if candidate in canonical_set:
                return candidate

    return "RELATED_TO"


def _load_relation_vocab(path: str) -> list[str]:
    if not path:
        vocab = list(_CANONICAL_RELATION_TYPES)
        if "RELATED_TO" not in vocab:
            vocab.insert(0, "RELATED_TO")
        return vocab
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("relation vocab must be a JSON array of strings")
    vocab = [str(item).strip().upper() for item in payload if str(item).strip()]
    if "RELATED_TO" not in vocab:
        vocab.insert(0, "RELATED_TO")
    return vocab


def _load_property_schema(path: str) -> dict[str, dict[str, str]]:
    if not path:
        return dict(_DEFAULT_PROPERTY_SCHEMA)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("property schema must be a JSON object")
    output: dict[str, dict[str, str]] = {}
    for label, props in payload.items():
        if not isinstance(props, dict):
            continue
        output[str(label)] = {str(k): str(v) for k, v in props.items()}
    return output


def _fetch_node_context(session, ids: list[int]) -> list[dict[str, Any]]:
    if not ids:
        return []
    query = (
        "MATCH (n) WHERE id(n) IN $ids "
        "CALL { "
        "  WITH n "
        "  MATCH (n)-[r]-(m) "
        "  RETURN collect({type: type(r), neighbor: coalesce(m.name, ''), labels: labels(m)})[0..6] AS rels "
        "} "
        "RETURN id(n) AS id, n.name AS name, labels(n) AS labels, rels"
    )
    return session.run(query, ids=ids).data()


def _coerce_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple, set)):
        out: list[Any] = []
        for item in value:
            coerced = _coerce_value(item)
            if coerced is not None:
                out.append(coerced)
        return out
    return str(value)


def _sanitize_props(props: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in props.items():
        coerced = _coerce_value(value)
        if coerced is None:
            continue
        if isinstance(coerced, str) and not coerced.strip():
            continue
        sanitized[str(key)] = coerced
    return sanitized


def _apply_relation_mapping(
    session,
    relation_items: list[dict[str, Any]],
    canonical: list[str],
    client: OpenAI,
    model_name: str,
    dry_run: bool,
    batch_size: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "total_relation_types": len(relation_items),
        "renamed": [],
        "skipped": [],
        "errors": [],
    }

    canonical_set = {item.upper() for item in canonical}
    mapping: dict[str, str] = {}

    for batch in _chunked(relation_items, batch_size):
        prompt = _relation_mapping_prompt(canonical, batch)
        try:
            rows = _llm_json_array(client, model_name, prompt)
        except Exception as exc:
            report["errors"].append(f"llm mapping failed: {exc}")
            rows = []

        for row in rows:
            source = str(row.get("source", "")).strip()
            target = str(row.get("target", "")).strip().upper()
            if not source:
                continue
            if target not in canonical_set:
                target = ""
            if not target:
                target = _fallback_relation_target(source, canonical_set)
            mapping[source] = target

    for item in relation_items:
        source = item["type"]
        if source in mapping:
            continue
        mapping[source] = _fallback_relation_target(source, canonical_set)

    for source, target in mapping.items():
        if source == target:
            report["skipped"].append({"source": source, "target": target})
            continue

        source_safe = source.replace("`", "")
        count_query = f"MATCH ()-[r:`{source_safe}`]->() RETURN count(r) AS c"
        try:
            count = session.run(count_query).single()["c"]
        except Exception as exc:
            report["errors"].append(f"count failed for {source}: {exc}")
            count = 0

        if not dry_run:
            try:
                session.run("CALL apoc.refactor.rename.type($old, $new)", old=source, new=target)
            except Exception as exc:
                report["errors"].append(f"rename failed for {source} -> {target}: {exc}")
                continue

        report["renamed"].append({"source": source, "target": target, "count": int(count)})

    return report


def _find_duplicate_groups(session) -> list[dict[str, Any]]:
    rows = session.run(
        "MATCH (n) "
        "WHERE n.name IS NOT NULL AND trim(n.name) <> '' "
        "OPTIONAL MATCH (n)-[r]-() "
        "WITH n, count(r) AS degree "
        "RETURN id(n) AS id, n.name AS name, labels(n) AS labels, degree"
    ).data()

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        norm = _normalize_name(str(row.get("name", "")))
        if not norm:
            continue
        groups.setdefault(norm, []).append(row)

    return [
        {"normalized": norm, "nodes": items}
        for norm, items in groups.items()
        if len(items) > 1
    ]


def _merge_duplicate_groups(session, groups: list[dict[str, Any]], dry_run: bool) -> dict[str, Any]:
    report: dict[str, Any] = {"groups": 0, "merged_nodes": 0, "errors": [], "samples": []}

    for group in groups:
        nodes = group["nodes"]
        nodes_sorted = sorted(nodes, key=lambda row: (-int(row.get("degree", 0)), int(row["id"])))
        primary = nodes_sorted[0]
        secondary_ids = [int(row["id"]) for row in nodes_sorted[1:]]
        report["groups"] += 1
        report["merged_nodes"] += len(secondary_ids)

        if len(report["samples"]) < 25:
            report["samples"].append(
                {
                    "normalized": group["normalized"],
                    "primary": primary["name"],
                    "secondary": [row["name"] for row in nodes_sorted[1:]],
                }
            )

        if dry_run or not secondary_ids:
            continue

        try:
            session.run(
                "MATCH (n) WHERE id(n) IN $ids "
                "WITH n ORDER BY CASE id(n) WHEN $primary THEN 0 ELSE 1 END, id(n) "
                "WITH collect(n) AS nodes "
                "CALL apoc.refactor.mergeNodes(nodes, {properties: 'discard', mergeRels: true}) "
                "YIELD node RETURN id(node) AS id",
                ids=[int(primary["id"])] + secondary_ids,
                primary=int(primary["id"]),
            ).consume()
        except Exception as exc:
            report["errors"].append(f"merge failed for {group['normalized']}: {exc}")

    return report


def _classify_concepts(
    session,
    labels: list[str],
    client: OpenAI,
    model_name: str,
    dry_run: bool,
    batch_size: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {"candidates": 0, "relabeled": 0, "label_counts": {}, "errors": []}

    rows = session.run(
        "MATCH (n) WHERE 'Concept' IN labels(n) AND size(labels(n)) = 1 "
        "RETURN id(n) AS id, n.name AS name"
    ).data()

    concept_ids = [int(row["id"]) for row in rows]
    report["candidates"] = len(concept_ids)

    for batch in _chunked(concept_ids, batch_size):
        context_rows = _fetch_node_context(session, batch)
        nodes_payload = [
            {
                "id": row["id"],
                "name": row.get("name", ""),
                "context": row.get("rels", []),
            }
            for row in context_rows
        ]

        prompt = _classify_concepts_prompt(labels, nodes_payload)
        try:
            mapped = _llm_json_array(client, model_name, prompt)
        except Exception as exc:
            report["errors"].append(f"concept classification failed: {exc}")
            continue

        for item in mapped:
            node_id = int(item.get("id", -1))
            label = str(item.get("label", "Concept")).strip()
            if label not in labels:
                label = "Concept"

            if label == "Concept":
                continue

            report["label_counts"][label] = report["label_counts"].get(label, 0) + 1
            report["relabeled"] += 1

            if dry_run:
                continue

            safe_label = _sanitize_label(label)
            session.run(
                f"MATCH (n) WHERE id(n) = $id SET n:`{safe_label}` REMOVE n:Concept",
                id=node_id,
            ).consume()

    return report


def _enrich_properties(
    session,
    schema: dict[str, dict[str, str]],
    client: OpenAI,
    model_name: str,
    dry_run: bool,
    batch_size: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {"candidates": 0, "updated_nodes": 0, "updated_props": {}, "errors": []}

    candidates: list[dict[str, Any]] = []
    candidate_lookup: dict[int, dict[str, Any]] = {}
    for label, props in schema.items():
        safe_label = _sanitize_label(label)
        rows = session.run(
            f"MATCH (n:`{safe_label}`) RETURN id(n) AS id, n.name AS name, properties(n) AS props"
        ).data()

        for row in rows:
            existing = row.get("props", {}) or {}
            missing = [
                key
                for key in props.keys()
                if key not in existing or existing.get(key) in (None, "")
            ]
            if not missing:
                continue
            candidate = {
                "id": int(row["id"]),
                "label": label,
                "name": row.get("name", ""),
                "missing": missing,
            }
            candidates.append(candidate)
            candidate_lookup[int(row["id"])] = candidate

    report["candidates"] = len(candidates)

    for batch in _chunked(candidates, batch_size):
        ids = [item["id"] for item in batch]
        context_rows = {row["id"]: row for row in _fetch_node_context(session, ids)}
        payload = []
        for item in batch:
            context = context_rows.get(item["id"], {})
            payload.append(
                {
                    "id": item["id"],
                    "label": item["label"],
                    "name": item["name"],
                    "missing": item["missing"],
                    "context": context.get("rels", []),
                }
            )

        prompt = _enrichment_prompt(schema, payload)
        try:
            rows = _llm_json_array(client, model_name, prompt)
        except Exception as exc:
            report["errors"].append(f"property enrichment failed: {exc}")
            continue

        for row in rows:
            node_id = int(row.get("id", -1))
            props = row.get("properties") or {}
            if not isinstance(props, dict):
                continue
            candidate = candidate_lookup.get(node_id)
            if not candidate:
                continue
            allowed_keys = set(schema.get(candidate["label"], {}).keys())
            allowed_keys &= set(candidate.get("missing", []))

            filtered_props = {key: value for key, value in props.items() if key in allowed_keys}
            clean_props = _sanitize_props(filtered_props)
            if not clean_props:
                continue

            report["updated_nodes"] += 1
            for key in clean_props.keys():
                report["updated_props"][key] = report["updated_props"].get(key, 0) + 1

            if dry_run:
                continue

            session.run("MATCH (n) WHERE id(n) = $id SET n += $props", id=node_id, props=clean_props).consume()

    return report


def _apply_constraints(
    session,
    all_labels: list[str],
    unique_labels: list[str],
    dry_run: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {"created": [], "skipped": [], "errors": []}

    for label in all_labels:
        safe_label = _sanitize_label(label)
        name = f"exists_{safe_label.lower()}_name"
        query = f"CREATE CONSTRAINT {name} IF NOT EXISTS FOR (n:`{safe_label}`) REQUIRE n.name IS NOT NULL"
        if dry_run:
            report["skipped"].append(query)
            continue
        try:
            session.run(query).consume()
            report["created"].append(query)
        except Exception as exc:
            report["errors"].append(f"constraint failed {name}: {exc}")

    for label in unique_labels:
        safe_label = _sanitize_label(label)
        name = f"uniq_{safe_label.lower()}_name"
        query = f"CREATE CONSTRAINT {name} IF NOT EXISTS FOR (n:`{safe_label}`) REQUIRE n.name IS UNIQUE"
        if dry_run:
            report["skipped"].append(query)
            continue
        try:
            session.run(query).consume()
            report["created"].append(query)
        except Exception as exc:
            report["errors"].append(f"constraint failed {name}: {exc}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="kg_pipeline/config.yaml")
    parser.add_argument("--database", default="")
    parser.add_argument("--relation-vocab", default="")
    parser.add_argument("--property-schema", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--log-file", default="")
    parser.add_argument("--relation-batch-size", type=int, default=120)
    parser.add_argument("--concept-batch-size", type=int, default=50)
    parser.add_argument("--enrich-batch-size", type=int, default=30)
    args = parser.parse_args()

    log_path = None
    if args.log_file.strip():
        log_path = Path(args.log_file).expanduser()
    else:
        log_dir = Path("kg_pipeline") / "logs"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"neo4j_postprocess_{timestamp}.log"
    _setup_logging(log_path)

    config = _load_yaml(Path(args.config))
    allowed_labels = [str(label) for label in config.get("ontology", {}).get("labels", [])]
    if "Concept" not in allowed_labels:
        allowed_labels.append("Concept")

    non_concept_labels = [label for label in allowed_labels if label != "Concept"] + ["Concept"]

    relation_vocab = _load_relation_vocab(args.relation_vocab)
    property_schema = _load_property_schema(args.property_schema)

    base_url, model_name, api_key = _resolve_llm_env()
    client = _build_llm_client(base_url=base_url, api_key=api_key)

    uri, user, password, env_db = _resolve_neo4j_env()
    database = args.database.strip() or env_db

    _confirm_db_changes(uri=uri, database=database, dry_run=args.dry_run, assume_yes=args.yes)

    report: dict[str, Any] = {"dry_run": bool(args.dry_run)}

    LOGGER.info("Starting Neo4j postprocess dry_run=%s", args.dry_run)
    LOGGER.info("Target database=%s", database or "<default>")

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session(database=database) as session:
            apoc_available = _has_apoc(session)
            report["apoc_available"] = apoc_available
            LOGGER.info("APOC available=%s", apoc_available)

            relation_items = _fetch_relation_types(session)
            if not apoc_available:
                report["step1_relation_mapping"] = {
                    "error": "APOC unavailable, cannot rename relationship types",
                    "total_relation_types": len(relation_items),
                }
            else:
                LOGGER.info("Step 1: mapping %d relation types", len(relation_items))
                report["step1_relation_mapping"] = _apply_relation_mapping(
                    session=session,
                    relation_items=relation_items,
                    canonical=relation_vocab,
                    client=client,
                    model_name=model_name,
                    dry_run=args.dry_run,
                    batch_size=args.relation_batch_size,
                )
                step1 = report["step1_relation_mapping"]
                LOGGER.info(
                    "Step 1 done: renamed=%d skipped=%d errors=%d",
                    len(step1.get("renamed", [])),
                    len(step1.get("skipped", [])),
                    len(step1.get("errors", [])),
                )

            duplicate_groups = _find_duplicate_groups(session)
            if not apoc_available:
                report["step2_dedup"] = {
                    "error": "APOC unavailable, cannot merge duplicate nodes",
                    "candidate_groups": len(duplicate_groups),
                }
            else:
                LOGGER.info("Step 2: merging %d duplicate groups", len(duplicate_groups))
                report["step2_dedup"] = _merge_duplicate_groups(
                    session=session,
                    groups=duplicate_groups,
                    dry_run=args.dry_run,
                )
                step2 = report["step2_dedup"]
                LOGGER.info(
                    "Step 2 done: groups=%d merged_nodes=%d errors=%d",
                    int(step2.get("groups", 0)),
                    int(step2.get("merged_nodes", 0)),
                    len(step2.get("errors", [])),
                )

            LOGGER.info("Step 3: relabel Concept nodes")
            report["step3_relabel_concept"] = _classify_concepts(
                session=session,
                labels=non_concept_labels,
                client=client,
                model_name=model_name,
                dry_run=args.dry_run,
                batch_size=args.concept_batch_size,
            )
            step3 = report["step3_relabel_concept"]
            LOGGER.info(
                "Step 3 done: candidates=%d relabeled=%d errors=%d",
                int(step3.get("candidates", 0)),
                int(step3.get("relabeled", 0)),
                len(step3.get("errors", [])),
            )

            LOGGER.info("Step 4: enrich node properties")
            report["step4_enrich_properties"] = _enrich_properties(
                session=session,
                schema=property_schema,
                client=client,
                model_name=model_name,
                dry_run=args.dry_run,
                batch_size=args.enrich_batch_size,
            )
            step4 = report["step4_enrich_properties"]
            LOGGER.info(
                "Step 4 done: candidates=%d updated_nodes=%d errors=%d",
                int(step4.get("candidates", 0)),
                int(step4.get("updated_nodes", 0)),
                len(step4.get("errors", [])),
            )

            unique_labels = [
                label
                for label in allowed_labels
                if label in {"Organization", "Region", "Event", "Indicator", "Dataset", "Method", "Policy", "Commodity"}
            ]
            LOGGER.info("Step 5: applying constraints")
            report["step5_constraints"] = _apply_constraints(
                session=session,
                all_labels=allowed_labels,
                unique_labels=unique_labels,
                dry_run=args.dry_run,
            )
            step5 = report["step5_constraints"]
            LOGGER.info(
                "Step 5 done: created=%d skipped=%d errors=%d",
                len(step5.get("created", [])),
                len(step5.get("skipped", [])),
                len(step5.get("errors", [])),
            )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
