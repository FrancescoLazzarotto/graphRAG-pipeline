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
from dotenv import load_dotenv

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

_INVERSE_RELATION_REWRITES = [
    {"from": "ESTABLISHED_BY", "to": "ESTABLISHES"},
    {"from": "USED_BY", "to": "USES"},
    {"from": "CAUSED_BY", "to": "CAUSES"},
    {"from": "REQUIRED_BY", "to": "REQUIRES"},
    {"from": "REGULATES", "to": "REGULATED_BY"},
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
_RELATED_TO_BATCH_SIZE = 50
_RELATION_RECLASS_BATCH_SIZE = 50

_AURA_RECLASS_TYPES = [
    "AFFECTS",
    "CONTRIBUTES_TO",
    "INCLUDES",
    "PART_OF",
    "ANALYZES",
    "RELATED_TO",
]

_AURA_REGION_GARBAGE_NAMES = [
    "REGIONS/SUBREGIONS/COUNTRIES/TERRITORIES",
    "ASIA*",
]

_AURA_INVERSE_REWRITES = [
    {"from": "REGULATES", "to": "REGULATED_BY"},
    {"from": "USED_BY", "to": "USES"},
    {"from": "ESTABLISHED_BY", "to": "ESTABLISHES"},
]

_AURA_RENAME_REWRITES = [
    {"from": "INFLUENCES", "to": "AFFECTS"},
]

_MICRO_RELATION_REWRITES = [
    {"from": "COMPOSED_OF", "to": "INCLUDES"},
    {"from": "USES_METHOD", "to": "USES"},
    {"from": "TARGETS", "to": "GOVERNS"},
    {"from": "MANAGES", "to": "GOVERNS"},
    {"from": "DEPENDS_ON", "to": "REQUIRES"},
    {"from": "PART_OF", "to": "HAS_COMPONENT"},
    {"from": "ASSOCIATED_WITH", "to": "RELATED_TO"},
    {"from": "OCCURS_IN", "to": "LOCATED_IN"},
    {"from": "LEADS_TO", "to": "AFFECTS"},
    {"from": "IMPACTS", "to": "AFFECTS"},
]


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


def _cypher_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


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


def _fetch_relation_types(session, max_patterns: int) -> list[dict[str, Any]]:
    query = (
        "MATCH (a)-[r]->(b) "
        "WITH type(r) AS type, labels(a) AS source_labels, labels(b) AS target_labels, count(*) AS count "
        "ORDER BY count DESC "
        "WITH type, collect({source_labels: source_labels, target_labels: target_labels, count: count}) AS patterns, sum(count) AS total "
        "RETURN type, total AS count, patterns[0..$max_patterns] AS patterns "
        "ORDER BY count DESC"
    )
    return session.run(query, max_patterns=max_patterns).data()


def _relation_mapping_prompt(canonical: list[str], items: list[dict[str, Any]]) -> str:
    return (
        "You map Neo4j relationship types to a fixed canonical list.\n"
        "Rules:\n"
        "- Use only the canonical list.\n"
        "- Preserve direction and semantics when possible.\n"
        "- Prefer the most specific relation; avoid RELATED_TO unless nothing fits.\n"
        "- Use endpoint label patterns if provided.\n"
        "Return JSON array of objects: {\"source\": str, \"target\": str}.\n\n"
        "Canonical list:\n"
        f"{json.dumps(canonical, indent=2)}\n\n"
        "Items:\n"
        f"{json.dumps(items, indent=2)}"
    )


def _related_to_refinement_prompt(canonical: list[str], items: list[dict[str, Any]]) -> str:
    return (
        "You refine RELATED_TO relationships to a more specific predicate.\n"
        "Rules:\n"
        "- Use only the canonical list.\n"
        "- Keep direction and semantics.\n"
        "- Prefer the most specific relation; use RELATED_TO only if nothing fits.\n"
        "Return JSON array of objects: {\"id\": int, \"type\": str}.\n\n"
        "Canonical list:\n"
        f"{json.dumps(canonical, indent=2)}\n\n"
        "Items:\n"
        f"{json.dumps(items, indent=2)}"
    )


def _relation_reclass_prompt(allowed: list[str], items: list[dict[str, Any]]) -> str:
    return (
        "You reclassify Neo4j relationships to a fixed allowed list.\n"
        "Rules:\n"
        "- Use only the allowed list.\n"
        "- Keep direction and semantics.\n"
        "- Use RELATED_TO only if nothing fits.\n"
        "Return JSON array of objects: {\"id\": int, \"type\": str}.\n\n"
        "Allowed list:\n"
        f"{json.dumps(allowed, indent=2)}\n\n"
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


def _labels_compatible(primary: list[str], secondary: list[str], mode: str) -> bool:
    if mode == "any":
        return True
    primary_set = set(primary or [])
    secondary_set = set(secondary or [])
    if mode == "exact":
        return primary_set == secondary_set
    return bool(primary_set & secondary_set)


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


def _fetch_related_to_ids(session) -> list[int]:
    rows = session.run("MATCH ()-[r:RELATED_TO]->() RETURN id(r) AS id ORDER BY id(r)").data()
    return [int(row["id"]) for row in rows]


def _fetch_related_to_context(session, rel_ids: list[int]) -> list[dict[str, Any]]:
    if not rel_ids:
        return []
    query = (
        "UNWIND $ids AS rid "
        "MATCH (s)-[r:RELATED_TO]->(t) "
        "WHERE id(r) = rid "
        "CALL { "
        "  WITH s, rid "
        "  MATCH (s)-[rs]-(sn) "
        "  WHERE id(rs) <> rid "
        "  RETURN collect({type: type(rs), neighbor: coalesce(sn.name, ''), labels: labels(sn)})[0..3] AS s_rels "
        "} "
        "CALL { "
        "  WITH t, rid "
        "  MATCH (t)-[rt]-(tn) "
        "  WHERE id(rt) <> rid "
        "  RETURN collect({type: type(rt), neighbor: coalesce(tn.name, ''), labels: labels(tn)})[0..3] AS t_rels "
        "} "
        "RETURN id(r) AS id, "
        "  {labels: labels(s), name: coalesce(s.name, '')} AS source, "
        "  {labels: labels(t), name: coalesce(t.name, '')} AS target, "
        "  s_rels AS source_context, t_rels AS target_context"
    )
    return session.run(query, ids=rel_ids).data()


def _fetch_relation_context(session, rel_ids: list[int], rel_type: str) -> list[dict[str, Any]]:
    if not rel_ids:
        return []
    safe_type = _normalize_rel_type(rel_type)
    query = (
        "UNWIND $ids AS rid "
        f"MATCH (s)-[r:`{safe_type}`]->(t) "
        "WHERE id(r) = rid "
        "CALL { "
        "  WITH s, rid "
        "  MATCH (s)-[rs]-(sn) "
        "  WHERE id(rs) <> rid "
        "  RETURN collect({type: type(rs), neighbor: coalesce(sn.name, ''), labels: labels(sn)})[0..3] AS s_rels "
        "} "
        "CALL { "
        "  WITH t, rid "
        "  MATCH (t)-[rt]-(tn) "
        "  WHERE id(rt) <> rid "
        "  RETURN collect({type: type(rt), neighbor: coalesce(tn.name, ''), labels: labels(tn)})[0..3] AS t_rels "
        "} "
        "RETURN id(r) AS id, type(r) AS current_type, "
        "  {labels: labels(s), name: coalesce(s.name, '')} AS source, "
        "  {labels: labels(t), name: coalesce(t.name, '')} AS target, "
        "  s_rels AS source_context, t_rels AS target_context"
    )
    return session.run(query, ids=rel_ids).data()


def _fetch_has_component_anomaly_ids(session) -> list[int]:
    rows = session.run(
        "MATCH (s:Organization)-[r:HAS_COMPONENT]->(t:Concept) RETURN id(r) AS id "
        "UNION "
        "MATCH (s:Concept)-[r:HAS_COMPONENT]->(t:Region) RETURN id(r) AS id"
    ).data()
    return [int(row["id"]) for row in rows]


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
    pending_items: list[dict[str, Any]] = []

    for item in relation_items:
        source = str(item.get("type", "")).strip()
        if not source:
            continue
        normalized = _normalize_rel_type(source)
        if normalized in canonical_set:
            mapping[source] = normalized
        else:
            pending_items.append(item)

    for batch in _chunked(pending_items, batch_size):
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


def _rewrite_inverse_relationships(
    session,
    rewrites: list[dict[str, str]],
    dry_run: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {"pairs": [], "errors": []}

    for item in rewrites:
        source = _normalize_rel_type(str(item.get("from", "")))
        target = _normalize_rel_type(str(item.get("to", "")))
        if not source or not target or source == target:
            continue

        count_query = f"MATCH ()-[r:`{source}`]->() RETURN count(r) AS c"
        try:
            count = session.run(count_query).single()["c"]
        except Exception as exc:
            report["errors"].append(f"inverse count failed for {source}: {exc}")
            continue

        if dry_run or int(count) == 0:
            report["pairs"].append({"source": source, "target": target, "count": int(count), "rewritten": 0})
            continue

        query = (
            f"MATCH (a)-[r:`{source}`]->(b) "
            "WITH a, b, r, properties(r) AS props "
            f"MERGE (b)-[r2:`{target}`]->(a) "
            "SET r2 += props "
            "DELETE r "
            "RETURN count(r2) AS rewritten"
        )
        try:
            rewritten = session.run(query).single()["rewritten"]
        except Exception as exc:
            report["errors"].append(f"inverse rewrite failed for {source} -> {target}: {exc}")
            continue

        report["pairs"].append(
            {"source": source, "target": target, "count": int(count), "rewritten": int(rewritten)}
        )

    return report


def _invert_published_direction(session, dry_run: bool) -> dict[str, Any]:
    report: dict[str, Any] = {"count": 0, "rewritten": 0, "errors": []}
    try:
        count = session.run(
            "MATCH (d:Document)-[r:PUBLISHED]->(o:Organization) RETURN count(r) AS c"
        ).single()["c"]
        report["count"] = int(count)
    except Exception as exc:
        report["errors"].append(f"published count failed: {exc}")
        return report

    if dry_run or report["count"] == 0:
        return report

    query = (
        "MATCH (d:Document)-[r:PUBLISHED]->(o:Organization) "
        "WITH d, o, r, properties(r) AS props "
        "MERGE (o)-[r2:PUBLISHED]->(d) "
        "SET r2 += props "
        "DELETE r "
        "RETURN count(r2) AS rewritten"
    )
    try:
        report["rewritten"] = int(session.run(query).single()["rewritten"])
    except Exception as exc:
        report["errors"].append(f"published inversion failed: {exc}")

    return report


def _cleanup_named_region_nodes(session, names: list[str], dry_run: bool) -> dict[str, Any]:
    report: dict[str, Any] = {
        "candidates": 0,
        "matched": 0,
        "deleted_nodes": 0,
        "rewired_relationships": 0,
        "deleted_relationships": 0,
        "errors": [],
        "samples": [],
        "by_name": [],
    }

    cleaned_names = [str(name).strip() for name in names if str(name).strip()]
    if not cleaned_names:
        return report

    for name in cleaned_names:
        name_report = {
            "name": name,
            "found": False,
            "candidates": 0,
            "matched": 0,
            "deleted_nodes": 0,
            "rewired_relationships": 0,
            "deleted_relationships": 0,
            "errors": [],
            "samples": [],
        }
        literal = _cypher_string_literal(name)
        rows = session.run(
            "MATCH (n:Region) "
            f"WHERE n.name = {literal} "
            "CALL { "
            "  WITH n "
            "  OPTIONAL MATCH (m:Region) "
            "  WHERE id(m) <> id(n) "
            "    AND toLower(trim(m.name)) = toLower(trim(n.name)) "
            "  WITH m "
            "  ORDER BY id(m) "
            "  LIMIT 1 "
            "  RETURN id(m) AS match_id, m.name AS match_name "
            "} "
            "RETURN id(n) AS id, n.name AS name, match_id, match_name"
        ).data()

        name_report["candidates"] = len(rows)
        name_report["found"] = bool(rows)
        report["candidates"] += len(rows)

        for row in rows:
            bad_id = int(row["id"])
            match_id = row.get("match_id")
            match_id = int(match_id) if match_id is not None else None

            rel_count = 0
            try:
                rel_count = int(
                    session.run(
                        "MATCH (n) WHERE id(n) = $id MATCH (n)-[r]-() RETURN count(r) AS c",
                        id=bad_id,
                    ).single()["c"]
                )
            except Exception as exc:
                error_msg = f"region rel count failed for {bad_id}: {exc}"
                report["errors"].append(error_msg)
                name_report["errors"].append(error_msg)

            if match_id is not None:
                report["matched"] += 1
                report["deleted_nodes"] += 1
                name_report["matched"] += 1
                name_report["deleted_nodes"] += 1
                if len(report["samples"]) < 20:
                    report["samples"].append({"from": row.get("name", ""), "to": row.get("match_name", "")})
                if len(name_report["samples"]) < 20:
                    name_report["samples"].append(
                        {"from": row.get("name", ""), "to": row.get("match_name", "")}
                    )

                if dry_run:
                    report["rewired_relationships"] += rel_count
                    name_report["rewired_relationships"] += rel_count
                    continue

                query = (
                    "MATCH (bad:Region) WHERE id(bad) = $bad_id "
                    "MATCH (match:Region) WHERE id(match) = $match_id "
                    "CALL { "
                    "  WITH bad, match "
                    "  MATCH (bad)-[r]->(n) "
                    "  WHERE id(n) <> id(match) "
                    "  WITH match, n, r, type(r) AS rel_type, properties(r) AS props "
                    "  CALL apoc.create.relationship(match, rel_type, props, n) YIELD rel "
                    "  DELETE r "
                    "  RETURN count(rel) AS out_count "
                    "} "
                    "CALL { "
                    "  WITH bad, match "
                    "  MATCH (n)-[r]->(bad) "
                    "  WHERE id(n) <> id(match) "
                    "  WITH match, n, r, type(r) AS rel_type, properties(r) AS props "
                    "  CALL apoc.create.relationship(n, rel_type, props, match) YIELD rel "
                    "  DELETE r "
                    "  RETURN count(rel) AS in_count "
                    "} "
                    "DETACH DELETE bad "
                    "RETURN out_count + in_count AS rewired"
                )
                try:
                    rewired = int(session.run(query, bad_id=bad_id, match_id=match_id).single()["rewired"])
                    report["rewired_relationships"] += rewired
                    name_report["rewired_relationships"] += rewired
                    deleted_rels = max(0, rel_count - rewired)
                    report["deleted_relationships"] += deleted_rels
                    name_report["deleted_relationships"] += deleted_rels
                except Exception as exc:
                    error_msg = f"region rewire failed for {bad_id}: {exc}"
                    report["errors"].append(error_msg)
                    name_report["errors"].append(error_msg)
                continue

            report["deleted_nodes"] += 1
            report["deleted_relationships"] += rel_count
            name_report["deleted_nodes"] += 1
            name_report["deleted_relationships"] += rel_count
            if len(report["samples"]) < 20:
                report["samples"].append({"from": row.get("name", ""), "to": None})
            if len(name_report["samples"]) < 20:
                name_report["samples"].append({"from": row.get("name", ""), "to": None})
            if dry_run:
                continue

            try:
                session.run("MATCH (n:Region) WHERE id(n) = $id DETACH DELETE n", id=bad_id).consume()
            except Exception as exc:
                error_msg = f"region delete failed for {bad_id}: {exc}"
                report["errors"].append(error_msg)
                name_report["errors"].append(error_msg)

        report["by_name"].append(name_report)

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


def _merge_duplicate_groups(
    session,
    groups: list[dict[str, Any]],
    dry_run: bool,
    label_mode: str,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "groups": 0,
        "merged_nodes": 0,
        "skipped_incompatible": 0,
        "errors": [],
        "samples": [],
    }

    for group in groups:
        nodes = group["nodes"]
        nodes_sorted = sorted(nodes, key=lambda row: (-int(row.get("degree", 0)), int(row["id"])))
        primary = nodes_sorted[0]
        primary_labels = primary.get("labels", []) or []
        compatible = []
        for row in nodes_sorted[1:]:
            if _labels_compatible(primary_labels, row.get("labels", []), label_mode):
                compatible.append(row)
            else:
                report["skipped_incompatible"] += 1

        secondary_ids = [int(row["id"]) for row in compatible]
        report["groups"] += 1
        report["merged_nodes"] += len(secondary_ids)

        if len(report["samples"]) < 25:
            report["samples"].append(
                {
                    "normalized": group["normalized"],
                    "primary": primary["name"],
                    "secondary": [row["name"] for row in compatible],
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
    label_mode: str,
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
            if label_mode == "add":
                session.run(
                    f"MATCH (n) WHERE id(n) = $id SET n:`{safe_label}`",
                    id=node_id,
                ).consume()
            else:
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


def _refine_related_to_relationships(
    session,
    canonical: list[str],
    client: OpenAI,
    model_name: str,
    dry_run: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "total_related_to": 0,
        "updated": 0,
        "skipped": 0,
        "type_counts": {},
        "batches": 0,
        "errors": [],
    }

    rel_ids = _fetch_related_to_ids(session)
    report["total_related_to"] = len(rel_ids)
    if not rel_ids:
        return report

    canonical_set = {item.upper() for item in canonical}

    for batch_ids in _chunked(rel_ids, _RELATED_TO_BATCH_SIZE):
        report["batches"] += 1
        context_rows = _fetch_related_to_context(session, batch_ids)
        prompt = _related_to_refinement_prompt(canonical, context_rows)
        try:
            rows = _llm_json_array(client, model_name, prompt)
        except Exception as exc:
            report["errors"].append(f"RELATED_TO refinement failed: {exc}")
            continue

        updates: list[dict[str, Any]] = []
        for row in rows:
            try:
                rel_id = int(row.get("id", -1))
            except (TypeError, ValueError):
                rel_id = -1
            if rel_id < 0:
                report["skipped"] += 1
                continue

            raw_type = str(row.get("type", "")).strip()
            normalized = _normalize_rel_type(raw_type)
            if normalized not in canonical_set:
                normalized = "RELATED_TO"

            report["type_counts"][normalized] = report["type_counts"].get(normalized, 0) + 1
            if normalized == "RELATED_TO":
                report["skipped"] += 1
                continue

            updates.append({"id": rel_id, "type": normalized})

        if not updates or dry_run:
            report["updated"] += 0
            continue

        try:
            result = session.run(
                "UNWIND $updates AS item "
                "MATCH ()-[r]->() WHERE id(r) = item.id "
                "CALL apoc.refactor.setType(r, item.type) YIELD output "
                "RETURN count(output) AS updated",
                updates=updates,
            ).single()
            report["updated"] += int(result["updated"])
        except Exception as exc:
            report["errors"].append(f"RELATED_TO update failed: {exc}")

    return report


def _reclassify_relationships(
    session,
    rel_ids: list[int],
    rel_type: str,
    allowed: list[str],
    client: OpenAI,
    model_name: str,
    dry_run: bool,
    batch_size: int,
    skip_when_type: str | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "total_candidates": len(rel_ids),
        "updated": 0,
        "skipped": 0,
        "type_counts": {},
        "batches": 0,
        "errors": [],
    }

    if not rel_ids:
        return report

    allowed_set = {item.upper() for item in allowed}
    normalized_skip = _normalize_rel_type(skip_when_type) if skip_when_type else None

    total_batches = (len(rel_ids) + batch_size - 1) // batch_size

    for batch_index, batch_ids in enumerate(_chunked(rel_ids, batch_size), start=1):
        report["batches"] += 1
        LOGGER.info("Reclass batch %d/%d: ids=%d", batch_index, total_batches, len(batch_ids))
        batch_set = {int(item) for item in batch_ids}
        context_rows = _fetch_relation_context(session, batch_ids, rel_type)
        prompt = _relation_reclass_prompt(allowed, context_rows)
        try:
            rows = _llm_json_array(client, model_name, prompt)
        except Exception as exc:
            report["errors"].append(f"relation reclass failed: {exc}")
            LOGGER.warning("Reclass batch %d/%d failed: %s", batch_index, total_batches, exc)
            continue

        updates: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        batch_skipped = 0

        for row in rows:
            try:
                rel_id = int(row.get("id", -1))
            except (TypeError, ValueError):
                rel_id = -1
            if rel_id < 0 or rel_id not in batch_set:
                continue

            seen_ids.add(rel_id)
            raw_type = str(row.get("type", "")).strip()
            normalized = _normalize_rel_type(raw_type)
            if normalized not in allowed_set:
                normalized = "RELATED_TO"

            report["type_counts"][normalized] = report["type_counts"].get(normalized, 0) + 1
            if normalized_skip and normalized == normalized_skip:
                report["skipped"] += 1
                batch_skipped += 1
                continue

            updates.append({"id": rel_id, "type": normalized})

        missing = len(batch_set - seen_ids)
        if missing:
            report["skipped"] += missing
            batch_skipped += missing

        if not updates:
            LOGGER.info(
                "Reclass batch %d/%d: updates=0 skipped=%d",
                batch_index,
                total_batches,
                batch_skipped,
            )
            continue

        if dry_run:
            LOGGER.info(
                "Reclass batch %d/%d: dry_run updates=%d skipped=%d",
                batch_index,
                total_batches,
                len(updates),
                batch_skipped,
            )
            continue

        try:
            result = session.run(
                "UNWIND $updates AS item "
                "MATCH ()-[r]->() WHERE id(r) = item.id "
                "CALL apoc.refactor.setType(r, item.type) YIELD output "
                "RETURN count(output) AS updated",
                updates=updates,
            ).single()
            updated_count = int(result["updated"])
            report["updated"] += updated_count
            LOGGER.info(
                "Reclass batch %d/%d: updated=%d skipped=%d",
                batch_index,
                total_batches,
                updated_count,
                batch_skipped,
            )
        except Exception as exc:
            report["errors"].append(f"relation reclass update failed: {exc}")
            LOGGER.warning("Reclass batch %d/%d update failed: %s", batch_index, total_batches, exc)

    return report


def _reclassify_has_component_anomalies(
    session,
    client: OpenAI,
    model_name: str,
    dry_run: bool,
    batch_size: int,
) -> dict[str, Any]:
    rel_ids = _fetch_has_component_anomaly_ids(session)
    return _reclassify_relationships(
        session=session,
        rel_ids=rel_ids,
        rel_type="HAS_COMPONENT",
        allowed=_AURA_RECLASS_TYPES,
        client=client,
        model_name=model_name,
        dry_run=dry_run,
        batch_size=batch_size,
        skip_when_type=None,
    )


def _reclassify_related_to_second_pass(
    session,
    client: OpenAI,
    model_name: str,
    allowed: list[str],
    dry_run: bool,
    batch_size: int,
) -> dict[str, Any]:
    rel_ids = _fetch_related_to_ids(session)
    return _reclassify_relationships(
        session=session,
        rel_ids=rel_ids,
        rel_type="RELATED_TO",
        allowed=allowed,
        client=client,
        model_name=model_name,
        dry_run=dry_run,
        batch_size=batch_size,
        skip_when_type="RELATED_TO",
    )


def _find_region_artifacts(session) -> list[dict[str, Any]]:
    query = (
        "MATCH (n:Region) "
        "WHERE n.name IS NOT NULL AND trim(n.name) <> '' "
        "  AND ( "
        "    n.name CONTAINS '/' "
        "    OR n.name CONTAINS '*' "
        "    OR ( "
        "      n.name = toUpper(n.name) "
        "      AND size([w IN split(trim(n.name), ' ') WHERE w <> '']) > 3 "
        "    ) "
        "  ) "
        "CALL { "
        "  WITH n "
        "  MATCH (m:Region) "
        "  WHERE id(m) <> id(n) "
        "    AND toLower(trim(m.name)) = toLower(trim(n.name)) "
        "  RETURN id(m) AS match_id, m.name AS match_name "
        "  ORDER BY id(m) "
        "  LIMIT 1 "
        "} "
        "RETURN id(n) AS id, n.name AS name, match_id, match_name"
    )
    return session.run(query).data()


def _cleanup_region_artifacts(session, dry_run: bool) -> dict[str, Any]:
    report: dict[str, Any] = {
        "candidates": 0,
        "matched": 0,
        "deleted_nodes": 0,
        "rewired_relationships": 0,
        "deleted_relationships": 0,
        "errors": [],
        "samples": [],
    }

    artifacts = _find_region_artifacts(session)
    report["candidates"] = len(artifacts)

    for row in artifacts:
        bad_id = int(row["id"])
        match_id = row.get("match_id")
        match_id = int(match_id) if match_id is not None else None

        rel_count = 0
        try:
            rel_count = int(
                session.run(
                    "MATCH (n) WHERE id(n) = $id MATCH (n)-[r]-() RETURN count(r) AS c",
                    id=bad_id,
                ).single()["c"]
            )
        except Exception as exc:
            report["errors"].append(f"region rel count failed for {bad_id}: {exc}")

        if match_id is not None:
            report["matched"] += 1
            report["rewired_relationships"] += rel_count
            if len(report["samples"]) < 20:
                report["samples"].append(
                    {"from": row.get("name", ""), "to": row.get("match_name", "")}
                )
            if dry_run:
                continue
            try:
                session.run(
                    "MATCH (match:Region) WHERE id(match) = $match_id "
                    "MATCH (bad:Region) WHERE id(bad) = $bad_id "
                    "WITH [match, bad] AS nodes "
                    "CALL apoc.refactor.mergeNodes(nodes, {properties: 'discard', mergeRels: true}) "
                    "YIELD node RETURN id(node) AS id",
                    match_id=match_id,
                    bad_id=bad_id,
                ).consume()
            except Exception as exc:
                report["errors"].append(f"region merge failed for {bad_id}: {exc}")
            continue

        report["deleted_nodes"] += 1
        report["deleted_relationships"] += rel_count
        if len(report["samples"]) < 20:
            report["samples"].append({"from": row.get("name", ""), "to": None})
        if dry_run:
            continue

        try:
            session.run("MATCH (n:Region) WHERE id(n) = $id DETACH DELETE n", id=bad_id).consume()
        except Exception as exc:
            report["errors"].append(f"region delete failed for {bad_id}: {exc}")

    return report


def _absorb_micro_relation_types(
    session,
    rewrites: list[dict[str, str]],
    dry_run: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {"pairs": [], "errors": []}

    for item in rewrites:
        source = _normalize_rel_type(str(item.get("from", "")))
        target = _normalize_rel_type(str(item.get("to", "")))
        if not source or not target or source == target:
            continue

        count_query = f"MATCH ()-[r:`{source}`]->() RETURN count(r) AS c"
        try:
            count = int(session.run(count_query).single()["c"])
        except Exception as exc:
            report["errors"].append(f"micro type count failed for {source}: {exc}")
            continue

        updated = 0
        if not dry_run and count > 0:
            try:
                session.run("CALL apoc.refactor.rename.type($old, $new)", old=source, new=target).consume()
                updated = count
            except Exception as exc:
                report["errors"].append(f"micro type rename failed for {source} -> {target}: {exc}")

        report["pairs"].append({"source": source, "target": target, "count": count, "updated": updated})

    return report


def _count_relationships(session, rel_type: str) -> int:
    safe_type = _normalize_rel_type(rel_type)
    query = f"MATCH ()-[r:`{safe_type}`]->() RETURN count(r) AS c"
    return int(session.run(query).single()["c"])


def _run_aura_issues(
    session,
    relation_vocab: list[str],
    dry_run: bool,
    batch_size: int,
    apoc_available: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {}

    if apoc_available:
        garbage_report = _cleanup_named_region_nodes(
            session=session,
            names=_AURA_REGION_GARBAGE_NAMES,
            dry_run=dry_run,
        )
    else:
        total_candidates = 0
        for name in _AURA_REGION_GARBAGE_NAMES:
            literal = _cypher_string_literal(str(name).strip())
            count = session.run(
                "MATCH (n:Region) "
                f"WHERE n.name = {literal} "
                "RETURN count(n) AS c"
            ).single()["c"]
            total_candidates += int(count)
        garbage_report = {
            "candidates": total_candidates,
            "matched": 0,
            "deleted_nodes": 0,
            "rewired_relationships": 0,
            "deleted_relationships": 0,
            "errors": ["APOC unavailable, cannot rewire/delete Region garbage nodes"],
            "samples": [],
            "by_name": [],
        }

    garbage_found = int(garbage_report.get("candidates", 0)) > 0
    garbage_edges = int(garbage_report.get("rewired_relationships", 0)) + int(
        garbage_report.get("deleted_relationships", 0)
    )
    report["garbage_nodes"] = {
        "found": garbage_found,
        "deleted_nodes": int(garbage_report.get("deleted_nodes", 0)),
        "edges_modified": garbage_edges,
        "details": garbage_report,
    }
    LOGGER.info(
        "Issue 1 garbage nodes: %s (deleted_nodes=%d edges_modified=%d)",
        "TROVATO" if garbage_found else "NON TROVATO",
        int(garbage_report.get("deleted_nodes", 0)),
        garbage_edges,
    )

    related_total = _count_relationships(session, "RELATED_TO")
    if related_total > 50:
        if not apoc_available:
            related_report = {
                "total_candidates": related_total,
                "updated": 0,
                "skipped": related_total,
                "errors": ["APOC unavailable, cannot reclassify RELATED_TO"],
            }
        else:
            base_url, model_name, api_key = _resolve_llm_env()
            client = _build_llm_client(base_url=base_url, api_key=api_key)
            related_report = _reclassify_related_to_second_pass(
                session=session,
                client=client,
                model_name=model_name,
                allowed=relation_vocab,
                dry_run=dry_run,
                batch_size=batch_size,
            )
        report["related_to_reclass"] = {
            "found": True,
            "total_related_to": related_total,
            "edges_modified": int(related_report.get("updated", 0)),
            "details": related_report,
        }
        LOGGER.info(
            "Issue 2 RELATED_TO reclass: TROVATO (total=%d updated=%d)",
            related_total,
            int(related_report.get("updated", 0)),
        )
    else:
        report["related_to_reclass"] = {
            "found": False,
            "total_related_to": related_total,
            "edges_modified": 0,
            "details": {"total_candidates": related_total, "updated": 0, "skipped": related_total},
        }
        LOGGER.info(
            "Issue 2 RELATED_TO reclass: NON TROVATO (total=%d <= 50)",
            related_total,
        )

    if apoc_available:
        micro_report = _absorb_micro_relation_types(
            session=session,
            rewrites=_MICRO_RELATION_REWRITES,
            dry_run=dry_run,
        )
    else:
        pairs = []
        for item in _MICRO_RELATION_REWRITES:
            source = _normalize_rel_type(str(item.get("from", "")))
            target = _normalize_rel_type(str(item.get("to", "")))
            if not source or not target or source == target:
                continue
            count = _count_relationships(session, source)
            pairs.append({"source": source, "target": target, "count": count, "updated": 0})
        micro_report = {
            "pairs": pairs,
            "errors": ["APOC unavailable, cannot rename relationship types"],
        }

    micro_pairs = micro_report.get("pairs", [])
    micro_found = any(int(pair.get("count", 0)) > 0 for pair in micro_pairs)
    micro_updated = sum(int(pair.get("updated", 0)) for pair in micro_pairs)
    report["micro_type_consolidation"] = {
        "found": micro_found,
        "edges_modified": micro_updated,
        "details": micro_report,
    }
    LOGGER.info(
        "Issue 3 micro-type consolidation: %s (edges_modified=%d)",
        "TROVATO" if micro_found else "NON TROVATO",
        micro_updated,
    )

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
    parser.add_argument("--env-file", default="kg_pipeline/.env")
    parser.add_argument("--database", default="")
    parser.add_argument("--relation-vocab", default="")
    parser.add_argument("--property-schema", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--log-file", default="")
    parser.add_argument("--relation-batch-size", type=int, default=120)
    parser.add_argument("--concept-batch-size", type=int, default=50)
    parser.add_argument("--enrich-batch-size", type=int, default=30)
    parser.add_argument("--reltype-patterns", type=int, default=6)
    parser.add_argument(
        "--fix",
        choices=["related-to", "region-artifacts", "micro-types", "aura-issues"],
        default="",
        help="Run a single cleanup task and skip the default pipeline",
    )
    parser.add_argument(
        "--dedup-label-mode",
        choices=["overlap", "exact", "any"],
        default="overlap",
        help="How strict label matching must be to merge duplicates",
    )
    parser.add_argument(
        "--concept-label-mode",
        choices=["replace", "add"],
        default="replace",
        help="Replace Concept label or add alongside it",
    )
    parser.add_argument(
        "--rewrite-inverses",
        action="store_true",
        help="Rewrite inverse relation pairs to a single direction",
    )
    args = parser.parse_args()

    log_path = None
    if args.log_file.strip():
        log_path = Path(args.log_file).expanduser()
    else:
        log_dir = Path("kg_pipeline") / "logs"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"neo4j_postprocess_{timestamp}.log"
    _setup_logging(log_path)

    load_dotenv(args.env_file, override=True)

    config = _load_yaml(Path(args.config))
    allowed_labels = [str(label) for label in config.get("ontology", {}).get("labels", [])]
    if "Concept" not in allowed_labels:
        allowed_labels.append("Concept")

    non_concept_labels = [label for label in allowed_labels if label != "Concept"] + ["Concept"]

    relation_vocab = _load_relation_vocab(args.relation_vocab)
    property_schema = _load_property_schema(args.property_schema)

    fix_mode = args.fix.strip()
    needs_llm = not fix_mode or fix_mode == "related-to"

    base_url = ""
    model_name = ""
    api_key = ""
    client: OpenAI | None = None
    if needs_llm:
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

            if fix_mode:
                if fix_mode == "aura-issues":
                    report["aura_issues"] = _run_aura_issues(
                        session=session,
                        relation_vocab=relation_vocab,
                        dry_run=args.dry_run,
                        batch_size=_RELATION_RECLASS_BATCH_SIZE,
                        apoc_available=apoc_available,
                    )
                    print(json.dumps(report, ensure_ascii=False, indent=2))
                    return

                if not apoc_available:
                    report["error"] = "APOC unavailable, cleanup tasks require APOC"
                    print(json.dumps(report, ensure_ascii=False, indent=2))
                    return

                if fix_mode == "related-to":
                    LOGGER.info("Fix: refine RELATED_TO relationships")
                    if client is None:
                        raise RuntimeError("LLM client required for RELATED_TO refinement")
                    report["related_to_refinement"] = _refine_related_to_relationships(
                        session=session,
                        canonical=relation_vocab,
                        client=client,
                        model_name=model_name,
                        dry_run=args.dry_run,
                    )
                elif fix_mode == "region-artifacts":
                    LOGGER.info("Fix: cleanup Region header artifacts")
                    report["region_artifact_cleanup"] = _cleanup_region_artifacts(
                        session=session,
                        dry_run=args.dry_run,
                    )
                elif fix_mode == "micro-types":
                    LOGGER.info("Fix: absorb micro relationship types")
                    report["micro_type_absorption"] = _absorb_micro_relation_types(
                        session=session,
                        rewrites=_MICRO_RELATION_REWRITES,
                        dry_run=args.dry_run,
                    )

                print(json.dumps(report, ensure_ascii=False, indent=2))
                return

            relation_items = _fetch_relation_types(session, max_patterns=args.reltype_patterns)
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

            if args.rewrite_inverses:
                LOGGER.info("Step 1b: rewrite inverse relations")
                report["step1b_rewrite_inverses"] = _rewrite_inverse_relationships(
                    session=session,
                    rewrites=_INVERSE_RELATION_REWRITES,
                    dry_run=args.dry_run,
                )
                step1b = report["step1b_rewrite_inverses"]
                LOGGER.info(
                    "Step 1b done: pairs=%d errors=%d",
                    len(step1b.get("pairs", [])),
                    len(step1b.get("errors", [])),
                )

            aura_report: dict[str, Any] = {}

            LOGGER.info("Aura cleanup step 1: invert PUBLISHED direction for Organization -> Document")
            aura_report["step1_published_direction_fix"] = _invert_published_direction(
                session=session,
                dry_run=args.dry_run,
            )
            step_a1 = aura_report["step1_published_direction_fix"]
            LOGGER.info(
                "Aura step 1 done: count=%d rewritten=%d errors=%d",
                int(step_a1.get("count", 0)),
                int(step_a1.get("rewritten", 0)),
                len(step_a1.get("errors", [])),
            )

            if not apoc_available:
                aura_report["step2_region_garbage_cleanup"] = {
                    "error": "APOC unavailable, cannot rewire/delete Region garbage nodes"
                }
            else:
                LOGGER.info("Aura cleanup step 2: redirect/delete named Region garbage nodes")
                aura_report["step2_region_garbage_cleanup"] = _cleanup_named_region_nodes(
                    session=session,
                    names=_AURA_REGION_GARBAGE_NAMES,
                    dry_run=args.dry_run,
                )
                step_a2 = aura_report["step2_region_garbage_cleanup"]
                LOGGER.info(
                    "Aura step 2 done: candidates=%d matched=%d deleted_nodes=%d rewired=%d deleted_rels=%d errors=%d",
                    int(step_a2.get("candidates", 0)),
                    int(step_a2.get("matched", 0)),
                    int(step_a2.get("deleted_nodes", 0)),
                    int(step_a2.get("rewired_relationships", 0)),
                    int(step_a2.get("deleted_relationships", 0)),
                    len(step_a2.get("errors", [])),
                )

            if not apoc_available:
                aura_report["step3_inverse_pairs"] = {
                    "error": "APOC unavailable, cannot rewrite inverse relationships"
                }
            else:
                LOGGER.info("Aura cleanup step 3: rewrite inverse pairs and rename INFLUENCES")
                inverse_report = _rewrite_inverse_relationships(
                    session=session,
                    rewrites=_AURA_INVERSE_REWRITES,
                    dry_run=args.dry_run,
                )
                rename_report = _absorb_micro_relation_types(
                    session=session,
                    rewrites=_AURA_RENAME_REWRITES,
                    dry_run=args.dry_run,
                )
                aura_report["step3_inverse_pairs"] = {
                    "inverse": inverse_report,
                    "rename": rename_report,
                }
                LOGGER.info(
                    "Aura step 3 done: inverse_pairs=%d inverse_errors=%d rename_pairs=%d rename_errors=%d",
                    len(inverse_report.get("pairs", [])),
                    len(inverse_report.get("errors", [])),
                    len(rename_report.get("pairs", [])),
                    len(rename_report.get("errors", [])),
                )

            if client is None:
                raise RuntimeError("LLM client required for Aura relationship reclassification")

            if not apoc_available:
                aura_report["step4_has_component_reclass"] = {
                    "error": "APOC unavailable, cannot reclassify HAS_COMPONENT relationships"
                }
            else:
                LOGGER.info("Aura cleanup step 4: reclassify anomalous HAS_COMPONENT relationships")
                aura_report["step4_has_component_reclass"] = _reclassify_has_component_anomalies(
                    session=session,
                    client=client,
                    model_name=model_name,
                    dry_run=args.dry_run,
                    batch_size=_RELATION_RECLASS_BATCH_SIZE,
                )
                step_a4 = aura_report["step4_has_component_reclass"]
                LOGGER.info(
                    "Aura step 4 done: candidates=%d updated=%d skipped=%d errors=%d",
                    int(step_a4.get("total_candidates", 0)),
                    int(step_a4.get("updated", 0)),
                    int(step_a4.get("skipped", 0)),
                    len(step_a4.get("errors", [])),
                )

            if not apoc_available:
                aura_report["step5_related_to_reclass"] = {
                    "error": "APOC unavailable, cannot reclassify RELATED_TO relationships"
                }
            else:
                LOGGER.info("Aura cleanup step 5: second-pass reclassify RELATED_TO relationships")
                aura_report["step5_related_to_reclass"] = _reclassify_related_to_second_pass(
                    session=session,
                    client=client,
                    model_name=model_name,
                    allowed=relation_vocab,
                    dry_run=args.dry_run,
                    batch_size=_RELATION_RECLASS_BATCH_SIZE,
                )
                step_a5 = aura_report["step5_related_to_reclass"]
                LOGGER.info(
                    "Aura step 5 done: candidates=%d updated=%d skipped=%d errors=%d",
                    int(step_a5.get("total_candidates", 0)),
                    int(step_a5.get("updated", 0)),
                    int(step_a5.get("skipped", 0)),
                    len(step_a5.get("errors", [])),
                )

            report["aura_cleanup"] = aura_report

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
                    label_mode=args.dedup_label_mode,
                )
                step2 = report["step2_dedup"]
                LOGGER.info(
                    "Step 2 done: groups=%d merged_nodes=%d skipped_incompatible=%d errors=%d",
                    int(step2.get("groups", 0)),
                    int(step2.get("merged_nodes", 0)),
                    int(step2.get("skipped_incompatible", 0)),
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
                label_mode=args.concept_label_mode,
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
