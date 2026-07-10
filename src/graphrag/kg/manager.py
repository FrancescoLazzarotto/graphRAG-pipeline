from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Sequence

try:
    from langchain_neo4j import Neo4jGraph
except ImportError:  # pragma: no cover - compatibility fallback
    from langchain_community.graphs import Neo4jGraph

try:  # pragma: no cover - depends on runtime dependency details
    from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError

    _RETRYABLE_NEO4J_EXCEPTIONS: tuple[type[BaseException], ...] = (
        SessionExpired,
        ServiceUnavailable,
        TransientError,
    )
except (
    Exception
):  # pragma: no cover - fallback if neo4j exception classes are unavailable
    _RETRYABLE_NEO4J_EXCEPTIONS = ()

from graphrag.config import KGConfig
from graphrag.types import KGNode, KGTriple

logger = logging.getLogger("graphrag")


class KnowledgeGraphManager:
    """High-level helper for import and query operations on Neo4j."""

    def __init__(self, config: KGConfig, graph: Neo4jGraph | None = None) -> None:
        self.config = config
        self.graph = graph or self._build_graph()

        retry_attempts_raw = os.getenv("GRAPHRAG_NEO4J_QUERY_RETRIES", "3").strip()
        retry_backoff_raw = os.getenv(
            "GRAPHRAG_NEO4J_QUERY_RETRY_BACKOFF_SEC", "1.0"
        ).strip()

        try:
            self.query_retry_attempts = max(1, int(retry_attempts_raw))
        except ValueError:
            self.query_retry_attempts = 3

        try:
            self.query_retry_backoff_sec = max(0.0, float(retry_backoff_raw))
        except ValueError:
            self.query_retry_backoff_sec = 1.0

        self.fulltext_index = os.getenv(
            "GRAPHRAG_FULLTEXT_INDEX", "node_search"
        ).strip()
        # None = not probed yet; False = index missing, use the CONTAINS scan.
        self._fulltext_available: bool | None = None

    def _build_graph(self) -> Neo4jGraph:
        return Neo4jGraph(
            url=self.config.url,
            username=self.config.username,
            password=self.config.password,
            database=self.config.database,
        )

    def _reconnect(self) -> None:
        self.graph = self._build_graph()

    @staticmethod
    def _is_retryable_query_error(exc: BaseException) -> bool:
        if _RETRYABLE_NEO4J_EXCEPTIONS and isinstance(exc, _RETRYABLE_NEO4J_EXCEPTIONS):
            return True

        text = f"{type(exc).__name__}: {exc}".lower()
        markers = (
            "sessionexpired",
            "serviceunavailable",
            "transienterror",
            "defunct connection",
            "connection reset",
            "connection aborted",
            "connection was closed",
            "failed to read from defunct connection",
            "failed to read",
            "network",
            "timed out",
        )
        return any(marker in text for marker in markers)

    @classmethod
    def from_env(
        cls,
        url_env: str = "NEO4J_URL",
        username_env: str = "NEO4J_USERNAME",
        password_env: str = "NEO4J_PASSWORD",
        database_env: str = "NEO4J_DATABASE",
    ) -> "KnowledgeGraphManager":
        return cls(
            KGConfig(
                url=os.environ[url_env],
                username=os.environ[username_env],
                password=os.environ[password_env],
                database=os.environ.get(database_env),
            )
        )

    @property
    def schema(self) -> str:
        return getattr(self.graph, "schema", "")

    def refresh_schema(self) -> str:
        self.graph.refresh_schema()
        return self.schema

    def run_query(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        payload = params or {}
        max_attempts = max(1, self.query_retry_attempts)

        for attempt in range(1, max_attempts + 1):
            try:
                return self.graph.query(cypher, payload)
            except Exception as exc:
                retryable = self._is_retryable_query_error(exc)
                if not retryable or attempt >= max_attempts:
                    raise

                backoff_sec = self.query_retry_backoff_sec * attempt
                logger.warning(
                    "Neo4j transient query failure. retry=%d/%d backoff_sec=%.2f error=%s",
                    attempt,
                    max_attempts,
                    backoff_sec,
                    exc,
                )

                try:
                    self._reconnect()
                except Exception as reconnect_exc:  # pragma: no cover - depends on runtime network state
                    logger.warning("Neo4j reconnect attempt failed: %s", reconnect_exc)

                if backoff_sec > 0:
                    time.sleep(backoff_sec)

        raise RuntimeError("unreachable: retry loop either returns or raises")

    def clear(self) -> None:
        self.run_query("MATCH (n) DETACH DELETE n")

    def import_triples(
        self,
        triples: Sequence[dict[str, Any]],
        subject_label: str = "Entity",
        object_label: str = "Entity",
        relationship_type: str = "RELATED_TO",
    ) -> int:
        if not triples:
            return 0

        subject_label = self._safe_identifier(subject_label)
        object_label = self._safe_identifier(object_label)
        relationship_type = self._safe_identifier(relationship_type)

        cypher = f"""
        UNWIND $triples AS triple
        MERGE (s:{subject_label} {{name: triple.subject}})
        MERGE (o:{object_label} {{name: triple.object}})
        MERGE (s)-[r:{relationship_type}]->(o)
        SET r.predicate = triple.predicate
        SET r += coalesce(triple.relationship_properties, {{}})
        RETURN count(r) AS relationships_written
        """
        rows = self.run_query(cypher, {"triples": list(triples)})
        return int(rows[0].get("relationships_written", 0)) if rows else 0

    def extract_nodes(
        self,
        text: str | None = None,
        labels: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[KGNode]:
        limit = limit or self.config.default_limit
        where_clauses: list[str] = []
        params: dict[str, Any] = {"limit": limit}

        if labels:
            where_clauses.append("any(label IN labels(n) WHERE label IN $labels)")
            params["labels"] = list(labels)

        if text:
            where_clauses.append(self._node_text_match_clause("n", "text"))
            params["text"] = text

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        cypher = f"""
        MATCH (n)
        {where_sql}
        RETURN DISTINCT
            elementId(n) AS node_id,
            labels(n) AS labels,
            properties(n) AS properties,
            {self._coalesce_name_expr("n")} AS text
        LIMIT $limit
        """
        return [self._row_to_node(row) for row in self.run_query(cypher, params)]

    def extract_triples(
        self,
        text: str | None = None,
        labels: Sequence[str] | None = None,
        relationship_types: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[KGTriple]:
        limit = limit or self.config.default_limit
        where_clauses: list[str] = []
        params: dict[str, Any] = {"limit": limit}

        if labels:
            where_clauses.append(
                "(any(label IN labels(s) WHERE label IN $labels) OR any(label IN labels(o) WHERE label IN $labels))"
            )
            params["labels"] = list(labels)

        if relationship_types:
            where_clauses.append("type(r) IN $relationship_types")
            params["relationship_types"] = list(relationship_types)

        if text:
            where_clauses.append(
                f"({self._node_text_match_clause('s', 'text')} OR {self._node_text_match_clause('o', 'text')})"
            )
            params["text"] = text

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        cypher = f"""
        MATCH (s)-[r]->(o)
        {where_sql}
        RETURN DISTINCT
            elementId(s) AS subject_id,
            {self._coalesce_name_expr("s")} AS subject,
            coalesce(toString(properties(r)['predicate']), type(r)) AS predicate,
            elementId(o) AS object_id,
            {self._coalesce_name_expr("o")} AS object,
            labels(s) AS subject_labels,
            labels(o) AS object_labels,
            properties(s) AS subject_properties,
            properties(o) AS object_properties,
            properties(r) AS relationship_properties
        LIMIT $limit
        """
        return [self._row_to_triple(row) for row in self.run_query(cypher, params)]

    _LUCENE_SPECIAL_RE = re.compile(r'(&&|\|\||[+\-!(){}\[\]^"~*?:\\/])')
    _FULLTEXT_MISSING_MARKERS = (
        "no such fulltext",
        "there is no such fulltext",
        "no such index",
        "not found",
        "db.index.fulltext.querynodes",
    )

    @classmethod
    def _lucene_query(cls, terms: Sequence[str]) -> str:
        """Build a Lucene OR-query from search terms (phrases quoted)."""
        parts: list[str] = []
        for term in terms:
            cleaned = str(term or "").strip()
            if not cleaned:
                continue
            escaped = cls._LUCENE_SPECIAL_RE.sub(r"\\\1", cleaned)
            parts.append(f'"{escaped}"' if " " in escaped else escaped)
        return " OR ".join(parts)

    def _handle_fulltext_error(self, exc: Exception) -> bool:
        """Return True (and disable full-text) when the index is unavailable."""
        text = f"{type(exc).__name__}: {exc}".lower()
        if any(marker in text for marker in self._FULLTEXT_MISSING_MARKERS):
            logger.warning(
                "Full-text index %r unavailable (%s) — falling back to the "
                "CONTAINS scan for this session. Run scripts/kg_search_index.py "
                "to create the index.",
                self.fulltext_index,
                exc,
            )
            self._fulltext_available = False
            return True
        return False

    def fulltext_search_nodes(
        self,
        terms: Sequence[str],
        labels: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[KGNode] | None:
        """Match nodes for all ``terms`` in one indexed query.

        Args:
            terms: Search terms; combined into a single Lucene OR-query.
            labels: Optional label whitelist applied after the index lookup.
            limit: Maximum nodes returned (best score first).

        Returns:
            Nodes ordered by Lucene score, or ``None`` when the full-text index
            is unavailable and the caller must fall back to the CONTAINS scan.
        """
        if self._fulltext_available is False:
            return None
        lucene = self._lucene_query(terms)
        if not lucene:
            return []
        limit = limit or self.config.default_limit
        params: dict[str, Any] = {
            "index": self.fulltext_index,
            "q": lucene,
            "limit": limit,
        }
        label_filter = ""
        if labels:
            label_filter = "WHERE any(label IN labels(node) WHERE label IN $labels)"
            params["labels"] = list(labels)
        cypher = f"""
        CALL db.index.fulltext.queryNodes($index, $q, {{limit: $limit}})
        YIELD node, score
        {label_filter}
        RETURN
            elementId(node) AS node_id,
            labels(node) AS labels,
            properties(node) AS properties,
            {self._coalesce_name_expr("node")} AS text
        """
        try:
            rows = self.run_query(cypher, params)
        except Exception as exc:  # noqa: BLE001 - narrowed by marker check
            if self._handle_fulltext_error(exc):
                return None
            raise
        self._fulltext_available = True
        return [self._row_to_node(row) for row in rows]

    def fulltext_search_triples(
        self,
        terms: Sequence[str],
        labels: Sequence[str] | None = None,
        relationship_types: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[KGTriple] | None:
        """Match triples around index-matched nodes in one query.

        Args:
            terms: Search terms; combined into a single Lucene OR-query.
            labels: Optional label whitelist for either endpoint.
            relationship_types: Optional relationship-type whitelist.
            limit: Maximum triples returned (best seed score first).

        Returns:
            Triples ordered by the matched endpoint's Lucene score, or ``None``
            when the full-text index is unavailable.
        """
        if self._fulltext_available is False:
            return None
        lucene = self._lucene_query(terms)
        if not lucene:
            return []
        limit = limit or self.config.default_limit
        params: dict[str, Any] = {
            "index": self.fulltext_index,
            "q": lucene,
            "limit": limit,
        }
        filters: list[str] = []
        if labels:
            filters.append(
                "(any(label IN labels(startNode(r)) WHERE label IN $labels) "
                "OR any(label IN labels(endNode(r)) WHERE label IN $labels))"
            )
            params["labels"] = list(labels)
        if relationship_types:
            filters.append("type(r) IN $relationship_types")
            params["relationship_types"] = list(relationship_types)
        where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""
        cypher = f"""
        CALL db.index.fulltext.queryNodes($index, $q, {{limit: $limit}})
        YIELD node, score
        MATCH (node)-[r]-()
        WITH r, max(score) AS seed_score
        {where_sql}
        RETURN
            elementId(startNode(r)) AS subject_id,
            {self._coalesce_name_expr("startNode(r)")} AS subject,
            coalesce(toString(properties(r)['predicate']), type(r)) AS predicate,
            elementId(endNode(r)) AS object_id,
            {self._coalesce_name_expr("endNode(r)")} AS object,
            labels(startNode(r)) AS subject_labels,
            labels(endNode(r)) AS object_labels,
            properties(startNode(r)) AS subject_properties,
            properties(endNode(r)) AS object_properties,
            properties(r) AS relationship_properties
        ORDER BY seed_score DESC
        LIMIT $limit
        """
        try:
            rows = self.run_query(cypher, params)
        except Exception as exc:  # noqa: BLE001 - narrowed by marker check
            if self._handle_fulltext_error(exc):
                return None
            raise
        self._fulltext_available = True
        return [self._row_to_triple(row) for row in rows]

    def extract_subgraph(
        self,
        entity: str,
        hops: int = 1,
        limit: int = 200,
        relationship_types: Sequence[str] | None = None,
    ) -> list[KGTriple]:
        hops = max(1, int(hops))
        params: dict[str, Any] = {"entity": entity, "limit": limit}

        rel_filter = "true"
        if relationship_types:
            rel_filter = "type(r) IN $relationship_types"
            params["relationship_types"] = list(relationship_types)

        cypher_exact = f"""
        MATCH (seed)
        WHERE {self._node_text_match_clause("seed", "entity", exact=True)}
        MATCH p = (seed)-[*1..{hops}]-(other)
        UNWIND relationships(p) AS r
        WITH DISTINCT r
        WHERE {rel_filter}
        RETURN DISTINCT
            elementId(startNode(r)) AS subject_id,
            {self._coalesce_name_expr("startNode(r)")} AS subject,
            coalesce(toString(properties(r)['predicate']), type(r)) AS predicate,
            elementId(endNode(r)) AS object_id,
            {self._coalesce_name_expr("endNode(r)")} AS object,
            labels(startNode(r)) AS subject_labels,
            labels(endNode(r)) AS object_labels,
            properties(startNode(r)) AS subject_properties,
            properties(endNode(r)) AS object_properties,
            properties(r) AS relationship_properties
        LIMIT $limit
        """

        rows = self.run_query(cypher_exact, params)
        if not rows:
            # fallback to a looser text match when exact matching returns nothing
            cypher_fallback = f"""
            MATCH (seed)
            WHERE {self._node_text_match_clause("seed", "entity", exact=False)}
            MATCH p = (seed)-[*1..{hops}]-(other)
            UNWIND relationships(p) AS r
            WITH DISTINCT r
            WHERE {rel_filter}
            RETURN DISTINCT
                elementId(startNode(r)) AS subject_id,
                {self._coalesce_name_expr("startNode(r)")} AS subject,
                coalesce(toString(properties(r)['predicate']), type(r)) AS predicate,
                elementId(endNode(r)) AS object_id,
                {self._coalesce_name_expr("endNode(r)")} AS object,
                labels(startNode(r)) AS subject_labels,
                labels(endNode(r)) AS object_labels,
                properties(startNode(r)) AS subject_properties,
                properties(endNode(r)) AS object_properties,
                properties(r) AS relationship_properties
            LIMIT $limit
            """
            rows = self.run_query(cypher_fallback, params)

        return [self._row_to_triple(row) for row in rows]

    def get_neighbors(
        self,
        entity: str,
        limit: int = 25,
        relationship_types: Sequence[str] | None = None,
    ) -> list[KGNode]:
        params: dict[str, Any] = {"entity": entity, "limit": limit}
        rel_clause = ""
        if relationship_types:
            rel_clause = "AND type(r) IN $relationship_types"
            params["relationship_types"] = list(relationship_types)
        cypher_exact = f"""
        MATCH (seed)-[r]-(neighbor)
        WHERE {self._node_text_match_clause("seed", "entity", exact=True)}
        {rel_clause}
        RETURN DISTINCT
            elementId(neighbor) AS node_id,
            labels(neighbor) AS labels,
            properties(neighbor) AS properties,
            {self._coalesce_name_expr("neighbor")} AS text
        LIMIT $limit
        """
        rows = self.run_query(cypher_exact, params)
        if not rows:
            cypher_fallback = f"""
            MATCH (seed)-[r]-(neighbor)
            WHERE {self._node_text_match_clause("seed", "entity", exact=False)}
            {rel_clause}
            RETURN DISTINCT
                elementId(neighbor) AS node_id,
                labels(neighbor) AS labels,
                properties(neighbor) AS properties,
                {self._coalesce_name_expr("neighbor")} AS text
            LIMIT $limit
            """
            rows = self.run_query(cypher_fallback, params)

        return [self._row_to_node(row) for row in rows]

    def get_entity_types(self, entity: str) -> list[str]:
        cypher = f"""
        MATCH (n)
        WHERE {self._node_text_match_clause("n", "entity", exact=True)}
        RETURN DISTINCT labels(n) AS labels
        LIMIT 1
        """
        rows = self.run_query(cypher, {"entity": entity})
        if not rows:
            return []
        return list(rows[0].get("labels", []))

    def get_shortest_path(
        self,
        entity_a: str,
        entity_b: str,
        max_depth: int = 6,
    ) -> list[KGTriple]:
        # max_depth is interpolated into a variable-length pattern, so force it
        # to a safe positive integer (defence-in-depth; mirrors extract_subgraph).
        max_depth = max(1, int(max_depth))
        cypher_exact = f"""
        MATCH (a)
        WHERE {self._node_text_match_clause("a", "entity_a", exact=True)}
        WITH DISTINCT a
        ORDER BY size({self._coalesce_name_expr("a")}) ASC
        LIMIT 8
        MATCH (b)
        WHERE {self._node_text_match_clause("b", "entity_b", exact=True)}
        WITH DISTINCT a, b
        ORDER BY size({self._coalesce_name_expr("b")}) ASC
        LIMIT 16
        MATCH p = shortestPath((a)-[*1..{max_depth}]-(b))
        UNWIND relationships(p) AS r
        RETURN DISTINCT
            elementId(startNode(r)) AS subject_id,
            {self._coalesce_name_expr("startNode(r)")} AS subject,
                        coalesce(toString(properties(r)['predicate']), type(r)) AS predicate,
            elementId(endNode(r)) AS object_id,
            {self._coalesce_name_expr("endNode(r)")} AS object,
            labels(startNode(r)) AS subject_labels,
            labels(endNode(r)) AS object_labels,
            properties(startNode(r)) AS subject_properties,
            properties(endNode(r)) AS object_properties,
            properties(r) AS relationship_properties
        """
        try:
            rows = self.run_query(cypher_exact, {"entity_a": entity_a, "entity_b": entity_b})
        except Exception as exc:
            # If the shortestPath call fails (e.g. same-node cartesian product),
            # return an empty result instead of propagating the DB error — but
            # log it, so "no path" stays distinguishable from "query failed".
            logger.warning(
                "shortestPath exact query failed for (%r, %r): %s",
                entity_a,
                entity_b,
                exc,
            )
            return []

        if not rows:
            # The CONTAINS fallback can match hundreds of nodes per side on
            # generic terms ("food"): unbounded, the a×b cartesian product of
            # shortestPath calls takes tens of seconds and floods the context
            # with deep-path noise. Shortest names first ≈ most canonical.
            cypher_fallback = f"""
            MATCH (a)
            WHERE {self._node_text_match_clause("a", "entity_a", exact=False)}
            WITH DISTINCT a
            ORDER BY size({self._coalesce_name_expr("a")}) ASC
            LIMIT 8
            MATCH (b)
            WHERE {self._node_text_match_clause("b", "entity_b", exact=False)}
            WITH DISTINCT a, b
            ORDER BY size({self._coalesce_name_expr("b")}) ASC
            LIMIT 16
            MATCH p = shortestPath((a)-[*1..{max_depth}]-(b))
            UNWIND relationships(p) AS r
            RETURN DISTINCT
                elementId(startNode(r)) AS subject_id,
                {self._coalesce_name_expr("startNode(r)")} AS subject,
                            coalesce(toString(properties(r)['predicate']), type(r)) AS predicate,
                elementId(endNode(r)) AS object_id,
                {self._coalesce_name_expr("endNode(r)")} AS object,
                labels(startNode(r)) AS subject_labels,
                labels(endNode(r)) AS object_labels,
                properties(startNode(r)) AS subject_properties,
                properties(endNode(r)) AS object_properties,
                properties(r) AS relationship_properties
            """
            try:
                rows = self.run_query(cypher_fallback, {"entity_a": entity_a, "entity_b": entity_b})
            except Exception as exc:
                logger.warning(
                    "shortestPath fallback query failed for (%r, %r): %s",
                    entity_a,
                    entity_b,
                    exc,
                )
                return []

        return [self._row_to_triple(row) for row in rows]

    def triples_to_text(self, triples: Sequence[KGTriple]) -> str:
        return "\n".join(
            f"({triple.get('subject', '')}, {triple.get('predicate', '')}, {triple.get('object', '')})"
            for triple in triples
        )

    def nodes_to_text(self, nodes: Sequence[KGNode]) -> str:
        return "\n".join(node.get("text", "") for node in nodes if node.get("text"))

    def get_subgraph_context(self, entity: str, hops: int = 1, limit: int = 200) -> str:
        return self.triples_to_text(
            self.extract_subgraph(entity=entity, hops=hops, limit=limit)
        )

    @staticmethod
    def _safe_identifier(value: str) -> str:
        cleaned = re.sub(r"[^0-9A-Za-z_]", "_", value.strip())
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        if not cleaned:
            return "RELATED_TO"
        if cleaned[0].isdigit():
            cleaned = f"_{cleaned}"
        return cleaned.upper()

    def _node_text_match_clause(
        self, alias: str, param_name: str, exact: bool = False
    ) -> str:
        operator = "=" if exact else "CONTAINS"
        properties_expr = f"properties({alias})"
        comparisons = [
            f"toLower(coalesce(toString({properties_expr}['{prop}']), '')) {operator} toLower(${param_name})"
            for prop in self.config.node_name_properties
        ]
        comparisons.append(f"elementId({alias}) {operator} ${param_name}")
        return "(" + " OR ".join(comparisons) + ")"

    def _coalesce_name_expr(self, alias: str) -> str:
        properties_expr = f"properties({alias})"
        props = ", ".join(
            f"toString({properties_expr}['{prop}'])"
            for prop in self.config.node_name_properties
        )
        return f"coalesce({props}, elementId({alias}))"

    @staticmethod
    def _row_to_node(row: dict[str, Any]) -> KGNode:
        return {
            "node_id": str(row.get("node_id", "")),
            "labels": list(row.get("labels", [])),
            "properties": dict(row.get("properties", {})),
            "text": str(row.get("text", "")),
        }

    @staticmethod
    def _row_to_triple(row: dict[str, Any]) -> KGTriple:
        return {
            "subject_id": str(row.get("subject_id", "")),
            "subject": str(row.get("subject", "")),
            "predicate": str(row.get("predicate", "")),
            "object_id": str(row.get("object_id", "")),
            "object": str(row.get("object", "")),
            "subject_labels": list(row.get("subject_labels", [])),
            "object_labels": list(row.get("object_labels", [])),
            "subject_properties": dict(row.get("subject_properties", {})),
            "object_properties": dict(row.get("object_properties", {})),
            "relationship_properties": dict(row.get("relationship_properties", {})),
        }
