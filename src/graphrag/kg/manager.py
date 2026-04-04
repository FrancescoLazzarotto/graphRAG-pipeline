from __future__ import annotations

import os
import re
from typing import Any, Sequence

try:
    from langchain_neo4j import Neo4jGraph
except ImportError:  # pragma: no cover - compatibility fallback
    from langchain_community.graphs import Neo4jGraph

from graphrag.config import KGConfig
from graphrag.types import KGNode, KGTriple


class KnowledgeGraphManager:
    """High-level helper for import and query operations on Neo4j."""

    def __init__(self, config: KGConfig, graph: Neo4jGraph | None = None) -> None:
        self.config = config
        self.graph = graph or Neo4jGraph(
            url=config.url,
            username=config.username,
            password=config.password,
            database=config.database,
        )

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

    def run_query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return self.graph.query(cypher, params or {})

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
            {self._coalesce_name_expr('n')} AS text
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
            {self._coalesce_name_expr('s')} AS subject,
            coalesce(toString(properties(r)['predicate']), type(r)) AS predicate,
            elementId(o) AS object_id,
            {self._coalesce_name_expr('o')} AS object,
            labels(s) AS subject_labels,
            labels(o) AS object_labels,
            properties(s) AS subject_properties,
            properties(o) AS object_properties,
            properties(r) AS relationship_properties
        LIMIT $limit
        """
        return [self._row_to_triple(row) for row in self.run_query(cypher, params)]

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

        cypher = f"""
        MATCH (seed)
        WHERE {self._node_text_match_clause('seed', 'entity', exact=True)}
        MATCH p = (seed)-[*1..{hops}]-(other)
        UNWIND relationships(p) AS r
        WITH DISTINCT r
        WHERE {rel_filter}
        RETURN DISTINCT
            elementId(startNode(r)) AS subject_id,
            {self._coalesce_name_expr('startNode(r)')} AS subject,
            coalesce(toString(properties(r)['predicate']), type(r)) AS predicate,
            elementId(endNode(r)) AS object_id,
            {self._coalesce_name_expr('endNode(r)')} AS object,
            labels(startNode(r)) AS subject_labels,
            labels(endNode(r)) AS object_labels,
            properties(startNode(r)) AS subject_properties,
            properties(endNode(r)) AS object_properties,
            properties(r) AS relationship_properties
        LIMIT $limit
        """
        return [self._row_to_triple(row) for row in self.run_query(cypher, params)]

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

        cypher = f"""
        MATCH (seed)-[r]-(neighbor)
        WHERE {self._node_text_match_clause('seed', 'entity', exact=True)}
        {rel_clause}
        RETURN DISTINCT
            elementId(neighbor) AS node_id,
            labels(neighbor) AS labels,
            properties(neighbor) AS properties,
            {self._coalesce_name_expr('neighbor')} AS text
        LIMIT $limit
        """
        return [self._row_to_node(row) for row in self.run_query(cypher, params)]

    def get_entity_types(self, entity: str) -> list[str]:
        cypher = f"""
        MATCH (n)
        WHERE {self._node_text_match_clause('n', 'entity', exact=True)}
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
        cypher = f"""
        MATCH (a), (b)
        WHERE {self._node_text_match_clause('a', 'entity_a', exact=True)}
          AND {self._node_text_match_clause('b', 'entity_b', exact=True)}
        MATCH p = shortestPath((a)-[*1..{max_depth}]-(b))
        UNWIND relationships(p) AS r
        RETURN DISTINCT
            elementId(startNode(r)) AS subject_id,
            {self._coalesce_name_expr('startNode(r)')} AS subject,
                        coalesce(toString(properties(r)['predicate']), type(r)) AS predicate,
            elementId(endNode(r)) AS object_id,
            {self._coalesce_name_expr('endNode(r)')} AS object,
            labels(startNode(r)) AS subject_labels,
            labels(endNode(r)) AS object_labels,
            properties(startNode(r)) AS subject_properties,
            properties(endNode(r)) AS object_properties,
            properties(r) AS relationship_properties
        """
        rows = self.run_query(cypher, {"entity_a": entity_a, "entity_b": entity_b})
        return [self._row_to_triple(row) for row in rows]

    def triples_to_text(self, triples: Sequence[KGTriple]) -> str:
        return "\n".join(
            f"({triple.get('subject', '')}, {triple.get('predicate', '')}, {triple.get('object', '')})"
            for triple in triples
        )

    def nodes_to_text(self, nodes: Sequence[KGNode]) -> str:
        return "\n".join(node.get("text", "") for node in nodes if node.get("text"))

    def get_subgraph_context(self, entity: str, hops: int = 1, limit: int = 200) -> str:
        return self.triples_to_text(self.extract_subgraph(entity=entity, hops=hops, limit=limit))

    @staticmethod
    def _safe_identifier(value: str) -> str:
        cleaned = re.sub(r"[^0-9A-Za-z_]", "_", value.strip())
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        if not cleaned:
            return "RELATED_TO"
        if cleaned[0].isdigit():
            cleaned = f"_{cleaned}"
        return cleaned.upper()

    def _node_text_match_clause(self, alias: str, param_name: str, exact: bool = False) -> str:
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
