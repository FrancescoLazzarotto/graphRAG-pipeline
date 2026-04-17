from __future__ import annotations

import re
from typing import Any, Sequence

from graphrag.config import AgentConfig
from graphrag.kg.manager import KnowledgeGraphManager
from graphrag.types import KGNode, KGTriple

_QUOTED_ENTITY_RE = re.compile(r"[\"']([^\"']{2,})[\"']")
_TITLE_ENTITY_RE = re.compile(r"\b(?:[A-Z][\w'-]*)(?:\s+[A-Z][\w'-]*)+\b")
_SINGLE_TOKEN_ENTITY_RE = re.compile(r"\b[A-Z][\w'-]{2,}\b")
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)

_QUESTION_STOPWORDS = {
    "chi",
    "come",
    "cosa",
    "quando",
    "dove",
    "quale",
    "quali",
    "perche",
    "what",
    "when",
    "where",
    "which",
    "who",
    "how",
    "why",
    "does",
    "is",
    "are",
    "did",
    "the",
    "a",
    "an",
}


class KGRetriever:
    def __init__(self, kg_store: KnowledgeGraphManager, config: AgentConfig) -> None:
        self.kg_store = kg_store
        self.config = config

    def retrieve(self, query: str | None = None) -> dict[str, Any]:
        query_text = (query or self.config.query or self.config.entity or "").strip()
        configured_entity = (self.config.entity or "").strip()
        search_terms = self._build_search_terms(query_text=query_text, configured_entity=configured_entity)

        nodes: list[KGNode] = []
        triples: list[KGTriple] = []
        neighbors: list[KGNode] = []
        subgraph: list[KGTriple] = []
        shortest_path: list[KGTriple] = []

        if self.config.include_nodes and search_terms:
            nodes = self._collect_nodes(search_terms=search_terms, limit=self.config.nodes_limit)

        if self.config.include_triples and search_terms:
            triples = self._collect_triples(search_terms=search_terms, limit=self.config.triples_limit)

        seed_entities = self._seed_entities(
            query_text=query_text,
            nodes=nodes,
            triples=triples,
            search_terms=search_terms,
        )
        resolved_entity = configured_entity or (seed_entities[0] if seed_entities else "")

        if self.config.include_neighbors and resolved_entity:
            neighbors = self.kg_store.get_neighbors(
                entity=resolved_entity,
                limit=self.config.neighbors_limit,
                relationship_types=self.config.relationship_types or None,
            )

        if self.config.include_subgraph and resolved_entity:
            subgraph = self.kg_store.extract_subgraph(
                entity=resolved_entity,
                hops=self.config.hops,
                limit=self.config.subgraph_limit,
                relationship_types=self.config.relationship_types or None,
            )

        if self.config.include_shortest_path:
            entity_a = self.config.entity_a or (seed_entities[0] if len(seed_entities) > 0 else None)
            entity_b = self.config.entity_b or (seed_entities[1] if len(seed_entities) > 1 else None)
            if entity_a and entity_b and entity_a != entity_b:
                shortest_path = self.kg_store.get_shortest_path(
                    entity_a=entity_a,
                    entity_b=entity_b,
                    max_depth=self.config.max_depth,
                )

        context_sections = self._build_context_sections(
            query_text=query_text,
            nodes=nodes,
            triples=triples,
            neighbors=neighbors,
            subgraph=subgraph,
            shortest_path=shortest_path,
        )

        return {
            "query": query_text,
            "entity": resolved_entity or None,
            "seed_entities": seed_entities,
            "search_terms": search_terms,
            "nodes": nodes,
            "triples": triples,
            "neighbors": neighbors,
            "subgraph": subgraph,
            "shortest_path": shortest_path,
            "context_sections": context_sections,
            "context_text": "\n\n".join(section for section in context_sections if section),
        }

    def multi_hop(
        self,
        entity: str | None = None,
        hops: int | None = None,
        limit: int | None = None,
        relationship_types: Sequence[str] | None = None,
    ) -> list[KGTriple]:
        target_entity = (entity or self.config.entity or self.config.query or "").strip()
        if not target_entity:
            return []
        return self.kg_store.extract_subgraph(
            entity=target_entity,
            hops=hops if hops is not None else self.config.hops,
            limit=limit if limit is not None else self.config.subgraph_limit,
            relationship_types=relationship_types or self.config.relationship_types or None,
        )

    def retrieve_context(self, query: str | None = None) -> str:
        return self.retrieve(query=query)["context_text"]

    def _seed_entities(
        self,
        query_text: str,
        nodes: Sequence[KGNode],
        triples: Sequence[KGTriple],
        search_terms: Sequence[str],
    ) -> list[str]:
        seeds: list[str] = []

        if self.config.entity:
            seeds.append(self.config.entity.strip())

        seeds.extend(search_terms)

        for node in nodes:
            text = node.get("text", "").strip()
            if text:
                seeds.append(text)

        for triple in triples:
            for candidate in (triple.get("subject", ""), triple.get("object", "")):
                candidate = candidate.strip()
                if candidate:
                    seeds.append(candidate)

        if not seeds and query_text:
            seeds.append(query_text)

        return self._unique_values(seeds)

    def _build_search_terms(self, query_text: str, configured_entity: str) -> list[str]:
        terms: list[str] = []

        if configured_entity:
            terms.append(configured_entity)

        if query_text:
            terms.extend(self._extract_entity_candidates(query_text))

            # Only use full question text when short, otherwise entity candidates are more precise.
            if len(_TOKEN_RE.findall(query_text)) <= 8:
                terms.append(query_text)

        if not terms and query_text:
            terms.append(query_text)

        return self._unique_values(terms)

    def _extract_entity_candidates(self, text: str) -> list[str]:
        candidates: list[str] = []

        for match in _QUOTED_ENTITY_RE.findall(text):
            value = match.strip()
            if value:
                candidates.append(value)

        for phrase in _TITLE_ENTITY_RE.findall(text):
            value = phrase.strip()
            if value:
                candidates.append(value)

        for token in _SINGLE_TOKEN_ENTITY_RE.findall(text):
            if token.lower() not in _QUESTION_STOPWORDS:
                candidates.append(token)

        return self._unique_values(candidates)

    def _collect_nodes(self, search_terms: Sequence[str], limit: int) -> list[KGNode]:
        if limit <= 0:
            return []

        collected: list[KGNode] = []
        seen: set[str] = set()

        for term in search_terms:
            rows = self.kg_store.extract_nodes(
                text=term,
                labels=self.config.labels or None,
                limit=limit,
            )
            for row in rows:
                key = self._node_key(row)
                if key in seen:
                    continue
                seen.add(key)
                collected.append(row)
                if len(collected) >= limit:
                    return collected

        return collected

    def _collect_triples(self, search_terms: Sequence[str], limit: int) -> list[KGTriple]:
        if limit <= 0:
            return []

        collected: list[KGTriple] = []
        seen: set[tuple[str, str, str]] = set()

        for term in search_terms:
            rows = self.kg_store.extract_triples(
                text=term,
                labels=self.config.labels or None,
                relationship_types=self.config.relationship_types or None,
                limit=limit,
            )
            for row in rows:
                key = self._triple_key(row)
                if key in seen:
                    continue
                seen.add(key)
                collected.append(row)
                if len(collected) >= limit:
                    return collected

        return collected

    def _build_context_sections(
        self,
        query_text: str,
        nodes: Sequence[KGNode],
        triples: Sequence[KGTriple],
        neighbors: Sequence[KGNode],
        subgraph: Sequence[KGTriple],
        shortest_path: Sequence[KGTriple],
    ) -> list[str]:
        sections: list[str] = []

        if query_text:
            sections.append(f"Query: {query_text}")

        if nodes:
            sections.append("Matched nodes:\n" + self.kg_store.nodes_to_text(nodes))

        if triples:
            sections.append("Matched triples:\n" + self.kg_store.triples_to_text(triples))

        if neighbors:
            sections.append("Neighbors:\n" + self.kg_store.nodes_to_text(neighbors))

        if subgraph:
            sections.append("Subgraph:\n" + self.kg_store.triples_to_text(subgraph))

        if shortest_path:
            sections.append("Shortest path:\n" + self.kg_store.triples_to_text(shortest_path))

        return sections

    @staticmethod
    def _unique_values(values: Sequence[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = value.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(normalized)
        return unique

    @staticmethod
    def _node_key(node: KGNode) -> str:
        node_id = str(node.get("node_id", "")).strip()
        if node_id:
            return f"id:{node_id}"
        text = str(node.get("text", "")).strip().lower()
        return f"text:{text}"

    @staticmethod
    def _triple_key(triple: KGTriple) -> tuple[str, str, str]:
        subject_id = str(triple.get("subject_id", "")).strip()
        object_id = str(triple.get("object_id", "")).strip()
        predicate = str(triple.get("predicate", "")).strip().lower()

        if subject_id and object_id:
            return (f"id:{subject_id}", predicate, f"id:{object_id}")

        subject = str(triple.get("subject", "")).strip().lower()
        obj = str(triple.get("object", "")).strip().lower()
        return (subject, predicate, obj)


