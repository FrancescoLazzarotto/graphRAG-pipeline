from __future__ import annotations

from typing import Any, Sequence

from graphrag.config import AgentConfig
from graphrag.kg.manager import KnowledgeGraphManager
from graphrag.types import KGNode, KGTriple


class KGRetriever:
    def __init__(self, kg_store: KnowledgeGraphManager, config: AgentConfig) -> None:
        self.kg_store = kg_store
        self.config = config

    def retrieve(self, query: str | None = None) -> dict[str, Any]:
        query_text = (query or self.config.query or self.config.entity or "").strip()
        entity = (self.config.entity or query_text or "").strip()

        nodes: list[KGNode] = []
        triples: list[KGTriple] = []
        neighbors: list[KGNode] = []
        subgraph: list[KGTriple] = []
        shortest_path: list[KGTriple] = []

        if self.config.include_nodes and query_text:
            nodes = self.kg_store.extract_nodes(
                text=query_text,
                labels=self.config.labels or None,
                limit=self.config.nodes_limit,
            )

        if self.config.include_triples and query_text:
            triples = self.kg_store.extract_triples(
                text=query_text,
                labels=self.config.labels or None,
                relationship_types=self.config.relationship_types or None,
                limit=self.config.triples_limit,
            )

        seed_entities = self._seed_entities(query_text=query_text, nodes=nodes, triples=triples)

        if self.config.include_neighbors and entity:
            neighbors = self.kg_store.get_neighbors(
                entity=entity,
                limit=self.config.neighbors_limit,
                relationship_types=self.config.relationship_types or None,
            )

        if self.config.include_subgraph and entity:
            subgraph = self.kg_store.extract_subgraph(
                entity=entity,
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
            "entity": entity or None,
            "seed_entities": seed_entities,
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
    ) -> list[str]:
        seeds: list[str] = []

        if self.config.entity:
            seeds.append(self.config.entity.strip())

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
