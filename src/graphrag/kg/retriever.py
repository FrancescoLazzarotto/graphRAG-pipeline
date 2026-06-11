from __future__ import annotations

import math
import re
from typing import Any, Sequence

from graphrag.config import AgentConfig
from graphrag.kg.manager import KnowledgeGraphManager
from graphrag.text_rag.pipeline import StandardTextRAGPipeline
from graphrag.types import KGNode, KGTriple

_QUOTED_ENTITY_RE = re.compile(r"[\"']([^\"']{2,})[\"']")
_TITLE_ENTITY_RE = re.compile(r"\b(?:[A-Z][\w'-]*)(?:\s+[A-Z][\w'-]*)+\b")
_SINGLE_TOKEN_ENTITY_RE = re.compile(r"\b[A-Z][\w'-]{2,}\b")
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
# Matches years (1900-2099) and quantities with explicit units so factual/numerical
# questions can seed KG lookup on DataValue nodes.
_NUMERIC_TERM_RE = re.compile(
    r"\b(?:(?:19|20)\d{2}|\d+(?:[.,]\d+)?\s*(?:%|kg|Mt|Gt|million|billion|tonnes|°C))(?!\w)",
    re.IGNORECASE,
)

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

_PLACEHOLDER_ENTITIES = {
    "entita a",
    "entità a",
    "entity a",
}


class KGRetriever:
    def __init__(
        self,
        kg_store: KnowledgeGraphManager,
        config: AgentConfig,
        text_pipeline: StandardTextRAGPipeline | None = None,
    ) -> None:
        self.kg_store = kg_store
        self.config = config
        self.text_pipeline = text_pipeline

    def retrieve(self, query: str | None = None) -> dict[str, Any]:
        query_text = (query or self.config.query or self.config.entity or "").strip()
        configured_entity = self._sanitize_entity_name(self.config.entity or "")
        search_terms = self._build_search_terms(
            query_text=query_text, configured_entity=configured_entity
        )

        nodes: list[KGNode] = []
        triples: list[KGTriple] = []
        neighbors: list[KGNode] = []
        subgraph: list[KGTriple] = []
        shortest_path: list[KGTriple] = []

        if self.config.include_nodes and search_terms:
            nodes = self._collect_nodes(
                search_terms=search_terms, limit=self.config.nodes_limit
            )

        if self.config.include_triples and search_terms:
            triples = self._collect_triples(
                search_terms=search_terms, limit=self.config.triples_limit
            )

        seed_entities = self._seed_entities(
            query_text=query_text,
            nodes=nodes,
            triples=triples,
            search_terms=search_terms,
        )
        resolved_entity = configured_entity or (seed_entities[0] if seed_entities else "")
        resolved_entity = self._sanitize_entity_name(resolved_entity)

        if self.config.include_neighbors and resolved_entity:
            neighbors = self.kg_store.get_neighbors(
                entity=resolved_entity,
                limit=self.config.neighbors_limit,
                relationship_types=self.config.relationship_types or None,
            )

        if self.config.include_subgraph and resolved_entity:
            if self.config.adaptive_hops:
                subgraph = self._adaptive_subgraph(
                    entity=resolved_entity,
                    hops=self.config.hops,
                    limit=self.config.subgraph_limit,
                    relationship_types=self.config.relationship_types or None,
                )
            else:
                subgraph = self.kg_store.extract_subgraph(
                    entity=resolved_entity,
                    hops=self.config.hops,
                    limit=self.config.subgraph_limit,
                    relationship_types=self.config.relationship_types or None,
                )

        if self.config.include_shortest_path:
            entity_a = self._sanitize_entity_name(self.config.entity_a or "") or (
                seed_entities[0] if len(seed_entities) > 0 else None
            )
            entity_b = self._sanitize_entity_name(self.config.entity_b or "") or (
                seed_entities[1] if len(seed_entities) > 1 else None
            )
            if entity_a and entity_b and entity_a != entity_b:
                shortest_path = self.kg_store.get_shortest_path(
                    entity_a=entity_a,
                    entity_b=entity_b,
                    max_depth=self.config.max_depth,
                )

        if self.config.rank_triples:
            triples = self._rank_triples(triples, query_text)
            subgraph = self._rank_triples(subgraph, query_text)

        text_chunks: list[str] = []
        if self.config.use_text_retriever and self.text_pipeline is not None:
            retrieved = self.text_pipeline.retrieve(
                query=query_text,
                top_k=self.config.text_retriever_top_k,
            )
            text_chunks = [chunk.content for chunk in retrieved if chunk.content.strip()]

        context_sections = self._build_context_sections(
            query_text=query_text,
            nodes=nodes,
            triples=triples,
            neighbors=neighbors,
            subgraph=subgraph,
            shortest_path=shortest_path,
            text_chunks=text_chunks,
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
            "context_text": "\n\n".join(
                section for section in context_sections if section
            ),
        }

    def resolve_entity_seed(self, query: str | None = None) -> str:
        query_text = (query or self.config.query or self.config.entity or "").strip()
        configured_entity = self._sanitize_entity_name(self.config.entity or "")
        search_terms = self._build_search_terms(
            query_text=query_text,
            configured_entity=configured_entity,
        )
        seed_entities = self._seed_entities(
            query_text=query_text,
            nodes=[],
            triples=[],
            search_terms=search_terms,
        )
        for candidate in [configured_entity, *seed_entities]:
            normalized = self._sanitize_entity_name(candidate)
            if normalized:
                return normalized
        return ""

    def multi_hop(
        self,
        entity: str | None = None,
        hops: int | None = None,
        limit: int | None = None,
        relationship_types: Sequence[str] | None = None,
    ) -> list[KGTriple]:
        target_entity = (
            entity or self.config.entity or self.config.query or ""
        ).strip()
        if not target_entity:
            return []
        return self.kg_store.extract_subgraph(
            entity=target_entity,
            hops=hops if hops is not None else self.config.hops,
            limit=limit if limit is not None else self.config.subgraph_limit,
            relationship_types=relationship_types
            or self.config.relationship_types
            or None,
        )

    def retrieve_context(self, query: str | None = None) -> str:
        return self.retrieve(query=query)["context_text"]

    def format_triples(self, triples: Sequence[KGTriple]) -> str:
        return self._format_triples(triples)

    def rank_triples(
        self, triples: Sequence[KGTriple], query_text: str
    ) -> list[KGTriple]:
        """Rank triples by the configured lexical/mention/confidence score."""
        return self._rank_triples(triples, query_text)

    def _format_triples(self, triples: Sequence[KGTriple]) -> str:
        if not triples:
            return ""

        if not self.config.include_triple_metadata:
            return self.kg_store.triples_to_text(triples)

        lines: list[str] = []
        for triple in triples:
            subject = str(triple.get("subject", "")).strip()
            predicate = str(triple.get("predicate", "")).strip()
            obj = str(triple.get("object", "")).strip()
            rel_props = dict(triple.get("relationship_properties", {}) or {})

            meta: list[str] = []
            source_doc = str(rel_props.get("source_doc", "")).strip()
            page_range = str(rel_props.get("page_range", "")).strip()
            mention_count = self._mention_count(triple)
            confidence = self._confidence_score(triple)

            if source_doc:
                meta.append(f"source={source_doc}")
            if page_range:
                meta.append(f"pages={page_range}")
            if mention_count > 1:
                meta.append(f"mentions={mention_count}")
            if confidence > 0:
                meta.append(f"conf={confidence:.2f}")
            for key in ("year", "value", "unit"):
                v = rel_props.get(key)
                if v is not None:
                    meta.append(f"{key}={v}")

            suffix = f" [{', '.join(meta)}]" if meta else ""
            lines.append(f"({subject}, {predicate}, {obj}){suffix}")

        return "\n".join(lines)

    def _adaptive_subgraph(
        self,
        entity: str,
        hops: int,
        limit: int,
        relationship_types: Sequence[str] | None,
    ) -> list[KGTriple]:
        min_triples = max(0, int(self.config.min_subgraph_triples))
        start_hops = max(1, int(hops))
        max_hops = max(start_hops, int(self.config.max_hops))

        collected: list[KGTriple] = []
        seen: set[tuple[str, str, str]] = set()

        for hop in range(start_hops, max_hops + 1):
            batch = self.kg_store.extract_subgraph(
                entity=entity,
                hops=hop,
                limit=limit,
                relationship_types=relationship_types,
            )
            for triple in batch:
                key = self._triple_key(triple)
                if key in seen:
                    continue
                seen.add(key)
                collected.append(triple)

            if min_triples and len(collected) >= min_triples:
                break

        return collected

    def _rank_triples(
        self, triples: Sequence[KGTriple], query_text: str
    ) -> list[KGTriple]:
        if not triples:
            return []

        query_tokens = self._tokenize(query_text)
        if not query_tokens:
            return list(triples)

        max_mention = max(self._mention_count(triple) for triple in triples)
        max_mention = max(1, int(max_mention))

        scored: list[tuple[float, KGTriple]] = []
        for triple in triples:
            score = self._score_triple(
                triple=triple,
                query_tokens=query_tokens,
                max_mention=max_mention,
            )
            scored.append((score, triple))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [triple for _, triple in scored]

    def _score_triple(
        self,
        triple: KGTriple,
        query_tokens: set[str],
        max_mention: int,
    ) -> float:
        subject = str(triple.get("subject", "")).lower()
        predicate = str(triple.get("predicate", "")).lower()
        obj = str(triple.get("object", "")).lower()
        triple_tokens = set(
            tok for tok in _TOKEN_RE.findall(f"{subject} {predicate} {obj}") if tok
        )

        lexical_hits = len(query_tokens & triple_tokens)
        lexical_score = (
            float(lexical_hits) / float(max(1, len(query_tokens)))
            if query_tokens
            else 0.0
        )

        mention_count = self._mention_count(triple)
        if max_mention > 1:
            mention_score = math.log1p(mention_count) / math.log1p(max_mention)
        else:
            mention_score = 1.0

        confidence = self._confidence_score(triple)

        score = (
            self.config.ranker_weight_lexical * lexical_score
            + self.config.ranker_weight_mention * mention_score
            + self.config.ranker_weight_confidence * confidence
        )

        if self._is_system_link(triple):
            penalty = max(0.0, min(1.0, float(self.config.ranker_system_link_penalty)))
            score *= max(0.0, 1.0 - penalty)

        return score

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        tokens = [tok.lower() for tok in _TOKEN_RE.findall(text) if len(tok) >= 3]
        return {tok for tok in tokens if tok not in _QUESTION_STOPWORDS}

    @staticmethod
    def _mention_count(triple: KGTriple) -> int:
        rel_props = triple.get("relationship_properties", {}) or {}
        value = rel_props.get("mention_count")
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 1

    @staticmethod
    def _confidence_score(triple: KGTriple) -> float:
        rel_props = triple.get("relationship_properties", {}) or {}
        value = rel_props.get("confidence")
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, score))

    @staticmethod
    def _is_system_link(triple: KGTriple) -> bool:
        rel_props = triple.get("relationship_properties", {}) or {}
        if str(rel_props.get("extraction_method", "")).lower() == "system_linking":
            return True
        return str(triple.get("predicate", "")) in {"MENTIONED_IN", "SAME_AS"}

    def _seed_entities(
        self,
        query_text: str,
        nodes: Sequence[KGNode],
        triples: Sequence[KGTriple],
        search_terms: Sequence[str],
    ) -> list[str]:
        seeds: list[str] = []

        configured_entity = self._sanitize_entity_name(self.config.entity or "")
        if configured_entity:
            seeds.append(configured_entity)

        seeds.extend(search_terms)

        for node in nodes:
            # prefer elementId/node_id when available (more reliable for exact matching)
            node_id = str(node.get("node_id", "") or "").strip()
            if node_id:
                seeds.append(node_id)
                continue
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

    @staticmethod
    def _sanitize_entity_name(value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            return ""
        normalized = " ".join(cleaned.lower().split())
        if normalized in _PLACEHOLDER_ENTITIES:
            return ""
        return cleaned

    def _build_search_terms(self, query_text: str, configured_entity: str) -> list[str]:
        terms: list[str] = []

        if configured_entity:
            terms.append(configured_entity)

        if query_text:
            candidates = self._extract_entity_candidates(query_text)
            terms.extend(candidates)

            # If no clear entity-like candidates were found, extract keyword tokens
            # from the question instead of using the entire question text as a single
            # search term (which later is used as an exact entity and therefore
            # typically yields no matches).
            if not candidates:
                tokens = [t for t in _TOKEN_RE.findall(query_text) if len(t) >= 3]
                # filter common stopwords and short tokens, preserve order
                filtered = [t for t in tokens if t.lower() not in _QUESTION_STOPWORDS]
                # prefer multi-word title-like candidates first
                if len(filtered) <= 1 and len(tokens) <= 8:
                    # for short questions, keep full question as a term
                    terms.append(query_text)
                else:
                    # include up to 6 token candidates to improve retrieval
                    for tok in filtered[:6]:
                        terms.append(tok)

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

        for term in _NUMERIC_TERM_RE.findall(text):
            value = term.strip()
            if value:
                candidates.append(value)

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

    def _collect_triples(
        self, search_terms: Sequence[str], limit: int
    ) -> list[KGTriple]:
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
        text_chunks: Sequence[str] = (),
    ) -> list[str]:
        sections: list[str] = []

        if query_text:
            sections.append(f"Query: {query_text}")

        if text_chunks:
            sections.append("Retrieved text:\n" + "\n\n---\n\n".join(text_chunks))

        if nodes:
            sections.append("Matched nodes:\n" + self.kg_store.nodes_to_text(nodes))

        if triples:
            sections.append(
                "Matched triples:\n" + self._format_triples(triples)
            )

        if neighbors:
            sections.append("Neighbors:\n" + self.kg_store.nodes_to_text(neighbors))

        if subgraph:
            sections.append("Subgraph:\n" + self._format_triples(subgraph))

        if shortest_path:
            sections.append(
                "Shortest path:\n" + self._format_triples(shortest_path)
            )

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
