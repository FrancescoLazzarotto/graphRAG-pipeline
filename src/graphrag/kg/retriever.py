from __future__ import annotations

import logging
import math
import re
from typing import Any, Sequence

from graphrag import embeddings, questions
from graphrag.config import AgentConfig
from graphrag.kg.manager import KnowledgeGraphManager
from graphrag.text_rag.pipeline import StandardTextRAGPipeline
from graphrag.types import KGNode, KGTriple

logger = logging.getLogger("graphrag")

# Double quotes always delimit an entity; single quotes only when they are not
# intra-word apostrophes (Italian elisions like "cos'è l'economia" would
# otherwise yield the bogus entity "è l").
_QUOTED_ENTITY_RE = re.compile(r"\"([^\"]{2,})\"|(?<!\w)'([^']{2,})'(?!\w)")
# Lowercase connectors allowed INSIDE a capitalized phrase: without them
# "Via del Campo" splits into the useless fragments "Via" + "Campo" and the
# Lucene OR-query drifts to unrelated nodes ("Campo Base", "via Roma").
_TITLE_CONNECTORS = (
    "di|del|della|delle|dei|degli|dello|da|dal|dalla|e|ed|il|lo|la|le|gli|"
    "of|the|and|for|de|van|von|der"
)
_TITLE_ENTITY_RE = re.compile(
    rf"\b[A-Z][\w'-]*(?:\s+(?:(?:{_TITLE_CONNECTORS})\s+)*[A-Z][\w'-]*)+\b"
)
_SINGLE_TOKEN_ENTITY_RE = re.compile(r"\b[A-Z][\w'-]{2,}\b")
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
# Matches years (1900-2099) and quantities with explicit units so factual/numerical
# questions can seed KG lookup on DataValue nodes.
_NUMERIC_TERM_RE = re.compile(
    r"\b(?:(?:19|20)\d{2}|\d+(?:[.,]\d+)?\s*(?:%|kg|Mt|Gt|million|billion|tonnes|°C))(?!\w)",
    re.IGNORECASE,
)
# Cap on lowercase content keywords added per query: enough to cover the
# topic terms of a long question without flooding the Lucene OR-query.
_MAX_KEYWORD_TERMS = 8

# Function words only. Deliberately NOT stopwords: "via", "campo", "stato",
# "stati" — they occur inside real entity names ("Via del Campo", "Stati
# Uniti") and dropping them would corrupt phrase candidates at the edges.
_QUESTION_STOPWORDS_IT = {
    "chi", "che", "cos", "come", "cosa", "quando", "dove", "quale", "quali",
    "qual", "quanto", "quanta", "quanti", "quante", "perche", "perché",
    "sono", "hanno", "essere", "viene", "vengono", "puo", "può", "possono",
    "il", "lo", "la", "le", "gli", "un", "una", "uno",
    "di", "del", "della", "delle", "dei", "degli", "dello",
    "da", "dal", "dalla", "dalle", "dai", "dagli",
    "in", "nel", "nella", "nelle", "nei", "negli",
    "su", "sul", "sulla", "sulle", "sui", "sugli", "sullo",
    "al", "alla", "alle", "ai", "agli", "allo",
    "con", "per", "tra", "fra", "cui", "non", "anche", "ogni", "sia",
    "questo", "questa", "questi", "queste", "quello", "quella",
    "piu", "più", "meno", "senza", "dopo", "prima", "due", "tre",
    "ed",
}
_QUESTION_STOPWORDS_EN = {
    "what", "when", "where", "which", "who", "whom", "whose", "how", "why",
    "does", "do", "did", "is", "are", "was", "were", "be", "been", "being",
    "will", "can", "could", "should", "would", "may", "might", "must",
    "shall", "has", "have", "had",
    "the", "a", "an", "this", "that", "these", "those",
    "in", "on", "at", "by", "to", "of", "and", "or", "but", "if", "as",
    "than", "then", "from", "with", "without", "within", "into", "over",
    "under", "between", "among", "through", "during", "before", "after",
    "about", "against", "per",
    "it", "its", "they", "their", "them", "there", "here",
    "we", "our", "you", "your",
    "not", "no", "nor", "also", "only", "both", "each", "every", "any",
    "some", "all", "such", "same", "other", "another", "more", "most",
    "two", "three", "four", "say", "says", "said", "according", "following",
}
_QUESTION_STOPWORDS = _QUESTION_STOPWORDS_IT | _QUESTION_STOPWORDS_EN

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
        self._token_df_cache: tuple[dict[str, int], int] | None = None
        self._query_vector_cache: tuple[str, list[float]] | None = None
        if self.config.use_text_retriever and self.text_pipeline is None:
            # Without this warning a "hybrid"/"text_only" run silently degrades
            # to KG-only and the experiment looks valid while measuring the
            # wrong strategy.
            logger.warning(
                "use_text_retriever is enabled but no text_pipeline was provided: "
                "the text channel will be skipped for every retrieval."
            )

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

        query_vector = self._query_vector(query_text)

        if self.config.include_nodes and (search_terms or query_vector):
            nodes = self._collect_nodes(
                search_terms=search_terms,
                limit=self.config.nodes_limit,
                query_vector=query_vector,
            )

        if self.config.include_triples and (search_terms or query_vector):
            triples = self._collect_triples(
                search_terms=search_terms,
                limit=self.config.triples_limit,
                query_vector=query_vector,
            )

        seed_entities = self._seed_entities(
            query_text=query_text,
            nodes=nodes,
            triples=triples,
            search_terms=search_terms,
        )
        # Anchors for the neighbour, subgraph and shortest-path channels. Those
        # three start from a relationship scan filtered on the seed, so a seed
        # that matches no node costs a full graph walk and returns nothing —
        # measured at 9.4 s, 19.8 s and 5.1 s on one gold question whose anchor
        # was the raw phrase "C's of the Circular Economy for Food". Restricting
        # the anchor to names retrieval actually returned removes the cost
        # without removing any evidence: a seed that matches no node could not
        # have produced any.
        anchors = (
            self._graph_anchors(nodes, triples)
            if self.config.verify_anchor_exists
            else [self._sanitize_entity_name(seed) for seed in seed_entities]
        )
        anchors = [anchor for anchor in anchors if anchor]
        resolved_entity = configured_entity or (anchors[0] if anchors else "")
        resolved_entity = self._sanitize_entity_name(resolved_entity)

        if self.config.include_neighbors and resolved_entity:
            neighbors = self.kg_store.get_neighbors(
                entity=resolved_entity,
                limit=self.config.neighbors_limit,
                relationship_types=self.config.relationship_types or None,
            )

        if self.config.include_subgraph and resolved_entity:
            # Anchoring on retrieved nodes made every seed accurate, which also
            # made the neighbourhood narrower: subgraph_2hop was the one
            # strategy that lost recall. Expanding from the top few anchors,
            # each with a share of the budget, restores breadth without
            # reverting to question-word seeds.
            seeds = [resolved_entity]
            for anchor in anchors[1 : max(1, int(self.config.subgraph_seed_count))]:
                if anchor and anchor not in seeds:
                    seeds.append(anchor)
            budget = max(1, self.config.subgraph_limit // len(seeds))
            seen_subgraph: set[tuple[str, str, str]] = set()
            for seed in seeds:
                if self.config.adaptive_hops:
                    batch = self._adaptive_subgraph(
                        entity=seed,
                        hops=self.config.hops,
                        limit=budget,
                        relationship_types=self.config.relationship_types or None,
                    )
                else:
                    batch = self.kg_store.extract_subgraph(
                        entity=seed,
                        hops=self.config.hops,
                        limit=budget,
                        relationship_types=self.config.relationship_types or None,
                    )
                for triple in batch:
                    key = self._triple_key(triple)
                    if key in seen_subgraph:
                        continue
                    seen_subgraph.add(key)
                    subgraph.append(triple)

        if self.config.include_shortest_path:
            entity_a = self._sanitize_entity_name(self.config.entity_a or "") or (
                anchors[0] if len(anchors) > 0 else None
            )
            entity_b = self._sanitize_entity_name(self.config.entity_b or "") or (
                anchors[1] if len(anchors) > 1 else None
            )
            if entity_a and entity_b and entity_a != entity_b:
                shortest_path = self.kg_store.get_shortest_path(
                    entity_a=entity_a,
                    entity_b=entity_b,
                    max_depth=self.config.max_depth,
                )

        triples = self._drop_empty_predicates(triples)
        subgraph = self._drop_empty_predicates(subgraph)
        shortest_path = self._drop_empty_predicates(shortest_path)

        if self.config.rank_triples:
            triples = self._rank_triples(triples, query_text)
            subgraph = self._rank_triples(subgraph, query_text)

        text_chunks: list[str] = []
        text_sources: list[dict[str, str]] = []
        text_units: list[dict[str, str]] = []
        if self.config.use_text_retriever and self.text_pipeline is not None:
            retrieved = self._retrieve_text_chunks(query_text)
            for chunk in retrieved:
                if not chunk.content.strip():
                    continue
                text_chunks.append(chunk.content)
                # Keep each chunk's source tag ("<path>#page=N#chunk=M") so
                # downstream provenance analysis can attribute retrieved text to
                # its origin document; the chunk text itself stays in the context.
                text_sources.append(
                    {"source": chunk.source or "", "chunk_id": chunk.chunk_id}
                )
                # Same chunk with content attached: the citation pipeline (WP1)
                # needs text and provenance in a single unit to number them.
                text_units.append(
                    {
                        "content": chunk.content,
                        "source": chunk.source or "",
                        "chunk_id": chunk.chunk_id,
                    }
                )

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
            "text_sources": text_sources,
            "text_chunks": text_units,
            "context_sections": context_sections,
            "context_text": "\n\n".join(
                section for section in context_sections if section
            ),
        }

    def _retrieve_text_chunks(self, query_text: str) -> list[Any]:
        """Retrieve the text channel, diversified (WP4) and re-ranked (WP3).

        Three steps, in this order and for a reason:

        1. fetch a candidate pool, with MMR when it is enabled;
        2. cap how many chunks a single document may contribute;
        3. promote the chunks that actually define the term, when the question
           asks for a definition.

        The cap runs before the boost so the boost can only reorder chunks that
        survived diversification — the other order lets the cap discard the
        definitional chunk the boost just found.

        Args:
            query_text: The retrieval query (already rewritten, when WP7 fired).

        Returns:
            At most ``text_retriever_top_k`` retrieved chunks.
        """
        top_k = max(1, int(self.config.text_retriever_top_k))
        cap = max(0, int(self.config.text_retriever_max_per_doc))
        term = (
            questions.definitional_term(query_text)
            if self.config.prefer_verbatim_definitions
            else ""
        )
        # A pool is only worth fetching when something downstream can reorder
        # it; otherwise the extra candidates are dead weight on the index.
        needs_pool = bool(cap or term)
        pool_size = top_k
        if needs_pool:
            pool_size = max(top_k, int(self.config.text_retriever_fetch_k) or top_k * 4)

        retrieved = list(
            self.text_pipeline.retrieve(
                query=query_text,
                top_k=pool_size,
                mmr_lambda=(
                    self.config.text_retriever_mmr_lambda
                    if self.config.text_retriever_mmr
                    else None
                ),
                fetch_k=self.config.text_retriever_fetch_k or None,
            )
        )

        if cap:
            # An enumeration is usually one list on contiguous pages of one
            # document: the cap that diversifies every other question truncates
            # exactly the answer here, so it gets twice the budget.
            effective_cap = cap * 2 if questions.is_enumerative(query_text) else cap
            retrieved = self._cap_per_document(retrieved, effective_cap, top_k)

        if term:
            retrieved = self._promote_definitions(retrieved, term)

        return retrieved[:top_k]

    @staticmethod
    def _document_key(source: str | None) -> str:
        """Document a chunk came from, dropping the page and chunk markers."""
        return str(source or "").split("#page=")[0].split("#chunk=")[0].strip()

    @staticmethod
    def _cap_per_document(
        chunks: Sequence[Any], max_per_doc: int, top_k: int
    ) -> list[Any]:
        """Keep at most ``max_per_doc`` chunks per source document.

        Chunks over the cap are not dropped, they are demoted: when the corpus
        has nothing else to say, a truncated context is worse than a
        single-source one.
        """
        kept: list[Any] = []
        overflow: list[Any] = []
        seen: dict[str, int] = {}
        for chunk in chunks:
            key = KGRetriever._document_key(getattr(chunk, "source", ""))
            count = seen.get(key, 0)
            if count < max_per_doc:
                seen[key] = count + 1
                kept.append(chunk)
            else:
                overflow.append(chunk)

        if len(kept) < top_k:
            kept.extend(overflow[: top_k - len(kept)])
        else:
            kept.extend(overflow)
        return kept

    @staticmethod
    def _promote_definitions(chunks: Sequence[Any], term: str) -> list[Any]:
        """Re-rank so chunks defining ``term`` come first.

        A stable sort on the definitional score alone: retrieval order breaks
        ties, so a query where nothing looks like a definition comes back in
        exactly the order the retriever produced.
        """
        scored = [
            (questions.definition_score(getattr(chunk, "content", ""), term), index, chunk)
            for index, chunk in enumerate(chunks)
        ]
        if not any(score > 0.5 for score, _, _ in scored):
            return list(chunks)

        scored.sort(key=lambda item: (-item[0], item[1]))
        logger.info(
            "Definitional boost for %r: chunk %d moved to the top (score %.1f)",
            term,
            scored[0][1],
            scored[0][0],
        )
        return [chunk for _, _, chunk in scored]

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

        from_graph: list[str] = []
        for node in nodes:
            # prefer elementId/node_id when available (more reliable for exact matching)
            node_id = str(node.get("node_id", "") or "").strip()
            if node_id:
                from_graph.append(node_id)
                continue
            text = node.get("text", "").strip()
            if text:
                from_graph.append(text)

        for triple in triples:
            for candidate in (triple.get("subject", ""), triple.get("object", "")):
                candidate = candidate.strip()
                if candidate:
                    from_graph.append(candidate)

        if self.config.seed_from_retrieved:
            # seeds[0] becomes the anchor for neighbors, subgraph and shortest
            # path. Search terms are raw question words: anchoring on them asked
            # the graph for neighbours of "valuable" or "implementation", which
            # match no node, so those three channels returned nothing while
            # looking like they had run. Nodes the index actually matched are
            # real node names, already ordered by score.
            seeds.extend(from_graph)
            seeds.extend(search_terms)
        else:
            seeds.extend(search_terms)
            seeds.extend(from_graph)

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

            # Content keywords are ALWAYS added alongside entity candidates: a
            # capitalized hit like "Via del Campo" must not silence the lowercase
            # content terms ("biogas", "digestate") that identify the topic —
            # candidates alone drift the Lucene query onto homonym nodes.
            covered = {
                tok.lower()
                for candidate in candidates
                for tok in _TOKEN_RE.findall(candidate)
            }
            keywords = [
                tok
                for tok in _TOKEN_RE.findall(query_text)
                if len(tok) >= 3
                and tok.lower() not in _QUESTION_STOPWORDS
                and tok.lower() not in covered
            ]
            if self.config.lexical_specificity:
                keywords = self._rank_keywords_by_specificity(keywords)
            terms.extend(keywords[:_MAX_KEYWORD_TERMS])

        if not terms and query_text:
            terms.append(query_text)

        return self._unique_values(terms)

    def _token_df(self) -> tuple[dict[str, int], int]:
        """Node-name document frequency, loaded once and reused."""
        if self._token_df_cache is None:
            try:
                self._token_df_cache = self.kg_store.token_document_frequency(
                    cache_path=self.config.lexical_df_cache_path or None
                )
            except Exception as exc:  # noqa: BLE001 - specificity is optional
                # Falling back to the flat query is always safe: it is what the
                # retriever did before this feature existed.
                logger.warning(
                    "token document frequency unavailable (%s): "
                    "lexical specificity disabled for this session",
                    exc,
                )
                self._token_df_cache = ({}, 0)
        return self._token_df_cache

    def _rank_keywords_by_specificity(self, keywords: Sequence[str]) -> list[str]:
        """Drop keywords that match too many node names, rarest first.

        Without this the ``_MAX_KEYWORD_TERMS`` cap keeps whichever tokens the
        question happens to state first, which is word order, not relevance.
        """
        token_df, total = self._token_df()
        if not token_df or total <= 0:
            return list(keywords)

        ceiling = max(1, int(total * self.config.lexical_df_max_ratio))
        scored: list[tuple[int, int, str]] = []
        for position, keyword in enumerate(keywords):
            df = token_df.get(keyword.lower(), 0)
            if df > ceiling:
                continue
            # Unknown tokens (df 0) are not evidence of specificity: they match
            # nothing on their own and only matter through the index's fuzzy
            # behaviour, so they rank after tokens that do occur.
            scored.append((df if df > 0 else total + 1, position, keyword))

        scored.sort()
        return [keyword for _, _, keyword in scored]

    def _term_boosts(self, terms: Sequence[str]) -> dict[str, float]:
        """Lucene weights: phrases first, then single tokens by rarity."""
        if not self.config.lexical_specificity:
            return {}
        token_df, total = self._token_df()
        boosts: dict[str, float] = {}
        for term in terms:
            if " " in term.strip():
                boosts[term] = self.config.lexical_phrase_boost
                continue
            if not token_df or total <= 0:
                continue
            df = token_df.get(term.lower(), 0)
            if df <= 0:
                continue
            # log10(total/df) lands around 1 for a token in ~10 % of names and
            # around 3 for one in ~0.1 %, which is the range worth separating.
            rarity = math.log10(total / df)
            boosts[term] = min(
                self.config.lexical_max_token_boost, max(1.0, rarity)
            )
        return boosts

    @staticmethod
    def _trim_stopword_edges(phrase: str) -> str:
        """Strip leading/trailing stopword tokens from a capitalized phrase.

        Sentence-initial function words are capitalized too, so the title regex
        captures "In the Via del Campo". As a quoted Lucene phrase that never
        matches the node "VIA DEL CAMPO"; trimming the edges (never the inside)
        recovers the real entity name.
        """
        tokens = phrase.split()
        while tokens and tokens[0].lower() in _QUESTION_STOPWORDS:
            tokens.pop(0)
        while tokens and tokens[-1].lower() in _QUESTION_STOPWORDS:
            tokens.pop()
        return " ".join(tokens)

    def _extract_entity_candidates(self, text: str) -> list[str]:
        candidates: list[str] = []

        for match in _QUOTED_ENTITY_RE.finditer(text):
            value = (match.group(1) or match.group(2) or "").strip()
            if value:
                candidates.append(value)

        for phrase in _TITLE_ENTITY_RE.findall(text):
            value = self._trim_stopword_edges(phrase.strip())
            if value:
                candidates.append(value)

        # Single capitalized tokens already inside a phrase candidate are noise:
        # "Via" and "Campo" alone match "via Roma" / "Campo Base" instead of
        # reinforcing "Via del Campo".
        covered = {
            tok.lower()
            for candidate in candidates
            for tok in _TOKEN_RE.findall(candidate)
        }
        for token in _SINGLE_TOKEN_ENTITY_RE.findall(text):
            low = token.lower()
            if low not in _QUESTION_STOPWORDS and low not in covered:
                candidates.append(token)

        for term in _NUMERIC_TERM_RE.findall(text):
            value = term.strip()
            if value:
                candidates.append(value)

        return self._unique_values(candidates)

    def _graph_anchors(
        self, nodes: Sequence[KGNode], triples: Sequence[KGTriple]
    ) -> list[str]:
        """Anchor candidates guaranteed to exist, best first.

        Node names come from the index and triple endpoints from real edges, so
        every candidate here matches a node without a verification query.
        """
        anchors: list[str] = []
        for node in nodes:
            node_id = str(node.get("node_id", "") or "").strip()
            anchors.append(node_id or str(node.get("text", "") or "").strip())
        for triple in triples:
            # elementId over name on purpose: matching a seed by name compares
            # six lowercased properties on every candidate, which is a scan
            # (0.92 s for neighbours, 2.29 s for a 1-hop subgraph), while the
            # id lookup is direct (0.04 s / 0.05 s). Strategies with
            # include_nodes off have only triples to anchor on, so without this
            # they pay the scan on every question.
            for id_key, name_key in (("subject_id", "subject"), ("object_id", "object")):
                candidate = str(triple.get(id_key, "") or "").strip()
                if not candidate:
                    candidate = str(triple.get(name_key, "") or "").strip()
                if candidate:
                    anchors.append(candidate)
        return self._unique_values(anchors)

    def _drop_empty_predicates(self, triples: Sequence[KGTriple]) -> list[KGTriple]:
        """Remove triples whose predicate cannot support an answer.

        Filtering here rather than in Cypher keeps the index lookup untouched:
        a dropped triple still counted toward the seed's score, so the ranking
        of what remains is the ranking the retriever intended.
        """
        dropped = {p.strip().upper() for p in self.config.drop_predicates if p.strip()}
        if not dropped:
            return list(triples)
        return [
            triple
            for triple in triples
            if str(triple.get("predicate", "")).strip().upper() not in dropped
        ]

    def _query_vector(self, query_text: str) -> list[float]:
        """Embed the question once per retrieval, or return [] if unavailable."""
        if not self.config.vector_retrieval or not query_text:
            return []
        if self._query_vector_cache is not None:
            cached_text, cached_vec = self._query_vector_cache
            if cached_text == query_text:
                return cached_vec
        try:
            vector = embeddings.encode_query(query_text)
        except embeddings.EmbeddingUnavailable as exc:
            # Lexical retrieval still works, so a missing encoder degrades the
            # result instead of failing the run — but it must be visible, since
            # the cross-lingual half is where most of the recall lives.
            logger.warning(
                "embedding endpoint unavailable (%s): vector channel skipped, "
                "retrieval is lexical-only for this query",
                exc,
            )
            vector = []
        self._query_vector_cache = (query_text, vector)
        return vector

    def _collect_nodes(
        self,
        search_terms: Sequence[str],
        limit: int,
        query_vector: Sequence[float] = (),
    ) -> list[KGNode]:
        if limit <= 0:
            return []

        collected: list[KGNode] = []
        seen: set[str] = set()

        # Vector matches go first: they are the only channel that can cross the
        # language gap, and the lexical channel below fills the rest of the
        # budget with exact surface matches.
        if query_vector:
            for row in self.kg_store.vector_search_nodes(
                vector=query_vector,
                limit=min(self.config.vector_nodes_limit, limit),
                index=self.config.vector_index,
                labels=self.config.labels or None,
                min_score=self.config.vector_min_score,
            ):
                key = self._node_key(row)
                if key in seen:
                    continue
                seen.add(key)
                collected.append(row)

        # One indexed query for all terms; rows arrive best-score first.
        indexed = self.kg_store.fulltext_search_nodes(
            terms=search_terms,
            labels=self.config.labels or None,
            limit=limit,
            boosts=self._term_boosts(search_terms),
        )
        if indexed is not None:
            for row in indexed:
                key = self._node_key(row)
                if key in seen:
                    continue
                seen.add(key)
                collected.append(row)
                if len(collected) >= limit:
                    break
            return collected

        # Fallback (no full-text index): legacy per-term CONTAINS scan.
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
        self,
        search_terms: Sequence[str],
        limit: int,
        query_vector: Sequence[float] = (),
    ) -> list[KGTriple]:
        if limit <= 0:
            return []

        collected: list[KGTriple] = []
        seen: set[tuple[str, str, str]] = set()

        if query_vector:
            for row in self.kg_store.vector_search_triples(
                vector=query_vector,
                limit=min(self.config.vector_triples_limit, limit),
                seed_limit=self.config.vector_seed_limit,
                index=self.config.vector_index,
                labels=self.config.labels or None,
                relationship_types=self.config.relationship_types or None,
            ):
                key = self._triple_key(row)
                if key in seen:
                    continue
                seen.add(key)
                collected.append(row)

        # One indexed query for all terms; rows arrive best-seed-score first.
        indexed = self.kg_store.fulltext_search_triples(
            terms=search_terms,
            labels=self.config.labels or None,
            relationship_types=self.config.relationship_types or None,
            limit=limit,
            boosts=self._term_boosts(search_terms),
        )
        if indexed is not None:
            for row in indexed:
                key = self._triple_key(row)
                if key in seen:
                    continue
                seen.add(key)
                collected.append(row)
                if len(collected) >= limit:
                    break
            return collected

        # Fallback (no full-text index): legacy per-term CONTAINS scan.
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
