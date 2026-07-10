from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from graphrag.agent.cache import LRUCache
from graphrag.agent.compression import ContextCompressor
from graphrag.config import AgentConfig
from graphrag.kg.retriever import KGRetriever
from graphrag.llm.manager import LLMManager
from graphrag.llm.prompts import PromptLibrary
from graphrag.llm.refusal import looks_like_refusal
from graphrag.types import RAGState

logger = logging.getLogger("graphrag")


class KGRAGAgent:
    def __init__(
        self,
        config: AgentConfig,
        kg_retriever: KGRetriever | None = None,
        llm: LLMManager | None = None,
    ) -> None:
        self.config = config
        self.kg_retriever = kg_retriever
        self.llm = llm
        self.compressor = ContextCompressor(
            config.max_content_tokens, config.token_estimator_ratio
        )
        self.cache = LRUCache(config.cache_maxsize) if config.enable_cache else None

        if self.llm is not None and self.config.llm_warmup:
            self.llm.warmup()

        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(RAGState)

        builder.add_node("decompose", self._decompose)
        builder.add_node("route", self._adaptive_route)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("grade", self._grade)
        builder.add_node("rewrite", self._rewrite)
        builder.add_node("generate", self._generate)

        builder.add_edge(START, "decompose")
        builder.add_edge("decompose", "route")
        # The retrieval mode chosen in `route` is read directly by `_retrieve`
        # from the state, so the edge is unconditional.
        builder.add_edge("route", "retrieve")
        builder.add_edge("retrieve", "grade")

        def grade_condition(state: RAGState):
            if int(state.get("rewrite_count", 0) or 0) >= 3:
                return "generate"
            if state.get("relevance") == "relevant":
                return "generate"
            return "rewrite"

        builder.add_conditional_edges("grade", grade_condition)
        builder.add_edge("rewrite", "retrieve")
        builder.add_edge("generate", END)

        return builder.compile()

    def _decompose(self, state: RAGState) -> dict:
        question = state.get("question", "").strip()
        if not question:
            return {"sub_questions": []}

        if not self.config.enable_decomposition_step:
            return {"sub_questions": [question]}

        prompt = PromptLibrary.decomposition_prompt(self.config)
        rendered = prompt.invoke({"question": question})

        if self.llm is None:
            return {"sub_questions": [question]}

        model = self.llm.load_llm()
        output = model.invoke(rendered)
        text = output.content if hasattr(output, "content") else str(output)
        text = text.strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return {
                    "sub_questions": [
                        str(item).strip() for item in parsed if str(item).strip()
                    ]
                }
        except json.JSONDecodeError:
            pass

        # Fallback for non-JSON output: strip bullets and "1." / "2)" numbering
        # so malformed sub-questions don't flow into retrieval.
        logger.warning(
            "Decomposition output was not a JSON array; falling back to "
            "line-based parsing (first 200 chars): %s",
            text[:200],
        )
        sub_questions = []
        for line in text.splitlines():
            cleaned = re.sub(r"^\s*(?:[-•*]|\d+[.)])\s*", "", line).strip()
            if cleaned:
                sub_questions.append(cleaned)
        return {"sub_questions": sub_questions or [question]}

    def _rewrite(self, state: RAGState) -> dict:
        question = state.get("question", "").strip()
        if not question:
            return {"rewritten_question": ""}

        prompt = PromptLibrary.rewrite_prompt(self.config)
        rendered = prompt.invoke({"question": question})

        if self.llm is None:
            return {"rewritten_question": question}

        model = self.llm.load_llm()
        output = model.invoke(rendered)
        rewritten = str(
            output.content if hasattr(output, "content") else output
        ).strip()
        rewrite_count = state.get("rewrite_count", 0) + 1
        # Generation is deterministic (temperature 0 / do_sample=False), so a
        # rewrite equal to the query already tried can never change retrieval:
        # cap the counter to exit the loop instead of burning identical
        # LLM + retrieval rounds.
        previous = str(
            state.get("rewritten_question") or state.get("question", "")
        ).strip()
        if rewritten and rewritten == previous:
            logger.info(
                "Rewrite produced an identical query; short-circuiting the rewrite loop."
            )
            rewrite_count = max(rewrite_count, 3)
        # The grade -> generate edge already short-circuits once rewrite_count
        # reaches 3, so a forced relevance flag here would never be read. Just
        # bump the counter.
        return {"rewritten_question": rewritten, "rewrite_count": rewrite_count}

    def _adaptive_route(self, state: RAGState) -> dict:
        question = state.get("question", "").strip()
        if not self.config.enable_adaptive_routing_step:
            return {"chosen_retrieval_mode": "HYBRID"}

        if not self.llm:
            # Without an LLM router available, prefer HYBRID to avoid dropping KG evidence.
            return {"chosen_retrieval_mode": "HYBRID"}

        prompt = PromptLibrary.adaptive_router_prompt(self.config)
        rendered = prompt.invoke({"question": question})
        model = self.llm.load_llm()
        output = model.invoke(rendered)
        mode = (
            str(output.content if hasattr(output, "content") else output)
            .strip()
            .upper()
        )

        if mode not in ["TEXT", "KG", "HYBRID", "MULTIHOP"]:
            mode = "HYBRID"

        return {"chosen_retrieval_mode": mode}

    def _retrieve(self, state: RAGState) -> dict:
        query = str(state.get("rewritten_question") or state.get("question", "")).strip()
        mode = state.get("chosen_retrieval_mode", "HYBRID")
        sub_questions = state.get("sub_questions", []) or []
        retrieval_queries = self._build_retrieval_queries(
            query=query,
            sub_questions=sub_questions if isinstance(sub_questions, list) else [],
        )
        cache_query = " || ".join(retrieval_queries) if retrieval_queries else query
        # The key already contains the (possibly rewritten) query text, and
        # retrieval is deterministic for a fixed query and graph. Adding the
        # rewrite counter here would only force a re-retrieval when a rewrite
        # produces the exact same text — pure wasted Neo4j round-trips.
        cache_mode = str(mode)

        if self.cache:
            hit = self.cache.get(cache_query, cache_mode)
            if hit is not None:
                return hit

        retrieved_data: dict[str, Any] = {
            "query": query,
            "context_text": "",
            "nodes": [],
            "triples": [],
            "neighbors": [],
            "subgraph": [],
            "shortest_path": [],
        }
        context = ""

        if self.kg_retriever:
            mm = str(mode).upper() if mode is not None else "HYBRID"
            if mm == "TEXT":
                text_sections: list[str] = []
                for candidate_query in retrieval_queries:
                    value = str(self.kg_retriever.retrieve_context(candidate_query)).strip()
                    if value:
                        text_sections.append(value)

                context = self._merge_context_sections(text_sections)
                retrieved_data["context_text"] = context

            elif mm in {"KG", "HYBRID"}:
                nodes: list[dict[str, Any]] = []
                triples: list[dict[str, Any]] = []
                neighbors: list[dict[str, Any]] = []
                subgraph: list[dict[str, Any]] = []
                shortest_path: list[dict[str, Any]] = []
                context_sections: list[str] = []

                node_seen: set[tuple[str, str]] = set()
                neighbor_seen: set[tuple[str, str]] = set()
                triple_seen: set[tuple[str, str, str]] = set()
                subgraph_seen: set[tuple[str, str, str]] = set()
                shortest_path_seen: set[tuple[str, str, str]] = set()

                for candidate_query in retrieval_queries:
                    batch = self.kg_retriever.retrieve(candidate_query)

                    nodes = self._merge_nodes(
                        existing=nodes,
                        incoming=batch.get("nodes", []),
                        seen=node_seen,
                        limit=max(1, int(self.config.nodes_limit)),
                    )
                    triples = self._merge_triples(
                        existing=triples,
                        incoming=batch.get("triples", []),
                        seen=triple_seen,
                        limit=max(1, int(self.config.triples_limit)),
                    )
                    neighbors = self._merge_nodes(
                        existing=neighbors,
                        incoming=batch.get("neighbors", []),
                        seen=neighbor_seen,
                        limit=max(1, int(self.config.neighbors_limit)),
                    )
                    subgraph = self._merge_triples(
                        existing=subgraph,
                        incoming=batch.get("subgraph", []),
                        seen=subgraph_seen,
                        limit=max(1, int(self.config.subgraph_limit)),
                    )
                    shortest_path = self._merge_triples(
                        existing=shortest_path,
                        incoming=batch.get("shortest_path", []),
                        seen=shortest_path_seen,
                        limit=max(1, int(self.config.subgraph_limit)),
                    )

                    candidate_context = str(batch.get("context_text", "")).strip()
                    if candidate_context:
                        context_sections.append(candidate_context)

                if (
                    self.config.rerank_merged_results
                    and self.config.rank_triples
                    and len(retrieval_queries) > 1
                ):
                    # Merged multi-query results keep arrival order by default;
                    # re-rank globally against the original question.
                    triples = self.kg_retriever.rank_triples(triples, query)
                    subgraph = self.kg_retriever.rank_triples(subgraph, query)

                context = self._merge_context_sections(context_sections)
                if mm == "KG" and not context:
                    context = self._format_triples_for_context(triples + subgraph + shortest_path)

                retrieved_data = {
                    "query": query,
                    "context_text": context,
                    "nodes": nodes,
                    "triples": triples,
                    "neighbors": neighbors,
                    "subgraph": subgraph,
                    "shortest_path": shortest_path,
                }

            elif mm == "MULTIHOP":
                subgraph: list[dict[str, Any]] = []
                triple_seen: set[tuple[str, str, str]] = set()
                seed_entities: list[str] = []
                seed_seen: set[str] = set()

                for candidate_query in retrieval_queries:
                    seed = self.kg_retriever.resolve_entity_seed(candidate_query)
                    normalized = seed.strip().lower()
                    if seed and normalized not in seed_seen:
                        seed_seen.add(normalized)
                        seed_entities.append(seed)

                for seed in seed_entities[:2]:
                    batch = self.kg_retriever.multi_hop(
                        entity=seed,
                        hops=self.config.hops,
                        limit=self.config.subgraph_limit,
                    )
                    subgraph = self._merge_triples(
                        existing=subgraph,
                        incoming=batch,
                        seen=triple_seen,
                        limit=max(1, int(self.config.subgraph_limit)),
                    )

                context = self._format_triples_for_context(subgraph)
                retrieved_data = {
                    "query": query,
                    "context_text": context,
                    "nodes": [],
                    "triples": [],
                    "neighbors": [],
                    "subgraph": subgraph,
                    "shortest_path": [],
                }

            else:
                retrieved_data = self.kg_retriever.retrieve(query)
                context = str(retrieved_data.get("context_text", ""))

        compressed_context = self.compressor.compress(context)
        triples = (
            retrieved_data.get("triples", [])
            if isinstance(retrieved_data, dict)
            else []
        )
        nodes = (
            retrieved_data.get("nodes", []) if isinstance(retrieved_data, dict) else []
        )
        neighbors = (
            retrieved_data.get("neighbors", [])
            if isinstance(retrieved_data, dict)
            else []
        )
        subgraph = (
            retrieved_data.get("subgraph", [])
            if isinstance(retrieved_data, dict)
            else []
        )
        shortest_path = (
            retrieved_data.get("shortest_path", [])
            if isinstance(retrieved_data, dict)
            else []
        )

        result = {
            "text_context": compressed_context,
            "kg_triples": triples if isinstance(triples, list) else [],
            "retrieved_nodes": nodes if isinstance(nodes, list) else [],
            "retrieved_nodes_count": len(nodes) if isinstance(nodes, list) else 0,
            "retrieved_neighbors": neighbors if isinstance(neighbors, list) else [],
            "retrieved_neighbors_count": len(neighbors)
            if isinstance(neighbors, list)
            else 0,
            "retrieved_subgraph": subgraph if isinstance(subgraph, list) else [],
            "retrieved_subgraph_count": len(subgraph)
            if isinstance(subgraph, list)
            else 0,
            "retrieved_shortest_path": shortest_path
            if isinstance(shortest_path, list)
            else [],
            "retrieved_shortest_path_count": len(shortest_path)
            if isinstance(shortest_path, list)
            else 0,
        }

        if self.cache:
            self.cache.put(cache_query, cache_mode, result)

        return result

    def _build_retrieval_queries(
        self,
        query: str,
        sub_questions: list[object],
        max_queries: int = 4,
    ) -> list[str]:
        queries: list[str] = []
        seen: set[str] = set()

        def add(candidate: str) -> None:
            value = " ".join(str(candidate).split()).strip()
            if not value:
                return
            key = value.lower()
            if key in seen:
                return
            seen.add(key)
            queries.append(value)

        add(query)
        if self.config.enable_decomposition_step:
            for sub in sub_questions:
                add(str(sub))
                if len(queries) >= max_queries:
                    break

        return queries or ([query] if query else [])

    @staticmethod
    def _node_key(node: dict[str, Any]) -> tuple[str, str]:
        node_id = str(node.get("node_id", "")).strip()
        if node_id:
            return ("id", node_id)
        return ("text", str(node.get("text", "")).strip().lower())

    @staticmethod
    def _triple_key(triple: dict[str, Any]) -> tuple[str, str, str]:
        subject_id = str(triple.get("subject_id", "")).strip()
        object_id = str(triple.get("object_id", "")).strip()
        predicate = str(triple.get("predicate", "")).strip().lower()

        if subject_id and object_id:
            return (f"id:{subject_id}", predicate, f"id:{object_id}")

        subject = str(triple.get("subject", "")).strip().lower()
        obj = str(triple.get("object", "")).strip().lower()
        return (subject, predicate, obj)

    def _merge_nodes(
        self,
        existing: list[dict[str, Any]],
        incoming: object,
        seen: set[tuple[str, str]],
        limit: int,
    ) -> list[dict[str, Any]]:
        if not isinstance(incoming, list):
            return existing

        for item in incoming:
            if not isinstance(item, dict):
                continue
            key = self._node_key(item)
            if key in seen:
                continue
            seen.add(key)
            existing.append(item)
            if len(existing) >= limit:
                break

        return existing

    def _merge_triples(
        self,
        existing: list[dict[str, Any]],
        incoming: object,
        seen: set[tuple[str, str, str]],
        limit: int,
    ) -> list[dict[str, Any]]:
        if not isinstance(incoming, list):
            return existing

        for item in incoming:
            if not isinstance(item, dict):
                continue
            key = self._triple_key(item)
            if key in seen:
                continue
            seen.add(key)
            existing.append(item)
            if len(existing) >= limit:
                break

        return existing

    @staticmethod
    def _merge_context_sections(sections: list[str]) -> str:
        merged: list[str] = []
        seen: set[str] = set()
        for section in sections:
            value = section.strip()
            if not value:
                continue
            key = " ".join(value.split()).lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(value)
        return "\n\n".join(merged)

    def _format_triples_for_context(self, triples: list[dict[str, Any]]) -> str:
        if not triples:
            return ""
        try:
            if hasattr(self.kg_retriever, "format_triples") and self.kg_retriever:
                return str(self.kg_retriever.format_triples(triples))
            if self.kg_retriever:
                return str(self.kg_retriever.kg_store.triples_to_text(triples))
        except Exception:
            logger.warning(
                "Triple formatting failed; dropping %d triples from the context.",
                len(triples),
                exc_info=True,
            )
            return ""
        return ""

    def _grade(self, state: RAGState) -> dict:
        nodes_count = int(state.get("retrieved_nodes_count", 0) or 0)
        triples_count = len(state.get("kg_triples", []) or [])
        subgraph_count = int(state.get("retrieved_subgraph_count", 0) or 0)
        shortest_path_count = int(state.get("retrieved_shortest_path_count", 0) or 0)
        text_context = str(state.get("text_context", "") or "")
        has_text_evidence = bool(text_context.strip())

        # Stronger semantic gating: ensure retrieved KG items actually match the
        # salient terms in the query/context instead of accepting any hit.
        kg_evidence_units = (
            nodes_count + triples_count + subgraph_count + shortest_path_count
        )
        evidence_units = kg_evidence_units + (1 if has_text_evidence else 0)
        if evidence_units == 0:
            return {"relevance": "not_relevant"}

        query = state.get("rewritten_question") or state.get("question", "")
        salient = set(self._extract_salient_terms_from_text(query))
        if not salient:
            salient = set(self._extract_salient_terms(query=query, context=""))

        matched = 0

        # examine triples for semantic overlap
        for triple in state.get("kg_triples", []) or []:
            hay = f"{triple.get('subject', '')} {triple.get('predicate', '')} {triple.get('object', '')}".lower()
            if any(term in hay for term in salient):
                matched += 1

        # examine nodes
        for node in state.get("retrieved_nodes", []) or state.get("nodes", []) or []:
            text = str(node.get("text", "")).lower()
            if any(term in text for term in salient):
                matched += 1

        # examine subgraph and shortest path textualizations
        for item in (
            state.get("retrieved_subgraph", []) or state.get("subgraph", []) or []
        ):
            hay = f"{item.get('subject', '')} {item.get('predicate', '')} {item.get('object', '')}".lower()
            if any(term in hay for term in salient):
                matched += 1

        for item in (
            state.get("retrieved_shortest_path", [])
            or state.get("shortest_path", [])
            or []
        ):
            hay = f"{item.get('subject', '')} {item.get('predicate', '')} {item.get('object', '')}".lower()
            if any(term in hay for term in salient):
                matched += 1

        if has_text_evidence:
            context_lower = text_context.lower()
            if any(term in context_lower for term in salient):
                matched += 1

        # Determine relevance: require at least one semantic match, and either
        # multiple matches or a reasonable match ratio to accept as relevant.
        match_ratio = matched / max(1, evidence_units)
        if kg_evidence_units == 0 and has_text_evidence:
            is_relevant = matched >= 1
        else:
            is_relevant = matched >= 1 and (matched >= 2 or match_ratio >= 0.30)

        logger.debug(
            "Grading retrieval: evidence_units=%d matched=%d match_ratio=%.2f salient=%s",
            evidence_units,
            matched,
            match_ratio,
            list(salient)[:8],
        )

        return {"relevance": "relevant" if is_relevant else "not_relevant"}

    def _generate(self, state: RAGState) -> dict:
        query = state.get("question", "")
        context = state.get("text_context", "")
        has_text_evidence = bool(str(context or "").strip())
        nodes_count = int(state.get("retrieved_nodes_count", 0) or 0)
        triples_count = len(state.get("kg_triples", []) or [])
        subgraph_count = int(state.get("retrieved_subgraph_count", 0) or 0)
        shortest_path_count = int(state.get("retrieved_shortest_path_count", 0) or 0)

        kg_evidence_units = (
            nodes_count + triples_count + subgraph_count + shortest_path_count
        )
        evidence_units = kg_evidence_units + (1 if has_text_evidence else 0)

        if evidence_units == 0:
            logger.warning(
                "Generation with zero evidence: retrieval mode=%s returned no nodes, "
                "triples, subgraph, shortest_path or text context for query=%r",
                state.get("chosen_retrieval_mode", "HYBRID"),
                str(query)[:200],
            )
            if LLMManager._detect_query_language(query) == "it":
                return {
                    "answer": (
                        "Il contesto disponibile non è sufficiente per dare una risposta fondata. "
                        "Prova a riformulare la domanda o a renderla più specifica."
                    )
                }
            return {
                "answer": (
                    "The provided context is insufficient to generate a grounded response. "
                    "Please provide additional context or a more specific question."
                )
            }

        sparse_context = evidence_units <= 2 and len(str(context or "").strip()) < 1600
        effective_query = query
        if sparse_context:
            # Match the instruction language to the question language: a fixed
            # Italian instruction on an English question pushes the model into
            # mixed-language answers.
            if LLMManager._detect_query_language(query) == "it":
                effective_query = (
                    query
                    + "\n\nIstruzione: rispondi direttamente usando solo il contesto disponibile. "
                    + "Se il contesto e limitato, fornisci comunque la migliore risposta possibile e aggiungi una breve sezione 'Limiti e fiducia'."
                )
            else:
                effective_query = (
                    query
                    + "\n\nInstruction: answer directly using only the available context. "
                    + "If the context is limited, still provide the best possible answer and add a short 'Limits and confidence' section."
                )

        if self.llm:
            result = self.llm.generate(
                query=effective_query, context=context, config=self.config
            )
            answer = result.get("answer", "")
            logger.info(
                "LLM returned (first 500 chars): %s | sparse_context=%s | evidence_units=%d",
                answer[:500],
                sparse_context,
                evidence_units,
            )
            if evidence_units > 0 and self._should_replace_with_fallback(
                answer=answer,
                query=query,
                context=context,
                triples=state.get("kg_triples", []) or [],
                sparse_context=sparse_context,
            ):
                logger.info(
                    "FALLBACK TRIGGERED: replacing LLM answer with evidence-based fallback"
                )
                answer = self._build_sparse_fallback_answer(
                    query=query,
                    context=context,
                    triples=state.get("kg_triples", []) or [],
                    language=LLMManager._detect_query_language(query),
                )
            verification_section = self._build_verification_section(
                triples=state.get("kg_triples", []) or [],
                nodes=state.get("retrieved_nodes", []) or [],
            )
            if verification_section:
                answer = answer.rstrip() + "\n\n" + verification_section

            return {"answer": answer}

        return {"answer": "LLM not available."}

    def _should_replace_with_fallback(
        self,
        answer: str,
        query: str,
        context: str,
        triples: list[dict[str, object]],
        sparse_context: bool,
    ) -> bool:
        # A genuine refusal / empty answer is always replaced with the evidence
        # block; this is the only unconditional trigger.
        if looks_like_refusal(answer):
            return True

        # Otherwise only intervene when the context was sparse AND the answer is
        # ungrounded. A well-formed answer that references a salient query/context
        # term or a retrieved triple is kept as-is. We deliberately avoid the old
        # "meta-marker" heuristic, which fired on common words (context,
        # information, analysis, ...) and replaced perfectly good answers.
        if not sparse_context:
            return False

        answer_lower = answer.lower()
        salient_terms = self._extract_salient_terms(query=query, context=context)
        triple_terms = self._extract_salient_terms_from_triples(triples)

        # Nothing to judge groundedness against: trust the model's answer.
        if not salient_terms and not triple_terms:
            return False

        if any(term in answer_lower for term in salient_terms):
            return False
        if triple_terms and any(term in answer_lower for term in triple_terms):
            return False

        return True

    @staticmethod
    def _build_sparse_fallback_answer(
        query: str,
        context: str,
        triples: list[dict[str, object]],
        language: str = "en",
    ) -> str:
        triple_summaries = KGRAGAgent._triple_summaries(triples, query=query)
        highlights = KGRAGAgent._extract_context_highlights(
            query=query, context=context
        )
        is_it = language == "it"

        if triple_summaries or highlights:
            evidence_block = "\n".join(
                f"- {line}" for line in (triple_summaries or highlights)
            )
            if is_it:
                return (
                    "Dal contesto disponibile emergono i seguenti elementi rilevanti. "
                    "La risposta e quindi parziale, ma contiene le evidenze trovate nel grafo.\n\n"
                    "Limiti e fiducia:\n"
                    "Il contesto e limitato, quindi non posso inferire l'intero perimetro tematico con alta fiducia.\n\n"
                    "Evidenze rilevanti:\n"
                    f"{evidence_block}"
                )
            return (
                "The available context surfaces the following relevant elements. "
                "The answer is therefore partial, but it reports the evidence found in the graph.\n\n"
                "Limits and confidence:\n"
                "The context is limited, so the full thematic scope cannot be inferred with high confidence.\n\n"
                "Relevant evidence:\n"
                f"{evidence_block}"
            )

        if is_it:
            return (
                "Il contesto disponibile e troppo scarno per costruire una risposta affidabile. "
                "Serve un recupero piu specifico o piu evidenza dal grafo."
            )
        return (
            "The available context is too sparse to build a reliable answer. "
            "A more specific retrieval or more graph evidence is needed."
        )

    @staticmethod
    def _build_verification_section(
        triples: list[dict[str, object]],
        nodes: list[dict[str, object]],
        limit: int = 4,
    ) -> str:
        lines: list[str] = []
        seen: set[str] = set()

        for triple in triples:
            subject = str(triple.get("subject", "")).strip()
            predicate = str(triple.get("predicate", "")).strip()
            obj = str(triple.get("object", "")).strip()
            if not (subject or predicate or obj):
                continue

            parts = [f"({subject}, {predicate}, {obj})"]

            subject_id = str(triple.get("subject_id", "")).strip()
            object_id = str(triple.get("object_id", "")).strip()
            if subject_id or object_id:
                id_bits = ", ".join(
                    bit
                    for bit in (
                        f"s={subject_id}" if subject_id else "",
                        f"o={object_id}" if object_id else "",
                    )
                    if bit
                )
                if id_bits:
                    parts.append(f"[{id_bits}]")

            rel_props = triple.get("relationship_properties", {})
            if isinstance(rel_props, dict):
                source_doc = str(
                    rel_props.get("source_doc", "") or rel_props.get("source", "") or ""
                ).strip()
                page_range = str(rel_props.get("page_range", "")).strip()
                provenance_bits = [bit for bit in (source_doc, page_range) if bit]
                if provenance_bits:
                    parts.append(f"<{' | '.join(provenance_bits)}>")

            line = " ".join(parts)
            if line in seen:
                continue
            seen.add(line)
            lines.append(f"- {line}")
            if len(lines) >= limit:
                break

        if not lines:
            for node in nodes:
                text = str(node.get("text", "")).strip()
                node_id = str(node.get("node_id", "")).strip()
                labels = node.get("labels", [])
                label_text = (
                    ", ".join(str(label) for label in labels)
                    if isinstance(labels, list)
                    else ""
                )
                if not text:
                    continue
                detail = f"({text})"
                if label_text:
                    detail += f" [{label_text}]"
                if node_id:
                    detail += f" [id={node_id}]"
                if detail in seen:
                    continue
                seen.add(detail)
                lines.append(f"- {detail}")
                if len(lines) >= limit:
                    break

        if not lines:
            return (
                "Verifica nel grafo:\n"
                "- Nessuna evidenza strutturata recuperata da mostrare in "
                "modo affidabile."
            )

        return "Verifica nel grafo:\n" + "\n".join(lines)

    @staticmethod
    def _extract_context_highlights(
        query: str, context: str, limit: int = 4
    ) -> list[str]:
        tokens = set(KGRAGAgent._extract_salient_terms(query=query, context=context))

        highlights: list[str] = []
        seen: set[str] = set()

        for raw_line in context.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            lowered = line.lower()
            if not any(token in lowered for token in tokens):
                continue

            if line in seen:
                continue

            seen.add(line)
            highlights.append(line)
            if len(highlights) >= limit:
                break

        if highlights:
            return highlights

        for raw_line in context.splitlines():
            line = raw_line.strip()
            if not line or line in seen:
                continue
            seen.add(line)
            highlights.append(line)
            if len(highlights) >= limit:
                break

        return highlights

    @staticmethod
    def _triple_summaries(
        triples: list[dict[str, object]], query: str, limit: int = 5
    ) -> list[str]:
        if not triples:
            return []

        query_terms = KGRAGAgent._extract_salient_terms_from_text(query)
        focus_terms = {term for term in query_terms if term}

        matched: list[str] = []
        fallback: list[str] = []

        for triple in triples:
            subject = str(triple.get("subject", "")).strip()
            predicate = str(triple.get("predicate", "")).strip()
            obj = str(triple.get("object", "")).strip()
            if not (subject or predicate or obj):
                continue

            summary = f"({subject}, {predicate}, {obj})"
            fallback.append(summary)

            haystack = f"{subject} {predicate} {obj}".lower()
            if any(term in haystack for term in focus_terms):
                matched.append(summary)
                if len(matched) >= limit:
                    break

        if matched:
            return matched

        return fallback[:limit]

    @staticmethod
    def _extract_salient_terms_from_triples(
        triples: list[dict[str, object]],
    ) -> list[str]:
        terms: list[str] = []
        seen: set[str] = set()

        for triple in triples:
            for field in ("subject", "predicate", "object"):
                raw_value = str(triple.get(field, "")).strip()
                if not raw_value:
                    continue
                for token in re.findall(r"\b[A-Z][A-Z0-9/&.-]{1,}\b", raw_value):
                    lowered = token.lower()
                    if lowered in seen:
                        continue
                    seen.add(lowered)
                    terms.append(lowered)

        return terms[:16]

    @staticmethod
    def _extract_salient_terms_from_text(text: str) -> list[str]:
        terms: list[str] = []
        seen: set[str] = set()

        for token in re.findall(r"\b[A-Z][A-Z0-9/&.-]{1,}\b", text):
            lowered = token.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            terms.append(lowered)

        return terms[:16]

    @staticmethod
    def _extract_salient_terms(query: str, context: str) -> list[str]:
        terms: list[str] = []
        seen: set[str] = set()

        STOPWORDS = {
            "parlami",
            "quali",
            "sono",
            "sue",
            "sui",
            "suo",
            "della",
            "delle",
            "degli",
            "dei",
            "con",
            "e",
            "le",
            "il",
            "lo",
            "la",
            "i",
            "gli",
            "un",
            "una",
            "in",
            "per",
            "di",
            "che",
            "da",
            "su",
            "al",
            "alla",
        }

        def add_term(value: str) -> None:
            normalized = value.strip().lower()
            if len(normalized) < 3:
                return
            if normalized in seen:
                return
            if normalized in STOPWORDS:
                return
            seen.add(normalized)
            terms.append(normalized)

        # capture capitalized tokens (acronyms, proper nouns)
        for token in re.findall(r"\b[A-Z][A-Z0-9/&.-]{1,}\b", query):
            add_term(token)
        for token in re.findall(r"\b[A-Z][A-Za-z0-9/&.-]{2,}\b", query):
            add_term(token)

        # also capture common words (lowercase) of length >=3, excluding stopwords
        for token in re.findall(r"\b[\wÀ-ÖØ-öø-ÿ'/-]{3,}\b", query, flags=re.UNICODE):
            add_term(token)

        for line in context.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            for token in re.findall(r"\b[A-Z][A-Z0-9/&.-]{1,}\b", stripped):
                add_term(token)
            for token in re.findall(r"\b[A-Z][A-Za-z0-9/&.-]{2,}\b", stripped):
                add_term(token)
            for token in re.findall(
                r"\b[\wÀ-ÖØ-öø-ÿ'/-]{3,}\b", stripped, flags=re.UNICODE
            ):
                add_term(token)

        return terms[:12]

    def invoke(self, question: str) -> dict:
        start = time.perf_counter()
        initial_state = {
            "question": question,
            "run_id": str(uuid.uuid4()),
            "rewrite_count": 0,
        }
        try:
            output = self.graph.invoke(
                initial_state, config={"recursion_limit": self.config.recursion_limit}
            )
        except GraphRecursionError:
            logger.warning(
                "Graph recursion limit reached (limit=%d) for question: %s",
                self.config.recursion_limit,
                question,
            )
            if LLMManager._detect_query_language(question) == "it":
                output = {
                    "answer": (
                        "Il processo ha raggiunto il limite di ricorsione dell'agente prima di convergere. "
                        "Prova con una domanda piu specifica o aumenta --recursion-limit."
                    )
                }
            else:
                output = {
                    "answer": (
                        "The agent hit its recursion limit before converging. "
                        "Try a more specific question or raise --recursion-limit."
                    )
                }
        latency_ms = (time.perf_counter() - start) * 1000.0
        output["latency_ms"] = latency_ms
        return output
