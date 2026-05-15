from __future__ import annotations

import json
import logging
import re
import time
import uuid

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from graphrag.agent.cache import LRUCache
from graphrag.agent.compression import ContextCompressor
from graphrag.config import AgentConfig
from graphrag.kg.retriever import KGRetriever
from graphrag.llm.manager import LLMManager
from graphrag.llm.prompts import PromptLibrary
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

        def route_condition(state: RAGState):
            _ = state.get("chosen_retrieval_mode", "TEXT")
            return "retrieve"

        builder.add_conditional_edges("route", route_condition)
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

        return {
            "sub_questions": [
                line.strip("-• \t") for line in text.splitlines() if line.strip()
            ]
        }

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
        if rewrite_count > 2:
            return {
                "rewritten_question": rewritten,
                "rewrite_count": rewrite_count,
                "relevance": "relevant",
            }
        return {"rewritten_question": rewritten, "rewrite_count": rewrite_count}

    def _adaptive_route(self, state: RAGState) -> dict:
        question = state.get("question", "").strip()
        if not self.config.enable_adaptive_routing_step:
            return {"chosen_retrieval_mode": "HYBRID"}

        if not self.llm:
            return {"chosen_retrieval_mode": "TEXT"}

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
        query = state.get("rewritten_question") or state.get("question", "")
        mode = state.get("chosen_retrieval_mode", "HYBRID")

        if self.cache:
            hit = self.cache.get(query, mode)
            if hit is not None:
                return hit

        # Choose retrieval strategy based on adaptive routing mode
        retrieved_data: dict = {}
        context = ""
        if self.kg_retriever:
            mm = str(mode).upper() if mode is not None else "HYBRID"
            if mm == "TEXT":
                # Text-only: return only the textual context (no KG evidence)
                context = str(self.kg_retriever.retrieve_context(query))
                retrieved_data = {
                    "query": query,
                    "context_text": context,
                    "nodes": [],
                    "triples": [],
                    "neighbors": [],
                    "subgraph": [],
                    "shortest_path": [],
                }
            elif mm == "KG":
                # KG-only: prioritize structural KG outputs and suppress free-text context
                raw = self.kg_retriever.retrieve(query)
                raw["context_text"] = ""
                retrieved_data = raw
                context = ""
            elif mm == "MULTIHOP":
                # Multi-hop: use the multi_hop API to extract a broader subgraph
                subgraph = self.kg_retriever.multi_hop(
                    entity=query,
                    hops=self.config.hops,
                    limit=self.config.subgraph_limit,
                )
                try:
                    context = str(self.kg_retriever.kg_store.triples_to_text(subgraph))
                except Exception:
                    context = ""
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
                # HYBRID and default: use the full retrieve implementation
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
            self.cache.put(query, mode, result)

        return result

    def _grade(self, state: RAGState) -> dict:
        nodes_count = int(state.get("retrieved_nodes_count", 0) or 0)
        triples_count = len(state.get("kg_triples", []) or [])
        subgraph_count = int(state.get("retrieved_subgraph_count", 0) or 0)
        shortest_path_count = int(state.get("retrieved_shortest_path_count", 0) or 0)

        # Stronger semantic gating: ensure retrieved KG items actually match the
        # salient terms in the query/context instead of accepting any hit.
        evidence_units = (
            nodes_count + triples_count + subgraph_count + shortest_path_count
        )
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

        # Determine relevance: require at least one semantic match, and either
        # multiple matches or a reasonable match ratio to accept as relevant.
        match_ratio = matched / max(1, evidence_units)
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
        nodes_count = int(state.get("retrieved_nodes_count", 0) or 0)
        triples_count = len(state.get("kg_triples", []) or [])
        subgraph_count = int(state.get("retrieved_subgraph_count", 0) or 0)
        shortest_path_count = int(state.get("retrieved_shortest_path_count", 0) or 0)

        evidence_units = (
            nodes_count + triples_count + subgraph_count + shortest_path_count
        )

        if evidence_units == 0:
            return {
                "answer": (
                    "The provided context is insufficient to generate a grounded response. "
                    "Please provide additional context or a more specific question."
                )
            }

        sparse_context = evidence_units <= 2
        effective_query = query
        if sparse_context:
            effective_query = (
                query
                + "\n\nIstruzione: rispondi direttamente usando solo il contesto disponibile. "
                + "Se il contesto e limitato, fornisci comunque la migliore risposta possibile e aggiungi una breve sezione 'Limiti e fiducia'."
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
                )
            verification_section = self._build_verification_section(
                triples=state.get("kg_triples", []) or [],
                nodes=state.get("retrieved_nodes", []) or [],
            )
            if verification_section:
                answer = answer.rstrip() + "\n\n" + verification_section

            return {"answer": answer}

        return {"answer": "LLM not available."}

    @staticmethod
    def _looks_like_generic_refusal(answer: str) -> bool:
        if not answer.strip():
            return True

        lowered = answer.lower()
        refusal_markers = (
            "context is insufficient",
            "provide additional context",
            "specific details regarding the question",
            "specific details regarding the context",
            "without a specific question",
            "without a specific question or detailed context",
            "detailed context",
            "without these elements",
            "could you specify the question",
            "serve specific details",
            "non ho abbastanza contesto",
            "contesto fornito e insufficiente",
            "contesto insufficiente",
            "ho bisogno di ulteriori informazioni",
            "crucial to first establish",
            "not feasible",
            "challenging to construct",
            "without these elements, crafting",
            "the current context does not provide sufficient information",
        )
        return any(marker in lowered for marker in refusal_markers)

    def _should_replace_with_fallback(
        self,
        answer: str,
        query: str,
        context: str,
        triples: list[dict[str, object]],
        sparse_context: bool,
    ) -> bool:
        if self._looks_like_generic_refusal(answer):
            return True

        if not sparse_context:
            return False

        salient_terms = self._extract_salient_terms(query=query, context=context)
        if not salient_terms:
            return False

        answer_lower = answer.lower()
        has_salient_reference = any(term in answer_lower for term in salient_terms)
        if has_salient_reference:
            return False

        if self._triple_summaries(triples, query=query):
            triple_terms = self._extract_salient_terms_from_triples(triples)
            if triple_terms and any(term in answer_lower for term in triple_terms):
                return False

        meta_markers = (
            "context",
            "question",
            "details",
            "specific",
            "grounded",
            "information",
            "analysis",
            "technical",
            "factual basis",
        )
        meta_hits = sum(1 for marker in meta_markers if marker in answer_lower)

        # Balanced fallback: intercept meta-discussions but allow short answers.
        # Trigger when sparse context AND high meta-marker signal (3+).
        return meta_hits >= 3

    @staticmethod
    def _build_sparse_fallback_answer(
        query: str, context: str, triples: list[dict[str, object]]
    ) -> str:
        triple_summaries = KGRAGAgent._triple_summaries(triples, query=query)
        highlights = KGRAGAgent._extract_context_highlights(
            query=query, context=context
        )

        if triple_summaries or highlights:
            evidence_block = "\n".join(
                f"- {line}" for line in (triple_summaries or highlights)
            )
            return (
                "Dal contesto disponibile emergono i seguenti elementi rilevanti. "
                "La risposta e quindi parziale, ma contiene le evidenze trovate nel grafo.\n\n"
                "Limiti e fiducia:\n"
                "Il contesto e limitato, quindi non posso inferire l'intero perimetro tematico con alta fiducia.\n\n"
                "Evidenze rilevanti:\n"
                f"{evidence_block}"
            )

        return (
            "Il contesto disponibile e troppo scarno per costruire una risposta affidabile. "
            "Serve un recupero piu specifico o piu evidenza dal grafo."
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
        if not tokens:
            tokens = {"fao", "who"}

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
        focus_terms.update({"fao", "who"})

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
            output = {
                "answer": (
                    "Il processo ha raggiunto il limite di ricorsione dell'agente prima di convergere. "
                    "Prova con una domanda piu specifica o aumenta --recursion-limit."
                )
            }
        latency_ms = (time.perf_counter() - start) * 1000.0
        output["latency_ms"] = latency_ms
        return output
