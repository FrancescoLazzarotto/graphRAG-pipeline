from __future__ import annotations

import json
import time
import uuid

from langgraph.graph import END, START, StateGraph

from graphrag.agent.cache import LRUCache
from graphrag.agent.compression import ContextCompressor
from graphrag.config import AgentConfig
from graphrag.kg.retriever import KGRetriever
from graphrag.llm.manager import LLMManager
from graphrag.llm.prompts import PromptLibrary
from graphrag.types import RAGState


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
        self.compressor = ContextCompressor(config.max_content_tokens, config.token_estimator_ratio)
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
                return {"sub_questions": [str(item).strip() for item in parsed if str(item).strip()]}
        except json.JSONDecodeError:
            pass

        return {"sub_questions": [line.strip("-• \t") for line in text.splitlines() if line.strip()]}

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
        rewritten = str(output.content if hasattr(output, "content") else output).strip()
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
        if not self.llm:
            return {"chosen_retrieval_mode": "TEXT"}

        prompt = PromptLibrary.adaptive_router_prompt(self.config)
        rendered = prompt.invoke({"question": question})
        model = self.llm.load_llm()
        output = model.invoke(rendered)
        mode = str(output.content if hasattr(output, "content") else output).strip().upper()

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

        if self.kg_retriever:
            retrieved_data = self.kg_retriever.retrieve(query)
            context = str(retrieved_data.get("context_text", ""))
        else:
            retrieved_data = {}
            context = ""

        compressed_context = self.compressor.compress(context)
        triples = retrieved_data.get("triples", []) if isinstance(retrieved_data, dict) else []
        neighbors = retrieved_data.get("neighbors", []) if isinstance(retrieved_data, dict) else []
        subgraph = retrieved_data.get("subgraph", []) if isinstance(retrieved_data, dict) else []
        shortest_path = retrieved_data.get("shortest_path", []) if isinstance(retrieved_data, dict) else []

        result = {
            "text_context": compressed_context,
            "kg_triples": triples if isinstance(triples, list) else [],
            "retrieved_neighbors_count": len(neighbors) if isinstance(neighbors, list) else 0,
            "retrieved_subgraph_count": len(subgraph) if isinstance(subgraph, list) else 0,
            "retrieved_shortest_path_count": len(shortest_path) if isinstance(shortest_path, list) else 0,
        }

        if self.cache:
            self.cache.put(query, mode, result)

        return result

    def _grade(self, state: RAGState) -> dict:
        return {"relevance": "relevant"}

    def _generate(self, state: RAGState) -> dict:
        query = state.get("question", "")
        context = state.get("text_context", "")

        if self.llm:
            result = self.llm.generate(query=query, context=context, config=self.config)
            return {"answer": result.get("answer", "")}

        return {"answer": "LLM not available."}

    def invoke(self, question: str) -> dict:
        start = time.perf_counter()
        initial_state = {
            "question": question,
            "run_id": str(uuid.uuid4()),
            "rewrite_count": 0,
        }
        output = self.graph.invoke(initial_state)
        latency_ms = (time.perf_counter() - start) * 1000.0
        output["latency_ms"] = latency_ms
        return output
