from __future__ import annotations

import time
import uuid

from graphrag.config import AgentConfig
from graphrag.llm.manager import LLMManager
from graphrag.text_rag.pipeline import StandardTextRAGPipeline


class StandardRAGAgent:
    """Simple text-only RAG agent built on top of StandardTextRAGPipeline."""

    def __init__(
        self,
        pipeline: StandardTextRAGPipeline,
        config: AgentConfig | None = None,
        llm: LLMManager | None = None,
        top_k: int = 4,
        include_sources: bool = True,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        self.pipeline = pipeline
        self.config = config or AgentConfig()
        self.llm = llm
        self.top_k = top_k
        self.include_sources = include_sources

        if self.llm is not None and self.config.llm_warmup:
            self.llm.warmup()

    def invoke(self, question: str) -> dict:
        start = time.perf_counter()

        context = self.pipeline.build_context(
            query=question,
            top_k=self.top_k,
            include_sources=self.include_sources,
        )
        retrieved = self.pipeline.retrieve(query=question, top_k=self.top_k)

        if self.llm is not None:
            generated = self.llm.generate(query=question, context=context, config=self.config)
            answer = generated.get("answer", "")
        else:
            if context:
                answer = f"Retrieved context preview:\n{context[:800]}"
            else:
                answer = "No context retrieved."

        latency_ms = (time.perf_counter() - start) * 1000.0

        return {
            "run_id": str(uuid.uuid4()),
            "question": question,
            "answer": answer,
            "text_context": context,
            "retrieved_text_chunks_count": len(retrieved),
            "kg_triples": [],
            "retrieved_neighbors_count": 0,
            "retrieved_subgraph_count": 0,
            "retrieved_shortest_path_count": 0,
            "sub_questions": [question],
            "reflection_passed": True,
            "latency_ms": latency_ms,
        }
