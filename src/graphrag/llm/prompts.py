from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from graphrag.config import AgentConfig, OUTPUT_COMPLEXITY, OUTPUT_TONE


class PromptLibrary:
    @staticmethod
    def answer_prompt(config: AgentConfig) -> ChatPromptTemplate:
        if config.answer_prompt:
            return ChatPromptTemplate.from_template(config.answer_prompt)

        tone_map = {
            OUTPUT_TONE.TECHNICAL: "Use precise technical terminology.",
            OUTPUT_TONE.SIMPLIFIED: "Explain in simple, accessible terms.",
            OUTPUT_TONE.FORMAL: "Use a formal, academic register.",
        }

        complexity_map = {
            OUTPUT_COMPLEXITY.LOW: "Keep the answer brief (2-3 sentences).",
            OUTPUT_COMPLEXITY.MEDIUM: "Provide a well-structured paragraph.",
            OUTPUT_COMPLEXITY.HIGH: "Provide a thorough, multi-paragraph analysis.",
        }

        structured = ""
        if config.use_structured_response:
            structured = (
                "\nAnswer using this structure:\n"
                "## Key Concepts\n## Relationships\n## Reasoning Chain\n## Conclusions\n"
            )

        # System message with explicit response rules
        system_message = (
            "You are a knowledge graph assistant. Answer using ONLY the provided context. "
            "If context does not answer the question, state this plainly. "
            "Do not invent or generate content outside the context. "
            "Preserve all entity names exactly as given. "
            "Respond in the same language as the question (English or Italian), "
            "even when the context is written in the other language: translate "
            "the evidence into the question's language, never switch language. "
            "Prefer a natural, human explanation over a mechanical list. "
            "If you mention a fact, tie it to an exact node, triple, or other "
            "explicit evidence from the context."
        )
        if config.cite_evidence:
            system_message += (
                " Evidence items in the context are numbered: reference them by "
                "their id so each specific claim stays traceable to its source "
                "document."
            )

        if config.always_include_limits:
            limits_block = (
                "Always include a short section titled 'Limits and confidence' "
                "assessing how strong the supporting evidence is. "
            )
            # Only meaningful without the citation protocol below, which replaces
            # the trailing evidence section with per-claim reference tags.
            no_inline_block = (
                "Keep the main paragraphs free of inline triple citations in "
                "parentheses; cite nodes and triples only in the dedicated "
                "evidence section. "
            )
        else:
            limits_block = (
                "If context is sparse, include a short section titled "
                "'Limits and confidence'. "
            )
            no_inline_block = ""

        if config.cite_evidence:
            # Deliberately restrictive: a tag on every sentence reads as noise
            # and stops carrying information. Citations belong on claims a
            # reader could want to check.
            evidence_block = (
                "Evidence items in the context are numbered: [S1], [S2], ... for "
                "source passages and [T1], [T2], ... for knowledge-graph facts. "
                "Put the id of the evidence you used in square brackets at the end "
                "of the sentence it supports. "
                "Cite only claims carrying specific content: figures, percentages, "
                "dates, proper names, article or standard numbers, definitions, and "
                "statements attributable to an author or a document. "
                "Do not cite generic, connective or summarising sentences, and use at "
                "most one tag per sentence. "
                "When several evidence items support the same claim, cite the single "
                "most specific one instead of stacking ids: never put more than two "
                "ids in a tag. "
                "Never write an id that is not in the context, and never cite the "
                "entity sections, which carry no source. "
                "Do not write a source list at the end: it is generated automatically. "
            )
        else:
            evidence_block = (
                no_inline_block
                + "When possible, add a short 'Evidence in graph' section with the "
                "exact node or triple names that support the answer. "
            )

        human_message_template = (
            f"Target audience: {config.target_audience}.\n"
            f"{tone_map[config.tone]}\n{complexity_map[config.complexity]}\n"
            f"{structured}\n"
            "Question:\n{question}\n\n"
            "Context:\n{context}\n\n"
            "If context has at least some factual evidence, provide the best "
            "grounded answer possible in 1-2 short paragraphs. "
            "Avoid a checklist style unless the user explicitly asks for a list. "
            + limits_block
            + evidence_block
            + "State that context is insufficient only when context is empty or "
            "lacks factual evidence."
        )

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("human", human_message_template),
            ]
        )

    @staticmethod
    def rewrite_prompt(config: AgentConfig) -> ChatPromptTemplate:
        if config.rewrite_prompt:
            return ChatPromptTemplate.from_template(config.rewrite_prompt)
        return ChatPromptTemplate.from_template(
            "Rewrite this question to improve retrieval over the target knowledge base. "
            "Add relevant synonyms or domain terms.\n\n"
            "Original: {question}\nRewritten:"
        )

    @staticmethod
    def decomposition_prompt(config: AgentConfig) -> ChatPromptTemplate:
        if config.decomposition_prompt:
            return ChatPromptTemplate.from_template(config.decomposition_prompt)
        return ChatPromptTemplate.from_template(
            "Break this complex question into 2-4 simpler, self-contained sub-questions "
            "that together cover the full scope of the original.\n"
            "Return ONLY a JSON array of strings.\n\n"
            "Question: {question}\n\nSub-questions:"
        )

    @staticmethod
    def reflection_prompt(config: AgentConfig) -> ChatPromptTemplate:
        if config.reflection_prompt:
            return ChatPromptTemplate.from_template(config.reflection_prompt)
        return ChatPromptTemplate.from_template(
            "You are a grounding verifier. Check whether the answer is faithful to "
            "the provided context. Look for hallucinations, unsupported claims, or "
            "logical errors.\n\n"
            "Context:\n{context}\n\n"
            "Answer:\n{answer}\n\n"
            "Respond with a JSON object:\n"
            '{{"passed": true/false, "confidence": 0.0-1.0, "feedback": "..."}}'
        )

    @staticmethod
    def adaptive_router_prompt(config: AgentConfig) -> ChatPromptTemplate:
        if config.adaptive_router_prompt:
            return ChatPromptTemplate.from_template(config.adaptive_router_prompt)
        return ChatPromptTemplate.from_template(
            "Given this question, choose the best retrieval strategy.\n"
            "Options:\n"
            "- TEXT: factual lookup, keyword-heavy queries\n"
            "- KG: relationship or reasoning queries\n"
            "- HYBRID: complex questions needing both facts and relationships\n"
            "- MULTIHOP: questions requiring chain reasoning across multiple concepts\n\n"
            "Question: {question}\n\n"
            "Respond with ONLY one word: TEXT, KG, HYBRID, or MULTIHOP."
        )

    @staticmethod
    def refusal_retry_prompt(language: str = "en") -> ChatPromptTemplate:
        """Stricter prompt used for the single fallback attempt after a refusal.

        Lives here (not inline in the backend) so vLLM and local HF keep
        rendering identical prompts — the invariant the whole PromptLibrary
        exists for.
        """
        if language == "it":
            return ChatPromptTemplate.from_template(
                "Usa solo il contesto fornito per rispondere in modo naturale e conciso alla domanda. "
                "Rispondi SEMPRE in italiano, anche se il contesto è in inglese: traduci le evidenze. "
                "Evita un elenco meccanico; preferisci una breve spiegazione in 1-2 paragrafi. "
                "Se possibile, aggiungi una piccola sezione 'Evidence in graph' con i nomi esatti dei nodi o dei tripletti che supportano la risposta. "
                "Contesto:\n{context}\n\nDomanda:\n{question}\n\nRisposta:"
            )
        return ChatPromptTemplate.from_template(
            "Use only the provided context to answer the question naturally and concisely. "
            "ALWAYS answer in English, even if the context is in Italian: translate the evidence. "
            "Avoid a mechanical list; prefer a short 1-2 paragraph explanation. "
            "When possible, add a short 'Evidence in graph' section with the exact node or triple names that support the answer. "
            "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        )

    @staticmethod
    def multihop_steer_prompt() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template(
            "You are exploring a knowledge graph to answer a question.\n"
            "So far you have gathered:\n{hop_history}\n\n"
            "Question: {question}\n\n"
            "Based on what you know so far, do you have enough information?\n"
            'Respond with JSON: {"enough": true/false, "next_entities": ["..."], '
            '"reasoning": "..."}'
        )
