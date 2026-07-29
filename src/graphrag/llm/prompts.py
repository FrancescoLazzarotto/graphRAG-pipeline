from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from graphrag.config import AgentConfig, OUTPUT_COMPLEXITY, OUTPUT_TONE


class PromptLibrary:
    # WP5: written in the target language on purpose — an Italian instruction
    # holds an Italian answer far better than an English sentence asking for
    # Italian, especially when the retrieved context is mostly English.
    LANGUAGE_DIRECTIVES = {
        "it": (
            "Rispondi SEMPRE in italiano, anche quando il contesto e in inglese: "
            "traduci le evidenze invece di cambiare lingua. Restano nella lingua "
            "originale solo i nomi propri, i titoli dei documenti e i termini "
            "tecnici privi di traduzione corrente."
        ),
        "en": (
            "ALWAYS answer in English, even when the context is written in "
            "Italian: translate the evidence instead of switching language. Only "
            "proper names, document titles and technical terms with no current "
            "translation stay in the original language."
        ),
    }
    # WP3 inside WP5. The language directive above is repeated as the very last
    # line of the prompt, which is the position models obey; on the first live
    # runs it beat the "copy word for word" rule three times out of eight, and
    # the definitions of SEeD, CEFF and metabolizzazione came back translated —
    # accurate prose between guillemets that the quote gate then had to strip.
    # The exception has to travel with the directive that overrides it.
    QUOTE_LANGUAGE_EXCEPTIONS = {
        "it": (
            " Unica eccezione: il passaggio citato fra «...» va copiato nella "
            "lingua originale della fonte, senza tradurlo, perche' una citazione "
            "tradotta non e' piu' una citazione; la traduzione va subito dopo, "
            "fuori dalle virgolette."
        ),
        "en": (
            " One exception: the passage quoted between «...» must be copied in "
            "the source's original language, untranslated, because a translated "
            "quotation is no longer a quotation; put your translation "
            "immediately after, outside the guillemets."
        ),
    }
    LANGUAGE_REINFORCEMENTS = {
        "it": (
            "VINCOLO ASSOLUTO: la risposta precedente era nella lingua sbagliata. "
            "Scrivi TUTTA la risposta in italiano, dalla prima all'ultima parola. "
        ),
        "en": (
            "ABSOLUTE CONSTRAINT: the previous answer was in the wrong language. "
            "Write the ENTIRE answer in English, from first word to last. "
        ),
    }

    @staticmethod
    def language_directive(
        language: str,
        reinforced: bool = False,
        quote_exception: bool = False,
    ) -> str:
        """Return the answer-language constraint, written in that language.

        Args:
            language: ``"it"`` or ``"en"``; anything else yields an empty string.
            reinforced: Prefix the stronger wording used on the retry that
                follows a wrong-language answer.
            quote_exception: Add the WP3 carve-out that keeps a quoted passage
                in the source's language. Only for definitional questions: on
                every other answer there is nothing to quote and the clause
                would just invite the model to leave English prose in.

        Returns:
            The directive, or an empty string when the language is unknown.
        """
        key = str(language or "").lower()
        directive = PromptLibrary.LANGUAGE_DIRECTIVES.get(key)
        if not directive:
            return ""
        if quote_exception:
            directive += PromptLibrary.QUOTE_LANGUAGE_EXCEPTIONS[key]
        if reinforced:
            return PromptLibrary.LANGUAGE_REINFORCEMENTS[key] + directive
        return directive

    @staticmethod
    def answer_prompt(
        config: AgentConfig,
        language: str | None = None,
        reinforce_language: bool = False,
        definitional: bool = False,
    ) -> ChatPromptTemplate:
        """Build the answer prompt.

        Args:
            config: Agent configuration; ``complexity`` drives answer depth and
                ``cite_evidence`` the citation protocol.
            language: Target answer language (``"it"``/``"en"``). ``None`` keeps
                the prompt byte-identical to the pre-WP5 one, which is what
                existing baselines and gold runs must keep seeing.
            reinforce_language: Use the stronger constraint (retry after a
                wrong-language answer). Ignored when ``language`` is ``None``.
            definitional: The question asks what something is (WP3). Adds the
                quote-then-explain structure the expert asked for.

        Returns:
            The chat prompt template with ``question`` and ``context`` slots.
        """
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
        language_block = (
            PromptLibrary.language_directive(
                language,
                reinforced=reinforce_language,
                quote_exception=definitional,
            )
            if language
            else ""
        )
        if language_block:
            system_message += " " + language_block
        if config.cite_evidence:
            system_message += (
                " Evidence items in the context are numbered: reference them by "
                "their id so each specific claim stays traceable to its source "
                "document."
            )

        # An English heading on top of an Italian answer is exactly the kind of
        # language leak WP5 removes, so the title follows the answer language.
        limits_title = (
            "Limiti e affidabilità" if language == "it" else "Limits and confidence"
        )
        if config.always_include_limits:
            limits_block = (
                f"Always include a short section titled '{limits_title}' "
                "assessing how strong the supporting evidence is, in at most "
                "three sentences: it closes the answer, it is not the answer. "
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
                f"'{limits_title}'. "
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
                "When consecutive sentences rest on the same evidence, cite it "
                "once for the whole passage instead of repeating it. "
                "Do not write a source list at the end: it is generated automatically. "
            )
        else:
            evidence_block = (
                no_inline_block
                + "When possible, add a short 'Evidence in graph' section with the "
                "exact node or triple names that support the answer. "
            )

        if config.complexity is OUTPUT_COMPLEXITY.HIGH:
            # WP2: "1-2 short paragraphs" contradicts a HIGH complexity setting
            # and is what turns answers into abstract summaries — a summary drops
            # exactly the figures, names and article numbers the expert asks for.
            depth_block = (
                "If context has at least some factual evidence, provide the best "
                "grounded answer possible, developing every point the evidence "
                "supports across several paragraphs. "
                "Avoid a checklist style unless the user explicitly asks for a list. "
                "Stay concrete: use the figures, proper names, years, percentages "
                "and article or standard numbers that appear in the evidence, and "
                "never generalise when a specific one is available — write 'a 65% "
                "impact reduction at Terra Madre Salone del Gusto', not 'a "
                "significant impact reduction'. "
            )
        else:
            depth_block = (
                "If context has at least some factual evidence, provide the best "
                "grounded answer possible in 1-2 short paragraphs. "
                "Avoid a checklist style unless the user explicitly asks for a list. "
            )

        # WP3: a definition *is* its wording, and the graph channel systematically
        # replaced it with relations — the answer on SEeD was built entirely out
        # of triples and never said "Systemic Event Design", which was sitting in
        # the corpus. The "only if it is there" clause is not politeness: the
        # instruction to quote is also an invitation to invent a quote, and the
        # quote gate downstream strips the guillemets off anything invented.
        definition_block = ""
        if definitional:
            definition_block = (
                "This question asks what something is. Open the answer with the "
                "source's own definition between «guillemets», followed by its "
                "reference tag — including the expansion of an acronym when the "
                "source gives one. "
                # The first live run failed exactly here: the model reordered
                # the source's words inside the guillemets, which reads as a
                # quotation and is not one. The gate strips those guillemets, so
                # the instruction has to be about copying, not about quoting.
                "Inside the guillemets copy the source word for word, in its "
                "original order, changing nothing: no reordering, no rewording, "
                "no shortening except a [...] for an omitted middle. "
                # The corpus is bilingual and the answer language is pinned to
                # the question's, so a definition taken from an English document
                # and answered in Italian can never be verbatim. Quoting in the
                # original and translating outside the guillemets is the only
                # form that satisfies both constraints.
                "This is the one place where you do not translate: copy the "
                "passage in the language the source wrote it in, then give your "
                "translation immediately after, outside the guillemets. "
                "When you cannot copy a passage exactly, or when no passage "
                "defines the term, use no guillemets at all: say in one sentence "
                "that the context carries no explicit definition and answer, in "
                "your own words and as a complete sentence, from what the context "
                "does say. Never invent a definition to quote. "
                "Then explain the definition and what it means in practice, and "
                "only after that use the graph facts, as a complement to the "
                "definition and never as a replacement for it. "
            )

        human_message_template = (
            f"Target audience: {config.target_audience}.\n"
            f"{tone_map[config.tone]}\n{complexity_map[config.complexity]}\n"
            f"{structured}\n"
            "Question:\n{question}\n\n"
            "Context:\n{context}\n\n"
            + definition_block
            + depth_block
            + limits_block
            + evidence_block
            + "State that context is insufficient only when context is empty or "
            "lacks factual evidence."
            # Repeated as the last line: the instruction closest to the
            # generation point is the one models follow when the context pulls
            # the other way.
            + (f"\n\n{language_block}" if language_block else "")
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
    def followup_rewrite_prompt(config: AgentConfig) -> ChatPromptTemplate:
        """Make an elliptical follow-up self-contained, for retrieval only.

        WP7 (`docs/demo_quality_plan_2026-07.md` §9.3). The output never reaches
        the answer prompt: it only feeds the retriever, so the instructions
        optimise for search terms, not for phrasing. The "repeat it unchanged"
        escape hatch matters — the detector is allowed to fire on a question
        that turns out to need nothing, and the model must be free to say so.
        """
        return ChatPromptTemplate.from_template(
            "You rewrite a follow-up question so that it can be understood on its own, "
            "outside the conversation.\n"
            "Topics active in this conversation: {entities}\n"
            "Previous question: {previous_question}\n"
            "Follow-up question: {question}\n\n"
            "Rules:\n"
            "- Keep the user's intent and the user's language.\n"
            "- Resolve pronouns and implicit references using the active topics above.\n"
            "- Use a topic only when the follow-up actually refers to it, and ignore "
            "the others: an unrelated topic pulls retrieval away from the question.\n"
            "- Add only the missing context. Do not add facts, and do not answer.\n"
            "- If the follow-up already stands on its own, repeat it unchanged.\n"
            "- Reply with the rewritten question only, on a single line.\n\n"
            "Rewritten question:"
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
