"""demo expert reads an answer and
asks a follow-up — "mi indichi le strategie nel settore vino" — where the topic
comes from the previous answer, not from the question. Retrieval receives that
question in isolation and searches for the wrong thing.

This module holds the state needed to make such a question self-contained again:
the entities the conversation is actually about. Three hard boundaries:

* **Never a source of facts.** Memory carries *entities* and the plain text of
  what was already said, never citable claims. The groundedness of a turn is
  always computed against the evidence retrieved in that turn; a model that
  could cite something "because it was said earlier" would be self-confirming.
  The transcript below is what makes the second half enforceable rather than
  merely intended: reference tags are stripped out of it, so there is no id in
  the transcript for the model to reuse.
* **Retrieval only, with one exception.** The rewritten question steers
  retrieval; generation still answers the question the user literally typed.
  The exception is the transcript: an expert who writes "hai scritto X, quali?"
  is quoting the assistant, and with no record of its own prose the model read
  that as an unsupported premise and told the expert the claim was false —
  twice, in the session of 2026-09-03, about a sentence it had written fifteen
  minutes earlier. The transcript is carried so the model can recognise its own
  words, and for nothing else.
* **Off unless asked.** No memory object means the previous behaviour, byte for
  byte — gold runs and experiment baselines stay comparable.

The follow-up detector is deterministic on purpose: an LLM classifier here would
add a call, a failure mode and a source of nondeterminism to every turn, on a
decision that a handful of surface markers already settles.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

__all__ = [
    "ActiveEntity",
    "Exchange",
    "ConversationMemory",
]


# Entity names shorter than this, or made only of digits, are noise as retrieval
# seeds.
_MIN_ENTITY_CHARS = 3
_MAX_ENTITY_CHARS = 60
# A document is where an answer came from, not what it was about. The graph
# holds document nodes under their file name, so "SEeD for Change.pdf" reached
# the seed list in two recorded sessions and led it in one of them, spending one
# of only four slots on a name that steers a rewrite towards the file rather
# than the subject.
_DOCUMENT_SUFFIXES = (
    ".pdf", ".doc", ".docx", ".odt", ".rtf",
    ".xls", ".xlsx", ".ods", ".csv",
    ".ppt", ".pptx", ".odp",
    ".txt", ".md", ".json", ".xml", ".html", ".htm",
)

# Reference tags as the answer prompt writes them: "[S1]", "[T12]", and the
# document form "[REPORT MATTM, p. 70]" with its multi-source "[A, p. 1; B, p. 2]"
# variant. They are removed from the transcript so a claim the model made earlier
# cannot come back carrying an id and be recited as if a document supported it.
# "[...]" is the omission marker the definitional prompt asks for and stays.
_REFERENCE_TAG_RE = re.compile(r"\[(?!\.\.\.\])[^\[\]\n]{1,200}\]")

# The generated source list closing an answer: pure citation machinery, the part
# of the text with the highest tag density and the least conversational value.
_SOURCE_LIST_RE = re.compile(
    r"\n\s*\*{0,2}(?:Fonti|Sources)\*{0,2}\s*:\s*\n.*\Z",
    re.IGNORECASE | re.DOTALL,
)

# Characters, not turns: an answer runs from 2.5k to 5k characters, so a turn
# budget would swing by a factor of two. 16k characters is roughly 4k tokens,
# about five stripped answers — comfortable inside a 32k window that also has to
# hold ~3k of retrieved context, and bounded by design because the corpus grows
# and the context block grows with it.
_DEFAULT_TRANSCRIPT_CHARS = 16_000


def _transcript_budget() -> int:
    """Character budget for the transcript, overridable per deployment."""
    raw = os.getenv("GRAPHRAG_TRANSCRIPT_MAX_CHARS", "")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_TRANSCRIPT_CHARS
    return value if value > 0 else _DEFAULT_TRANSCRIPT_CHARS


def _strip_references(answer: str) -> str:
    """The prose of an answer, without the apparatus that makes it citable."""
    text = _SOURCE_LIST_RE.sub("", str(answer or ""))
    text = _REFERENCE_TAG_RE.sub("", text)
    # Removing an inline tag leaves " ." and doubled spaces behind.
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" +([.,;:!?])", r"\1", text)
    return text.strip()


_TOKEN_RE = re.compile(r"[\wÀ-ÿ'’-]+")

# Word units for entity matching. Narrower than `_TOKEN_RE` on purpose:
# apostrophes and hyphens separate here, so "l'economia" contains the word
# "economia" and "sotto-prodotti" contains "prodotti".
_WORD_RE = re.compile(r"[\wÀ-ÿ]+")


def _words(text: str) -> tuple[str, ...]:
    """Lowercased word units of `text`."""
    return tuple(match.lower() for match in _WORD_RE.findall(str(text or "")))


def _contains_span(outer: Sequence[str], inner: Sequence[str]) -> bool:
    """True when `inner` occurs inside `outer` as a run of whole words.

    Substring containment is not usable on entity names: with a 3-character
    floor, "Riso" sits inside "risorse", "Eni" inside "sostenibile" and "tema"
    inside "sistema". Measured on the 2026-07 demo logs, a plain `in` test
    marked roughly a third of the matching names as mentioned when they never
    were.
    """
    if not inner or len(inner) > len(outer):
        return False
    span = tuple(inner)
    width = len(span)
    return any(
        tuple(outer[start : start + width]) == span
        for start in range(len(outer) - width + 1)
    )


@dataclass
class ActiveEntity:
    """A KG entity the conversation has touched, with its recency."""

    name: str
    turn: int
    mentions: int = 1


@dataclass
class Exchange:
    """One completed turn as plain conversational text.

    The answer is stored stripped of reference tags and of its generated source
    list: what stays is what the assistant said, not what it cited.
    """

    question: str
    answer: str


@dataclass
class ConversationMemory:
    """Entities in play for the current session.

    Lives in `st.session_state` for the demo and nowhere else: no persistence
    across sessions, no shared domain memory. Entities older than `window`
    turns are dropped — without decay the seed list grows until it describes
    half the graph and stops discriminating.
    """

    window: int = 3
    max_seed_entities: int = 4
    turn: int = 0
    # Turns that raised before producing an answer. Counted apart from `turn`
    # so a retry after a graph failover does not spend two turns of the decay
    # window on one question; see `observe_failure`.
    failed_turns: int = 0
    active_entities: list[ActiveEntity] = field(default_factory=list)
    last_answer_entities: list[str] = field(default_factory=list)
    last_question: str = ""
    # The conversation as text, oldest first. Unlike `active_entities` this is
    # not subject to the `window` decay: a user can refer to something said six
    # turns ago, and the character budget already bounds it.
    exchanges: list[Exchange] = field(default_factory=list)
    max_transcript_chars: int = field(default_factory=_transcript_budget)

    def reset(self) -> None:
        """Forget the current topic (the 'Nuovo argomento' button)."""
        self.turn = 0
        self.failed_turns = 0
        self.active_entities = []
        self.last_answer_entities = []
        self.last_question = ""
        self.exchanges = []

    def has_context(self) -> bool:
        """Whether anything has been said yet in this session.

        Turn count, not entity count. Entities are observed only from the KG
        channel, and on a question the graph answers with nothing — measured:
        0 nodes and 0 triples for "Quali sono le 3C e cosa vogliono dire?",
        answered entirely from text — the entity list stays empty and every
        follow-up looked like a fresh question. The rewrite step exits on this
        flag before it runs, so an empty graph turn silently disabled
        follow-up handling for the rest of the session.
        """
        return self.turn > 0 or self.failed_turns > 0

    def observe_failure(self, question: str) -> None:
        """Record that a turn happened even though it produced no answer.

        `observe` runs only after a successful `graph.invoke`, so a turn that
        raised left no trace: if the failure was the first turn of a session,
        `has_context()` stayed false, the rewrite step never ran, and the next
        legitimate follow-up was treated as a fresh question.

        Deliberately not `observe` with an empty answer: that would increment
        `turn`, and the demo retries the same question after rebuilding onto
        the fallback graph, so one question would consume two turns and shorten
        the entity decay window by one. The turn counter stays the truth about
        answered turns; this only records that the conversation has started.
        """
        self.failed_turns += 1
        self.last_question = " ".join(str(question or "").split())

    def seed_entities(self, limit: int | None = None) -> list[str]:
        """Entities to resolve a follow-up against, most useful first.

        Only entities the previous answer actually named: they are what the
        expert just read, and therefore what an elliptical follow-up refers to.

        Retrieved-but-unused entities are deliberately excluded rather than
        ranked below. On "Quali sono le 3C dell'economia circolare per il cibo?"
        the graph returned 35 nodes, none of which the answer mentioned — it
        discussed Capitale, Ciclicità and Coevoluzione — so the old fallback
        ranking seeded the rewrite with "Economia circolare ittica" and the
        follow-up went out asking about fish. An empty seed list costs nothing:
        `_rewrite_with_memory` then keeps the question as typed, which the
        retriever handles, while a wrong seed sends it somewhere else entirely.
        """
        cap = self.max_seed_entities if limit is None else limit
        recent = {name.lower() for name in self.last_answer_entities}
        if not recent:
            return []
        ranked = sorted(
            (item for item in self.active_entities if item.name.lower() in recent),
            key=lambda item: (item.turn, item.mentions),
            reverse=True,
        )

        selected: list[str] = []
        selected_words: list[tuple[str, ...]] = []
        for item in ranked:
            if len(selected) >= cap:
                break
            words = _words(item.name)
            if not words:
                continue
            if any(_contains_span(chosen, words) for chosen in selected_words):
                continue
            # "Regione" next to "Regione Piemonte" wastes one of the few slots
            # and makes the rewrite vaguer, not richer. Either order can occur —
            # ranking decides which of the two is seen first — so the specific
            # name wins whether it arrives before or after the broader one.
            # One new entity can subsume more than one already-selected one
            # (e.g. "Politica Agricola Comune" absorbs both "Politica Agricola"
            # and "Agricola Comune"), so every absorbed slot is dropped, not
            # just the first found.
            absorbed = [
                pos for pos, chosen in enumerate(selected_words)
                if _contains_span(words, chosen)
            ]
            if absorbed:
                keep_at = absorbed[0]
                for pos in reversed(absorbed[1:]):
                    del selected[pos]
                    del selected_words[pos]
                # Replace in place: the slot keeps the rank it earned.
                selected[keep_at] = item.name
                selected_words[keep_at] = words
                continue
            selected.append(item.name)
            selected_words.append(words)
        return selected

    def observe(
        self,
        question: str,
        answer: str,
        nodes: Sequence[dict[str, Any]] = (),
        triples: Sequence[dict[str, Any]] = (),
    ) -> None:
        """Record one completed turn.

        Args:
            question: The question as typed.
            answer: The generated answer, used only to tell which retrieved
                entities the model actually talked about.
            nodes: Retrieved KG nodes for the turn.
            triples: Retrieved triples (and subgraph) for the turn.
        """
        self.turn += 1
        self.last_question = " ".join(str(question or "").split())

        retrieved = _entity_names(nodes=nodes, triples=triples)
        index = {item.name.lower(): item for item in self.active_entities}
        for name in retrieved:
            existing = index.get(name.lower())
            if existing is None:
                item = ActiveEntity(name=name, turn=self.turn)
                self.active_entities.append(item)
                index[name.lower()] = item
            else:
                existing.turn = self.turn
                existing.mentions += 1

        # Whole-word match: this list is the top of the seed ranking, so a name
        # that only happens to sit inside a longer word steers the rewrite
        # towards something the answer never discussed.
        answer_words = _words(answer)
        self.last_answer_entities = [
            name for name in retrieved if _contains_span(answer_words, _words(name))
        ]

        cutoff = self.turn - self.window
        self.active_entities = [
            item for item in self.active_entities if item.turn > cutoff
        ]

        self._record_exchange(question=self.last_question, answer=answer)

    def _record_exchange(self, question: str, answer: str) -> None:
        """Append the turn to the transcript and trim it to the budget.

        Oldest first out: a reference to what was just said is what breaks
        without a transcript, and the far end of a long conversation is the part
        the user is least likely to be quoting.
        """
        prose = _strip_references(answer)
        if not question and not prose:
            return
        self.exchanges.append(Exchange(question=question, answer=prose))

        budget = self.max_transcript_chars
        while len(self.exchanges) > 1 and self._transcript_size() > budget:
            self.exchanges.pop(0)

    def _transcript_size(self) -> int:
        return sum(len(item.question) + len(item.answer) for item in self.exchanges)

    def transcript(self) -> str:
        """The conversation so far, as text for the answer prompt.

        Empty until a turn has completed, so the first question of a session
        renders the prompt exactly as it did before this existed.

        Labels are English because the prompt around them is: the model has to
        read these as speaker turns, not as retrieved material. The answers
        carry no reference tags, so nothing here can be cited.
        """
        parts: list[str] = []
        for item in self.exchanges:
            if item.question:
                parts.append(f"User: {item.question}")
            if item.answer:
                parts.append(f"Assistant: {item.answer}")
        return "\n\n".join(parts)


def _entity_names(
    nodes: Sequence[dict[str, Any]] = (),
    triples: Sequence[dict[str, Any]] = (),
) -> list[str]:
    """Canonical entity names from one turn's retrieval, in retrieval order.

    Retrieval order is relevance order, so the first names are the ones the turn
    was really about.
    """
    names: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        name = " ".join(str(value or "").split())
        if not (_MIN_ENTITY_CHARS <= len(name) <= _MAX_ENTITY_CHARS):
            return
        if not any(char.isalpha() for char in name):
            return
        if name.lower().endswith(_DOCUMENT_SUFFIXES):
            return
        key = name.lower()
        if key in seen:
            return
        seen.add(key)
        names.append(name)

    for node in _as_dicts(nodes):
        add(node.get("text") or dict(node.get("properties", {}) or {}).get("name"))

    for triple in _as_dicts(triples):
        add(triple.get("subject"))
        add(triple.get("object"))

    return names


def _as_dicts(items: Iterable[Any]) -> list[dict[str, Any]]:
    return [item for item in (items or []) if isinstance(item, dict)]
