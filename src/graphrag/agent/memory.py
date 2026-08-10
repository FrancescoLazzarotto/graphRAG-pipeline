"""Intra-session conversational memory (WP7).

See `docs/demo_quality_plan_2026-07.md` §9. The demo expert reads an answer and
asks a follow-up — "mi indichi le strategie nel settore vino" — where the topic
comes from the previous answer, not from the question. Retrieval receives that
question in isolation and searches for the wrong thing.

This module holds the state needed to make such a question self-contained again:
the entities the conversation is actually about. Three hard boundaries:

* **Never a source of facts.** Memory carries *entities*, never claims. The
  groundedness of a turn is always computed against the evidence retrieved in
  that turn; a model that could cite something "because it was said earlier"
  would be self-confirming.
* **Retrieval only.** The rewritten question steers retrieval; generation still
  answers the question the user literally typed.
* **Off unless asked.** No memory object means the previous behaviour, byte for
  byte — gold runs and experiment baselines stay comparable.

The follow-up detector is deterministic on purpose: an LLM classifier here would
add a call, a failure mode and a source of nondeterminism to every turn, on a
decision that a handful of surface markers already settles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

__all__ = [
    "ActiveEntity",
    "ConversationMemory",
    "is_follow_up",
]

# Anaphora and continuity markers. These never occur in a self-contained
# question: they point at something said before.
_CONTINUITY_MARKERS = re.compile(
    r"\b("
    r"approfondisci|approfondiscilo|dimmi di piu|dimmi di più|dammi piu|dammi più|"
    r"puoi approfondire|espandi|continua|prosegui|vai avanti|"
    r"lo stesso|la stessa|come sopra|come prima|il precedente|la precedente|"
    r"di questi|tra questi|fra questi|di queste|tra queste|"
    r"tell me more|say more|what about|how about|elaborate|expand on|go deeper|"
    r"the same|the previous|of these|among these"
    r")\b",
    re.IGNORECASE,
)

# Demonstratives. Weaker than the markers above — "questo" appears in
# self-contained questions too — so they only count on a short question.
# English "it"/"them" are deliberately absent: they are pervasive in ordinary
# self-contained questions ("how does it differ from whey?").
_DEICTICS = re.compile(
    r"\b(questo|questa|questi|queste|quello|quella|quelli|quelle|ciò|cio|"
    r"these|those)\b",
    re.IGNORECASE,
)

# A question opening with a conjunction is grammatically a continuation.
_OPENING_CONJUNCTION = re.compile(
    r"^\s*(e|ed|ma|invece|inoltre|poi|and|but|also|so)\b", re.IGNORECASE
)

# Second-person request forms: the shape a follow-up takes when the expert stops
# writing full questions and starts talking to the system.
_REQUEST_OPENER = re.compile(
    r"^\s*("
    r"mi\s+(?:dai|dia|indichi|indica|puoi|potresti|spieghi|spiega|elenchi|elenca|"
    r"riporti|riporta|fai|fa|mostri|mostra|descrivi|descriva)|"
    # The bare modal, without the "mi" that used to be required: "puoi
    # spiegarmelo in modo più semplice?" was read as a fresh question, and with
    # the domain gate on it was refused outright as out of domain.
    r"(?:puoi|potresti|riesci\s+a|sai)\s+\w+|"
    r"dammi|dimmi|fammi|spiegami|elencami|indicami|riportami|mostrami|"
    r"give me|show me|list me|(?:can|could)\s+you\s+\w+"
    r")\b",
    re.IGNORECASE,
)

# Above this length a question carries its own context even without proper
# nouns; the elliptical follow-ups we care about are short.
_MAX_ELLIPTIC_TOKENS = 16

# Entity names shorter than this, or made only of digits, are noise as retrieval
# seeds.
_MIN_ENTITY_CHARS = 3
_MAX_ENTITY_CHARS = 60

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


def _looks_self_contained(question: str) -> bool:
    """True when the question names its own subject.

    A proper noun past the opening word ("Piemonte"), an internal capital
    ("SEeD") or an alphanumeric code ("3C", "ISO20121") is enough: the
    retriever's search-term builder will find it, so nothing from the
    conversation is needed. The first token is skipped because every sentence
    starts capitalised.
    """
    for token in _TOKEN_RE.findall(question)[1:]:
        if any(char.isupper() for char in token):
            return True
        # Alphanumeric codes survive lowercasing in the question ("3c").
        if any(char.isdigit() for char in token) and any(
            char.isalpha() for char in token
        ):
            return True
    return False


def is_follow_up(question: str, has_context: bool = True) -> bool:
    """Decide whether `question` depends on what was said earlier.

    Deliberately conservative: when it returns False the question travels
    unchanged, which is the behaviour of the whole system before WP7. A false
    positive costs one short LLM call and a rewrite that may add nothing; a
    false negative just leaves today's behaviour in place.

    Args:
        question: The raw question typed by the user.
        has_context: Whether the memory holds any entity yet. On the first turn
            of a session there is nothing to resolve a reference against.

    Returns:
        True when the question should be rewritten before retrieval.
    """
    text = " ".join(str(question or "").split())
    if not text or not has_context:
        return False

    if _CONTINUITY_MARKERS.search(text) or _OPENING_CONJUNCTION.match(text):
        return True

    tokens = _TOKEN_RE.findall(text)
    if len(tokens) > _MAX_ELLIPTIC_TOKENS:
        return False

    if _DEICTICS.search(text):
        return True

    # Short imperative request with no subject of its own: the Q5 case,
    # "Mi indichi le strategie nel settore vino individuate dalla ricerca?"
    return bool(_REQUEST_OPENER.match(text)) and not _looks_self_contained(text)


@dataclass
class ActiveEntity:
    """A KG entity the conversation has touched, with its recency."""

    name: str
    turn: int
    mentions: int = 1


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
    active_entities: list[ActiveEntity] = field(default_factory=list)
    last_answer_entities: list[str] = field(default_factory=list)
    last_question: str = ""

    def reset(self) -> None:
        """Forget the current topic (the 'Nuovo argomento' button)."""
        self.turn = 0
        self.active_entities = []
        self.last_answer_entities = []
        self.last_question = ""

    def has_context(self) -> bool:
        """Whether anything has been said yet in this session.

        Turn count, not entity count. Entities are observed only from the KG
        channel, and on a question the graph answers with nothing — measured:
        0 nodes and 0 triples for "Quali sono le 3C e cosa vogliono dire?",
        answered entirely from text — the entity list stays empty and every
        follow-up looked like a fresh question. `is_follow_up` exits on this
        flag before it ever reads its markers, so an empty graph turn silently
        disabled follow-up detection for the rest of the session.
        """
        return self.turn > 0

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
            # "Regione" next to "Regione Piemonte" wastes one of the few slots
            # and makes the rewrite vaguer, not richer. Either order can occur —
            # ranking decides which of the two is seen first — so the specific
            # name wins whether it arrives before or after the broader one.
            broader = next(
                (pos for pos, chosen in enumerate(selected_words)
                 if _contains_span(words, chosen)),
                None,
            )
            if any(_contains_span(chosen, words) for chosen in selected_words):
                continue
            if broader is not None:
                # Replace in place: the slot keeps the rank it earned.
                selected[broader] = item.name
                selected_words[broader] = words
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
