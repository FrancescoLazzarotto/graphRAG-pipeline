"""Recognise the questions that are about the assistant, not about the corpus.

An expert opening the demo does not start with a gold question. They type
"ciao", or "prova, sistema operativo?", and both reach retrieval as if they
were subject questions: the first carries no search terms at all and comes back
with an empty context, so the model answers that it does not know; the second
is refused with a sentence about documents that never says what the assistant
is. Either way the first exchange of the session teaches the reader that the
system is broken.

These are not domain questions and no gate can make them into one — there is
nothing to retrieve. They are answered here, before retrieval, with a fixed
sentence saying what this assistant covers plus the questions worth asking.

Detection is deterministic on purpose. An LLM classifier would cost a call on
every turn and, worse, would be free to read "come funziona la simbiosi
industriale?" as a question about itself. The patterns below match the whole
normalised question or a specific phrase that has no reading inside the domain,
under a word ceiling: a real question that happens to contain "chi sei" in a
subordinate clause is longer than anything here.
"""

from __future__ import annotations

import re
import unicodedata

# Above this many words a question carries a subject of its own, whatever
# phrase it contains. Measured against the frozen gold set: the shortest of the
# 30 questions is 7 words, and none of them matches a pattern below anyway —
# the ceiling is the second lock, not the first.
_MAX_META_WORDS = 10

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Whole-question greetings and pings. Matched against the entire normalised
# string, never as a substring: "ciao, cos'e l'economia circolare del cibo?"
# is a domain question with a greeting glued to its front, and answering it
# with an introduction would be worse than what happens today.
_WHOLE_IT = {
    "ciao", "ciao ciao", "salve", "buongiorno", "buon giorno", "buonasera",
    "buona sera", "buonpomeriggio", "buon pomeriggio", "ehi", "ehila", "hey",
    "ciao come stai", "come stai", "come va", "tutto bene", "prova",
    "prova prova", "sei attivo", "ci sei", "funzioni",
}
_WHOLE_EN = {
    "hi", "hi there", "hello", "hello there", "hey", "hey there", "yo",
    "good morning", "good afternoon", "good evening", "how are you",
    "are you there", "test", "testing", "ping", "are you working",
}

# Phrases with no reading inside the domain. Each is anchored on the verb or
# the possessive that makes it about the assistant: "che modello" alone matches
# "che modello di economia circolare descrive il documento?", "che modello sei"
# cannot.
_PHRASES_IT = (
    r"chi sei",
    r"chi e' che (mi )?risponde",
    r"(che )?cosa sei",
    r"che cos'?e'? (questo|sto) (assistente|sistema|servizio|chatbot|bot|strumento)",
    r"come ti chiami",
    r"(qual e'|qualè) il tuo nome",
    r"(che )?cosa (sai|puoi) fare$",
    r"cosa (mi )?sai dire di te",
    r"a (che )?cosa servi",
    r"come funzioni",
    r"come funziona (questo|il|la) (assistente|sistema|servizio|chat|chatbot|bot|demo|strumento)",
    r"di (che )?cosa (parli|ti occupi)",
    r"(che|quali) domande posso (farti|fare|porti)",
    r"(che )?cosa posso (chiederti|domandarti)",
    r"su (che )?cosa (posso chiederti|rispondi)",
    r"che tipo di domande",
    # No reading inside the domain: no document here is about an OS, so the
    # phrase is a probe of the machine whoever typed it thinks they are talking
    # to. Unanchored for the same reason.
    r"sistema operativo",
    r"(che|quale) modello (sei|usi|stai usando)",
    r"che (llm|intelligenza artificiale|ia) (sei|usi)",
    r"sei (chatgpt|un umano|una persona|un bot|un'?intelligenza artificiale)",
    r"^aiuto$",
)
# The `$` on the capability phrases is load-bearing: "what can you do with
# grape pomace?" is a domain question that opens with one of them, and answering
# it with an introduction would be a worse failure than the one being fixed.
_PHRASES_EN = (
    r"who are you",
    r"what are you",
    r"what is this (assistant|system|service|chatbot|bot|tool|demo)",
    r"what'?s your name",
    r"what can you do$",
    r"what do you do$",
    r"what are you for",
    r"how do you work",
    r"how does this (assistant|system|service|chat|chatbot|bot|demo|tool) work",
    r"what (questions )?(can|should) i ask",
    r"what can i ask you",
    r"what kind of questions",
    r"operating system",
    r"(what|which) model (are you|do you use|are you using)",
    r"(what|which) (llm|ai) (are you|do you use)",
    r"are you (chatgpt|human|a human|a bot|an ai)",
    r"^help$",
)

_RE_IT = tuple(re.compile(p) for p in _PHRASES_IT)
_RE_EN = tuple(re.compile(p) for p in _PHRASES_EN)


def _normalise(question: str) -> str:
    """Lowercase, unaccented, punctuation-free, single-spaced.

    Accents go because the demo is typed into a browser by people who write
    "qual e'", "qual è" and "qual e" in the same session, and the patterns
    would otherwise need three spellings each.
    """
    text = unicodedata.normalize("NFD", question.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    # The apostrophe stays: it is what separates "cos'e" from "cose".
    text = re.sub(r"[^\w'\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def detect_meta_question(question: str) -> str | None:
    """Say whether this is a question about the assistant, and in which language.

    Args:
        question: The question as typed.

    Returns:
        ``"it"`` or ``"en"`` when the question is a greeting, a ping or a
        question about the assistant itself; ``None`` when it is anything else,
        which is every question that must keep reaching retrieval.
    """
    text = _normalise(question)
    if not text:
        return None
    if text in _WHOLE_IT:
        return "it"
    if text in _WHOLE_EN:
        return "en"
    # Language detection on two words is unreliable, and these two sets share
    # "hey": the tie goes to Italian, the language the demo is opened in.
    if len(_WORD_RE.findall(text)) > _MAX_META_WORDS:
        return None
    for pattern in _RE_IT:
        if pattern.search(text):
            return "it"
    for pattern in _RE_EN:
        if pattern.search(text):
            return "en"
    return None
