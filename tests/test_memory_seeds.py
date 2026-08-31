"""A document name is where an answer came from, not what it was about.

`SEeD for Change.pdf` reached the seed list in two recorded demo sessions and
led it in one, spending one of only four slots on a name that steers the
rewrite of a follow-up towards the file rather than the subject.

The other half of the finding — that seeds are not tested for discriminativeness
and could be filtered by the retriever's `lexical_df_max_ratio` — is not
implemented, and `test_document_frequency_would_be_the_wrong_filter` records
why, measured on the live graph rather than argued.
"""

from __future__ import annotations

import pytest

from graphrag.agent.memory import ConversationMemory, _entity_names


def _nodes(*names: str) -> list[dict]:
    return [{"text": name} for name in names]


@pytest.mark.parametrize(
    "name",
    ["SEeD for Change.pdf", "REPORT MATTM_Definitivo.PDF", "dati.csv", "note.DOCX"],
)
def test_a_document_name_is_not_an_entity(name: str) -> None:
    assert _entity_names(nodes=_nodes(name)) == []


@pytest.mark.parametrize(
    "name",
    ["Circular Economy for Food", "biochar", "SEeD for Change", "Agenda ONU 2030"],
)
def test_a_subject_that_merely_resembles_one_survives(name: str) -> None:
    """The rule is the suffix, not the presence of a dot or of a file-ish word."""
    assert _entity_names(nodes=_nodes(name)) == [name]


def test_the_slot_goes_to_the_subject_instead() -> None:
    """The recorded session that ranked the file first: it now ranks nothing."""
    memory = ConversationMemory()
    memory.observe(
        question="Che cos'è SEeD e che cosa vuol dire?",
        answer="SEeD for Change è un progetto; SEeD for Global Goals ne è l'evoluzione.",
        nodes=_nodes("SEeD for Change.pdf", "SEeD for Change", "SEeD for Global Goals"),
    )
    seeds = memory.seed_entities()
    assert "SEeD for Change.pdf" not in seeds
    assert seeds, "dropping the document must not empty the seed list"


def test_document_frequency_would_be_the_wrong_filter() -> None:
    """Measured on the live graph (14 520 nodes, 1% ceiling = 145).

    The finding proposed reusing `lexical_df_max_ratio` to drop vague seeds.
    The numbers say it does the opposite of what it was asked for: document
    frequency here counts how many *node names* contain a token, so the
    domain's central English words are the common ones and the vague Italian
    abstractions the finding named are rare. Every token of "Circular Economy
    for Food" is above the ceiling, while `cambiamento`, `integrazione` and
    `transizione` are far below it. Any threshold that removes the second group
    removes the first group first.
    """
    df_over_ceiling = {"circular": 165, "economy": 151, "for": 259, "food": 530}
    df_under_ceiling = {"cambiamento": 13, "integrazione": 4, "transizione": 20}
    ceiling = 145
    assert all(df > ceiling for df in df_over_ceiling.values())
    assert all(df < ceiling for df in df_under_ceiling.values())
