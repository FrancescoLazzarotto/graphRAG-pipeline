"""Two ways a failing graph used to be indistinguishable from a working one.

The driver's own retry window is the first: at its 30 s default, one
unreachable graph cost 301 s of waiting in a measured demo session, because
every query in a retrieval burned the window independently and the manager's
retry loop multiplied it.

The postprocess pass is the second: every step collects its failures into an
`errors` list, and `main()` exited 0 whether that list was empty or not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphrag.kg import manager as manager_module
from kg_pipeline.stages.neo4j_postprocess import _collect_errors


class _Recorder:
    """Stands in for Neo4jGraph and keeps the kwargs it was built with."""

    last_kwargs: dict = {}

    def __init__(self, **kwargs) -> None:
        type(self).last_kwargs = kwargs


@pytest.fixture
def built(monkeypatch) -> dict:
    monkeypatch.setattr(manager_module, "Neo4jGraph", _Recorder)
    bare = manager_module.KnowledgeGraphManager.__new__(manager_module.KnowledgeGraphManager)
    bare.config = manager_module.KGConfig(
        url="bolt://localhost:7687", username="neo4j", password="x", database="neo4j"
    )
    bare._build_graph()
    return _Recorder.last_kwargs


def test_the_retry_window_is_shorter_than_the_drivers_default(built) -> None:
    """30 s per query per retry is what made a failover take five minutes."""
    assert built["driver_config"]["max_transaction_retry_time"] < 30.0


def test_connecting_gives_up_before_a_person_does(built) -> None:
    assert built["driver_config"]["connection_timeout"] <= 10.0


def test_a_query_may_run_well_past_the_slowest_one_measured(built) -> None:
    """34 of 36 queries measured under 0.23 s; the two slow ones took ~24 s.
    The cap has to clear those, or it turns a slow answer into no answer."""
    assert built["timeout"] > 24.3


@pytest.mark.parametrize(
    "name,value,expected",
    [
        ("GRAPHRAG_NEO4J_MAX_RETRY_TIME_SEC", "2.5", 2.5),
        ("GRAPHRAG_NEO4J_MAX_RETRY_TIME_SEC", "0", 8.0),
        ("GRAPHRAG_NEO4J_MAX_RETRY_TIME_SEC", "-1", 8.0),
        ("GRAPHRAG_NEO4J_MAX_RETRY_TIME_SEC", "abc", 8.0),
        ("GRAPHRAG_NEO4J_MAX_RETRY_TIME_SEC", "", 8.0),
    ],
)
def test_an_unusable_override_falls_back_instead_of_crashing(
    monkeypatch, name: str, value: str, expected: float
) -> None:
    """An operator typo must not stop the demo from connecting at all."""
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(manager_module, "Neo4jGraph", _Recorder)
    bare = manager_module.KnowledgeGraphManager.__new__(manager_module.KnowledgeGraphManager)
    bare.config = manager_module.KGConfig(
        url="bolt://localhost:7687", username="neo4j", password="x", database="neo4j"
    )
    bare._build_graph()
    assert _Recorder.last_kwargs["driver_config"]["max_transaction_retry_time"] == expected


def test_errors_are_found_however_deep_the_report_nests_them() -> None:
    report = {
        "step1": {"renamed": 3, "errors": ["rename A -> B: apoc missing"]},
        "aura": {"sub": [{"errors": []}, {"errors": ["merge failed"]}]},
        "step5": {"created": ["c"], "errors": []},
    }
    assert _collect_errors(report) == ["rename A -> B: apoc missing", "merge failed"]


def test_a_clean_report_reports_nothing() -> None:
    assert _collect_errors({"step1": {"errors": []}, "step2": {"renamed": 2}}) == []
