"""The named profiles must reproduce what the campaigns actually ran.

`THESIS_CAMPAIGN` is checked against `tests/fixtures/thesis_campaign_config.json`,
which is the `strategy_configs` block written by `graphrag.cli --experiment` for
a real thesis run (gemma3_27b_awq, 2026-08-18). If a profile field drifts from
what that campaign was measured with, these tests fail -- which is the point:
the profile is only useful if switching a runner over to it cannot move a
published number.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from graphrag.config import AgentConfig
from graphrag.profiles import (
    DEMO,
    PROFILES,
    RESEARCH_BASELINE,
    THESIS_CAMPAIGN,
    build_config,
)
from graphrag.strategies import STRATEGY_PRESETS

FIXTURE = Path(__file__).parent / "fixtures" / "thesis_campaign_config.json"

# Per-run values, not part of any profile: the fixture has them stripped.
PER_RUN_FIELDS = ("query", "entity")


def _recorded() -> dict[str, dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["strategy_configs"]


def _serialise(config: AgentConfig) -> dict:
    """Render a config the way `graphrag.cli` writes it into config.json."""
    raw = dataclasses.asdict(config)
    for field in PER_RUN_FIELDS:
        raw.pop(field, None)
    # cli.py dumps with `default=str`, which is what turns enums into
    # "OUTPUT_TONE.TECHNICAL" and tuples into lists.
    return json.loads(json.dumps(raw, default=str))


@pytest.mark.parametrize("strategy", sorted(_recorded()))
def test_thesis_campaign_reproduces_the_recorded_run(strategy: str) -> None:
    """Every field of every strategy matches the campaign that was published."""
    expected = _recorded()[strategy]
    actual = _serialise(build_config("thesis_campaign", strategy=strategy))

    assert set(actual) == set(expected), "field set drifted from the recorded run"
    differing = {k: (expected[k], actual[k]) for k in expected if expected[k] != actual[k]}
    assert not differing, f"{strategy}: profile differs from the recorded run: {differing}"


def test_fixture_covers_every_strategy() -> None:
    """The contract is only as strong as its coverage of the preset list."""
    assert set(_recorded()) == set(STRATEGY_PRESETS)


def test_profiles_only_declare_real_fields() -> None:
    """A typo in a profile key would otherwise surface as a silent no-op."""
    fields = {f.name for f in AgentConfig.__dataclass_fields__.values()}
    for name, profile in PROFILES.items():
        unknown = sorted(set(profile) - fields)
        assert not unknown, f"profile '{name}' names non-existent field(s): {unknown}"


def test_profiles_only_declare_actual_overrides() -> None:
    """A profile entry equal to the library default is noise that will rot.

    The campaign shell scripts pass several flags whose values are already the
    default; carrying them here would make the profile look like it controls
    something it does not.
    """
    default = AgentConfig()
    for name, profile in PROFILES.items():
        redundant = [k for k, v in profile.items() if getattr(default, k) == v]
        assert not redundant, f"profile '{name}' restates library defaults: {redundant}"


def test_research_baseline_is_the_library_default() -> None:
    assert RESEARCH_BASELINE == {}
    assert _serialise(build_config("research_baseline")) == _serialise(AgentConfig())


def test_demo_differs_from_the_campaign_where_it_should() -> None:
    """The demo is not the campaign renamed; the differences are deliberate."""
    # A demo may abstain and may answer from parametric knowledge; a measurement
    # run may do neither.
    assert DEMO["enable_domain_gate"] is True
    assert DEMO["allow_parametric_fallback"] is True
    assert "enable_domain_gate" not in THESIS_CAMPAIGN
    assert "allow_parametric_fallback" not in THESIS_CAMPAIGN
    # The demo does not narrow the answer or re-seed graph expansion, so a demo
    # answer is not a retrieval measurement.
    for field in ("focused_answer", "seed_from_retrieved", "subgraph_seed_count"):
        assert field in THESIS_CAMPAIGN
        assert field not in DEMO


def test_build_config_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="Unknown profile"):
        build_config("does_not_exist")


def test_build_config_rejects_unknown_field() -> None:
    with pytest.raises(ValueError, match="Unknown AgentConfig field"):
        build_config("demo", nonexistent_field=1)


def test_build_config_does_not_mutate_the_profile() -> None:
    before = dict(THESIS_CAMPAIGN)
    build_config("thesis_campaign", strategy="hybrid", query="x")
    assert THESIS_CAMPAIGN == before


def test_overrides_win_over_profile_and_strategy() -> None:
    config = build_config("thesis_campaign", strategy="text_only", include_nodes=True)
    # text_only switches include_nodes off; the explicit override comes last.
    assert config.include_nodes is True
    assert build_config("thesis_campaign", strategy="text_only").include_nodes is False
