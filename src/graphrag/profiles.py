"""Named configuration profiles.

An ``AgentConfig`` has 88 fields and roughly a dozen of them are feature flags
that default to off, each one off "to keep existing baselines unchanged". That
is the right default for a research codebase and the wrong starting point for
anyone who just wants the system as it is actually run. The combinations that
are actually run live in four places: three shell scripts under
``scripts/runners/`` repeat the same ~30 CLI flags, and ``product/config.py``
carries a fourth variant as Python constants.

This module is the data those four surfaces encode, written once. It only
declares overrides against the library defaults, so a field that does not appear
here is at its ``AgentConfig`` default -- which is also how the profiles stay
short: the campaign script passes ``--complexity medium``,
``--max-context-tokens 6000``, ``--citation-policy mark`` and
``--text-retriever-backend tfidf``, all of which are already the default and so
are absent below.

Nothing in the codebase calls this module yet. It is introduced alongside
``tests/test_profiles.py``, which checks ``THESIS_CAMPAIGN`` against the
resolved configuration recorded by a real campaign run, so that switching the
callers over later is a verifiable step rather than a hopeful one.
"""

from __future__ import annotations

import copy
from typing import Any

from graphrag.config import OUTPUT_COMPLEXITY, AgentConfig
from graphrag.strategies import apply_strategy

# The eight-strategy gold campaign: scripts/runners/run_gold_variant.sh,
# run_italian_arm.sh and run_abstention_arms.sh, whose flag blocks are identical.
# Verified field by field against the run recorded in
# tests/fixtures/thesis_campaign_config.json.
THESIS_CAMPAIGN: dict[str, Any] = {
    # Numbered evidence, tag verification, and reader-facing labels rather than
    # the [S1]/[T1] ids, which a reader cannot check against anything.
    "cite_evidence": True,
    "citation_display": "label",
    # Answer in the language of the question, with one retry.
    "enforce_language": True,
    # Answer what was asked and leave out related material the evidence carries.
    "focused_answer": True,
    # Definitional questions open with the quoted defining passage.
    "prefer_verbatim_definitions": True,
    # Cross-lingual retrieval: the graph is largely Italian, the gold English.
    "vector_retrieval": True,
    # Anchor expansion on retrieved node names rather than question words, and
    # spread the subgraph over the top three anchors to keep breadth.
    "seed_from_retrieved": True,
    "subgraph_seed_count": 3,
    # Source diversification in the text channel.
    "text_retriever_mmr": True,
    "text_retriever_max_per_doc": 2,
}

# The demo both expert-facing surfaces run: product/config.py, whose constants
# are all overridable through DEMO_* environment variables. The values here are
# those defaults.
#
# It is not the campaign profile with a different name. A demo answers a person
# waiting at a screen, so it is more verbose (complexity high, limits always
# shown) and it may abstain (domain gate) or answer from parametric knowledge
# where retrieval missed, marked as such -- neither of which a measurement run
# may do. It also leaves `focused_answer` and the seeding options at their
# defaults, so a demo answer is not a retrieval measurement.
DEMO: dict[str, Any] = {
    "complexity": OUTPUT_COMPLEXITY.HIGH,
    "always_include_limits": True,
    "cite_evidence": True,
    "citation_display": "label",
    "enforce_language": True,
    "prefer_verbatim_definitions": True,
    "vector_retrieval": True,
    "enable_domain_gate": True,
    "allow_parametric_fallback": True,
    "text_retriever_top_k": 8,
    "text_retriever_mmr": True,
    "text_retriever_max_per_doc": 2,
    # `product/config.py` has run this since the demo existed, while the
    # library default is "tfidf" — so this profile, which claims to describe
    # the demo, described a retriever the demo does not use (MNT-7). Measured
    # 2026-09-04 on the 30 EN gold questions: dense 0.625 concept recall
    # against 0.580, and 22 of 30 questions reaching one of the 9 Italian
    # documents against 7 of 30. Declared here so the drift cannot recur
    # silently; `test_demo_profile_matches_the_product` holds it to
    # product/config.py.
    "text_retriever_backend": "dense",
}

# The control arm: library defaults, every optional channel and prompt change
# off. Empty on purpose -- it is what the campaigns are measured against, and
# naming it keeps that explicit instead of implicit in an absence.
RESEARCH_BASELINE: dict[str, Any] = {}

PROFILES: dict[str, dict[str, Any]] = {
    "thesis_campaign": THESIS_CAMPAIGN,
    "demo": DEMO,
    "research_baseline": RESEARCH_BASELINE,
}


def build_config(
    profile: str,
    strategy: str | None = None,
    **overrides: Any,
) -> AgentConfig:
    """Build an ``AgentConfig`` from a named profile.

    Args:
        profile: One of :data:`PROFILES`.
        strategy: Optional retrieval-strategy preset to apply on top, from
            ``graphrag.strategies.STRATEGY_PRESETS``.
        **overrides: Field values applied last, after the profile and the
            strategy. Used for the per-run values a profile cannot carry, such
            as ``query``.

    Returns:
        A new ``AgentConfig``. The profile dictionaries are never mutated.

    Raises:
        ValueError: If ``profile`` is unknown, or if an override names a field
            that ``AgentConfig`` does not have.
    """
    if profile not in PROFILES:
        allowed = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown profile '{profile}'. Allowed: {allowed}")

    fields = {f.name for f in AgentConfig.__dataclass_fields__.values()}
    unknown = sorted(set(overrides) - fields)
    if unknown:
        raise ValueError(f"Unknown AgentConfig field(s): {', '.join(unknown)}")

    config = AgentConfig(**copy.deepcopy(PROFILES[profile]))
    if strategy is not None:
        config = apply_strategy(config, strategy)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config
