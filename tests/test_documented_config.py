"""Every `DEMO_*` variable the product reads must be documented.

`docs/configuration.md` is where an operator looks to find out what they can
change. It omitted eleven variables, and they were not the harmless ones: the
domain gate, intra-session memory, the vector channel, the parametric fallback
— the switches that change what the demo answers. A variable nobody documented
is a behaviour nobody can turn off deliberately.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _ROOT / "product" / "config.py"
_DOC = _ROOT / "docs" / "configuration.md"
_VAR = re.compile(r"DEMO_[A-Z0-9_]+")


def test_every_demo_variable_is_documented() -> None:
    letti = set(_VAR.findall(_CONFIG.read_text(encoding="utf-8")))
    documentati = set(_VAR.findall(_DOC.read_text(encoding="utf-8")))

    mancanti = sorted(letti - documentati)

    assert not mancanti, (
        f"{len(mancanti)} DEMO_* variables are read by product/config.py and "
        f"absent from docs/configuration.md: {mancanti}"
    )


def test_the_documentation_invents_no_variable() -> None:
    """The other direction: a documented switch that does nothing misleads too."""
    letti = set(_VAR.findall(_CONFIG.read_text(encoding="utf-8")))
    documentati = set(_VAR.findall(_DOC.read_text(encoding="utf-8")))

    inventate = sorted(documentati - letti)

    assert not inventate, (
        f"documented but not read by product/config.py: {inventate}"
    )
