from __future__ import annotations

import re


_LONG_SHORT_RE = re.compile(r"\b([A-Za-z][A-Za-z \-/]{3,}?)\s*\(([A-Z]{2,10})\)")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_surface(text: str) -> str:
    return _NON_ALNUM_RE.sub("", text.lower())


def update_acronym_map(acronym_map: dict[str, str], text: str) -> None:
    for long_form, short_form in _LONG_SHORT_RE.findall(text):
        long_clean = " ".join(long_form.split()).strip()
        short_clean = short_form.strip().upper()
        if len(long_clean) > 2 and len(short_clean) > 1:
            acronym_map[short_clean] = long_clean


def expand_acronym(surface: str, acronym_map: dict[str, str]) -> str:
    stripped = surface.strip()
    if not stripped:
        return stripped
    if stripped.upper() in acronym_map:
        return acronym_map[stripped.upper()]
    return stripped
