from __future__ import annotations

import re
import unicodedata

# Fixed before evaluation per gold_entity_eval_protocol.md §3 and §6.
# Every pipeline is normalised through these functions, symmetrically.

_STRIP_LEADING = r"^[\s\-—–\"'“”«»]+"
_STRIP_TRAILING = r"[\s\-—–\"'“”«».,;:]+$"


def normalise(text: str) -> str:
    """Canonical surface form used for every concept-level comparison.

    Protocol §3: lowercase, strip surrounding whitespace/punctuation, collapse
    internal whitespace. No fuzzy matching, no accent folding (see fold_accents),
    no plural stripping (see the resolver's lexicon).

    Parentheses are deliberately NOT stripped: a trailing ')' that closes a
    parenthetical gloss ('potassium bitartrate (cream of tartar)') is part of the
    term, and removing it corrupts the label into an unmatchable string.

    Args:
        text: Raw surface form from a gold entry or a pipeline's output.

    Returns:
        The normalised form; empty string if nothing survives normalisation.
    """
    out = text.strip().lower()
    out = re.sub(_STRIP_LEADING, "", out)
    out = re.sub(_STRIP_TRAILING, "", out)
    return re.sub(r"\s+", " ", out).strip()


def fold_accents(text: str) -> str:
    """Return an accent-stripped key ('ciclicità' -> 'ciclicita').

    Used only as a secondary matching key: Italian surface forms reach us both
    accented (from source text) and unaccented (from lossy extractors), and the
    protocol's no-fuzzy-matching rule forbids absorbing that difference with edit
    distance. Folding is deterministic and auditable, so it is allowed; it is
    applied symmetrically to gold and to every pipeline.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def match_key(text: str) -> str:
    """Full comparison key: normalised + accent-folded."""
    return fold_accents(normalise(text))
