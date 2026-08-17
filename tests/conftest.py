from __future__ import annotations

import sys
from pathlib import Path

# Make the editable `graphrag` package importable even without `pip install -e .`.
_ROOT = Path(__file__).resolve().parent.parent
# `kg_pipeline` is a top-level package in the repo root and `evalkit` lives under
# evaluation/. Both were reachable only because pytest is normally invoked from
# the repo root, which puts the cwd on sys.path — so a test importing either one
# failed when the suite was run by absolute path from anywhere else.
for _path in (_ROOT / "src", _ROOT, _ROOT / "evaluation"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
