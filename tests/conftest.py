from __future__ import annotations

import sys
from pathlib import Path

# Make the editable `graphrag` package importable even without `pip install -e .`.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
