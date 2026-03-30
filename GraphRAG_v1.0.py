"""Backward-compatible entrypoint.

The original monolithic implementation has been refactored into the src/graphrag package.
Run this file exactly as before, or prefer the CLI entrypoint:

    graphrag-demo --seed-movie-dataset
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graphrag.cli import main


if __name__ == "__main__":
    main()
