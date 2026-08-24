"""Refuse to modify a graph that nobody asked to modify.

Helper module, not an entrypoint.

The kg_repair passes rewrite the graph in place: they delete nodes, reverse
relationships and rename relationship types. They take no arguments, print no
usage and ask for no confirmation, and they read their target from
``kg_pipeline/.env`` — which points at the hosted graph the demo serves.

On 2026-08-24 that combination cost the demo graph 1 661 vector carriers, 43
entities, all 532 ``PART_OF`` relationships and the direction of 39 more. The
passes were started by someone checking that a file move had not broken
anything, the ordinary way: running each script with ``--help``. Having no
argument parser, they ignored the flag and did the work.

So the rule here is narrow and absolute: **asking a repair pass what it does
must never make it do it.** Confirmation is explicit, and the target is printed
before anything is written, because "which graph is this pointing at" is the
question that actually matters.
"""

from __future__ import annotations

import os
import sys

CONFIRM_ENV = "KG_REPAIR_CONFIRM"
CONFIRM_FLAG = "--yes"
_HELP_FLAGS = {"-h", "--help"}


def _looks_hosted(uri: str) -> bool:
    """Whether this URI is a remote instance rather than a local one."""
    lowered = uri.lower()
    local_markers = ("localhost", "127.0.0.1", "0.0.0.0", "::1")
    return not any(marker in lowered for marker in local_markers)


def require_confirmation(
    title: str,
    what_it_does: str,
    uri: str,
    database: str | None,
    argv: list[str] | None = None,
) -> None:
    """Return only if the caller has explicitly confirmed a destructive run.

    Args:
        title: Human name of the pass, e.g. "KG Repair 3".
        what_it_does: One or more lines describing the writes it performs.
        uri: The Neo4j URI the pass would write to.
        database: The target database, or None for the server default.
        argv: Command line to inspect; defaults to ``sys.argv[1:]``.

    Raises:
        SystemExit: Always, unless the run was confirmed. Exit code 0 — being
            asked for a plan and getting one is a success, not a failure.
    """
    args = sys.argv[1:] if argv is None else argv
    target = f"{uri or '<unset>'} (db: {database or '<default>'})"
    confirmed = CONFIRM_FLAG in args or os.getenv(CONFIRM_ENV, "").strip().lower() == "yes"

    if not any(flag in args for flag in _HELP_FLAGS) and confirmed:
        if _looks_hosted(uri):
            print(f"!! {title}: writing to the HOSTED graph {target}", file=sys.stderr)
        return

    asked_for_help = any(flag in args for flag in _HELP_FLAGS)
    print(f"{title} — modifies the graph in place. Nothing has been written.")
    print()
    print("What it would do:")
    for line in what_it_does.strip().splitlines():
        print(f"  {line.strip()}")
    print()
    print(f"Target: {target}")
    if _looks_hosted(uri):
        print("        ^ this is the hosted graph the demo serves, not a scratch copy.")
        print("        Export NEO4J_URL/NEO4J_USERNAME/NEO4J_PASSWORD to point elsewhere;")
        print("        exported values win over kg_pipeline/.env.")
    print()
    print("To run it for real:")
    print(f"  python {sys.argv[0]} {CONFIRM_FLAG}")
    print(f"  {CONFIRM_ENV}=yes python {sys.argv[0]}")
    if asked_for_help and confirmed:
        print()
        print(f"({CONFIRM_FLAG} was given together with a help flag; help wins.)")
    raise SystemExit(0)
