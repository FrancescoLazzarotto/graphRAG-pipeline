#!/usr/bin/env python3
"""Preflight for the demo: imports, graph, generator, encoder.

This used to print SMOKE CHECK PASSED while the graph was suspended and no
model was served: the Neo4j probe was opt-in behind ``--check-neo4j`` and the
LLM was never contacted at all, so the check only proved that four modules
import. A health check that passes when the system cannot answer a question is
worse than none, because it is believed.

Every check now runs by default and any failure is a non-zero exit. The two
that depend on a running server can be waived explicitly (``--skip-llm``,
``--skip-encoder``) for offline use, and the waiver is printed.

Usage:
    conda run -n graphllm python scripts/smoke_check.py
    python scripts/smoke_check.py --check-imports-only
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

# Both indexes are required by the demo's retrieval: the full-text one backs the
# lexical channel, the vector one the cross-lingual channel. A graph that
# answers RETURN 1 but has neither is a graph that answers every question badly.
REQUIRED_INDEXES = ("node_search", "node_embedding")


def _check_import(module_name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(module_name)
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def _check_neo4j_connectivity() -> tuple[bool, str]:
    try:
        from neo4j import GraphDatabase
    except Exception as exc:
        return False, f"neo4j driver unavailable: {exc}"

    uri = os.getenv("NEO4J_URL")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    missing = [
        key
        for key, value in (
            ("NEO4J_URL", uri),
            ("NEO4J_USERNAME", username),
            ("NEO4J_PASSWORD", password),
        )
        if not value
    ]
    if missing:
        return False, f"missing environment variables: {', '.join(missing)}"

    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        with driver.session(database=database) as session:
            row = session.run("RETURN 1 AS ok").single()
            if not row or row.get("ok") != 1:
                return False, "unexpected neo4j preflight result"
            nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            if not nodes:
                return False, f"{uri} answers but holds no nodes"
            online = {
                record["name"]
                for record in session.run("SHOW INDEXES YIELD name, state")
                if record["state"] == "ONLINE"
            }
            absent = [name for name in REQUIRED_INDEXES if name not in online]
            if absent:
                return False, (
                    f"{nodes} nodes, but index(es) not ONLINE: {', '.join(absent)} "
                    "— run scripts/kg_search_index.py / scripts/kg_vector_index.py"
                )
    except Exception as exc:
        return False, str(exc)
    finally:
        driver.close()

    return True, f"{uri} — {nodes} nodes, indexes online"


def _served_models(base_url: str, timeout_sec: float) -> list[str]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/models", method="GET"
    )
    api_key = os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        payload = json.load(response)
    return [entry["id"] for entry in payload.get("data", [])]


def _check_openai_endpoint(
    base_url: str | None,
    expected_model: str | None,
    label: str,
    timeout_sec: float,
) -> tuple[bool, str]:
    """Confirm an OpenAI-compatible server answers and serves the wanted model."""
    if not base_url:
        return False, f"{label}: base URL not configured"
    try:
        served = _served_models(base_url, timeout_sec)
    except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError) as exc:
        return False, f"{label}: {base_url} unreachable ({exc})"
    if not served:
        return False, f"{label}: {base_url} serves no model"
    if expected_model and expected_model not in served:
        return False, (
            f"{label}: {base_url} serves {', '.join(served)} — "
            f"configured model is {expected_model}"
        )
    return True, f"{label}: {base_url} — {', '.join(served)}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        default=str(ROOT / "kg_pipeline" / ".env"),
        help="Values fill in variables that are not already exported",
    )
    parser.add_argument(
        "--check-imports-only",
        action="store_true",
        help="Only check imports, skip everything that needs a server",
    )
    parser.add_argument("--skip-neo4j", action="store_true", help="Waive the graph check")
    parser.add_argument("--skip-llm", action="store_true", help="Waive the generator check")
    parser.add_argument(
        "--skip-encoder", action="store_true", help="Waive the embedding-server check"
    )
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    parser.add_argument(
        "--check-neo4j",
        action="store_true",
        help=argparse.SUPPRESS,  # kept so old invocations keep working; now the default
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    # Keep pre-existing environment variables, but populate missing values from
    # the env file: exporting NEO4J_URL inline must still win.
    load_dotenv(args.env_file, override=False)
    load_dotenv(override=False)

    required_modules = ["torch", "langgraph", "transformers", "neo4j"]

    failures: list[str] = []
    for module_name in required_modules:
        ok, reason = _check_import(module_name)
        if not ok:
            failures.append(f"import {module_name}: {reason}")

    if args.check_imports_only:
        if failures:
            print("IMPORT CHECK FAILED")
            for item in failures:
                print(f"- {item}")
            return 1
        print("IMPORT CHECK PASSED")
        return 0

    src_path = str(ROOT / "src")
    env = os.environ.copy()
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [sys.executable, "-m", "graphrag.cli", "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        failures.append("cli --help failed")
        if result.stderr.strip():
            failures.append(result.stderr.strip())

    probes = {
        "neo4j": (args.skip_neo4j, _check_neo4j_connectivity),
        "llm": (
            args.skip_llm,
            lambda: _check_openai_endpoint(
                os.getenv("VLLM_BASE_URL"),
                os.getenv("VLLM_MODEL_NAME"),
                "generator",
                args.timeout_sec,
            ),
        ),
        "encoder": (
            args.skip_encoder,
            lambda: _check_openai_endpoint(
                os.getenv("GRAPHRAG_EMBED_BASE_URL"),
                os.getenv("GRAPHRAG_EMBED_MODEL"),
                "encoder",
                args.timeout_sec,
            ),
        ),
    }
    for name, (skipped, probe) in probes.items():
        if skipped:
            print(f"SKIPPED {name} (waived on the command line)")
            continue
        ok, detail = probe()
        print(f"{'OK     ' if ok else 'FAILED '}{name}: {detail}")
        if not ok:
            failures.append(f"{name} check failed: {detail}")

    if failures:
        print("SMOKE CHECK FAILED")
        for item in failures:
            print(f"- {item}")
        return 1

    print("SMOKE CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
