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
    conda run -n graphllm python scripts/smoke/smoke_check.py
    python scripts/smoke/smoke_check.py --check-imports-only
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

ROOT = Path(__file__).resolve().parents[2]

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


def _probe_graph(uri: str, username: str, password: str, database: str) -> tuple[bool, str]:
    """Whether one graph answers, holds nodes, and has both indexes ONLINE."""
    try:
        from neo4j import GraphDatabase
    except Exception as exc:
        return False, f"neo4j driver unavailable: {exc}"

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
                    "— run scripts/kg/kg_search_index.py / scripts/kg/kg_vector_index.py"
                )
    except Exception as exc:
        return False, str(exc)
    finally:
        driver.close()

    return True, f"{uri} — {nodes} nodes, indexes online"


def _check_neo4j_connectivity() -> tuple[bool, str]:
    """Pass if *either* graph the demo can use is healthy, and say which.

    The demo falls back from the hosted Aura instance to the local mirror when
    the first is unreachable — Aura Free suspends itself after three idle days.
    The preflight did not know that, so it failed the launch in exactly the
    outage the fallback exists for, and `start_demo.sh` stopped before starting
    a demo that would have worked. Passing on the fallback is not silent: the
    line says which graph answered, because a session served by the mirror
    during an outage is not the same thing as a healthy one.
    """
    primary = (
        os.getenv("NEO4J_URL"),
        os.getenv("NEO4J_USERNAME"),
        os.getenv("NEO4J_PASSWORD"),
        os.getenv("NEO4J_DATABASE", "neo4j"),
    )
    missing = [
        key
        for key, value in zip(("NEO4J_URL", "NEO4J_USERNAME", "NEO4J_PASSWORD"), primary)
        if not value
    ]
    if missing:
        primary_detail = f"missing environment variables: {', '.join(missing)}"
    else:
        ok, primary_detail = _probe_graph(*primary)  # type: ignore[arg-type]
        if ok:
            return True, f"primary {primary_detail}"

    fallback_url = (os.getenv("DEMO_NEO4J_FALLBACK_URL") or "").strip()
    if not fallback_url:
        return False, (
            f"primary unusable ({primary_detail}) and no fallback configured "
            "— set DEMO_NEO4J_FALLBACK_URL / _USERNAME / _PASSWORD / _DATABASE"
        )
    # An unset fallback database is not "no database": the driver would read
    # NEO4J_DATABASE and the mirror would inherit Aura's name. Same default as
    # product/config.build_kg_manager, so the two agree on what they probe.
    ok, fallback_detail = _probe_graph(
        fallback_url,
        os.getenv("DEMO_NEO4J_FALLBACK_USERNAME") or "",
        os.getenv("DEMO_NEO4J_FALLBACK_PASSWORD") or "",
        (os.getenv("DEMO_NEO4J_FALLBACK_DATABASE") or "neo4j").strip() or "neo4j",
    )
    if ok:
        return True, (
            f"PRIMARY DOWN ({primary_detail}) — running on the fallback: {fallback_detail}"
        )
    return False, (
        f"no usable graph. primary: {primary_detail}. fallback: {fallback_detail}"
    )


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


def _check_generators(args: argparse.Namespace) -> tuple[bool, str]:
    """Probe every generator the caller named, or the one the environment names.

    Args:
        args: Parsed command line; reads ``llm_base_url``, ``llm_model`` and
            ``timeout_sec``.

    Returns:
        Whether every endpoint answered, and one detail line per endpoint.
    """
    urls = args.llm_base_url or [os.getenv("VLLM_BASE_URL")]
    # The identity assertion belongs to a caller who states which model they
    # expect. It used to fall back to VLLM_MODEL_NAME, which does not mean
    # that: in kg_pipeline/.env that variable pins the model the *ingestion*
    # pipeline extracts with — the one the current graph was built by — while
    # the demo probes its endpoints and answers with whatever is served. The two
    # diverged when serving moved to Qwen3.8-27B on 2026-08-26 and the ingestion
    # pin stayed, so this check reported FAILED on a healthy demo. Reading the
    # ingestion pin as a serving requirement made the documented health check
    # lie, and the fix is not to edit that pin: changing it would silently
    # change which model a future rebuild extracts with.
    expected = None if args.llm_base_url else args.llm_model

    results = [
        _check_openai_endpoint(url, expected, "generator", args.timeout_sec)
        for url in urls
    ]
    return all(ok for ok, _ in results), "; ".join(detail for _, detail in results)


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
    parser.add_argument(
        "--llm-base-url",
        action="append",
        default=None,
        metavar="URL",
        help=(
            "Generator endpoint to probe; repeatable, once per running generator. "
            "Defaults to VLLM_BASE_URL. Passing it also drops the model-identity "
            "assertion, which only ever held for the single pinned generator: the "
            "launcher serves eight models across two ports, so a caller that knows "
            "which port it started is checking reachability, not identity"
        ),
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help=(
            "Model id required at the endpoint. Unset, any served model passes: "
            "the demo probes its endpoints and answers with whatever is up. Only "
            "applies when --llm-base-url is not given"
        ),
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
            lambda: _check_generators(args),
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
