from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path


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
        key for key, value in (
            ("NEO4J_URL", uri),
            ("NEO4J_USERNAME", username),
            ("NEO4J_PASSWORD", password),
        ) if not value
    ]
    if missing:
        return False, f"missing environment variables: {', '.join(missing)}"

    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        with driver.session(database=database) as session:
            row = session.run("RETURN 1 AS ok").single()
            if not row or row.get("ok") != 1:
                return False, "unexpected neo4j preflight result"
    except Exception as exc:
        return False, str(exc)
    finally:
        driver.close()

    return True, "ok"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal environment smoke checks")
    parser.add_argument(
        "--check-neo4j",
        action="store_true",
        help="Also validate Neo4j connectivity with current env vars",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    required_modules = [
        "torch",
        "langgraph",
        "transformers",
        "neo4j",
    ]

    failures: list[str] = []
    for module_name in required_modules:
        ok, reason = _check_import(module_name)
        if not ok:
            failures.append(f"import {module_name}: {reason}")

    src_path = str(Path(__file__).resolve().parents[1] / "src")
    env = os.environ.copy()
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [sys.executable, "-m", "graphrag.cli", "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        failures.append("cli --help failed")
        if result.stderr.strip():
            failures.append(result.stderr.strip())

    if args.check_neo4j:
        ok, reason = _check_neo4j_connectivity()
        if not ok:
            failures.append(f"neo4j connectivity check failed: {reason}")

    if failures:
        print("SMOKE CHECK FAILED")
        for item in failures:
            print(f"- {item}")
        return 1

    print("SMOKE CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
