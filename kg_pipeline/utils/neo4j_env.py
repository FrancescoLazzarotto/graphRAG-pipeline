"""One place that decides which Neo4j a process talks to.

`GraphDatabase.driver` was constructed in 27 files, and each one had worked
out for itself what a connection is. They disagreed on all of it:

* the variable names — some read ``NEO4J_URI``, some ``NEO4J_URL``, some both,
  and `visualize_kg.py` alone knew about ``NEO4J_DB``;
* what a missing variable means — `os.environ[...]` raised a bare ``KeyError``
  naming one variable, while `os.getenv(..., "")` handed an empty string to
  the driver and produced a connection error instead;
* what the fallback is — the repair passes defaulted the user to ``neo4j``,
  and `kg_evaluator.py` defaulted the URI to the literal placeholder
  ``neo4j+s://<id>.databases.neo4j.io`` with the password ``<password>``.

Which matters more here than it would elsewhere: ``localhost`` does not
identify the target. Ports 7688 and 7689 are both live instances, the hosted
Aura graph serves the demo, and the difference between them is a few
characters in whichever variable that particular file happened to read.

So: one resolver, every accepted spelling, one error that lists them, and a
target object that can say where it points without saying the password.
"""

from __future__ import annotations

import os
from typing import Any, NamedTuple
from urllib.parse import urlsplit

from neo4j import Driver, GraphDatabase

# Both spellings of each name are read, oldest first. They are all in use
# across the repo and in people's `.env` files; picking one and breaking the
# other would be a migration nobody asked for.
URI_VARS = ("NEO4J_URI", "NEO4J_URL")
USER_VARS = ("NEO4J_USER", "NEO4J_USERNAME")
PASSWORD_VARS = ("NEO4J_PASSWORD",)
DATABASE_VARS = ("NEO4J_DATABASE", "NEO4J_DB")


def _first_env(names: tuple[str, ...]) -> str:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return ""


def _or_names(names: tuple[str, ...]) -> str:
    return " or ".join(names)


class Neo4jTarget(NamedTuple):
    """Where a process is about to connect, and how."""

    uri: str
    user: str
    password: str
    database: str | None

    @property
    def auth(self) -> tuple[str, str]:
        return (self.user, self.password)

    @property
    def host(self) -> str:
        """Hostname alone, for deciding whether a target is local."""
        parsed = urlsplit(self.uri if "://" in self.uri else f"//{self.uri}")
        return (parsed.hostname or "").lower()

    @property
    def is_local(self) -> bool:
        return self.host in {"localhost", "127.0.0.1", "::1", ""}

    def session_kwargs(self) -> dict[str, Any]:
        """Kwargs for ``driver.session()``, empty when no database is named."""
        return {"database": self.database} if self.database else {}

    def describe(self) -> str:
        """A line safe to print or log: never the password."""
        return f"{self.uri} (user {self.user}, database {self.database or '<default>'})"


def resolve_target(
    *,
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
    database: str | None = None,
    require: bool = True,
) -> Neo4jTarget:
    """Work out the target from explicit arguments first, then the environment.

    Args:
        uri: Overrides the environment when given and non-empty. Same for the
            three below — a script with a ``--uri`` flag passes it straight in
            rather than writing its own precedence rule.
        user: Overrides ``NEO4J_USER`` / ``NEO4J_USERNAME``.
        password: Overrides ``NEO4J_PASSWORD``.
        database: Overrides ``NEO4J_DATABASE`` / ``NEO4J_DB``. An empty value
            means "the server's default database", not "missing".
        require: Raise when the URI, user or password cannot be found. Pass
            False only to report what is configured without connecting.

    Returns:
        The resolved target.

    Raises:
        ValueError: When ``require`` and something is missing. The message
            names every variable that would have satisfied it.
    """
    resolved_uri = (uri or "").strip() or _first_env(URI_VARS)
    resolved_user = (user or "").strip() or _first_env(USER_VARS)
    resolved_password = password if password else _first_env(PASSWORD_VARS)
    resolved_database = (database or "").strip() or _first_env(DATABASE_VARS)

    if require:
        missing = []
        if not resolved_uri:
            missing.append(_or_names(URI_VARS))
        if not resolved_user:
            missing.append(_or_names(USER_VARS))
        if not resolved_password:
            missing.append(_or_names(PASSWORD_VARS))
        if missing:
            raise ValueError(
                "Missing Neo4j connection settings: "
                + "; ".join(missing)
                + ". Set them in the environment or kg_pipeline/.env."
            )

    return Neo4jTarget(
        uri=resolved_uri,
        user=resolved_user,
        password=resolved_password,
        database=resolved_database or None,
    )


def connect(target: Neo4jTarget | None = None, **overrides: Any) -> Driver:
    """Build a driver for ``target``, resolving one from the environment if absent.

    The driver is a context manager, so the caller keeps the usual
    ``with connect() as driver:`` shape.
    """
    if target is None:
        target = resolve_target(**overrides)
    return GraphDatabase.driver(target.uri, auth=target.auth)
