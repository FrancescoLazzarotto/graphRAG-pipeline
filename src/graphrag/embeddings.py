"""Multilingual sentence encoder shared by the KG vector index and the retriever.

The graph was extracted from a bilingual corpus and most concepts carry their
Italian surface form, while the questions are English; lexical retrieval cannot
cross that gap (``exp_results/KG_VS_RETRIEVAL.md``). Encoding node names and
questions into one multilingual space is the bridge, so index build time and
query time must use the *same* model, prefixes and pooling — a mismatch there
degrades similarity silently instead of failing. Hence a single module, and a
single transport.

That transport is an OpenAI-compatible ``/v1/embeddings`` endpoint, the pattern
this project already uses for generation::

    CUDA_VISIBLE_DEVICES=1 vllm serve intfloat/multilingual-e5-base \\
        --runner pooling --port 8002 --gpu-memory-utilization 0.12 \\
        --max-model-len 512

Running the encoder in-process was the obvious alternative and does not work
here: the ``graphllm`` environment pairs torch 2.5.1 with torchvision 0.25.0, so
importing any ``transformers`` text model dies with ``operator torchvision::nms
does not exist`` (the torch/torchvision row in CLAUDE.md). Serving it from the
``vllm-serve`` virtualenv keeps both environments untouched.

The e5 family is trained with ``query:`` / ``passage:`` prefixes; they are not
optional and are applied here so no caller can forget one.
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

import requests

logger = logging.getLogger("graphrag")

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

DEFAULT_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_BASE_URL = "http://localhost:8002/v1"


class EmbeddingUnavailable(RuntimeError):
    """The embedding endpoint could not be reached or returned an error."""


def base_url() -> str:
    """Endpoint from ``GRAPHRAG_EMBED_BASE_URL``, else the local default."""
    return os.getenv("GRAPHRAG_EMBED_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def model_id() -> str:
    """Encoder id from ``GRAPHRAG_EMBED_MODEL``, else the e5 default."""
    return os.getenv("GRAPHRAG_EMBED_MODEL", DEFAULT_MODEL)


def encode(
    texts: Sequence[str],
    prefix: str,
    model: str | None = None,
    url: str | None = None,
    batch_size: int = 256,
    timeout: float = 120.0,
) -> list[list[float]]:
    """Embed ``texts``, in input order.

    Args:
        texts: Strings to encode.
        prefix: :data:`QUERY_PREFIX` or :data:`PASSAGE_PREFIX`.
        model: Encoder id; must match the one the index was built with.
        url: Base URL of the OpenAI-compatible endpoint.
        batch_size: Texts per HTTP request.
        timeout: Per-request timeout in seconds.

    Returns:
        One vector per input.

    Raises:
        EmbeddingUnavailable: The endpoint is unreachable or replied with an
            error, so the caller can fall back to lexical retrieval instead of
            silently returning nothing.
    """
    if not texts:
        return []

    endpoint = f"{url or base_url()}/embeddings"
    name = model or model_id()
    out: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = [f"{prefix}{text}" for text in texts[start : start + batch_size]]
        try:
            response = requests.post(
                endpoint,
                json={"model": name, "input": batch},
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise EmbeddingUnavailable(f"{endpoint}: {exc}") from exc

        rows = payload.get("data") or []
        if len(rows) != len(batch):
            raise EmbeddingUnavailable(
                f"{endpoint}: asked for {len(batch)} embeddings, got {len(rows)}"
            )
        # The API is not required to preserve order, and silently mispairing a
        # vector with its node would poison the index in a way no later check
        # would catch.
        for row in sorted(rows, key=lambda item: item.get("index", 0)):
            out.append([float(value) for value in row["embedding"]])
    return out


def encode_query(
    text: str, model: str | None = None, url: str | None = None
) -> list[float]:
    """Embed one question with the query prefix."""
    vectors = encode([text], QUERY_PREFIX, model=model, url=url)
    return vectors[0] if vectors else []


def available(url: str | None = None, timeout: float = 5.0) -> bool:
    """Whether the endpoint answers, for a cheap pre-flight check."""
    try:
        response = requests.get(f"{url or base_url()}/models", timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False
