from __future__ import annotations

import logging
import os
from typing import Any

from evalkit.judge.base import JudgeBackend

logger = logging.getLogger("graphrag")


class VLLMBackend:
    """Judge backend using an OpenAI-compatible vLLM endpoint.

    Reuses the same connection pattern as LLMManager in src/graphrag/llm/manager.py.
    """

    def __init__(
        self,
        model_id: str,
        base_url: str = "",
        api_key: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        self.api_key = api_key or os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from langchain_openai import ChatOpenAI  # type: ignore

            self._client = ChatOpenAI(
                model=self.model_id,
                base_url=self.base_url,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
            )
        return self._client

    def complete(self, system: str, user: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore

        client = self._get_client()
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        for attempt in range(3):
            try:
                output = client.invoke(messages)
                return str(output.content).strip()
            except Exception as exc:
                logger.warning("VLLMBackend attempt %d failed: %s", attempt + 1, exc)
        return ""


class LocalHFBackend:
    """Judge backend using a local HuggingFace model.

    Wraps LLMManager from src/graphrag/llm/manager.py to reuse retry logic.
    """

    def __init__(self, model_id: str, max_new_tokens: int = 256) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._llm: Any = None

    def _get_llm(self) -> Any:
        if self._llm is None:
            import sys
            from pathlib import Path

            # Ensure src/ is importable
            src_path = Path(__file__).resolve().parents[4] / "src"
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))

            from graphrag.llm.manager import LLMManager  # type: ignore

            mgr = LLMManager(
                model_id=self.model_id,
                max_new_tokens=self.max_new_tokens,
                use_vllm=False,
            )
            self._llm = mgr.load_llm()
        return self._llm

    def complete(self, system: str, user: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore

        llm = self._get_llm()
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        for attempt in range(3):
            try:
                output = llm.invoke(messages)
                return str(output.content if hasattr(output, "content") else output).strip()
            except Exception as exc:
                logger.warning("LocalHFBackend attempt %d failed: %s", attempt + 1, exc)
        return ""


class APIBackend:
    """Judge backend using an external API (Anthropic or OpenAI).

    Requires the appropriate SDK to be installed and API key set.
    """

    def __init__(
        self,
        model_id: str,
        provider: str = "anthropic",
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        self.model_id = model_id
        self.provider = provider.lower()
        self.max_tokens = max_tokens
        self.temperature = temperature

    def complete(self, system: str, user: str) -> str:
        if self.provider == "anthropic":
            return self._anthropic(system, user)
        if self.provider == "openai":
            return self._openai(system, user)
        raise ValueError(f"Unknown API provider: {self.provider!r}")

    def _anthropic(self, system: str, user: str) -> str:
        try:
            import anthropic  # type: ignore
        except ImportError as exc:
            raise ImportError("Install anthropic: pip install anthropic") from exc

        client = anthropic.Anthropic()
        for attempt in range(3):
            try:
                message = client.messages.create(
                    model=self.model_id,
                    max_tokens=self.max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return str(message.content[0].text).strip()
            except Exception as exc:
                logger.warning("APIBackend (anthropic) attempt %d failed: %s", attempt + 1, exc)
        return ""

    def _openai(self, system: str, user: str) -> str:
        try:
            import openai  # type: ignore
        except ImportError as exc:
            raise ImportError("Install openai: pip install openai") from exc

        client = openai.OpenAI()
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=self.model_id,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                return str(response.choices[0].message.content or "").strip()
            except Exception as exc:
                logger.warning("APIBackend (openai) attempt %d failed: %s", attempt + 1, exc)
        return ""


class ClaudeCodeBackend:
    """Judge backend driving the Claude Code CLI (`claude -p`) headless.

    Uses the Claude Code subscription auth (Pro/Max) rather than a metered API
    key: the `claude` binary must be installed and logged in to a claude.ai
    account. This is the only way to use the subscription programmatically.

    The user prompt is piped via stdin (avoids arg-length/quoting limits on
    large prompts); the system prompt is appended via --append-system-prompt.

    Env overrides:
        CLAUDE_CODE_BIN: path to the claude binary (default "claude").
        CLAUDE_CODE_TIMEOUT: per-call timeout in seconds (default 300).
        CLAUDE_CODE_EXTRA_ARGS: extra CLI args, whitespace-separated.
    """

    def __init__(
        self,
        model_id: str = "sonnet",
        max_tokens: int = 256,  # noqa: ARG002 - kept for backend interface parity
        bin_path: str = "",
        timeout: int = 300,
    ) -> None:
        self.model_id = model_id or "sonnet"
        self.bin = bin_path or os.getenv("CLAUDE_CODE_BIN", "claude")
        self.timeout = int(os.getenv("CLAUDE_CODE_TIMEOUT", str(timeout)))
        extra = os.getenv("CLAUDE_CODE_EXTRA_ARGS", "").strip()
        self.extra_args = extra.split() if extra else []

    def _backoff(self, attempt: int) -> None:
        import random
        import time

        time.sleep(min(2.0**attempt + random.random(), 30.0))

    def complete(self, system: str, user: str) -> str:
        import json as _json
        import subprocess

        cmd = [self.bin, "-p", "--output-format", "json", "--model", self.model_id, "--max-turns", "1"]
        if system:
            cmd += ["--append-system-prompt", system]
        cmd += self.extra_args

        for attempt in range(3):
            try:
                proc = subprocess.run(
                    cmd,
                    input=user,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"claude CLI not found ({self.bin!r}). Install Claude Code and run "
                    "`claude login` with your Pro/Max account, or set CLAUDE_CODE_BIN."
                ) from exc
            except subprocess.TimeoutExpired:
                logger.warning("ClaudeCodeBackend attempt %d timed out", attempt + 1)
                self._backoff(attempt)
                continue

            if proc.returncode != 0:
                logger.warning(
                    "ClaudeCodeBackend attempt %d exit=%d stderr=%s",
                    attempt + 1, proc.returncode, proc.stderr[:200],
                )
                self._backoff(attempt)
                continue

            out = proc.stdout.strip()
            try:
                payload = _json.loads(out)
            except _json.JSONDecodeError:
                return out  # already plain text
            if isinstance(payload, dict):
                if payload.get("is_error"):
                    logger.warning("ClaudeCodeBackend is_error: %s", str(payload)[:200])
                    self._backoff(attempt)
                    continue
                return str(payload.get("result", "")).strip()
            return out
        return ""


def make_backend(
    backend: str,
    model_id: str,
    vllm_base_url: str = "",
    vllm_api_key: str = "",
    api_provider: str = "anthropic",
    max_new_tokens: int = 256,
    claude_code_bin: str = "",
) -> JudgeBackend:
    """Factory: return the appropriate JudgeBackend based on *backend* string."""
    if backend == "vllm":
        return VLLMBackend(
            model_id=model_id,
            base_url=vllm_base_url,
            api_key=vllm_api_key,
            max_new_tokens=max_new_tokens,
        )
    if backend == "local_hf":
        return LocalHFBackend(model_id=model_id, max_new_tokens=max_new_tokens)
    if backend == "api":
        return APIBackend(
            model_id=model_id,
            provider=api_provider,
            max_tokens=max_new_tokens,
        )
    if backend == "claude_code":
        return ClaudeCodeBackend(
            model_id=model_id,
            max_tokens=max_new_tokens,
            bin_path=claude_code_bin,
        )
    raise ValueError(
        f"Unknown backend: {backend!r}. Choose from: vllm, local_hf, api, claude_code"
    )
