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


def make_backend(
    backend: str,
    model_id: str,
    vllm_base_url: str = "",
    vllm_api_key: str = "",
    api_provider: str = "anthropic",
    max_new_tokens: int = 256,
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
    raise ValueError(f"Unknown backend: {backend!r}. Choose from: vllm, local_hf, api")
