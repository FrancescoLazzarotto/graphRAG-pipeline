from __future__ import annotations

import importlib.metadata
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

import torch

from graphrag.config import AgentConfig, DEFAULT_MODEL_ID
from graphrag.llm.prompts import PromptLibrary

logger = logging.getLogger("graphrag")


class LLMManager:
    _LARGE_MODEL_THRESHOLD_B = 30.0

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        warmup: bool = False,
        max_new_tokens: int = 256,
        gpu_memory_fraction: float = 0.92,
        allow_large_model_fp16_fallback: bool = False,
    ) -> None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if gpu_memory_fraction <= 0 or gpu_memory_fraction > 1:
            raise ValueError("gpu_memory_fraction must be in (0, 1]")

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.gpu_memory_fraction = gpu_memory_fraction
        env_allow_fallback = os.getenv("GRAPHRAG_ALLOW_LARGE_MODEL_FP16_FALLBACK", "").strip().lower()
        self.allow_large_model_fp16_fallback = allow_large_model_fp16_fallback or env_allow_fallback in {
            "1",
            "true",
            "yes",
            "on",
        }

        self._cached_model: Any | None = None
        self._cached_model_id: str | None = None
        self._load_lock = threading.Lock()

        if warmup:
            self.load_llm()

    @staticmethod
    def _import_hf_stack() -> tuple[Any, Any, Any, Any, Any, Any]:
        try:
            from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
        except Exception as exc:  # pragma: no cover - depends on runtime env
            text = str(exc).lower()
            if "huggingface-hub" in text and "required" in text:
                raise RuntimeError(
                    "Incompatible transformers/huggingface-hub versions detected. "
                    "Fix with: conda run -n graphllm python -m pip install \"huggingface-hub>=0.34.0,<1.0\""
                ) from exc
            raise

        return ChatHuggingFace, HuggingFacePipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

    @staticmethod
    def _hf_token() -> str | None:
        return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    @staticmethod
    def _model_size_billions(model_id: str) -> float | None:
        match = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?:\b|[-_/])", model_id)
        if match is None:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    @classmethod
    def _is_large_model(cls, model_id: str) -> bool:
        size_b = cls._model_size_billions(model_id)
        return size_b is not None and size_b >= cls._LARGE_MODEL_THRESHOLD_B

    def _build_max_memory(self) -> dict[int | str, str] | None:
        if not torch.cuda.is_available():
            return None

        gpu_count = torch.cuda.device_count()
        if gpu_count < 1:
            return None

        max_memory: dict[int | str, str] = {}
        for index in range(gpu_count):
            total_gib = torch.cuda.get_device_properties(index).total_memory / (1024**3)
            usable_gib = max(1, int(total_gib * self.gpu_memory_fraction))
            max_memory[index] = f"{usable_gib}GiB"

        try:
            cpu_offload_gib = int(os.getenv("GRAPHRAG_CPU_OFFLOAD_GIB", "64"))
        except ValueError:
            cpu_offload_gib = 64
        max_memory["cpu"] = f"{max(4, cpu_offload_gib)}GiB"
        return max_memory

    @staticmethod
    def _offload_folder() -> str:
        offload_dir = Path(os.getenv("GRAPHRAG_OFFLOAD_DIR", "/tmp/graphrag-offload"))
        offload_dir.mkdir(parents=True, exist_ok=True)
        return str(offload_dir)

    @staticmethod
    def _fp16_fallback_message(model_id: str, root_exc: BaseException) -> RuntimeError:
        return RuntimeError(
            "4-bit quantized load failed for large model '"
            + model_id
            + "'. To keep production reliability, fp16 fallback is disabled for large models by default. "
            + "Set --allow-large-model-fp16-fallback (or GRAPHRAG_ALLOW_LARGE_MODEL_FP16_FALLBACK=1) "
            + "if you explicitly want this fallback. Root cause: "
            + str(root_exc)
        )

    @staticmethod
    def _is_hf_auth_error(exc: BaseException) -> bool:
        details: list[str] = []
        current: BaseException | None = exc
        depth = 0
        while current is not None and depth < 6:
            details.append(f"{type(current).__name__}: {current}")
            current = current.__cause__ or current.__context__
            depth += 1

        text = " ".join(details).lower()
        markers = (
            "gated repo",
            "cannot access gated repo",
            "unauthorized",
            "401",
            "localtokennotfounderror",
            "access to model",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _raise_hf_access_error(model_id: str, exc: BaseException) -> None:
        raise RuntimeError(
            "Cannot load Hugging Face model '"
            + model_id
            + "': access denied or authentication missing.\n"
            + "If the model is gated, request access at https://huggingface.co/"
            + model_id
            + "\n"
            + "Fast path (recommended): export HF_TOKEN and rerun the same command.\n"
            + "  export HF_TOKEN=\"<your-hf-token>\"\n"
            + "Optional persistent login from this conda env:\n"
            + "  $CONDA_PREFIX/bin/python -m huggingface_hub.commands.huggingface_cli login --token \"$HF_TOKEN\"\n"
            + "You can also use HUGGINGFACE_HUB_TOKEN instead of HF_TOKEN.\n"
            + "You can also switch to an ungated model, for example Qwen/Qwen2.5-7B-Instruct."
        ) from exc

    def _build_llm(self, model_id: str) -> Any:
        (
            ChatHuggingFace,
            HuggingFacePipeline,
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            hf_pipeline,
        ) = self._import_hf_stack()

        logger.info("Loading LLM model: %s", model_id)
        hf_token = self._hf_token()
        model_is_large = self._is_large_model(model_id)

        common_load_kwargs: dict[str, Any] = {
            "token": hf_token,
            "low_cpu_mem_usage": True,
        }
        if torch.cuda.is_available():
            common_load_kwargs["device_map"] = "auto"
            max_memory = self._build_max_memory()
            if max_memory:
                common_load_kwargs["max_memory"] = max_memory
                common_load_kwargs["offload_folder"] = self._offload_folder()

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            if torch.cuda.is_available():
                try:
                    importlib.metadata.version("bitsandbytes")
                    base_model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        quantization_config=BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                        ),
                        **common_load_kwargs,
                    )
                except importlib.metadata.PackageNotFoundError:
                    if model_is_large and not self.allow_large_model_fp16_fallback:
                        raise RuntimeError(
                            "bitsandbytes is required for large models (>=30B) in this production profile. "
                            "Install bitsandbytes or use a smaller model, or explicitly allow fp16 fallback."
                        )
                    logger.warning("bitsandbytes not installed: loading model on GPU without 4-bit quantization.")
                    base_model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16,
                        **common_load_kwargs,
                    )
                except Exception as exc:  # pragma: no cover - depends on GPU/runtime setup
                    if self._is_hf_auth_error(exc):
                        raise
                    if model_is_large and not self.allow_large_model_fp16_fallback:
                        raise self._fp16_fallback_message(model_id=model_id, root_exc=exc) from exc
                    logger.warning(
                        "bitsandbytes is installed, but 4-bit loading failed (%s). "
                        "Falling back to standard fp16 GPU loading.",
                        exc,
                    )
                    base_model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16,
                        **common_load_kwargs,
                    )
            else:
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    **common_load_kwargs,
                )
        except Exception as exc:
            if self._is_hf_auth_error(exc):
                self._raise_hf_access_error(model_id=model_id, exc=exc)
            raise

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "return_full_text": False,
            "repetition_penalty": 1.05,
        }
        if tokenizer.pad_token_id is not None:
            generation_kwargs["pad_token_id"] = tokenizer.pad_token_id

        generation = hf_pipeline(
            "text-generation",
            model=base_model,
            tokenizer=tokenizer,
            **generation_kwargs,
        )
        return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=generation))

    def load_llm(self, model_id: str | None = None) -> Any:
        target_model_id = model_id or self.model_id

        if self._cached_model is not None and self._cached_model_id == target_model_id:
            return self._cached_model

        with self._load_lock:
            if self._cached_model is not None and self._cached_model_id == target_model_id:
                return self._cached_model

            self._cached_model = self._build_llm(target_model_id)
            self._cached_model_id = target_model_id
            self.model_id = target_model_id
            return self._cached_model

    def warmup(self) -> None:
        self.load_llm()

    def generate(self, query: str, context: str, config: AgentConfig) -> dict[str, str]:
        prompt = PromptLibrary.answer_prompt(config)
        rendered = prompt.invoke({
            "question": query,
            "context": context,
        })
        model = self.load_llm()
        output = model.invoke(rendered)
        answer = str(output.content if hasattr(output, "content") else output).strip()
        return {"answer": answer}
    
        