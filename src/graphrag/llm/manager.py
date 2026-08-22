from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import torch

from graphrag import questions
from graphrag.config import AgentConfig, DEFAULT_MODEL_ID
from graphrag.llm.prompts import PromptLibrary
from graphrag.llm.refusal import looks_like_refusal

logger = logging.getLogger("graphrag")

# Orthographic markers of Italian used to break ties on short questions, where
# function words alone are too thin a signal (WP5).
_ITALIAN_ACCENTED = re.compile(r"[àèéìòù]")
_ITALIAN_ELISION = re.compile(r"\b(?:l|dell|nell|all|sull|dall|un|quell|c|d)'")


class LLMManager:
    _LARGE_MODEL_THRESHOLD_B = 30.0

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        warmup: bool = False,
        max_new_tokens: int = 256,
        gpu_memory_fraction: float = 0.92,
        allow_large_model_fp16_fallback: bool = False,
        use_vllm: bool = False,
        vllm_base_url: str = "http://localhost:8000/v1",
    ) -> None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if gpu_memory_fraction <= 0 or gpu_memory_fraction > 1:
            raise ValueError("gpu_memory_fraction must be in (0, 1]")
        if not vllm_base_url.strip():
            raise ValueError("vllm_base_url must be non-empty")

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.gpu_memory_fraction = gpu_memory_fraction
        self.use_vllm = bool(use_vllm)
        self.vllm_base_url = vllm_base_url.strip().rstrip("/")
        self.vllm_api_key = (
            os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
        )
        env_allow_fallback = (
            os.getenv("GRAPHRAG_ALLOW_LARGE_MODEL_FP16_FALLBACK", "").strip().lower()
        )
        self.allow_large_model_fp16_fallback = (
            allow_large_model_fp16_fallback
            or env_allow_fallback
            in {
                "1",
                "true",
                "yes",
                "on",
            }
        )

        self._cached_model: Any | None = None
        self._cached_model_id: str | None = None
        self._load_lock = threading.Lock()
        self._vllm_endpoint_checked = False

        try:
            self.generate_retry_attempts = max(
                1, int(os.getenv("GRAPHRAG_LLM_GENERATE_RETRIES", "2"))
            )
        except ValueError:
            self.generate_retry_attempts = 2
        try:
            self.generate_retry_backoff_sec = max(
                0.0, float(os.getenv("GRAPHRAG_LLM_GENERATE_RETRY_BACKOFF_SEC", "1.0"))
            )
        except ValueError:
            self.generate_retry_backoff_sec = 1.0

        if self.use_vllm:
            logger.info(
                "vLLM mode enabled: gpu_memory_fraction and allow_large_model_fp16_fallback are ignored "
                "for client-side inference (server controls memory/precision)."
            )

        if warmup:
            self.warmup()

    @staticmethod
    def _import_vllm_stack() -> Any:
        try:
            from langchain_openai import ChatOpenAI
        except Exception as exc:
            raise RuntimeError(
                "vLLM mode requires langchain-openai. Install it in your runtime env, for example: "
                'conda run -n graphllm python -m pip install "langchain-openai>=0.2,<0.4"'
            ) from exc

        return ChatOpenAI

    @staticmethod
    def _import_hf_stack() -> tuple[Any, Any, Any, Any, Any, Any]:
        try:
            from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                pipeline,
            )
        except Exception as exc:  # pragma: no cover - depends on runtime env
            text = str(exc).lower()
            if "huggingface-hub" in text and "required" in text:
                raise RuntimeError(
                    "Incompatible transformers/huggingface-hub versions detected. "
                    'Fix with: conda run -n graphllm python -m pip install "huggingface-hub>=0.34.0,<1.0"'
                ) from exc
            raise

        return (
            ChatHuggingFace,
            HuggingFacePipeline,
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            pipeline,
        )

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

    @staticmethod
    def _is_awq_model(model_id: str) -> bool:
        return bool(re.search(r"[-_/]awq(?:[-_/]|$)", model_id, re.IGNORECASE))

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
            + '  export HF_TOKEN="<your-hf-token>"\n'
            + "Optional persistent login from this conda env:\n"
            + '  $CONDA_PREFIX/bin/python -m huggingface_hub.commands.huggingface_cli login --token "$HF_TOKEN"\n'
            + "You can also use HUGGINGFACE_HUB_TOKEN instead of HF_TOKEN.\n"
            + "You can also switch to an ungated model, for example Qwen/Qwen2.5-7B-Instruct."
        ) from exc

    def _build_llm(self, model_id: str) -> Any:
        if self.use_vllm:
            return self._build_vllm_llm(model_id)

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

            # Enable Flash Attention 2 if flash-attn package is installed (A40/Ampere+ supported)
            try:
                importlib.metadata.version("flash-attn")
                common_load_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Flash Attention 2 enabled.")
            except importlib.metadata.PackageNotFoundError:
                pass

        torch_compile = (
            os.getenv("GRAPHRAG_TORCH_COMPILE", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )

        def _load_model_fa2_safe(**kw: Any) -> Any:
            """Load model; strip flash_attention_2 and retry if architecture doesn't support it."""
            try:
                return AutoModelForCausalLM.from_pretrained(model_id, **kw)
            except (ValueError, NotImplementedError) as exc:
                if "attn_implementation" in kw and "flash" in str(exc).lower():
                    kw.pop("attn_implementation")
                    logger.warning(
                        "Flash Attention 2 not supported for '%s', retrying without it: %s",
                        model_id,
                        exc,
                    )
                    return AutoModelForCausalLM.from_pretrained(model_id, **kw)
                raise

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            if torch.cuda.is_available():
                if self._is_awq_model(model_id):
                    try:
                        importlib.metadata.version("autoawq")
                    except importlib.metadata.PackageNotFoundError:
                        raise RuntimeError(
                            "autoawq is required to load AWQ models. "
                            "Install it with: pip install autoawq>=0.2"
                        ) from None
                    # AWQ model: weights already quantized on HuggingFace; skip bitsandbytes
                    logger.info("AWQ model detected; loading in fp16 (skipping bitsandbytes).")
                    base_model = _load_model_fa2_safe(
                        torch_dtype=torch.float16,
                        **common_load_kwargs,
                    )
                    if torch_compile:
                        logger.info("Applying torch.compile(mode='reduce-overhead') to AWQ model.")
                        base_model = torch.compile(base_model, mode="reduce-overhead")
                else:
                    try:
                        importlib.metadata.version("bitsandbytes")
                        base_model = _load_model_fa2_safe(
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
                        logger.warning(
                            "bitsandbytes not installed: loading model on GPU without 4-bit quantization."
                        )
                        base_model = _load_model_fa2_safe(
                            torch_dtype=torch.float16,
                            **common_load_kwargs,
                        )
                        if torch_compile:
                            logger.info("Applying torch.compile(mode='reduce-overhead').")
                            base_model = torch.compile(base_model, mode="reduce-overhead")
                    except (
                        Exception
                    ) as exc:  # pragma: no cover - depends on GPU/runtime setup
                        if self._is_hf_auth_error(exc):
                            raise
                        if model_is_large and not self.allow_large_model_fp16_fallback:
                            raise self._fp16_fallback_message(
                                model_id=model_id, root_exc=exc
                            ) from exc
                        logger.warning(
                            "bitsandbytes is installed, but 4-bit loading failed (%s). "
                            "Falling back to standard fp16 GPU loading.",
                            exc,
                        )
                        base_model = _load_model_fa2_safe(
                            torch_dtype=torch.float16,
                            **common_load_kwargs,
                        )
                        if torch_compile:
                            logger.info("Applying torch.compile(mode='reduce-overhead').")
                            base_model = torch.compile(base_model, mode="reduce-overhead")
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

    def classify_in_domain(self, question: str, config: Any) -> bool:
        """Whether the question belongs to the corpus domain.

        One classification call before retrieval, answering a single word.

        The prompt asks for one token, but nothing enforces it: `_build_vllm_llm`
        binds `max_tokens=self.max_new_tokens` on the shared client, so the gate
        call carries the full answer budget and a reasoning model will spend it
        on a <think> block. The parsing below is written for that.

        Args:
            question: The question as typed.
            config: The agent config, read for ``domain_scope``.

        Returns:
            ``True`` when in domain, and on any failure — a broken gate must not
            silence a working demo.
        """
        prompt = PromptLibrary.domain_gate_prompt(getattr(config, "domain_scope", ""))
        try:
            model = self.load_llm()
            output = self._invoke_with_retry(
                model, prompt.invoke({"question": question})
            )
        except Exception as exc:
            logger.warning("Domain gate failed (%s); treating question as in domain", exc)
            return True

        verdict = str(output.content if hasattr(output, "content") else output).strip()
        logger.info("Domain gate: %r -> %s", question[:80], verdict[:64])
        # `startswith("OUT")` read only the first three characters, so any
        # preamble flipped a refusal into an acceptance — and three of the six
        # campaign generators are reasoning models that open with a <think>
        # block. Strip the reasoning block, then look for the verdict as a whole
        # word anywhere in what remains. See docs/code_audit_2026-08-15.md §1.6.
        cleaned = re.sub(
            r"<think>.*?</think>", " ", verdict, flags=re.DOTALL | re.IGNORECASE
        )
        cleaned = re.sub(r"<think>.*\Z", " ", cleaned, flags=re.DOTALL | re.IGNORECASE)
        upper = cleaned.upper()
        out_of_domain = re.search(r"\bOUT(?:[_\- ]?OF[_\- ]?DOMAIN)?\b", upper)
        in_domain = re.search(r"\bIN(?:[_\- ]?DOMAIN)?\b", upper)
        if out_of_domain and not in_domain:
            return False
        if out_of_domain and in_domain:
            # Both words present: trust whichever the model said last, which is
            # its conclusion rather than its restatement of the options.
            return in_domain.start() > out_of_domain.start()
        return True

    def _build_vllm_llm(self, model_id: str) -> Any:
        ChatOpenAI = self._import_vllm_stack()
        logger.info(
            "Using vLLM OpenAI-compatible endpoint at %s for model %s",
            self.vllm_base_url,
            model_id,
        )
        # Without an explicit timeout the OpenAI SDK waits 600 s and retries
        # twice on its own, under this class's own retry loop: a vLLM server
        # that is wedged rather than down left an interactive session hanging
        # on a spinner for the better part of an hour. Generous on purpose —
        # 2 048 tokens out of a 32B model on one A40 is a couple of minutes,
        # and a cap that fires on a slow-but-working answer is worse than the
        # hang it prevents. The worst case is this value times
        # GRAPHRAG_LLM_GENERATE_RETRIES, since a timeout counts as transient.
        # The KG pipeline has its own, much longer budget (VLLM_HTTP_TIMEOUT):
        # nobody is watching a batch run.
        timeout_sec = float(os.getenv("GRAPHRAG_LLM_HTTP_TIMEOUT_SEC", "300"))
        return ChatOpenAI(
            model=model_id,
            base_url=self.vllm_base_url,
            api_key=self.vllm_api_key,
            temperature=0,
            max_tokens=self.max_new_tokens,
            timeout=timeout_sec,
            # This class already retries transient failures with backoff;
            # the SDK's own retries only multiply the wait.
            max_retries=0,
        )

    @staticmethod
    def _models_url(base_url: str) -> str:
        return urllib.parse.urljoin(base_url.rstrip("/") + "/", "models")

    def _check_vllm_endpoint(self, target_model_id: str) -> None:
        models_url = self._models_url(self.vllm_base_url)
        request = urllib.request.Request(models_url, method="GET")
        if self.vllm_api_key:
            request.add_header("Authorization", f"Bearer {self.vllm_api_key}")

        timeout_sec = float(os.getenv("GRAPHRAG_VLLM_HEALTHCHECK_TIMEOUT_SEC", "5"))
        try:
            with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                status_code = int(getattr(response, "status", 200))
                if status_code >= 400:
                    raise RuntimeError(
                        f"vLLM endpoint returned HTTP {status_code} on {models_url}"
                    )
                payload = response.read().decode("utf-8", errors="ignore")
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            raise RuntimeError(
                "Cannot reach vLLM endpoint at '"
                + models_url
                + "'. Ensure server is running and reachable from this process. "
                + "If using sbatch mode, enable USE_VLLM=1 and verify port forwarding/bind address."
            ) from exc

        reported_models: list[str] = []
        try:
            parsed = json.loads(payload)
            data = parsed.get("data", []) if isinstance(parsed, dict) else []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and isinstance(item.get("id"), str):
                        reported_models.append(item["id"])
        except json.JSONDecodeError:
            reported_models = []

        if reported_models and target_model_id not in reported_models:
            preview = ", ".join(reported_models[:5])
            logger.warning(
                "Requested model '%s' not listed by vLLM /models. Reported models: %s",
                target_model_id,
                preview,
            )

    def load_llm(self, model_id: str | None = None) -> Any:
        target_model_id = model_id or self.model_id

        if self._cached_model is not None and self._cached_model_id == target_model_id:
            return self._cached_model

        with self._load_lock:
            if (
                self._cached_model is not None
                and self._cached_model_id == target_model_id
            ):
                return self._cached_model

            if self.use_vllm and not self._vllm_endpoint_checked:
                self._check_vllm_endpoint(target_model_id)
                self._vllm_endpoint_checked = True

            self._cached_model = self._build_llm(target_model_id)
            self._cached_model_id = target_model_id
            self.model_id = target_model_id
            return self._cached_model

    def warmup(self) -> None:
        self.load_llm()

    @staticmethod
    def _is_transient_error(exc: BaseException) -> bool:
        text = f"{type(exc).__name__}: {exc}".lower()
        markers = (
            "timeout",
            "timed out",
            "connection",
            "temporarily unavailable",
            "service unavailable",
            "serviceunavailable",
            "bad gateway",
            "gateway timeout",
            "rate limit",
            "ratelimiterror",
            "apiconnectionerror",
            "apitimeouterror",
            "internalservererror",
        )
        return any(marker in text for marker in markers)

    def _invoke_with_retry(self, model: Any, payload: Any) -> Any:
        """Invoke the model, retrying only on transient network/server errors."""
        attempts = max(1, self.generate_retry_attempts)
        for attempt in range(1, attempts + 1):
            try:
                return model.invoke(payload)
            except Exception as exc:
                if attempt >= attempts or not self._is_transient_error(exc):
                    raise
                backoff_sec = self.generate_retry_backoff_sec * attempt
                logger.warning(
                    "LLM invoke transient failure. retry=%d/%d backoff_sec=%.2f error=%s",
                    attempt,
                    attempts,
                    backoff_sec,
                    exc,
                )
                if backoff_sec > 0:
                    time.sleep(backoff_sec)

    def generate(self, query: str, context: str, config: AgentConfig) -> dict[str, str]:
        response_language = self._detect_query_language(query)

        # PromptLibrary is the single source of truth for prompts: both the
        # vLLM and local HF backends must see the same prompt so their answers
        # stay comparable across experiments.
        # WP3: detected on the question, not asked of the model — one regex
        # instead of a classifier call on every turn.
        definitional = config.prefer_verbatim_definitions and questions.is_definitional(
            query
        )
        if definitional:
            logger.info("Definitional question detected; asking for a verbatim opening")

        prompt = PromptLibrary.answer_prompt(
            config,
            language=response_language if config.enforce_language else None,
            definitional=definitional,
        )
        rendered = prompt.invoke(
            {
                "question": query,
                "context": context,
            }
        )

        logger.info("Rendered prompt (first 500 chars): %s", str(rendered)[:500])
        logger.info("Context length (chars): %d", len(context))

        model = self.load_llm()
        output = self._invoke_with_retry(model, rendered)

        answer = str(output.content if hasattr(output, "content") else output).strip()
        logger.info("LLM raw output (first 800 chars): %s", answer[:800])

        # Kept so abstention can be measured on what the model said before any
        # rescue retry rewrote it (audit §1.5).
        pre_retry_answer = answer
        refusal_retry_applied = False

        if self._hit_token_limit(output):
            logger.warning(
                "Answer hit max_new_tokens=%d and was cut mid-sentence; trimming "
                "to the last complete sentence",
                self.max_new_tokens,
            )
            answer = self._trim_to_last_sentence(answer)

        # If model returned empty or a generic refusal, try a stricter fallback prompt once.
        #
        # Skipped when parametric fallback is authorised: there the model may
        # answer from its own knowledge as long as it marks the statement, so a
        # refusal means it has nothing to offer from either source. Retrying with
        # "use only the provided context" would talk it out of a decision it was
        # entitled to make, which is how a correct abstention became an answer.
        if (
            looks_like_refusal(answer)
            and context
            and str(context).strip()
            and not config.allow_parametric_fallback
        ):
            try:
                logger.info("LLM refusal detected; attempting fallback retry...")
                fallback_prompt = PromptLibrary.refusal_retry_prompt(
                    language=response_language
                ).invoke({"question": query, "context": context})
                output2 = self._invoke_with_retry(model, fallback_prompt)
                answer2 = str(
                    output2.content if hasattr(output2, "content") else output2
                ).strip()
                if answer2 and not looks_like_refusal(answer2):
                    logger.info("Fallback retry succeeded: %s", answer2[:500])
                    # Keep the pre-retry answer: any abstention measured on the
                    # final answer is measuring post-retry behaviour, which is
                    # why abstention was unmeasurable on runs without the
                    # parametric flag. See docs/code_audit_2026-08-15.md §1.5.
                    refusal_retry_applied = True
                    answer = answer2
                elif answer2:
                    # A second refusal is not a rescue. Keeping the first
                    # formulation preserves the model's own wording.
                    logger.info("Fallback retry also refused; keeping first answer")
                else:
                    logger.info("Fallback retry returned empty answer")
            except Exception as exc:
                # best-effort retry — ignore errors and keep original answer
                logger.warning("Fallback retry failed: %s", exc)
                pass

        # Do not apply aggressive rule-based extraction when the model refuses
        # to answer. Preserve the model's refusal so failures remain transparent
        # to the caller instead of being converted into questionable lists.

        if config.enforce_language and answer:
            answer = self._enforce_answer_language(
                model=model,
                query=query,
                context=context,
                config=config,
                answer=answer,
                target_language=response_language,
            )

        return {
            "answer": answer,
            "pre_retry_answer": pre_retry_answer,
            "refusal_retry_applied": refusal_retry_applied,
        }

    def _enforce_answer_language(
        self,
        model: Any,
        query: str,
        context: str,
        config: AgentConfig,
        answer: str,
        target_language: str,
    ) -> str:
        """Regenerate once when the answer came back in the wrong language (WP5).

        A single retry: a second wrong-language answer means the constraint is
        not what is failing, and further calls only add latency.

        Args:
            model: The already-loaded chat model.
            query: The user question, as sent to the first attempt.
            context: The retrieved context, unchanged.
            config: Agent configuration.
            answer: The answer produced by the first attempt.
            target_language: ``"it"`` or ``"en"``, detected on the question.

        Returns:
            The retried answer when it is in the target language, otherwise the
            original one — a wrong-language answer still beats a worse answer.
        """
        if self._detect_text_language(answer) == target_language:
            return answer

        logger.warning(
            "Answer language mismatch: expected=%s. Retrying once with a "
            "reinforced constraint.",
            target_language,
        )
        try:
            retry_prompt = PromptLibrary.answer_prompt(
                config,
                language=target_language,
                reinforce_language=True,
                definitional=(
                    config.prefer_verbatim_definitions
                    and questions.is_definitional(query)
                ),
            ).invoke({"question": query, "context": context})
            output = self._invoke_with_retry(model, retry_prompt)
            retried = str(
                output.content if hasattr(output, "content") else output
            ).strip()
        except Exception as exc:  # noqa: BLE001 - best effort, keep first answer
            logger.warning("Language retry failed: %s", exc)
            return answer

        if retried and self._detect_text_language(retried) == target_language:
            logger.info("Language retry produced a %s answer", target_language)
            return retried

        logger.warning(
            "Language retry did not fix the language; keeping the first answer"
        )
        return answer

    @staticmethod
    def _hit_token_limit(output: Any) -> bool:
        """Whether generation stopped because it ran out of tokens.

        Args:
            output: The backend response; only OpenAI-compatible backends carry
                ``response_metadata['finish_reason']``.

        Returns:
            ``True`` when the backend reported a length stop.
        """
        metadata = getattr(output, "response_metadata", None)
        if not isinstance(metadata, dict):
            return False
        return str(metadata.get("finish_reason", "")).lower() == "length"

    @staticmethod
    def _trim_to_last_sentence(answer: str) -> str:
        """Drop a dangling half-sentence left by the token cap.

        Only the trailing fragment goes: a reader seeing "…e coerente con le
        fonti, la specifica" reads it as a bug, which it is, and the fragment
        carries no information anyway.

        Args:
            answer: The generated answer, possibly cut mid-sentence.

        Returns:
            The answer up to the last sentence end, or unchanged when no
            sentence boundary is left to cut back to.
        """
        text = str(answer or "").rstrip()
        if not text or text[-1] in ".!?:»\"')":
            return text

        cut = max(text.rfind(mark) for mark in (". ", ".\n", "! ", "? ", ".", "!", "?"))
        if cut <= 0:
            return text
        # Keep at least a paragraph: an answer trimmed down to one sentence is
        # worse than the fragment it was meant to hide.
        if cut < len(text) * 0.4:
            return text
        return text[: cut + 1].rstrip()

    @staticmethod
    def _detect_text_language(text: str) -> str:
        """Detect the language of generated prose.

        Strips what carries no language signal but plenty of foreign tokens:
        reference tags, the trailing source list whose document titles are
        mostly English even under an Italian answer, and (WP3) verbatim
        quotations, which are deliberately left in the source's language.

        Args:
            text: The generated answer.

        Returns:
            ``"it"`` or ``"en"``.
        """
        body = re.split(r"\n\s*(?:Fonti|Sources)\s*:", str(text or ""))[0]
        body = re.sub(r"\[[STst]\s*\d+(?:\s*,\s*[STst]\s*\d+)*\]", " ", body)
        body = re.sub(r"«[^»]*»", " ", body)
        return LLMManager._detect_query_language(body)

    @staticmethod
    def _detect_query_language(query: str) -> str:
        text = str(query or "").strip().lower()
        if not text:
            return "en"

        italian_markers = {
            "il",
            "la",
            "gli",
            "della",
            "delle",
            "perche",
            "perché",
            "quali",
            "quale",
            "come",
            "sono",
            "rispetto",
            "tra",
            "sulla",
            # Elided forms tokenize with the apostrophe attached, so plain
            # articles never match them; accented "è" and "cos'è" are the
            # strongest single-token signals in short Italian questions.
            "è",
            "che",
            "cosa",
            "cos'è",
            "qual",
            "dove",
            "chi",
            "quando",
            "quanto",
            "quanti",
            "quante",
            "del",
            "dei",
            "dello",
            "degli",
            "nel",
            "nella",
            "una",
            "uno",
            "più",
            "può",
            "viene",
            "vengono",
            # Bare prepositions/articles that have no English homograph:
            # they carry verbless fragments like "Esempi di X a Torino?".
            "di",
            "da",
            "su",
            "un",
            # Imperative openings the expert uses in follow-ups ("Mi indichi…",
            # "Approfondisci…"): often the only Italian tokens in a short turn.
            "mi",
            "dimmi",
            "dammi",
            "elenca",
            "indica",
            "indichi",
            "spiega",
            "approfondisci",
            "riporta",
            "esempi",
            "secondo",
            "per",
            "con",
            "questo",
            "questa",
            "questi",
            "queste",
            "quel",
            "quella",
            "anche",
            "invece",
            "quindi",
            "senza",
            "sulle",
            "sui",
            "nelle",
            "negli",
            "alle",
            "agli",
        }
        # Kept at comparable coverage to the Italian set above. A ~70-vs-20 split
        # gave Italian a structural advantage on every mixed sentence, which on a
        # bilingual corpus with `enforce_language` on meant English questions
        # being answered in Italian. See docs/code_audit_2026-08-15.md §1.12.
        # Homographs are deliberately absent from both sets: "in" and "a" are
        # high-frequency function words in Italian too, so scoring them as
        # English turned "Approfondisci l'economia circolare in Piemonte" into an
        # English question.
        english_markers = {
            "about", "all", "an", "and", "any", "are", "as", "at", "be",
            "been", "between", "both", "but", "by", "can", "could", "did", "do",
            "does", "each", "for", "from", "give", "has", "have", "how",
            "into", "is", "it", "its", "list", "many", "may", "much", "must",
            "of", "on", "or", "other", "our", "over", "regarding", "should",
            "some", "such", "than", "that", "the", "their", "them", "then",
            "there", "these", "they", "this", "those", "through", "to", "under",
            "was", "were", "what", "what's", "when", "where", "which", "while",
            "who", "why", "will", "with", "within", "would",
        }

        tokens = re.findall(r"[a-zà-öø-ÿ']+", text)
        it_score = sum(1 for token in tokens if token in italian_markers)
        en_score = sum(1 for token in tokens if token in english_markers)

        # Short questions carry almost no function words ("Definizione di SEeD?",
        # "Cos'è la coevoluzione?"): orthography is then the strongest remaining
        # signal. It is only decisive when no English function word appeared at
        # all — otherwise a borrowed or accented noun ("café", "Fassio's
        # coevoluzione") handed a free point to Italian in a plainly English
        # sentence.
        if en_score == 0:
            if _ITALIAN_ACCENTED.search(text):
                it_score += 1
            if _ITALIAN_ELISION.search(text):
                it_score += 1

        return "it" if it_score > en_score else "en"

