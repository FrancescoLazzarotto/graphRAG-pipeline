from __future__ import annotations

import logging
import threading

import torch
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

from graphrag.config import AgentConfig, DEFAULT_MODEL_ID
from graphrag.llm.prompts import PromptLibrary

logger = logging.getLogger("graphrag")


class LLMManager:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, warmup: bool = False) -> None:
        self.model_id = model_id
        self._cached_model: ChatHuggingFace | None = None
        self._cached_model_id: str | None = None
        self._load_lock = threading.Lock()

        if warmup:
            self.load_llm()

    @staticmethod
    def _build_llm(model_id: str) -> ChatHuggingFace:
        logger.info("Loading LLM model: %s", model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        if torch.cuda.is_available():
            base_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                ),
                device_map="auto",
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
            )

        generation = pipeline(
            "text-generation",
            model=base_model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            do_sample=False,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05,
        )
        return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=generation))

    def load_llm(self, model_id: str | None = None) -> ChatHuggingFace:
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
