from __future__ import annotations

import torch
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

from graphrag.config import AgentConfig, DEFAULT_MODEL_ID
from graphrag.llm.prompts import PromptLibrary


class LLMManager:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id

    @staticmethod
    def load_llm(model_id: str = DEFAULT_MODEL_ID) -> ChatHuggingFace:
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

    def generate(self, query: str, context: str, config: AgentConfig) -> dict[str, str]:
        prompt = PromptLibrary.answer_prompt(config)
        rendered = prompt.invoke({
            "question": query,
            "context": context,
        })
        model = self.load_llm(self.model_id)
        output = model.invoke(rendered)
        answer = str(output.content if hasattr(output, "content") else output).strip()
        return {"answer": answer}
