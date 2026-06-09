from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_BOOTSTRAP_N = 1000
DEFAULT_BOOTSTRAP_CI = 0.95
DEFAULT_BOOTSTRAP_SEED = 42
DEFAULT_REGRESSION_THRESHOLD = 0.05


@dataclass
class JudgeConfig:
    backend: str = "vllm"  # "local_hf" | "vllm" | "api"
    model_id: str = ""
    vllm_base_url: str = field(
        default_factory=lambda: os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    )
    vllm_api_key: str = field(
        default_factory=lambda: os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
    )
    api_provider: str = "anthropic"  # "anthropic" | "openai"
    rubrics: list[str] = field(
        default_factory=lambda: ["answer_correctness", "groundedness", "relevance"]
    )
    max_new_tokens: int = 256
    cache_size: int = 256


@dataclass
class EvalConfig:
    gold_dir: Path = field(default_factory=lambda: Path("evaluation/gold"))
    baselines_path: Path = field(
        default_factory=lambda: Path("evaluation/baselines/baseline_metrics.json")
    )

    # Retrieval metrics
    k: int | None = None
    bertscore: bool = False

    # RAGAS (optional)
    ragas_enabled: bool = False
    ragas_metrics: list[str] = field(
        default_factory=lambda: [
            "faithfulness",
            "answer_relevancy",
            "answer_correctness",
            "context_precision",
            "context_recall",
        ]
    )

    # Bootstrap
    n_bootstrap: int = DEFAULT_BOOTSTRAP_N
    ci: float = DEFAULT_BOOTSTRAP_CI
    seed: int = DEFAULT_BOOTSTRAP_SEED

    # Regression
    regression_threshold: float = DEFAULT_REGRESSION_THRESHOLD

    # Judge
    judge: JudgeConfig = field(default_factory=JudgeConfig)

    extra: dict[str, Any] = field(default_factory=dict)
