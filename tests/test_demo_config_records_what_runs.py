"""The demo's config has to name the retriever the demo actually used.

`AgentConfig.text_retriever_backend` is the one field that records which text
retriever answered, and `build_agent_config` left it at its default while
`build_text_pipeline` built something else. Every session log and every bug
report read from it therefore named the wrong backend — the same defect the
CLI had and fixed (audit §5.4), on the surface the expert actually uses.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def config(monkeypatch):
    """A freshly imported product.config, so env overrides are read."""

    def _load(**env: str):
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        import product.config as module

        return importlib.reload(module)

    yield _load
    import product.config as module

    importlib.reload(module)


def test_the_config_names_the_backend_that_will_be_built(config):
    settings = config()

    built = settings.TEXT_RETRIEVER_BACKEND
    recorded = settings.build_agent_config().text_retriever_backend

    assert recorded == built


@pytest.mark.parametrize("backend", ["dense", "tfidf"])
def test_whichever_backend_is_chosen_is_the_one_recorded(config, backend):
    settings = config(DEMO_TEXT_RETRIEVER_BACKEND=backend)

    assert settings.build_agent_config().text_retriever_backend == backend


def test_the_default_is_dense_not_the_dataclass_default(config):
    # `AgentConfig.text_retriever_backend` defaults to "tfidf"; the demo has
    # measured with dense since 2026-09-04, and used to report tfidf.
    settings = config()
    monkeypatched_default = settings.build_agent_config().text_retriever_backend

    assert settings.TEXT_RETRIEVER_BACKEND == "dense"
    assert monkeypatched_default == "dense"


def test_the_embedding_model_is_named_once_for_both_uses(config):
    # It was a literal inside `build_text_pipeline`, with nothing tying it to
    # what `build_agent_config` reported.
    settings = config()

    assert (
        settings.build_agent_config().dense_embedding_model
        == settings.DENSE_EMBEDDING_MODEL
    )


def test_the_embedding_model_can_be_overridden(config):
    settings = config(DEMO_DENSE_EMBEDDING_MODEL="intfloat/multilingual-e5-large")

    assert (
        settings.build_agent_config().dense_embedding_model
        == "intfloat/multilingual-e5-large"
    )


def test_the_vector_index_directory_is_recorded_too(config):
    settings = config()

    recorded = settings.build_agent_config().vector_index_dir

    assert recorded.endswith("artifacts/vector_index")
