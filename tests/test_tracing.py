"""Tests for LangSmith tracing configuration."""

from __future__ import annotations

import os

import pytest

from src.config.settings import Settings
from src.utils.tracing import build_graph_run_config, configure_langsmith, is_langsmith_enabled

_LANGSMITH_KEYS = [
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGSMITH_ENDPOINT",
    "LANGSMITH_WORKSPACE_ID",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_PROJECT",
]


@pytest.fixture(autouse=True)
def clear_langsmith_env():
    saved = {key: os.environ.get(key) for key in _LANGSMITH_KEYS}
    for key in _LANGSMITH_KEYS:
        os.environ.pop(key, None)
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_configure_langsmith_sets_env_from_langsmith_vars():
    settings = Settings(
        langsmith_tracing=True,
        langsmith_api_key="ls-test-key",
        langsmith_project="researchiq-test",
        langsmith_endpoint="https://api.smith.langchain.com",
    )

    assert configure_langsmith(settings) is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "ls-test-key"
    assert os.environ["LANGSMITH_PROJECT"] == "researchiq-test"
    assert os.environ["LANGSMITH_ENDPOINT"] == "https://api.smith.langchain.com"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGCHAIN_API_KEY"] == "ls-test-key"
    assert is_langsmith_enabled(settings) is True


def test_configure_langsmith_supports_legacy_langchain_vars():
    settings = Settings(
        langsmith_tracing=False,
        langchain_tracing_v2=True,
        langchain_api_key="ls-legacy-key",
        langchain_project="legacy-project",
    )

    assert configure_langsmith(settings) is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "ls-legacy-key"
    assert os.environ["LANGSMITH_PROJECT"] == "legacy-project"


def test_configure_langsmith_requires_api_key():
    settings = Settings(langsmith_tracing=True, langsmith_api_key=None)

    assert configure_langsmith(settings) is False
    assert "LANGSMITH_TRACING" not in os.environ
    assert is_langsmith_enabled(settings) is False


def test_configure_langsmith_sets_workspace_id_when_present():
    settings = Settings(
        langsmith_tracing=True,
        langsmith_api_key="ls-test-key",
        langsmith_workspace_id="workspace-123",
    )

    configure_langsmith(settings)
    assert os.environ["LANGSMITH_WORKSPACE_ID"] == "workspace-123"


def test_build_graph_run_config_includes_metadata():
    config = build_graph_run_config("Compare transformer and Mamba", source="api")

    assert config["run_name"] == "researchiq"
    assert "api" in config["tags"]
    assert config["metadata"]["query"].startswith("Compare transformer")
    assert config["metadata"]["source"] == "api"
