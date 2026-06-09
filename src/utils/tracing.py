"""LangSmith tracing configuration for ResearchIQ."""

from __future__ import annotations

import os
from typing import Any

from src.config.settings import Settings, get_settings


def _resolve_api_key(settings: Settings) -> str | None:
    if settings.langsmith_tracing:
        key = settings.langsmith_api_key
        if key and key.strip():
            return key.strip()
    if settings.langchain_tracing_v2:
        key = settings.langchain_api_key
        if key and key.strip():
            return key.strip()
    if _resolve_tracing_enabled(settings):
        for candidate in (settings.langsmith_api_key, settings.langchain_api_key):
            if candidate and candidate.strip():
                return candidate.strip()
    return None


def _resolve_project(settings: Settings) -> str:
    if settings.langsmith_tracing and settings.langsmith_project:
        return settings.langsmith_project
    if settings.langchain_tracing_v2 and settings.langchain_project:
        return settings.langchain_project
    return settings.langsmith_project or settings.langchain_project or "researchiq"


def _resolve_tracing_enabled(settings: Settings) -> bool:
    return settings.langsmith_tracing or settings.langchain_tracing_v2


def is_langsmith_enabled(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    return bool(_resolve_tracing_enabled(settings) and _resolve_api_key(settings))


def configure_langsmith(settings: Settings | None = None) -> bool:
    """Sync LangSmith env vars from settings for LangChain/LangGraph auto-tracing."""
    settings = settings or get_settings()
    api_key = _resolve_api_key(settings)
    if not _resolve_tracing_enabled(settings) or not api_key:
        return False

    project = _resolve_project(settings)

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint

    if settings.langsmith_workspace_id:
        os.environ["LANGSMITH_WORKSPACE_ID"] = settings.langsmith_workspace_id
    else:
        os.environ.pop("LANGSMITH_WORKSPACE_ID", None)

    # Legacy names still read by some LangChain versions.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project

    return True


def build_graph_run_config(
    query: str,
    *,
    run_name: str = "researchiq",
    source: str = "api",
) -> dict[str, Any]:
    """RunnableConfig metadata for LangGraph runs visible in LangSmith."""
    return {
        "run_name": run_name,
        "tags": ["researchiq", source],
        "metadata": {
            "query": query[:200],
            "source": source,
        },
    }
