"""Tests for API rate limiting."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.rate_limit import limiter
from src.config.settings import Settings


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_limiter():
    limiter.reset()
    yield
    limiter.reset()


@pytest.fixture
def strict_settings():
    return Settings(
        rate_limit_enabled=True,
        rate_limit_research_per_minute=2,
        rate_limit_ingest_per_minute=2,
    )


def test_research_rate_limit_returns_429(client, strict_settings):
    async def _fake_stream(query: str, *, export_report: bool = True, settings=None):
        yield '{"type": "done", "result": {"query": "q", "sub_tasks": [], "web_findings_count": 0, "doc_findings_count": 0, "report_title": "T", "synthesis": "# T"}}'

    with (
        patch("src.api.rate_limit.get_settings", return_value=strict_settings),
        patch("src.api.routes.research.stream_research", _fake_stream),
    ):
        first = client.post("/research", json={"query": "What is RAG?", "export_report": False})
        second = client.post("/research", json={"query": "What is RAG again?", "export_report": False})
        third = client.post("/research", json={"query": "Third query here", "export_report": False})

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    assert "Rate limit exceeded" in third.json()["detail"]
    assert third.headers.get("retry-after") == "60"


def test_rate_limit_disabled_allows_burst(client, strict_settings):
    disabled = strict_settings.model_copy(update={"rate_limit_enabled": False})

    async def _fake_stream(query: str, *, export_report: bool = True, settings=None):
        yield '{"type": "done", "result": {"query": "q", "sub_tasks": [], "web_findings_count": 0, "doc_findings_count": 0, "report_title": "T", "synthesis": "# T"}}'

    with (
        patch("src.api.rate_limit.get_settings", return_value=disabled),
        patch("src.api.routes.research.stream_research", _fake_stream),
    ):
        for _ in range(5):
            response = client.post(
                "/research",
                json={"query": "What is agentic RAG?", "export_report": False},
            )
            assert response.status_code == 200
