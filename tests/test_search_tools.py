"""Tests for Tavily web search utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.config.settings import Settings
from src.models.schemas import SearchResult
from src.tools.search_tools import (
    apply_web_findings_limit,
    sort_results_by_recent,
    tavily_search_sync,
)


def test_sort_results_by_recent_orders_by_published_date():
    results = [
        SearchResult(title="Old", url="https://a.com", published_date="2024-01-01"),
        SearchResult(title="New", url="https://b.com", published_date="2025-06-01"),
        SearchResult(title="Mid", url="https://c.com", published_date="2024-12-01"),
    ]
    sorted_results = sort_results_by_recent(results)
    assert [result.url for result in sorted_results] == [
        "https://b.com",
        "https://c.com",
        "https://a.com",
    ]


def test_apply_web_findings_limit_disabled():
    settings = Settings(web_search_recent_limit_enabled=False)
    results = [SearchResult(title="A", url=f"https://{index}.com") for index in range(5)]
    assert apply_web_findings_limit(results, settings) == results


def test_apply_web_findings_limit_enabled():
    settings = Settings(web_search_recent_limit_enabled=True, web_search_recent_limit=2)
    results = [
        SearchResult(title="Old", url="https://a.com", published_date="2024-01-01"),
        SearchResult(title="New", url="https://b.com", published_date="2025-06-01"),
        SearchResult(title="Mid", url="https://c.com", published_date="2024-12-01"),
    ]
    limited = apply_web_findings_limit(results, settings)
    assert len(limited) == 2
    assert limited[0].url == "https://b.com"
    assert limited[1].url == "https://c.com"


def test_apply_web_findings_limit_is_global_across_sub_tasks():
    settings = Settings(web_search_recent_limit_enabled=True, web_search_recent_limit=20)
    merged: list[SearchResult] = []
    for index in range(3):
        merged.extend(
            SearchResult(
                title=f"Item {offset}",
                url=f"https://example{offset}.com",
                published_date=f"2025-01-{offset + 1:02d}",
            )
            for offset in range(index * 20, (index + 1) * 20)
        )

    limited = apply_web_findings_limit(merged, settings)
    assert len(limited) == 20


def test_tavily_search_sync_recent_limit_uses_news_topic_per_call():
    settings = Settings(
        tavily_api_key="test-key",
        web_search_recent_limit_enabled=True,
        web_search_recent_limit=20,
    )
    mock_client = MagicMock()
    mock_client.search.return_value = {"results": []}

    with patch("src.tools.search_tools._get_tavily_client", return_value=mock_client):
        tavily_search_sync("agentic rag", max_results=5, settings=settings)

    assert mock_client.search.call_args.kwargs["topic"] == "news"
    assert mock_client.search.call_args.kwargs["max_results"] == 5


def test_tavily_search_sync_no_limit_when_disabled():
    settings = Settings(tavily_api_key="test-key", web_search_recent_limit_enabled=False)
    mock_client = MagicMock()
    mock_client.search.return_value = {"results": []}

    with patch("src.tools.search_tools._get_tavily_client", return_value=mock_client):
        tavily_search_sync("agentic rag", max_results=7, settings=settings)

    assert mock_client.search.call_args.kwargs["max_results"] == 7
    assert "topic" not in mock_client.search.call_args.kwargs
