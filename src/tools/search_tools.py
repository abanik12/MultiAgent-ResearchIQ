"""Tavily web search utilities."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential
from tavily import TavilyClient

from src.config.settings import Settings, get_settings
from src.models.schemas import SearchResult

TAVILY_MAX_RESULTS_PER_CALL = 20


def _get_tavily_client(settings: Settings | None = None) -> TavilyClient:
    settings = settings or get_settings()
    if not settings.tavily_api_key:
        raise ValueError("TAVILY_API_KEY is not set")
    return TavilyClient(api_key=settings.tavily_api_key)


def _parse_published_date(value: str) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def sort_results_by_recent(results: list[SearchResult]) -> list[SearchResult]:
    """Sort web results newest-first, using published_date then Tavily score."""
    return sorted(
        results,
        key=lambda result: (
            _parse_published_date(result.published_date) or datetime.min,
            result.score,
        ),
        reverse=True,
    )


def apply_web_findings_limit(
    results: list[SearchResult],
    settings: Settings | None = None,
) -> list[SearchResult]:
    """When recent limiting is enabled, keep the top N newest unique URLs across the full run."""
    settings = settings or get_settings()
    if not settings.web_search_recent_limit_enabled:
        return results

    deduped: list[SearchResult] = []
    seen_urls: set[str] = set()
    for result in sort_results_by_recent(results):
        if not result.url or result.url in seen_urls:
            continue
        seen_urls.add(result.url)
        deduped.append(result)

    return deduped[: settings.web_search_recent_limit]


def _parse_tavily_results(response: dict[str, Any]) -> list[SearchResult]:
    results: list[SearchResult] = []
    for item in response.get("results", []):
        results.append(
            SearchResult(
                title=item.get("title", "") or "",
                url=item.get("url", "") or "",
                snippet=item.get("content", "") or item.get("snippet", "") or "",
                source="tavily",
                published_date=item.get("published_date", "") or "",
                score=float(item.get("score", 0.0) or 0.0),
            )
        )
    return results


def _resolve_max_results(max_results: int | None, settings: Settings) -> int:
    if max_results is None:
        return 5
    return max_results


def _search_kwargs_for_settings(
    query: str,
    max_results: int,
    settings: Settings,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "query": query,
        "max_results": min(max_results, TAVILY_MAX_RESULTS_PER_CALL),
    }
    if settings.web_search_recent_limit_enabled:
        kwargs["topic"] = "news"
        kwargs["days"] = 365
    return kwargs


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def tavily_search_sync(
    query: str,
    max_results: int | None = 5,
    settings: Settings | None = None,
) -> list[SearchResult]:
    settings = settings or get_settings()
    client = _get_tavily_client(settings)
    resolved = _resolve_max_results(max_results, settings)
    response = client.search(**_search_kwargs_for_settings(query, resolved, settings))
    return _parse_tavily_results(response)


async def tavily_search(
    query: str,
    max_results: int | None = 5,
    settings: Settings | None = None,
) -> list[SearchResult]:
    """Search the web via Tavily (async wrapper)."""
    return await asyncio.to_thread(tavily_search_sync, query, max_results, settings)


def format_search_results(results: list[SearchResult]) -> str:
    return json.dumps([result.model_dump() for result in results], indent=2)
