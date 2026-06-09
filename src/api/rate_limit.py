"""In-memory per-IP rate limiting for FastAPI routes."""

from __future__ import annotations

from collections import defaultdict
from threading import Lock
from time import time

from fastapi import HTTPException, Request

from src.config.settings import Settings, get_settings

_WINDOW_SECONDS = 60


class InMemoryRateLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def allow(self, key: str, *, limit: int, window_seconds: int = _WINDOW_SECONDS) -> bool:
        now = time()
        with self._lock:
            hits = self._hits[key]
            hits[:] = [timestamp for timestamp in hits if now - timestamp < window_seconds]
            if len(hits) >= limit:
                return False
            hits.append(now)
            return True

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


limiter = InMemoryRateLimiter()


def _client_key(request: Request) -> str:
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def enforce_rate_limit(request: Request, *, limit: int, scope: str) -> None:
    settings = get_settings()
    if not settings.rate_limit_enabled:
        return

    key = f"{scope}:{_client_key(request)}"
    if not limiter.allow(key, limit=limit):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded for {scope}. Try again in a minute.",
            headers={"Retry-After": str(_WINDOW_SECONDS)},
        )


def research_rate_limit(request: Request) -> None:
    settings = get_settings()
    enforce_rate_limit(
        request,
        limit=settings.rate_limit_research_per_minute,
        scope="research",
    )


def ingest_rate_limit(request: Request) -> None:
    settings = get_settings()
    enforce_rate_limit(
        request,
        limit=settings.rate_limit_ingest_per_minute,
        scope="ingest",
    )
