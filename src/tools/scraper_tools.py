"""Web page fetching and text extraction."""

from __future__ import annotations

import re
from html import unescape

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

DEFAULT_USER_AGENT = "ResearchIQ/0.3 (+https://github.com/abanik12/MultiAgent-ResearchIQ)"
DEFAULT_HEADERS = {
    "User-Agent": DEFAULT_USER_AGENT,
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _html_to_text(html: str) -> str:
    cleaned = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    text = unescape(cleaned)
    return re.sub(r"\s+", " ", text).strip()


def _fetch_error_message(url: str, status_code: int | None, detail: str) -> str:
    code = status_code if status_code is not None else "unknown"
    return (
        f"Could not fetch page content from {url} (HTTP {code}: {detail}). "
        "Many sites block automated access — rely on the web_search snippet instead."
    )


def _should_retry(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception(_should_retry),
)
async def fetch_page_content(
    url: str,
    max_chars: int = 8000,
    timeout: float = 30.0,
) -> str:
    """Fetch a URL and return cleaned plain text, or a graceful error message."""
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
            headers=DEFAULT_HEADERS,
        ) as client:
            response = await client.get(url)

            if response.status_code >= 400:
                if response.status_code >= 500:
                    response.raise_for_status()
                return _fetch_error_message(
                    url,
                    response.status_code,
                    response.reason_phrase,
                )

            text = _html_to_text(response.text)
            if not text:
                return (
                    f"Fetched {url} but no readable text was extracted. "
                    "Use the web_search snippet instead."
                )
            if len(text) <= max_chars:
                return text
            return text[:max_chars] + "\n...[truncated]"
    except httpx.TimeoutException:
        return _fetch_error_message(url, None, f"request timed out after {timeout}s")
    except httpx.RequestError as exc:
        return _fetch_error_message(url, None, str(exc))
