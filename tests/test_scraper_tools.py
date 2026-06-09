"""Tests for web page scraping utilities."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.tools.scraper_tools import fetch_page_content


@pytest.mark.asyncio
async def test_fetch_page_content_returns_text():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html><body><p>Hello world</p></body></html>"

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("src.tools.scraper_tools.httpx.AsyncClient", return_value=mock_client):
        content = await fetch_page_content("https://example.com")

    assert "Hello world" in content


@pytest.mark.asyncio
async def test_fetch_page_content_handles_403_gracefully():
    mock_response = MagicMock()
    mock_response.status_code = 403
    mock_response.reason_phrase = "Forbidden"

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("src.tools.scraper_tools.httpx.AsyncClient", return_value=mock_client):
        content = await fetch_page_content("https://openai.com/index/new-tools")

    assert "403" in content
    assert "Could not fetch page content" in content


@pytest.mark.asyncio
async def test_fetch_page_content_retries_server_errors():
    mock_ok = MagicMock()
    mock_ok.status_code = 200
    mock_ok.text = "<html><body>Recovered</body></html>"

    mock_error = MagicMock()
    mock_error.status_code = 503
    mock_error.reason_phrase = "Service Unavailable"
    mock_error.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "Service Unavailable",
            request=MagicMock(),
            response=mock_error,
        )
    )

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=[mock_error, mock_ok])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("src.tools.scraper_tools.httpx.AsyncClient", return_value=mock_client):
        content = await fetch_page_content("https://example.com")

    assert "Recovered" in content
    assert mock_client.get.await_count == 2
