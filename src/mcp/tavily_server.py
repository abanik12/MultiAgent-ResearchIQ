#!/usr/bin/env python3
"""Tavily MCP server exposing web search and page fetch tools."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from src.config.settings import get_settings
from src.tools.scraper_tools import fetch_page_content
from src.tools.search_tools import apply_web_findings_limit, format_search_results, tavily_search

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

mcp = FastMCP("tavily-research")


@mcp.tool()
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for current information on a topic."""
    settings = get_settings()
    effective_max = max_results
    results = await tavily_search(query, max_results=effective_max, settings=settings)
    results = apply_web_findings_limit(results, settings)
    return format_search_results(results)


@mcp.tool()
async def get_page_content(url: str) -> str:
    """Fetch and clean text content from a web page URL."""
    return await fetch_page_content(url)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
