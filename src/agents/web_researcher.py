"""Web research agent using Tavily search tools via ReAct."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent

from src.agents.coordinator import _build_llm, _configure_tracing
from src.config.settings import Settings, get_settings
from src.models.schemas import SearchResult
from src.tools.scraper_tools import fetch_page_content
from src.tools.search_tools import (
    format_search_results,
    tavily_search,
)

WEB_RESEARCHER_PROMPT = """You are a web research specialist for ResearchIQ.

Given a research sub-task, search the web for relevant, current information.
Use web_search first. If a result looks important but the snippet is thin,
try get_page_content on its URL for more detail.

If get_page_content reports an error (403, timeout, etc.), do not retry that URL.
Continue with the web_search snippets you already have.

Rules:
- Focus only on the assigned sub-task
- Prefer authoritative sources (papers, official docs, reputable news)
- Do not invent URLs or facts
- Stop after you have enough evidence (2-3 strong sources)
"""


def _build_web_tools(settings: Settings | None = None) -> list[StructuredTool]:
    settings = settings or get_settings()

    async def web_search(query: str, max_results: int = 5) -> str:
        results = await tavily_search(query, max_results=max_results, settings=settings)
        return format_search_results(results)

    async def get_page_content(url: str) -> str:
        return await fetch_page_content(url)

    return [
        StructuredTool.from_function(
            coroutine=web_search,
            name="web_search",
            description="Search the web for current information on a topic.",
        ),
        StructuredTool.from_function(
            coroutine=get_page_content,
            name="get_page_content",
            description="Fetch and clean text content from a web page URL.",
        ),
    ]


def _parse_tool_results(messages: list) -> list[SearchResult]:
    findings: list[SearchResult] = []
    seen_urls: set[str] = set()

    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        if message.name != "web_search":
            continue
        content = message.content
        if not isinstance(content, str):
            continue
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict) or not item.get("url"):
                continue
            url = item["url"]
            if url in seen_urls:
                continue
            seen_urls.add(url)
            findings.append(SearchResult(**item))
    return findings


async def create_web_researcher(settings: Settings | None = None):
    """Create a ReAct agent with Tavily web search tools."""
    settings = settings or get_settings()
    _configure_tracing(settings)
    llm = _build_llm(settings)
    tools = _build_web_tools(settings)
    return create_react_agent(llm, tools, prompt=WEB_RESEARCHER_PROMPT)


async def create_web_researcher_via_mcp(settings: Settings | None = None):
    """Create a ReAct agent whose tools come from the Tavily MCP server."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    settings = settings or get_settings()
    _configure_tracing(settings)
    llm = _build_llm(settings)

    server_path = Path(__file__).resolve().parents[1] / "mcp" / "tavily_server.py"
    client = MultiServerMCPClient(
        {
            "tavily": {
                "command": "python",
                "args": [str(server_path)],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()
    return create_react_agent(llm, tools, prompt=WEB_RESEARCHER_PROMPT)


async def research_web(
    task: str,
    settings: Settings | None = None,
    *,
    use_mcp: bool = False,
) -> list[SearchResult]:
    """Run web research for a single sub-task."""
    settings = settings or get_settings()
    if not settings.tavily_api_key:
        return []

    if use_mcp:
        agent = await create_web_researcher_via_mcp(settings)
    else:
        agent = await create_web_researcher(settings)

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=f"Research sub-task: {task}")]}
    )
    findings = _parse_tool_results(result["messages"])

    if not findings:
        findings = await tavily_search(task, max_results=5, settings=settings)

    return findings
