from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.report_writer import report_to_markdown, write_report
from src.config.settings import get_settings
from src.models.schemas import DocumentChunk, ResearchReport, SearchResult


@pytest.mark.asyncio
async def test_report_to_markdown_splits_sources_by_origin():
    report = ResearchReport(
        title="Test Report",
        summary="Executive summary.",
        sections=["## Section One\nBody text."],
    )
    markdown = report_to_markdown(
        report,
        web_findings=[
            SearchResult(
                title="Agent Blog",
                url="https://example.com/agents",
                snippet="snippet",
                source="tavily",
            )
        ],
        doc_findings=[
            DocumentChunk(
                text="chunk",
                source="data/sample_docs/attention.pdf",
                score=0.9,
                title="Attention Is All You Need",
            )
        ],
    )
    assert "## Sources" in markdown
    assert "### From Web" in markdown
    assert "### From Document KB" in markdown
    assert "https://example.com/agents" in markdown
    assert "Attention Is All You Need" in markdown


@pytest.mark.asyncio
async def test_report_to_markdown():
    report = ResearchReport(
        title="Test Report",
        summary="Executive summary.",
        sections=["## Section One\nBody text."],
        sources=["https://example.com", "Attention Is All You Need"],
    )
    markdown = report_to_markdown(report)
    assert "# Test Report" in markdown
    assert "Executive summary." in markdown
    assert "## Sources" in markdown


@pytest.mark.asyncio
async def test_write_report_with_mock_llm():
    mock_report = ResearchReport(
        title="Agentic RAG Overview",
        summary="A brief overview.",
        sections=["## Background\nRAG combines retrieval with generation."],
        sources=["https://example.com"],
    )

    with patch("src.agents.report_writer._build_llm") as mock_build_llm:
        mock_llm = MagicMock()
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=mock_report)
        mock_llm.with_structured_output.return_value = mock_structured
        mock_build_llm.return_value = mock_llm

        report, synthesis = await write_report(
            query="What is agentic RAG?",
            sub_tasks=["Overview", "Recent work"],
            web_findings=[
                SearchResult(
                    title="Example",
                    url="https://example.com",
                    snippet="Snippet text",
                    source="tavily",
                )
            ],
            doc_findings=[
                DocumentChunk(
                    text="Transformer architecture details.",
                    source="attention.pdf",
                    score=0.9,
                    title="Attention Is All You Need",
                )
            ],
        )

    assert report.title == "Agentic RAG Overview"
    assert "Agentic RAG Overview" in synthesis


@pytest.mark.asyncio
async def test_research_web_falls_back_to_direct_search():
    from src.agents.web_researcher import research_web

    mock_findings = [
        SearchResult(
            title="Result",
            url="https://example.com",
            snippet="content",
            source="tavily",
        )
    ]

    with (
        patch("src.agents.web_researcher.create_web_researcher", new=AsyncMock()) as mock_create,
        patch("src.agents.web_researcher.tavily_search", new=AsyncMock(return_value=mock_findings)),
        patch("src.config.settings.get_settings") as mock_settings,
    ):
        mock_agent = AsyncMock()
        mock_agent.ainvoke = AsyncMock(return_value={"messages": []})
        mock_create.return_value = mock_agent
        mock_settings.return_value.tavily_api_key = "test-key"

        findings = await research_web("transformer architectures")

    assert len(findings) == 1
    assert findings[0].url == "https://example.com"


@pytest.mark.asyncio
async def test_analyze_documents_respects_rag_skip_rerank():
    from src.agents.doc_analyst import analyze_documents

    mock_chunks = [
        DocumentChunk(text="chunk", source="paper.pdf", score=0.8, title="Paper")
    ]
    settings = get_settings().model_copy(update={"rag_skip_rerank": True})

    with patch(
        "src.agents.doc_analyst.hybrid_search",
        return_value=mock_chunks,
    ) as mock_search:
        await analyze_documents("attention mechanisms", settings=settings)

    assert mock_search.call_args.args[5] is True


@pytest.mark.asyncio
async def test_analyze_documents_calls_hybrid_search():
    from src.agents.doc_analyst import analyze_documents

    mock_chunks = [
        DocumentChunk(text="chunk", source="paper.pdf", score=0.8, title="Paper")
    ]

    with patch(
        "src.agents.doc_analyst.hybrid_search",
        return_value=mock_chunks,
    ) as mock_search:
        findings = await analyze_documents("attention mechanisms")

    mock_search.assert_called_once()
    assert len(findings) == 1
    assert findings[0].title == "Paper"
