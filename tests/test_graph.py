from unittest.mock import AsyncMock, patch

import pytest

from src.graph.graph import build_graph
from src.models.schemas import DocumentChunk, ResearchPlan, ResearchReport, SearchResult, SubTask


@pytest.fixture
def mock_plan():
    return ResearchPlan(
        original_query="test query",
        sub_tasks=[
            SubTask(id=1, description="Web research task", rationale="Need web data"),
            SubTask(id=2, description="Doc analysis task", rationale="Need KB data"),
        ],
        research_strategy="Parallel web and doc research",
    )


@pytest.fixture
def mock_report():
    return ResearchReport(
        title="Test Research Report",
        summary="Summary of findings.",
        sections=["## Findings\nDetails here."],
        sources=["https://example.com"],
    )


def test_graph_compiles():
    app = build_graph()
    assert app is not None


@pytest.mark.asyncio
async def test_graph_end_to_end_with_mocks(mock_plan, mock_report):
    app = build_graph()
    mock_web = [
        SearchResult(
            title="Web hit",
            url="https://example.com",
            snippet="web snippet",
            source="tavily",
        )
    ]
    mock_doc = [
        DocumentChunk(text="doc text", source="paper.pdf", score=0.9, title="Paper")
    ]

    with (
        patch("src.graph.nodes.plan_research", new=AsyncMock(return_value=mock_plan)),
        patch("src.graph.nodes.research_web", new=AsyncMock(return_value=(mock_web, []))),
        patch("src.graph.nodes.analyze_documents", new=AsyncMock(return_value=(mock_doc, []))),
        patch(
            "src.graph.nodes.write_report",
            new=AsyncMock(return_value=(mock_report, "# Test Research Report\n\nSummary.")),
        ),
    ):
        result = await app.ainvoke(
            {
                "query": "test query",
                "sub_tasks": [],
                "web_findings": [],
                "doc_findings": [],
                "synthesis": "",
                "report": None,
                "messages": [],
            }
        )

    assert result["sub_tasks"] == ["Web research task", "Doc analysis task"]
    assert len(result["web_findings"]) == 2
    assert len(result["doc_findings"]) == 2
    assert result["report"] is not None
    assert result["report"].title == "Test Research Report"
    assert "Test Research Report" in result["synthesis"]
