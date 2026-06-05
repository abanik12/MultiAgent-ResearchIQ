from unittest.mock import AsyncMock, patch

import pytest

from src.graph.graph import build_graph
from src.models.schemas import ResearchPlan, SubTask


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


def test_graph_compiles():
    app = build_graph()
    assert app is not None


@pytest.mark.asyncio
async def test_graph_end_to_end_with_mock_coordinator(mock_plan):
    app = build_graph()

    with patch("src.graph.nodes.plan_research", new=AsyncMock(return_value=mock_plan)):
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
    assert "Phase 1 stub" in result["synthesis"]
    assert result["report"] is None
