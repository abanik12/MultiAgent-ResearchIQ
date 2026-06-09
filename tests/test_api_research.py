"""Tests for the research streaming API."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


async def _fake_stream(query: str, *, export_report: bool = True, settings=None):
    yield json.dumps({"type": "status", "message": "Starting"})
    yield json.dumps({"type": "phase", "phase": "planning", "message": "Planning research"})
    yield json.dumps(
        {
            "type": "agent_start",
            "agent": "coordinator",
            "message": "Decomposing query into sub-tasks",
        }
    )
    yield json.dumps(
        {
            "type": "plan",
            "sub_tasks": ["Task A", "Task B"],
            "strategy": "Parallel research",
            "message": "Plan ready — 2 sub-tasks",
        }
    )
    yield json.dumps(
        {
            "type": "agent_start",
            "agent": "web_researcher",
            "task_id": 0,
            "task": "Task A",
            "message": "Searching the web (task 1)",
        }
    )
    yield json.dumps(
        {
            "type": "tool_call",
            "agent": "web_researcher",
            "tool": "web_search",
            "input_summary": "agentic RAG overview",
            "task_id": 0,
        }
    )
    yield json.dumps({"type": "phase", "phase": "synthesis", "message": "Synthesizing final report"})
    yield json.dumps(
        {
            "type": "done",
            "result": {
                "query": query,
                "sub_tasks": ["Task A", "Task B"],
                "web_findings_count": 2,
                "doc_findings_count": 4,
                "report_title": "Demo Report",
                "synthesis": "# Demo Report\n\nSummary.",
                "export": {
                    "report_id": "20250101-demo",
                    "markdown_path": "data/reports/20250101-demo.md",
                    "pdf_path": "data/reports/20250101-demo.pdf",
                },
            },
        }
    )


@pytest.fixture
def client():
    return TestClient(app)


def test_research_stream_returns_sse_events(client):
    with patch("src.api.routes.research.stream_research", _fake_stream):
        response = client.post(
            "/research",
            json={"query": "What is agentic RAG?", "export_report": True},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert '"type": "plan"' in body
    assert '"type": "agent_start"' in body
    assert '"type": "tool_call"' in body
    assert '"type": "done"' in body
    assert "Demo Report" in body

    event_types = []
    for block in body.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                event_types.append(json.loads(line.removeprefix("data: ").strip())["type"])

    assert event_types.index("phase") < event_types.index("plan")
    assert event_types.index("agent_start") < event_types.index("tool_call")
    assert event_types.index("tool_call") < event_types.index("done")


def test_download_markdown_report(client, tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report_id = "20250101-demo"
    md_file = reports_dir / f"{report_id}.md"
    md_file.write_text("# Demo", encoding="utf-8")

    with patch("src.api.routes.research.resolve_report_path", return_value=md_file):
        response = client.get(f"/research/reports/{report_id}/markdown")

    assert response.status_code == 200
    assert response.text == "# Demo"
