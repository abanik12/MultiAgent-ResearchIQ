"""Unit tests for curated SSE trace event emission."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agents.doc_analyst import build_doc_trace_steps
from src.agents.web_researcher import extract_react_trace_steps
from src.graph.trace_context import (
    activate_trace_queue,
    deactivate_trace_queue,
    drain_trace_events,
    emit_trace_event,
)
from src.models.schemas import DocumentChunk, SearchResult


def test_emit_trace_event_accepts_event_dict():
    activate_trace_queue()
    try:
        emit_trace_event(
            **{
                "type": "tool_call",
                "agent": "web_researcher",
                "tool": "web_search",
                "input_summary": "agentic RAG",
                "task_id": 0,
            }
        )
        events = drain_trace_events()
        assert events[0]["type"] == "tool_call"
        assert events[0]["tool"] == "web_search"
    finally:
        deactivate_trace_queue()


def test_trace_context_queue_lifecycle():
    activate_trace_queue()
    try:
        emit_trace_event("phase", phase="planning", message="Planning")
        emit_trace_event("agent_start", agent="coordinator", message="Starting")
        events = drain_trace_events()
        assert len(events) == 2
        assert events[0]["type"] == "phase"
        assert events[1]["agent"] == "coordinator"
        assert drain_trace_events() == []
    finally:
        deactivate_trace_queue()

    emit_trace_event("phase", phase="planning")
    assert drain_trace_events() == []


def test_extract_react_trace_steps_tool_calls():
    messages = [
        HumanMessage(content="Research sub-task: agentic RAG"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "web_search", "args": {"query": "agentic RAG 2025"}, "id": "1"},
            ],
        ),
        ToolMessage(
            content='[{"title":"Blog","url":"https://example.com","snippet":"x","source":"tavily"}]',
            name="web_search",
            tool_call_id="1",
        ),
    ]

    steps = extract_react_trace_steps(messages, task_id=0)

    assert steps[0]["type"] == "tool_call"
    assert steps[0]["tool"] == "web_search"
    assert "agentic RAG 2025" in steps[0]["input_summary"]
    assert any("returned 1 result" in step["message"] for step in steps if step["type"] == "trace")


def test_build_doc_trace_steps_with_scores():
    chunks = [
        DocumentChunk(text="chunk", source="paper.pdf", score=0.91, title="Paper"),
    ]
    steps = build_doc_trace_steps("attention mechanisms", chunks, task_id=1)

    assert steps[0]["message"].startswith('Searching KB: "')
    assert "Retrieved 1 chunk(s) (top score 0.91)" in steps[1]["message"]
    assert steps[1]["task_id"] == 1


async def _collect_stream_events(query: str = "test query") -> list[dict]:
    from src.graph.streaming import stream_research

    events: list[dict] = []
    async for line in stream_research(query, export_report=False):
        events.append(json.loads(line))
    return events


@pytest.mark.asyncio
async def test_stream_research_emits_lifecycle_events():
    mock_web = [
        SearchResult(
            title="Web hit",
            url="https://example.com",
            snippet="snippet",
            source="tavily",
        )
    ]
    mock_doc = [
        DocumentChunk(text="doc text", source="paper.pdf", score=0.9, title="Paper")
    ]

    async def fake_astream(_state, stream_mode="updates", config=None):
        assert stream_mode == "updates"
        emit_trace_event(
            "agent_start",
            agent="coordinator",
            message="Decomposing query into sub-tasks",
        )
        yield {
            "coordinator": {
                "sub_tasks": ["Task A"],
                "messages": [],
            }
        }
        emit_trace_event(
            "plan",
            sub_tasks=["Task A"],
            strategy="Parallel research",
            message="Plan ready — 1 sub-tasks",
        )
        emit_trace_event(
            "agent_end",
            agent="coordinator",
            message="Plan ready — 1 sub-tasks",
        )
        emit_trace_event("phase", phase="research", message="Running parallel research")
        emit_trace_event(
            "agent_start",
            agent="web_researcher",
            task_id=0,
            task="Task A",
            message="Searching the web (task 1)",
        )
        yield {"web_researcher": {"web_findings": mock_web}}
        emit_trace_event(
            "agent_end",
            agent="web_researcher",
            task_id=0,
            message="Web research complete — 1 source(s)",
        )
        emit_trace_event(
            "agent_start",
            agent="doc_analyst",
            task_id=0,
            task="Task A",
            message="Searching knowledge base (task 1)",
        )
        yield {"doc_analyst": {"doc_findings": mock_doc}}
        emit_trace_event(
            "agent_end",
            agent="doc_analyst",
            task_id=0,
            message="Document analysis complete — 1 chunk(s)",
        )
        emit_trace_event("phase", phase="synthesis", message="Synthesizing final report")
        emit_trace_event("agent_start", agent="report_writer", message="Writing structured report")
        yield {
            "report_writer": {
                "synthesis": "# Report\n\nBody.",
                "report": MagicMock(title="Report Title"),
            }
        }
        emit_trace_event("agent_end", agent="report_writer", message="Report draft complete")

    mock_app = MagicMock()
    mock_app.astream = fake_astream

    with (
        patch("src.graph.streaming.build_graph", return_value=mock_app),
        patch("src.graph.streaming.close_index_store"),
        patch(
            "src.graph.streaming.apply_web_findings_limit",
            side_effect=lambda findings, _settings: findings,
        ),
    ):
        events = await _collect_stream_events()

    types = [event["type"] for event in events]
    assert types[0] == "status"
    assert "phase" in types
    assert "agent_start" in types
    assert "plan" in types
    assert "agent_end" in types
    assert "web_finding" in types
    assert "doc_finding" in types
    assert types[-1] == "done"
    assert events[-1]["result"]["report_title"] == "Report Title"


@pytest.mark.asyncio
async def test_stream_research_emits_trace_events_from_nodes():
    async def fake_astream(_state, stream_mode="updates", config=None):
        emit_trace_event("agent_start", agent="coordinator", message="Planning")
        yield {"coordinator": {"sub_tasks": ["Task A"], "messages": []}}
        emit_trace_event(
            "plan",
            sub_tasks=["Task A"],
            strategy="Parallel research",
            message="Plan ready — 1 sub-tasks",
        )
        emit_trace_event("phase", phase="synthesis", message="Synthesizing final report")
        yield {"report_writer": {"synthesis": "# Done", "report": MagicMock(title="Done")}}

    mock_app = MagicMock()
    mock_app.astream = fake_astream

    with (
        patch("src.graph.streaming.build_graph", return_value=mock_app),
        patch("src.graph.streaming.close_index_store"),
        patch(
            "src.graph.streaming.apply_web_findings_limit",
            side_effect=lambda findings, _settings: findings,
        ),
    ):
        events = await _collect_stream_events()

    coordinator_events = [
        event
        for event in events
        if event.get("agent") == "coordinator" or event.get("phase") == "planning"
    ]
    assert coordinator_events
    plan_events = [event for event in events if event["type"] == "plan"]
    assert len(plan_events) == 1
    assert plan_events[0]["sub_tasks"] == ["Task A"]
