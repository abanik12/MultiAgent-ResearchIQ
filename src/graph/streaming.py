"""Server-sent event streaming for the research graph."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from src.config.settings import Settings, get_settings
from src.graph.graph import build_graph
from src.graph.state import AgentState
from src.graph.trace_context import (
    activate_trace_queue,
    deactivate_trace_queue,
    drain_trace_events,
)
from src.models.schemas import DocumentChunk, ReportExportPaths, ResearchDonePayload, SearchResult
from src.rag.index_store import close_index_store
from src.tools.search_tools import apply_web_findings_limit
from src.utils.report_export import export_report_files
from src.utils.tracing import build_graph_run_config, configure_langsmith


def _serialize_finding(finding: SearchResult | DocumentChunk | dict[str, Any]) -> dict[str, Any]:
    if isinstance(finding, (SearchResult, DocumentChunk)):
        return finding.model_dump()
    return finding


def _event(event_type: str, **payload: Any) -> str:
    return json.dumps({"type": event_type, **payload})


def _merge_state(state: AgentState, update: dict[str, Any]) -> None:
    for key, value in update.items():
        if key in {"web_findings", "doc_findings", "messages"}:
            state[key] = [*state.get(key, []), *value]  # type: ignore[list-item]
        else:
            state[key] = value  # type: ignore[literal-required]


def _yield_pending_trace_events() -> list[str]:
    return [json.dumps(payload) for payload in drain_trace_events()]


async def stream_research(
    query: str,
    *,
    export_report: bool = True,
    settings: Settings | None = None,
) -> AsyncIterator[str]:
    """Yield SSE-formatted JSON events while the research graph runs."""
    settings = settings or get_settings()
    configure_langsmith(settings)
    app = build_graph()
    run_config = build_graph_run_config(query, source="api")
    initial_state: AgentState = {
        "query": query,
        "sub_tasks": [],
        "web_findings": [],
        "doc_findings": [],
        "synthesis": "",
        "report": None,
        "messages": [],
    }
    accumulated: AgentState = {
        "query": query,
        "sub_tasks": [],
        "web_findings": [],
        "doc_findings": [],
        "synthesis": "",
        "report": None,
        "messages": [],
    }

    activate_trace_queue()

    try:
        yield _event("status", message="Starting research pipeline")
        yield _event("phase", phase="planning", message="Starting research pipeline")

        async for chunk in app.astream(initial_state, stream_mode="updates", config=run_config):
            for pending in _yield_pending_trace_events():
                yield pending

            for node_name, update in chunk.items():
                _merge_state(accumulated, update)

                for finding in update.get("web_findings", []):
                    yield _event("web_finding", finding=_serialize_finding(finding))

                for finding in update.get("doc_findings", []):
                    yield _event("doc_finding", finding=_serialize_finding(finding))

                if node_name == "report_writer" and update.get("synthesis"):
                    yield _event(
                        "trace",
                        agent="report_writer",
                        message="Report markdown ready",
                    )

        for pending in _yield_pending_trace_events():
            yield pending

        if not accumulated.get("synthesis"):
            yield _event("error", message="Research pipeline did not produce a result")
            return

        accumulated = {
            **accumulated,
            "web_findings": apply_web_findings_limit(accumulated["web_findings"], settings),
        }

        export_paths: ReportExportPaths | None = None
        if export_report and settings.report_export_enabled and accumulated.get("synthesis"):
            yield _event("phase", phase="export", message="Exporting markdown and PDF")
            export_paths = export_report_files(
                accumulated["synthesis"],
                query,
                settings=settings,
            )
            yield _event(
                "export",
                report_id=export_paths.report_id,
                markdown_path=export_paths.markdown_path,
                pdf_path=export_paths.pdf_path,
                message="Report exported to markdown and PDF",
            )

        report = accumulated.get("report")
        done = ResearchDonePayload(
            query=query,
            sub_tasks=accumulated.get("sub_tasks") or [],
            web_findings_count=len(accumulated.get("web_findings") or []),
            doc_findings_count=len(accumulated.get("doc_findings") or []),
            report_title=report.title if report else "",
            synthesis=accumulated.get("synthesis") or "",
            export=export_paths,
        )
        yield _event("done", result=done.model_dump())
    except Exception as exc:
        for pending in _yield_pending_trace_events():
            yield pending
        yield _event("error", message=str(exc))
    finally:
        deactivate_trace_queue()
        close_index_store()
