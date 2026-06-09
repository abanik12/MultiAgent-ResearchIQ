#!/usr/bin/env python3
"""Streamlit UI for ResearchIQ — streaming research with markdown/PDF download."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import httpx
import streamlit as st
from dotenv import load_dotenv

from ui.trace_renderer import TraceState, render_trace_markdown

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

API_URL = os.getenv("RESEARCHIQ_API_URL", "http://localhost:8000").rstrip("/")


def _parse_sse_line(line: str) -> dict | None:
    if not line.startswith("data: "):
        return None
    return json.loads(line.removeprefix("data: ").strip())


def stream_research_events(
    query: str,
    export_report: bool,
    on_event: Callable[[dict], None],
) -> dict | None:
    done_payload: dict | None = None

    with httpx.Client(timeout=None) as client:
        with client.stream(
            "POST",
            f"{API_URL}/research",
            json={"query": query, "export_report": export_report},
            headers={"Accept": "text/event-stream"},
        ) as response:
            response.raise_for_status()
            buffer = ""
            for chunk in response.iter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    block, buffer = buffer.split("\n\n", 1)
                    for line in block.splitlines():
                        event = _parse_sse_line(line)
                        if event is None:
                            continue
                        on_event(event)
                        if event.get("type") == "done":
                            done_payload = event.get("result")
    return done_payload


st.set_page_config(page_title="ResearchIQ", page_icon="🔎", layout="wide")
st.title("ResearchIQ")
st.caption("Multi-agent research with live agent tracing, markdown export, and PDF download.")

with st.sidebar:
    st.subheader("Settings")
    st.text_input("API URL", value=API_URL, disabled=True)
    export_report = st.checkbox("Export markdown + PDF", value=True)
    st.markdown(
        "Start the API first:\n\n"
        "`uvicorn src.api.main:app --reload --port 8000`"
    )

query = st.text_area(
    "Research question",
    placeholder="What are the latest advances in agentic RAG?",
    height=120,
)

if st.button("Run research", type="primary", disabled=not query.strip()):
    trace_state = TraceState()
    status = st.status(trace_state.status_label, expanded=True)
    metrics_cols = status.columns(3)
    web_metric = metrics_cols[0].empty()
    doc_metric = metrics_cols[1].empty()
    task_metric = metrics_cols[2].empty()
    trace_panel = status.empty()
    report_container = st.container()

    def refresh_ui() -> None:
        status.update(label=trace_state.status_label, state="running")
        web_metric.metric("Web findings", trace_state.web_count)
        doc_metric.metric("Document findings", trace_state.doc_count)
        task_metric.metric("Sub-tasks", len(trace_state.sub_tasks))
        trace_panel.markdown(render_trace_markdown(trace_state))

    def on_event(event: dict) -> None:
        trace_state.apply_event(event)
        refresh_ui()

    try:
        done = stream_research_events(query.strip(), export_report, on_event)

        if trace_state.error:
            status.update(label=trace_state.error, state="error")
            st.error(trace_state.error)
        elif done is None:
            status.update(label="Research failed", state="error")
            st.error("No final result returned from the API.")
        else:
            status.update(label="Research complete", state="complete")
            refresh_ui()

            with report_container:
                st.subheader(done.get("report_title") or "Research report")
                st.markdown(done.get("synthesis") or "")

                export = done.get("export")
                if export:
                    report_id = export["report_id"]
                    download_cols = st.columns(2)

                    md_response = httpx.get(f"{API_URL}/research/reports/{report_id}/markdown")
                    pdf_response = httpx.get(f"{API_URL}/research/reports/{report_id}/pdf")
                    md_response.raise_for_status()
                    pdf_response.raise_for_status()

                    download_cols[0].download_button(
                        "Download markdown",
                        data=md_response.content,
                        file_name=f"{report_id}.md",
                        mime="text/markdown",
                    )
                    download_cols[1].download_button(
                        "Download PDF",
                        data=pdf_response.content,
                        file_name=f"{report_id}.pdf",
                        mime="application/pdf",
                    )
    except httpx.HTTPError as exc:
        status.update(label="Request failed", state="error")
        st.error(f"Could not reach the API at {API_URL}: {exc}")
