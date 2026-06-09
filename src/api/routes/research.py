"""Research pipeline API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from src.graph.streaming import stream_research
from src.models.schemas import ResearchRequest
from src.utils.report_export import resolve_report_path

router = APIRouter(prefix="/research", tags=["research"])


@router.post("")
async def run_research_stream(request: ResearchRequest) -> StreamingResponse:
    """Run the full research pipeline and stream progress as SSE."""

    async def event_generator():
        async for event in stream_research(
            request.query,
            export_report=request.export_report,
        ):
            yield f"data: {event}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/reports/{report_id}/markdown")
async def download_markdown_report(report_id: str) -> FileResponse:
    """Download an exported markdown report by id."""
    try:
        path = resolve_report_path(report_id, "md")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return FileResponse(
        path,
        media_type="text/markdown",
        filename=f"{report_id}.md",
    )


@router.get("/reports/{report_id}/pdf")
async def download_pdf_report(report_id: str) -> FileResponse:
    """Download an exported PDF report by id."""
    try:
        path = resolve_report_path(report_id, "pdf")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return FileResponse(
        path,
        media_type="application/pdf",
        filename=f"{report_id}.pdf",
    )
