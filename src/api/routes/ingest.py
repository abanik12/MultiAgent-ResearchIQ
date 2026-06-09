"""Document ingestion API routes."""

from fastapi import APIRouter, Depends, HTTPException

from src.api.rate_limit import ingest_rate_limit
from src.models.schemas import IngestRequest, IngestResponse
from src.rag.ingestion import ingest_document

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
    _: None = Depends(ingest_rate_limit),
) -> IngestResponse:
    """Add a PDF, URL, or text document to the knowledge base."""
    provided = [request.url, request.text, request.pdf_path]
    if sum(field is not None for field in provided) != 1:
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of: url, text, pdf_path",
        )

    try:
        chunks_indexed, source = await ingest_document(
            url=request.url,
            text=request.text,
            pdf_path=request.pdf_path,
            title=request.title,
            category=request.category,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return IngestResponse(
        chunks_indexed=chunks_indexed,
        source=source,
        message=f"Successfully indexed {chunks_indexed} chunks",
    )
