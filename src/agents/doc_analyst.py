"""Document analyst agent — hybrid RAG over the curated knowledge base."""

from __future__ import annotations

import asyncio

from src.config.settings import Settings, get_settings
from src.models.schemas import DocumentChunk
from src.rag.hybrid_search import hybrid_search

DOC_ANALYST_TOP_K = 5


async def analyze_documents(
    task: str,
    top_k: int = DOC_ANALYST_TOP_K,
    settings: Settings | None = None,
) -> list[DocumentChunk]:
    """Retrieve relevant knowledge-base chunks for a sub-task."""
    settings = settings or get_settings()
    return await asyncio.to_thread(
        hybrid_search,
        task,
        top_k,
        20,
        None,
        settings,
        settings.rag_skip_rerank,
    )
