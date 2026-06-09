"""Document analyst agent — hybrid RAG over the curated knowledge base."""

from __future__ import annotations

import asyncio
from typing import Any

from src.config.settings import Settings, get_settings
from src.models.schemas import DocumentChunk
from src.rag.hybrid_search import hybrid_search

DOC_ANALYST_TOP_K = 5


def build_doc_trace_steps(
    task: str,
    chunks: list[DocumentChunk],
    *,
    task_id: int | None = None,
) -> list[dict[str, Any]]:
    summary_task = task if len(task) <= 80 else f"{task[:77]}..."
    steps: list[dict[str, Any]] = [
        {
            "type": "trace",
            "agent": "doc_analyst",
            "message": f'Searching KB: "{summary_task}"',
            "task_id": task_id,
        }
    ]
    if not chunks:
        steps.append(
            {
                "type": "trace",
                "agent": "doc_analyst",
                "message": "No matching knowledge-base chunks found",
                "task_id": task_id,
            }
        )
        return steps

    top_score = chunks[0].score
    steps.append(
        {
            "type": "trace",
            "agent": "doc_analyst",
            "message": f"Retrieved {len(chunks)} chunk(s) (top score {top_score:.2f})",
            "task_id": task_id,
        }
    )
    return steps


async def analyze_documents(
    task: str,
    top_k: int = DOC_ANALYST_TOP_K,
    settings: Settings | None = None,
    *,
    task_id: int | None = None,
) -> tuple[list[DocumentChunk], list[dict[str, Any]]]:
    """Retrieve relevant knowledge-base chunks for a sub-task."""
    settings = settings or get_settings()
    chunks = await asyncio.to_thread(
        hybrid_search,
        task,
        top_k,
        20,
        None,
        settings,
        settings.rag_skip_rerank,
    )
    return chunks, build_doc_trace_steps(task, chunks, task_id=task_id)
