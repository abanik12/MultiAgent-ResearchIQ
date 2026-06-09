"""Hybrid retrieval: BM25 + dense vectors fused with RRF, then cross-encoder re-rank."""

from __future__ import annotations

from src.config.settings import Settings, get_settings
from src.models.schemas import DocumentChunk
from src.rag.index_store import IndexStore, get_index_store
from src.rag.reranker import rerank

RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (chunk_id, _raw_score) in enumerate(ranked):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def hybrid_search(
    query: str,
    top_k: int = 5,
    retrieval_k: int = 20,
    store: IndexStore | None = None,
    settings: Settings | None = None,
    skip_rerank: bool = False,
) -> list[DocumentChunk]:
    """Run hybrid search with optional cross-encoder re-ranking."""
    settings = settings or get_settings()
    store = store or get_index_store(settings)

    dense = store.dense_search(query, limit=retrieval_k)
    sparse = store.bm25_search(query, limit=retrieval_k)
    fused = reciprocal_rank_fusion([dense, sparse])

    candidates: list[DocumentChunk] = []
    for chunk_id, rrf_score in fused[: max(top_k * 2, 10)]:
        stored = store.get_chunk_by_id(chunk_id)
        if stored:
            candidates.append(stored.to_document_chunk(score=rrf_score))

    if skip_rerank:
        return candidates[:top_k]

    return rerank(query, candidates, top_k=top_k, settings=settings)
