"""Cross-encoder re-ranking for hybrid search results."""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from src.config.settings import Settings, get_settings
from src.models.schemas import DocumentChunk


@lru_cache
def _get_cross_encoder(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


def rerank(
    query: str,
    candidates: list[DocumentChunk],
    top_k: int = 5,
    settings: Settings | None = None,
) -> list[DocumentChunk]:
    """Re-rank candidate chunks with a cross-encoder model."""
    if not candidates:
        return []

    settings = settings or get_settings()
    model = _get_cross_encoder(settings.cross_encoder_model)
    pairs = [[query, chunk.text] for chunk in candidates]
    scores = model.predict(pairs)

    scored = sorted(
        zip(candidates, scores, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        chunk.model_copy(update={"score": float(score)})
        for chunk, score in scored[:top_k]
    ]
