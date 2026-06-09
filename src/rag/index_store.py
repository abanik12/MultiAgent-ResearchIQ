"""Qdrant dense index + BM25 sparse index with JSON persistence."""

from __future__ import annotations

import atexit
import json
import threading
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from src.config.settings import Settings, get_settings
from src.models.schemas import DocumentChunk

_local_qdrant_lock = threading.RLock()
_index_store: "IndexStore | None" = None
_index_store_key: tuple[Any, ...] | None = None


def create_qdrant_client(settings: Settings | None = None) -> QdrantClient:
    """Create a Qdrant client — local embedded mode by default (no Docker required)."""
    settings = settings or get_settings()
    if settings.qdrant_mode == "server":
        return QdrantClient(url=settings.qdrant_url)
    local_path = Path(settings.qdrant_local_path)
    local_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(local_path))


class StoredChunk:
    """Internal chunk record shared by Qdrant and BM25 indexes."""

    __slots__ = ("chunk_id", "text", "source", "title", "category")

    def __init__(
        self,
        chunk_id: str,
        text: str,
        source: str,
        title: str = "",
        category: str = "",
    ) -> None:
        self.chunk_id = chunk_id
        self.text = text
        self.source = source
        self.title = title
        self.category = category

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "title": self.title,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StoredChunk:
        return cls(
            chunk_id=data["chunk_id"],
            text=data["text"],
            source=data.get("source", ""),
            title=data.get("title", ""),
            category=data.get("category", ""),
        )

    def to_document_chunk(self, score: float = 0.0) -> DocumentChunk:
        return DocumentChunk(
            chunk_id=self.chunk_id,
            text=self.text,
            source=self.source,
            title=self.title,
            category=self.category,
            score=score,
        )


class IndexStore:
    """Dual-index store: Qdrant (dense) + BM25 (sparse, JSON-backed)."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client or create_qdrant_client(self.settings)
        self._embeddings = OpenAIEmbeddings(
            model=self.settings.openai_embedding_model,
            api_key=self.settings.openai_api_key or None,
        )
        self._bm25_path = Path(self.settings.bm25_index_path)
        self._chunks: list[StoredChunk] = []
        self._bm25: BM25Okapi | None = None
        self._closed = False
        self._load_bm25()

    def close(self) -> None:
        """Release the Qdrant client before interpreter shutdown."""
        if self._closed:
            return
        with _local_qdrant_guard(self.settings):
            close_fn = getattr(self._client, "close", None)
            if close_fn is not None:
                close_fn()
        self._client = None  # type: ignore[assignment]
        self._closed = True

    @property
    def embedding_dimension(self) -> int:
        probe = self._embeddings.embed_query("dimension probe")
        return len(probe)

    def ensure_collection(self) -> None:
        with _local_qdrant_guard(self.settings):
            collections = [c.name for c in self._client.get_collections().collections]
            if self.settings.qdrant_collection not in collections:
                dim = self.embedding_dimension
                self._client.create_collection(
                    collection_name=self.settings.qdrant_collection,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )

    def _load_bm25(self) -> None:
        if not self._bm25_path.exists():
            self._chunks = []
            self._bm25 = None
            return
        raw = json.loads(self._bm25_path.read_text(encoding="utf-8"))
        self._chunks = [StoredChunk.from_dict(item) for item in raw.get("chunks", [])]
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        if not self._chunks:
            self._bm25 = None
            return
        tokenized = [chunk.text.lower().split() for chunk in self._chunks]
        self._bm25 = BM25Okapi(tokenized)

    def _save_bm25(self) -> None:
        self._bm25_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"chunks": [chunk.to_dict() for chunk in self._chunks]}
        self._bm25_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def get_chunk_by_id(self, chunk_id: str) -> StoredChunk | None:
        for chunk in self._chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None

    def upsert_chunks(self, chunks: list[StoredChunk], vectors: list[list[float]]) -> int:
        if not chunks:
            return 0

        self.ensure_collection()
        existing_ids = {c.chunk_id for c in self._chunks}
        new_pairs = [
            (chunk, vector)
            for chunk, vector in zip(chunks, vectors, strict=True)
            if chunk.chunk_id not in existing_ids
        ]
        if not new_pairs:
            return 0

        new_chunks = [pair[0] for pair in new_pairs]
        new_vectors = [pair[1] for pair in new_pairs]
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                vector=vector,
                payload=chunk.to_dict(),
            )
            for chunk, vector in zip(new_chunks, new_vectors, strict=True)
        ]
        with _local_qdrant_guard(self.settings):
            self._client.upsert(
                collection_name=self.settings.qdrant_collection,
                points=points,
            )

        self._chunks.extend(new_chunks)
        self._rebuild_bm25()
        self._save_bm25()
        return len(new_chunks)

    def dense_search(self, query: str, limit: int = 20) -> list[tuple[str, float]]:
        if not self._chunks:
            return []
        self.ensure_collection()
        query_vector = self._embeddings.embed_query(query)
        with _local_qdrant_guard(self.settings):
            response = self._client.query_points(
                collection_name=self.settings.qdrant_collection,
                query=query_vector,
                limit=limit,
            )
        return [
            (hit.payload["chunk_id"], hit.score)
            for hit in response.points
            if hit.payload
        ]

    def bm25_search(self, query: str, limit: int = 20) -> list[tuple[str, float]]:
        if not self._bm25 or not self._chunks:
            return []
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [
            (self._chunks[idx].chunk_id, float(score))
            for idx, score in ranked[:limit]
        ]

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)


def make_chunk_id(source: str, index: int) -> str:
    return f"{uuid.uuid5(uuid.NAMESPACE_URL, source)}-{index}"


def _settings_store_key(settings: Settings) -> tuple[Any, ...]:
    return (
        settings.qdrant_mode,
        settings.qdrant_local_path,
        settings.qdrant_url,
        settings.qdrant_collection,
        settings.bm25_index_path,
        settings.openai_embedding_model,
    )


def get_index_store(settings: Settings | None = None) -> IndexStore:
    """Return a process-wide IndexStore singleton (required for local Qdrant mode)."""
    global _index_store, _index_store_key

    settings = settings or get_settings()
    key = _settings_store_key(settings)

    with _local_qdrant_lock:
        if _index_store is None or _index_store_key != key:
            _index_store = IndexStore(settings=settings)
            _index_store_key = key
        return _index_store


def close_index_store() -> None:
    """Close and release the cached store (call before process exit)."""
    global _index_store, _index_store_key

    with _local_qdrant_lock:
        if _index_store is not None:
            _index_store.close()
        _index_store = None
        _index_store_key = None


def reset_index_store() -> None:
    """Clear the cached store (useful in tests)."""
    close_index_store()


def _local_qdrant_guard(settings: Settings):
    if settings.qdrant_mode == "local":
        return _local_qdrant_lock
    return nullcontext()


atexit.register(close_index_store)
