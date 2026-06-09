"""Tests for RAG ingestion, hybrid search, and re-ranking."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from src.api.main import app
from src.config.settings import Settings
from src.models.schemas import DocumentChunk
from src.rag.hybrid_search import hybrid_search, reciprocal_rank_fusion
from src.rag.index_store import IndexStore, StoredChunk
from src.rag.reranker import rerank


@pytest.fixture
def test_settings(tmp_path):
    return Settings(
        openai_api_key="test-key",
        qdrant_url="http://localhost:6333",
        qdrant_collection="test_researchiq",
        bm25_index_path=str(tmp_path / "bm25_index.json"),
    )


@pytest.fixture
def memory_store(test_settings, tmp_path):
    client = QdrantClient(":memory:")
    client.upsert = MagicMock()
    store = IndexStore(settings=test_settings, client=client)
    with patch.object(
        store,
        "_embeddings",
        MagicMock(
            embed_query=lambda q: [0.1, 0.2, 0.3],
            embed_documents=lambda docs: [[0.1, 0.2, 0.3] for _ in docs],
        ),
    ):
        with patch.object(store, "ensure_collection"):
            yield store


def test_reciprocal_rank_fusion_combines_lists():
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    sparse = [("b", 5.0), ("d", 4.0), ("a", 3.0)]
    fused = reciprocal_rank_fusion([dense, sparse])
    ids = [chunk_id for chunk_id, _ in fused]
    assert ids[0] in {"a", "b"}
    assert "d" in ids


def test_index_store_bm25_persistence(test_settings, tmp_path):
    client = QdrantClient(":memory:")
    store = IndexStore(settings=test_settings, client=client)
    client.upsert = MagicMock()

    chunks = [
        StoredChunk("c1", "transformer self-attention mechanism", "doc1", "Attention", "transformers"),
        StoredChunk("c2", "vision transformer patch embeddings", "doc2", "ViT", "vision"),
    ]
    vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    with patch.object(store, "ensure_collection"):
        store.upsert_chunks(chunks, vectors)

    reloaded = IndexStore(settings=test_settings, client=QdrantClient(":memory:"))
    assert reloaded.chunk_count == 2
    assert reloaded.get_chunk_by_id("c1") is not None

    results = reloaded.bm25_search("self-attention transformers", limit=5)
    assert results[0][0] == "c1"


@pytest.mark.asyncio
async def test_ingest_text_indexes_chunks(memory_store, test_settings):
    mock_embeddings = MagicMock(
        embed_documents=lambda docs: [[0.1, 0.2, 0.3] for _ in docs],
    )
    with (
        patch("src.rag.ingestion._semantic_split") as mock_split,
        patch("src.rag.ingestion._build_embeddings", return_value=mock_embeddings),
    ):
        from langchain_core.documents import Document

        from src.rag.ingestion import ingest_document

        mock_split.return_value = [
            Document(page_content="Transformers use self-attention.", metadata={"source": "Test Doc"})
        ]
        chunks_indexed, source = await ingest_document(
            text="Transformers use self-attention for sequence modeling.",
            title="Test Doc",
            category="transformers",
            store=memory_store,
            settings=test_settings,
        )
    assert chunks_indexed >= 1
    assert source == "Test Doc"
    assert memory_store.chunk_count >= 1


def test_hybrid_search_with_mocked_store():
    store = MagicMock()
    store.dense_search.return_value = [("c1", 0.9), ("c2", 0.8)]
    store.bm25_search.return_value = [("c2", 5.0), ("c3", 4.0)]
    store.get_chunk_by_id.side_effect = lambda cid: {
        "c1": StoredChunk("c1", "attention is all you need", "doc1", "Attention", "transformers"),
        "c2": StoredChunk("c2", "vision transformer patches", "doc2", "ViT", "vision"),
        "c3": StoredChunk("c3", "clip contrastive learning", "doc3", "CLIP", "vision"),
    }.get(cid)

    with patch("src.rag.hybrid_search.rerank") as mock_rerank:
        mock_rerank.return_value = [
            DocumentChunk(chunk_id="c1", text="attention is all you need", source="doc1", score=0.95)
        ]
        results = hybrid_search("transformer attention", top_k=1, store=store, skip_rerank=False)

    assert len(results) == 1
    assert results[0].chunk_id == "c1"
    mock_rerank.assert_called_once()


def test_rerank_orders_by_score():
    candidates = [
        DocumentChunk(chunk_id="c1", text="low relevance", source="s1"),
        DocumentChunk(chunk_id="c2", text="high relevance match", source="s2"),
    ]
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.1, 0.9]

    with patch("src.rag.reranker._get_cross_encoder", return_value=mock_model):
        ranked = rerank("high relevance", candidates, top_k=2)

    assert ranked[0].chunk_id == "c2"
    assert ranked[0].score == pytest.approx(0.9)


def test_ingest_api_text():
    client = TestClient(app)
    with patch(
        "src.api.routes.ingest.ingest_document",
        return_value=(3, "inline-text"),
    ) as mock_ingest:
        response = client.post(
            "/ingest",
            json={"text": "Sample transformer architecture overview.", "title": "Test"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["chunks_indexed"] == 3
    assert data["source"] == "inline-text"
    mock_ingest.assert_called_once()


def test_ingest_api_validation_error():
    client = TestClient(app)
    response = client.post("/ingest", json={"text": "a", "url": "https://example.com"})
    assert response.status_code == 400
