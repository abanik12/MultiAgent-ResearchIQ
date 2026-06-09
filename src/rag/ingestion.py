"""Document loading, semantic chunking, and dual-index ingestion."""

from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import Settings, get_settings
from src.rag.index_store import IndexStore, StoredChunk, make_chunk_id


def _build_embeddings(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key or None,
    )


def _semantic_split(documents: list[Document], settings: Settings) -> list[Document]:
    """Split documents using semantic chunking with fallback to recursive split."""
    try:
        splitter = SemanticChunker(
            _build_embeddings(settings),
            breakpoint_threshold_type="percentile",
        )
        return splitter.split_documents(documents)
    except Exception:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
        )
        return splitter.split_documents(documents)


def load_pdf(path: str | Path, title: str = "", category: str = "") -> list[Document]:
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    source = str(Path(path).resolve())
    for doc in docs:
        doc.metadata.setdefault("source", source)
        doc.metadata.setdefault("title", title or Path(path).stem)
        doc.metadata.setdefault("category", category)
    return docs


def load_url(url: str, title: str = "", category: str = "") -> list[Document]:
    loader = WebBaseLoader(url)
    docs = loader.load()
    for doc in docs:
        doc.metadata.setdefault("source", url)
        doc.metadata.setdefault("title", title or url)
        doc.metadata.setdefault("category", category)
    return docs


def load_text(text: str, source: str, title: str = "", category: str = "") -> list[Document]:
    return [
        Document(
            page_content=text,
            metadata={
                "source": source,
                "title": title or source,
                "category": category,
            },
        )
    ]


def _documents_to_stored_chunks(documents: list[Document]) -> list[StoredChunk]:
    chunks: list[StoredChunk] = []
    for index, doc in enumerate(documents):
        source = doc.metadata.get("source", "unknown")
        chunk_id = make_chunk_id(f"{source}:{hashlib.md5(doc.page_content.encode()).hexdigest()[:8]}", index)
        chunks.append(
            StoredChunk(
                chunk_id=chunk_id,
                text=doc.page_content,
                source=source,
                title=doc.metadata.get("title", ""),
                category=doc.metadata.get("category", ""),
            )
        )
    return chunks


async def ingest_document(
    *,
    url: str | None = None,
    text: str | None = None,
    pdf_path: str | None = None,
    title: str | None = None,
    category: str | None = None,
    store: IndexStore | None = None,
    settings: Settings | None = None,
) -> tuple[int, str]:
    """Load, chunk, embed, and index a document. Returns (chunks_indexed, source)."""
    settings = settings or get_settings()
    store = store or IndexStore(settings)

    if sum(bool(x) for x in (url, text, pdf_path)) != 1:
        raise ValueError("Provide exactly one of: url, text, pdf_path")

    if pdf_path:
        documents = load_pdf(pdf_path, title=title or "", category=category or "")
        source = str(Path(pdf_path).resolve())
    elif url:
        documents = load_url(url, title=title or "", category=category or "")
        source = url
    else:
        source = title or "inline-text"
        documents = load_text(text or "", source=source, title=title or "", category=category or "")

    if not documents:
        return 0, source

    split_docs = _semantic_split(documents, settings)
    stored_chunks = _documents_to_stored_chunks(split_docs)
    if not stored_chunks:
        return 0, source

    embeddings = _build_embeddings(settings)
    vectors = embeddings.embed_documents([chunk.text for chunk in stored_chunks])
    indexed = store.upsert_chunks(stored_chunks, vectors)
    return indexed, source
