"""LangChain tools wrapping the local hybrid RAG pipeline."""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from src.rag.hybrid_search import hybrid_search


def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """Search the curated knowledge base and return formatted passages."""
    chunks = hybrid_search(query, top_k=top_k)
    if not chunks:
        return "No relevant documents found."

    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        title = chunk.title or chunk.source or "Untitled"
        lines.append(
            f"[{index}] {title} (score={chunk.score:.3f}, source={chunk.source})\n"
            f"{chunk.text[:600]}"
        )
    return "\n\n".join(lines)


def get_rag_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=search_knowledge_base,
            name="search_knowledge_base",
            description=(
                "Search the curated research knowledge base (papers on LLMs, RAG, "
                "transformers, etc.) for relevant passages."
            ),
        )
    ]
