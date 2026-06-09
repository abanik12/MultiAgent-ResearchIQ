#!/usr/bin/env python3
"""Query the local knowledge base with hybrid search (BM25 + Qdrant + rerank)."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

# Explicit path — load_dotenv() without args fails in some invocation contexts (heredoc)
load_dotenv(ROOT / ".env")

from src.rag.hybrid_search import hybrid_search
from src.rag.index_store import IndexStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the ResearchIQ knowledge base")
    parser.add_argument("query", help="Search query")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of results")
    parser.add_argument(
        "--skip-rerank",
        action="store_true",
        help="Skip cross-encoder re-ranking (faster)",
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    store = IndexStore()
    print(f"Indexed chunks: {store.chunk_count}", file=sys.stderr)

    if store.chunk_count == 0:
        print("No chunks indexed. Run: python scripts/seed_knowledge_base.py", file=sys.stderr)
        sys.exit(1)

    results = hybrid_search(
        args.query,
        top_k=args.top_k,
        store=store,
        skip_rerank=args.skip_rerank,
    )

    if args.json:
        print(json.dumps([r.model_dump() for r in results], indent=2))
        return

    print(f"\nQuery: {args.query}\n")
    if not results:
        print("No results found.")
        return

    for i, r in enumerate(results, 1):
        preview = r.text[:300].replace("\n", " ")
        if len(r.text) > 300:
            preview += "..."
        print(f"--- Result {i} (score={r.score:.4f}) ---")
        print(f"Title:    {r.title}")
        print(f"Category: {r.category}")
        print(f"Source:   {r.source}")
        print(f"Text:     {preview}\n")


if __name__ == "__main__":
    main()
