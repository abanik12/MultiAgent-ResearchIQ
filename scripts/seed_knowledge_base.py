#!/usr/bin/env python3
"""Ingest all PDFs from the seed manifest into Qdrant + BM25."""

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag.ingestion import ingest_document

MANIFEST_PATH = ROOT / "data" / "sample_docs" / "manifest.json"
PDF_DIR = ROOT / "data" / "sample_docs" / "pdfs"


async def main() -> None:
    load_dotenv()

    if not MANIFEST_PATH.exists():
        print(f"Manifest not found: {MANIFEST_PATH}")
        sys.exit(1)

    entries = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    total_chunks = 0

    for entry in entries:
        pdf_path = PDF_DIR / entry["filename"]
        if not pdf_path.exists():
            print(f"Missing PDF: {pdf_path} — run scripts/download_seed_docs.py first")
            sys.exit(1)

        print(f"Ingesting: {entry['title']}")
        chunks, source = await ingest_document(
            pdf_path=str(pdf_path),
            title=entry["title"],
            category=entry["category"],
        )
        print(f"  → {chunks} chunks indexed from {source}")
        total_chunks += chunks

    print(f"\nSeed complete. Total new chunks indexed: {total_chunks}")


if __name__ == "__main__":
    asyncio.run(main())
