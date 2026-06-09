#!/usr/bin/env python3
"""Download ArXiv PDFs listed in data/sample_docs/manifest.json."""

import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "sample_docs" / "manifest.json"
PDF_DIR = ROOT / "data" / "sample_docs" / "pdfs"


def download_pdf(arxiv_id: str, destination: Path) -> None:
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    response = httpx.get(url, follow_redirects=True, timeout=120.0)
    response.raise_for_status()
    destination.write_bytes(response.content)


def main() -> None:
    if not MANIFEST_PATH.exists():
        print(f"Manifest not found: {MANIFEST_PATH}")
        sys.exit(1)

    entries = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    for entry in entries:
        destination = PDF_DIR / entry["filename"]
        if destination.exists() and destination.stat().st_size > 0:
            print(f"Skip (exists): {entry['title']}")
            skipped += 1
            continue
        print(f"Downloading: {entry['title']} ({entry['arxiv_id']})")
        download_pdf(entry["arxiv_id"], destination)
        downloaded += 1

    print(f"\nDone. Downloaded: {downloaded}, skipped: {skipped}, total: {len(entries)}")


if __name__ == "__main__":
    main()
