#!/usr/bin/env python3
"""CLI to run the full ResearchIQ multi-agent research pipeline."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.config.settings import get_settings
from src.graph.graph import run_research


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full ResearchIQ research pipeline")
    parser.add_argument("query", help="Research question")
    parser.add_argument(
        "--skip-rerank",
        action="store_true",
        help="Skip Hugging Face cross-encoder reranking (faster; uses RAG_SKIP_RERANK from .env if set)",
    )
    args = parser.parse_args()

    if args.skip_rerank:
        os.environ["RAG_SKIP_RERANK"] = "true"
        get_settings.cache_clear()

    settings = get_settings()
    skip_rerank = settings.rag_skip_rerank
    recent_limit = settings.web_search_recent_limit_enabled

    print(f"Running full research pipeline for: {args.query}")
    print(
        "RAG reranking: "
        + ("disabled (RAG_SKIP_RERANK=true)" if skip_rerank else "enabled")
    )
    print(
        "Web search recent limit: "
        + (
            f"enabled (top {settings.web_search_recent_limit} total, by date)"
            if recent_limit
            else "disabled"
        )
        + "\n"
    )

    result = await run_research(args.query)

    settings = get_settings()
    if settings.report_export_enabled and result.get("synthesis"):
        from src.utils.report_export import export_report_files

        export_paths = export_report_files(result["synthesis"], args.query, settings=settings)
        print(f"\nExported markdown: {export_paths.markdown_path}")
        print(f"Exported PDF:      {export_paths.pdf_path}")

    print("Sub-tasks:")
    for index, task in enumerate(result["sub_tasks"], start=1):
        print(f"  {index}. {task}")

    print(f"\nWeb findings: {len(result['web_findings'])}")
    for finding in result["web_findings"][:5]:
        print(f"  - {finding.title} ({finding.url})")

    print(f"\nDocument findings: {len(result['doc_findings'])}")
    for finding in result["doc_findings"][:5]:
        title = finding.title or finding.source
        print(f"  - {title} (score={finding.score:.3f})")

    print("\n" + "=" * 60)
    print(result["synthesis"])
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
