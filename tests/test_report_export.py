"""Tests for markdown/PDF report export."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.settings import Settings
from src.utils.report_export import (
    export_report_files,
    markdown_to_pdf,
    parse_markdown_blocks,
    resolve_report_path,
)


@pytest.fixture
def export_settings(tmp_path):
    return Settings(report_output_dir=str(tmp_path / "reports"))


def test_parse_markdown_blocks_merges_paragraph_lines():
    markdown = "# Title\n\nFirst sentence.\nSecond sentence.\n\n## Section\n\nBody text."
    blocks = parse_markdown_blocks(markdown)
    assert blocks[0].kind == "title"
    assert blocks[1].kind == "paragraph"
    assert blocks[1].text == "First sentence. Second sentence."
    assert blocks[2].kind == "h2"


def test_parse_markdown_blocks_converts_links():
    blocks = parse_markdown_blocks("- [Example](https://example.com)")
    assert blocks[0].text == "Example (https://example.com)"


def test_export_report_files_writes_markdown_and_pdf(export_settings):
    markdown = (
        "# Demo Report\n\n"
        "Summary paragraph with enough text to wrap cleanly across the page width.\n\n"
        "## Findings\n\n"
        "A longer paragraph that should render as a single block rather than one line per row.\n\n"
        "## Sources\n\n"
        "### From Web\n"
        "- Example Source (https://example.com)\n"
        "### From Document KB\n"
        "- Attention Is All You Need (data/sample_docs/attention.pdf)"
    )
    paths = export_report_files(
        markdown,
        "Demo query about transformers",
        settings=export_settings,
    )

    md_path = Path(paths.markdown_path)
    pdf_path = Path(paths.pdf_path)
    assert md_path.exists()
    assert pdf_path.exists()
    assert md_path.read_text(encoding="utf-8") == markdown
    assert pdf_path.stat().st_size > 1000


def test_resolve_report_path_rejects_invalid_ids(export_settings, tmp_path):
    export_report_files("# Title\n\nBody", "safe query", settings=export_settings)
    with pytest.raises(ValueError):
        resolve_report_path("../escape", "md", settings=export_settings)


def test_markdown_to_pdf_creates_file(tmp_path):
    output = tmp_path / "sample.pdf"
    markdown_to_pdf(
        "# Heading\n\nParagraph text that reads like a real report section.",
        output,
        query="Sample query",
    )
    assert output.exists()
    assert output.stat().st_size > 500
