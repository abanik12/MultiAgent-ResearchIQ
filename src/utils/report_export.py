"""Export research reports to markdown and PDF files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from fpdf import FPDF

from src.config.settings import Settings, get_settings
from src.models.schemas import ReportExportPaths


@dataclass(frozen=True)
class MarkdownBlock:
    kind: str
    text: str


class ReportPDF(FPDF):
    """Styled PDF renderer for ResearchIQ markdown reports."""

    def __init__(self) -> None:
        super().__init__()
        self.set_auto_page_break(auto=True, margin=22)
        self.set_margins(left=22, top=22, right=22)
        self._title_for_header = "ResearchIQ Report"

    def set_document_title(self, title: str) -> None:
        self._title_for_header = title[:80] or "ResearchIQ Report"

    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(130, 130, 140)
        self.cell(0, 8, self._title_for_header, align="L")
        self.ln(10)

    def footer(self) -> None:
        self.set_y(-14)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(130, 130, 140)
        self.cell(0, 8, f"ResearchIQ  ·  Page {self.page_no()}", align="C")

    def _rule(self) -> None:
        y = self.get_y() + 1.5
        self.set_draw_color(210, 214, 220)
        self.set_line_width(0.3)
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.ln(5)

    def render_block(self, block: MarkdownBlock) -> None:
        width = self.epw

        if block.kind == "title":
            self.ln(2)
            self.set_font("Helvetica", "B", 22)
            self.set_text_color(26, 32, 44)
            self.multi_cell(width, 11, block.text)
            self._rule()
            return

        if block.kind == "h2":
            self.ln(5)
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(31, 41, 55)
            self.multi_cell(width, 8, block.text)
            self.ln(2)
            return

        if block.kind == "h3":
            self.ln(3)
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(55, 65, 81)
            self.multi_cell(width, 6, block.text)
            self.ln(1)
            return

        if block.kind == "paragraph":
            self.set_font("Helvetica", "", 11)
            self.set_text_color(55, 65, 81)
            self.multi_cell(width, 6, block.text)
            self.ln(3)
            return

        if block.kind == "bullet":
            self.set_font("Helvetica", "", 10)
            self.set_text_color(55, 65, 81)
            bullet_x = self.l_margin + 2
            text_x = self.l_margin + 8
            self.set_x(text_x)
            self.cell(4, 6, "-")
            self.multi_cell(width - (text_x - self.l_margin) - 2, 5.5, block.text)
            self.ln(1.5)
            return

        self.set_text_color(0, 0, 0)


def _slugify(text: str, max_length: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    if not slug:
        slug = "research-report"
    return slug[:max_length].strip("-") or "research-report"


def _pdf_safe(text: str) -> str:
    """Normalize text for Helvetica and break very long tokens (URLs)."""
    cleaned = (
        text.replace("\u2014", "-")
        .replace("\u2013", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .encode("latin-1", errors="replace")
        .decode("latin-1")
    )
    wrapped_words: list[str] = []
    for word in cleaned.split():
        if len(word) <= 90:
            wrapped_words.append(word)
            continue
        wrapped_words.extend(word[index : index + 90] for index in range(0, len(word), 90))
    return " ".join(wrapped_words)


def _clean_inline_markdown(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return _pdf_safe(text.strip())


def parse_markdown_blocks(markdown: str) -> list[MarkdownBlock]:
    """Parse report markdown into renderable blocks with merged paragraphs."""
    blocks: list[MarkdownBlock] = []
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        blocks.append(MarkdownBlock("paragraph", _clean_inline_markdown(" ".join(paragraph_lines))))
        paragraph_lines.clear()

    for raw_line in markdown.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            flush_paragraph()
            continue

        if stripped.startswith("# "):
            flush_paragraph()
            blocks.append(MarkdownBlock("title", _clean_inline_markdown(stripped[2:])))
        elif stripped.startswith("## "):
            flush_paragraph()
            blocks.append(MarkdownBlock("h2", _clean_inline_markdown(stripped[3:])))
        elif stripped.startswith("### "):
            flush_paragraph()
            blocks.append(MarkdownBlock("h3", _clean_inline_markdown(stripped[4:])))
        elif stripped.startswith("- "):
            flush_paragraph()
            blocks.append(MarkdownBlock("bullet", _clean_inline_markdown(stripped[2:])))
        else:
            paragraph_lines.append(stripped)

    flush_paragraph()
    return blocks


def markdown_to_pdf(
    markdown: str,
    output_path: Path,
    *,
    query: str | None = None,
) -> None:
    """Convert markdown text to a styled PDF report."""
    blocks = parse_markdown_blocks(markdown)
    pdf = ReportPDF()
    pdf.add_page()

    title = next((block.text for block in blocks if block.kind == "title"), "ResearchIQ Report")
    pdf.set_document_title(title)

    if query:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(100, 116, 139)
        pdf.multi_cell(pdf.epw, 5, _pdf_safe(f"Query: {query}"))
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(
            pdf.epw,
            5,
            datetime.now(timezone.utc).strftime("Generated %Y-%m-%d %H:%M UTC"),
        )
        pdf.ln(4)

    for block in blocks:
        pdf.render_block(block)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))


def export_report_files(
    markdown: str,
    query: str,
    *,
    settings: Settings | None = None,
) -> ReportExportPaths:
    """Write markdown and PDF report files and return their paths."""
    settings = settings or get_settings()
    output_dir = Path(settings.report_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    report_id = f"{timestamp}_{_slugify(query)}"
    markdown_path = output_dir / f"{report_id}.md"
    pdf_path = output_dir / f"{report_id}.pdf"

    markdown_path.write_text(markdown, encoding="utf-8")
    markdown_to_pdf(markdown, pdf_path, query=query)

    return ReportExportPaths(
        report_id=report_id,
        markdown_path=str(markdown_path),
        pdf_path=str(pdf_path),
    )


def resolve_report_path(report_id: str, extension: str, settings: Settings | None = None) -> Path:
    """Resolve a report file path and ensure it stays under the output directory."""
    settings = settings or get_settings()
    output_dir = Path(settings.report_output_dir).resolve()
    if ".." in report_id or "/" in report_id or "\\" in report_id:
        raise ValueError("Invalid report id")

    candidate = (output_dir / f"{report_id}.{extension}").resolve()
    if not str(candidate).startswith(str(output_dir)):
        raise ValueError("Invalid report path")
    if not candidate.exists():
        raise FileNotFoundError(f"Report not found: {report_id}.{extension}")
    return candidate
