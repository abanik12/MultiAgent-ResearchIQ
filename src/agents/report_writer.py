"""Report writer — synthesizes findings into a structured markdown report."""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.coordinator import _build_llm, _configure_tracing
from src.config.settings import Settings, get_settings
from src.models.schemas import DocumentChunk, ResearchReport, SearchResult
from src.tools.search_tools import apply_web_findings_limit

REPORT_WRITER_PROMPT = """You are a research report writer for ResearchIQ.

Synthesize web and document findings into a clear, well-structured research report.
Each section string should be markdown with a heading and body paragraphs.

Rules:
- Base the report only on provided findings — do not invent facts
- Include a concise executive summary in the summary field
- Organize sections by theme, not by agent
- Do not populate the sources field — sources are appended automatically from findings
- Be balanced and note gaps or conflicting evidence when present
"""


def _format_findings_context(
    query: str,
    sub_tasks: list[str],
    web_findings: list[SearchResult],
    doc_findings: list[DocumentChunk],
) -> str:
    parts = [f"# Original query\n{query}\n", "## Sub-tasks"]
    for index, task in enumerate(sub_tasks, start=1):
        parts.append(f"{index}. {task}")

    parts.append("\n## Web findings")
    if web_findings:
        for index, finding in enumerate(web_findings, start=1):
            parts.append(
                f"{index}. [{finding.title}]({finding.url})\n   {finding.snippet[:400]}"
            )
    else:
        parts.append("(none)")

    parts.append("\n## Document findings")
    if doc_findings:
        for index, finding in enumerate(doc_findings, start=1):
            parts.append(
                f"{index}. {finding.title or finding.source} "
                f"(score={finding.score:.3f})\n   {finding.text[:400]}"
            )
    else:
        parts.append("(none)")

    return "\n".join(parts)


def _unique_web_sources(findings: list[SearchResult]) -> list[str]:
    seen_urls: set[str] = set()
    sources: list[str] = []
    for finding in findings:
        if not finding.url or finding.url in seen_urls:
            continue
        seen_urls.add(finding.url)
        if finding.title:
            sources.append(f"{finding.title} — {finding.url}")
        else:
            sources.append(finding.url)
    return sources


def _unique_doc_sources(findings: list[DocumentChunk]) -> list[str]:
    seen_keys: set[str] = set()
    sources: list[str] = []
    for finding in findings:
        key = finding.source or finding.title or finding.chunk_id
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        title = finding.title or Path(finding.source).stem if finding.source else "Unknown document"
        if finding.source and finding.source != title:
            sources.append(f"{title} ({finding.source})")
        else:
            sources.append(title)
    return sources


def report_to_markdown(
    report: ResearchReport,
    *,
    web_findings: list[SearchResult] | None = None,
    doc_findings: list[DocumentChunk] | None = None,
) -> str:
    lines = [f"# {report.title}", "", report.summary, ""]
    for section in report.sections:
        lines.append(section)
        lines.append("")

    web_sources = _unique_web_sources(web_findings or [])
    doc_sources = _unique_doc_sources(doc_findings or [])

    if not web_sources and not doc_sources and report.sources:
        lines.append("## Sources")
        for source in report.sources:
            lines.append(f"- {source}")
    elif web_sources or doc_sources:
        lines.append("## Sources")
        if web_sources:
            lines.append("")
            lines.append("### From Web")
            for source in web_sources:
                lines.append(f"- {source}")
        if doc_sources:
            lines.append("")
            lines.append("### From Document KB")
            for source in doc_sources:
                lines.append(f"- {source}")

    return "\n".join(lines).strip()


async def write_report(
    query: str,
    sub_tasks: list[str],
    web_findings: list[SearchResult],
    doc_findings: list[DocumentChunk],
    settings: Settings | None = None,
) -> tuple[ResearchReport, str]:
    settings = settings or get_settings()
    _configure_tracing(settings)
    llm = _build_llm(settings)
    structured_llm = llm.with_structured_output(ResearchReport)

    web_findings = apply_web_findings_limit(web_findings, settings)
    context = _format_findings_context(query, sub_tasks, web_findings, doc_findings)
    messages = [
        SystemMessage(content=REPORT_WRITER_PROMPT),
        HumanMessage(content=context),
    ]
    report: ResearchReport = await structured_llm.ainvoke(messages)
    synthesis = report_to_markdown(
        report,
        web_findings=web_findings,
        doc_findings=doc_findings,
    )
    return report, synthesis
