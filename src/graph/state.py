import operator
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

from src.models.schemas import DocumentChunk, ResearchReport, SearchResult


class AgentState(TypedDict):
    query: str
    sub_tasks: list[str]
    web_findings: Annotated[list[SearchResult], operator.add]
    doc_findings: Annotated[list[DocumentChunk], operator.add]
    synthesis: str
    report: ResearchReport | None
    messages: Annotated[list, add_messages]


class WorkerState(TypedDict):
    """State passed to parallel worker nodes via Send()."""

    query: str
    task: str
    task_id: int
