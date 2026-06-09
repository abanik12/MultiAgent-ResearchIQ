from langgraph.types import Send
from langgraph.graph import END, START, StateGraph

from src.graph.nodes import (
    coordinator_node,
    doc_analyst_node,
    report_writer_node,
    web_researcher_node,
)
from src.graph.state import AgentState, WorkerState
from src.config.settings import get_settings
from src.rag.index_store import close_index_store
from src.tools.search_tools import apply_web_findings_limit
from src.utils.tracing import build_graph_run_config, configure_langsmith


def route_to_agents(state: AgentState) -> list[Send]:
    """Dispatch each sub-task to web and doc agents in parallel."""
    sends: list[Send] = []
    for index, task in enumerate(state["sub_tasks"]):
        worker_state = WorkerState(query=state["query"], task=task, task_id=index)
        sends.append(Send("web_researcher", worker_state))
        sends.append(Send("doc_analyst", worker_state))
    return sends


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("coordinator", coordinator_node)
    graph.add_node("web_researcher", web_researcher_node)
    graph.add_node("doc_analyst", doc_analyst_node)
    graph.add_node("report_writer", report_writer_node)

    graph.add_edge(START, "coordinator")
    graph.add_conditional_edges(
        "coordinator",
        route_to_agents,
        ["web_researcher", "doc_analyst"],
    )
    graph.add_edge("web_researcher", "report_writer")
    graph.add_edge("doc_analyst", "report_writer")
    graph.add_edge("report_writer", END)

    return graph.compile()


async def run_research(query: str) -> AgentState:
    """Run the full research graph for a query."""
    settings = get_settings()
    configure_langsmith(settings)
    app = build_graph()
    run_config = build_graph_run_config(query, source="cli")
    initial_state: AgentState = {
        "query": query,
        "sub_tasks": [],
        "web_findings": [],
        "doc_findings": [],
        "synthesis": "",
        "report": None,
        "messages": [],
    }
    try:
        result = await app.ainvoke(initial_state, config=run_config)
        result = {
            **result,
            "web_findings": apply_web_findings_limit(result["web_findings"], settings),
        }
        return result
    finally:
        close_index_store()
