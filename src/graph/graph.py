from langgraph.types import Send
from langgraph.graph import END, START, StateGraph

from src.graph.nodes import (
    coordinator_node,
    doc_analyst_node,
    report_writer_node,
    web_researcher_node,
)
from src.graph.state import AgentState, WorkerState


async def research_worker(state: WorkerState) -> dict:
    """Run web + doc stubs for a single sub-task (Phase 1)."""
    stub_state: AgentState = {
        "query": state["task"],
        "sub_tasks": [state["task"]],
        "web_findings": [],
        "doc_findings": [],
        "synthesis": "",
        "report": None,
        "messages": [],
    }
    await web_researcher_node(stub_state)
    await doc_analyst_node(stub_state)
    return {"web_findings": [], "doc_findings": []}


def route_to_agents(state: AgentState) -> list[Send]:
    """Dispatch each sub-task to a research worker in parallel."""
    return [
        Send(
            "research_worker",
            WorkerState(query=state["query"], task=task, task_id=i),
        )
        for i, task in enumerate(state["sub_tasks"])
    ]


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("coordinator", coordinator_node)
    graph.add_node("research_worker", research_worker)
    graph.add_node("report_writer", report_writer_node)

    graph.add_edge(START, "coordinator")
    graph.add_conditional_edges("coordinator", route_to_agents, ["research_worker"])
    graph.add_edge("research_worker", "report_writer")
    graph.add_edge("report_writer", END)

    return graph.compile()


async def run_research(query: str) -> AgentState:
    """Run the full research graph for a query."""
    app = build_graph()
    initial_state: AgentState = {
        "query": query,
        "sub_tasks": [],
        "web_findings": [],
        "doc_findings": [],
        "synthesis": "",
        "report": None,
        "messages": [],
    }
    result = await app.ainvoke(initial_state)
    return result
