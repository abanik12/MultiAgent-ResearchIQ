from langchain_core.messages import AIMessage

from src.agents.coordinator import plan_research
from src.graph.state import AgentState


async def coordinator_node(state: AgentState) -> dict:
    """Decompose the query into sub-tasks via structured LLM output."""
    plan = await plan_research(state["query"])
    sub_task_descriptions = [task.description for task in plan.sub_tasks]

    return {
        "sub_tasks": sub_task_descriptions,
        "messages": [
            AIMessage(
                content=(
                    f"Research plan: {plan.research_strategy}\n"
                    + "\n".join(
                        f"{i + 1}. {task.description} — {task.rationale}"
                        for i, task in enumerate(plan.sub_tasks)
                    )
                )
            )
        ],
    }


async def web_researcher_node(state: AgentState) -> dict:
    """Phase 1 stub — real web research in Phase 3."""
    return {"web_findings": []}


async def doc_analyst_node(state: AgentState) -> dict:
    """Phase 1 stub — real RAG search in Phase 2/3."""
    return {"doc_findings": []}


async def report_writer_node(state: AgentState) -> dict:
    """Phase 1 stub — real synthesis in Phase 3."""
    return {
        "synthesis": "Phase 1 stub — report writer not yet implemented.",
        "report": None,
    }
