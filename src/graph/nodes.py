from langchain_core.messages import AIMessage

from src.agents.coordinator import plan_research
from src.agents.doc_analyst import analyze_documents
from src.agents.report_writer import write_report
from src.agents.web_researcher import research_web
from src.graph.state import AgentState, WorkerState


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
                        f"{index + 1}. {task.description} — {task.rationale}"
                        for index, task in enumerate(plan.sub_tasks)
                    )
                )
            )
        ],
    }


async def web_researcher_node(state: WorkerState) -> dict:
    """Search the live web for a single sub-task."""
    findings = await research_web(state["task"])
    return {"web_findings": findings}


async def doc_analyst_node(state: WorkerState) -> dict:
    """Search the curated knowledge base for a single sub-task."""
    findings = await analyze_documents(state["task"])
    return {"doc_findings": findings}


async def report_writer_node(state: AgentState) -> dict:
    """Synthesize accumulated findings into a structured report."""
    report, synthesis = await write_report(
        query=state["query"],
        sub_tasks=state["sub_tasks"],
        web_findings=state["web_findings"],
        doc_findings=state["doc_findings"],
    )
    return {"report": report, "synthesis": synthesis}
