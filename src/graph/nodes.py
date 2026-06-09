from langchain_core.messages import AIMessage

from src.agents.coordinator import plan_research
from src.agents.doc_analyst import analyze_documents
from src.agents.report_writer import write_report
from src.agents.web_researcher import research_web
from src.graph.state import AgentState, WorkerState
from src.graph.trace_context import emit_trace_event


async def coordinator_node(state: AgentState) -> dict:
    """Decompose the query into sub-tasks via structured LLM output."""
    emit_trace_event(
        "phase",
        phase="planning",
        message="Planning research",
    )
    emit_trace_event(
        "agent_start",
        agent="coordinator",
        message="Decomposing query into sub-tasks",
    )

    plan = await plan_research(state["query"])
    sub_task_descriptions = [task.description for task in plan.sub_tasks]

    emit_trace_event(
        "plan",
        sub_tasks=sub_task_descriptions,
        strategy=plan.research_strategy,
        message=f"Plan ready — {len(sub_task_descriptions)} sub-tasks",
    )
    emit_trace_event(
        "agent_end",
        agent="coordinator",
        message=f"Plan ready — {len(sub_task_descriptions)} sub-tasks",
    )
    emit_trace_event(
        "phase",
        phase="research",
        message="Running parallel web and document research",
    )

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
    task_id = state["task_id"]
    emit_trace_event(
        "agent_start",
        agent="web_researcher",
        task_id=task_id,
        task=state["task"],
        message=f"Searching the web (task {task_id + 1})",
    )

    findings, trace_steps = await research_web(state["task"], task_id=task_id)
    for step in trace_steps:
        emit_trace_event(**step)

    emit_trace_event(
        "agent_end",
        agent="web_researcher",
        task_id=task_id,
        web_findings_count=len(findings),
        message=f"Web research complete — {len(findings)} source(s)",
    )
    return {"web_findings": findings}


async def doc_analyst_node(state: WorkerState) -> dict:
    """Search the curated knowledge base for a single sub-task."""
    task_id = state["task_id"]
    emit_trace_event(
        "agent_start",
        agent="doc_analyst",
        task_id=task_id,
        task=state["task"],
        message=f"Searching knowledge base (task {task_id + 1})",
    )

    findings, trace_steps = await analyze_documents(state["task"], task_id=task_id)
    for step in trace_steps:
        emit_trace_event(**step)

    emit_trace_event(
        "agent_end",
        agent="doc_analyst",
        task_id=task_id,
        doc_findings_count=len(findings),
        message=f"Document analysis complete — {len(findings)} chunk(s)",
    )
    return {"doc_findings": findings}


async def report_writer_node(state: AgentState) -> dict:
    """Synthesize accumulated findings into a structured report."""
    emit_trace_event(
        "phase",
        phase="synthesis",
        message="Synthesizing final report",
    )
    emit_trace_event(
        "agent_start",
        agent="report_writer",
        message="Writing structured report",
    )

    report, synthesis = await write_report(
        query=state["query"],
        sub_tasks=state["sub_tasks"],
        web_findings=state["web_findings"],
        doc_findings=state["doc_findings"],
    )

    emit_trace_event(
        "agent_end",
        agent="report_writer",
        message="Report draft complete",
    )
    return {"report": report, "synthesis": synthesis}
