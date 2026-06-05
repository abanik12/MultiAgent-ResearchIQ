import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config.settings import get_settings
from src.models.schemas import ResearchPlan, ResearchPlanResult
from src.utils.token_cost import extract_token_usage

COORDINATOR_SYSTEM_PROMPT = """You are a research planning coordinator for ResearchIQ.

Your job is to decompose a research query into 2-4 focused, non-overlapping sub-tasks.
Each sub-task must be actionable for either web research or document/knowledge-base analysis.

Rules:
- Produce exactly 2-4 sub-tasks (never 1, never more than 4)
- Each sub-task needs a clear description and rationale
- Sub-tasks should cover different angles (e.g. overview, comparison, recent developments, limitations)
- Do NOT answer the query — only plan how to research it
- Set original_query to the user's exact query
- Write a brief research_strategy explaining the overall approach
"""


def _configure_tracing(settings) -> None:
    if settings.langchain_tracing_v2 and settings.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project


def _build_llm(settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )


async def plan_research_with_usage(query: str) -> ResearchPlanResult:
    """Decompose a research query into structured sub-tasks with token usage."""
    settings = get_settings()
    _configure_tracing(settings)

    llm = _build_llm(settings)
    structured_llm = llm.with_structured_output(ResearchPlan, include_raw=True)

    messages = [
        SystemMessage(content=COORDINATOR_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    response = await structured_llm.ainvoke(messages)
    plan: ResearchPlan = response["parsed"].model_copy(update={"original_query": query})
    usage = extract_token_usage(response["raw"], settings.openai_model)

    return ResearchPlanResult(plan=plan, usage=usage)


async def plan_research(query: str) -> ResearchPlan:
    """Decompose a research query into structured sub-tasks."""
    result = await plan_research_with_usage(query)
    return result.plan
