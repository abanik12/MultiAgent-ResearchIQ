import pytest
from pydantic import ValidationError

from src.models.schemas import ResearchPlan, SubTask


def test_research_plan_validates_sub_task_count():
    tasks = [
        SubTask(id=1, description="Task one", rationale="Why one"),
        SubTask(id=2, description="Task two", rationale="Why two"),
    ]
    plan = ResearchPlan(
        original_query="test query",
        sub_tasks=tasks,
        research_strategy="Cover two angles",
    )
    assert len(plan.sub_tasks) == 2


def test_research_plan_rejects_too_few_tasks():
    with pytest.raises(ValidationError):
        ResearchPlan(
            original_query="test",
            sub_tasks=[SubTask(id=1, description="Only one", rationale="Why")],
            research_strategy="Too narrow",
        )


def test_research_plan_rejects_too_many_tasks():
    tasks = [
        SubTask(id=i, description=f"Task {i}", rationale=f"Why {i}")
        for i in range(1, 6)
    ]
    with pytest.raises(ValidationError):
        ResearchPlan(
            original_query="test",
            sub_tasks=tasks,
            research_strategy="Too broad",
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_plan_research_integration():
    from src.agents.coordinator import plan_research
    from src.config.settings import get_settings

    settings = get_settings()
    if not settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY not set")

    plan = await plan_research("What are the latest advances in agentic RAG?")
    assert 2 <= len(plan.sub_tasks) <= 4
    assert plan.original_query == "What are the latest advances in agentic RAG?"
