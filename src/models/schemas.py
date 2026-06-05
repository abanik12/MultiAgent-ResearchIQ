from pydantic import BaseModel, Field


class SubTask(BaseModel):
    id: int
    description: str
    rationale: str


class ResearchPlan(BaseModel):
    """Structured output from the coordinator agent."""

    original_query: str
    sub_tasks: list[SubTask] = Field(min_length=2, max_length=4)
    research_strategy: str


class TokenUsage(BaseModel):
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    total_cost_usd: float = 0.0


class ResearchPlanResult(BaseModel):
    """Coordinator output plus token/cost metadata."""

    plan: ResearchPlan
    usage: TokenUsage


class SearchResult(BaseModel):
    """Web research finding (Phase 2+)."""

    title: str = ""
    url: str = ""
    snippet: str = ""
    source: str = ""


class DocumentChunk(BaseModel):
    """Knowledge base chunk (Phase 2+)."""

    text: str = ""
    source: str = ""
    score: float = 0.0


class ResearchReport(BaseModel):
    """Final synthesized report (Phase 3+)."""

    title: str = ""
    summary: str = ""
    sections: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
