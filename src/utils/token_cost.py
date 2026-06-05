"""Token usage extraction and cost estimation for LLM calls."""

from src.config.settings import Settings, get_settings
from src.models.schemas import TokenUsage

# USD per 1M tokens — override via OPENAI_INPUT_PRICE_PER_1M / OPENAI_OUTPUT_PRICE_PER_1M
DEFAULT_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5.4-mini": (0.15, 0.60),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


def extract_token_usage(raw_message, model: str) -> TokenUsage:
    """Pull token counts from a LangChain AIMessage response."""
    usage_metadata = getattr(raw_message, "usage_metadata", None) or {}

    input_tokens = int(usage_metadata.get("input_tokens", 0) or 0)
    output_tokens = int(usage_metadata.get("output_tokens", 0) or 0)
    total_tokens = int(
        usage_metadata.get("total_tokens", input_tokens + output_tokens) or 0
    )

    if total_tokens == 0:
        token_usage = (getattr(raw_message, "response_metadata", None) or {}).get(
            "token_usage", {}
        )
        input_tokens = int(token_usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(token_usage.get("completion_tokens", 0) or 0)
        total_tokens = int(token_usage.get("total_tokens", input_tokens + output_tokens) or 0)

    usage = TokenUsage(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )
    return calculate_cost(usage)


def calculate_cost(usage: TokenUsage, settings: Settings | None = None) -> TokenUsage:
    """Estimate USD cost from token counts and model pricing."""
    settings = settings or get_settings()

    input_rate, output_rate = _resolve_pricing(usage.model, settings)
    input_cost = (usage.input_tokens / 1_000_000) * input_rate
    output_cost = (usage.output_tokens / 1_000_000) * output_rate

    return usage.model_copy(
        update={
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "total_cost_usd": input_cost + output_cost,
        }
    )


def _resolve_pricing(model: str, settings: Settings) -> tuple[float, float]:
    if settings.openai_input_price_per_1m is not None and settings.openai_output_price_per_1m is not None:
        return settings.openai_input_price_per_1m, settings.openai_output_price_per_1m
    return DEFAULT_MODEL_PRICING.get(model, DEFAULT_MODEL_PRICING["gpt-4o-mini"])


def format_usage_summary(usage: TokenUsage) -> str:
    """Human-readable summary for CLI output."""
    return (
        f"\n--- Token & cost summary ({usage.model}) ---\n"
        f"Input tokens:  {usage.input_tokens:,}\n"
        f"Output tokens: {usage.output_tokens:,}\n"
        f"Total tokens:  {usage.total_tokens:,}\n"
        f"Est. input cost:  ${usage.input_cost_usd:.6f}\n"
        f"Est. output cost: ${usage.output_cost_usd:.6f}\n"
        f"Est. total cost:  ${usage.total_cost_usd:.6f}\n"
        f"(Costs are estimates — verify against OpenAI/LangSmith billing.)"
    )
