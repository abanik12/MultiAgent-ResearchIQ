import pytest

from src.models.schemas import TokenUsage
from src.utils.token_cost import calculate_cost, extract_token_usage, format_usage_summary


class FakeRawMessage:
    def __init__(self, usage_metadata=None, response_metadata=None):
        self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata or {}


def test_extract_token_usage_from_usage_metadata():
    raw = FakeRawMessage(usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150})
    usage = extract_token_usage(raw, "gpt-5.4-mini")
    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    assert usage.total_tokens == 150


def test_extract_token_usage_fallback_to_response_metadata():
    raw = FakeRawMessage(
        response_metadata={
            "token_usage": {
                "prompt_tokens": 200,
                "completion_tokens": 80,
                "total_tokens": 280,
            }
        }
    )
    usage = extract_token_usage(raw, "gpt-5.4-mini")
    assert usage.input_tokens == 200
    assert usage.output_tokens == 80
    assert usage.total_tokens == 280


def test_calculate_cost_uses_default_pricing():
    usage = TokenUsage(model="gpt-5.4-mini", input_tokens=1_000_000, output_tokens=1_000_000)
    priced = calculate_cost(usage)
    assert priced.input_cost_usd == pytest.approx(0.15)
    assert priced.output_cost_usd == pytest.approx(0.60)
    assert priced.total_cost_usd == pytest.approx(0.75)


def test_format_usage_summary_includes_token_breakdown():
    usage = TokenUsage(
        model="gpt-5.4-mini",
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        input_cost_usd=0.000015,
        output_cost_usd=0.000030,
        total_cost_usd=0.000045,
    )
    summary = format_usage_summary(usage)
    assert "Input tokens:  100" in summary
    assert "Output tokens: 50" in summary
    assert "Est. total cost:" in summary
