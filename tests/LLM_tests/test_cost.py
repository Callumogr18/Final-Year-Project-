"""
Cost tests - testing.md Performance Metrics: Latency and Cost.

Cost per call is derived from token usage (prompt + completion) and the
published per-1M-token pricing. These tests verify the cost calculation is:
  - correct for known token counts
  - non-negative
  - proportional to token usage (more tokens => higher cost)
"""
import pytest

# SOURCE: https://azure.microsoft.com/en-us/pricing/details/ai-foundry-models/aoai/
# Prices are USD per 1,000,000 tokens.
PRICING = {
    'gpt-4.1-mini': {'input': 0.40,  'output': 1.60},
    'phi-4':        {'input': 0.125, 'output': 0.50},
    'grok-4-fast':  {'input': 0.20,  'output': 0.50},
    'kimi-k2.5':    {'input': 0.60,  'output': 3.00},
    # llama: no pricing available - intentionally omitted
}

PRICED_MODELS = list(PRICING.keys())


def compute_cost(model, prompt_tokens, completion_tokens):
    """USD cost for a single call given token usage and per-1M pricing."""
    p = PRICING[model]
    return (prompt_tokens * p['input'] + completion_tokens * p['output']) / 1_000_000


@pytest.mark.parametrize('model', PRICED_MODELS)
def test_cost_non_negative(model):
    """Cost for any non-negative usage must be non-negative."""
    assert compute_cost(model, 0, 0) == 0.0
    assert compute_cost(model, 100, 50) > 0.0


def test_known_cost_gpt_4_1_mini():
    """1M input + 1M output of gpt-4.1-mini = $0.40 + $1.60 = $2.00."""
    cost = compute_cost('gpt-4.1-mini', 1_000_000, 1_000_000)
    assert cost == pytest.approx(2.00)


def test_known_cost_phi_4():
    """1M input + 1M output of phi-4 = $0.125 + $0.50 = $0.625."""
    cost = compute_cost('phi-4', 1_000_000, 1_000_000)
    assert cost == pytest.approx(0.625)


@pytest.mark.parametrize('model', PRICED_MODELS)
def test_cost_scales_with_token_usage(model):
    """Doubling token usage must double the cost."""
    small = compute_cost(model, 100, 50)
    large = compute_cost(model, 200, 100)
    assert large == pytest.approx(2 * small)


@pytest.mark.parametrize('model', PRICED_MODELS)
def test_output_tokens_at_least_as_expensive_as_input(model):
    """Per published pricing, output tokens are never cheaper than input tokens."""
    assert PRICING[model]['output'] >= PRICING[model]['input']


def test_llama_has_no_pricing():
    """llama is intentionally excluded from PRICING - testing.md cost comparison skips it."""
    assert 'llama' not in PRICING
