"""
Latency is measured inside OpenAIClient.generate as (end - start) * 1000.
We simulate a known-duration API call by patching the underlying chat client
to sleep for a fixed amount, then assert latency_ms reflects that delay.
"""
import time
import pytest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from LLM.clients.openai_client import OpenAIClient
from DB.prompts.Prompt import Prompt

MODELS = ['gpt-4.1-mini', 'phi-4', 'grok-4-fast', 'kimi-k2.5', 'llama']


def _fake_completion(model_name, delay_s, prompt_tokens=10, completion_tokens=5):
    """
    Returns a callable that sleeps then returns an OpenAI-shaped response
    """
    def _create(**kwargs):
        time.sleep(delay_s)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='stub response'))],
            usage=SimpleNamespace(
                completion_tokens=completion_tokens,
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            model=model_name,
        )
    return _create


def _make_client(model_name, delay_s):
    with patch('LLM.clients.openai_client.OpenAI') as mock_openai:
        instance = MagicMock()
        instance.chat.completions.create.side_effect = _fake_completion(model_name, delay_s)
        mock_openai.return_value = instance
        return OpenAIClient(endpoint='http://stub', key='stub', model_name=model_name)


def _prompt():
    return Prompt(
        id=1,
        task_type='QA',
        input_text='What is the capital of France?',
        reference_output='Paris',
        answer='Paris',
        contexts=['Paris is the capital of France.'],
    )


@pytest.mark.parametrize('model', MODELS)
def test_latency_is_measured_per_model(model):
    """
    latency_ms must be present, non-negative, and roughly match the simulated delay
    """
    client = _make_client(model, delay_s=0.05)
    result = client.generate(_prompt())

    assert 'latency_ms' in result
    assert isinstance(result['latency_ms'], float)
    assert result['latency_ms'] >= 0.0
    # 50 ms simulated; allow generous upper bound for CI jitter.
    assert 40.0 <= result['latency_ms'] < 2000.0, f'{model} latency {result["latency_ms"]} ms'


def test_longer_call_has_higher_latency():
    """
    A longer simulated call must produce a larger latency_ms value
    """
    fast = _make_client('gpt-4.1-mini', delay_s=0.02).generate(_prompt())
    slow = _make_client('gpt-4.1-mini', delay_s=0.15).generate(_prompt())
    assert slow['latency_ms'] > fast['latency_ms']


def test_latency_reported_in_milliseconds():
    """
    A 100 ms call must report latency_ms in the hundreds, not seconds
    """
    result = _make_client('gpt-4.1-mini', delay_s=0.1).generate(_prompt())
    assert 80.0 <= result['latency_ms'] <= 1000.0
