from openai import OpenAI, BadRequestError
from dotenv import load_dotenv
from uuid import uuid4
import os
import time
import logging

from .base import SYSTEM_PROMPTS, BaseLLMClient

load_dotenv()
logger = logging.getLogger(__name__)

# Default per-request timeout in seconds. Callers can override via the
# `request_timeout` constructor argument.  main.py benefits from the same
# default; the performance test suite passes a tighter value so no single
# stuck request can block the whole run.
_DEFAULT_TIMEOUT_S: float = 30.0


class OpenAIClient(BaseLLMClient):
    def __init__(self, endpoint, key, model_name, request_timeout: float = _DEFAULT_TIMEOUT_S):
        self.model_name = model_name
        self._request_timeout = request_timeout
        self.client = OpenAI(
            base_url=endpoint,
            api_key=key
        )

    def build_messages(self, prompt, few_shot_example=None):
        system_message = SYSTEM_PROMPTS.get(prompt.task_type.upper(), "You are a helpful assistant.")

        if prompt.task_type.upper() == "SUMMARISATION":
            messages = [{"role": "system", "content": system_message}]
            if few_shot_example:
                messages.append({"role": "user", "content": f"Article:\n{few_shot_example.article}"})
                messages.append({"role": "assistant", "content": few_shot_example.highlights})
            messages.append({"role": "user", "content": f"Article:\n{prompt.article}"})
            return messages
        else:
            user_content = (
                f"Context:\n{' '.join(prompt.contexts) if isinstance(prompt.contexts, list) else prompt.contexts}\n\nQuestion:\n{prompt.input_text}"
                if prompt.contexts
                else prompt.input_text
            )
            return [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content},
            ]

    def generate(self, prompt, few_shot_example=None):
        self.prompt = prompt

        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                messages=self.build_messages(prompt, few_shot_example),
                temperature=0.6,
                model=self.model_name,
                timeout=self._request_timeout,
            )
            end_time = time.time()

            return {
                'response_id': str(uuid4()),
                'llm_response': response.choices[0].message.content,
                'prompt_id': prompt.id,
                'tokens_generated': response.usage.completion_tokens,
                'tokens_prompt': response.usage.prompt_tokens,
                'total_tokens': response.usage.total_tokens,
                'model_name': response.model,
                'latency_ms': (end_time - start_time) * 1000
            }
        except BadRequestError as e:
            if e.code == 'content_filter' or 'content_filter' in str(e).lower():
                logger.warning(f"Prompt {prompt.id} filtered by content policy, skipping.")
                return {'response_id': str(uuid4()), 'llm_response': None, 'prompt_id': prompt.id,
                        'tokens_generated': 0, 'tokens_prompt': 0, 'total_tokens': 0,
                        'model_name': self.model_name, 'latency_ms': 0}
            raise
        except Exception as e:
            # Covers httpx.ReadTimeout, openai.APITimeoutError, and similar
            if 'timeout' in type(e).__name__.lower() or 'timeout' in str(e).lower():
                logger.warning(
                    "Request timed out for model=%s prompt=%s after %.0fs, skipping.",
                    self.model_name, prompt.id, self._request_timeout,
                )
                return {
                    'response_id': str(uuid4()),
                    'llm_response': None,
                    'prompt_id': prompt.id,
                    'tokens_generated': 0,
                    'tokens_prompt': 0,
                    'total_tokens': 0,
                    'model_name': self.model_name,
                    'latency_ms': self._request_timeout * 1000,
                }
            raise
