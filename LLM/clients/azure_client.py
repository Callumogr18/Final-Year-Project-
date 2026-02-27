from openai import AzureOpenAI
from dotenv import load_dotenv
from uuid import uuid4
import os
import time
import logging

from .base import SYSTEM_PROMPTS, BaseLLMClient

load_dotenv()
logger = logging.getLogger(__name__)

class AzureClient(BaseLLMClient):
    def __init__(self):
        self.client = AzureOpenAI(
            api_version=os.getenv("AZURE_API_V"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_key=os.getenv("AZURE_API_KEY")
        )

    def build_messages(self, prompt):
        system_message = SYSTEM_PROMPTS.get(prompt.task_type, "You are a helpful assistant.")

        if prompt.task_type.upper() == "SUMMARISATION":
            highlights_section = f"\n\nKey highlights to focus on:\n{prompt.highlights}" if prompt.highlights else ""
            user_content = f"Article:\n{prompt.article}{highlights_section}"
        else:
            user_content = (
                f"Context:\n{prompt.contexts}\n\nQuestion:\n{prompt.input_text}"
                if prompt.contexts
                else prompt.input_text
            )

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ]

    def generate(self, prompt):
        self.prompt = prompt
        logger.info(f"Generating response for task type: {prompt.task_type}")

        start_time = time.time()
        response = self.client.chat.completions.create(
            messages=self.build_messages(prompt),
            temperature=0.2,
            model=os.getenv("AZURE_MODEL")
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