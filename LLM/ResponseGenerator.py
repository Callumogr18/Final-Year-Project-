import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import logging
import time
from uuid import uuid4

load_dotenv()

logger = logging.getLogger(__name__)

class ResponseGenerator:
    def __init__(self, prompt, prompt_id):
        self.prompt = prompt
        self.prompt_id = prompt_id

        self.generation_data = None
        if prompt != None and prompt_id != None:
            self.generation_data = self._azure_caller()

    def _azure_caller(self):
        logger.info(f"Creating Azure client for LLM response generation")

        client = AzureOpenAI(
            api_version=os.getenv("AZURE_API_V"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_key=os.getenv("AZURE_API_KEY"),
        )

        logger.info("Generating response...")

        start_time = time.time()
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": self.prompt.input_text,
                }
            ],
            temperature=0.5,
            model=os.getenv("AZURE_MODEL")
        )
        end_time = time.time()
        
        generation_data = {
            'response_id' : str(uuid4()),
            'llm_response' : response.choices[0].message.content,
            'prompt_id' : self.prompt_id,
            'tokens_generated' : response.usage.completion_tokens,
            'tokens_prompt' : response.usage.prompt_tokens,
            'total_tokens' : response.usage.total_tokens,
            'model_name' : response.model,
            'latency_ms' : (end_time - start_time) * 1000
        }

        return generation_data