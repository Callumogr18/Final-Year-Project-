import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import logging
import time
from uuid import uuid4

load_dotenv()

logger = logging.getLogger(__name__)

SYSTEM_PROMPTS = {
    'QA': 'You are a question answering assistant. Answer the question directly and concisely using only the information provided. Do not add unnecessary detail.',
    'Reasoning': 'You are a logical reasoning assistant. Think through the problem step by step and provide a clear, structured answer.',
    'Summarisation': 'You are a summarisation assistant. Provide a concise summary that captures the key points. Do not add information beyond what is provided.',
}

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
                    "content": SYSTEM_PROMPTS.get(self.prompt.task_type, "You are a helpful assistant."),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{self.prompt.contexts}\n\nQuestion:\n{self.prompt.input_text}" if self.prompt.contexts else self.prompt.input_text,
       }
            ],
            temperature=0.2,
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