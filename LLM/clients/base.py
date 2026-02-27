from abc import ABC, abstractmethod

SYSTEM_PROMPTS = {
    'QA': 'You are a question answering assistant. Answer the question directly and concisely using only the information provided. Do not add unnecessary detail.',
    'Reasoning': 'You are a logical reasoning assistant. Think through the problem step by step and provide a clear, structured answer.',
    'Summarisation': 'You are a summarisation assistant. Provide a concise summary that captures the key points. Do not add information beyond what is provided.',
}

class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt):
        pass