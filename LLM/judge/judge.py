import os
import re
import json
import logging
import time
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from openai import RateLimitError

from LLM.judge.pydantic_models import JudgeEvaluation

load_dotenv()
logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluation judge. "
    "You will be given an input, a context, and a model response. "
    "For each question, answer YES or NO. "
    "When you answer NO, you must also provide a brief explanation (1-2 sentences) citing the specific part of the response that caused the failure. "
    "When you answer YES, leave the explanation as an empty string. "
    "You MUST respond with a single valid JSON object. Do not include any text outside the JSON. "
    "Do not use newlines or special characters inside string values. "
    f"The JSON must match this schema exactly: {json.dumps(JudgeEvaluation.model_json_schema())}"
)


METRIC_QUESTIONS = {
    # Metric
    "Hallucination": [
        # Sub questions for every metric
        "Is every factual claim in the response supported by the provided context?",
        "Does the response avoid introducing names, dates, or figures not present in the context?",
        "Does the response avoid contradicting any information in the context?",
        "Does the response stay within the scope of the provided context?",
    ],
    "Fluency": [
        "Is the response free from grammatical errors?",
        "Does the response read naturally without awkward phrasing?",
        "Are all sentences complete and well-formed?",
        "Is the language clear and appropriate for the task?",
    ],
    "Consistency": [
        "Is the response internally consistent with no self-contradictions?",
        "Does the response maintain a consistent viewpoint or stance throughout?",
        "Does the response avoid repeating the same claim with conflicting details?",
        "Is the response consistent with the information provided in the context?",
    ],
    "Reasoning": [
        "Does the response directly address the question or task?",
        "Are the conclusions in the response logically supported by the reasoning given?",
        "Is the reasoning free from obvious logical fallacies?",
        "Does the response follow a clear logical thought process?",
    ],
    "Coherence": [
        "Does the response have a clear and logical structure?",
        "Do ideas flow logically from one to the next?",
        "Is the response focused and on-topic throughout?",
        "Does the response avoid unnecessary tangents or contradictions?",
    ],
    "Factual Accuracy": [
        "Are all named entities (people, places, organisations) accurately represented?",
        "Are numerical values such as dates, figures, and statistics correct relative to the context?",
        "Does the response avoid making claims beyond what the context supports?",
        "Are cause-and-effect relationships accurately stated?",
    ],
}

class LLMAsJudge:
    def __init__(self):
        self.client = AzureChatOpenAI(
            azure_deployment=os.getenv("JUDGE_MODEL"),
            azure_endpoint=os.getenv("JUDGE_ENDPOINT"),
            api_key=os.getenv("JUDGE_KEY"),
            api_version=os.getenv("JUDGE_API_V"),
            temperature=0
        )

    def build_message(self, prompt, llm_response):
        context = (
            "\n".join(prompt.contexts) if isinstance(prompt.contexts, list)
            else prompt.contexts or prompt.article or ""
        )

        if not context and prompt.reference_output:
            context_block = f"Reference Answer:\n{prompt.reference_output}"
        else:
            context_block = f"Context:\n{context}"

        questions_block = ""
        for metric, questions in METRIC_QUESTIONS.items():
            questions_block += f"\n{metric}:\n"
            for i, q in enumerate(questions, 1):
                questions_block += f"  {i}. {q}\n"

        human_content = (
            f"Input:\n{prompt.input_text}\n\n"
            f"{context_block}\n\n"
            f"Response:\n{llm_response}\n\n"
            f"For each metric below, answer every sub-question YES or NO.\n"
            f"{questions_block}"
        )

        return [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

    def evaluate(self, prompt, llm_response, max_retries=5):
        messages = self.build_message(prompt, llm_response)

        for attempt in range(max_retries):
            try:
                response = self.client.invoke(messages)
                break
            except RateLimitError:
                wait = 2 ** attempt
                logger.warning(f"Judge rate limited on prompt {prompt.id}, retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Judge failed after {max_retries} retries due to rate limiting on prompt {prompt.id}")

        raw_text = response.content
        # Strip control characters that make JSON invalid (keep \t, \n, \r)
        clean = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', raw_text)
        # Extract JSON object in case there is any surrounding text
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in judge response for prompt {prompt.id}")
        result: JudgeEvaluation = JudgeEvaluation.model_validate_json(match.group())

        return result
