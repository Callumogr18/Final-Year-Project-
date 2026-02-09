from dataclasses import dataclass

@dataclass
class Prompt:
    id: int
    task_type: str
    input_text: str
    reference_output: str
    answer: str
    contexts: str

    def to_dict(self):
        return {
            "id": self.id,
            "task_type": self.task_type,
            "input_text": self.input_text,
            "reference_output": self.reference_output,
            "answer": self.answer,
            "contexts": self.contexts
        }

    def validate(self) -> bool:
        if not self.task_type or not isinstance(self.task_type, str):
            return False
        if not self.input_text or not isinstance(self.input_text, str):
            return False
        return True

    def get_word_count(self) -> int:
        return len(self.input_text.split())
