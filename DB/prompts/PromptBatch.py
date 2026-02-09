from dataclasses import dataclass, field
from typing import List
from uuid import UUID, uuid4
from .Prompt import Prompt


@dataclass
class PromptBatch:
    prompts: List[Prompt]
    task_type: str
    batch_id: UUID = field(default_factory=uuid4)

    def get_task_type(self) -> str:
        return self.task_type

    def get_prompt_ids(self) -> List[int]:
        return [prompt.id for prompt in self.prompts]

    def size(self) -> int:
        return len(self.prompts)


class PromptBatcher:
    def __init__(self, default_batch_size: int = 10):
        self._default_batch_size = default_batch_size

    def create_batches(self, prompts) -> List[PromptBatch]:
        batch_size = self._default_batch_size
        batches = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            if batch_prompts:
                task_type = batch_prompts[0].task_type
                batches.append(PromptBatch(prompts=batch_prompts, task_type=task_type))

        return batches
