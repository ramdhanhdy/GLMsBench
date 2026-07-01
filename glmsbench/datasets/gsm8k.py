from __future__ import annotations
import re
from .base import DatasetItem


def _load_hf(*args, **kwargs):
    from datasets import load_dataset
    return load_dataset(*args, **kwargs)


class GSM8KLoader:
    name = "gsm8k"
    # Budget covers hidden reasoning_content (low effort) + the visible
    # step-by-step solution + "#### N" line.
    max_tokens = 768

    def load(self, n: int) -> list[DatasetItem]:
        ds = _load_hf("openai/gsm8k", "main", split="test")
        items: list[DatasetItem] = []
        for i, row in enumerate(ds):
            if i >= n:
                break
            answer_text = row["answer"]
            m = re.search(r"####\s*([-+]?\d+)", answer_text)
            gold = int(m.group(1)) if m else None
            prompt = _format_prompt(row["question"])
            items.append(DatasetItem(
                id=f"gsm8k-{i:04d}",
                prompt=prompt,
                gold=gold,
                meta={},
            ))
        return items


def _format_prompt(question: str) -> str:
    return (
        f"Solve the math problem. End your response with a line of the form "
        f"'#### <integer answer>'.\n\n"
        f"Question: {question}\n\nAnswer:"
    )
