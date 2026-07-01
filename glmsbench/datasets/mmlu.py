from __future__ import annotations
from .base import DatasetItem

LETTERS = ["A", "B", "C", "D"]


def _load_hf(*args, **kwargs):
    """Indirection so tests can monkeypatch without touching the network."""
    from datasets import load_dataset
    return load_dataset(*args, **kwargs)


class MMLULoader:
    name = "mmlu"
    # GLM-5.2 reasons by default (see Generation.reasoning_effort); reasoning
    # tokens count against this cap before the visible answer letter.
    # Measured ~70 reasoning tokens for a trivial MCQ at reasoning_effort="low";
    # 300 gives headroom for harder questions.
    max_tokens = 300

    def load(self, n: int) -> list[DatasetItem]:
        ds = _load_hf("cais/mmlu", "all", split="test")
        items: list[DatasetItem] = []
        for i, row in enumerate(ds):
            if i >= n:
                break
            choices = row["choices"]
            letters = LETTERS[:len(choices)]
            body = f"{row['question']}\n" + "\n".join(
                f"{l}. {c}" for l, c in zip(letters, choices)
            )
            prompt = _format_prompt(body, letters)
            items.append(DatasetItem(
                id=f"mmlu-{i:04d}",
                prompt=prompt,
                gold=LETTERS[row["answer"]],
                meta={"subject": row.get("subject", "")},
            ))
        return items


def _format_prompt(body: str, letters: list[str]) -> str:
    # Zero-shot MCQ with explicit answer instruction.
    options = ", ".join(letters)
    return (
        f"Answer the following multiple-choice question. "
        f"Respond with only the letter ({options}).\n\n"
        f"{body}\n\nAnswer:"
    )
