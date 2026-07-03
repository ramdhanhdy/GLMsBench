from __future__ import annotations
from .base import DatasetItem


def _load_hf(*args, **kwargs):
    from datasets import load_dataset
    return load_dataset(*args, **kwargs)


class ARCLoader:
    name = "arc"
    # GLM-5.2 can spend 300-1000 hidden reasoning tokens on hard ARC items
    # before emitting the visible answer. ARC uses 1200 so strict success is
    # not dominated by artificial length stops.
    max_tokens = 1200

    def load(self, n: int) -> list[DatasetItem]:
        ds = _load_hf("allenai/ai2_arc", "ARC-Challenge", split="test")
        items: list[DatasetItem] = []
        for i, row in enumerate(ds):
            if i >= n:
                break
            choices = row["choices"]
            labels = list(choices["label"])
            texts = list(choices["text"])
            prompt = _format_prompt(row["question"], labels, texts)
            items.append(DatasetItem(
                id=f"arc-{i:04d}",
                prompt=prompt,
                gold=row["answerKey"],
                meta={},
            ))
        return items


def _format_prompt(question: str, labels: list[str], texts: list[str]) -> str:
    body = f"{question}\n" + "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
    options = ", ".join(labels)
    return (
        f"Answer the following multiple-choice question. "
        f"Respond with only the letter ({options}).\n\n"
        f"{body}\n\nAnswer:"
    )
