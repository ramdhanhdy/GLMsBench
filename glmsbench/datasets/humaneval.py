from __future__ import annotations
from .base import DatasetItem


def _load_hf(*args, **kwargs):
    from datasets import load_dataset
    return load_dataset(*args, **kwargs)


class HumanEvalLoader:
    name = "humaneval"
    # Budget covers hidden reasoning_content (low effort) + a full function body.
    max_tokens = 2048

    def load(self, n: int) -> list[DatasetItem]:
        ds = _load_hf("openai/openai_humaneval", split="test")
        items: list[DatasetItem] = []
        for i, row in enumerate(ds):
            if i >= n:
                break
            items.append(DatasetItem(
                id=row["task_id"],
                prompt=_format_prompt(row),
                gold=[row["test"]],
                meta={
                    "entry_point": row["entry_point"],
                    "canonical_solution": row.get("canonical_solution", ""),
                },
            ))
        return items


def _format_prompt(row: dict) -> str:
    # Zero-shot: signature + docstring verbatim, ask to complete the body.
    return (
        f"Complete the following Python function. Respond with only the code, "
        f"no explanation.\n\n"
        f"{row['prompt']}"
    )
