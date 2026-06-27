from __future__ import annotations
import re
from typing import Optional


_LETTER_RE = re.compile(r"\b([A-D])\b")


def extract_letter(text: str) -> Optional[str]:
    """Extract the first standalone A-D letter from model output."""
    m = _LETTER_RE.search(text)
    return m.group(1) if m else None


def score_letter(gold: str, output: str) -> dict:
    parsed = extract_letter(output)
    return {"correct": parsed == gold, "parsed": parsed, "gold": gold}
