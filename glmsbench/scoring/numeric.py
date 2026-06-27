from __future__ import annotations
import re
from typing import Optional


_HASH_RE = re.compile(r"####\s*([-+]?\d[\d,]*)")
_LAST_NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def extract_number(text: str) -> Optional[int]:
    """Extract the GSM8K-style answer: prefer #### N, else last number."""
    m = _HASH_RE.search(text)
    if m:
        return int(m.group(1).replace(",", ""))
    nums = _LAST_NUM_RE.findall(text)
    if nums:
        try:
            return int(float(nums[-1].replace(",", "")))
        except ValueError:
            return None
    return None


def score_numeric(gold: int, output: str) -> dict:
    parsed = extract_number(output)
    gold_i = int(gold)
    return {"correct": parsed is not None and parsed == gold_i, "parsed": parsed, "gold": gold_i}
