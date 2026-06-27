from __future__ import annotations
from typing import Callable, Optional


def compare_text_parity(
    gold,
    output_a: str,
    output_b: str,
    extract_fn: Callable[[str], Optional[str]],
) -> dict:
    """Compare two providers' outputs on canonical answer + raw text.

    extract_fn maps a raw output to its canonical form (e.g. extract_letter).
    """
    canon_a = extract_fn(output_a)
    canon_b = extract_fn(output_b)
    return {
        "answer_agree": canon_a is not None and canon_a == canon_b,
        "raw_text_agree": output_a == output_b,
        "canonical_a": canon_a,
        "canonical_b": canon_b,
        "gold": gold,
    }


def aggregate_parity(results: list[dict]) -> dict:
    """Compute disagreement rate from a list of per-item parity results."""
    n = len(results)
    if n == 0:
        return {"n": 0, "disagreement_rate": None, "raw_text_agreement_rate": None}
    n_disagree = sum(1 for r in results if not r["answer_agree"])
    n_raw = sum(1 for r in results if r["raw_text_agree"])
    return {
        "n": n,
        "disagreement_rate": n_disagree / n,
        "raw_text_agreement_rate": n_raw / n,
    }
