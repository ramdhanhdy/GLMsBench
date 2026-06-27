from __future__ import annotations
import re
from ..sandbox.runner import run_code_against_tests


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(output: str) -> str:
    """Extract the first fenced code block, or fall back to whole output."""
    m = _CODE_BLOCK_RE.search(output)
    if m:
        return m.group(1).strip()
    return output.strip()


def score_code(output: str, test_code: str, entry_point: str, timeout_s: float = 10.0) -> dict:
    code = extract_code(output)
    return run_code_against_tests(code, test_code, entry_point, timeout_s)
