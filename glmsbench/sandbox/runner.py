from __future__ import annotations
import subprocess
import sys


def run_code_against_tests(
    code: str, test_code: str, entry_point: str, timeout_s: float = 10.0
) -> dict:
    """Run generated code against HumanEval-style tests in a subprocess.

    The test harness calls ``check(entry_point_fn)``. We assemble both pieces
    into a single script (each at column 0 so embedded indentation survives),
    run it via ``python -c`` in a subprocess with a hard timeout.
    Returns {passed, n_tests_passed, n_tests, error}.
    """
    # Assemble without textwrap.dedent: dedent would strip the internal
    # indentation of the embedded code/test bodies and break them.
    full = (
        f"{code}\n\n"
        f"{test_code}\n\n"
        f"try:\n"
        f"    fn = {entry_point}\n"
        f"    check(fn)\n"
        f"    print('__PASS__')\n"
        f"except Exception as e:\n"
        f"    print('__FAIL__:' + repr(e))\n"
    )

    try:
        proc = subprocess.run(
            [sys.executable, "-c", full],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {"passed": False, "n_tests_passed": 0, "n_tests": 1, "error": "timeout"}
    out = proc.stdout
    if "__PASS__" in out:
        return {"passed": True, "n_tests_passed": 1, "n_tests": 1, "error": None}
    err = proc.stderr.strip().splitlines()
    return {
        "passed": False,
        "n_tests_passed": 0,
        "n_tests": 1,
        "error": (out + "\n" + "\n".join(err[-3:])).strip() or "unknown error",
    }
