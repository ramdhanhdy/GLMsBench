from glmsbench.sandbox.runner import run_code_against_tests


def test_passing_code():
    code = "def f():\n    return 1\n"
    test_code = "def check(c):\n    assert c() == 1\n"
    r = run_code_against_tests(code, test_code, entry_point="f", timeout_s=5)
    assert r["passed"] is True
    assert r["n_tests_passed"] == 1
    assert r["n_tests"] == 1


def test_failing_code():
    code = "def f():\n    return 2\n"
    test_code = "def check(c):\n    assert c() == 1\n"
    r = run_code_against_tests(code, test_code, entry_point="f", timeout_s=5)
    assert r["passed"] is False
    assert r["n_tests_passed"] == 0


def test_syntax_error():
    code = "def f(\n"
    test_code = "def check(c):\n    assert c() == 1\n"
    r = run_code_against_tests(code, test_code, entry_point="f", timeout_s=5)
    assert r["passed"] is False
    assert r["error"] is not None


def test_timeout():
    code = "def f():\n    while True: pass\n"
    test_code = "def check(c):\n    c()\n"
    r = run_code_against_tests(code, test_code, entry_point="f", timeout_s=2)
    assert r["passed"] is False
    assert "timeout" in (r["error"] or "").lower() or r["error"] is not None
