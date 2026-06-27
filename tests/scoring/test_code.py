from glmsbench.scoring.code import extract_code, score_code


def test_extract_code_from_markdown_block():
    output = "Here is the code:\n```python\ndef f():\n    return 1\n```\nDone."
    code = extract_code(output)
    assert "def f():" in code
    assert "return 1" in code
    assert "```" not in code


def test_extract_code_plain():
    assert "def f():" in extract_code("def f():\n    return 1\n")


def test_score_code_passes():
    output = "```python\ndef f():\n    return 1\n```"
    test_code = "def check(c):\n    assert c() == 1\n"
    r = score_code(output, test_code, entry_point="f")
    assert r["passed"] is True


def test_score_code_fails():
    output = "```python\ndef f():\n    return 2\n```"
    test_code = "def check(c):\n    assert c() == 1\n"
    r = score_code(output, test_code, entry_point="f")
    assert r["passed"] is False
