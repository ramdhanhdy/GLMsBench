def test_humaneval_loader_extracts_tests(monkeypatch):
    from glmsbench.datasets import humaneval as mod

    fake_rows = [
        {
            "task_id": "HumanEval/0",
            "prompt": "def has_close_elements(x):\n    pass\n",
            "canonical_solution": "    return True",
            "test": "def check(candidate):\n    assert candidate([1]) is True\n",
            "entry_point": "has_close_elements",
        }
    ]

    def fake_load_dataset(*a, **k):
        class _DS:
            def __init__(self, rows): self._rows = rows
            def __iter__(self): return iter(self._rows)
        return _DS(fake_rows)

    monkeypatch.setattr(mod, "_load_hf", fake_load_dataset)
    items = mod.HumanEvalLoader().load(1)
    assert items[0].id == "HumanEval/0"
    assert isinstance(items[0].gold, list)
    assert "def check(candidate)" in items[0].gold[0]
    assert items[0].meta["entry_point"] == "has_close_elements"


def test_humaneval_prompt_is_function_plus_docstring(monkeypatch):
    from glmsbench.datasets import humaneval as mod

    fake_rows = [{
        "task_id": "HumanEval/1", "prompt": "def f():\n    pass\n",
        "canonical_solution": "    return 1", "test": "def check(c): assert c() == 1\n",
        "entry_point": "f",
    }]

    def fake_load_dataset(*a, **k):
        class _DS:
            def __init__(self, rows): self._rows = rows
            def __iter__(self): return iter(self._rows)
        return _DS(fake_rows)

    monkeypatch.setattr(mod, "_load_hf", fake_load_dataset)
    items = mod.HumanEvalLoader().load(1)
    # the loader should preserve the prompt (signature + docstring) verbatim
    assert "def f():" in items[0].prompt
