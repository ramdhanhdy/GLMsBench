import pytest


def test_mmlu_loader_formats_prompt(monkeypatch):
    from glmsbench.datasets import mmlu as mmlu_mod
    from glmsbench.datasets.base import DatasetItem

    fake_rows = [
        {"question": "What is 2+2?", "subject": "math", "choices": ["1", "2", "3", "4"], "answer": 3},
        {"question": "Capital of France?", "subject": "geography", "choices": ["London", "Paris", "Rome", "Berlin"], "answer": 1},
    ]

    def fake_load_dataset(*a, **k):
        class _DS:
            def __init__(self, rows):
                self._rows = rows
            def __iter__(self):
                return iter(self._rows)
        return _DS(fake_rows)

    monkeypatch.setattr(mmlu_mod, "_load_hf", fake_load_dataset)

    loader = mmlu_mod.MMLULoader()
    items = loader.load(2)
    assert len(items) == 2
    assert items[0].gold == "D"  # answer index 3 -> D
    assert "What is 2+2?" in items[0].prompt
    assert "A. 1" in items[0].prompt
    assert items[1].gold == "B"


def test_mmlu_caps_n(monkeypatch):
    from glmsbench.datasets import mmlu as mmlu_mod

    fake_rows = [{"question": f"Q{i}", "subject": "s", "choices": ["a", "b", "c", "d"], "answer": 0} for i in range(10)]

    def fake_load_dataset(*a, **k):
        class _DS:
            def __init__(self, rows):
                self._rows = rows
            def __iter__(self):
                return iter(self._rows)
        return _DS(fake_rows)

    monkeypatch.setattr(mmlu_mod, "_load_hf", fake_load_dataset)
    loader = mmlu_mod.MMLULoader()
    items = loader.load(3)
    assert len(items) == 3
