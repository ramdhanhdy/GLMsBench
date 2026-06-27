def test_gsm8k_loader_extracts_int(monkeypatch):
    from glmsbench.datasets import gsm8k as mod

    fake_rows = [
        {"question": "Janet has 2 apples and buys 3. How many?", "answer": "Janet has 2+3=5 apples.\n#### 5"},
    ]

    def fake_load_dataset(*a, **k):
        class _DS:
            def __init__(self, rows):
                self._rows = rows
            def __iter__(self):
                return iter(self._rows)
        return _DS(fake_rows)

    monkeypatch.setattr(mod, "_load_hf", fake_load_dataset)
    loader = mod.GSM8KLoader()
    items = loader.load(1)
    assert len(items) == 1
    assert items[0].gold == 5
    assert "Janet" in items[0].prompt


def test_gsm8k_caps_n(monkeypatch):
    from glmsbench.datasets import gsm8k as mod
    fake_rows = [
        {"question": f"Q{i}", "answer": f"x\n#### {i}"} for i in range(20)
    ]

    def fake_load_dataset(*a, **k):
        class _DS:
            def __init__(self, rows):
                self._rows = rows
            def __iter__(self):
                return iter(self._rows)
        return _DS(fake_rows)

    monkeypatch.setattr(mod, "_load_hf", fake_load_dataset)
    items = mod.GSM8KLoader().load(5)
    assert len(items) == 5
    assert [it.gold for it in items] == [0, 1, 2, 3, 4]
