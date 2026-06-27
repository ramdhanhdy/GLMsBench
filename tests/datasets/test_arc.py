def test_arc_loader_formats(monkeypatch):
    from glmsbench.datasets import arc as mod

    # ARC labels are letters; choices come as {label: text} dict or {labels, text}
    fake_rows = [
        {
            "question": "Which is a mammal?",
            "choices": {"label": ["A", "B", "C", "D"], "text": ["shark", "whale", "tuna", "eel"]},
            "answerKey": "B",
        }
    ]

    def fake_load_dataset(*a, **k):
        class _DS:
            def __init__(self, rows): self._rows = rows
            def __iter__(self): return iter(self._rows)
        return _DS(fake_rows)

    monkeypatch.setattr(mod, "_load_hf", fake_load_dataset)
    items = mod.ARCLoader().load(1)
    assert items[0].gold == "B"
    assert "Which is a mammal?" in items[0].prompt
    assert "A. shark" in items[0].prompt
