from glmsbench.scoring.parity import compare_text_parity
from glmsbench.scoring.exact import extract_letter


def test_answer_agree_true():
    # Both outputs resolve to canonical "C" despite different prose.
    r = compare_text_parity(
        gold="C",
        output_a="The answer is (C).",
        output_b="C",
        extract_fn=extract_letter,
    )
    assert r["answer_agree"] is True
    assert r["raw_text_agree"] is False


def test_raw_text_agree_true():
    r = compare_text_parity(
        gold="C",
        output_a="C",
        output_b="C",
        extract_fn=extract_letter,
    )
    assert r["raw_text_agree"] is True and r["answer_agree"] is True


def test_answer_disagree():
    r = compare_text_parity(
        gold="C",
        output_a="A",
        output_b="C",
        extract_fn=extract_letter,
    )
    assert r["answer_agree"] is False
