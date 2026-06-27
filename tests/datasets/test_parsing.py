import pytest
from glmsbench.datasets.base import DatasetItem
from glmsbench.scoring.exact import extract_letter
from glmsbench.scoring.numeric import extract_number


def test_extract_letter_plain():
    assert extract_letter("B") == "B"


def test_extract_letter_wrapped():
    assert extract_letter("The answer is (C).") == "C"


def test_extract_letter_none():
    assert extract_letter("I don't know") is None


def test_extract_number_hash():
    assert extract_number("...so the answer is\n#### 42") == 42


def test_extract_number_bare():
    assert extract_number("Final: 7") == 7


def test_extract_number_none():
    assert extract_number("no number here") is None
