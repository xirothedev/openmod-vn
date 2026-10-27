from core.normalizer import normalize


def test_teencode_maps():
    assert normalize("k0 đi đc").norm == "không đi được"


def test_punctuation_injection_with_offsets():
    n = normalize("c.ặ.c")
    assert "cặc" in n.norm
    assert n.align[0] == 0


def test_repeated_collapse():
    assert normalize("điiiiên").norm == "điiên"


def test_zero_width_stripped():
    assert normalize("a\u200bb").norm == "ab"
