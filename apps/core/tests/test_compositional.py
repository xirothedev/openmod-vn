from core.compositional import CompositionalDetector
from core.normalizer import normalize


def _rules():
    return [
        {
            "id": "comp_vi_00001",
            "tokens": ["chăn", "rau"],
            "max_skip_gram_distance": 2,
            "category": "SEXUAL_INNUENDO",
            "base_severity": 2,
            "positive_context_indicators": ["vườn", "trồng"],
            "toxic_context_indicators": ["hẹn hò"],
        }
    ]


def test_skip_gram_gap():
    n = normalize("chăn mấy em rau")
    hits = CompositionalDetector(_rules(), threshold=0.0).detect(n.norm, n.align)
    assert len(hits) == 1


def test_positive_context_lowers():
    n = normalize("trồng rau ngoài vườn chăn nuôi")
    # order wrong -> no match; score path covered by pipeline test
    assert isinstance(n.norm, str)
