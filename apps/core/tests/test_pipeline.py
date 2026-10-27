from core.pipeline import CorePipeline

_pipe: CorePipeline | None = None


def pipe() -> CorePipeline:
    global _pipe
    if _pipe is None:
        _pipe = CorePipeline()
    return _pipe


def test_whitelist_no_false_positive():
    d = pipe().moderate("Cho tôi một bát bún bò Huế.")
    assert d.action == "ALLOW"


def test_explicit_blocked():
    d = pipe().moderate("Thằng này địt cả lò nhà mày.")
    assert d.action in ("BLOCK", "WARN", "MASK")
    assert d.overall_severity >= 2


def test_compositional_benign_allowed():
    d = pipe().moderate("Nông dân trồng rau sạch ngoài vườn.")
    assert d.action == "ALLOW"


def test_compositional_toxic_flagged():
    d = pipe().moderate("Đi chăn rau khu Cầu Giấy, cho xin số điện thoại em nhé?")
    assert d.action in ("WARN", "BLOCK", "MASK")
    assert any(s.is_compositional for s in d.detected_spans)


def test_bare_phrase_no_indicators_allowed():
    # bare "chăn rau" with neither toxic nor positive context stays below threshold
    d = pipe().moderate("Hôm qua chăn rau xong về nhà.")
    assert d.action == "ALLOW"


def test_policy_kid_safe_stricter():
    d_std = pipe().moderate("Loại mặt dày như mày ai cũng ghét.", policy="standard_social")
    d_kid = pipe().moderate("Loại mặt dày như mày ai cũng ghét.", policy="kid_safe")
    order = {"ALLOW": 0, "MASK": 1, "WARN": 2, "BLOCK": 3, "ESCALATE": 4}
    assert order[d_kid.action] >= order[d_std.action]


def test_policy_relaxed_gaming_lenient():
    d_std = pipe().moderate("Loại mặt dày như mày ai cũng ghét.", policy="standard_social")
    d_rel = pipe().moderate("Loại mặt dày như mày ai cũng ghét.", policy="relaxed_gaming")
    order = {"ALLOW": 0, "MASK": 1, "WARN": 2, "BLOCK": 3, "ESCALATE": 4}
    assert order[d_rel.action] <= order[d_std.action]


def test_context_dependent_benign():
    d = pipe().moderate("Con chó nhà tôi rất ngoan.")
    assert d.action == "ALLOW"


def test_context_dependent_toxic():
    d = pipe().moderate("Đồ con chó, cút khỏi đây.")
    assert d.action in ("WARN", "BLOCK", "MASK")
