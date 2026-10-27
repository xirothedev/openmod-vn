"""Tests for encoder wiring: shadow sidecar (default) + live veto/flag modes."""

from __future__ import annotations

import core.pipeline as pipe_mod
from core.pipeline import CorePipeline


class FakeEncoder:
    def __init__(self, label="ALLOW", conf=0.9):
        self.label = label
        self.conf = conf
        self.calls = []

    def decide_fragment(self, phrase, left, right, phrase_start=0, policy="standard_social"):
        self.calls.append((phrase, left, right))
        return self.label, self.conf, []

    def decide(self, text, policy="standard_social"):
        from core.decision import ModerationDecision

        action = {"ALLOW": "ALLOW", "BORDERLINE": "WARN", "BLOCK": "BLOCK"}[self.label]
        return ModerationDecision(
            action=action,
            overall_severity=2,
            confidence=self.conf,
            rationale=f"fake {self.label}",
        )


def make_pipe(monkeypatch, encoder, **kw):
    monkeypatch.setattr(pipe_mod, "TransformerSpanJudge", lambda model_dir, **kw: encoder)
    return CorePipeline(encoder_dir="/tmp/l3", **kw)


def test_shadow_never_changes_behavior(monkeypatch):
    enc = FakeEncoder("BLOCK", 0.99)
    pipe = make_pipe(monkeypatch, enc)
    d = pipe.moderate("Cho tôi một bát bún bò Huế.")
    assert d.action == "ALLOW"
    assert "shadow encoder=BLOCK" in d.rationale


def test_shadow_notes_error(monkeypatch):
    class Boom:
        def decide(self, *a, **k):
            raise RuntimeError("gpu gone")

        def decide_fragment(self, *a, **k):
            raise RuntimeError("gpu gone")

    pipe = make_pipe(monkeypatch, Boom())
    d = pipe.moderate("Thằng này địt cả lò nhà mày.")
    assert d.action in ("BLOCK", "WARN", "MASK")
    assert "shadow encoder=error" in d.rationale


def test_live_veto_clears_false_positive(monkeypatch):
    pipe = make_pipe(monkeypatch, FakeEncoder("ALLOW", 0.95), shadow=False)
    d = pipe.moderate("Thằng này địt cả lò nhà mày.")
    assert d.action == "ALLOW"


def test_live_flag_catches_miss(monkeypatch):
    pipe = make_pipe(monkeypatch, FakeEncoder("BLOCK", 0.9), shadow=False)
    d = pipe.moderate("Cho tôi một bát bún bò Huế.")
    assert d.action == "BLOCK"


def test_no_encoder_unchanged():
    pipe = CorePipeline()
    assert pipe.encoder is None
    d = pipe.moderate("Cho tôi một bát bún bò Huế.")
    assert d.action == "ALLOW"


def test_bio_spans_merge():
    from core.decision import TransformerSpanJudge

    offsets = [(0, 3), (4, 7), (7, 10), (11, 14)]
    tags = [0, 1, 2, 0]
    assert TransformerSpanJudge._bio_spans("abc def ghijkl", offsets, tags) == [(4, 10)]
    assert TransformerSpanJudge._bio_spans("abc", [(0, 3)], [2]) == [(0, 3)]
    assert TransformerSpanJudge._bio_spans("abc", [(0, 3)], [0]) == []
