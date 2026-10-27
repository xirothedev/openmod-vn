"""Tests for encoder veto wiring (fake encoder, no weights)."""

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


def make_pipe(monkeypatch, encoder):
    monkeypatch.setattr(pipe_mod, "TransformerSpanJudge", lambda model_dir: encoder)
    return CorePipeline(encoder_dir="/tmp/l3")


def test_veto_clears_false_positive(monkeypatch):
    pipe = make_pipe(monkeypatch, FakeEncoder("ALLOW", 0.95))
    d = pipe.moderate("Nông dân trồng rau sạch ngoài vườn.")
    assert d.action == "ALLOW"


def test_veto_keeps_encoder_block(monkeypatch):
    pipe = make_pipe(monkeypatch, FakeEncoder("BLOCK", 0.9))
    d = pipe.moderate("Đi chăn rau khu Cầu Giấy, cho xin số điện thoại em nhé?")
    assert d.action in ("WARN", "BLOCK", "MASK")
    assert len(pipe.encoder.calls) >= 1
    phrase = pipe.encoder.calls[0][0]
    assert "chăn rau" in phrase or "rau" in phrase


def test_encoder_error_keeps_heuristic(monkeypatch):
    class Boom:
        def decide_fragment(self, *a, **k):
            raise RuntimeError("gpu gone")

    pipe = make_pipe(monkeypatch, Boom())
    d = pipe.moderate("Thằng này địt cả lò nhà mày.")
    assert d.action in ("BLOCK", "WARN", "MASK")


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
