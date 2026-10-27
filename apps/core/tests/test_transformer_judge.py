"""Tests for TransformerSpanJudge (fake ORT session + tokenizer, no weights)."""

from __future__ import annotations

import sys
import types
from typing import ClassVar

import pytest

from core.decision import TransformerSpanJudge


class FakeEncoding:
    def __init__(self, ids, offsets):
        self.ids = ids
        self.offsets = offsets


class FakeTokenizer:
    IDS: ClassVar = {"[CLS]": 101, "[SEP]": 102, "[PAD]": 0}

    def __init__(self, ids, offsets):
        self._ids = ids
        self._offsets = offsets

    def encode(self, text, add_special_tokens=True):
        assert isinstance(text, str)
        if text == "":
            return FakeEncoding([], [])
        return FakeEncoding(self._ids, self._offsets)

    def token_to_id(self, token):
        return self.IDS.get(token)


class FakeSession:
    def __init__(self, cls_logits, bio_logits):
        self._cls = cls_logits
        self._bio = bio_logits
        self.feed = None

    def run(self, _names, feed):
        self.feed = feed
        return [self._cls, self._bio]


def install_fake(monkeypatch, cls_logits, bio_logits, ids, offsets):
    fake_ort = types.ModuleType("onnxruntime")

    def session_factory(path, providers=None):
        assert str(path).endswith("model.int8.onnx")
        return FakeSession(cls_logits, bio_logits)

    fake_ort.InferenceSession = session_factory
    fake_tok_mod = types.ModuleType("tokenizers")

    class FakeTokenizerMod:
        @staticmethod
        def from_file(path):
            assert str(path).endswith("tokenizer.json")
            return FakeTokenizer(ids, offsets)

    fake_tok_mod.Tokenizer = FakeTokenizerMod
    fake_np = types.ModuleType("numpy")
    fake_np.array = lambda x, dtype=None: x
    fake_np.int64 = "int64"
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "tokenizers", fake_tok_mod)
    monkeypatch.setitem(sys.modules, "numpy", fake_np)


def test_block_with_span(monkeypatch):
    install_fake(
        monkeypatch,
        cls_logits=[[0.1, 0.2, 2.0]],
        bio_logits=[
            [[0.9, 0.05, 0.05]] * 2 + [[0.1, 0.8, 0.1], [0.1, 0.2, 0.7]] + [[0.9, 0.05, 0.05]] * 3
        ],
        ids=[1, 2, 3],
        offsets=[(0, 3), (4, 7), (7, 10)],
    )
    j = TransformerSpanJudge(model_dir="/tmp/l3")
    d = j.decide("abc def ghi")
    assert d.action == "BLOCK"
    assert d.overall_severity == 3
    assert [(s.start, s.end) for s in d.detected_spans] == [(4, 10)]
    assert 0.0 < d.confidence <= 1.0


def test_allow_has_no_spans(monkeypatch):
    install_fake(
        monkeypatch,
        cls_logits=[[3.0, 0.1, 0.1]],
        bio_logits=[[[0.9, 0.05, 0.05]]],
        ids=[1],
        offsets=[(0, 5)],
    )
    j = TransformerSpanJudge(model_dir="/tmp/l3")
    d = j.decide("hello")
    assert d.action == "ALLOW"
    assert d.detected_spans == []


def test_borderline_maps_to_warn(monkeypatch):
    install_fake(
        monkeypatch,
        cls_logits=[[0.1, 2.0, 0.1]],
        bio_logits=[[[0.9, 0.05, 0.05]]],
        ids=[1],
        offsets=[(0, 5)],
    )
    j = TransformerSpanJudge(model_dir="/tmp/l3")
    d = j.decide("hello", policy="standard_social")
    assert d.action == "WARN"
    assert d.overall_severity == 2


def test_no_model_dir_raises():
    with pytest.raises(RuntimeError):
        TransformerSpanJudge().decide("hello")


def test_load_failure_raises(monkeypatch):
    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.InferenceSession = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk gone"))
    fake_tok_mod = types.ModuleType("tokenizers")
    fake_tok_mod.Tokenizer = types.SimpleNamespace(from_file=lambda p: FakeTokenizer([1], [(0, 1)]))
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "tokenizers", fake_tok_mod)
    with pytest.raises(RuntimeError):
        TransformerSpanJudge(model_dir="/tmp/l3").decide("hello")
