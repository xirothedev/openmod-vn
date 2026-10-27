from __future__ import annotations

import math
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

Action = Literal["ALLOW", "MASK", "WARN", "BLOCK", "ESCALATE"]


class DetectedSpan(BaseModel):
    start: int
    end: int
    text: str
    category: str
    is_compositional: bool = False
    explanation: str = ""


class ModerationDecision(BaseModel):
    action: Action
    overall_severity: int = Field(ge=0, le=4)
    confidence: float
    detected_spans: list[DetectedSpan] = []
    masked_text: str = ""
    rationale: str = ""


POLICIES: dict[str, dict[str, int]] = {
    "kid_safe": {"escalate": 3, "block": 2, "warn": 1},
    "standard_social": {"escalate": 4, "block": 3, "warn": 2},
    "relaxed_gaming": {"escalate": 5, "block": 4, "warn": 3},
}


class HeuristicContextJudge:
    def decide(
        self,
        original_text: str,
        norm: str,
        lex_hits: list,
        candidates: list,
        whitelist_terms: set[str] | None = None,
        policy: str = "standard_social",
    ) -> ModerationDecision:
        spans: list[DetectedSpan] = []
        severity = 0
        # unconditional hits first
        for h in lex_hits:
            if whitelist_terms and h.term in whitelist_terms:
                continue
            if h.requires_context_check:
                if not self._aimed_at_person(norm):
                    continue
                spans.append(
                    DetectedSpan(
                        start=h.orig_span[0],
                        end=h.orig_span[1],
                        text=original_text[h.orig_span[0] : h.orig_span[1]],
                        category="INSULT_DEHUMANIZATION",
                        is_compositional=False,
                        explanation=f"context-dependent term aimed at person: {h.term}",
                    )
                )
                severity = max(severity, h.severity)
            else:
                spans.append(
                    DetectedSpan(
                        start=h.orig_span[0],
                        end=h.orig_span[1],
                        text=original_text[h.orig_span[0] : h.orig_span[1]],
                        category="PROFANITY",
                        explanation=f"explicit match: {h.term}",
                    )
                )
                severity = max(severity, h.severity)
        for c in candidates:
            spans.append(
                DetectedSpan(
                    start=c.orig_span[0],
                    end=c.orig_span[1],
                    text=original_text[c.orig_span[0] : c.orig_span[1]],
                    category=c.category,
                    is_compositional=True,
                    explanation=f"compositional {c.rule_id} suspicion={c.suspicion:.2f}",
                )
            )
            severity = max(severity, c.base_severity)
        if not spans:
            return ModerationDecision(
                action="ALLOW",
                overall_severity=0,
                confidence=0.95,
                masked_text=original_text,
                rationale="clean",
            )
        action: Action
        th = POLICIES.get(policy, POLICIES["standard_social"])
        if severity >= th["escalate"]:
            action = "ESCALATE"
        elif severity >= th["block"]:
            action = "BLOCK"
        elif severity >= th["warn"]:
            action = "WARN"
        else:
            action = "MASK"
        masked = original_text
        for s in sorted(spans, key=lambda x: x.start, reverse=True):
            masked = masked[: s.start] + "***" + masked[s.end :]
        return ModerationDecision(
            action=action,
            overall_severity=severity,
            confidence=0.8,
            detected_spans=spans,
            masked_text=masked,
            rationale=f"{len(spans)} span(s), max severity {severity}",
        )

    @staticmethod
    def _aimed_at_person(norm: str) -> bool:
        cues = ["đồ", "lũ", "tụi", "bọn", "mày", "thằng", "cút", "chửi"]
        return any(c in norm for c in cues)


class TransformerSpanJudge:
    """ONNX-backed L3 encoder judge (replaces the v0 stub).

    Loads ``model_dir/model.int8.onnx`` + ``tokenizer.json`` lazily on first
    :meth:`decide`. Any load/inference failure raises RuntimeError — callers
    (pipeline) own the fallback to :class:`HeuristicContextJudge`.
    """

    CLS_LABELS = ("ALLOW", "BORDERLINE", "BLOCK")
    SEVERITY: ClassVar = {"ALLOW": 0, "BORDERLINE": 2, "BLOCK": 3}

    def __init__(self, model_dir: str | Path | None = None):
        self.model_dir = Path(model_dir) if model_dir else None
        self._session = None
        self._tokenizer = None

    def _load(self):
        if self._session is not None:
            return
        if self.model_dir is None:
            raise RuntimeError("TransformerSpanJudge: no model_dir configured")
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            raise RuntimeError(f"TransformerSpanJudge: missing dependency: {e}") from e
        try:
            self._tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
            if self._tokenizer.token_to_id("[SEP]") is not None:
                self._cls_id, self._sep_id, self._pad_id = (
                    self._tokenizer.token_to_id("[CLS]"),
                    self._tokenizer.token_to_id("[SEP]"),
                    self._tokenizer.token_to_id("[PAD]"),
                )
            else:
                self._cls_id, self._sep_id, self._pad_id = (
                    self._tokenizer.token_to_id("<s>"),
                    self._tokenizer.token_to_id("</s>"),
                    self._tokenizer.token_to_id("<pad>"),
                )
            self._session = ort.InferenceSession(
                str(self.model_dir / "model.int8.onnx"),
                providers=["CPUExecutionProvider"],
            )
        except Exception as e:
            raise RuntimeError(f"TransformerSpanJudge: load failed: {e}") from e

    def _encode_fragment(
        self, phrase: str, left: str, right: str, seq_max: int = 96
    ) -> tuple[list[int], list[int], list[tuple[int, int]], list[int]]:
        """Mirror training encoding: `[CLS] phrase [SEP] left [SEP] right [SEP]`.

        Returns (ids, mask, offsets, seg_id) where offsets are char offsets
        *within the joined fragment string* and seg_id marks phrase tokens (1).
        """
        parts = [(phrase, 0), (left, len(phrase) + 6), (right, len(phrase) + 6 + len(left) + 6)]
        ids, offsets, seg_id = [self._cls_id], [(0, 0)], [0]
        for text, base in parts:
            enc = self._tokenizer.encode(text, add_special_tokens=False)
            ids += enc.ids
            offsets += [(base + s, base + e) for s, e in enc.offsets]
            seg_id += [1 if base == 0 else 0] * len(enc.ids)
            ids.append(self._sep_id)
            offsets.append((0, 0))
            seg_id.append(0)
        ids, offsets, seg_id = ids[:seq_max], offsets[:seq_max], seg_id[:seq_max]
        mask = [1] * len(ids)
        pad = seq_max - len(ids)
        return ids + [self._pad_id] * pad, mask + [0] * pad, offsets, seg_id

    def decide_fragment(
        self,
        phrase: str,
        left: str,
        right: str,
        phrase_start: int = 0,
        policy: str = "standard_social",
    ) -> tuple[str, float, list[tuple[int, int]]]:
        """Judge one training-format fragment.

        Returns (label, confidence, spans) with spans in original-text coords
        (phrase-segment BIO tags shifted by phrase_start).
        """
        self._load()
        try:
            import numpy as np

            ids, mask, offsets, seg_id = self._encode_fragment(phrase, left, right)
            cls_logits, bio_logits = self._session.run(
                None,
                {
                    "ids": np.array([ids], dtype=np.int64),
                    "mask": np.array([mask], dtype=np.int64),
                },
            )
            cls_logits = cls_logits[0]
            top = max(cls_logits)
            exps = [math.exp(float(v) - float(top)) for v in cls_logits]
            total = sum(exps)
            best = max(range(3), key=lambda i: exps[i])
            tags = [
                int(max(range(3), key=lambda c: bio_logits[0][i][c])) for i in range(len(offsets))
            ]
            frag_spans = self._bio_spans("", offsets, tags)
            spans = [
                (phrase_start + s, phrase_start + e)
                for s, e in frag_spans
                if self._in_phrase(offsets, tags, s, e, seg_id)
            ]
            return self.CLS_LABELS[best], exps[best] / total, spans
        except Exception as e:
            raise RuntimeError(f"TransformerSpanJudge: inference failed: {e}") from e

    @staticmethod
    def _in_phrase(offsets, tags, s, e, seg_id) -> bool:
        return any(
            seg_id[i] == 1 and offsets[i][0] >= s and offsets[i][1] <= e
            for i in range(len(offsets))
        )

    def decide(self, text: str, policy: str = "standard_social") -> ModerationDecision:
        self._load()
        try:
            import numpy as np

            enc = self._tokenizer.encode(text)
            ids = enc.ids[:96] + [0] * max(0, 96 - len(enc.ids))
            mask = [1] * min(len(enc.ids), 96) + [0] * max(0, 96 - len(enc.ids))
            cls_logits, bio_logits = self._session.run(
                None,
                {
                    "ids": np.array([ids], dtype=np.int64),
                    "mask": np.array([mask], dtype=np.int64),
                },
            )
            cls_logits = cls_logits[0]
            top = max(cls_logits)
            exps = [math.exp(float(v) - float(top)) for v in cls_logits]
            total = sum(exps)
            best = max(range(3), key=lambda i: exps[i])
            confidence = exps[best] / total
            label = self.CLS_LABELS[best]
            severity = self.SEVERITY[label]
            offsets = enc.offsets[:96]
            tags = [
                int(max(range(3), key=lambda c: bio_logits[0][i][c])) for i in range(len(offsets))
            ]
        except Exception as e:
            raise RuntimeError(f"TransformerSpanJudge: inference failed: {e}") from e
        spans: list[DetectedSpan] = []
        if label != "ALLOW":
            for s, e in self._bio_spans(text, offsets, tags):
                spans.append(
                    DetectedSpan(
                        start=s,
                        end=e,
                        text=text[s:e],
                        category="MODEL_SPAN",
                        explanation=f"encoder {label} conf={confidence:.2f}",
                    )
                )
        th = POLICIES.get(policy, POLICIES["standard_social"])
        if label == "ALLOW":
            action: Action = "ALLOW"
        elif severity >= th["block"]:
            action = "BLOCK"
        elif severity >= th["warn"]:
            action = "WARN"
        else:
            action = "MASK"
        return ModerationDecision(
            action=action,
            overall_severity=severity,
            confidence=confidence,
            detected_spans=spans,
            masked_text=text,
            rationale=f"encoder {label}",
        )

    @staticmethod
    def _bio_spans(
        text: str, offsets: list[tuple[int, int]], tags: list[int]
    ) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        start: int | None = None
        prev = 0
        for (s, e), t in zip(offsets, tags):
            if t == 1 or (t == 2 and start is None):
                if start is not None:
                    spans.append((start, prev))
                start = s
            elif t == 0 and start is not None:
                spans.append((start, prev))
                start = None
            prev = e
        if start is not None:
            spans.append((start, prev))
        if text:
            return [(s, e) for s, e in spans if 0 <= s < e <= len(text)]
        return [(s, e) for s, e in spans if 0 <= s < e]


class LLMDecisionJudge:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-3B-Instruct"):
        self.model_id = model_id

    def decide(self, *args, **kwargs) -> ModerationDecision:
        raise NotImplementedError("LLM judge is a stub in v0; serve via vLLM guided_json")
