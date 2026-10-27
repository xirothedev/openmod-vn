from __future__ import annotations

from typing import Literal

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
    def __init__(self, model_id: str = "Fsoft-AIC/videberta-xsmall"):
        self.model_id = model_id

    def decide(self, *args, **kwargs) -> ModerationDecision:
        raise NotImplementedError(
            "Transformer judge is a lazy stub in v0; wire weights in docs/training"
        )


class LLMDecisionJudge:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-3B-Instruct"):
        self.model_id = model_id

    def decide(self, *args, **kwargs) -> ModerationDecision:
        raise NotImplementedError("LLM judge is a stub in v0; serve via vLLM guided_json")
