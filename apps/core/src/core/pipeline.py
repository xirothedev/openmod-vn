from __future__ import annotations

import json
from pathlib import Path

from .compositional import CompositionalDetector
from .decision import HeuristicContextJudge, ModerationDecision, TransformerSpanJudge
from .matcher import LexicalMatcher
from .normalizer import normalize


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class CorePipeline:
    def __init__(
        self,
        data_dir: Path | None = None,
        encoder_dir: Path | str | None = None,
        encoder_veto_conf: float = 0.8,
        encoder_flag_conf: float = 0.7,
        shadow: bool = True,
        encoder_temperature: float = 1.0,
    ):
        root = data_dir or (_repo_root() / "data")
        lexdir = root / "raw_lexicons" / "vi"
        lex: list[dict] = []
        for p in sorted(lexdir.glob("*.jsonl")):
            if p.name == "safe_whitelist.jsonl":
                continue
            lex.extend(_load_jsonl(p))
        white = _load_jsonl(root / "raw_lexicons" / "vi" / "safe_whitelist.jsonl")
        rules = _load_jsonl(root / "compositional" / "vi_compositional_phrases.jsonl")
        self.matcher = LexicalMatcher(lex)
        self.whitelist = {w["term"].lower() for w in white}
        self.detector = CompositionalDetector(rules)
        self.judge = HeuristicContextJudge()
        self.encoder = (
            TransformerSpanJudge(encoder_dir, temperature=encoder_temperature)
            if encoder_dir
            else None
        )
        self.encoder_veto_conf = encoder_veto_conf
        self.encoder_flag_conf = encoder_flag_conf
        self.shadow = shadow

    def moderate(self, text: str, policy: str = "standard_social") -> ModerationDecision:
        n = normalize(text)
        # whitelist suppression: drop hits covered by benign text
        hits = [
            h for h in self.matcher.find(n.norm, n.align) if h.term.lower() not in self.whitelist
        ]
        # extra guard: if norm itself contains whitelist phrase, suppress overlapping hits
        cands = self.detector.detect(n.norm, n.align)
        decision = self.judge.decide(text, n.norm, hits, cands, self.whitelist, policy)
        if self.encoder is not None:
            if self.shadow:
                self._shadow_note(text, policy, decision)
            elif decision.detected_spans:
                decision = self._encoder_veto(text, n.norm, hits, cands, decision, policy)
            else:
                flagged = self._encoder_flag(text, policy)
                if flagged is not None:
                    decision = flagged
        return decision

    def _shadow_note(self, text: str, policy: str, decision: ModerationDecision) -> None:
        """Run the encoder sidecar: annotate rationale, never change behavior."""
        assert self.encoder is not None
        try:
            enc = self.encoder.decide(text, policy)
            decision.rationale += f" | shadow encoder={enc.action} conf={enc.confidence:.2f}"
        except Exception:  # noqa: BLE001 - sidecar must never break moderate()
            decision.rationale += " | shadow encoder=error"

    def _encoder_flag(self, text: str, policy: str) -> ModerationDecision | None:
        """Flag text the heuristic missed when the encoder is confident.

        Whole text judged as phrase (short social texts resemble training
        fragments). Returns None to keep heuristic ALLOW.
        """
        assert self.encoder is not None
        try:
            enc_decision = self.encoder.decide(text, policy)
        except Exception:  # noqa: BLE001 - recall flag is advisory; heuristic stands
            return None
        if enc_decision.action == "ALLOW":
            return None
        if enc_decision.confidence < self.encoder_flag_conf:
            return None
        return enc_decision

    def _encoder_veto(self, text, norm, hits, cands, decision, policy) -> ModerationDecision:
        """Drop heuristic spans the encoder clears (ALLOW, high confidence).

        The encoder only vetoes — it never creates new blocks. Any encoder
        error keeps the heuristic decision unchanged.
        """
        assert self.encoder is not None
        kept_hits, kept_cands = list(hits), list(cands)
        try:
            for h in hits:
                s, e = h.orig_span
                if self._cleared(text, s, e):
                    kept_hits.remove(h)
            for c in cands:
                s, e = c.orig_span
                if self._cleared(text, s, e):
                    kept_cands.remove(c)
        except Exception:  # noqa: BLE001 - veto is advisory; heuristic stands on error
            return decision
        if len(kept_hits) == len(hits) and len(kept_cands) == len(cands):
            return decision
        return self.judge.decide(text, norm, kept_hits, kept_cands, self.whitelist, policy)

    def _cleared(self, text: str, s: int, e: int) -> bool:
        assert self.encoder is not None
        phrase = text[s:e]
        left, right = text[max(0, s - 40) : s], text[e : e + 40]
        try:
            label, conf, _ = self.encoder.decide_fragment(phrase, left, right, s)
        except Exception:  # noqa: BLE001 - single-span veto failure must not break moderate()
            return False
        return label == "ALLOW" and conf >= self.encoder_veto_conf
