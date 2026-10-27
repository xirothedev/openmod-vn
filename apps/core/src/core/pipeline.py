from __future__ import annotations

import json
from pathlib import Path

from .compositional import CompositionalDetector
from .decision import HeuristicContextJudge, ModerationDecision
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
    def __init__(self, data_dir: Path | None = None):
        root = data_dir or (_repo_root() / "data")
        lex = _load_jsonl(root / "raw_lexicons" / "vi" / "explicit_profanity.jsonl")
        white = _load_jsonl(root / "raw_lexicons" / "vi" / "safe_whitelist.jsonl")
        rules = _load_jsonl(root / "compositional" / "vi_compositional_phrases.jsonl")
        self.matcher = LexicalMatcher(lex)
        self.whitelist = {w["term"].lower() for w in white}
        self.detector = CompositionalDetector(rules)
        self.judge = HeuristicContextJudge()

    def moderate(self, text: str, policy: str = "standard_social") -> ModerationDecision:
        n = normalize(text)
        # whitelist suppression: drop hits covered by benign text
        hits = [
            h for h in self.matcher.find(n.norm, n.align) if h.term.lower() not in self.whitelist
        ]
        # extra guard: if norm itself contains whitelist phrase, suppress overlapping hits
        cands = self.detector.detect(n.norm, n.align)
        return self.judge.decide(text, n.norm, hits, cands, self.whitelist, policy)
