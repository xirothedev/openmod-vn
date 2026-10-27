from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LexHit:
    pattern_id: str
    term: str
    norm_span: tuple[int, int]
    orig_span: tuple[int, int]
    requires_context_check: bool
    severity: int


class LexicalMatcher:
    def __init__(self, patterns: list[dict]):
        self.patterns = patterns
        expanded: list[str] = []
        self._index: list[dict] = []
        for p in patterns:
            terms = [p["term"], *(p.get("variants") or [])]
            for t in terms:
                expanded.append(t.lower())
                self._index.append(p)
        self._expanded = expanded
        try:
            from ahocorasick_rs import AhoCorasick  # ponytail: fallback to trie when lib missing

            self._auto = AhoCorasick(patterns=expanded)
            self._backend = "rs"
        except ImportError:
            self._auto = None
            self._backend = "trie"

    def find(self, norm: str, align: list[int]) -> list[LexHit]:
        hits: list[LexHit] = []
        if self._backend == "rs" and self._auto is not None:
            for pid, s, e in self._auto.find_matches_as_indexes(norm, overlapping=True):
                p = self._index[pid]
                if not self._boundary(norm, s, e):
                    continue
                hits.append(self._hit(p, s, e, align))
            return hits
        for p, term in zip(self._index, [t for t in self._expanded]):
            start = norm.find(term)
            while start != -1:
                end = start + len(term)
                if self._boundary(norm, start, end):
                    hits.append(self._hit(p, start, end, align))
                start = norm.find(term, start + 1)
        return hits

    @staticmethod
    def _boundary(norm: str, s: int, e: int) -> bool:
        left = s == 0 or not norm[s - 1].isalnum()
        right = e >= len(norm) or not norm[e].isalnum()
        return left and right

    @staticmethod
    def _hit(p: dict, s: int, e: int, align: list[int]) -> LexHit:
        os_ = align[s] if s < len(align) else s
        oe = (align[e - 1] + 1) if e - 1 < len(align) else e
        return LexHit(
            pattern_id=p["id"],
            term=p["term"],
            norm_span=(s, e),
            orig_span=(os_, oe),
            requires_context_check=bool(p.get("requires_context_check", False)),
            severity=int(p.get("severity", 1)),
        )


def load_patterns(path: str) -> list[dict]:
    import json

    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
