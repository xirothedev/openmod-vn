from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CandidateSpan:
    rule_id: str
    text: str
    norm_span: tuple[int, int]
    orig_span: tuple[int, int]
    suspicion: float
    category: str
    base_severity: int


def _tokenize(norm: str) -> tuple[list[str], list[tuple[int, int]]]:
    toks: list[str] = []
    spans: list[tuple[int, int]] = []
    for m in __import__("re").finditer(r"[^\s]+", norm):
        toks.append(m.group(0))
        spans.append((m.start(), m.end()))
    return toks, spans


class CompositionalDetector:
    def __init__(self, rules: list[dict], threshold: float = 0.5):
        self.rules = rules
        self.threshold = threshold

    def detect(self, norm: str, align: list[int]) -> list[CandidateSpan]:
        toks, tspans = _tokenize(norm)
        out: list[CandidateSpan] = []
        for r in self.rules:
            want = [t.lower() for t in r["tokens"]]
            gap = int(r.get("max_skip_gram_distance", 2))
            idx = self._match(toks, want, gap)
            if idx is None:
                continue
            s_tok, e_tok = idx
            ns = tspans[s_tok][0]
            ne = tspans[e_tok][1]
            score = self._score(norm, r)
            if score < self.threshold:
                continue
            os_ = align[ns] if ns < len(align) else ns
            oe = (align[ne - 1] + 1) if ne - 1 < len(align) else ne
            out.append(
                CandidateSpan(
                    rule_id=r["id"],
                    text=norm[ns:ne],
                    norm_span=(ns, ne),
                    orig_span=(os_, oe),
                    suspicion=score,
                    category=r.get("category", "HARASSMENT"),
                    base_severity=int(r.get("base_severity", 1)),
                )
            )
        return out

    @staticmethod
    def _match(toks: list[str], want: list[str], gap: int) -> tuple[int, int] | None:
        # ordered skip-gram within gap
        for i, t in enumerate(toks):
            if t != want[0]:
                continue
            j = i
            k = 1
            while k < len(want):
                found = -1
                for d in range(1, gap + 2):
                    if j + d < len(toks) and toks[j + d] == want[k]:
                        found = j + d
                        break
                if found == -1:
                    break
                j = found
                k += 1
            if k == len(want):
                return (i, j)
        return None

    @staticmethod
    def _score(norm: str, rule: dict) -> float:
        base = 0.3
        low = norm.lower()
        for t in rule.get("toxic_context_indicators", []):
            if t.lower() in low:
                base += 0.35
                break
        for t in rule.get("positive_context_indicators", []):
            if t.lower() in low:
                base -= 0.25
                break
        return max(0.0, min(1.0, base))
