from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path

_ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\ufeff", "\u2060", "\u00ad"}
_SEPARATORS = set("._*|`~^:;!-+/\\")

_HOMOGLYPH = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "6": "g",
    "7": "t",
    "8": "b",
    "9": "g",
    "@": "a",
    "$": "s",
}


@dataclass
class Normalized:
    norm: str
    align: list[int]


def _load_tables() -> tuple[dict[str, str], dict[str, str]]:
    base = Path(__file__).resolve().parents[5] / "data" / "normalization"
    teen: dict[str, str] = {}
    homo = dict(_HOMOGLYPH)
    try:
        teen = json.loads((base / "teencode_mapping.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        pass
    try:
        homo.update(json.loads((base / "leetspeak_homoglyphs.json").read_text(encoding="utf-8")))
    except FileNotFoundError:
        pass
    return teen, homo


def normalize(text: str) -> Normalized:
    text = unicodedata.normalize("NFC", text)
    teen, homo = _load_tables()
    # zero-width strip, keep align
    chars: list[str] = []
    align: list[int] = []
    for i, ch in enumerate(text):
        if ch in _ZERO_WIDTH:
            continue
        if unicodedata.category(ch) in ("Cc", "Cf"):
            continue
        chars.append(ch)
        align.append(i)
    s = "".join(chars)
    # NFC already applied to text before align built, so positions hold.
    # Fold per char with offset tracking
    out: list[str] = []
    out_align: list[int] = []
    raw = list(s)
    raw_align = align[: len(raw)] + [align[-1] if align else 0] * max(0, len(raw) - len(align))
    i = 0
    buf_word: list[str] = []
    buf_idx: list[int] = []

    def flush_word() -> None:
        if not buf_word:
            return
        w = "".join(buf_word).lower()
        # longest-first teencode handled by full-word lookup (tables are word-level)
        rep = teen.get(w, w)
        if rep == w:
            for c, ai in zip(buf_word, buf_idx):
                out.append(c)
                out_align.append(ai)
        else:
            for c in rep:
                out.append(c)
                out_align.append(buf_idx[0])
        buf_word.clear()
        buf_idx.clear()

    while i < len(raw):
        ch = raw[i]
        ai = raw_align[i] if i < len(raw_align) else (align[-1] if align else 0)
        low = ch.lower()
        folded = homo.get(low, low)
        if folded.isalnum():
            buf_word.append(folded)
            buf_idx.append(ai)
            i += 1
            continue
        # separator: flush word, decide keep/delete
        flush_word()
        nxt = raw[i + 1] if i + 1 < len(raw) else ""
        prv = out[-1] if out else ""
        if ch in _SEPARATORS and prv.isalnum() and nxt != "" and nxt.lower().isalnum():
            i += 1  # delete injection inside word: c.ặ.c -> cặc
            continue
        if ch.isspace():
            if not out or out[-1] == " ":
                i += 1
                continue
            out.append(" ")
            out_align.append(ai)
            i += 1
            continue
        i += 1  # drop other punctuation
    flush_word()
    norm = "".join(out).strip()
    # collapse 3+ repeats -> 2
    collapsed: list[str] = []
    collapsed_align: list[int] = []
    for idx, c in enumerate(norm):
        if len(collapsed) >= 2 and collapsed[-1] == c and collapsed[-2] == c:
            continue
        collapsed.append(c)
        collapsed_align.append(out_align[idx] if idx < len(out_align) else 0)
    return Normalized(norm="".join(collapsed), align=collapsed_align)
