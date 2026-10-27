#!/usr/bin/env python3
"""Download allowed external sources and refactor them into our schemas.

Bundled (license-compatible, with citation on every row):
  - blue-eyes-vn/vietnamese-offensive-words (MIT) -> raw_lexicons/vi/offensive_words_import.jsonl
  - phusroyal/ViHOS span rows (MIT) -> benchmarks/vihos_spans.jsonl
  - HateCheck templates adapted to VI (CC-BY-4.0) -> benchmarks/hatecheck_vi_templates.jsonl

Reference-only (staging, never bundled): ViHSD, ViCTSD, VOZ-HSD, ViLexNorm, ToxiGen, Jigsaw, SALT.
Raw downloads live in data/_staging/ (gitignored).
"""

from __future__ import annotations

import ast
import csv
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STAGE = DATA / "_staging"
DAY = "2026-10-27"

SRC_OFF = "https://github.com/blue-eyes-vn/vietnamese-offensive-words"
SRC_VIHOS = "https://github.com/phusroyal/vihos"
SRC_HC = "https://github.com/paul-rottger/hatecheck-data"

FILES = {
    STAGE
    / "vn_offensive_words.txt": "https://raw.githubusercontent.com/blue-eyes-vn/vietnamese-offensive-words/master/vn_offensive_words.txt",
    STAGE
    / "vihos_train.csv": "https://raw.githubusercontent.com/phusroyal/ViHOS/master/data/Span_Extraction_based_version/train.csv",
    STAGE
    / "vihos_dev.csv": "https://raw.githubusercontent.com/phusroyal/ViHOS/master/data/Span_Extraction_based_version/dev.csv",
    STAGE
    / "hatecheck_cases.csv": "https://raw.githubusercontent.com/paul-rottger/hatecheck-data/main/test_suite_cases.csv",
}


def fetch() -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    for dest, url in FILES.items():
        if dest.exists() and dest.stat().st_size > 0:
            print(f"reuse {dest.name}")
            continue
        print(f"get {url}")
        urllib.request.urlretrieve(url, dest)


def convert_offensive() -> int:
    group = "ungrouped"
    rows: list[dict] = []
    n = 0
    for line in (
        (STAGE / "vn_offensive_words.txt").read_text(encoding="utf-8").splitlines()
    ):
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            group = s.lstrip("# ").lower().replace(" ", "_")[:40] or "ungrouped"
            continue
        n += 1
        rows.append(
            {
                "id": f"imp_lex_{n:05d}",
                "term": s.lower(),
                "variants": [],
                "language": "vi",
                "category": "PROFANITY",
                "severity": 2,
                "requires_context_check": True,
                "tags": ["imported", "needs-review", group],
                "source": SRC_OFF,
                "source_url": SRC_OFF,
                "license": "MIT",
                "retrieved_at": DAY,
            }
        )
    out = DATA / "raw_lexicons" / "vi" / "offensive_words_import.jsonl"
    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    return len(rows)


def _spans_to_texts(content: str, idx: list[int]) -> list[str]:
    idx = sorted(set(idx))
    if not idx:
        return []
    parts, cur = [], [idx[0]]
    for x in idx[1:]:
        if x == cur[-1] + 1:
            cur.append(x)
        else:
            parts.append("".join(content[i] for i in cur if 0 <= i < len(content)))
            cur = [x]
    parts.append("".join(content[i] for i in cur if 0 <= i < len(content)))
    return [p for p in parts if p.strip()]


def convert_vihos() -> int:
    rows: list[dict] = []
    n = 0
    for name in ("vihos_train.csv", "vihos_dev.csv"):
        with open(STAGE / name, encoding="utf-8") as f:
            for rec in csv.DictReader(f):
                try:
                    idx = ast.literal_eval(rec.get("index_spans") or "[]")
                except (ValueError, SyntaxError):
                    continue
                if not idx:
                    continue
                texts = _spans_to_texts(
                    rec.get("content", ""), [int(x) for x in idx if str(x).isdigit()]
                )
                if not texts:
                    continue
                n += 1
                rows.append(
                    {
                        "id": f"vihos_{n:05d}",
                        "text": rec.get("content", ""),
                        "label": "OFFENSIVE",
                        "severity": 2,
                        "is_compositional": False,
                        "target_spans": texts,
                        "functional_test_type": "vihos_span",
                        "source": SRC_VIHOS,
                        "source_url": SRC_VIHOS,
                        "license": "MIT",
                        "retrieved_at": DAY,
                    }
                )
    out = DATA / "benchmarks" / "vihos_spans.jsonl"
    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    return len(rows)


# (functionality, vi_text, label, severity, spans, test_type)
TEMPLATES: list[tuple[str, str, str, int, list[str], str]] = [
    (
        "derogation",
        "Tao ghét bọn người miền núi.",
        "HATE",
        3,
        ["ghét"],
        "derogation_vi",
    ),
    ("derogation", "Tao ghét phụ nữ.", "HATE", 3, ["ghét"], "derogation_vi"),
    (
        "threat",
        "Tao sẽ đánh mày chết ở cổng trường.",
        "HATE",
        4,
        ["đánh mày chết"],
        "threat_vi",
    ),
    ("slur", "Lũ mọi rợ này cút khỏi đây.", "HATE", 3, ["mọi rợ"], "slur_vi"),
    (
        "negation",
        "Tôi không ghét phụ nữ, tôi tôn trọng họ.",
        "CLEAN",
        0,
        [],
        "negated_hate_vi",
    ),
    (
        "quote",
        'Nó nói "ghét bọn nhà quê" rồi cười.',
        "OFFENSIVE",
        2,
        ["ghét"],
        "quoted_hate_vi",
    ),
    (
        "spelling",
        "Taaaao ghééét lũ chó hùaaa.",
        "HATE",
        3,
        ["ghééét"],
        "spelling_variation_vi",
    ),
    ("teencode", "k0 ua phu nu chut nao.", "HATE", 3, ["k0 ua"], "teencode_hate_vi"),
    (
        "dehumanization",
        "Lũ bốn chân này biến khỏi đây.",
        "HATE",
        3,
        ["bốn chân"],
        "dehumanization_vi",
    ),
    (
        "insult",
        "Óc heo như mày thì đừng lên tiếng.",
        "OFFENSIVE",
        2,
        ["óc heo"],
        "insult_vi",
    ),
    (
        "innuendo",
        "Bị đồn đi bán hoa ở vũ trường.",
        "OFFENSIVE",
        2,
        ["bán hoa"],
        "innuendo_vi",
    ),
    (
        "benign_group",
        "Phụ nữ miền núi dệt thổ cẩm rất đẹp.",
        "CLEAN",
        0,
        [],
        "benign_group_mention_vi",
    ),
    (
        "benign_reclaim",
        "Cháo óc heo bổ dưỡng cho người ốm.",
        "CLEAN",
        0,
        [],
        "benign_overlap_vi",
    ),
    (
        "counter",
        "Tôi phản đối kẻ nói ghét người đồng tính.",
        "CLEAN",
        0,
        [],
        "counterspeech_vi",
    ),
    (
        "target_individual",
        "Mày là đồ ăn bám, cút đi.",
        "OFFENSIVE",
        2,
        ["ăn bám"],
        "targeted_individual_vi",
    ),
    (
        "target_group",
        "Bọn nhà quê này chỉ giỏi ăn bám.",
        "HATE",
        3,
        ["ăn bám"],
        "targeted_group_vi",
    ),
    (
        "positive",
        "Buổi tối ở phố cổ rất đông vui.",
        "CLEAN",
        0,
        [],
        "benign_homograph_vi",
    ),
    (
        "positive",
        "Cô Lan bán hoa quả ngoài chợ.",
        "CLEAN",
        0,
        [],
        "benign_slang_overlap_vi",
    ),
]


def convert_hatecheck_vi() -> int:
    rows = [
        {
            "id": f"hcvit_{i + 1:05d}",
            "text": t,
            "label": lab,
            "severity": sev,
            "is_compositional": False,
            "target_spans": spans,
            "functional_test_type": ftype,
            "source": "HateCheck adapted to Vietnamese",
            "source_url": SRC_HC,
            "license": "CC-BY-4.0",
            "retrieved_at": DAY,
        }
        for i, (_, t, lab, sev, spans, ftype) in enumerate(TEMPLATES)
    ]
    out = DATA / "benchmarks" / "hatecheck_vi_templates.jsonl"
    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    return len(rows)


def main() -> None:
    fetch()
    a = convert_offensive()
    b = convert_vihos()
    c = convert_hatecheck_vi()
    print(f"OK lexicon_import={a} vihos_spans={b} hatecheck_vi_templates={c}")


if __name__ == "__main__":
    main()
