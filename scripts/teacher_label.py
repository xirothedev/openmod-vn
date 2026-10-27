#!/usr/bin/env python3
"""Offline teacher-labeling for L3 rows (research-only teachers, UNREVIEWED output).

Reads ViHOS staging CSVs (content + char index_spans), labels each row with
ViHateT5 (sentence label + spans) and optionally ViSoBERT-HSD (agreement),
and writes candidate L3 rows to --out. Output is UNREVIEWED: never write it
directly into data/. Human review must approve rows first (see
docs/review/review_batch_*.md). Validator fields (source/source_url/license/
retrieved_at) are filled from the data ledger (data/SOURCES.md).

Teacher label map (reviewable): HATE -> BLOCK, OFFENSIVE -> BORDERLINE,
CLEAN -> ALLOW.

Usage (on bvpt3):
  ~/venvs/openmod-train/bin/python scripts/teacher_label.py \
    --input data/_staging/vihos_train.csv --limit 50 --out ~/openmod-train/review/l3_t5_smoke.jsonl
  ~/venvs/openmod-train/bin/python scripts/teacher_label.py \
    --input data/_staging/vihos_train.csv --out ~/openmod-train/review/l3_teacher_labels.jsonl
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

VIHOS_SOURCE = "https://github.com/phusroyal/vihos"
VIHOS_LICENSE = "MIT"
RETRIEVED_AT = "2026-10-27"
T5_MODEL = "tarudesu/ViHateT5-base-HSD"
VISOBERT_MODEL = "tarudesu/ViSoBERT-HSD"

LABEL_MAP = {"HATE": "BLOCK", "OFFENSIVE": "BORDERLINE", "CLEAN": "ALLOW"}


def parse_index_spans(raw: str) -> list[tuple[int, int]]:
    """Staging index_spans is a flat list of single char positions. Merge runs."""
    try:
        nums = sorted(set(ast.literal_eval(raw) if raw.strip() else []))
    except (SyntaxError, ValueError):
        return []
    spans, start, prev = [], None, None
    for n in nums:
        n = int(n)
        if start is None:
            start, prev = n, n
        elif n == prev + 1:
            prev = n
        else:
            spans.append((start, prev + 1))
            start, prev = n, n
    if start is not None:
        spans.append((start, prev + 1))
    return spans


def parse_t5_spans(raw: str) -> list[str]:
    import re

    return [s for s in re.findall(r"\[hate\](.*?)\[hate\]", raw) if s.strip()]


def window(text: str, spans: list[tuple[int, int]], pad: int = 40):
    if not spans:
        return text[:120], "", text[:120]
    s, e = spans[0]
    return (
        text[s:e],
        text[max(0, s - pad) : s],
        text[e : e + pad],
    )


def norm_label(raw: str) -> str:
    t = raw.strip().upper()
    for k in ("HATE", "OFFENSIVE", "CLEAN"):
        if k in t:
            return k
    return "UNKNOWN"


def load_t5(device: str):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(T5_MODEL, trust_remote_code=False)
    model = (
        AutoModelForSeq2SeqLM.from_pretrained(
            T5_MODEL,
            trust_remote_code=False,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        .to(device)
        .eval()
    )
    return tok, model


def generate(tok, model, device: str, text: str, prefix: str, max_tokens: int) -> str:
    import torch

    ids = tok.encode(f"{prefix}: {text}", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(ids, max_length=max_tokens)
    return tok.decode(out[0], skip_special_tokens=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Offline teacher labeling (UNREVIEWED output)"
    )
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-note", default="")
    a = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"teacher_label: device={device} model={T5_MODEL}")
    tok, model = load_t5(device)

    rows = []
    with open(a.input, encoding="utf-8") as f:
        for i, rec in enumerate(csv.DictReader(f)):
            if a.limit and i >= a.limit:
                break
            text = (rec.get("content") or "").strip()
            if not text:
                continue
            spans_idx = parse_index_spans(rec.get("index_spans") or "")
            span_texts = [text[s:e] for s, e in spans_idx if 0 <= s < e <= len(text)]
            raw_label = generate(tok, model, device, text, "hate-speech-detection", 16)
            raw_spans = generate(tok, model, device, text, "hate-spans-detection", 64)
            t5_spans = parse_t5_spans(raw_spans)
            if t5_spans:
                span_texts = t5_spans
            sent = norm_label(raw_label)
            phrase, left, right = window(text, spans_idx)
            rows.append(
                {
                    "id": f"t5_{i:05d}",
                    "text": text,
                    "phrase": phrase,
                    "left": left,
                    "right": right,
                    "teacher_label": sent,
                    "label": LABEL_MAP.get(sent, "BORDERLINE"),
                    "spans": span_texts,
                    "teacher_spans_raw": raw_spans,
                    "policy": "standard_social",
                    "is_teencode": False,
                    "status": "UNREVIEWED",
                    "source": VIHOS_SOURCE,
                    "source_url": VIHOS_SOURCE,
                    "license": VIHOS_LICENSE,
                    "retrieved_at": RETRIEVED_AT,
                    "teacher": {"model": T5_MODEL, "label_raw": raw_label},
                }
            )
            if (i + 1) % 200 == 0:
                print(f"teacher_label: {i + 1} rows")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    from collections import Counter

    print(f"teacher_label: wrote {len(rows)} rows to {out}")
    print("teacher_label: dist", dict(Counter(r["teacher_label"] for r in rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
