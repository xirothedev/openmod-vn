#!/usr/bin/env python3
"""Build data/l3_vi.jsonl from approved teacher-label slices (review batch 03).

Approved: S1 CLEAN no-span -> ALLOW, S2 HATE/OFFENSIVE with spans -> mapped label,
S3 CLEAN with spans -> ALLOW with spans stripped. Dropped: S4 no-span HATE/OFFENSIVE.
Held out (not written here): S5 political rows -> --hold-out for a later policy call.

Usage:
  python3 scripts/build_l3_vi.py --mirror ~/.cache/openmod-review --out data/l3_vi.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SEVERITY = {"HATE": 3, "OFFENSIVE": 2, "CLEAN": 0}
S5_PATTERN = re.compile(
    r"phản động|\+ sản|cộng sản|đảng |đảng\.|chính trị|bầu cử|biểu tình"
)


def load(path: Path, prefix: str) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        r["id"] = f"{prefix}_{r['id']}"
        rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Build approved L3 rows")
    ap.add_argument("--mirror", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hold-out", required=True)
    ap.add_argument(
        "--allow-held",
        action="store_true",
        help="approved: include S5 political rows (review batch 03)",
    )
    a = ap.parse_args()

    mirror = Path(a.mirror).expanduser()
    src = load(mirror / "l3_teacher_train.jsonl", "l3_train") + load(
        mirror / "l3_teacher_dev.jsonl", "l3_dev"
    )
    kept, held, dropped = [], [], 0
    for r in src:
        if S5_PATTERN.search(r["text"]):
            held.append(r)
            continue
        has_spans = bool(r["spans"])
        if r["teacher_label"] == "CLEAN":
            r["spans"] = []  # S1 empty already; S3 stripped per approval
            kept.append(r)
        elif has_spans:
            kept.append(r)  # S2
        else:
            dropped += 1  # S4

    out = Path(a.out)
    if a.allow_held:
        kept += held
        held = []
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(
            json.dumps(
                {
                    "id": r["id"],
                    "text": r["text"],
                    "phrase": r["phrase"],
                    "left": r["left"],
                    "right": r["right"],
                    "label": r["label"],
                    "spans": r["spans"],
                    "policy": r["policy"],
                    "is_teencode": r["is_teencode"],
                    "severity": SEVERITY[r["teacher_label"]],
                    "source": r["source"],
                    "source_url": r["source_url"],
                    "license": r["license"],
                    "retrieved_at": r["retrieved_at"],
                    "teacher": r["teacher"],
                },
                ensure_ascii=False,
            )
            + "\n"
            for r in kept
        )
    hold = Path(a.hold_out)
    hold.parent.mkdir(parents=True, exist_ok=True)
    with open(hold, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in held)
    print(f"kept={len(kept)} held_s5={len(held)} dropped_s4={dropped} -> {out}")
    print("held_ids:", [r["id"] for r in held][:20])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
