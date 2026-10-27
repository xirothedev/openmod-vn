#!/usr/bin/env python3
"""Build binary filter-classifier data from l3_vi + benchmarks (stdlib only).

Maps 3-way L3 labels to binary filter labels:
  ALLOW -> CLEAN, BORDERLINE|BLOCK -> TOXIC (severity kept for policy mapping).

Usage:
  python3 scripts/build_classifier_data.py --out data/classifier_vi.jsonl
  python3 scripts/build_classifier_data.py --out data/classifier_vi.jsonl --limit 200
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TODAY = datetime.date.today().isoformat()
MAP = {"ALLOW": "CLEAN", "BORDERLINE": "TOXIC", "BLOCK": "TOXIC"}


def load_jsonl(p: Path, limit: int = 0) -> list[dict]:
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Build binary classifier data")
    ap.add_argument("--l3", default="data/l3_vi.jsonl")
    ap.add_argument("--bench", default="data/benchmarks/vihos_spans.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    rows = load_jsonl(ROOT / a.l3, a.limit)
    out: list[dict] = []
    for r in rows:
        label3 = r.get("label", "ALLOW")
        out.append(
            {
                "id": f"clf_{r.get('id', '?')}",
                "text": r.get("text", ""),
                "label": MAP.get(label3, "CLEAN"),
                "severity": int(r.get("severity", 0)),
                "from_l3_label": label3,
                "spans": r.get("spans", []),
                "is_teencode": bool(r.get("is_teencode", False)),
                "source": r.get("source", ""),
                "source_url": r.get("source_url", ""),
                "license": r.get("license", ""),
                "retrieved_at": TODAY,
            }
        )
    bench = load_jsonl(ROOT / a.bench, 0)
    for b in bench:
        lab = "CLEAN" if b.get("label") == "CLEAN" else "TOXIC"
        out.append(
            {
                "id": f"clf_{b.get('id', '?')}",
                "text": b.get("text", ""),
                "label": lab,
                "severity": int(b.get("severity", 0)),
                "from_l3_label": "ALLOW" if lab == "CLEAN" else "BLOCK",
                "spans": b.get("target_spans", []),
                "is_teencode": False,
                "source": b.get("source", ""),
                "source_url": b.get("source_url", ""),
                "license": b.get("license", ""),
                "retrieved_at": TODAY,
            }
        )

    outp = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
    with open(outp, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_toxic = sum(1 for r in out if r["label"] == "TOXIC")
    print(f"OK {outp} rows={len(out)} toxic={n_toxic} clean={len(out) - n_toxic}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
