#!/usr/bin/env python3
"""Benchmark heuristic-only vs encoder-veto hybrid on data/benchmarks rows.

Row labels: HATE/OFFENSIVE expect flagged (not ALLOW), CLEAN expects ALLOW.
Reports FP (clean flagged), FN (toxic allowed), and veto counts.

Usage (on bvpt3):
  PYTHONPATH=apps/core/src ~/venvs/openmod-train/bin/python scripts/bench_pipeline.py \
    --encoder-dir ~/openmod-train/bakeoff2/xlmr
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_bench() -> list[dict]:
    rows = []
    for p in sorted((ROOT / "data/benchmarks").glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                r["_file"] = p.name
                rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark veto hybrid vs heuristic")
    ap.add_argument("--encoder-dir", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--flag-conf", type=float, default=0.7)
    a = ap.parse_args()

    from core.pipeline import CorePipeline

    base = CorePipeline()
    hybrid = (
        CorePipeline(
            encoder_dir=a.encoder_dir, encoder_flag_conf=a.flag_conf, shadow=False
        )
        if a.encoder_dir
        else None
    )
    rows = load_bench()
    if a.limit:
        rows = rows[: a.limit]

    stats = {"n": 0, "fp_base": 0, "fn_base": 0, "fp_hyb": 0, "fn_hyb": 0, "vetoed": 0}
    t0 = time.perf_counter()
    for r in rows:
        text = r.get("text", "")
        if not text:
            continue
        expect_flag = r.get("label") in ("HATE", "OFFENSIVE")
        if r.get("label") not in ("HATE", "OFFENSIVE", "CLEAN"):
            continue
        stats["n"] += 1
        d0 = base.moderate(text)
        flagged0 = d0.action != "ALLOW"
        stats["fp_base"] += flagged0 and not expect_flag
        stats["fn_base"] += (not flagged0) and expect_flag
        if hybrid is not None:
            d1 = hybrid.moderate(text)
            flagged1 = d1.action != "ALLOW"
            stats["fp_hyb"] += flagged1 and not expect_flag
            stats["fn_hyb"] += (not flagged1) and expect_flag
            stats["vetoed"] += flagged0 and not flagged1
    dt = time.perf_counter() - t0
    stats["seconds"] = round(dt, 1)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
