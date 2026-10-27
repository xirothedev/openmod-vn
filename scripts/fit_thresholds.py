#!/usr/bin/env python3
"""Fit (temperature, flag-threshold) on half of ViHSD test, report on the other.

Eval-only dataset, never bundled. Run on bvpt3 (needs datasets + torch env).

Usage:
  PYTHONPATH=apps/core/src ~/venvs/openmod-train/bin/python scripts/fit_thresholds.py \
    --encoder-dir ~/openmod-train/v2/xlmr
"""

from __future__ import annotations

import argparse
import random

TEMPS = (1.0, 1.5, 2.0, 3.0)
TAUS = (0.5, 0.7, 0.9)


def main() -> int:
    ap = argparse.ArgumentParser(description="Fit temperature + threshold")
    ap.add_argument("--encoder-dir", required=True)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    from datasets import load_dataset

    from core.pipeline import CorePipeline

    ds = load_dataset("stair-lab/ViHSD", split="test")
    texts = [r["free_text"] for r in ds]
    exp = [r["label_id"] != 0 for r in ds]
    idx = list(range(len(texts)))
    random.Random(a.seed).shuffle(idx)
    fit_idx, rep_idx = idx[: len(idx) // 2], idx[len(idx) // 2 :]

    def score(pipe, ids):
        fp = fn = 0
        for i in ids:
            f = pipe.moderate(texts[i]).action != "ALLOW"
            fp += f and not exp[i]
            fn += (not f) and exp[i]
        tp = sum(exp[i] for i in ids) - fn
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, sum(exp[i] for i in ids))
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        return fp, fn, round(prec, 3), round(rec, 3), round(f1, 3)

    best, results = None, {}
    for temp in TEMPS:
        for tau in TAUS:
            pipe = CorePipeline(
                encoder_dir=a.encoder_dir,
                encoder_flag_conf=tau,
                encoder_temperature=temp,
                shadow=False,
            )
            fp, fn, prec, rec, f1 = score(pipe, fit_idx)
            results[(temp, tau)] = (fp, fn, prec, rec, f1)
            print(
                f"fit T={temp} tau={tau} FP={fp} FN={fn} P={prec} R={rec} F1={f1}",
                flush=True,
            )
            if best is None or f1 > best[1]:
                best = ((temp, tau), f1)
    (temp, tau), _ = best
    pipe = CorePipeline(
        encoder_dir=a.encoder_dir,
        encoder_flag_conf=tau,
        encoder_temperature=temp,
        shadow=False,
    )
    fp, fn, prec, rec, f1 = score(pipe, rep_idx)
    print(f"REPORT T={temp} tau={tau} FP={fp} FN={fn} P={prec} R={rec} F1={f1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
