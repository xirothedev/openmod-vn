#!/usr/bin/env python3
"""Binary filter-classifier finetune skeleton (CLEAN|TOXIC, CLS-only, OOM-safe).

Prep skeleton for bvpt3. Full loop mirrors scripts/finetune_encoder.py
(3-way + BIO) with a 2-way CLS head on full text. Today implements
--check/--smoke + config validation only; training body is an explicit TODO.

Usage:
  python3 scripts/train_classifier.py --check --data data/classifier_vi.jsonl
  python3 scripts/train_classifier.py --smoke
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["CLEAN", "TOXIC"]
BASE_MODEL = "FacebookAI/xlm-roberta-base"


@dataclass
class ClassifierConfig:
    micro_batch: int = 8
    grad_accum: int = 4
    fp16: bool = True
    workers: int = 2
    seq_max: int = 96
    lr: float = 2e-5
    epochs: int = 10

    @property
    def effective_batch(self) -> int:
        return self.micro_batch * self.grad_accum

    def validate(self) -> list[str]:
        errs = []
        if self.micro_batch < 1:
            errs.append("micro_batch must be >= 1")
        if self.grad_accum < 1:
            errs.append("grad_accum must be >= 1")
        if not 0 <= self.workers <= 2:
            errs.append("workers must be 0..2 (bvpt3 OOM-safe)")
        if self.seq_max > 96:
            errs.append("seq_max must be <= 96 (encoder gate)")
        if self.lr <= 0:
            errs.append("lr must be > 0")
        if self.epochs < 1:
            errs.append("epochs must be >= 1")
        return errs


def check(data: str) -> int:
    p = ROOT / data
    if not p.exists():
        print(f"NOT READY: {data} missing; run scripts/build_classifier_data.py first")
        return 1
    n = toxic = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("label") not in LABELS:
            print(f"NOT READY: bad label {r.get('label')} in {r.get('id')}")
            return 1
        n += 1
        toxic += r["label"] == "TOXIC"
    print(f"READY: rows={n} toxic={toxic} clean={n - toxic} base={BASE_MODEL}")
    return 0


def smoke(cfg: ClassifierConfig) -> int:
    errs = cfg.validate()
    if errs:
        print("smoke: FAIL " + "; ".join(errs))
        return 2
    print(
        f"smoke: micro={cfg.micro_batch} accum={cfg.grad_accum} "
        f"eff={cfg.effective_batch} fp16={cfg.fp16} seq={cfg.seq_max} "
        f"lr={cfg.lr} epochs={cfg.epochs} base={BASE_MODEL}"
    )
    print(
        "smoke: PASS (training loop TODO — mirror finetune_encoder.py CLS head 2-way)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Filter classifier skeleton")
    ap.add_argument("--data", default="data/classifier_vi.jsonl")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    cfg = ClassifierConfig()
    if a.check:
        return check(a.data)
    if a.smoke:
        return smoke(cfg)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
