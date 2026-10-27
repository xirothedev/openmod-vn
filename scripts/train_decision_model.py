#!/usr/bin/env python3
"""Fine-tune L3 fast encoder + Qwen judge (offline-safe skeleton).

Usage:
  python scripts/train_decision_model.py --help
  python scripts/train_decision_model.py --list-steps
  python scripts/train_decision_model.py --check  (validates data readiness, no GPU needed)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAST_MODEL = "Fsoft-AIC/videberta-xsmall"
FALLBACK_MODEL = "FacebookAI/xlm-roberta-base"
JUDGE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
JUDGE_SWAP = "Qwen/Qwen3-4B"


def check() -> int:
    n_rules = len(
        (ROOT / "data/compositional/vi_compositional_phrases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    n_bench = sum(
        len(p.read_text(encoding="utf-8").splitlines())
        for p in (ROOT / "data/benchmarks").glob("*.jsonl")
    )
    print(f"rules={n_rules} benchmark_rows={n_bench}")
    print(f"fast={FAST_MODEL} fallback={FALLBACK_MODEL}")
    print(f"judge={JUDGE_MODEL} swap={JUDGE_SWAP}")
    bad = []
    for p in (ROOT / "data/compositional").glob("*.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            ce = r["contrastive_examples"]
            if len(ce["benign"]) < 2 or len(ce["toxic"]) < 2:
                bad.append(r["id"])
    if bad:
        print(f"NOT READY: {bad}")
        return 1
    print("READY: data meets 2+2 contrastive minimum")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="L3 training driver (see docs/training/decision-model.md)"
    )
    ap.add_argument("--list-steps", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        return check()
    print(
        "1. teacher-label ViHOS spans + HateCheck-VI with ViHateT5/ViSoBERT (research-only)"
    )
    print("2. human review every teacher label before it enters data/")
    print(f"3. fine-tune {FAST_MODEL} 3-way + span head (fallback {FALLBACK_MODEL})")
    print("4. export ONNX opset17 + dynamic int8, gate p99 <15ms CPU batch1 seq96")
    print(f"5. LoRA r=16 {JUDGE_MODEL} (swap {JUDGE_SWAP}) via vLLM guided_json")
    print("6. nightly distill judge corrections into encoder set")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
