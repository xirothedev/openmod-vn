#!/usr/bin/env python3
"""Fine-tune L3 fast encoder + Qwen judge (offline-safe driver).

Usage:
  python scripts/train_decision_model.py --help
  python scripts/train_decision_model.py --list-steps
  python scripts/train_decision_model.py --check  (validates data readiness, no GPU needed)
  python scripts/train_decision_model.py --smoke  (config + mem guard + tokenizer trial)
  python scripts/train_decision_model.py --export-onnx DIR  (writes export manifest stub)

OOM policy (bvpt3): one job at a time, micro-batch 8 x accum 4 (=32),
fp16, workers 2, seq <= 96. Kill-switch: RAM available <8GB or VRAM
free <4GB. See docs/training/decision-v1-bvpt3-plan.md.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAST_MODEL = "Fsoft-AIC/videberta-xsmall"
FALLBACK_MODEL = "FacebookAI/xlm-roberta-base"
JUDGE_SMOKE = "Qwen/Qwen3-0.6B"
JUDGE_MODEL = "Qwen/Qwen3-4B"

MIN_RAM_GB = 8.0
MIN_VRAM_GB = 4.0


@dataclass
class TrainConfig:
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
        if not 0 <= self.workers <= 8:
            errs.append("workers must be 0..8 (bvpt3: use <= 2)")
        if self.seq_max > 128:
            errs.append("seq_max must be <= 128 (encoder gate uses 96)")
        if self.lr <= 0:
            errs.append("lr must be > 0")
        if self.epochs < 1:
            errs.append("epochs must be >= 1")
        return errs


def _ram_available_gb(meminfo_path: Path) -> float | None:
    try:
        for line in meminfo_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    except OSError:
        return None
    return None


def _vram_free_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _ = torch.cuda.mem_get_info()
        return free / 1024 / 1024 / 1024
    except Exception:  # noqa: BLE001 - torch is optional; any backend failure means skip
        return None


def mem_guard(
    min_ram_gb: float = MIN_RAM_GB,
    min_vram_gb: float = MIN_VRAM_GB,
    meminfo_path: Path = Path("/proc/meminfo"),
    vram_free_gb: float | None = None,
) -> tuple[bool, str]:
    ram = _ram_available_gb(meminfo_path)
    notes = []
    if ram is None:
        notes.append("RAM unknown, skipped")
    elif ram < min_ram_gb:
        return False, f"KILL-SWITCH: RAM available {ram:.1f}GB < {min_ram_gb}GB"
    vram = vram_free_gb if vram_free_gb is not None else _vram_free_gb()
    if vram is None:
        notes.append("VRAM unknown, skipped")
    elif vram < min_vram_gb:
        return False, f"KILL-SWITCH: VRAM free {vram:.1f}GB < {min_vram_gb}GB"
    suffix = f" ({'; '.join(notes)})" if notes else ""
    return True, "mem guard pass" + suffix


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
    print(f"judge_smoke={JUDGE_SMOKE} judge={JUDGE_MODEL}")
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


def smoke(cfg: TrainConfig) -> int:
    print(
        f"smoke: micro={cfg.micro_batch} accum={cfg.grad_accum} "
        f"eff={cfg.effective_batch} fp16={cfg.fp16} workers={cfg.workers} "
        f"seq={cfg.seq_max} lr={cfg.lr} epochs={cfg.epochs}"
    )
    errs = cfg.validate()
    if errs:
        print("smoke: FAIL " + "; ".join(errs))
        return 2
    if cfg.workers > 2 or cfg.seq_max > 96 or cfg.effective_batch > 32:
        print("smoke: WARN exceeds v1 OOM-safe defaults (workers<=2 seq<=96 eff<=32)")
    ok, reason = mem_guard()
    print(f"smoke: {reason}")
    if not ok:
        return 3
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            FAST_MODEL, local_files_only=True, trust_remote_code=False
        )
        print(f"smoke: tokenizer OK vocab={tok.vocab_size}")
    except Exception as e:  # noqa: BLE001 - best-effort trial; offline-safe by design
        print(f"smoke: tokenizer SKIP ({type(e).__name__}, offline-safe)")
    print("smoke: PASS")
    return 0


def export_onnx_stub(out_dir: Path | str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": FAST_MODEL,
        "opset": 17,
        "quant": "int8-dynamic",
        "gate": {"p99_ms": 15, "batch": 1, "seq": 96},
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"onnx stub manifest written to {out / 'manifest.json'}")
    return out


def list_steps() -> int:
    print(
        "0. prep venv (peft/bnb/optimum) + license verify + ROLLBACK.txt, one job in tmux"
    )
    print(
        "1. teacher-label ViHOS spans + HateCheck-VI with ViHateT5/ViSoBERT (research-only)"
    )
    print("2. human review every teacher label before it enters data/")
    print(f"3. fine-tune {FAST_MODEL} 3-way + span head (fallback {FALLBACK_MODEL})")
    print("4. export ONNX opset17 + dynamic int8, gate p99 <15ms CPU batch1 seq96")
    print(f"5. QLoRA r=16 {JUDGE_MODEL} (smoke {JUDGE_SMOKE}) via vLLM guided_json")
    print("6. nightly distill judge corrections into encoder set")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="L3 training driver (see docs/training/decision-v1-bvpt3-plan.md)"
    )
    ap.add_argument("--list-steps", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--export-onnx", metavar="DIR", default=None)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-fp16", action="store_true")
    a = ap.parse_args(argv)
    cfg = TrainConfig(
        micro_batch=a.micro_batch,
        grad_accum=a.grad_accum,
        fp16=not a.no_fp16,
        workers=a.workers,
    )
    if a.check:
        return check()
    if a.smoke:
        return smoke(cfg)
    if a.export_onnx:
        export_onnx_stub(a.export_onnx)
        return 0
    if a.list_steps:
        return list_steps()
    return list_steps()


if __name__ == "__main__":
    raise SystemExit(main())
