"""Tests for scripts/train_decision_model.py (offline-safe, no GPU needed)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/train_decision_model.py"


def load_driver():
    spec = importlib.util.spec_from_file_location("train_decision_model", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_decision_model"] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        del sys.modules["train_decision_model"]
    return mod


drv = load_driver()


def test_model_ids_match_v1_plan():
    assert drv.FAST_MODEL == "Fsoft-AIC/videberta-xsmall"
    assert drv.FALLBACK_MODEL == "FacebookAI/xlm-roberta-base"
    assert drv.JUDGE_SMOKE == "Qwen/Qwen3-0.6B"
    assert drv.JUDGE_MODEL == "Qwen/Qwen3-4B"


def test_config_defaults_are_oom_safe():
    c = drv.TrainConfig()
    assert (c.micro_batch, c.grad_accum) == (8, 4)
    assert c.effective_batch == 32
    assert c.fp16 is True
    assert c.workers == 2
    assert c.seq_max == 96
    assert c.lr == 2e-5
    assert c.epochs == 10


def test_mem_guard_ok(tmp_path):
    mem = tmp_path / "meminfo"
    mem.write_text("MemAvailable:    48000000 kB\n", encoding="utf-8")
    ok, _ = drv.mem_guard(meminfo_path=mem, vram_free_gb=10.0)
    assert ok is True


def test_mem_guard_fails_low_ram(tmp_path):
    mem = tmp_path / "meminfo"
    mem.write_text("MemAvailable:     1000000 kB\n", encoding="utf-8")
    ok, reason = drv.mem_guard(meminfo_path=mem, vram_free_gb=10.0)
    assert ok is False
    assert "RAM" in reason


def test_mem_guard_fails_low_vram(tmp_path):
    mem = tmp_path / "meminfo"
    mem.write_text("MemAvailable:    48000000 kB\n", encoding="utf-8")
    ok, reason = drv.mem_guard(meminfo_path=mem, vram_free_gb=1.0)
    assert ok is False
    assert "VRAM" in reason


def test_smoke_passes_offline(monkeypatch, capsys):
    monkeypatch.setattr(drv, "mem_guard", lambda **kw: (True, ""))
    rc = drv.smoke(drv.TrainConfig())
    assert rc == 0
    out = capsys.readouterr().out
    assert "smoke" in out.lower()


def test_smoke_fails_bad_config(monkeypatch, capsys):
    monkeypatch.setattr(drv, "mem_guard", lambda **kw: (True, ""))
    bad = drv.TrainConfig(micro_batch=0)
    assert drv.smoke(bad) == 2


def test_smoke_fails_kill_switch(monkeypatch, capsys):
    monkeypatch.setattr(drv, "mem_guard", lambda **kw: (False, "KILL-SWITCH: RAM"))
    assert drv.smoke(drv.TrainConfig()) == 3
    assert "KILL-SWITCH" in capsys.readouterr().out


def test_smoke_warns_over_defaults(monkeypatch, capsys):
    monkeypatch.setattr(drv, "mem_guard", lambda **kw: (True, ""))
    assert drv.smoke(drv.TrainConfig(workers=8)) == 0
    assert "WARN" in capsys.readouterr().out


def test_export_onnx_stub_writes_manifest(tmp_path):
    out = drv.export_onnx_stub(tmp_path / "onnx")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["opset"] == 17
    assert manifest["quant"] == "int8-dynamic"
    assert manifest["model"] == drv.FAST_MODEL
    assert manifest["gate"] == {"p99_ms": 15, "batch": 1, "seq": 96}


def test_list_steps_mentions_v1_models(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["train_decision_model.py", "--list-steps"])
    assert drv.main() == 0
    out = capsys.readouterr().out
    assert "Qwen/Qwen3-4B" in out
    assert "videberta-xsmall" in out
