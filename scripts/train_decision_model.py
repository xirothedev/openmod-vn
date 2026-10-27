#!/usr/bin/env python3
"""Skeleton fine-tune for L3 fast encoder + Qwen judge (offline-safe stub)."""

from __future__ import annotations


def main() -> None:
    print("TODO wire: load data/benchmarks + ViHOS-teacher labels")
    print(
        "TODO train: Fsoft-AIC/videberta-xsmall 3-way + span head, export ONNX int8, gate p99<15ms"
    )
    print("TODO judge: Qwen2.5-3B-Instruct (or Qwen3-4B) vLLM guided_json LoRA")
    print("See docs/training/decision-model.md")


if __name__ == "__main__":
    main()
