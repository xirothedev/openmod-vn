# Training & eval

## ViGuard-X (shipped, v0.2.0)

- Base: xlm-roberta-base (MIT).
- Task: span-level toxicity judge (BLOCK / BORDERLINE / ALLOW).
- Export: ONNX opset17, int8 dynamic quant.
- Metrics: MF1 0.8214; int8 p99 15.84 ms CPU (batch 1, seq 96).
- Calibration: temperature T=1.5, threshold tau=0.5.
- Weights: GitHub release v0.2.0 (`model.int8.onnx`, `tokenizer.json`, `metrics.json`).
- Runtime: `TransformerSpanJudge(model_dir)`; shadow sidecar by default.

## Training plan

See `docs/training/decision-model.md`. Summary:

1. Teacher-label ViHOS spans + HateCheck-VI with ViHateT5 / ViSoBERT.
2. Human review every teacher label before it enters `data/`.
3. Fine-tune xlm-roberta-base 3-way + span BIO head (AdamW 2e-5, 10 epochs, batch 32).
4. Export ONNX, int8 quant, gate p99 < 15 ms.
5. Optional deep judge (async GPU): Qwen2.5-3B-Instruct with LoRA.
6. Nightly distill judge corrections; weekly refresh.

## Encoder usage modes

- `shadow=True` (default): encoder annotates rationale, never changes decision.
- `shadow=False` + spans present: encoder vetoes cleared spans.
- `shadow=False` + no spans: encoder flags missed toxic text (recall).

## Calibration

Temperature scaling (T=1.5) + threshold (tau=0.5) fit on ViHSD half, reported on the other half.
See commit `d0f7c81` and `docs/training/decision-v1-bvpt3-plan.md`.

## Eval suites

- `data/benchmarks/hatecheck_vi.jsonl` + `hatecheck_vi_templates.jsonl` — functional suite.
- `data/benchmarks/vihos_spans.jsonl` — span-level labels.
- `data/benchmarks/test_functional_suite.jsonl` — regression cases.
- Bias slices: religion, gender, region.

## Benchmarks

```bash
python scripts/bench_pipeline.py
```

Reports hybrid (heuristic + encoder) vs heuristic on the functional suite.

## Filter classifier (planned)

Binary full-text head (`CLEAN | TOXIC` + severity) for block/pass use cases. Maps from encoder label.
Trains on `data/classifier_vi.jsonl` (14747 rows). Ships in shadow first.
See `docs/training/classifier-v1-plan.md` and `scripts/train_classifier.py`.
