# ADR-0002: Product/model naming + videberta license stance

Date: 2026-10-28. Status: accepted.

## Context

Repo `openmod-vn`, package `core`. Shipped v1 encoder is xlm-roberta-base (MIT);
videberta-xsmall is eval-only (HF weights declare no license; upstream repo LICENSE is MIT
for code). Earlier candidates: VinaShield, CompoxGuard, LexiJudge, VnToxicGraph, OpenModVN.

## Decision

- Ecosystem (repo, dataset, SDK): keep `openmod-vn` / OpenModVN. Descriptive,
  international, already released (v0.1.0).
- Model card name: `ViGuard-X` (X = XLM-R base). Short, international, architecture-hinting;
  Vietnamese alternatives considered (Chắn/Gác/Lọc — strong but risk tone issues in EN contexts),
  CompoxGuard-vi kept as product-line name if a compositional-focused sibling lands.
- Videberta weights: stay eval-only. No prod use until Fsoft-AIC declares a weights
  license on the HF card. Revisit only if videberta becomes the primary encoder.

## Consequences

- HF release (when cut): card `ViGuard-X`, dataset keeps `openmod-vn`.
- If upstream adds a license tag, re-evaluate videberta as the gated sync fallback.

## v1 weights (ViGuard-X, xlm-roberta-base finetune)

GitHub release `v0.2.0` (MF1 0.8214, int8 p99 15.84ms CPU b1/s96):

- https://github.com/xirothedev/openmod-vn/releases/download/v0.2.0/model.int8.onnx
- https://github.com/xirothedev/openmod-vn/releases/download/v0.2.0/tokenizer.json
- https://github.com/xirothedev/openmod-vn/releases/download/v0.2.0/metrics.json

Runtime: `TransformerSpanJudge(model_dir)` with both files; shadow sidecar by default.
