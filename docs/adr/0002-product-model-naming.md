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

- HF release (when cut): card `CompoxGuard-vi`, dataset keeps `openmod-vn`.
- If upstream adds a license tag, re-evaluate videberta as the gated sync fallback.
