# Architecture (L0-L3)

openmod-vn is a four-layer moderation pipeline. Each layer narrows the decision and adds evidence.

## Data flow

```
text
  -> L0 normalize (NFC, teencode, leetspeak)
  -> L1 lexical match (Aho-Corasick, variants, Scunthorpe defense)
  -> L2 compositional detection (skip-gram rules, suspicion score)
  -> L3 decision (heuristic context judge + optional encoder)
  -> ModerationDecision { action, severity, spans, rationale }
```

## L0 — Normalizer

- Unicode NFC first.
- Teencode mapping (`data/normalization/teencode_mapping.json`).
- Leetspeak / homoglyph mapping (`data/normalization/leetspeak_homoglyphs.json`).
- Tracks alignment so spans map back to the original text.

## L1 — Lexical matcher

- Aho-Corasick over `data/raw_lexicons/vi/*.jsonl`.
- Variant spellings and confusable characters.
- Scunthorpe defense: whitelist check stops substring matches inside clean words (e.g. "bún").
- Output: hits with term, category, original span.

## L2 — Compositional detector

- 13 skip-gram rules in `data/compositional/vi_compositional_phrases.jsonl`.
- Each rule carries tokens, suspicion score, category, and contrastive pairs.
- Positive context lowers suspicion; toxic context raises it.
- Threshold (`>=`) triggers L3.
- Output: candidate spans with normalized + original span.

## L3 — Decision engine

- `HeuristicContextJudge` combines hits + candidates + whitelist + policy.
- Actions: `ALLOW | MASK | WARN | BLOCK | ESCALATE`, severity 0-4, category, spans, rationale.
- Optional `TransformerSpanJudge` (ViGuard-X, xlm-roberta-base, int8) in two modes:
  - **shadow** (default): annotate rationale, never change behavior.
  - **veto**: drop heuristic spans the encoder clears; flag text the heuristic missed.
- Encoder is advisory only; heuristic stands on error.

## Policy profiles

| Profile | Use |
| ------- | --- |
| `kid_safe` | Strictest thresholds; escalates explicit content |
| `standard_social` | Default for social feeds |
| `relaxed_gaming` | Lenient; masks instead of blocks |

## Key files

- `apps/core/src/core/pipeline.py` — `CorePipeline.moderate()`
- `apps/core/src/core/normalizer/`
- `apps/core/src/core/matcher/`
- `apps/core/src/core/compositional/`
- `apps/core/src/core/decision/`
- `apps/core/src/core/api/server.py` — FastAPI `/moderate`, `/health`

## ADRs

- [ADR-0001 single apps/core](../adr/0001-single-apps-core.md)
- [ADR-0002 naming & license stance](../adr/0002-product-model-naming.md)
