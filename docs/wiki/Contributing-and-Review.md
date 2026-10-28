# Contributing & review

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md). General flow:

1. Fork / branch.
2. Run tests: `cd apps/core && uv run pytest -q`.
3. Validate data: `python scripts/validate_datasets.py`.
4. Lint: `ruff check .` (line length 100).
5. Open PR with a clear description.

## Lexicon review

Lexicon changes (promotions, new terms, whitelist) go through review batches:

- `docs/review/review_batch_01.md` — items 1-30.
- `docs/review/review_batch_02.md` — items 31-60.
- `docs/review/review_batch_03_teacher_labels.md` — teacher labels.

Each batch lists candidate terms with context and a decision (promote / keep context-dependent / whitelist).

## Adding a compositional rule

Rules live in `data/compositional/vi_compositional_phrases.jsonl`. Each rule requires:

- Tokens (skip-gram).
- Suspicion base score.
- Category.
- Contrastive pairs: 2 benign + 2 toxic using the same tokens.
- `scripts/validate_datasets.py` enforces token consistency across contrastive examples.

## ADRs

Architecture decisions are recorded in `docs/adr/`:

- `0001-single-apps-core.md` — single app now, split later.
- `0002-product-model-naming.md` — ViGuard-X name + videberta license stance.

Add a new ADR when a decision is hard to reverse, surprising without context, and the result of a real trade-off.

## Domain language

See `GLOSSARY.md` for ubiquitous terms (explicit profanity, compositional toxicity, candidate span, etc.). Keep implementation details out of the glossary.

## Code of conduct

Be respectful. This project handles toxic content; contributors should handle examples with care and avoid amplifying harm.
