# Contributing

## Contrastive PR rule

Every new `data/compositional/` rule must include 2+ benign and 2+ toxic examples using the same tokens. CI fails without them (`scripts/validate_datasets.py`).

## Citations

Every `data/**/*.jsonl` record needs `source`, `source_url`, `license`, `retrieved_at`. Add the source to `data/SOURCES.md`.
