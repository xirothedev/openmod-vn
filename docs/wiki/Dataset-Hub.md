# Dataset hub

All data lives under `data/`. License: CC-BY-SA-4.0 for derived content; source licenses vary (see `data/SOURCES.md`).

## Layout

```
data/
  SOURCES.md                 # citations + row counts per slice
  normalization/             # teencode, leetspeak maps
  raw_lexicons/vi/           # explicit profanity, offensive words, whitelist
  compositional/             # vi_compositional_phrases.jsonl (13 rules)
  l3_vi.jsonl                # L3 training samples (9918 rows)
  classifier_vi.jsonl        # binary CLEAN|TOXIC (14747 rows)
  benchmarks/                # ViHOS spans, HateCheck-VI, functional suite
  schemas/                   # JSON schemas for each slice
  _staging/                  # raw downloads (not committed as final)
```

## Slices

| Slice | File | Rows | Purpose |
| ----- | ---- | ---- | ------- |
| L3 samples | `data/l3_vi.jsonl` | 9918 | Phrase + left/right context + label (BLOCK/BORDERLINE/ALLOW) + spans |
| Classifier | `data/classifier_vi.jsonl` | 14747 | Binary CLEAN/TOXIC + severity for full-text head |
| Compositional rules | `data/compositional/vi_compositional_phrases.jsonl` | 13 | Skip-gram rules with contrastive pairs |
| Lexicons | `data/raw_lexicons/vi/*.jsonl` | 435+ | Explicit profanity, context-dependent words, whitelist |
| Benchmarks | `data/benchmarks/*.jsonl` | — | ViHOS spans, HateCheck-VI templates, functional suite |

## Validation

```bash
python scripts/validate_datasets.py
```

Checks schemas, citations, and contrastive-pair token consistency.

## Releases

`scripts/build_releases.py` produces downloadable SQLite DB + flat JSON per slice. Attached to GitHub releases (`dist/`).

## Sources

See `data/SOURCES.md` for per-source citations, licenses, and row counts. Sources include ViHOS, ViHSD, HateCheck-VI templates, and imported offensive-word lists.

## Adding data

1. Stage raw files under `data/_staging/`.
2. Transform with scripts in `scripts/` (download_datasets, build_l3_vi, build_classifier_data).
3. Human review (batch reviews in `docs/review/`).
4. Validate and commit.
