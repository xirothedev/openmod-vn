# openmod-vn

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](apps/core/pyproject.toml)
[![Release v0.2.0](https://img.shields.io/github/v/release/xirothedev/openmod-vn)](https://github.com/xirothedev/openmod-vn/releases)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](apps/core/tests)

Open Vietnamese moderation: a **dataset hub** plus a **compositional toxicity pipeline (L0-L3)**.
Built for context-dependent and implicit hate speech in Vietnamese, where benign words combine into toxic meaning.

> "chăn rau" alone is harmless. "Đi chăn rau khu Cầu Giấy, cho xin số điện thoại em nhé?" is not. openmod-vn catches the second one.

## Why

Most filters match single slurs. Vietnamese toxicity is often **compositional**: two clean words
("chăn", "rau") form an innuendo only in context. openmod-vn normalizes, matches, composes, and
decides — with spans, severity, and a rationale — so moderators see *why* a message was flagged.

## Features

- **L0 Normalizer** — NFC-first, teencode + leetspeak homoglyph mapping, alignment tracking.
- **L1 Lexical matcher** — Aho-Corasick over 435+ terms with variant spellings and Scunthorpe defense.
- **L2 Compositional detector** — 13 skip-gram rules with suspicion scores and contrastive pairs.
- **L3 Decision engine** — Heuristic context judge + optional ViGuard-X encoder (shadow or veto).
- **Policy profiles** — `kid_safe`, `standard_social`, `relaxed_gaming` thresholds per surface.
- **Dataset hub** — CC-BY-SA-4.0 sources with citations, schemas, and validation.
- **ViGuard-X** — xlm-roberta-base span judge, int8, p99 15.84 ms CPU (v0.2.0).
- **TypeScript examples** — Discord bot (offline + remote) and NestJS message guard.

## Demo

Run the Discord example in offline mode (no server needed):

```bash
cd examples/discord-word-filter
bun install
bun run demo
```

Sample output:

```
mode=OFFLINE_LITE
lexicon=435 rules=13 whitelist=2

## policy=standard_social
- [ALLOW delete=false] "Cho tôi một bát bún bò Huế." :: ALLOW (whitelist bún)
- [ALLOW delete=false] "Con chó nhà tôi rất ngoan." :: ALLOW (context-dependent, benign)
- [BLOCK delete=true] "Thằng này địt cả lò nhà mày." :: delete+reply (explicit)
- [WARN delete=true] "Đồ con chó, cút khỏi đây." :: delete+reply (aimed at person)
- [ALLOW delete=false] "Nông dân trồng rau sạch ngoài vườn." :: ALLOW (compositional benign)
- [WARN delete=true] "Đi chăn rau khu Cầu Giấy, cho xin số điện thoại em nhé?" :: delete (compositional toxic)
```

Or hit the API directly:

```bash
cd apps/core
uv sync --extra test
uv run uvicorn core.api.server:app --port 8000
curl -s -X POST localhost:8000/moderate \
  -H 'content-type: application/json' \
  -d '{"text":"Đi chăn rau khu Cầu Giấy, cho xin số điện thoại em nhé?","policy":"standard_social"}'
```

```json
{
  "action": "WARN",
  "overall_severity": 2,
  "detected_spans": [
    { "text": "chăn rau", "category": "SEXUAL_INNUENDO", "is_compositional": true }
  ],
  "masked_text": "Đi *** khu Cầu Giấy, cho xin số điện thoại em nhé?"
}
```

## Quickstart

```bash
cd apps/core
uv sync --extra test
uv run pytest -q                    # 40+ tests
python ../../scripts/validate_datasets.py
uv run uvicorn core.api.server:app --port 8000
```

## Layout

- `apps/core/src/core/` — L0 normalizer, L1 matcher, L2 compositional, L3 decision, pipeline, FastAPI
- `data/` — CC-BY-SA-4.0 dataset hub with citations (`data/SOURCES.md`)
- `docs/` — architecture, training plans, ADRs, wiki pages
- `examples/` — Discord bot + NestJS message guard
- `scripts/` — validate, build releases, train skeleton

## Releases

- **v0.2.0** (latest) — ViGuard-X weights (int8 ONNX + tokenizer + metrics), calibration (T=1.5, tau=0.5).
  [Download weights](https://github.com/xirothedev/openmod-vn/releases/tag/v0.2.0)
- **v0.1.0** — scaffold + dataset hub.

## Documentation

- [GitHub Wiki](https://github.com/xirothedev/openmod-vn/wiki) (publish with `scripts/publish-wiki.sh` after first page)
- [Hosted docs (GitHub Pages)](https://xirothedev.github.io/openmod-vn/)
- [Wiki: Architecture (L0-L3)](docs/wiki/Architecture.md)
- [Wiki: API reference](docs/wiki/API-Reference.md)
- [Wiki: Dataset hub](docs/wiki/Dataset-Hub.md)
- [Wiki: Training & eval](docs/wiki/Training-and-Eval.md)
- [Wiki: Contributing & review](docs/wiki/Contributing-and-Review.md)
- [Decision model training](docs/training/decision-model.md)
- [ADR-0001 single apps/core](docs/adr/0001-single-apps-core.md)
- [ADR-0002 naming & license](docs/adr/0002-product-model-naming.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Lexicon changes go through the review batches in `docs/review/`.

## License

MIT. Dataset content is CC-BY-SA-4.0; see `data/SOURCES.md`.
