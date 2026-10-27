# openmod-vn

Open Vietnamese moderation: dataset hub + compositional toxicity pipeline (L0-L3).

## Quickstart

```bash
cd apps/core
uv sync --extra test
uv run pytest -q
python ../../scripts/validate_datasets.py
uv run uvicorn core.api.server:app --port 8000
curl -X POST localhost:8000/moderate -H 'content-type: application/json' -d '{"text":"Đi chăn rau sạch khu Cầu Giấy tối nay không?"}'
```

## Layout

- `apps/core/src/core/` — L0 normalizer, L1 matcher, L2 compositional, L3 decision, pipeline, FastAPI
- `data/` — CC-BY-SA-4.0 dataset hub with citations (`data/SOURCES.md`)
- `docs/training/decision-model.md` — L3 training plan
- `scripts/` — validate, build releases, train skeleton
```
