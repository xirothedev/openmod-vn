# API reference

FastAPI service in `apps/core/src/core/api/server.py`.

## Run

```bash
cd apps/core
uv sync --extra test
uv run uvicorn core.api.server:app --port 8000
```

## Endpoints

### `GET /health`

```json
{ "ok": true }
```

### `POST /moderate`

Request:

```json
{
  "text": "Đi chăn rau khu Cầu Giấy, cho xin số điện thoại em nhé?",
  "policy": "standard_social"
}
```

- `text` (string, required): input message.
- `policy` (string, optional): `kid_safe` | `standard_social` (default) | `relaxed_gaming`.

Response:

```json
{
  "action": "WARN",
  "overall_severity": 2,
  "confidence": 0.8,
  "detected_spans": [
    {
      "start": 3,
      "end": 11,
      "text": "chăn rau",
      "category": "SEXUAL_INNUENDO",
      "is_compositional": true,
      "explanation": "compositional comp_vi_00001 suspicion=0.65"
    }
  ],
  "masked_text": "Đi *** khu Cầu Giấy, cho xin số điện thoại em nhé?",
  "rationale": "1 span(s), max severity 2"
}
```

## Decision shape

| Field | Type | Notes |
| ----- | ---- | ----- |
| `action` | string | `ALLOW \| MASK \| WARN \| BLOCK \| ESCALATE` |
| `overall_severity` | int 0-4 | Max severity across spans |
| `confidence` | float 0-1 | Decision confidence |
| `detected_spans` | array | Span objects with offsets, category, compositional flag |
| `masked_text` | string | Text with toxic spans masked |
| `rationale` | string | Human-readable reason |

## Span object

| Field | Type |
| ----- | ---- |
| `start`, `end` | int | Offsets into original text |
| `text` | string | Matched span |
| `category` | string | E.g. `EXPLICIT_PROFANITY`, `SEXUAL_INNUENDO` |
| `is_compositional` | bool | True for L2 spans |
| `explanation` | string | Rule id + suspicion score |

## TypeScript clients

- `examples/discord-word-filter` — offline bundle loader + remote API client.
- `examples/nestjs-message-guard` — NestJS guard calling `/moderate`.

Both support `MODERATION_MODE=REMOTE_API` and `OFFLINE_LITE`.
