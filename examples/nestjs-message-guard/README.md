# NestJS message guard example

Backend use case: block toxic chat messages before they reach other users.

## Modes

- `OFFLINE_LITE` (default): fetch `data/*.jsonl` via GitHub raw, run L0-L2 lite
  + heuristic 5-way in TypeScript. No Python server needed.
- `REMOTE_API`: `POST /moderate` to a running `apps/core` API
  (`uv run uvicorn core.api.server:app --port 8000`). Covers full L0-L3
  including ViGuard-X shadow notes.

## Run

```bash
bun install
bun run demo
MODERATION_MODE=REMOTE_API bun run demo
```

Demo runs 6 cases x 3 policies (`standard_social`, `kid_safe`, `relaxed_gaming`)
and prints lexicon/rules/whitelist counts plus `l3_vi` + benchmarks samples,
so all 4 data slices are touched.

## Drop into NestJS

Copy `src/moderation.ts` + `src/message-guard.service.ts` into your app,
provide `MessageGuardService`, call `canPass(text, policy)` in a guard.
