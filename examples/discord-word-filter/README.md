# Discord word filter example

Discord use case: delete toxic messages and reply with a warning.

## Modes

- `OFFLINE_LITE` (default): fetch `data/*.jsonl` via GitHub raw, filter in
  TypeScript. No Python server needed.
- `REMOTE_API`: `POST /moderate` to a running `apps/core` API for full L0-L3
  including ViGuard-X notes.

## Run

```bash
npm install -g tsx
tsx src/demo.ts
MODERATION_MODE=REMOTE_API tsx src/demo.ts
DISCORD_TOKEN=xxx tsx src/bot.ts
```

Demo runs 6 cases x 3 policies and prints lexicon/rules counts plus `l3_vi`
+ benchmarks samples, so all 4 data slices are touched.
