# Discord word filter example

Discord use case: delete toxic messages and reply with a warning.

## Modes

- `OFFLINE_LITE` (default): fetch `data/*.jsonl` via GitHub raw, filter in
  TypeScript. No Python server needed.
- `REMOTE_API`: `POST /moderate` to a running `apps/core` API for full L0-L3
  including ViGuard-X notes.

## Run

```bash
bun install
bun run demo
MODERATION_MODE=REMOTE_API bun run demo
DISCORD_TOKEN=xxx bun run bot
```

Demo runs 6 cases x 3 policies and prints lexicon/rules counts plus `l3_vi`
+ benchmarks samples, so all 4 data slices are touched.
