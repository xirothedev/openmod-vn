# ADR-0001: Single apps/core for v0, split later

Date: 2026-10-27. Status: accepted.

## Context

Monorepo with one Python engine (L0-L3) plus FastAPI. No second consumer yet (no NestJS wrapper, no JS SDK).

## Decision

Ship single `apps/core` (`import core`). Defer `packages/core` lib + `apps/api` service split to v1 when a second app arrives.

## Consequences

- Keep JSON I/O boundary (`POST /moderate`) stable so the split needs no logic rewrite.
- Revisit when NestJS sidecar or batch worker lands.
