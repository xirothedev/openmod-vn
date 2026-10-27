#!/usr/bin/env python3
"""Validate data schemas, duplicate IDs, citations, contrastive pairs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load(p: Path) -> list[dict]:
    out = []
    with open(p, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"FAIL {p}:{i} bad json: {e}")
                    raise SystemExit(1)
    return out


def need(d: dict, fields: list[str], where: str) -> None:
    for k in fields:
        if not d.get(k):
            print(f"FAIL {where} {d.get('id', '?')} missing {k}")
            raise SystemExit(1)


def main() -> None:
    seen: set[str] = set()
    files = list((DATA).rglob("*.jsonl"))
    if not files:
        print("FAIL no jsonl found")
        raise SystemExit(1)
    for p in files:
        for d in load(p):
            i = d.get("id", str(p))
            if i in seen:
                print(f"FAIL duplicate id {i}")
                raise SystemExit(1)
            seen.add(i)
            need(d, ["id", "source", "source_url", "license", "retrieved_at"], str(p))
            if "tokens" in d:  # compositional
                need(
                    d,
                    ["contrastive_examples", "literal_meaning", "toxic_meaning"],
                    str(p),
                )
                ce = d["contrastive_examples"]
                if len(ce.get("benign", [])) < 2 or len(ce.get("toxic", [])) < 2:
                    print(f"FAIL {i} needs 2+2 contrastive examples")
                    raise SystemExit(1)
                toks = [str(t).lower() for t in d.get("tokens", [])]
                for side in ("benign", "toxic"):
                    for ex in ce.get(side, []):
                        low = str(ex).lower()
                        missing = [t for t in toks if t not in low]
                        if missing:
                            print(
                                f"FAIL {i} {side} example misses tokens {missing}: {ex[:60]}"
                            )
                            raise SystemExit(1)
                if d.get("individual_tokens_clean") is not True:
                    print(f"FAIL {i} individual_tokens_clean must be true")
                    raise SystemExit(1)
    print(f"OK {len(files)} files, {len(seen)} records")


if __name__ == "__main__":
    sys.exit(main())
