#!/usr/bin/env python3
"""Export jsonl to csv/sqlite (light, stdlib-first; parquet needs pyarrow, deferred)."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    out = ROOT / "dist"
    out.mkdir(exist_ok=True)
    for p in (ROOT / "data").rglob("*.jsonl"):
        rows = [
            json.loads(l)
            for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        if not rows:
            continue
        rel = p.relative_to(ROOT / "data").with_suffix("")
        csvp = out / f"{str(rel).replace('/', '_')}.csv"
        with open(csvp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)
        dbp = out / "openmod.db"
        con = sqlite3.connect(dbp)
        tbl = str(rel).replace("/", "_")
        cols = sorted({k for r in rows for k in r})
        con.execute(f"DROP TABLE IF EXISTS [{tbl}]")
        con.execute(f"CREATE TABLE [{tbl}] ({', '.join(f'[{c}] TEXT' for c in cols)})")
        for r in rows:
            con.execute(
                f"INSERT INTO [{tbl}] VALUES ({', '.join('?' for _ in cols)})",
                [
                    json.dumps(r.get(c), ensure_ascii=False)
                    if isinstance(r.get(c), (list, dict))
                    else r.get(c)
                    for c in cols
                ],
            )
        con.commit()
        con.close()
    print("OK dist/")


if __name__ == "__main__":
    main()
