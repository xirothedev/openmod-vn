#!/usr/bin/env python3
"""Export jsonl to csv/sqlite/json (light, stdlib-first; parquet needs pyarrow, deferred)."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _col_type(values: list) -> str:
    kinds = set()
    for v in values:
        if v is None:
            continue
        if isinstance(v, (bool, int)):
            kinds.add("int")
        elif isinstance(v, float):
            kinds.add("real")
        else:
            kinds.add("text")
    if kinds == {"int"}:
        return "INTEGER"
    if kinds <= {"int", "real"}:
        return "REAL"
    return "TEXT"


def _cell(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return v


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
        # Flat JSON array: direct download use, no sqlite client needed.
        # JS: const rows = await (await fetch(URL)).json()
        jsonp = out / f"{str(rel).replace('/', '_')}.json"
        jsonp.write_text(
            json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        dbp = out / "openmod.db"
        con = sqlite3.connect(dbp)
        tbl = str(rel).replace("/", "_")
        cols = sorted({k for r in rows for k in r})
        coltypes = {c: _col_type([r.get(c) for r in rows]) for c in cols}
        con.execute(f"DROP TABLE IF EXISTS [{tbl}]")
        con.execute(
            f"CREATE TABLE [{tbl}] ({', '.join(f'[{c}] {coltypes[c]}' for c in cols)})"
        )
        for r in rows:
            con.execute(
                f"INSERT INTO [{tbl}] VALUES ({', '.join('?' for _ in cols)})",
                [_cell(r.get(c)) for c in cols],
            )
        con.commit()
        con.close()
    print("OK dist/")


if __name__ == "__main__":
    main()
