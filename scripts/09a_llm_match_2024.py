"""LLM step 0: match a random sample of 2024 archive narratives to the official export.

The LLM module needs only about 50 items from 2024. Step 1b with a 2024 window was killed
for memory on the laptop, so this script reads both exports row by row with the csv module
(no pandas chunks) and keeps a fixed random share of 2024 Complaint IDs, chosen by a hash of
the ID so the sample does not depend on file order. Output has the same columns as
data/interim/matched.parquet. Original data and tables are not touched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from datetime import date
from pathlib import Path
from typing import Iterator

import pandas as pd

from triage.io import COL_DATE, COL_ID, COL_NARRATIVE, COL_RESPONSE

OFFICIAL_COLS = [COL_ID, COL_DATE, COL_RESPONSE, "Product", "Sub-product", "Issue", "Sub-issue", "Company"]
csv.field_size_limit(sys.maxsize)


def rows(path: str) -> Iterator[dict]:
    """Stream rows of a .csv or a .csv.zip export as dicts."""
    p = Path(path)
    if p.suffix == ".zip":
        z = zipfile.ZipFile(p)
        member = next(n for n in z.namelist() if n.endswith(".csv"))
        f = io.TextIOWrapper(z.open(member), encoding="utf-8-sig", newline="")
    else:
        f = open(p, encoding="utf-8-sig", newline="")
    with f:
        yield from csv.DictReader(f)


def parse_day(s: str) -> date | None:
    """Exports use YYYY-MM-DD or MM/DD/YY (or MM/DD/YYYY)."""
    s = (s or "").strip()[:10]
    try:
        if len(s) >= 10 and s[4] == "-":
            return date(int(s[:4]), int(s[5:7]), int(s[8:10]))
        m, d, y = s.split("/")[:3]
        y = int(y[:4])
        return date(y + 2000 if y < 100 else y, int(m), int(d))
    except (ValueError, IndexError):
        return None


def in_sample(complaint_id: str, share: float) -> bool:
    return int(hashlib.sha256(complaint_id.encode()).hexdigest()[:8], 16) < int(share * 0xFFFFFFFF)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", default="data/archive/complaints_20251209.csv.zip")
    ap.add_argument("--official", default="data/raw/complaints.csv.zip")
    ap.add_argument("--start-date", default="2024-01-01")
    ap.add_argument("--end-date", default="2024-12-31")
    ap.add_argument("--share", type=float, default=0.05)
    ap.add_argument("--out", default="data/interim/matched_2024.parquet")
    args = ap.parse_args()
    start, end = date.fromisoformat(args.start_date), date.fromisoformat(args.end_date)

    arch, n_rows, n_window = {}, 0, 0
    for r in rows(args.archive):
        n_rows += 1
        if n_rows % 1_000_000 == 0:
            print(f"archive: {n_rows:,} rows read, {n_window:,} in window, {len(arch):,} sampled", flush=True)
        text = r.get(COL_NARRATIVE) or ""
        if not text.strip():
            continue
        d = parse_day(r[COL_DATE])
        if d is None or not (start <= d <= end):
            continue
        n_window += 1
        cid = r[COL_ID].strip()
        if in_sample(cid, args.share) and cid not in arch:
            arch[cid] = (d, text)
    print(f"archive done: {n_rows:,} rows, {n_window:,} narratives in window, {len(arch):,} sampled", flush=True)

    off, n_off = {}, 0
    for r in rows(args.official):
        n_off += 1
        if n_off % 2_000_000 == 0:
            print(f"official: {n_off:,} rows read, {len(off):,} matched", flush=True)
        cid = r[COL_ID].strip()
        if cid in arch and cid not in off:
            off[cid] = {c: (r.get(c) or None) for c in OFFICIAL_COLS}

    m = pd.DataFrame([{COL_ID: cid, "date_archive": pd.Timestamp(arch[cid][0]), "narrative": arch[cid][1],
                       **{k: v for k, v in o.items() if k != COL_DATE},
                       "date_official": pd.to_datetime(o[COL_DATE], errors="coerce", format="mixed")}
                      for cid, o in off.items()])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(args.out, index=False, compression="zstd")
    print(json.dumps({"archive_rows": n_rows, "archive_narratives_in_window": n_window, "sampled": len(arch),
                      "share": args.share, "official_rows": n_off, "matched": len(m),
                      "match_rate": round(len(m) / len(arch), 6) if arch else None,
                      "date_mismatches": int((m["date_archive"] != m["date_official"].dt.normalize()).sum()),
                      "output": args.out}, indent=2))


if __name__ == "__main__":
    main()
