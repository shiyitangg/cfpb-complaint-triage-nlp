"""Step 1b: take narratives from an archived CFPB export and labels from the current official export.

The archive supplies Complaint ID, Date received and the narrative. The official export
supplies the response used for the label and the intake fields. Rows are joined on
Complaint ID and the match rate is reported by year. Only rows inside
[--start-date, --end-date] are kept to bound memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from triage.io import COL_DATE, COL_ID, COL_NARRATIVE, COL_RESPONSE, iter_chunks, parse_dates

OFFICIAL_COLS = [COL_ID, COL_DATE, COL_RESPONSE, "Product", "Sub-product", "Issue", "Sub-issue", "Company"]


def sha256(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(block):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", required=True, help="archived export with narratives (.csv or .csv.zip)")
    ap.add_argument("--archive-url", required=True, help="source URL of the archive, recorded in the profile")
    ap.add_argument("--official", required=True, help="current official export (.csv or .csv.zip)")
    ap.add_argument("--start-date", default="2018-01-01")
    ap.add_argument("--end-date", default="2023-12-31")
    ap.add_argument("--out", default="data/interim/matched.parquet")
    ap.add_argument("--tables", default="reports/tables")
    ap.add_argument("--chunksize", type=int, default=200_000)
    args = ap.parse_args()

    archive, official = Path(args.archive), Path(args.official)
    tables = Path(args.tables)
    tables.mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    start, end = pd.Timestamp(args.start_date), pd.Timestamp(args.end_date)

    # 1. Archive: keep rows with a narrative inside the window.
    parts = []
    n_arch_rows = n_arch_narr = 0
    arch_min = arch_max = None
    for i, c in enumerate(iter_chunks(archive, [COL_ID, COL_DATE, COL_NARRATIVE], args.chunksize)):
        d = parse_dates(c[COL_DATE])
        has = c[COL_NARRATIVE].fillna("").str.strip().str.len() > 0
        n_arch_rows += len(c)
        n_arch_narr += int(has.sum())
        if d[has].notna().any():
            lo, hi = d[has].min(), d[has].max()
            arch_min = lo if arch_min is None else min(arch_min, lo)
            arch_max = hi if arch_max is None else max(arch_max, hi)
        keep = has & (d >= start) & (d <= end)
        parts.append(pd.DataFrame({COL_ID: c.loc[keep, COL_ID], "date_archive": d[keep], "narrative": c.loc[keep, COL_NARRATIVE]}))
        print(f"archive chunk {i}: {n_arch_rows:,} rows", flush=True)
    arch = pd.concat(parts, ignore_index=True)
    n_dup_arch = int(arch[COL_ID].duplicated().sum())
    arch = arch.drop_duplicates(COL_ID, keep="first")
    # A pd.Index keeps its hash table between lookups; Series.isin(set) rebuilds it every chunk.
    ids = pd.Index(arch[COL_ID])

    # 2. Official: stream and keep only IDs present in the archive window.
    parts = []
    n_off_rows = 0
    for i, c in enumerate(iter_chunks(official, OFFICIAL_COLS, args.chunksize)):
        n_off_rows += len(c)
        parts.append(c[ids.get_indexer(c[COL_ID]) >= 0])
        print(f"official chunk {i}: {n_off_rows:,} rows", flush=True)
    off = pd.concat(parts, ignore_index=True).drop_duplicates(COL_ID, keep="first")
    off["date_official"] = parse_dates(off[COL_DATE])
    off = off.drop(columns=[COL_DATE])

    # 3. Join and report.
    m = arch.merge(off, on=COL_ID, how="left", indicator=True)
    m["matched"] = m.pop("_merge") == "both"
    m["date_mismatch"] = m["matched"] & (m["date_archive"].dt.normalize() != m["date_official"].dt.normalize())
    m["year"] = m["date_archive"].dt.year

    by_year = m.groupby("year").agg(
        archive_narratives=("matched", "size"),
        matched=("matched", "sum"),
        date_mismatch=("date_mismatch", "sum"),
    )
    by_year["match_rate"] = (by_year["matched"] / by_year["archive_narratives"]).round(4)
    by_year.loc["total"] = by_year[["archive_narratives", "matched", "date_mismatch"]].sum().tolist() + [
        round(m["matched"].mean(), 4)
    ]
    int_cols = ["archive_narratives", "matched", "date_mismatch"]
    by_year[int_cols] = by_year[int_cols].astype(int)
    by_year.to_csv(tables / "01b_archive_match_by_year.csv")

    resp_by_year = pd.crosstab(m.loc[m["matched"], COL_RESPONSE].fillna("<missing>"), m.loc[m["matched"], "year"], margins=True, margins_name="total")
    resp_by_year.to_csv(tables / "01b_response_by_year_matched.csv")

    m[m["matched"]].drop(columns=["matched"]).to_parquet(args.out, index=False, compression="zstd")

    profile = {
        "archive_file": str(archive),
        "archive_url": args.archive_url,
        "archive_sha256": sha256(archive),
        "archive_rows": n_arch_rows,
        "archive_rows_with_narrative": n_arch_narr,
        "archive_narrative_date_min": None if arch_min is None else str(arch_min.date()),
        "archive_narrative_date_max": None if arch_max is None else str(arch_max.date()),
        "archive_duplicate_ids_in_window": n_dup_arch,
        "official_file": str(official),
        "official_rows": n_off_rows,
        "window": [args.start_date, args.end_date],
        "archive_narratives_in_window": int(len(m)),
        "matched": int(m["matched"].sum()),
        "unmatched": int((~m["matched"]).sum()),
        "match_rate": round(float(m["matched"].mean()), 6) if len(m) else None,
        "date_mismatches_among_matched": int(m["date_mismatch"].sum()),
        "output": args.out,
    }
    (tables / "01b_archive_profile.json").write_text(json.dumps(profile, indent=2))

    pd.set_option("display.width", 250)
    print("\n=== Match rate by year ===")
    print(by_year.to_string())
    print("\n=== Company response to consumer by year, matched rows ===")
    print(resp_by_year.to_string())
    print("\n=== Profile ===")
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
