"""Step 1: profile the raw export before any filtering.

Outputs value_counts of 'Company response to consumer' by year (all rows and rows
with a narrative), the Date received range, and file metadata used to judge the
export date. Nothing is filtered or modelled here.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from triage.io import COL_DATE, COL_ID, COL_NARRATIVE, COL_RESPONSE, iter_chunks, parse_dates, read_header

MISSING = "<missing>"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True, help="path to complaints.csv or complaints.csv.zip")
    ap.add_argument("--out", default="reports/tables")
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--download-date", default=None, help="date you downloaded the file, YYYY-MM-DD")
    ap.add_argument("--last-modified", default=None, help="Last-Modified header from curl -sI, pasted as text")
    args = ap.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    counts_all: list[pd.Series] = []
    counts_narr: list[pd.Series] = []
    n_rows = 0
    n_narr = 0
    n_bad_date = 0
    id_parts: list[pd.Series] = []
    min_date = None
    max_date = None
    max_date_narr = None

    header = read_header(raw)
    has_narr_col = COL_NARRATIVE in header
    if not has_narr_col:
        print(f"WARNING: '{COL_NARRATIVE}' is not in this export. Narrative tables are skipped.", flush=True)
    usecols = [COL_DATE, COL_RESPONSE, COL_ID] + ([COL_NARRATIVE] if has_narr_col else [])
    for i, chunk in enumerate(iter_chunks(raw, usecols, args.chunksize)):
        dates = parse_dates(chunk[COL_DATE])
        n_bad_date += int(dates.isna().sum())
        year = dates.dt.year.astype("Int64").astype(str).replace("<NA>", "unparsed")
        resp = chunk[COL_RESPONSE].fillna(MISSING)
        if has_narr_col:
            has_narr = chunk[COL_NARRATIVE].fillna("").str.strip().str.len() > 0
        else:
            has_narr = pd.Series(False, index=chunk.index)

        counts_all.append(pd.crosstab(year, resp).stack())
        if has_narr_col:
            counts_narr.append(pd.crosstab(year[has_narr], resp[has_narr]).stack())

        id_parts.append(chunk[COL_ID].dropna())

        n_rows += len(chunk)
        n_narr += int(has_narr.sum())
        cmin, cmax = dates.min(), dates.max()
        if pd.notna(cmin):
            min_date = cmin if min_date is None else min(min_date, cmin)
            max_date = cmax if max_date is None else max(max_date, cmax)
        cmax_n = dates[has_narr].max()
        if pd.notna(cmax_n):
            max_date_narr = cmax_n if max_date_narr is None else max(max_date_narr, cmax_n)
        print(f"chunk {i}: {n_rows:,} rows read", flush=True)

    n_dup_id = int(pd.concat(id_parts, ignore_index=True).duplicated().sum())

    def to_table(parts: list[pd.Series]) -> pd.DataFrame:
        s = pd.concat(parts).groupby(level=[0, 1]).sum()
        t = s.unstack(fill_value=0)
        t.index.name = "year"
        t.columns.name = None
        t = t.sort_index()
        t.loc["total"] = t.sum()
        return t.T  # rows = response category, columns = year

    t_all = to_table(counts_all)
    t_all.to_csv(out / "01_response_by_year_all.csv")
    # Column shares within each year, easier to read than raw counts.
    (t_all / t_all.sum()).round(4).to_csv(out / "01_response_share_by_year_all.csv")
    t_narr = None
    if has_narr_col and counts_narr:
        t_narr = to_table(counts_narr)
        t_narr.to_csv(out / "01_response_by_year_narrative.csv")
        (t_narr / t_narr.sum()).round(4).to_csv(out / "01_response_share_by_year_narrative.csv")

    stat = raw.stat()
    profile = {
        "raw_file": str(raw),
        "file_size_bytes": stat.st_size,
        "file_mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "download_date_reported": args.download_date,
        "last_modified_header": args.last_modified,
        "n_rows": n_rows,
        "columns_in_export": header,
        "narrative_column_present": has_narr_col,
        "n_rows_with_narrative": n_narr if has_narr_col else None,
        "n_unparsed_dates": n_bad_date,
        "n_duplicate_complaint_ids": n_dup_id,
        "date_received_min": None if min_date is None else str(min_date.date()),
        "date_received_max": None if max_date is None else str(max_date.date()),
        "date_received_max_with_narrative": None if max_date_narr is None else str(max_date_narr.date()),
        "response_categories": sorted(t_all.index.tolist()),
    }
    (out / "01_raw_profile.json").write_text(json.dumps(profile, indent=2))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print("\n=== Company response to consumer by year, all rows ===")
    print(t_all.to_string())
    if t_narr is not None:
        print("\n=== Same, rows with narrative ===")
        print(t_narr.to_string())
    print("\n=== Profile ===")
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
