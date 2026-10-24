"""Step 2: build the cleaned, labelled dataset from the matched archive + official rows.

Each filter is logged in a funnel table. Unknown response categories stop the script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from triage.labels import EXCLUDED, make_label
from triage.text import clean_text

KEEP = {
    "Complaint ID": "complaint_id",
    "date_official": "date_received",
    "text": "text",
    "Product": "product",
    "Sub-product": "sub_product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "Company": "company",  # ablation only
    "label": "label",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="data/interim/matched.parquet")
    ap.add_argument("--start-date", default="2018-01-01")
    ap.add_argument("--end-date", default="2023-12-31")
    ap.add_argument("--min-chars", type=int, default=50, help="minimum length after removing masks")
    ap.add_argument("--out", default="data/interim/complaints_clean.parquet")
    ap.add_argument("--tables", default="reports/tables")
    args = ap.parse_args()
    tables = Path(args.tables)
    tables.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.inp)
    n0 = len(df)
    funnel = [("matched input", n0)]

    day = df["date_official"].dt.normalize()
    df = df[(day >= pd.Timestamp(args.start_date)) & (day <= pd.Timestamp(args.end_date))]
    funnel.append((f"Date received in [{args.start_date}, {args.end_date}]", len(df)))

    resp = df["Company response to consumer"]
    df = df.assign(label=make_label(resp), year=df["date_official"].dt.year)

    # Unresolved / unknown outcome by year, before dropping them.
    resp = df["Company response to consumer"]
    status = resp.fillna("<missing>").where(resp.isna() | resp.isin(EXCLUDED), "resolved")
    unresolved = pd.crosstab(df["year"], status, margins=True, margins_name="total")
    for c in [*EXCLUDED, "<missing>"]:
        if c not in unresolved.columns:
            unresolved[c] = 0
    unresolved["excluded"] = unresolved[[*EXCLUDED, "<missing>"]].sum(axis=1)
    unresolved["excluded_share"] = (unresolved["excluded"] / unresolved["total"]).round(6)
    unresolved.to_csv(tables / "02_unresolved_by_year.csv")

    df = df[df["label"].notna()]
    funnel.append(("label known (drop In progress, Untimely response, missing)", len(df)))

    df = df.assign(text=clean_text(df["narrative"]))
    df = df[df["text"].str.len() >= args.min_chars]
    funnel.append((f"text >= {args.min_chars} chars after removing masks", len(df)))

    df = df.sort_values(["date_official", "Complaint ID"]).drop_duplicates("Complaint ID", keep="first")
    funnel.append(("unique Complaint ID", len(df)))

    out = df[list(KEEP)].rename(columns=KEEP)
    out["label"] = out["label"].astype("int8")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False, compression="zstd")

    f = pd.DataFrame(funnel, columns=["step", "rows"])
    f["dropped"] = (-f["rows"].diff()).fillna(0).astype(int)
    f["dropped_share_of_previous"] = (f["dropped"] / f["rows"].shift(1)).round(6)
    f["remaining_share_of_input"] = (f["rows"] / n0).round(6)
    f.to_csv(tables / "02_filter_funnel.csv", index=False)

    by_year = out.assign(year=out["date_received"].dt.year).groupby("year")["label"].agg(n="size", positives="sum")
    by_year["positive_rate"] = (by_year["positives"] / by_year["n"]).round(6)
    by_year.to_csv(tables / "02_label_by_year.csv")

    pd.set_option("display.width", 250)
    print("\n=== Filter funnel ===")
    print(f.to_string(index=False))
    print("\n=== Excluded (unknown outcome) by year ===")
    print(unresolved.to_string())
    print("\n=== Label by year after cleaning ===")
    print(by_year.to_string())
    print(json.dumps({"output": args.out, "rows": len(out), "columns": list(out.columns)}, indent=2))


if __name__ == "__main__":
    main()
