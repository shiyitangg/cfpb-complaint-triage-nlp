"""Step 3: time split, exact text dedup, near-duplicate check across splits, split figure.

Exact duplicates (same normalized text) keep only the earliest complaint. Near duplicates
(see triage.dedup) are flagged, not dropped: near_dup_train marks val/test rows that have a
near duplicate in train, near_dup_val marks test rows with one in val. Later steps use the
main setting (drop near_dup_train rows) and report the unfiltered setting as a check.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from triage.dedup import BANDS, NUM_PERM, ROWS, SHINGLE, compute_band_keys, find_near_dups
from triage.split import SplitDates, assign_split
from triage.text import normalize_for_dedup

ORDER = ["train", "val", "test"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="data/interim/complaints_clean.parquet")
    ap.add_argument("--train-start", default="2018-01-01")
    ap.add_argument("--train-end", default="2021-12-31")
    ap.add_argument("--val-start", default="2022-01-01")
    ap.add_argument("--val-end", default="2022-12-31")
    ap.add_argument("--test-start", default="2023-01-01")
    ap.add_argument("--test-end", default="2023-12-31")
    ap.add_argument("--jaccard", type=float, default=0.8)
    ap.add_argument("--max-ref-per-bucket", type=int, default=5)
    ap.add_argument("--n-examples", type=int, default=30)
    ap.add_argument("--workers", type=int, default=None, help="processes for MinHash; default = CPU cores - 1")
    ap.add_argument("--out", default="data/processed/modeling.parquet")
    ap.add_argument("--tables", default="reports/tables")
    ap.add_argument("--figures", default="reports/figures")
    args = ap.parse_args()
    tables, figures = Path(args.tables), Path(args.figures)
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    dates = SplitDates(*(pd.Timestamp(x) for x in (args.train_start, args.train_end, args.val_start,
                                                   args.val_end, args.test_start, args.test_end)))
    df = pd.read_parquet(args.inp)
    df["split"] = assign_split(df["date_received"], dates)
    n_outside = int(df["split"].isna().sum())
    df = df[df["split"].notna()].copy()
    for s in ORDER:
        r = df.loc[df["split"] == s, "date_received"]
        print(f"{s}: {len(r):,} rows, {r.min().date()} .. {r.max().date()}")
    assert df.groupby("split")["date_received"].agg(["min", "max"]).loc[ORDER].pipe(
        lambda t: (t["max"].iloc[:-1].to_numpy() < t["min"].iloc[1:].to_numpy()).all()), "split windows overlap"

    # Exact duplicates on normalized text; keep the earliest.
    df = df.sort_values(["date_received", "complaint_id"]).reset_index(drop=True)
    df["norm"] = normalize_for_dedup(df["text"])
    df["text_hash"] = pd.util.hash_pandas_object(df["norm"], index=False).to_numpy()
    first = df.groupby("text_hash")["split"].transform("first")
    is_dup = df["text_hash"].duplicated(keep="first")
    mixed_label_groups = int((df.groupby("text_hash")["label"].nunique() > 1).sum())
    exact = pd.DataFrame({
        "rows_before": df.groupby("split").size(),
        "exact_dup_removed": df[is_dup].groupby("split").size(),
        "of_which_first_copy_in_earlier_split": df[is_dup & (first != df["split"])].groupby("split").size(),
    }).reindex(ORDER).fillna(0).astype(int)
    df = df[~is_dup].reset_index(drop=True)

    # Near duplicates.
    print("computing MinHash signatures", flush=True)
    keys = compute_band_keys(df["norm"], workers=args.workers)
    split = df["split"].to_numpy()
    print("matching val/test against train", flush=True)
    vs_train = find_near_dups(df["norm"], keys, split == "train", split != "train", args.jaccard, args.max_ref_per_bucket)
    print("matching test against val", flush=True)
    test_vs_val = find_near_dups(df["norm"], keys, split == "val", split == "test", args.jaccard, args.max_ref_per_bucket)
    df["near_dup_train"] = False
    df.loc[vs_train["query_pos"].to_numpy(), "near_dup_train"] = True
    df["near_dup_val"] = False
    df.loc[test_vs_val["query_pos"].to_numpy(), "near_dup_val"] = True

    near = pd.DataFrame({
        "rows_after_exact_dedup": df.groupby("split").size(),
        "near_dup_of_train": df.groupby("split")["near_dup_train"].sum(),
        "near_dup_of_val": df.groupby("split")["near_dup_val"].sum(),
    }).reindex(ORDER).fillna(0).astype(int)
    near["near_dup_of_train_share"] = (near["near_dup_of_train"] / near["rows_after_exact_dedup"]).round(6)
    near["near_dup_of_val_share"] = (near["near_dup_of_val"] / near["rows_after_exact_dedup"]).round(6)
    near["rows_main"] = near["rows_after_exact_dedup"] - near["near_dup_of_train"]
    dedup = exact.join(near)
    dedup.to_csv(tables / "03_dedup_summary.csv")

    # Pair examples with short snippets stay under data/ (git-ignored); the report table keeps ids only.
    ex = pd.concat([vs_train.assign(kind="vs_train"), test_vs_val.assign(kind="test_vs_val")])
    ex = ex.sample(min(args.n_examples, len(ex)), random_state=0) if len(ex) else ex
    ex_ids = pd.DataFrame({
        "kind": ex["kind"].to_numpy(),
        "jaccard": ex["jaccard"].round(4).to_numpy(),
        "query_id": df["complaint_id"].to_numpy()[ex["query_pos"].to_numpy()] if len(ex) else [],
        "query_split": split[ex["query_pos"].to_numpy()] if len(ex) else [],
        "ref_id": df["complaint_id"].to_numpy()[ex["ref_pos"].to_numpy()] if len(ex) else [],
        "ref_split": split[ex["ref_pos"].to_numpy()] if len(ex) else [],
    })
    ex_ids.to_csv(tables / "03_near_duplicate_examples_ids.csv", index=False)
    if len(ex):
        snip = ex_ids.assign(
            query_snippet=df["text"].to_numpy()[ex["query_pos"].to_numpy()],
            ref_snippet=df["text"].to_numpy()[ex["ref_pos"].to_numpy()],
        )
        snip["query_snippet"] = snip["query_snippet"].str[:120]
        snip["ref_snippet"] = snip["ref_snippet"].str[:120]
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        snip.to_csv(Path(args.out).parent / "03_near_duplicate_examples_with_text.csv", index=False)

    # Split summary, both settings.
    rows = []
    for setting, d in [("all_after_exact_dedup", df), ("main_drop_near_dup_train", df[~df["near_dup_train"]])]:
        for s in ORDER:
            x = d[d["split"] == s]
            rows.append({"setting": setting, "split": s, "date_min": x["date_received"].min().date(),
                         "date_max": x["date_received"].max().date(), "rows": len(x),
                         "positives": int(x["label"].sum()), "positive_rate": round(float(x["label"].mean()), 6)})
    summary = pd.DataFrame(rows)
    summary.to_csv(tables / "03_split_summary.csv", index=False)

    # Figure: monthly volume by split and monthly positive rate.
    m = df.assign(month=df["date_received"].dt.to_period("M").dt.to_timestamp())
    vol = m.groupby(["month", "split"]).size().unstack(fill_value=0).reindex(columns=ORDER, fill_value=0)
    rate = m.groupby("month")["label"].mean()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    colors = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
    for s in ORDER:
        a1.bar(vol.index, vol[s], width=25, color=colors[s], label=s)
    a1.set_ylabel("complaints per month")
    a1.legend(frameon=False)
    a1.set_title("Time split on Date received (after cleaning and exact dedup)")
    a2.plot(rate.index, rate.values, color="#333333")
    a2.set_ylabel("monetary-relief rate")
    for x in (dates.val_start, dates.test_start):
        for a in (a1, a2):
            a.axvline(x, color="#999999", ls="--", lw=1)
    fig.tight_layout()
    fig.savefig(figures / "03_time_split.png", dpi=150)

    out = df.drop(columns=["norm"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False, compression="zstd")

    meta = {
        "split_dates": {k: str(v.date()) for k, v in vars(dates).items()},
        "rows_outside_windows": n_outside,
        "exact_dup_groups_with_mixed_labels": mixed_label_groups,
        "near_dup_definition": {"shingle_words": SHINGLE, "jaccard_threshold": args.jaccard, "num_perm": NUM_PERM,
                                "bands": BANDS, "rows_per_band": ROWS, "max_ref_per_bucket": args.max_ref_per_bucket,
                                "verification": "exact Jaccard on shingle sets"},
        "output": args.out,
    }
    (tables / "03_split_meta.json").write_text(json.dumps(meta, indent=2))

    pd.set_option("display.width", 250)
    print("\n=== Dedup summary ===")
    print(dedup.to_string())
    print("\n=== Split summary ===")
    print(summary.to_string(index=False))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
