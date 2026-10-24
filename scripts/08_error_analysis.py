"""Step 9: error analysis on test (main population) for the primary model.

Review rule: the top 10% of test complaints by score. Slices by product, text length and
quarter. Example tables: highest-scored false positives and lowest-scored false negatives.
The report table keeps a 100-character snippet; a 300-character version is written under
data/ (git-ignored) for reading locally.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from triage.metrics import top_k_mask

LENGTH_BINS = [0, 250, 500, 1000, 2000, np.inf]
LENGTH_LABELS = ["<250", "250-499", "500-999", "1000-1999", "2000+"]


def slice_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by, observed=True)
    t = pd.DataFrame({
        "rows": g.size(),
        "positive_rate": g["label"].mean(),
        "review_rate": g["reviewed"].mean(),
        "precision": g.apply(lambda x: x.loc[x["reviewed"], "label"].mean() if x["reviewed"].any() else np.nan),
        "recall": g.apply(lambda x: x.loc[x["reviewed"], "label"].sum() / x["label"].sum() if x["label"].sum() else np.nan),
    })
    t = t.reset_index().rename(columns={by: "value"})
    t.insert(0, "slice", by)
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="reports/frozen_config.json")
    ap.add_argument("--clean", default="data/interim/complaints_clean.parquet")
    ap.add_argument("--pred-dir", default="data/predictions")
    ap.add_argument("--model", default=None)
    ap.add_argument("--capacity", type=float, default=0.10)
    ap.add_argument("--n-examples", type=int, default=15)
    ap.add_argument("--min-product-rows", type=int, default=1000)
    ap.add_argument("--tables", default="reports/tables")
    ap.add_argument("--local-out", default="data/interim/09_examples_with_text.csv")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    name = args.model or cfg["primary_model"]
    tables = Path(args.tables)

    t = pd.read_parquet(Path(args.pred_dir) / f"test_{name}.parquet")
    t = t[t["in_dedup"] & ~t["near_dup_train"]].reset_index(drop=True)
    t["reviewed"] = top_k_mask(t["score"].to_numpy(), args.capacity)
    t["quarter"] = t["date_received"].dt.to_period("Q").astype(str)
    t["length_bin"] = pd.cut(t["text_chars"], LENGTH_BINS, labels=LENGTH_LABELS, right=False)
    big = t["product"].map(t["product"].value_counts()) >= args.min_product_rows
    t["product_slice"] = t["product"].where(big, "other (small products)")

    slices = pd.concat([slice_table(t, "product_slice"), slice_table(t, "length_bin"), slice_table(t, "quarter")])
    slices.to_csv(tables / "09_slices.csv", index=False)

    fp = t[(t["label"] == 0)].nlargest(args.n_examples, "score").assign(kind="false_positive_highest_score")
    fn = t[(t["label"] == 1)].nsmallest(args.n_examples, "score").assign(kind="false_negative_lowest_score")
    ex = pd.concat([fp, fn])
    text = pd.read_parquet(args.clean, columns=["complaint_id", "text", "issue"])
    ex = ex.merge(text, on="complaint_id", how="left")
    cols = ["kind", "complaint_id", "product", "issue", "quarter", "score", "text_chars"]
    ex[cols].assign(snippet=ex["text"].str[:100]).to_csv(tables / "09_error_examples.csv", index=False)
    Path(args.local_out).parent.mkdir(parents=True, exist_ok=True)
    ex[cols].assign(snippet=ex["text"].str[:300]).to_csv(args.local_out, index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    pd.set_option("display.max_colwidth", 110)
    print("=== Slices (test main, global top {:.0%}) ===".format(args.capacity))
    print(slices.to_string(index=False))
    print("\n=== Examples (100-character snippets) ===")
    print(ex[["kind", "product", "issue", "score"]].assign(snippet=ex["text"].str[:100]).to_string(index=False))
    print(f"\n300-character versions for local reading: {args.local_out}")


if __name__ == "__main__":
    main()
