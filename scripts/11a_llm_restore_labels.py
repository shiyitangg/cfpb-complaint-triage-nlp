"""LLM reproduce helper: rebuild data/llm/to_label_gold.csv from the committed gold labels.

The labeling CSV holds narrative text, so it is not committed. reports/llm/gold_labels.csv holds
item_id, Complaint ID and the three labels. After scripts/09_llm_sample.py has rebuilt
data/llm/gold.parquet, this joins the two so scripts/11_llm_eval.py can score the gold set.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default="reports/llm/gold_labels.csv")
    ap.add_argument("--gold", default="data/llm/gold.parquet")
    ap.add_argument("--out", default="data/llm/to_label_gold.csv")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and (pd.read_csv(out, dtype=str, keep_default_na=False).iloc[:, 2:5] != "").any().any():
        raise SystemExit(f"{out} already has labels; it is not overwritten.")
    gold = pd.read_parquet(args.gold, columns=["item_id", "complaint_id", "text"])
    gold["complaint_id"] = gold["complaint_id"].astype(str)
    lab = pd.read_csv(args.labels, dtype=str, keep_default_na=False)
    m = gold.merge(lab, on=["item_id", "complaint_id"], how="inner")
    if len(m) != len(gold):
        raise SystemExit(f"only {len(m)} of {len(gold)} gold items match {args.labels}; the sample differs")
    m.drop(columns=["complaint_id"]).assign(notes="").to_csv(out, index=False)
    print(f"wrote {out} ({len(m)} items)")


if __name__ == "__main__":
    main()
