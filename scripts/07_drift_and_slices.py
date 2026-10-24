"""Step 8: drift between periods, quarterly performance, per-product results.

Reads stored predictions only; nothing is refitted and no threshold changes. Review rules:
  frozen cutoff  the validation score cut-off for a 10% review capacity, applied unchanged
  top-k          the top 10% within the group being measured
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from triage.metrics import at_capacity, summarize, top_k_mask

CREDIT_REPORTING = "credit reporting"


def pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="reports/frozen_config.json")
    ap.add_argument("--modeling", default="data/processed/modeling.parquet")
    ap.add_argument("--pred-dir", default="data/predictions")
    ap.add_argument("--model", default=None, help="default: primary model in the config")
    ap.add_argument("--capacity", type=float, default=0.10)
    ap.add_argument("--min-product-rows", type=int, default=1000)
    ap.add_argument("--tables", default="reports/tables")
    ap.add_argument("--figures", default="reports/figures")
    args = ap.parse_args()
    tables, figures = Path(args.tables), Path(args.figures)
    cfg = json.loads(Path(args.config).read_text())
    name = args.model or cfg["primary_model"]
    cutoff = cfg["models"][name]["capacity_thresholds"][str(args.capacity)]
    tag = f"{round(args.capacity * 100)}pct"

    # Drift: train vs val vs test, main setting throughout.
    md = pd.read_parquet(args.modeling, columns=["split", "near_dup_train", "product", "label", "text"])
    md["text_chars"] = md.pop("text").str.len()
    md = md[~md["near_dup_train"]]
    rows = []
    for s in ["train", "val", "test"]:
        x = md[md["split"] == s]
        q = x["text_chars"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        rows.append({"split": s, "rows": len(x), "positive_rate": x["label"].mean(),
                     "text_chars_mean": x["text_chars"].mean(), **{f"text_chars_p{int(k * 100)}": v for k, v in q.items()}})
    drift = pd.DataFrame(rows)
    drift.to_csv(tables / "08_drift.csv", index=False)
    mix = (md.groupby(["product", "split"]).agg(rows=("label", "size"), positive_rate=("label", "mean"))
             .reset_index())
    mix["share"] = mix["rows"] / mix.groupby("split")["rows"].transform("sum")
    mix = mix.pivot(index="product", columns="split", values=["share", "positive_rate"])
    mix.columns = [f"{a}_{b}" for a, b in mix.columns]
    mix = mix.reindex(columns=[f"{a}_{b}" for a in ("share", "positive_rate") for b in ("train", "val", "test")])
    mix = mix.sort_values("share_test", ascending=False)
    mix.to_csv(tables / "08_product_mix.csv")

    # Quarterly: validation and test quarters, frozen cut-off and within-quarter top-k.
    val = pd.read_parquet(Path(args.pred_dir) / f"val_{name}.parquet")
    val = val[~val["near_dup_train"]].assign(split="val")
    test = pd.read_parquet(Path(args.pred_dir) / f"test_{name}.parquet")
    test = test[test["in_dedup"] & ~test["near_dup_train"]].assign(split="test")
    both = pd.concat([val, test], ignore_index=True)
    both["quarter"] = both["date_received"].dt.to_period("Q").astype(str)
    qrows = []
    for (split, qt), g in both.groupby(["split", "quarter"], sort=True):
        y, s = g["label"].to_numpy(), g["score"].to_numpy()
        r = s >= cutoff
        tk = at_capacity(y, s, args.capacity)
        qrows.append({"split": split, "quarter": qt, "rows": len(g), "positive_rate": y.mean(), "pr_auc": pr_auc(y, s),
                      "frozen_cutoff_review_rate": r.mean(),
                      "frozen_cutoff_precision": y[r].mean() if r.any() else np.nan,
                      "frozen_cutoff_recall": y[r].sum() / y.sum() if y.sum() else np.nan,
                      f"precision@{tag}": tk["precision"], f"recall@{tag}": tk["recall"]})
    quarterly = pd.DataFrame(qrows).sort_values("quarter")
    quarterly.to_csv(tables / "08_quarterly.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(quarterly))
    ax.plot(x, quarterly[f"recall@{tag}"], marker="o", label=f"recall, top {tag.replace('pct', '%')} per quarter")
    ax.plot(x, quarterly[f"precision@{tag}"], marker="o", label=f"precision, top {tag.replace('pct', '%')} per quarter")
    ax.plot(x, quarterly["frozen_cutoff_recall"], marker="x", ls="--", label="recall, frozen validation cut-off")
    ax.plot(x, quarterly["positive_rate"], color="#999999", ls=":", label="base rate")
    ax.axvline((quarterly["split"] == "val").sum() - 0.5, color="#999999", lw=1)
    ax.set_xticks(x, quarterly["quarter"], rotation=45)
    ax.set_ylim(0, 1)
    ax.set_title(f"{name}: validation (2022) and test (2023) by quarter")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "08_quarterly.png", dpi=150)

    # Per product on test: global top-k ranking, plus per-product PR-AUC.
    y, s = test["label"].to_numpy(), test["score"].to_numpy()
    reviewed = top_k_mask(s, args.capacity)
    test = test.assign(reviewed=reviewed)
    prow = []
    for prod, g in test.groupby("product"):
        if len(g) < args.min_product_rows:
            continue
        gy, gs, gr = g["label"].to_numpy(), g["score"].to_numpy(), g["reviewed"].to_numpy()
        prow.append({"product": prod, "rows": len(g), "share_of_test": len(g) / len(test), "positives": int(gy.sum()),
                     "share_of_positives": gy.sum() / y.sum(), "positive_rate": gy.mean(), "pr_auc": pr_auc(gy, gs),
                     "review_rate": gr.mean(), "share_of_reviewed": gr.sum() / reviewed.sum(),
                     "precision": gy[gr].mean() if gr.any() else np.nan,
                     "recall": gy[gr].sum() / gy.sum() if gy.sum() else np.nan})
    by_product = pd.DataFrame(prow).sort_values("rows", ascending=False)
    by_product.to_csv(tables / "08_by_product.csv", index=False)

    # Does credit reporting dominate the overall numbers?
    is_cr = test["product"].fillna("").str.lower().str.contains(CREDIT_REPORTING).to_numpy()
    cr_rows = []
    for label, m in (("all", np.ones(len(test), bool)), ("excluding_credit_reporting", ~is_cr),
                     ("credit_reporting_only", is_cr)):
        cr_rows.append({"subset": label, "share_of_test": m.mean(), **summarize(y[m], s[m])})
    cr = pd.DataFrame(cr_rows)
    cr.to_csv(tables / "08_excluding_credit_reporting.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    pd.set_option("display.max_colwidth", 60)
    print("=== Drift (main setting) ===")
    print(drift.to_string(index=False))
    print("\n=== Product mix (top 12 by test share) ===")
    print(mix.head(12).to_string())
    print("\n=== Quarterly ===")
    print(quarterly.to_string(index=False))
    print(f"\n=== Test by product (global top {tag.replace('pct', '%')}) ===")
    print(by_product.to_string(index=False))
    print("\n=== Credit reporting check (each subset ranked within itself) ===")
    print(cr[["subset", "share_of_test", "n", "positives", "base_rate", "pr_auc", f"precision@{tag}",
              f"recall@{tag}", f"lift@{tag}"]].to_string(index=False))


if __name__ == "__main__":
    main()
