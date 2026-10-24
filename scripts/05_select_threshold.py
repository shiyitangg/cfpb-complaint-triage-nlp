"""Step 6: choose review thresholds on validation only, under an illustrative cost model.

Cost = C_FN * FN + C_FP * FP in assumed cost units (not dollars). For each model and each
C_FN / C_FP ratio, the minimum-cost threshold is found on validation and frozen in
reports/frozen_config.json together with the score cut-offs for fixed review capacities.
Both no-model baselines (review none, review all) are reported next to every result.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from triage.cost import baselines, best_threshold, cost_curve
from triage.metrics import CAPACITIES, at_capacity


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-dir", default="data/predictions")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--primary", default="tfidf_structured")
    ap.add_argument("--c-fp", type=float, default=1.0)
    ap.add_argument("--ratios", nargs="+", type=float, default=[5, 10, 20], help="C_FN / C_FP values")
    ap.add_argument("--primary-ratio", type=float, default=10)
    ap.add_argument("--capacities", nargs="+", type=float, default=list(CAPACITIES))
    ap.add_argument("--setting", choices=["main", "all"], default="main",
                    help="main = drop val rows that near-duplicate a train row")
    ap.add_argument("--tables", default="reports/tables")
    ap.add_argument("--figures", default="reports/figures")
    ap.add_argument("--config-out", default="reports/frozen_config.json")
    ap.add_argument("--lock", default="reports/test_opened.lock")
    args = ap.parse_args()
    if Path(args.lock).exists():
        raise SystemExit(f"{args.lock} exists: the test set was already opened, thresholds must not change now")
    tables, figures = Path(args.tables), Path(args.figures)
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.pred_dir).glob("val_*.parquet"))
    if not files:
        raise SystemExit(f"no validation predictions in {args.pred_dir}; run step 5 first")
    if not (Path(args.pred_dir) / f"val_{args.primary}.parquet").exists():
        raise SystemExit(f"no validation predictions for the primary model {args.primary}")
    if args.primary_ratio not in args.ratios:
        raise SystemExit("--primary-ratio must be one of --ratios")
    sens, capr, frozen_models, curves = [], [], {}, {}
    for f in files:
        name = f.stem[len("val_"):]
        v = pd.read_parquet(f)
        if args.setting == "main":
            v = v[~v["near_dup_train"]]
        y, s = v["label"].to_numpy(), v["score"].to_numpy()
        entry = {"model_path": str(Path(args.models_dir) / f"{name}.joblib"), "cost_thresholds": {}, "capacity_thresholds": {}}
        mp = Path(entry["model_path"])
        if mp.exists():
            entry["C"] = joblib.load(mp)["C"]
        for r in args.ratios:
            c_fn = r * args.c_fp
            b = best_threshold(y, s, c_fn, args.c_fp)
            sens.append({"model": name, "ratio": r, **b})
            entry["cost_thresholds"][str(r)] = b["threshold"]
            if name == args.primary:
                curves[r] = cost_curve(y, s, c_fn, args.c_fp)
        order = np.sort(s)[::-1]
        for c in args.capacities:
            k = math.ceil(c * len(s))
            t = float(order[k - 1])
            row = {"model": name, **at_capacity(y, s, c), "score_cutoff": t}
            fp = row["reviewed"] - row["tp"]
            fn = int(y.sum()) - row["tp"]
            for r in args.ratios:
                row[f"cost_ratio_{r:g}"] = r * args.c_fp * fn + args.c_fp * fp
            capr.append(row)
            entry["capacity_thresholds"][str(c)] = t
        frozen_models[name] = entry

    sens = pd.DataFrame(sens)
    capr = pd.DataFrame(capr)
    sens.to_csv(tables / "06_cost_sensitivity_val.csv", index=False)
    capr.to_csv(tables / "06_capacity_scenarios_val.csv", index=False)

    v = pd.read_parquet(Path(args.pred_dir) / f"val_{args.primary}.parquet")
    if args.setting == "main":
        v = v[~v["near_dup_train"]]
    fig, axes = plt.subplots(1, len(args.ratios), figsize=(4.2 * len(args.ratios), 3.8), sharey=False)
    for ax, r in zip(np.atleast_1d(axes), args.ratios):
        cc = curves[r]
        b = baselines(v["label"].to_numpy(), r * args.c_fp, args.c_fp)
        ax.plot(cc["review_rate"], cc["cost"], color="#333333", lw=1.2, label="model")
        ax.axhline(b["review_none"], color="#C44E52", ls="--", lw=1, label="review none")
        ax.axhline(b["review_all"], color="#4C72B0", ls="--", lw=1, label="review all")
        i = cc["cost"].idxmin()
        ax.scatter([cc.loc[i, "review_rate"]], [cc.loc[i, "cost"]], color="#DD8452", zorder=3, label="minimum")
        ax.set_title(f"C_FN / C_FP = {r:g}")
        ax.set_xlabel("share of complaints reviewed")
        ax.set_ylabel("cost (illustrative units)")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Validation cost by review share, {args.primary} (assumed costs)")
    fig.tight_layout()
    fig.savefig(figures / "06_cost_vs_threshold.png", dpi=150)

    config = {
        "selected_on": "validation 2022", "setting": args.setting, "primary_model": args.primary,
        "cost_model": {"description": "illustrative operational cost units, not dollars",
                       "c_fp": args.c_fp, "ratios": args.ratios, "primary_ratio": args.primary_ratio,
                       "reviewed_true_positive_cost": 0},
        "capacities": args.capacities,
        "models": frozen_models,
    }
    Path(args.config_out).write_text(json.dumps(config, indent=2))

    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    cols = ["model", "ratio", "threshold", "review_rate", "precision", "recall", "cost",
            "baseline_review_none", "baseline_review_all", "saving_vs_best_baseline_pct"]
    print("\n=== Cost-minimizing threshold on validation ===")
    print(sens[cols].to_string(index=False))
    print("\n=== Capacity scenarios on validation ===")
    print(capr[["model", "capacity", "reviewed", "precision", "recall", "lift", "score_cutoff"]
               + [c for c in capr.columns if c.startswith("cost_ratio_")]].to_string(index=False))
    print(f"\nfrozen config written to {args.config_out}")


if __name__ == "__main__":
    main()
