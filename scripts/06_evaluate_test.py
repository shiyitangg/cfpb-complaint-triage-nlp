"""Step 7: one-time evaluation on the test split with everything frozen on validation.

The first run writes reports/test_opened.lock. Later runs stop unless --force is given, and
every forced run is appended to reports/test_open_log.txt, so the record shows how many
times the test set was opened.

Populations:
  main       test rows after exact dedup, minus rows that near-duplicate a train row
  dedup_all  test rows after exact dedup
  raw_queue  every cleaned test row, exact duplicates included (closer to a real queue)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from triage.company_mask import CompanyMasker
from triage.cost import apply_threshold, baselines
from triage.metrics import at_capacity, summarize

POPULATIONS = ["main", "dedup_all", "raw_queue"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="reports/frozen_config.json")
    ap.add_argument("--modeling", default="data/processed/modeling.parquet")
    ap.add_argument("--clean", default="data/interim/complaints_clean.parquet")
    ap.add_argument("--split-meta", default="reports/tables/03_split_meta.json")
    ap.add_argument("--pred-dir", default="data/predictions")
    ap.add_argument("--tables", default="reports/tables")
    ap.add_argument("--figures", default="reports/figures")
    ap.add_argument("--results", default="reports/results.json")
    ap.add_argument("--lock", default="reports/test_opened.lock")
    ap.add_argument("--force", action="store_true", help="re-open the test set; the run is logged")
    args = ap.parse_args()
    tables, figures = Path(args.tables), Path(args.figures)
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    cfg_text = Path(args.config).read_text()
    cfg = json.loads(cfg_text)
    cfg_hash = hashlib.sha256(cfg_text.encode()).hexdigest()
    lock = Path(args.lock)
    now = datetime.now(timezone.utc).isoformat()
    if lock.exists():
        if not args.force:
            raise SystemExit(f"{lock} exists: the test set was already opened ({lock.read_text().strip()}). "
                             "Use --force only with a reason; the run will be logged.")
        with open(Path(args.lock).parent / "test_open_log.txt", "a") as f:
            f.write(f"{now} forced re-open, config sha256 {cfg_hash}\n")
    else:
        lock.write_text(json.dumps({"opened_at_utc": now, "config_sha256": cfg_hash}) + "\n")

    meta = json.loads(Path(args.split_meta).read_text())["split_dates"]
    t0, t1 = pd.Timestamp(meta["test_start"]), pd.Timestamp(meta["test_end"])

    clean = pd.read_parquet(args.clean)
    day = clean["date_received"].dt.normalize()
    raw = clean[(day >= t0) & (day <= t1)].reset_index(drop=True)
    flags = pd.read_parquet(args.modeling, columns=["complaint_id", "split", "near_dup_train", "near_dup_val"],
                            filters=[("split", "==", "test")])
    raw = raw.merge(flags.drop(columns="split"), on="complaint_id", how="left", indicator=True)
    raw["in_dedup"] = raw.pop("_merge") == "both"
    raw["near_dup_train"] = raw["near_dup_train"].fillna(False).astype(bool)
    raw["near_dup_val"] = raw["near_dup_val"].fillna(False).astype(bool)
    masks = {"main": (raw["in_dedup"] & ~raw["near_dup_train"]).to_numpy(),
             "dedup_all": raw["in_dedup"].to_numpy(),
             "raw_queue": np.ones(len(raw), dtype=bool)}
    print({k: int(v.sum()) for k, v in masks.items()}, flush=True)
    y_all = raw["label"].to_numpy()

    c_fp = cfg["cost_model"]["c_fp"]
    ratios = cfg["cost_model"]["ratios"]
    comp, costs, caps, curves = [], [], [], {}
    for name, entry in cfg["models"].items():
        art = joblib.load(entry["model_path"])
        df = raw
        if "company_masker_phrases" in art:
            df = raw.assign(text_masked=CompanyMasker(art["company_masker_phrases"]).transform(raw["text"]))
        s = art["model"].predict_proba(art["featurizer"].transform(df))[:, 1]
        pd.DataFrame({"complaint_id": raw["complaint_id"], "date_received": raw["date_received"],
                      "product": raw["product"], "text_chars": raw["text"].str.len(), "label": y_all, "score": s,
                      "in_dedup": raw["in_dedup"], "near_dup_train": raw["near_dup_train"],
                      "near_dup_val": raw["near_dup_val"]}).to_parquet(Path(args.pred_dir) / f"test_{name}.parquet", index=False)
        print(f"[{name}] scored {len(s):,} test rows", flush=True)

        for pop in POPULATIONS:
            m = masks[pop]
            y, sc = y_all[m], s[m]
            comp.append({"model": name, "population": pop, **summarize(y, sc, cfg["capacities"])})
            for r in ratios:
                c_fn = r * c_fp
                b = baselines(y, c_fn, c_fp)
                res = apply_threshold(y, sc, entry["cost_thresholds"][str(r)], c_fn, c_fp)
                best = min(b.values())
                costs.append({"model": name, "population": pop, "ratio": r, **res,
                              "baseline_review_none": b["review_none"], "baseline_review_all": b["review_all"],
                              "saving_vs_best_baseline_pct": (best - res["cost"]) / best})
            for c in cfg["capacities"]:
                top = at_capacity(y, sc, c)
                fixed = apply_threshold(y, sc, entry["capacity_thresholds"][str(c)], c_fp, c_fp)
                row = {"model": name, "population": pop, "capacity": c,
                       "topk_precision": top["precision"], "topk_recall": top["recall"], "topk_lift": top["lift"],
                       "frozen_cutoff_review_rate": fixed["review_rate"], "frozen_cutoff_precision": fixed["precision"],
                       "frozen_cutoff_recall": fixed["recall"]}
                fn = int(y.sum()) - top["tp"]
                fp = top["reviewed"] - top["tp"]
                for r in ratios:
                    row[f"topk_cost_ratio_{r:g}"] = r * c_fp * fn + c_fp * fp
                caps.append(row)
            if pop == "main":
                p, rc, _ = precision_recall_curve(y, sc)
                curves[name] = (rc, p)

    comp, costs, caps = pd.DataFrame(comp), pd.DataFrame(costs), pd.DataFrame(caps)
    comp.to_csv(tables / "07_test_model_comparison.csv", index=False)
    costs.to_csv(tables / "07_test_cost_frozen_thresholds.csv", index=False)
    caps.to_csv(tables / "07_test_capacity.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (rc, p) in curves.items():
        a = comp.query("model == @name and population == 'main'")["pr_auc"].iloc[0]
        ax.plot(rc, p, label=f"{name} (AP {a:.3f})")
    ax.axhline(y_all[masks["main"]].mean(), color="#999999", ls="--", lw=1, label="base rate")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Test (2023) precision-recall, main population")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "07_test_pr_curves.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    cm = caps[caps["population"] == "main"]
    for name, g in cm.groupby("model", sort=False):
        ax.plot(g["capacity"] * 100, g["topk_recall"], marker="o", label=name)
    ax.set_xlabel("share of complaints reviewed (%)")
    ax.set_ylabel("recall of monetary-relief complaints")
    ax.set_title("Test (2023): recall at fixed review capacity")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "07_test_recall_at_k.png", dpi=150)

    prim, pr = cfg["primary_model"], cfg["cost_model"]["primary_ratio"]
    head = comp.query("model == @prim and population == 'main'").iloc[0].to_dict()
    hc = costs.query("model == @prim and population == 'main' and ratio == @pr").iloc[0].to_dict()
    results = {
        "test_opened_at_utc": now, "config_sha256": cfg_hash, "primary_model": prim,
        "populations": {k: int(v.sum()) for k, v in masks.items()},
        "primary_test_main": head,
        "primary_test_main_cost_at_primary_ratio": hc,
        "pr_auc_by_model_and_population": comp.pivot(index="model", columns="population", values="pr_auc").to_dict(),
        "cost_units": "illustrative, assumed; not dollars",
    }
    Path(args.results).write_text(json.dumps(results, indent=2, default=float))

    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n=== Test model comparison ===")
    print(comp[["model", "population", "n", "positives", "base_rate", "pr_auc", "precision@10pct", "recall@10pct",
                "lift@10pct", "recall@5pct", "recall@20pct", "recall@30pct"]].to_string(index=False))
    print("\n=== Test cost with thresholds frozen on validation ===")
    print(costs[["model", "population", "ratio", "review_rate", "precision", "recall", "cost", "baseline_review_none",
                 "baseline_review_all", "saving_vs_best_baseline_pct"]].to_string(index=False))
    print("\n=== Test capacity (main population) ===")
    print(caps[caps["population"] == "main"].to_string(index=False))
    print(f"\nresults written to {args.results}")


if __name__ == "__main__":
    main()
