"""Step 5: train the four models on train and choose C on validation PR-AUC.

The test split is never read: the parquet is loaded with a filter on split in {train, val}.
Validation metrics are reported for the main setting (drop val rows that near-duplicate a
train row) and for all val rows.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve

from triage.company_mask import CompanyMasker, build_phrases
from triage.leakage import check_features
from triage.metrics import summarize
from triage.models import SPECS, Featurizer, Spec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/processed/modeling.parquet")
    ap.add_argument("--models", nargs="+", default=list(SPECS))
    ap.add_argument("--c-grid", nargs="+", type=float, default=[0.25, 1.0, 4.0])
    ap.add_argument("--max-train-rows", type=int, default=None, help="random subsample of train, for a quick run")
    ap.add_argument("--vocab-sample", type=int, default=200_000)
    ap.add_argument("--max-features", type=int, default=200_000)
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--n-terms", type=int, default=30)
    ap.add_argument("--mask-min-count", type=int, default=50,
                    help="companies with at least this many train complaints are masked in text_masked")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models")
    ap.add_argument("--pred-dir", default="data/predictions")
    ap.add_argument("--tables", default="reports/tables")
    ap.add_argument("--figures", default="reports/figures")
    ap.add_argument("--lock", default="reports/test_opened.lock")
    args = ap.parse_args()
    if Path(args.lock).exists():
        raise SystemExit(f"{args.lock} exists: the test set was already opened, models must not be retrained now")
    for d in (args.out, args.pred_dir, args.tables, args.figures):
        Path(d).mkdir(parents=True, exist_ok=True)
    tables, figures = Path(args.tables), Path(args.figures)

    df = pd.read_parquet(args.data, filters=[("split", "in", ["train", "val"])])
    assert set(df["split"].unique()) <= {"train", "val"}, "test rows must not be loaded here"
    train = df[df["split"] == "train"]
    if args.max_train_rows and len(train) > args.max_train_rows:
        train = train.sample(args.max_train_rows, random_state=args.seed)
    val = df[df["split"] == "val"].reset_index(drop=True)
    main_mask = ~val["near_dup_train"].to_numpy()
    y_tr, y_val = train["label"].to_numpy(), val["label"].to_numpy()
    if any(SPECS[m].text_col == "text_masked" for m in args.models):
        t0 = time.time()
        masker = CompanyMasker(build_phrases(train["company"], min_count=args.mask_min_count))
        train = train.assign(text_masked=masker.transform(train["text"]))
        val = val.assign(text_masked=masker.transform(val["text"]))
        Path(tables / "05_company_mask_phrases.txt").write_text("\n".join(masker.phrases) + "\n")
        print(f"company mask: {len(masker.phrases):,} phrases from train, applied in {time.time() - t0:.0f}s", flush=True)
    print(f"train {len(train):,} rows ({y_tr.mean():.4%} positive); val {len(val):,} rows, main {main_mask.sum():,}")

    grid_rows, comp_rows, audits, terms = [], [], {}, []
    pr_curves = {}
    # One TF-IDF vocabulary per text column, fitted on train, shared by the models that use it.
    shared = {}
    for col in sorted({SPECS[m].text_col for m in args.models if SPECS[m].text}):
        t0 = time.time()
        f = Featurizer(Spec("tfidf_" + col, True, (), text_col=col), min_df=args.min_df,
                       max_features=args.max_features, vocab_sample=args.vocab_sample, seed=args.seed).fit(train)
        shared[col] = (f.tfidf, f.transform(train), f.transform(val))
        print(f"TF-IDF on {col}: {shared[col][1].shape[1]:,} terms, {shared[col][1].nnz:,} non-zeros in train, "
              f"built in {time.time() - t0:.0f}s", flush=True)
    for name in args.models:
        spec = SPECS[name]
        audits[name] = check_features(spec.columns)
        t0 = time.time()
        tf, T_tr, T_val = shared[spec.text_col] if spec.text else (None, None, None)
        feat = Featurizer(spec, tfidf=tf).fit(train)
        X_tr = feat.transform(train, T_tr)
        X_val = feat.transform(val, T_val)
        print(f"[{name}] features {X_tr.shape[1]:,}, built in {time.time() - t0:.0f}s", flush=True)

        best = None
        for c in args.c_grid:
            t1 = time.time()
            clf = LogisticRegression(C=c, solver="liblinear", max_iter=500)
            clf.fit(X_tr, y_tr)
            s = clf.predict_proba(X_val)[:, 1]
            ap_main = average_precision_score(y_val[main_mask], s[main_mask])
            grid_rows.append({"model": name, "C": c, "val_pr_auc_main": ap_main,
                              "val_pr_auc_all": average_precision_score(y_val, s), "fit_seconds": round(time.time() - t1, 1)})
            print(f"[{name}] C={c}: val PR-AUC (main) {ap_main:.4f}, {time.time() - t1:.0f}s", flush=True)
            if best is None or ap_main > best[0]:
                best = (ap_main, c, clf, s)
        _, c_best, clf, s = best
        if c_best in (min(args.c_grid), max(args.c_grid)) and len(args.c_grid) > 1:
            print(f"[{name}] WARNING: best C={c_best} is at the edge of the grid; consider widening --c-grid", flush=True)

        artifact = {"spec": spec, "featurizer": feat, "model": clf, "C": c_best}
        if spec.text_col == "text_masked":
            artifact["company_masker_phrases"] = masker.phrases
        joblib.dump(artifact, Path(args.out) / f"{name}.joblib")
        pd.DataFrame({"complaint_id": val["complaint_id"], "date_received": val["date_received"],
                      "product": val["product"], "label": y_val, "score": s,
                      "near_dup_train": val["near_dup_train"]}).to_parquet(Path(args.pred_dir) / f"val_{name}.parquet", index=False)
        for setting, m in (("main", main_mask), ("all", np.ones_like(main_mask))):
            comp_rows.append({"model": name, "C": c_best, "setting": setting, **summarize(y_val[m], s[m])})
        p, r, _ = precision_recall_curve(y_val[main_mask], s[main_mask])
        pr_curves[name] = (r, p)

        if spec.text:
            names = feat.feature_names()
            coef = clf.coef_.ravel()
            is_text = np.char.startswith(names.astype(str), "text:")
            idx = np.flatnonzero(is_text)
            order = idx[np.argsort(coef[idx])]
            for direction, sel in (("positive", order[::-1][: args.n_terms]), ("negative", order[: args.n_terms])):
                for rank, j in enumerate(sel, 1):
                    terms.append({"model": name, "direction": direction, "rank": rank,
                                  "term": names[j][5:], "coef": float(coef[j])})

    # Running a subset of models updates their rows and keeps the others from earlier runs.
    def merge(path: Path, new: pd.DataFrame) -> pd.DataFrame:
        if path.exists() and len(new):
            old = pd.read_csv(path)
            new = pd.concat([old[~old["model"].isin(new["model"].unique())], new], ignore_index=True)
        new.to_csv(path, index=False)
        return new

    grid = merge(tables / "05_val_c_grid.csv", pd.DataFrame(grid_rows))
    comp = merge(tables / "05_val_model_comparison.csv", pd.DataFrame(comp_rows))
    merge(tables / "05_top_terms.csv", pd.DataFrame(terms))
    audit_path = tables / "05_leakage_audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {"test_rows_loaded": 0, "models": {}}
    audit["models"].update(audits)
    audit.update({"test_rows_loaded": 0, "train_rows": int(len(train)), "val_rows": int(len(val))})
    audit_path.write_text(json.dumps(audit, indent=2))

    for f in sorted(Path(args.pred_dir).glob("val_*.parquet")):
        name = f.stem[len("val_"):]
        if name not in pr_curves and name in set(comp["model"]):
            v = pd.read_parquet(f)
            v = v[~v["near_dup_train"]]
            p, r, _ = precision_recall_curve(v["label"], v["score"])
            pr_curves[name] = (r, p)
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (r, p) in sorted(pr_curves.items()):
        ap_v = comp.query("model == @name and setting == 'main'")["pr_auc"].iloc[0]
        ax.plot(r, p, label=f"{name} (AP {ap_v:.3f})")
    ax.axhline(y_val[main_mask].mean(), color="#999999", ls="--", lw=1, label="base rate")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Validation (2022) precision-recall, main setting")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "05_val_pr_curves.png", dpi=150)

    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n=== C grid ===")
    print(grid.to_string(index=False))
    print("\n=== Validation comparison ===")
    print(comp.to_string(index=False))
    t = pd.DataFrame(terms)
    for m in ("tfidf", "tfidf_structured_masked"):
        if len(t) and (t["model"] == m).any():
            print(f"\n=== Top terms ({m}) ===")
            tt = t[t["model"] == m]
            print(tt.pivot(index="rank", columns="direction", values="term").head(20).to_string())


if __name__ == "__main__":
    main()
