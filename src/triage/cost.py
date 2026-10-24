"""Illustrative operational cost model. Costs are assumed units, not dollars.

Expected cost = C_FN * FN + C_FP * FP. A reviewed true positive costs nothing extra.
Reviewing a complaint means its score is >= the threshold.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def cost_curve(y: np.ndarray, scores: np.ndarray, c_fn: float, c_fp: float) -> pd.DataFrame:
    """Cost for every distinct threshold, from reviewing nothing to reviewing everything.

    Rows are cut only where the score changes, so each row is a threshold rule `score >= t`
    (t = inf means review nothing).
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s, kind="stable")
    ss, yy = s[order], y[order]
    n, p = len(y), int(y.sum())
    cut = np.flatnonzero(np.r_[ss[1:] != ss[:-1], True]) + 1  # k after each run of equal scores
    k = np.r_[0, cut]
    tp = np.r_[0, np.cumsum(yy)[cut - 1]]
    fp = k - tp
    fn = p - tp
    thr = np.r_[np.inf, ss[cut - 1]]
    return pd.DataFrame({
        "threshold": thr, "reviewed": k, "review_rate": k / n, "tp": tp, "fp": fp, "fn": fn,
        "precision": np.divide(tp, k, out=np.full(len(k), np.nan), where=k > 0),
        "recall": tp / p if p else np.nan,
        "cost": c_fn * fn + c_fp * fp,
    })


def baselines(y: np.ndarray, c_fn: float, c_fp: float) -> dict:
    y = np.asarray(y).astype(int)
    p, n = int(y.sum()), len(y)
    return {"review_none": c_fn * p, "review_all": c_fp * (n - p)}


def best_threshold(y: np.ndarray, scores: np.ndarray, c_fn: float, c_fp: float) -> dict:
    """Minimum-cost threshold. Ties go to the smallest review rate."""
    cc = cost_curve(y, scores, c_fn, c_fp)
    row = cc.loc[cc["cost"].idxmin()]
    b = baselines(y, c_fn, c_fp)
    best_base = min(b.values())
    return {**{k: (float(v) if not isinstance(v, (int, np.integer)) else int(v)) for k, v in row.items()},
            "c_fn": c_fn, "c_fp": c_fp, **{f"baseline_{k}": v for k, v in b.items()},
            "saving_vs_best_baseline": best_base - float(row["cost"]),
            "saving_vs_best_baseline_pct": (best_base - float(row["cost"])) / best_base if best_base else math.nan}


def apply_threshold(y: np.ndarray, scores: np.ndarray, threshold: float, c_fn: float, c_fp: float) -> dict:
    y = np.asarray(y).astype(int)
    r = np.asarray(scores) >= threshold
    tp = int((r & (y == 1)).sum())
    fp = int((r & (y == 0)).sum())
    fn = int(y.sum()) - tp
    return {"threshold": threshold, "reviewed": int(r.sum()), "review_rate": float(r.mean()), "tp": tp, "fp": fp,
            "fn": fn, "precision": tp / r.sum() if r.sum() else math.nan,
            "recall": tp / y.sum() if y.sum() else math.nan, "cost": c_fn * fn + c_fp * fp}
