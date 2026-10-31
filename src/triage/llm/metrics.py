"""Metrics for extraction outputs. Values are True / False / None (null)."""
from __future__ import annotations

import numpy as np
import pandas as pd

LEVELS = ("true", "false", "null")


def to_level(v) -> str:
    return "null" if v is None or (isinstance(v, float) and np.isnan(v)) else ("true" if v else "false")


def prf_true(gold: list, pred: list) -> dict:
    """Precision, recall and F1 for the value True; False and null both count as not-true."""
    g = np.array([to_level(x) == "true" for x in gold])
    p = np.array([to_level(x) == "true" for x in pred])
    tp, fp, fn = int((g & p).sum()), int((~g & p).sum()), int((g & ~p).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else (0.0 if tp + fp + fn else float("nan"))
    return {"n": len(g), "gold_true": int(g.sum()), "pred_true": int(p.sum()), "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def bootstrap_f1(gold: list, pred: list, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    g, p = np.array(gold, dtype=object), np.array(pred, dtype=object)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(g), len(g))
        f = prf_true(list(g[i]), list(p[i]))["f1"]
        if not np.isnan(f):
            vals.append(f)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"),) * 2


def confusion(gold: list, pred: list) -> pd.DataFrame:
    g = pd.Categorical([to_level(x) for x in gold], categories=LEVELS)
    p = pd.Categorical([to_level(x) for x in pred], categories=LEVELS)
    return pd.crosstab(pd.Series(g, name="gold"), pd.Series(p, name="pred"), dropna=False)


def agreement(a: list, b: list) -> float:
    """Share of items where two runs (or two prompt versions, or two labeling passes) give the same level."""
    if not a:
        return float("nan")
    return float(np.mean([to_level(x) == to_level(y) for x, y in zip(a, b)]))


def amount_agreement(pred_sets: list[set], base_sets: list[set]) -> dict:
    exact = [p == b for p, b in zip(pred_sets, base_sets)]
    tp = sum(len(p & b) for p, b in zip(pred_sets, base_sets))
    npred = sum(len(p) for p in pred_sets)
    nbase = sum(len(b) for b in base_sets)
    return {"items": len(exact), "exact_set_match": float(np.mean(exact)) if exact else float("nan"),
            "values_pred": npred, "values_regex": nbase,
            "precision_vs_regex": tp / npred if npred else float("nan"),
            "recall_vs_regex": tp / nbase if nbase else float("nan")}
