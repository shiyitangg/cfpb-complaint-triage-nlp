"""Ranking metrics for an imbalanced label. Accuracy is deliberately absent."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

CAPACITIES = (0.05, 0.10, 0.20, 0.30)


def top_k_mask(scores: np.ndarray, frac: float) -> np.ndarray:
    """Boolean mask of the top ceil(frac * n) scores. Ties are broken by position (stable sort),
    so the reviewed count is exactly the capacity."""
    n = len(scores)
    k = math.ceil(frac * n)
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def at_capacity(y: np.ndarray, scores: np.ndarray, frac: float) -> dict:
    y = np.asarray(y).astype(int)
    m = top_k_mask(scores, frac)
    tp = int(y[m].sum())
    pos = int(y.sum())
    base = pos / len(y) if len(y) else float("nan")
    precision = tp / m.sum() if m.sum() else float("nan")
    recall = tp / pos if pos else float("nan")
    return {"capacity": frac, "reviewed": int(m.sum()), "tp": tp, "precision": precision,
            "recall": recall, "lift": precision / base if base else float("nan")}


def summarize(y: np.ndarray, scores: np.ndarray, capacities: Sequence[float] = CAPACITIES) -> dict:
    y = np.asarray(y).astype(int)
    out = {"n": int(len(y)), "positives": int(y.sum()), "base_rate": float(y.mean()),
           "pr_auc": float(average_precision_score(y, scores))}
    for c in capacities:
        r = at_capacity(y, scores, c)
        tag = f"{round(c * 100)}pct"
        out[f"precision@{tag}"] = r["precision"]
        out[f"recall@{tag}"] = r["recall"]
        out[f"lift@{tag}"] = r["lift"]
    return out


def summarize_frame(df: pd.DataFrame, y: str = "label", s: str = "score") -> dict:
    return summarize(df[y].to_numpy(), df[s].to_numpy())
