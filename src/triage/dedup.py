"""Exact and near-duplicate detection.

Near duplicate: two normalized texts whose sets of 5-word shingles have Jaccard >= 0.8.
Candidates come from MinHash LSH (128 hash functions, 16 bands x 8 rows, which catches pairs
well below 0.8 so few true pairs are missed); every candidate is then checked with the exact
Jaccard, so reported pairs never rest on the MinHash estimate.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable

import numpy as np
import pandas as pd

SHINGLE = 5
NUM_PERM = 128
BANDS = 16
ROWS = NUM_PERM // BANDS

_rng = np.random.default_rng(20261023)
# Multiply-shift hashing on uint64 with wraparound; one (a, b) pair per hash function.
_A = (_rng.integers(1, 2**63, NUM_PERM, dtype=np.uint64) | np.uint64(1))
_B = _rng.integers(0, 2**63, NUM_PERM, dtype=np.uint64)
_BAND_MIX = _rng.integers(1, 2**63, ROWS, dtype=np.uint64) | np.uint64(1)


def shingles(text: str, k: int = SHINGLE) -> list[str]:
    w = text.split()
    if len(w) <= k:
        return [" ".join(w)] if w else [""]
    return [" ".join(w[i : i + k]) for i in range(len(w) - k + 1)]


def jaccard(a: str, b: str) -> float:
    sa, sb = set(shingles(a)), set(shingles(b))
    return len(sa & sb) / len(sa | sb)


def band_keys(texts: Iterable[str]) -> np.ndarray:
    """MinHash signature folded into one uint64 key per band. Shape (n_docs, BANDS)."""
    texts = list(texts)
    sh, owner = [], []
    for i, t in enumerate(texts):
        s = set(shingles(t))
        sh.extend(s)
        owner.extend([i] * len(s))
    h = pd.util.hash_array(np.array(sh, dtype=object))  # stable across runs
    owner = np.asarray(owner)
    starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
    with np.errstate(over="ignore"):
        sig = np.empty((len(texts), NUM_PERM), dtype=np.uint64)
        for p in range(NUM_PERM):
            v = (h * _A[p] + _B[p]) >> np.uint64(32)
            sig[:, p] = np.minimum.reduceat(v, starts)
        sig = sig.reshape(len(texts), BANDS, ROWS)
        keys = (sig * _BAND_MIX).sum(axis=2, dtype=np.uint64)
    return keys


def compute_band_keys(texts: pd.Series, batch: int = 20_000, workers: int | None = None) -> np.ndarray:
    """Band keys for every text, computed in parallel batches. Results do not depend on workers."""
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    batches = [texts.iloc[i : i + batch].tolist() for i in range(0, len(texts), batch)]
    if not batches:
        return np.empty((0, BANDS), dtype=np.uint64)
    out = []
    if workers == 1:
        it = map(band_keys, batches)
    else:
        ex = ProcessPoolExecutor(max_workers=workers)
        it = ex.map(band_keys, batches)
    done = 0
    for k in it:
        out.append(k)
        done += len(k)
        print(f"  minhash {done:,} / {len(texts):,}", flush=True)
    if workers != 1:
        ex.shutdown()
    return np.vstack(out)


def find_near_dups(
    texts: pd.Series,
    keys: np.ndarray,
    is_ref: np.ndarray,
    is_query: np.ndarray,
    threshold: float = 0.8,
    max_ref_per_bucket: int = 5,
) -> pd.DataFrame:
    """For each query document, find one reference document with Jaccard >= threshold.

    texts, keys, is_ref, is_query are aligned by position. Within each LSH bucket a query is
    compared with at most max_ref_per_bucket reference documents (lowest positions first);
    this bounds the work on large template clusters. Returns one row per matched query:
    query_pos, ref_pos, jaccard.
    """
    pos = np.arange(len(texts))
    pairs = []
    for b in range(keys.shape[1]):
        ref = pd.DataFrame({"key": keys[is_ref, b], "ref_pos": pos[is_ref]})
        ref = ref[ref.groupby("key").cumcount() < max_ref_per_bucket]
        q = pd.DataFrame({"key": keys[is_query, b], "query_pos": pos[is_query]})
        pairs.append(q.merge(ref, on="key")[["query_pos", "ref_pos"]])
    cand = pd.concat(pairs).drop_duplicates()
    cand = cand[cand["query_pos"] != cand["ref_pos"]].sort_values(["query_pos", "ref_pos"])
    print(f"  {len(cand):,} candidate pairs for {cand['query_pos'].nunique():,} queries", flush=True)
    rows = []
    vals = texts.to_numpy()
    last = -1
    for qp, rp in zip(cand["query_pos"].to_numpy(), cand["ref_pos"].to_numpy()):
        if qp == last:
            continue  # already matched
        j = jaccard(vals[qp], vals[rp])
        if j >= threshold:
            rows.append((int(qp), int(rp), j))
            last = qp
    return pd.DataFrame(rows, columns=["query_pos", "ref_pos", "jaccard"])
