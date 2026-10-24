import numpy as np
import pandas as pd

from triage.dedup import compute_band_keys, find_near_dups, jaccard

BASE = ("i am writing to dispute the late fee charged on my credit card account because the payment "
        "was submitted on time through the online portal and the bank still refused to reverse it")


def test_jaccard_identical_and_disjoint():
    assert jaccard(BASE, BASE) == 1.0
    assert jaccard(BASE, "completely different words about a mortgage escrow shortage letter") == 0.0


def test_near_dup_found_and_unrelated_not():
    near = BASE + " again"            # one extra word: Jaccard well above 0.8
    other = " ".join(reversed(BASE.split()))
    texts = pd.Series([BASE, near, other])
    keys = compute_band_keys(texts, workers=1)
    is_ref = np.array([True, False, False])
    is_query = ~is_ref
    res = find_near_dups(texts, keys, is_ref, is_query, threshold=0.8)
    assert res["query_pos"].tolist() == [1]
    assert res["ref_pos"].tolist() == [0]
    assert (res["jaccard"] >= 0.8).all()


def test_band_keys_stable():
    texts = pd.Series([BASE])
    assert (compute_band_keys(texts, workers=1) == compute_band_keys(texts, workers=1)).all()
