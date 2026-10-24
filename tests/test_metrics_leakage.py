import numpy as np
import pytest

from triage.leakage import LeakageError, check_features
from triage.metrics import at_capacity, summarize, top_k_mask


def test_top_k_rounds_up_and_breaks_ties_by_position():
    s = np.array([0.5, 0.5, 0.5, 0.1])
    assert top_k_mask(s, 0.5).tolist() == [True, True, False, False]
    assert top_k_mask(s, 0.3).sum() == 2  # ceil(1.2)


def test_at_capacity_hand_computed():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    r = at_capacity(y, s, 0.2)          # top 2: labels 1, 0
    assert r["tp"] == 1 and r["precision"] == 0.5
    assert r["recall"] == pytest.approx(1 / 3)
    assert r["lift"] == pytest.approx(0.5 / 0.3)


def test_summarize_perfect_ranking():
    y = np.array([0] * 90 + [1] * 10)
    s = np.arange(100) / 100
    out = summarize(y, s)
    assert out["pr_auc"] == pytest.approx(1.0)
    assert out["recall@10pct"] == pytest.approx(1.0)
    assert out["precision@5pct"] == pytest.approx(1.0)


def test_leakage_forbidden_column_raises():
    with pytest.raises(LeakageError, match="Timely response"):
        check_features(["text", "Timely response?"])


def test_leakage_unknown_column_raises():
    with pytest.raises(LeakageError, match="State"):
        check_features(["text", "State"])


def test_leakage_allowed_columns_pass():
    assert check_features(["text", "product", "issue", "company"])["passed"]


def test_company_mask_phrases_and_boundaries():
    import pandas as pd

    from triage.company_mask import PLACEHOLDER, CompanyMasker, build_phrases

    ph = build_phrases(["TRANSUNION INTERMEDIATE HOLDINGS, INC."] * 60 + ["Rare Lender LLC"] * 3, min_count=50)
    assert "transunion" in ph and "rare lender" not in ph
    m = CompanyMasker(ph)
    out = m.transform(pd.Series(["TransUnion and Bank-of-America, then td. Bank; chaser; the bank"])).iloc[0]
    assert out == f"{PLACEHOLDER} and {PLACEHOLDER}, then {PLACEHOLDER}. Bank; chaser; the bank"
