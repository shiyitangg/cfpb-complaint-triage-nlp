import numpy as np
import pytest

from triage.cost import apply_threshold, baselines, best_threshold, cost_curve


Y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
S = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])


def test_baselines():
    assert baselines(Y, 10, 1) == {"review_none": 20, "review_all": 8}


def test_cost_curve_endpoints():
    cc = cost_curve(Y, S, 10, 1)
    assert cc["cost"].iloc[0] == 20   # review nothing: both positives missed
    assert cc["cost"].iloc[-1] == 8   # review all: eight false positives


def test_best_threshold_hand_computed():
    # review top 3 (0.9, 0.8, 0.7): FN = 0, FP = 1, cost 1, the minimum
    b = best_threshold(Y, S, 10, 1)
    assert b["reviewed"] == 3 and b["cost"] == 1 and b["threshold"] == pytest.approx(0.7)
    assert b["saving_vs_best_baseline"] == 7


def test_ties_cut_together():
    cc = cost_curve(np.array([1, 0, 0]), np.array([0.5, 0.5, 0.1]), 10, 1)
    assert cc["reviewed"].tolist() == [0, 2, 3]


def test_apply_threshold_matches_curve():
    r = apply_threshold(Y, S, 0.7, 10, 1)
    assert (r["tp"], r["fp"], r["fn"], r["cost"]) == (2, 1, 0, 1)


def test_high_ratio_prefers_reviewing_more():
    low = best_threshold(Y, S, 1, 1)["reviewed"]
    high = best_threshold(Y, S, 20, 1)["reviewed"]
    assert high >= low
