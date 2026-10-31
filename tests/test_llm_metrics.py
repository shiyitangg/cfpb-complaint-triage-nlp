import math

from triage.llm.metrics import agreement, amount_agreement, bootstrap_f1, confusion, prf_true


def test_prf_true_hand_computed():
    gold = [True, True, False, None, True]
    pred = [True, None, True, None, True]
    r = prf_true(gold, pred)
    assert (r["tp"], r["fp"], r["fn"]) == (2, 1, 1)
    assert r["precision"] == 2 / 3 and r["recall"] == 2 / 3 and math.isclose(r["f1"], 2 / 3)


def test_prf_no_positive_anywhere_is_nan():
    assert math.isnan(prf_true([None, False], [False, None])["f1"])


def test_confusion_has_all_levels():
    c = confusion([True, None], [None, None])
    assert list(c.index) == ["true", "false", "null"] and list(c.columns) == ["true", "false", "null"]
    assert c.loc["true", "null"] == 1 and c.loc["null", "null"] == 1 and c.values.sum() == 2


def test_agreement_treats_nan_as_null():
    assert agreement([True, None, False], [True, float("nan"), True]) == 2 / 3


def test_amount_agreement():
    r = amount_agreement([{1.0, 2.0}, set()], [{1.0}, set()])
    assert r["exact_set_match"] == 0.5 and r["precision_vs_regex"] == 0.5 and r["recall_vs_regex"] == 1.0


def test_bootstrap_interval_contains_point():
    gold = [True] * 10 + [False] * 10
    pred = [True] * 8 + [False] * 2 + [True] * 2 + [False] * 8
    lo, hi = bootstrap_f1(gold, pred, n_boot=500)
    assert lo <= prf_true(gold, pred)["f1"] <= hi
