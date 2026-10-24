import pandas as pd
import pytest

from triage.split import SplitDates, assign_split


def _dates(*xs):
    return SplitDates(*(pd.Timestamp(x) for x in xs))


def test_assign_split_boundaries():
    d = _dates("2018-01-01", "2021-12-31", "2022-01-01", "2022-12-31", "2023-01-01", "2023-12-31")
    s = pd.to_datetime(pd.Series(["2017-12-31", "2018-01-01", "2021-12-31 23:59", "2022-01-01", "2022-12-31",
                                  "2023-01-01", "2023-12-31 12:00", "2024-01-01"]), format="mixed")
    assert assign_split(s, d).tolist() == [pd.NA, "train", "train", "val", "val", "test", "test", pd.NA]


def test_overlapping_windows_raise():
    with pytest.raises(ValueError):
        _dates("2018-01-01", "2022-01-01", "2022-01-01", "2022-12-31", "2023-01-01", "2023-12-31")


def test_unordered_windows_raise():
    with pytest.raises(ValueError):
        _dates("2018-01-01", "2021-12-31", "2023-01-01", "2023-12-31", "2022-01-01", "2022-12-31")
