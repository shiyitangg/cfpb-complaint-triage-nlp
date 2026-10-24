"""Time-based split on Date received."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitDates:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __post_init__(self) -> None:
        order = [self.train_start, self.train_end, self.val_start, self.val_end, self.test_start, self.test_end]
        if not all(a <= b for a, b in zip(order, order[1:])):
            raise ValueError(f"split dates must be ordered: {order}")
        if not (self.train_end < self.val_start and self.val_end < self.test_start):
            raise ValueError("train, val and test windows must not overlap")


def assign_split(dates: pd.Series, d: SplitDates) -> pd.Series:
    """Return 'train' / 'val' / 'test', or <NA> for dates outside every window. Dates are
    compared at day level, so an end date includes that whole day."""
    day = dates.dt.normalize()
    out = pd.Series(pd.NA, index=dates.index, dtype="string")
    out[(day >= d.train_start) & (day <= d.train_end)] = "train"
    out[(day >= d.val_start) & (day <= d.val_end)] = "val"
    out[(day >= d.test_start) & (day <= d.test_end)] = "test"
    return out
