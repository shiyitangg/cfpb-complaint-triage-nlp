"""Label definition. Positive = 'Closed with monetary relief'. This is a proxy label for the
eventual outcome, not a measure of complaint severity."""
from __future__ import annotations

import pandas as pd

POSITIVE = "Closed with monetary relief"
NEGATIVE = ("Closed with non-monetary relief", "Closed with explanation", "Closed")
# Outcome unknown: excluded, never treated as negative.
EXCLUDED = ("In progress", "Untimely response")


def make_label(response: pd.Series) -> pd.Series:
    """Return 1 / 0 / <NA> (excluded). Raises on any category not listed above, so a new or
    renamed category is never silently mapped."""
    known = {POSITIVE, *NEGATIVE, *EXCLUDED}
    seen = set(response.dropna().unique())
    unknown = seen - known
    if unknown:
        raise ValueError(f"unexpected 'Company response to consumer' values: {sorted(unknown)}")
    out = pd.Series(pd.NA, index=response.index, dtype="Int8")
    out[response == POSITIVE] = 1
    out[response.isin(NEGATIVE)] = 0
    return out
