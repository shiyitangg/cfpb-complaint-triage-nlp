import pandas as pd
import pytest

from triage.labels import make_label
from triage.text import clean_text, normalize_for_dedup


def test_label_mapping():
    r = pd.Series(["Closed with monetary relief", "Closed with explanation", "Closed with non-monetary relief",
                   "Closed", "In progress", "Untimely response", None])
    assert make_label(r).tolist() == [1, 0, 0, 0, pd.NA, pd.NA, pd.NA]


def test_unknown_category_raises():
    with pytest.raises(ValueError, match="Closed with relief"):
        make_label(pd.Series(["Closed with relief"]))


def test_mask_removal():
    s = pd.Series(["On XX/XX/2019 XXXX charged me {$1200.00} at XXXX XXXX bank."])
    assert clean_text(s).iloc[0] == "On charged me $1200.00 at bank."


def test_normalize():
    assert normalize_for_dedup(pd.Series(["Hello,  World!"])).iloc[0] == "hello world"
