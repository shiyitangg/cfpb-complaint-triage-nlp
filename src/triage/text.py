"""Narrative cleaning."""
from __future__ import annotations

import re

import pandas as pd

# CFPB redacts personal details with runs of X, including dates such as XX/XX/2019 or XX/XX/XXXX.
_MASK_DATE = re.compile(r"\b(?:X{2,}/)+(?:X{2,}|\d{2,4})\b")
_MASK = re.compile(r"\bX{2,}\b")
# Dollar amounts are written as {$1200.00}; keep the amount, drop the braces.
_BRACES = re.compile(r"\{(\$?[\d,\.]+)\}")
_WS = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\s]")


def clean_text(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str)
    s = s.str.replace(_MASK_DATE, " ", regex=True)
    s = s.str.replace(_MASK, " ", regex=True)
    s = s.str.replace(_BRACES, r"\1", regex=True)
    return s.str.replace(_WS, " ", regex=True).str.strip()


def normalize_for_dedup(s: pd.Series) -> pd.Series:
    """Lowercase, drop punctuation, collapse whitespace. Input should already be cleaned."""
    s = s.str.lower().str.replace(_NON_WORD, " ", regex=True)
    return s.str.replace(_WS, " ", regex=True).str.strip()
