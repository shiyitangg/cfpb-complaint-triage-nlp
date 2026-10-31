"""Regex baseline for dollar amounts, and parsing of amounts the model returns."""
from __future__ import annotations

import re

_AMOUNT = re.compile(r"\$\s?((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)")
_NUM = re.compile(r"^\$?\s?((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)$")


def parse_amount(s: str) -> float | None:
    """'$1,200.00' -> 1200.0. Returns None for anything that is not a plain dollar number."""
    m = _NUM.match(s.strip().rstrip("."))
    return round(float(m.group(1).replace(",", "")), 2) if m else None


def regex_amounts(text: str) -> set[float]:
    return {round(float(m.replace(",", "")), 2) for m in _AMOUNT.findall(text)}


def model_amounts(items: list[str]) -> tuple[set[float], int]:
    """Parsed values and the count of strings that could not be parsed."""
    vals = [parse_amount(a) for a in items]
    return {v for v in vals if v is not None}, sum(v is None for v in vals)
