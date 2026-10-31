"""Check that evidence quotes are copied from the narrative."""
from __future__ import annotations

import re

from triage.llm.schema import BOOL_FIELDS

_WS = re.compile(r"\s+")
_EDGE = "\"'“”‘’`.,;:!?…- "


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip()


def quote_status(text: str, quote: str | None) -> str:
    """'exact': substring after collapsing whitespace. 'loose': substring after also ignoring case
    and stray quote marks or punctuation at the ends. 'fail': neither. 'missing': no quote."""
    if quote is None or not quote.strip():
        return "missing"
    t, q = _norm(text), _norm(quote)
    if q in t:
        return "exact"
    q2 = q.strip(_EDGE).casefold()
    if q2 and q2 in t.casefold():
        return "loose"
    return "fail"


def field_status(text: str, value, quote: str | None) -> str:
    """Evidence outcome for one field. A non-null value needs a quote; a null value should have none."""
    if value is None:
        return "null_with_quote" if quote and quote.strip() else "null"
    s = quote_status(text, quote)
    return "unsupported" if s == "missing" else s


def check_record(text: str, obj: dict) -> dict:
    return {f: field_status(text, obj[f]["value"], obj[f]["evidence"]) for f in BOOL_FIELDS}
