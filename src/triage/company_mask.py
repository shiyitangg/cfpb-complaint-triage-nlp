"""Mask company names inside narratives, for the "is the model memorizing companies?" ablation.

Phrases come from (a) company names seen in the training split with at least `min_count`
complaints, reduced to a core name by dropping legal suffixes, plus the first word of that
core name when it is not a generic word, and (b) a short hand-written list of common
abbreviations. The phrase list is fitted on train only and saved with the model.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import pandas as pd

PLACEHOLDER = "zzcompanyzz"

ALIASES = [
    "bofa", "boa", "b of a", "bank of america", "wells", "wf", "wells fargo", "chase", "jpmorgan", "jp morgan",
    "jpmc", "jpmcb", "citi", "citibank", "amex", "american express", "capital one", "cap one", "td", "td bank",
    "usaa", "pnc", "synchrony", "navient", "nelnet", "ocwen", "paypal", "venmo", "cash app", "cashapp", "chime",
    "coinbase", "experian", "equifax", "transunion", "trans union", "discover", "barclays", "hsbc", "ally",
    "sofi", "mr cooper", "nationstar", "pennymac", "loancare", "amerisave", "netspend", "rocket mortgage", "quicken",
]

_SUFFIX = {"inc", "llc", "na", "n", "a", "corp", "corporation", "co", "company", "ltd", "lp", "llp", "plc",
           "holdings", "holding", "group", "the", "fsb", "financial", "services", "intermediate", "association",
           "national", "usa", "us", "bancorp", "incorporated", "limited"}
_GENERIC = {"bank", "first", "national", "american", "capital", "citizens", "credit", "financial", "united", "us",
            "the", "new", "home", "community", "federal", "state", "mortgage", "services", "service", "loan",
            "loans", "america", "one", "general", "global", "pacific", "western", "eastern", "southern",
            "northern", "central", "fifth", "third", "great", "security", "guaranty", "peoples", "farmers",
            "union", "savings", "trust", "consumer", "auto", "student", "card", "cards", "money", "cash", "pay",
            "debt", "collection", "collections", "recovery", "portfolio", "acceptance", "finance", "lending",
            "group", "holdings", "company", "midland", "select", "premier", "regional", "summit", "freedom",
            "integrity", "advance", "world", "direct", "capitol", "city", "county", "investment", "investments"}
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")


def _norm(name: str) -> list[str]:
    return _NON_ALNUM.sub(" ", name.lower()).split()


def build_phrases(train_companies: Iterable[str], min_count: int = 50) -> list[str]:
    counts = Counter(c for c in train_companies if isinstance(c, str))
    phrases = set(ALIASES)
    for name, n in counts.items():
        if n < min_count:
            continue
        w = _norm(name)
        while w and w[-1] in _SUFFIX:
            w = w[:-1]
        if not w:
            continue
        phrases.add(" ".join(w))
        if w[0] not in _GENERIC and len(w[0]) >= 3 and w[0].isalpha():
            phrases.add(w[0])
    return sorted(p for p in phrases if p)


def _trie_pattern(phrases: list[str]) -> str:
    """Regex alternation built as a character trie, so each position branches on its next
    character instead of trying every phrase in turn. A space in a phrase matches any run of
    spaces or hyphens ("bank of america", "bank-of-america"), never across punctuation such as ". "."""
    trie: dict = {}
    for p in phrases:
        node = trie
        for ch in p:
            node = node.setdefault(ch, {})
        node[""] = {}

    def emit(node: dict) -> str:
        end = "" in node
        alts = []
        for ch, child in sorted((k, v) for k, v in node.items() if k):
            alts.append((r"[ \t\-]+" if ch == " " else re.escape(ch)) + emit(child))
        if not alts:
            return ""
        body = alts[0] if len(alts) == 1 else "(?:" + "|".join(alts) + ")"
        if end:
            # Shorter phrase is complete here: the longer continuation is optional, tried first.
            return "(?:" + body + ")?"
        return body

    return emit(trie)


class CompanyMasker:
    def __init__(self, phrases: list[str]):
        self.phrases = phrases
        self.regex = re.compile(r"(?<![A-Za-z0-9])" + _trie_pattern(phrases) + r"(?![A-Za-z0-9])", re.IGNORECASE)

    def transform(self, texts: pd.Series) -> pd.Series:
        return texts.str.replace(self.regex, PLACEHOLDER, regex=True)
