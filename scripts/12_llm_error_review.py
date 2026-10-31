"""LLM failure analysis: give each gold error a category in the terminal, one at a time.

Shows the narrative, the field, the gold label, the model's value and its evidence quote, then
asks for a category key and an optional comment. Saves reports/llm/gold_errors.csv after every
error, so it can be stopped with q and resumed. Errors are shown in a fixed shuffled order, so
the first 20 or 30 reviewed are a fair mix of fields. Gold labels are never changed here: if the
gold label looks wrong now, that is recorded with the g key.

Keys: a m o r g v (see CATEGORIES), b goes back one error, q (or Ctrl+C) saves and quits.
"""
from __future__ import annotations

import argparse
import os
import shutil
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = {
    "a": "label_ambiguity",      # the rule allows both readings
    "m": "missed_statement",     # the narrative says it, the model did not pick it up
    "o": "over_reading",         # the model inferred more than the text says
    "r": "rule_misapplied",      # the model found the passage but applied the rule wrongly
    "g": "gold_label_wrong",     # on a second look the gold label is the mistake
    "v": "evidence_not_verbatim",
}
SEED = 20261031


def save(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def ask(prompt: str, valid: set[str]) -> str:
    while True:
        try:
            a = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return "q"
        if a in valid:
            return a
        print("  type one of: " + " ".join(sorted(valid)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--errors", default="reports/llm/gold_errors.csv")
    ap.add_argument("--texts", default="data/llm/gold_errors_with_text.csv")
    args = ap.parse_args()
    path = Path(args.errors)
    err = pd.read_csv(path, dtype=str, keep_default_na=False)
    full = pd.read_csv(args.texts, dtype=str, keep_default_na=False)
    if len(full) != len(err) or list(full["item_id"]) != list(err["item_id"]):
        raise SystemExit(f"{args.texts} does not line up with {path}")
    order = np.random.default_rng(SEED).permutation(len(err))
    width = min(shutil.get_terminal_size((100, 20)).columns, 110)
    valid_cat = set(CATEGORIES.values())
    help_line = "  ".join(f"{k}={v}" for k, v in CATEGORIES.items())

    def n_done() -> int:
        return int(err["error_category"].isin(valid_cat).sum())

    pos = next((k for k, i in enumerate(order) if err.at[i, "error_category"] not in valid_cat), None)
    if pos is None:
        print(f"All {len(err)} errors in {path} have a category.")
        return
    while pos is not None and pos < len(order):
        i = order[pos]
        os.system("clear")
        print(f"error {pos + 1}/{len(err)}   ({n_done()} categorized)   {err.at[i, 'item_id']}\n")
        for para in full.at[i, "text"].split("\n"):
            print(textwrap.fill(para, width))
        print(f"\nFIELD  {err.at[i, 'field']}")
        print(f"GOLD   {err.at[i, 'gold']}")
        print(f"MODEL  {err.at[i, 'pred']}")
        print(f"QUOTE  {full.at[i, 'quote'] or '(none)'}\n")
        print(help_line + "  b=back q=quit")
        old = err.at[i, "error_category"]
        a = ask(f"Category{' (was ' + old + ')' if old else ''}: ", set(CATEGORIES) | {"b", "q"})
        if a == "q":
            break
        if a == "b":
            pos = max(pos - 1, 0)
            continue
        try:
            comment = input("Comment (optional, Enter to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nError not saved.")
            break
        err.at[i, "error_category"] = CATEGORIES[a]
        if comment:
            err.at[i, "comment"] = comment
        save(err, path)
        pos += 1
    save(err, path)
    print(f"\nSaved. {n_done()}/{len(err)} errors categorized in {path}.")
    if n_done():
        print(err.loc[err["error_category"] != "", ["field", "error_category"]].value_counts().to_string())


if __name__ == "__main__":
    main()
