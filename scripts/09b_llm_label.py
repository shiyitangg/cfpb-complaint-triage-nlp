"""LLM labeling tool: label the gold set in the terminal, one narrative at a time.

Shows one narrative, asks the three fields (t = true, f = false, n = null) and an optional
note, and saves the CSV after every item, so it can be stopped with q and resumed later.
Never shows model output. Rules are in docs/LLM_MODULE.md, section 3.

Keys at any question: t / f / n answer, b goes back one item, q (or Ctrl+C) saves and quits.
"""
from __future__ import annotations

import argparse
import os
import shutil
import textwrap
from pathlib import Path

import pandas as pd

FIELDS = [
    ("prior_contact_with_company", "Contacted the company before?"),
    ("requested_refund_or_reimbursement", "Asks for money back / paid out?"),
    ("fee_or_charge_dispute", "Disputes a specific fee or charge?"),
]
KEYS = {"t": "true", "f": "false", "n": "null"}
VALID = set(KEYS.values())


def save(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def ask(prompt: str) -> str:
    while True:
        try:
            a = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return "q"
        if a in KEYS or a in ("b", "q"):
            return a
        print("  type t, f, n (or b = back, q = quit)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", default="data/llm/to_label_gold.csv")
    ap.add_argument("--item", help="start at this item_id, even if it is already labeled")
    args = ap.parse_args()
    path = Path(args.file)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    width = min(shutil.get_terminal_size((100, 20)).columns, 110)

    def done(i: int) -> bool:
        return all(df.at[i, f] in VALID for f, _ in FIELDS)

    if args.item:
        hits = df.index[df["item_id"] == args.item].tolist()
        if not hits:
            raise SystemExit(f"{args.item} not found in {path}")
        i = hits[0]
    else:
        i = next((k for k in range(len(df)) if not done(k)), None)
    if i is None:
        print(f"All {len(df)} items in {path} are labeled.")
        return
    while i is not None and i < len(df):
        os.system("clear")
        n_done = sum(done(k) for k in range(len(df)))
        print(f"{df.at[i, 'item_id']}   ({n_done}/{len(df)} done)   t=true f=false n=null  b=back q=quit\n")
        for para in df.at[i, "text"].split("\n"):
            print(textwrap.fill(para, width))
        print()
        answers, back, quit_ = {}, False, False
        for f, question in FIELDS:
            old = df.at[i, f]
            a = ask(f"{question:<36} [{f}]{' (was ' + old + ')' if old in VALID else ''}: ")
            if a == "b":
                back = True
                break
            if a == "q":
                quit_ = True
                break
            answers[f] = KEYS[a]
        if quit_:
            break
        if back:
            i = max(i - 1, 0)
            continue
        try:
            note = input("Note (optional, Enter to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nItem not saved.")
            break
        for f, v in answers.items():
            df.at[i, f] = v
        if note:
            df.at[i, "notes"] = note
        save(df, path)
        i += 1
    save(df, path)
    n_done = sum(done(k) for k in range(len(df)))
    print(f"\nSaved. {n_done}/{len(df)} items labeled in {path}.")


if __name__ == "__main__":
    main()
