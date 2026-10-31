"""LLM step 1: draw the dev set (2022) and the gold set (2022 + 2024) and write the labeling CSV.

2022 rows come from the validation split, main setting. 2024 rows come from a separate run of
step 09a, a random sample of 2024 (data/interim/matched_2024.parquet), and are cleaned here with the
same rules as step 2, and rows whose normalized text matches any modeling row are dropped.
Before sampling, the outcome completeness of 2024 narratives is printed; if --skip-2024 is
given, the gold set is drawn from 2022 only. 2023 is never read.

Sampling is stratified by product within each window, with square-root allocation.
Outputs under data/llm/ (git-ignored, contain text) and an id-only manifest under reports/llm/.
With --relabel N, instead writes N gold items in a new order for the self-agreement pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from triage.labels import EXCLUDED, make_label
from triage.text import clean_text, normalize_for_dedup

SEED = 20261025
LABEL_COLS = ["prior_contact_with_company", "requested_refund_or_reimbursement", "fee_or_charge_dispute"]


def sqrt_allocation(counts: pd.Series, n: int) -> pd.Series:
    """Split n across groups in proportion to sqrt(count), largest remainder, capped at the group size."""
    counts = counts[counts > 0]
    alloc = pd.Series(0, index=counts.index)
    remaining = n
    active = counts.index
    while remaining > 0 and len(active):
        w = np.sqrt(counts[active])
        raw = w / w.sum() * remaining
        add = np.floor(raw).astype(int)
        rem = (raw - add).sort_values(ascending=False)
        add[rem.index[: remaining - add.sum()]] += 1
        add = np.minimum(add, counts[active] - alloc[active])
        alloc[active] += add
        remaining = n - alloc.sum()
        active = counts.index[alloc < counts]
        if add.sum() == 0:
            break
    return alloc


def stratified(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    alloc = sqrt_allocation(pool["product"].value_counts(), n)
    parts = [pool[pool["product"] == p].sample(k, random_state=seed) for p, k in alloc.items() if k > 0]
    return pd.concat(parts)


def text_hashes(s: pd.Series) -> pd.Series:
    return normalize_for_dedup(s).map(lambda t: hashlib.md5(t.encode()).digest())


def modeling_hashes(path: str) -> set:
    """Hashes of the normalized text of every modeling row, read in batches to bound memory."""
    out = set()
    for b in pq.ParquetFile(path).iter_batches(batch_size=100_000, columns=["text"]):
        out.update(text_hashes(b.column("text").to_pandas()))
    return out


def load_2024(matched: str, modeling_norm: set, min_chars: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["Complaint ID", "date_official", "narrative", "Product", "Issue", "Company response to consumer"]
    df = pd.read_parquet(matched, columns=cols)
    day = df["date_official"].dt.normalize()
    df = df[(day >= "2024-01-01") & (day <= "2024-12-31")]
    resp = df["Company response to consumer"]
    status = resp.fillna("<missing>").where(resp.isna() | resp.isin(EXCLUDED), "resolved")
    completeness = status.value_counts().rename_axis("status").reset_index(name="rows")
    completeness["share"] = (completeness["rows"] / len(df)).round(6)
    df = df.assign(label=make_label(resp))
    df = df[df["label"].notna()]
    df = df.assign(text=clean_text(df["narrative"]))
    df = df[df["text"].str.len() >= min_chars]
    df = df.sort_values(["date_official", "Complaint ID"]).drop_duplicates("Complaint ID", keep="first")
    df = df.assign(norm=text_hashes(df["text"]))
    df = df.drop_duplicates("norm", keep="first")
    df = df[~df["norm"].isin(modeling_norm)]
    out = df.rename(columns={"Complaint ID": "complaint_id", "date_official": "date_received",
                             "Product": "product", "Issue": "issue"})
    out = out.assign(complaint_id=out["complaint_id"].astype(str))
    out = out[["complaint_id", "date_received", "text", "product", "issue", "label"]].assign(window="2024")
    return out, completeness


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modeling", default="data/processed/modeling.parquet")
    ap.add_argument("--matched", default="data/interim/matched_2024.parquet")
    ap.add_argument("--min-chars", type=int, default=50)
    ap.add_argument("--max-chars", type=int, default=3000)
    ap.add_argument("--n-dev", type=int, default=50)
    ap.add_argument("--n-gold-2022", type=int, default=50)
    ap.add_argument("--n-gold-2024", type=int, default=50)
    ap.add_argument("--skip-2024", action="store_true")
    ap.add_argument("--check-only", action="store_true", help="print 2024 completeness and pool sizes, write nothing")
    ap.add_argument("--relabel", type=int, default=0)
    ap.add_argument("--out-dir", default="data/llm")
    ap.add_argument("--reports", default="reports/llm")
    args = ap.parse_args()
    out_dir, reports = Path(args.out_dir), Path(args.reports)

    if args.relabel:
        target = out_dir / "to_relabel_gold.csv"
        if target.exists():
            raise SystemExit(f"{target} already exists; it is not overwritten. Continue it with 09b_llm_label.py.")
        gold = pd.read_parquet(out_dir / "gold.parquet")
        pick = gold.sample(args.relabel, random_state=SEED + 1)
        sheet = pick[["item_id", "text"]].assign(**{c: "" for c in LABEL_COLS}, notes="")
        sheet.to_csv(out_dir / "to_relabel_gold.csv", index=False)
        print(f"wrote {out_dir / 'to_relabel_gold.csv'} ({len(sheet)} items, new order)")
        return

    val = pd.read_parquet(args.modeling, columns=["complaint_id", "date_received", "text", "product", "issue",
                                                  "label", "near_dup_train"], filters=[("split", "==", "val")])
    val = val[~val["near_dup_train"]].drop(columns=["near_dup_train"]).assign(window="2022")
    pools = {"2022": val.assign(complaint_id=val["complaint_id"].astype(str))}
    rows = []
    if not args.skip_2024:
        y24, completeness = load_2024(args.matched, modeling_hashes(args.modeling), args.min_chars)
        print("=== 2024 narratives (random sample from step 09a): outcome completeness ===")
        print(completeness.to_string(index=False))
        pools["2024"] = y24

    for w, p in pools.items():
        within = p["text"].str.len() <= args.max_chars
        rows.append({"window": w, "rows": len(p), "rows_within_max_chars": int(within.sum()),
                     "share_removed_by_max_chars": round(1 - within.mean(), 6), "products": p["product"].nunique()})
        pools[w] = p[within]
    pool_table = pd.DataFrame(rows)
    print("\n=== Pools ===")
    print(pool_table.to_string(index=False))
    if args.check_only:
        return

    dev = stratified(pools["2022"], args.n_dev, SEED)
    rest = pools["2022"][~pools["2022"]["complaint_id"].isin(dev["complaint_id"])]
    gold_parts = [stratified(rest, args.n_gold_2022, SEED)]
    if "2024" in pools:
        gold_parts.append(stratified(pools["2024"], args.n_gold_2024, SEED))
    gold = pd.concat(gold_parts)

    rng = np.random.default_rng(SEED)
    dev = dev.iloc[rng.permutation(len(dev))].reset_index(drop=True)
    gold = gold.iloc[rng.permutation(len(gold))].reset_index(drop=True)
    dev.insert(0, "item_id", [f"dev-{i + 1:03d}" for i in range(len(dev))])
    gold.insert(0, "item_id", [f"gold-{i + 1:03d}" for i in range(len(gold))])
    assert not set(dev["complaint_id"]) & set(gold["complaint_id"])

    label_path = out_dir / "to_label_gold.csv"
    if label_path.exists():
        raise SystemExit(f"{label_path} already exists; move it away first so labels are not overwritten")
    out_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    dev.to_parquet(out_dir / "dev.parquet", index=False)
    gold.to_parquet(out_dir / "gold.parquet", index=False)
    gold[["item_id", "text"]].assign(**{c: "" for c in LABEL_COLS}, notes="").to_csv(label_path, index=False)

    both = pd.concat([dev, gold], ignore_index=True)
    manifest = both[["item_id", "complaint_id", "window", "product", "issue"]].assign(text_chars=both["text"].str.len())
    manifest.to_csv(reports / "sample_manifest.csv", index=False)
    pool_table.to_csv(reports / "sample_pools.csv", index=False)
    if not args.skip_2024:
        completeness.to_csv(reports / "sample_2024_completeness.csv", index=False)

    ids = ",".join(manifest["complaint_id"].astype(str))
    print("\n=== Sample by product (columns: set, window) ===")
    print(pd.crosstab(manifest["product"], [manifest["item_id"].str[:-4], manifest["window"]], margins=True).to_string())
    print(json.dumps({"dev": len(dev), "gold": len(gold), "seed": SEED, "max_chars": args.max_chars,
                      "manifest_sha256": hashlib.sha256(ids.encode()).hexdigest(),
                      "labeling_csv": str(label_path)}, indent=2))


if __name__ == "__main__":
    main()
