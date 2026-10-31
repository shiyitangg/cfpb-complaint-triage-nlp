"""LLM step 3: evaluate extraction outputs.

--set dev   unlabeled checks only: schema validity, evidence, amounts vs regex, null rates,
            fee field vs CFPB Issue; --compare-prompt gives agreement with another version.
--set gold  the checks above plus hand labels. The first run writes
            reports/llm_eval_opened.lock; later runs stop unless --force is given, and forced
            runs are appended to reports/llm_eval_open_log.txt. Items whose output failed
            validation count as null predictions in precision/recall/F1.

Tables go to reports/llm/. Gold labels are copied there as Complaint ID plus values, no text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from triage.llm.amounts import model_amounts, regex_amounts
from triage.llm.evidence import check_record
from triage.llm.metrics import agreement, amount_agreement, bootstrap_f1, confusion, prf_true, to_level
from triage.llm.schema import BOOL_FIELDS

VALID_LABELS = {"true": True, "false": False, "null": None}


def load_outputs(data: Path, set_name: str, version: str, run: int) -> pd.DataFrame:
    path = data / "outputs" / f"{set_name}_{version}_run{run}.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found: run scripts/10_llm_extract.py --set {set_name} --run {run} first")
    o = pd.read_parquet(path)
    o["obj"] = o["output"].map(lambda s: json.loads(s) if isinstance(s, str) else None)
    return o


def values(o: pd.DataFrame, f: str) -> list:
    return [obj[f]["value"] if isinstance(obj, dict) else None for obj in o["obj"]]


def read_labels(path: Path) -> pd.DataFrame:
    lab = pd.read_csv(path, dtype=str, keep_default_na=False)
    bad = []
    for f in BOOL_FIELDS:
        v = lab[f].str.strip().str.lower()
        for i in lab.index[~v.isin(VALID_LABELS)]:
            bad.append(f"{lab.at[i, 'item_id']} {f}={lab.at[i, f]!r}")
        lab[f] = v.map(VALID_LABELS)
    if bad:
        raise SystemExit(f"{len(bad)} label cells are not true/false/null, for example: {bad[:5]}")
    return lab


def read_relabel(path: Path) -> pd.DataFrame | None:
    """None if the self-agreement file does not exist or has not been filled in yet."""
    if not path.exists():
        return None
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    if (raw[list(BOOL_FIELDS)].apply(lambda c: c.str.strip()) == "").all().all():
        return None
    return read_labels(path)


def unlabeled_checks(items: pd.DataFrame, o: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    m = items.merge(o, on=["item_id", "complaint_id"], how="inner")
    valid = m[m["valid"]]
    ev = pd.DataFrame([check_record(t, obj) for t, obj in zip(valid["text"], valid["obj"])])
    ev_table = pd.concat({f: ev[f].value_counts() for f in BOOL_FIELDS}, axis=1).fillna(0).astype(int)
    nonnull = ev.isin(["exact", "loose", "fail", "unsupported"])
    pred_sets, base_sets, unparsed, verbatim, n_strings = [], [], 0, 0, 0
    for t, obj in zip(valid["text"], valid["obj"]):
        vals, bad = model_amounts(obj["amounts_mentioned"])
        pred_sets.append(vals)
        base_sets.append(regex_amounts(t))
        unparsed += bad
        n_strings += len(obj["amounts_mentioned"])
        verbatim += sum(a.strip() in t for a in obj["amounts_mentioned"])
    amt = amount_agreement(pred_sets, base_sets)
    amt.update({"amount_strings": n_strings, "unparsed_strings": unparsed,
                "strings_found_verbatim": verbatim / n_strings if n_strings else float("nan")})
    fee = pd.crosstab(valid["issue"], [to_level(v) for v in values(valid, "fee_or_charge_dispute")])
    summary = {
        "items": len(m), "schema_valid_rate": float(m["valid"].mean()),
        "needed_retry_rate": float((m["attempts"] > 1).mean()),
        "null_rate_model": {f: float(np.mean([v is None for v in values(valid, f)])) for f in BOOL_FIELDS},
        "evidence_exact_rate_nonnull": float((ev[nonnull] == "exact").sum().sum() / nonnull.sum().sum())
        if nonnull.values.any() else float("nan"),
        "evidence_exact_or_loose_rate_nonnull": float(ev[nonnull].isin(["exact", "loose"]).sum().sum()
                                                       / nonnull.sum().sum()) if nonnull.values.any() else float("nan"),
        "amounts": amt,
        "cost_per_1000_items_usd": float(m["cost_usd"].mean() * 1000),
        "latency_median_s": float(m["latency_s"].median()), "latency_p95_s": float(m["latency_s"].quantile(0.95)),
        "tokens_mean": {"input": float(m["input_tokens"].mean()), "output": float(m["output_tokens"].mean())},
    }
    return summary, ev_table, fee


def show(items: pd.DataFrame, o: pd.DataFrame, which: str, chars: int) -> None:
    """Print outputs next to the start of each narrative, for reading during prompt iteration (dev only)."""
    m = items.merge(o, on=["item_id", "complaint_id"], how="inner")
    for r in m.itertuples(index=False):
        obj = r.obj if isinstance(r.obj, dict) else None
        status = check_record(r.text, obj) if obj else {}
        amt_model = model_amounts(obj["amounts_mentioned"])[0] if obj else set()
        amt_flag = obj is not None and amt_model != regex_amounts(r.text)
        problem = obj is None or amt_flag or any(v in ("fail", "loose", "unsupported", "null_with_quote")
                                                 for v in status.values())
        if which == "problems" and not problem:
            continue
        print(f"\n--- {r.item_id} | {r.product} | {len(r.text)} chars ---")
        print(r.text[:chars] + (" [...]" if len(r.text) > chars else ""))
        if obj is None:
            print("  INVALID OUTPUT")
            continue
        print(f"  amounts: {obj['amounts_mentioned']}" + (f"  regex: {sorted(regex_amounts(r.text))}" if amt_flag else ""))
        for f in BOOL_FIELDS:
            q = obj[f]["evidence"]
            print(f"  {f}: {to_level(obj[f]['value'])} [{status[f]}]" + (f' "{q[:160]}"' if q else ""))


def open_lock(lock: Path, frozen: Path, force: bool) -> None:
    cfg_hash = hashlib.sha256(frozen.read_bytes()).hexdigest()
    now = datetime.now(timezone.utc).isoformat()
    if lock.exists():
        if not force:
            raise SystemExit(f"{lock} exists: the gold set was already evaluated ({lock.read_text().strip()}). "
                             "Use --force only with a reason; the run will be logged.")
        with open(lock.parent / "llm_eval_open_log.txt", "a") as f:
            f.write(f"{now} forced re-open, frozen config sha256 {cfg_hash}\n")
    else:
        lock.write_text(json.dumps({"opened_at_utc": now, "frozen_config_sha256": cfg_hash}) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", choices=["dev", "gold"], required=True)
    ap.add_argument("--prompt-version", default="extract_v1")
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--compare-prompt", default=None, help="dev only: another prompt version, same run number")
    ap.add_argument("--data-dir", default="data/llm")
    ap.add_argument("--reports", default="reports/llm")
    ap.add_argument("--frozen", default="reports/llm/frozen_prompt.json")
    ap.add_argument("--lock", default="reports/llm_eval_opened.lock")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--show", choices=["problems", "all"], default=None, help="dev only: print outputs with text")
    ap.add_argument("--chars", type=int, default=400, help="narrative characters printed by --show")
    args = ap.parse_args()
    data, reports = Path(args.data_dir), Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    tag = f"{args.set}_{args.prompt_version}_run{args.run}"

    if args.set == "gold":
        frozen = json.loads(Path(args.frozen).read_text())
        if frozen["prompt_version"] != args.prompt_version:
            raise SystemExit(f"gold is evaluated only with the frozen prompt {frozen['prompt_version']}")
        if args.compare_prompt:
            raise SystemExit("older prompt versions are not scored on gold")
        if args.run not in (1, 2):
            raise SystemExit("gold has run 1 and run 2 only")
    # Everything that can fail is loaded before the gold lock is written.
    items = pd.read_parquet(data / f"{args.set}.parquet")
    o = load_outputs(data, args.set, args.prompt_version, args.run)
    if args.set == "gold":
        labels = read_labels(data / "to_label_gold.csv")
        missing = set(items["item_id"]) - set(labels["item_id"])
        if missing:
            raise SystemExit(f"{len(missing)} gold items have no label row, for example {sorted(missing)[:5]}")
        other = data / "outputs" / f"gold_{args.prompt_version}_run{3 - args.run}.parquet"
        o_other = load_outputs(data, "gold", args.prompt_version, 3 - args.run) if other.exists() else None
        relabel = read_relabel(data / "to_relabel_gold.csv")
        open_lock(Path(args.lock), Path(args.frozen), args.force)
    if args.show and args.set == "dev":
        show(items, o, args.show, args.chars)
    summary, ev_table, fee = unlabeled_checks(items, o)
    ev_table.to_csv(reports / f"{tag}_evidence.csv")
    fee.to_csv(reports / f"{tag}_fee_vs_issue.csv")

    if args.compare_prompt:
        o2 = load_outputs(data, args.set, args.compare_prompt, args.run).set_index("item_id").loc[o["item_id"]]
        summary["agreement_with_" + args.compare_prompt] = {f: agreement(values(o, f), values(o2, f)) for f in BOOL_FIELDS}

    if args.set == "gold":
        g = items[["item_id", "complaint_id"]].merge(labels.drop(columns=["text"]), on="item_id", how="left")
        g[["item_id", "complaint_id"]].assign(**{f: g[f].map(lambda v: to_level(None if pd.isna(v) else v))
                                                 for f in BOOL_FIELDS}).to_csv(reports / "gold_labels.csv", index=False)
        p = g.merge(o, on=["item_id", "complaint_id"], how="left")
        rows, confs = [], []
        for f in BOOL_FIELDS:
            gold, pred = [None if pd.isna(x) else x for x in g[f]], values(p, f)
            r = prf_true(gold, pred)
            r["f1_ci95_low"], r["f1_ci95_high"] = bootstrap_f1(gold, pred)
            r["null_rate_gold"] = float(np.mean([x is None for x in gold]))
            r["null_rate_pred"] = float(np.mean([x is None for x in pred]))
            r["three_level_agreement"] = agreement(gold, pred)
            rows.append({"field": f, **r})
            confs.append(confusion(gold, pred).assign(field=f))
        scores = pd.DataFrame(rows)
        scores.to_csv(reports / f"{tag}_scores.csv", index=False)
        pd.concat(confs).to_csv(reports / f"{tag}_confusion.csv")

        if o_other is not None:
            o2 = o_other.set_index("item_id").loc[o["item_id"]]
            summary["run_to_run_agreement"] = {f: agreement(values(o, f), values(o2, f)) for f in BOOL_FIELDS}

        if relabel is not None:
            r2 = relabel.set_index("item_id")
            first = labels.set_index("item_id").loc[r2.index]
            summary["self_agreement"] = {"items": len(r2), **{f: agreement(list(first[f]), list(r2[f]))
                                                              for f in BOOL_FIELDS}}

        text = items.set_index("item_id")["text"]
        err = []
        for f in BOOL_FIELDS:
            for it, gv, obj in zip(p["item_id"], g[f], p["obj"]):
                obj = obj if isinstance(obj, dict) else None
                gl, pl = to_level(None if pd.isna(gv) else gv), to_level(None if obj is None else obj[f]["value"])
                if gl != pl:
                    err.append({"item_id": it, "field": f, "gold": gl, "pred": pl, "valid_output": obj is not None,
                                "quote": None if obj is None else obj[f]["evidence"], "text": text[it]})
        err = pd.DataFrame(err)
        if len(err):
            err.drop(columns=["text"]).assign(quote=err["quote"].str[:100], error_category="", comment="").to_csv(
                reports / "gold_errors.csv", index=False)
            err.to_csv(data / "gold_errors_with_text.csv", index=False)
        summary["errors"] = len(err)
        pd.set_option("display.width", 250)
        pd.set_option("display.float_format", lambda v: f"{v:.4f}")
        print("=== Gold scores (positive class = true) ===")
        print(scores.to_string(index=False))

    print("\n=== Evidence status by field ===")
    print(ev_table.to_string())
    (reports / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
