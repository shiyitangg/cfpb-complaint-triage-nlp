"""LLM step 2: run the extraction prompt on the dev or gold set, with cache and spend cap.

  --dry-run   render prompts and print the worst-case cost; no API call, no key needed
  --freeze    write reports/llm/frozen_prompt.json for the given prompt and settings; no calls

On --set gold the prompt must match the frozen hash. Outputs go to data/llm/outputs/.
Reruns read the cache, so they are free. The API key is read from ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from triage.llm.budget import Budget, BudgetExceeded, call_cost, estimate_tokens
from triage.llm.cache import DiskCache
from triage.llm.client import AnthropicBackend, Extractor
from triage.llm.evidence import check_record
from triage.llm.prompts import load_prompt
from triage.llm.schema import JSON_SCHEMA


def settings(args, prompt) -> dict:
    return {"provider": "anthropic", "model": args.model, "thinking": args.thinking, "max_tokens": args.max_tokens,
            "prompt_version": prompt.version, "prompt_sha256": prompt.sha256,
            "schema_sha256": hashlib.sha256(json.dumps(JSON_SCHEMA, sort_keys=True).encode()).hexdigest()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", choices=["dev", "gold"], required=True)
    ap.add_argument("--prompt", default="prompts/extract_v1.md")
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--thinking", default="disabled", choices=["disabled", "adaptive"])
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--n", type=int, default=None, help="only the first n items")
    ap.add_argument("--max-usd", type=float, default=10.0, help="total cap across all runs (ledger)")
    ap.add_argument("--max-calls", type=int, default=200, help="cap on new calls in this run")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--data-dir", default="data/llm")
    ap.add_argument("--frozen", default="reports/llm/frozen_prompt.json")
    args = ap.parse_args()
    data = Path(args.data_dir)
    prompt = load_prompt(args.prompt)
    cfg = settings(args, prompt)

    if args.freeze:
        fz = Path(args.frozen)
        if fz.exists():
            old = json.loads(fz.read_text())
            raise SystemExit(f"{fz} already exists (frozen {old['prompt_version']} at {old['frozen_at_utc']}); "
                             "the prompt is not refrozen")
        fz.parent.mkdir(parents=True, exist_ok=True)
        fz.write_text(json.dumps({**cfg, "frozen_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
        print(fz.read_text())
        return

    if args.set == "gold":
        fz = Path(args.frozen)
        if not fz.exists():
            raise SystemExit("gold runs need a frozen prompt: run with --freeze first")
        frozen = json.loads(fz.read_text())
        diff = {k: (frozen.get(k), v) for k, v in cfg.items() if frozen.get(k) != v}
        if diff:
            raise SystemExit(f"settings differ from the frozen config: {diff}")

    items = pd.read_parquet(data / f"{args.set}.parquet")
    if args.n:
        items = items.head(args.n)

    if args.dry_run:
        chars = np.array([len(prompt.system) + len(prompt.render(t)) for t in items["text"]])
        worst = sum(call_cost(args.model, estimate_tokens(c), args.max_tokens) for c in chars)
        print(json.dumps({"items": len(items), "prompt_chars_median": int(np.median(chars)),
                          "prompt_chars_max": int(chars.max()), "worst_case_usd_one_attempt": round(worst, 4),
                          "settings": cfg}, indent=2))
        print("\n=== First rendered prompt ===\n[system]\n" + prompt.system + "\n\n[user]\n" + prompt.render(items["text"].iloc[0])[:600])
        return

    budget = Budget(data / "spend.json", args.max_usd, args.max_calls)
    ex = Extractor(AnthropicBackend(args.model, thinking=args.thinking), prompt, DiskCache(data / "cache"), budget,
                   max_tokens=args.max_tokens)
    rows = []
    stopped = None
    for it in items.itertuples(index=False):
        try:
            rec = ex.extract(it.text, run=args.run)
        except BudgetExceeded as e:
            stopped = str(e)
            break
        a = rec["attempts"]
        row = {"item_id": it.item_id, "complaint_id": it.complaint_id, "valid": rec["valid"],
               "attempts": len(a), "input_tokens": sum(x["input_tokens"] for x in a),
               "output_tokens": sum(x["output_tokens"] for x in a), "cost_usd": sum(x["cost_usd"] for x in a),
               "latency_s": sum(x["latency_s"] for x in a), "model_reported": a[-1]["model_reported"],
               "called_at_utc": rec["called_at_utc"], "output": json.dumps(rec["output"]) if rec["valid"] else None}
        if rec["valid"]:
            row.update({f"evidence_{k}": v for k, v in check_record(it.text, rec["output"]).items()})
        rows.append(row)

    out = pd.DataFrame(rows)
    out_dir = data / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.set}_{prompt.version}_run{args.run}.parquet"
    out.to_parquet(path, index=False)

    ev_cols = [c for c in out.columns if c.startswith("evidence_")]
    ev = out[ev_cols].melt()["value"].value_counts() if ev_cols else pd.Series(dtype=int)
    summary = {
        "set": args.set, "prompt": prompt.version, "run": args.run, "items_done": len(out),
        "items_requested": len(items), "new_calls": ex.stats["called"], "from_cache": ex.stats["cached"],
        "schema_valid_rate": round(float(out["valid"].mean()), 4) if len(out) else None,
        "needed_retry": int((out["attempts"] > 1).sum()) if len(out) else 0,
        "input_tokens_mean": round(float(out["input_tokens"].mean()), 1) if len(out) else None,
        "output_tokens_mean": round(float(out["output_tokens"].mean()), 1) if len(out) else None,
        "cost_per_1000_items_usd": round(float(out["cost_usd"].mean()) * 1000, 4) if len(out) else None,
        "latency_median_s": round(float(out["latency_s"].median()), 3) if len(out) else None,
        "latency_p95_s": round(float(out["latency_s"].quantile(0.95)), 3) if len(out) else None,
        "evidence_status_counts": {k: int(v) for k, v in ev.items()},
        "ledger_total_usd": round(budget.spent, 4), "cap_usd": args.max_usd,
        "stopped_early": stopped, "output": str(path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
