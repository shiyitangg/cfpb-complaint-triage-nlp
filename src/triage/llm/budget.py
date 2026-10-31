"""Spend cap. Prices are USD per million tokens; check them on the provider's pricing page on
the day of the run and record that date in docs/LLM_MODULE.md."""
from __future__ import annotations

import json
from pathlib import Path

# model: (input, output) USD per 1M tokens, for prompts up to 100K tokens.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
}
MAX_PROMPT_TOKENS = 100_000


class BudgetExceeded(RuntimeError):
    pass


def call_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in PRICES:
        raise KeyError(f"no price for {model}; add it to triage.llm.budget.PRICES first")
    if input_tokens > MAX_PROMPT_TOKENS:
        raise ValueError(f"prompt of {input_tokens} tokens is above the price tier in PRICES")
    pin, pout = PRICES[model]
    return (input_tokens * pin + output_tokens * pout) / 1e6


def estimate_tokens(chars: int) -> int:
    """Deliberately high: about 2.5 characters per token, where English prose is closer to 4."""
    return int(chars / 2.5) + 50


class Budget:
    """Ledger of actual spend, kept on disk so the cap holds across runs."""

    def __init__(self, ledger: str | Path, cap_usd: float, max_calls: int | None = None):
        self.ledger = Path(ledger)
        self.cap = cap_usd
        self.max_calls = max_calls
        self.calls_this_run = 0
        self.state = json.loads(self.ledger.read_text()) if self.ledger.exists() else {
            "total_usd": 0.0, "calls": 0, "input_tokens": 0, "output_tokens": 0}

    @property
    def spent(self) -> float:
        return self.state["total_usd"]

    def check(self, model: str, prompt_chars: int, max_tokens: int) -> None:
        if self.max_calls is not None and self.calls_this_run >= self.max_calls:
            raise BudgetExceeded(f"reached --max-calls {self.max_calls}")
        worst = call_cost(model, estimate_tokens(prompt_chars), max_tokens)
        if self.spent + worst > self.cap:
            raise BudgetExceeded(f"spent {self.spent:.4f} USD; a call could cost up to {worst:.4f}; cap {self.cap:.2f}")

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        cost = call_cost(model, input_tokens, output_tokens)
        s = self.state
        s["total_usd"] += cost
        s["calls"] += 1
        s["input_tokens"] += input_tokens
        s["output_tokens"] += output_tokens
        self.calls_this_run += 1
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ledger.with_suffix(".tmp")
        tmp.write_text(json.dumps(s, indent=1))
        tmp.replace(self.ledger)
        return cost
