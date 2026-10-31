"""Extraction calls: a thin backend per provider, plus the cached, budgeted, validated loop."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol

from triage.llm.budget import Budget
from triage.llm.cache import DiskCache, cache_key
from triage.llm.prompts import Prompt
from triage.llm.schema import JSON_SCHEMA, validate


@dataclass
class Reply:
    text: str
    input_tokens: int
    output_tokens: int
    stop_reason: str
    model: str  # as reported by the API


class Backend(Protocol):
    model: str

    def complete(self, system: str, messages: list[dict], max_tokens: int) -> Reply: ...


class AnthropicBackend:
    """Claude Messages API with JSON outputs constrained to JSON_SCHEMA. The key is read from
    ANTHROPIC_API_KEY by the SDK; it is never passed in code."""

    def __init__(self, model: str, thinking: str = "disabled", max_retries: int = 3):
        import anthropic  # imported here so tests and other steps do not need the package

        self.model = model
        self.thinking = thinking
        self._client = anthropic.Anthropic(max_retries=max_retries)

    def complete(self, system: str, messages: list[dict], max_tokens: int) -> Reply:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            thinking={"type": self.thinking},
            output_config={"format": {"type": "json_schema", "schema": JSON_SCHEMA}},
        )
        text = next((b.text for b in resp.content if b.type == "text"), "")
        return Reply(text, resp.usage.input_tokens, resp.usage.output_tokens, resp.stop_reason, resp.model)


def parse_reply(text: str) -> tuple[dict | None, list[str]]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return None, [f"not valid JSON: {e.msg}"]
    errors = validate(obj)
    return (obj if not errors else None), errors


@dataclass
class Extractor:
    backend: Backend
    prompt: Prompt
    cache: DiskCache
    budget: Budget
    max_tokens: int = 1024
    stats: dict = field(default_factory=lambda: {"cached": 0, "called": 0})

    def extract(self, text: str, run: int = 1) -> dict:
        """Return the cached record if there is one; otherwise call (at most twice) and cache."""
        key = cache_key(self.backend.model, self.prompt.version, self.prompt.sha256, run, text)
        hit = self.cache.get(key)
        if hit is not None:
            self.stats["cached"] += 1
            return hit
        user = self.prompt.render(text)
        messages = [{"role": "user", "content": user}]
        attempts = []
        obj = None
        for attempt in range(2):
            self.budget.check(self.backend.model, len(self.prompt.system) + sum(len(m["content"]) for m in messages),
                              self.max_tokens)
            t0 = time.perf_counter()
            r = self.backend.complete(self.prompt.system, messages, self.max_tokens)
            latency = time.perf_counter() - t0
            cost = self.budget.record(self.backend.model, r.input_tokens, r.output_tokens)
            obj, errors = parse_reply(r.text)
            if r.stop_reason not in ("end_turn", "stop_sequence"):
                errors = [f"stop_reason {r.stop_reason}", *errors]
                obj = None
            attempts.append({"reply": r.text, "errors": errors, "stop_reason": r.stop_reason,
                             "model_reported": r.model, "input_tokens": r.input_tokens,
                             "output_tokens": r.output_tokens, "cost_usd": cost, "latency_s": latency})
            if obj is not None:
                break
            messages = messages + [
                {"role": "assistant", "content": r.text or "(empty)"},
                {"role": "user", "content": "That output did not pass validation: " + "; ".join(errors)
                 + ". Return only a JSON object that matches the schema."},
            ]
        self.stats["called"] += 1
        record = {"key": key, "model": self.backend.model, "prompt_version": self.prompt.version,
                  "prompt_sha256": self.prompt.sha256, "run": run,
                  "called_at_utc": datetime.now(timezone.utc).isoformat(),
                  "valid": obj is not None, "output": obj, "attempts": attempts}
        self.cache.put(key, record)
        return record
