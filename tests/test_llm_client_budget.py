import json

import pytest

from triage.llm.budget import Budget, BudgetExceeded, call_cost
from triage.llm.cache import DiskCache, cache_key
from triage.llm.client import Extractor, Reply
from triage.llm.prompts import parse_prompt
from triage.llm.schema import BOOL_FIELDS

MODEL = "claude-haiku-5-5"
PROMPT = parse_prompt("## system\nExtract.\n\n## user\n{narrative}\n", "v_test")
GOOD = json.dumps({"amounts_mentioned": [], **{f: {"value": None, "evidence": None} for f in BOOL_FIELDS}})


class FakeBackend:
    """Returns scripted replies and records every call; no network."""

    def __init__(self, replies, model=MODEL):
        self.model = model
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, messages, max_tokens):
        self.calls.append(messages)
        text = self.replies.pop(0)
        return Reply(text, input_tokens=1000, output_tokens=100, stop_reason="end_turn", model=self.model)


def make(tmp_path, replies, cap=10.0, max_calls=None):
    b = FakeBackend(replies)
    budget = Budget(tmp_path / "spend.json", cap, max_calls)
    return b, Extractor(b, PROMPT, DiskCache(tmp_path / "cache"), budget, max_tokens=200)


def test_cache_hit_makes_no_call(tmp_path):
    b, ex = make(tmp_path, [GOOD])
    r1 = ex.extract("some text")
    r2 = ex.extract("some text")
    assert r1 == r2 and r1["valid"] and len(b.calls) == 1
    assert ex.stats == {"cached": 1, "called": 1}


def test_cache_key_separates_runs_and_prompts():
    k = cache_key(MODEL, "v1", "a" * 64, 1, "t")
    assert k != cache_key(MODEL, "v1", "a" * 64, 2, "t")
    assert k != cache_key(MODEL, "v1", "b" * 64, 1, "t")
    assert k != cache_key("other", "v1", "a" * 64, 1, "t")


def test_one_retry_then_valid(tmp_path):
    b, ex = make(tmp_path, ["not json", GOOD])
    r = ex.extract("text")
    assert r["valid"] and len(r["attempts"]) == 2
    retry_msgs = b.calls[1]
    assert [m["role"] for m in retry_msgs] == ["user", "assistant", "user"]
    assert "did not pass validation" in retry_msgs[-1]["content"]


def test_two_failures_recorded_as_invalid(tmp_path):
    b, ex = make(tmp_path, ["{}", "still bad"])
    r = ex.extract("text")
    assert not r["valid"] and r["output"] is None and len(b.calls) == 2


def test_non_end_turn_is_invalid(tmp_path):
    class Truncating(FakeBackend):
        def complete(self, system, messages, max_tokens):
            r = super().complete(system, messages, max_tokens)
            r.stop_reason = "max_tokens"
            return r
    b = Truncating([GOOD, GOOD])
    ex = Extractor(b, PROMPT, DiskCache(tmp_path / "c"), Budget(tmp_path / "s.json", 10.0), max_tokens=200)
    r = ex.extract("text")
    assert not r["valid"] and "stop_reason max_tokens" in r["attempts"][0]["errors"][0]


def test_ledger_adds_up_and_persists(tmp_path):
    _, ex = make(tmp_path, [GOOD, GOOD])
    ex.extract("a")
    ex.extract("b")
    expected = 2 * call_cost(MODEL, 1000, 100)
    assert ex.budget.spent == pytest.approx(expected)
    assert Budget(tmp_path / "spend.json", 10.0).spent == pytest.approx(expected)


def test_cap_blocks_before_calling(tmp_path):
    b, ex = make(tmp_path, [GOOD], cap=1e-9)
    with pytest.raises(BudgetExceeded):
        ex.extract("text")
    assert b.calls == []


def test_max_calls(tmp_path):
    b, ex = make(tmp_path, [GOOD, GOOD], max_calls=1)
    ex.extract("a")
    with pytest.raises(BudgetExceeded, match="max-calls"):
        ex.extract("b")
    assert len(b.calls) == 1


def test_unknown_model_price_refused():
    with pytest.raises(KeyError):
        call_cost("unpriced-model", 10, 10)
