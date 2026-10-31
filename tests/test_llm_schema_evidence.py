import json

from triage.llm.amounts import model_amounts, parse_amount, regex_amounts
from triage.llm.client import parse_reply
from triage.llm.evidence import check_record, field_status, quote_status
from triage.llm.prompts import parse_prompt
from triage.llm.schema import BOOL_FIELDS, validate

import pytest

GOOD = {"amounts_mentioned": ["$25.00"],
        **{f: {"value": None, "evidence": None} for f in BOOL_FIELDS}}
GOOD["fee_or_charge_dispute"] = {"value": True, "evidence": "charged me a $25.00 fee"}
TEXT = "The bank   charged me a $25.00 fee twice. I called them on the phone."


def test_valid_object_passes():
    assert validate(GOOD) == []
    obj, errors = parse_reply(json.dumps(GOOD))
    assert obj == GOOD and errors == []


@pytest.mark.parametrize("bad, fragment", [
    ({**GOOD, "extra": 1}, "unexpected key"),
    ({k: v for k, v in GOOD.items() if k != "amounts_mentioned"}, "missing key"),
    ({**GOOD, "amounts_mentioned": [25.0]}, "list of strings"),
    ({**GOOD, "fee_or_charge_dispute": {"value": "yes", "evidence": None}}, "true, false or null"),
    ({**GOOD, "fee_or_charge_dispute": {"value": True}}, "exactly the keys"),
])
def test_invalid_objects(bad, fragment):
    assert any(fragment in e for e in validate(bad))


def test_parse_reply_not_json():
    obj, errors = parse_reply("Sure! Here is the JSON:")
    assert obj is None and "not valid JSON" in errors[0]


def test_quote_status():
    assert quote_status(TEXT, "charged me a $25.00 fee") == "exact"
    assert quote_status(TEXT, "bank charged me") == "exact"            # whitespace collapsed
    assert quote_status(TEXT, '"I CALLED THEM on the phone."') == "loose"
    assert quote_status(TEXT, "I phoned the bank") == "fail"           # paraphrase
    assert quote_status(TEXT, "charged me ... twice") == "fail"        # ellipsis join
    assert quote_status(TEXT, "  ") == "missing"


def test_field_status_null_and_unsupported():
    assert field_status(TEXT, None, None) == "null"
    assert field_status(TEXT, None, "I called them") == "null_with_quote"
    assert field_status(TEXT, True, None) == "unsupported"
    assert check_record(TEXT, GOOD)["fee_or_charge_dispute"] == "exact"


def test_amounts():
    assert regex_amounts("paid $1,200.00 and then $45 and $45.00, not 300 dollars") == {1200.0, 45.0}
    assert parse_amount("$1,200.00") == 1200.0
    assert parse_amount("300 dollars") is None
    assert model_amounts(["$1,200.00", "$45", "three hundred"]) == ({1200.0, 45.0}, 1)


def test_prompt_parsing():
    p = parse_prompt("## system\nBe careful.\n\n## user\nText: {narrative}\n", "v_test")
    assert p.system == "Be careful." and p.render("abc") == "Text: abc" and len(p.sha256) == 64
    with pytest.raises(ValueError):
        parse_prompt("## system\nx\n## user\nno placeholder\n", "bad")
    with pytest.raises(ValueError):
        parse_prompt("## user\n{narrative}\n", "bad")


def test_repo_prompt_loads():
    from pathlib import Path
    from triage.llm.prompts import load_prompt
    p = load_prompt(Path(__file__).parents[1] / "prompts" / "extract_v1.md")
    assert p.version == "extract_v1" and "{narrative}" not in p.render("x")
