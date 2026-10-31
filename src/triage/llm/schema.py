"""Output schema for the extraction prompt, and a validator that does not depend on the API."""
from __future__ import annotations

BOOL_FIELDS = ("prior_contact_with_company", "requested_refund_or_reimbursement", "fee_or_charge_dispute")

_NULLABLE_BOOL = {"anyOf": [{"type": "boolean"}, {"type": "null"}]}
_NULLABLE_STR = {"anyOf": [{"type": "string"}, {"type": "null"}]}
_FIELD = {
    "type": "object",
    "properties": {"value": _NULLABLE_BOOL, "evidence": _NULLABLE_STR},
    "required": ["value", "evidence"],
    "additionalProperties": False,
}

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "amounts_mentioned": {"type": "array", "items": {"type": "string"}},
        **{f: _FIELD for f in BOOL_FIELDS},
    },
    "required": ["amounts_mentioned", *BOOL_FIELDS],
    "additionalProperties": False,
}


def validate(obj) -> list[str]:
    """Return a list of problems; empty means the object matches JSON_SCHEMA."""
    if not isinstance(obj, dict):
        return ["top level is not an object"]
    errors = []
    expected = {"amounts_mentioned", *BOOL_FIELDS}
    for k in sorted(set(obj) - expected):
        errors.append(f"unexpected key {k!r}")
    for k in sorted(expected - set(obj)):
        errors.append(f"missing key {k!r}")
    amounts = obj.get("amounts_mentioned")
    if "amounts_mentioned" in obj and not (isinstance(amounts, list) and all(isinstance(a, str) for a in amounts)):
        errors.append("amounts_mentioned must be a list of strings")
    for f in BOOL_FIELDS:
        if f not in obj:
            continue
        v = obj[f]
        if not isinstance(v, dict):
            errors.append(f"{f} must be an object")
            continue
        if set(v) != {"value", "evidence"}:
            errors.append(f"{f} must have exactly the keys value and evidence")
            continue
        if v["value"] is not None and not isinstance(v["value"], bool):
            errors.append(f"{f}.value must be true, false or null")
        if v["evidence"] is not None and not isinstance(v["evidence"], str):
            errors.append(f"{f}.evidence must be a string or null")
    return errors
