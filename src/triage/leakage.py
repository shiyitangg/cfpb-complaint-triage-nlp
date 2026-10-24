"""Feature allow-list. Only fields known when a complaint enters the review queue may be used."""
from __future__ import annotations

from typing import Iterable

ALLOWED = {"text", "text_masked", "product", "sub_product", "issue", "sub_issue", "company"}

# Fields created during or after handling, plus identifiers and bookkeeping columns.
FORBIDDEN = {
    "Company response to consumer", "company_response", "Company public response", "company_public_response",
    "Timely response?", "timely", "Consumer disputed?", "consumer_disputed", "Date sent to company",
    "date_sent_to_company", "Tags", "Submitted via", "State", "ZIP code",
    "date_received", "label", "split", "complaint_id", "near_dup_train", "near_dup_val", "text_hash",
}


class LeakageError(ValueError):
    pass


def check_features(columns: Iterable[str]) -> dict:
    """Raise LeakageError if any column is forbidden or not on the allow-list. Returns the audit."""
    cols = list(columns)
    forbidden = sorted(set(cols) & FORBIDDEN)
    not_allowed = sorted(set(cols) - ALLOWED)
    audit = {"features": cols, "forbidden_used": forbidden, "not_in_allow_list": not_allowed,
             "passed": not forbidden and not not_allowed}
    if not audit["passed"]:
        raise LeakageError(f"leakage check failed: forbidden={forbidden}, not_in_allow_list={not_allowed}")
    return audit
