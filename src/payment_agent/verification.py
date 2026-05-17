"""Identity verification.

The assignment is explicit: a user is verified if and only if

    full name matches exactly AND at least one of
        (date of birth, last 4 of Aadhaar, pincode) matches.

Matching is strict — no fuzzy logic, no case folding for names. The verifier
returns structured outcomes so the state machine can decide how to respond
(prompt for a different factor, count a retry, terminate).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

from .models import Account


class VerificationOutcome(str, Enum):
    """High-level result of a verification check."""

    VERIFIED = "verified"
    MISSING_NAME = "missing_name"
    NAME_MISMATCH = "name_mismatch"
    NEED_SECONDARY = "need_secondary"
    SECONDARY_MISMATCH = "secondary_mismatch"


@dataclass(frozen=True)
class VerificationResult:
    outcome: VerificationOutcome
    # Which secondary factor (if any) succeeded — useful for telemetry, never
    # surfaced to the user.
    secondary_used: Optional[str] = None
    # When `outcome == SECONDARY_MISMATCH`, the names of factors the user
    # supplied that did not match. Used to drive retry messaging.
    failed_secondary: tuple[str, ...] = ()


def verify(
    account: Account,
    *,
    full_name: Optional[str],
    dob: Optional[str],
    aadhaar_last4: Optional[str],
    pincode: Optional[str],
) -> VerificationResult:
    """Run the verification rule against fields collected from the user.

    All inputs are pre-validated by the validators module — this function only
    performs equality checks and does not normalise further. The single
    intentional exception is whitespace collapsing inside `full_name`, since
    "Nithin   Jain" and "Nithin Jain" should be treated as the same string.
    """

    if not full_name:
        return VerificationResult(outcome=VerificationOutcome.MISSING_NAME)

    normalised_name = " ".join(full_name.split())
    if normalised_name != account.full_name:
        return VerificationResult(outcome=VerificationOutcome.NAME_MISMATCH)

    supplied: list[tuple[str, str, str]] = []
    if dob:
        supplied.append(("dob", dob, account.dob))
    if aadhaar_last4:
        supplied.append(("aadhaar_last4", aadhaar_last4, account.aadhaar_last4))
    if pincode:
        supplied.append(("pincode", pincode, account.pincode))

    if not supplied:
        return VerificationResult(outcome=VerificationOutcome.NEED_SECONDARY)

    failed: list[str] = []
    for label, user_value, account_value in supplied:
        if user_value == account_value:
            return VerificationResult(
                outcome=VerificationOutcome.VERIFIED,
                secondary_used=label,
            )
        failed.append(label)

    return VerificationResult(
        outcome=VerificationOutcome.SECONDARY_MISMATCH,
        failed_secondary=tuple(failed),
    )


def supplied_secondary_labels(
    dob: Optional[str], aadhaar_last4: Optional[str], pincode: Optional[str]
) -> Iterable[str]:
    """Return human-readable labels for whichever secondary factors are set."""

    if dob:
        yield "date of birth"
    if aadhaar_last4:
        yield "last 4 of Aadhaar"
    if pincode:
        yield "pincode"
