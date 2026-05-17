"""Conversation state used by the agent's state machine.

State is intentionally a plain dataclass rather than a heavy object — easier to
test, easier to reason about. The agent owns exactly one of these per session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Optional

from .models import Account


class Phase(str, Enum):
    """Top-level conversation phases.

    Sub-phases (e.g. which card field we're missing) are derived from the
    populated fields on `ConversationState` rather than enumerated separately.
    """

    GREETING = "greeting"
    AWAITING_ACCOUNT = "awaiting_account"
    AWAITING_VERIFICATION = "awaiting_verification"
    AWAITING_AMOUNT = "awaiting_amount"
    AWAITING_CARD = "awaiting_card"
    CONFIRM_PAYMENT = "confirm_payment"
    DONE_SUCCESS = "done_success"
    DONE_FAILED = "done_failed"
    DONE_TERMINATED = "done_terminated"


TERMINAL_PHASES = frozenset({Phase.DONE_SUCCESS, Phase.DONE_FAILED, Phase.DONE_TERMINATED})


@dataclass
class ConversationState:
    """All state for a single conversation."""

    phase: Phase = Phase.GREETING

    # Account context — set once after a successful lookup.
    account: Optional[Account] = None
    account_id: Optional[str] = None

    # Verification candidates supplied by the user. Stored separately from
    # `account` because we must not surface the account's true values to the
    # user, and we want to detect when the user supplies a new candidate.
    candidate_name: Optional[str] = None
    candidate_dob: Optional[str] = None
    candidate_aadhaar_last4: Optional[str] = None
    candidate_pincode: Optional[str] = None
    verification_attempts: int = 0
    verified: bool = False

    # Payment details.
    payment_amount: Optional[Decimal] = None
    pay_full_balance_intent: bool = False
    cardholder_name: Optional[str] = None
    card_number: Optional[str] = None
    cvv: Optional[str] = None
    expiry_month: Optional[int] = None
    expiry_year: Optional[int] = None
    payment_attempts: int = 0
    last_transaction_id: Optional[str] = None
    last_payment_error: Optional[str] = None

    # Conversation transcript (latest 12 turns, oldest first). Only used to
    # provide minimal context to the LLM extractor and for debugging.
    history: list[tuple[str, str]] = field(default_factory=list)

    # ---- Convenience helpers ------------------------------------------------

    def card_fields_complete(self) -> bool:
        return all(
            [
                self.cardholder_name,
                self.card_number,
                self.cvv,
                self.expiry_month is not None,
                self.expiry_year is not None,
            ]
        )

    def is_terminal(self) -> bool:
        return self.phase in TERMINAL_PHASES

    def push_history(self, role: str, message: str, *, max_len: int = 24) -> None:
        self.history.append((role, message))
        if len(self.history) > max_len:
            del self.history[: len(self.history) - max_len]

    def reset_card_fields(self) -> None:
        """Wipe collected card fields, e.g. after a retryable payment error."""

        self.card_number = None
        self.cvv = None
        self.expiry_month = None
        self.expiry_year = None
        # Keep cardholder_name and payment_amount — those are unlikely to be the
        # cause of `invalid_card` / `invalid_cvv` / `invalid_expiry`.
