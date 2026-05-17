"""Pydantic models for the payment API and internal state."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Account(BaseModel):
    """Account record returned by `/api/lookup-account`."""

    model_config = ConfigDict(frozen=True)

    account_id: str
    full_name: str
    dob: str  # YYYY-MM-DD
    aadhaar_last4: str
    pincode: str
    balance: Decimal


class CardDetails(BaseModel):
    """Card details collected from the user before payment."""

    model_config = ConfigDict(frozen=True)

    cardholder_name: str
    card_number: str  # digits only, no spaces
    cvv: str
    expiry_month: int = Field(ge=1, le=12)
    expiry_year: int = Field(ge=2000, le=2100)


class PaymentMethod(BaseModel):
    type: Literal["card"] = "card"
    card: CardDetails


class ProcessPaymentRequest(BaseModel):
    account_id: str
    amount: Decimal
    payment_method: PaymentMethod


class PaymentSuccess(BaseModel):
    success: Literal[True]
    transaction_id: str


class PaymentFailure(BaseModel):
    success: Literal[False]
    error_code: str


class ExtractedFields(BaseModel):
    """Structured fields extracted from a single user turn.

    All fields are optional; the extractor populates only what it is confident
    in for this turn. The state machine then merges these into conversation
    state.
    """

    account_id: Optional[str] = None
    full_name: Optional[str] = None
    dob: Optional[str] = None  # normalized to YYYY-MM-DD when possible
    aadhaar_last4: Optional[str] = None
    pincode: Optional[str] = None
    payment_amount: Optional[Decimal] = None
    pay_full_balance: Optional[bool] = None
    cardholder_name: Optional[str] = None
    card_number: Optional[str] = None
    cvv: Optional[str] = None
    expiry_month: Optional[int] = None
    expiry_year: Optional[int] = None
    user_intent: Optional[
        Literal[
            "provide_info",
            "confirm",
            "decline",
            "cancel",
            "ask_help",
            "smalltalk",
            "out_of_scope",
        ]
    ] = None
