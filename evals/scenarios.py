"""Scripted evaluation scenarios.

Each scenario is a structured description of an expected conversation. The
runner replays the user turns into a fresh `Agent`, captures every agent
response, and grades the run against the scenario's expectations.

Scenarios are scripted (not LLM-driven) so they form a deterministic regression
suite that runs in CI. The LLM-driven persona evaluator lives in
`persona_eval.py` for richer, exploratory testing when an API key is present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable, Optional

from payment_agent.models import Account, PaymentFailure, PaymentSuccess
from payment_agent.state import Phase


# ---------------------------------------------------------------- fixtures ---

ACC1001 = Account(
    account_id="ACC1001",
    full_name="Nithin Jain",
    dob="1990-05-14",
    aadhaar_last4="4321",
    pincode="400001",
    balance=Decimal("1250.75"),
)
ACC1002 = Account(
    account_id="ACC1002",
    full_name="Rajarajeswari Balasubramaniam",
    dob="1985-11-23",
    aadhaar_last4="9876",
    pincode="400002",
    balance=Decimal("540.00"),
)
ACC1003_ZERO = Account(
    account_id="ACC1003",
    full_name="Priya Agarwal",
    dob="1992-08-10",
    aadhaar_last4="2468",
    pincode="400003",
    balance=Decimal("0.00"),
)
ACC1004_LEAP = Account(
    account_id="ACC1004",
    full_name="Rahul Mehta",
    dob="1988-02-29",
    aadhaar_last4="1357",
    pincode="400004",
    balance=Decimal("3200.50"),
)

DEFAULT_ACCOUNTS = {
    "ACC1001": ACC1001,
    "ACC1002": ACC1002,
    "ACC1003": ACC1003_ZERO,
    "ACC1004": ACC1004_LEAP,
}


# ---------------------------------------------------------- scenario types ---


@dataclass
class Expect:
    """Expectations a scenario asserts about a run."""

    final_phase: Phase
    min_payment_calls: int = 0
    max_payment_calls: int = 99
    min_lookup_calls: int = 0
    max_lookup_calls: int = 99
    payment_amount: Optional[Decimal] = None
    last_transaction_id_present: bool = False
    must_contain: list[str] = field(default_factory=list)
    must_not_contain_in_any_message: list[str] = field(default_factory=list)
    # Hook for custom assertions: `(state, transcript, api) -> Optional[str]`.
    custom: Optional[Callable] = None


@dataclass
class Scenario:
    name: str
    description: str
    accounts: dict[str, Account]
    user_turns: list[str]
    expect: Expect
    payment_results: list[PaymentSuccess | PaymentFailure] = field(default_factory=list)


# -------------------------------------------------------------- scenarios ----


def all_scenarios() -> list[Scenario]:
    return [
        _happy_path_full_balance(),
        _happy_path_partial(),
        _verification_exhaustion(),
        _name_mismatch_then_recover(),
        _account_not_found_then_retry(),
        _zero_balance_close(),
        _leap_year_dob_acc1004(),
        _payment_invalid_card_then_success(),
        _payment_invalid_cvv_terminal(),
        _amount_above_balance_local_block(),
        _messy_phrasing_full_flow(),
        _out_of_order_info(),
    ]


def _out_of_order_info() -> Scenario:
    return Scenario(
        name="out_of_order_info",
        description="User volunteers account + name in one turn; agent must not re-ask.",
        accounts=DEFAULT_ACCOUNTS,
        user_turns=[
            "Hi I am Nithin Jain account ACC1001",
            "1990-05-14",
            "pay 100",
            "Nithin Jain",
            "4532 0151 1283 0366",
            "12/27 CVV 123",
            "yes",
        ],
        expect=Expect(
            final_phase=Phase.DONE_SUCCESS,
            min_payment_calls=1,
            max_payment_calls=1,
            payment_amount=Decimal("100.00"),
        ),
    )


def _happy_path_full_balance() -> Scenario:
    return Scenario(
        name="happy_path_full_balance",
        description="User pays the full balance on ACC1001 in a clean flow.",
        accounts=DEFAULT_ACCOUNTS,
        user_turns=[
            "hi",
            "my account is ACC1001",
            "Nithin Jain",
            "1990-05-14",
            "full amount",
            "Nithin Jain",
            "card 4532 0151 1283 0366",
            "expires 12/27 CVV 123",
            "yes",
        ],
        expect=Expect(
            final_phase=Phase.DONE_SUCCESS,
            min_payment_calls=1,
            max_payment_calls=1,
            payment_amount=Decimal("1250.75"),
            last_transaction_id_present=True,
            must_contain=["Payment", "transaction ID"],
            must_not_contain_in_any_message=["1990-05-14", "4321", "400001"],
        ),
    )


def _happy_path_partial() -> Scenario:
    return Scenario(
        name="happy_path_partial",
        description="User pays a partial amount.",
        accounts=DEFAULT_ACCOUNTS,
        user_turns=[
            "hi",
            "ACC1001",
            "Nithin Jain",
            "1990-05-14",
            "pay 500",
            "Nithin Jain",
            "4532 0151 1283 0366",
            "12/27 CVV 123",
            "yes",
        ],
        expect=Expect(
            final_phase=Phase.DONE_SUCCESS,
            min_payment_calls=1,
            max_payment_calls=1,
            payment_amount=Decimal("500.00"),
            last_transaction_id_present=True,
        ),
    )


def _verification_exhaustion() -> Scenario:
    return Scenario(
        name="verification_exhaustion",
        description="Three bad secondary factors → terminate.",
        accounts=DEFAULT_ACCOUNTS,
        user_turns=[
            "hi",
            "ACC1001",
            "Nithin Jain",
            "DOB 2000-01-01",
            "DOB 2001-01-01",
            "DOB 2002-01-01",
            # Post-terminal turn should remain closed.
            "1990-05-14",
        ],
        expect=Expect(
            final_phase=Phase.DONE_TERMINATED,
            max_payment_calls=0,
            min_lookup_calls=1,
            max_lookup_calls=1,
        ),
    )


def _name_mismatch_then_recover() -> Scenario:
    return Scenario(
        name="name_mismatch_then_recover",
        description="Wrong name first, then correct name and DOB.",
        accounts=DEFAULT_ACCOUNTS,
        user_turns=[
            "hi",
            "ACC1001",
            "Nithan Jain",  # one-letter typo, must NOT verify
            "Nithin Jain",
            "1990-05-14",
            "pay 100",
            "Nithin Jain",
            "4532 0151 1283 0366",
            "12/27 CVV 123",
            "yes",
        ],
        expect=Expect(
            final_phase=Phase.DONE_SUCCESS,
            min_payment_calls=1,
            payment_amount=Decimal("100.00"),
        ),
    )


def _account_not_found_then_retry() -> Scenario:
    return Scenario(
        name="account_not_found_then_retry",
        description="First account ID missing, then user corrects it.",
        accounts=DEFAULT_ACCOUNTS,
        user_turns=[
            "hi",
            "ACC9999",
            "ACC1001",
            "Nithin Jain",
            "1990-05-14",
            "pay 100",
            "Nithin Jain",
            "4532 0151 1283 0366",
            "12/27 CVV 123",
            "yes",
        ],
        expect=Expect(
            final_phase=Phase.DONE_SUCCESS,
            min_lookup_calls=2,
            max_lookup_calls=2,
            min_payment_calls=1,
        ),
    )


def _zero_balance_close() -> Scenario:
    return Scenario(
        name="zero_balance_close",
        description="ACC1003 has ₹0 — agent should close without payment.",
        accounts=DEFAULT_ACCOUNTS,
        user_turns=["hi", "ACC1003"],
        expect=Expect(
            final_phase=Phase.DONE_SUCCESS,
            max_payment_calls=0,
            must_contain=["nothing outstanding"],
        ),
    )


def _leap_year_dob_acc1004() -> Scenario:
    return Scenario(
        name="leap_year_dob_acc1004",
        description="Leap-year DOB 1988-02-29 is valid; off-by-one should fail.",
        accounts=DEFAULT_ACCOUNTS,
        user_turns=[
            "hi",
            "ACC1004",
            "Rahul Mehta",
            "1988-02-28",  # close but wrong
            "1988-02-29",  # actual leap-year DOB
            "pay 200",
            "Rahul Mehta",
            "4532 0151 1283 0366",
            "12/27 CVV 123",
            "yes",
        ],
        expect=Expect(
            final_phase=Phase.DONE_SUCCESS,
            min_payment_calls=1,
            payment_amount=Decimal("200.00"),
        ),
    )


def _payment_invalid_card_then_success() -> Scenario:
    return Scenario(
        name="payment_invalid_card_then_success",
        description="API reports invalid_card; agent reprompts and second try succeeds.",
        accounts=DEFAULT_ACCOUNTS,
        payment_results=[
            PaymentFailure(success=False, error_code="invalid_card"),
            PaymentSuccess(success=True, transaction_id="txn_retry_ok"),
        ],
        user_turns=[
            "hi",
            "ACC1001",
            "Nithin Jain",
            "1990-05-14",
            "full amount",
            "Nithin Jain",
            "4532 0151 1283 0366",
            "12/27 CVV 123",
            "yes",
            "4532 0151 1283 0366",
            "12/27 CVV 123",
            "yes",
        ],
        expect=Expect(
            final_phase=Phase.DONE_SUCCESS,
            min_payment_calls=2,
            max_payment_calls=2,
        ),
    )


def _payment_invalid_cvv_terminal() -> Scenario:
    return Scenario(
        name="payment_invalid_cvv_terminal",
        description="API keeps returning invalid_cvv; eventually terminates.",
        accounts=DEFAULT_ACCOUNTS,
        payment_results=[
            PaymentFailure(success=False, error_code="invalid_cvv"),
            PaymentFailure(success=False, error_code="invalid_cvv"),
            PaymentFailure(success=False, error_code="invalid_cvv"),
        ],
        user_turns=[
            "hi",
            "ACC1001",
            "Nithin Jain",
            "1990-05-14",
            "full amount",
            "Nithin Jain",
            "4532 0151 1283 0366",
            "12/27 CVV 123",
            "yes",
            "4532 0151 1283 0366",
            "12/27 CVV 123",
            "yes",
            "4532 0151 1283 0366",
            "12/27 CVV 123",
            "yes",
        ],
        expect=Expect(
            final_phase=Phase.DONE_FAILED,
            min_payment_calls=3,
        ),
    )


def _amount_above_balance_local_block() -> Scenario:
    return Scenario(
        name="amount_above_balance_local_block",
        description="User asks for more than balance — agent reprompts without an API call.",
        accounts=DEFAULT_ACCOUNTS,
        user_turns=[
            "hi",
            "ACC1001",
            "Nithin Jain",
            "1990-05-14",
            "pay 5000",
            "pay 500",
            "Nithin Jain",
            "4532 0151 1283 0366",
            "12/27 CVV 123",
            "yes",
        ],
        expect=Expect(
            final_phase=Phase.DONE_SUCCESS,
            min_payment_calls=1,
            max_payment_calls=1,
            payment_amount=Decimal("500.00"),
        ),
    )


def _messy_phrasing_full_flow() -> Scenario:
    """The cleanest deterministic-only flow we can muster from the spec examples.

    For richer messy-phrasing tests, see the LLM persona evaluator — that path
    needs an Anthropic API key and is run separately.
    """

    return Scenario(
        name="messy_phrasing_full_flow",
        description="Mixed phrasing through the flow (deterministic-friendly variants).",
        accounts=DEFAULT_ACCOUNTS,
        user_turns=[
            "hi there",
            "yeah my account number is ACC 1001 I think",
            "my name is Nithin Jain",
            "DOB is 1990-05-14",
            "I want to pay a thousand rupees",
            "name on card is Nithin Jain",
            "card number 4532 0151 1283 0366",
            "expiry 12/27 CVV one two three",
            "yes please",
        ],
        expect=Expect(
            final_phase=Phase.DONE_SUCCESS,
            min_payment_calls=1,
            payment_amount=Decimal("1000.00"),
        ),
    )
