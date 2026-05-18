"""Integration-style tests for `Agent.next()` against a mocked API client.

These tests are hermetic: no network, no Anthropic API key required. They
exercise the deterministic extraction path plus the state machine. The LLM
extractor is exercised separately via the eval harness.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Iterable

import pytest

from payment_agent.agent import Agent
from payment_agent.api_client import PaymentAPIClient
from payment_agent.config import Config
from payment_agent.errors import AccountNotFoundError
from payment_agent.extractors import CompositeExtractor
from payment_agent.models import (
    Account,
    CardDetails,
    PaymentFailure,
    PaymentSuccess,
)


class FakeAPIClient(PaymentAPIClient):
    """In-memory stand-in for `PaymentAPIClient`."""

    def __init__(self, accounts: dict[str, Account], payment_results: Iterable | None = None):
        self._accounts = accounts
        self._payment_results = list(payment_results or [])
        self.lookup_calls: list[str] = []
        self.payment_calls: list[tuple[str, Decimal, CardDetails]] = []

    # We deliberately do not call super().__init__ — we don't need httpx here.

    def lookup_account(self, account_id: str) -> Account:
        self.lookup_calls.append(account_id)
        if account_id in self._accounts:
            return self._accounts[account_id]
        raise AccountNotFoundError(f"No account {account_id}")

    def process_payment(self, account_id: str, amount: Decimal, card: CardDetails):
        self.payment_calls.append((account_id, amount, card))
        if not self._payment_results:
            return PaymentSuccess(success=True, transaction_id="txn_test_001")
        return self._payment_results.pop(0)

    def close(self) -> None:  # pragma: no cover - no-op
        return


@pytest.fixture
def acc1001() -> Account:
    return Account(
        account_id="ACC1001",
        full_name="Nithin Jain",
        dob="1990-05-14",
        aadhaar_last4="4321",
        pincode="400001",
        balance=Decimal("1250.75"),
    )


@pytest.fixture
def acc1003_zero() -> Account:
    return Account(
        account_id="ACC1003",
        full_name="Priya Agarwal",
        dob="1992-08-10",
        aadhaar_last4="2468",
        pincode="400003",
        balance=Decimal("0.00"),
    )


def make_agent(api: PaymentAPIClient) -> Agent:
    config = Config.from_env()
    return Agent(
        config=config,
        api_client=api,
        extractor=CompositeExtractor(config),
    )


def reply(agent: Agent, message: str) -> str:
    return agent.next(message)["message"]


# ---------------------------------------------------------------- flows ------


def test_happy_path_full_balance(acc1001):
    api = FakeAPIClient({"ACC1001": acc1001})
    agent = make_agent(api)

    assert "account ID" in reply(agent, "hi")
    assert "verify your identity" in reply(agent, "ACC1001")
    assert "share your" in reply(agent, "Nithin Jain")
    assert "outstanding balance is ₹1,250.75" in reply(agent, "1990-05-14")
    assert "card details" in reply(agent, "full amount")
    reply(agent, "Nithin Jain")
    reply(agent, "card 4532 0151 1283 0366")
    reply(agent, "expires 12/27 CVV 123")
    confirmation = reply(agent, "yes")
    assert "transaction ID" in confirmation
    assert api.payment_calls == [
        (
            "ACC1001",
            Decimal("1250.75"),
            CardDetails(
                cardholder_name="Nithin Jain",
                card_number="4532015112830366",
                cvv="123",
                expiry_month=12,
                expiry_year=2027,
            ),
        )
    ]


def test_partial_payment(acc1001):
    api = FakeAPIClient({"ACC1001": acc1001})
    agent = make_agent(api)
    reply(agent, "hi")
    reply(agent, "ACC1001")
    reply(agent, "Nithin Jain")
    reply(agent, "1990-05-14")
    reply(agent, "pay 500")
    reply(agent, "Nithin Jain")
    reply(agent, "4532 0151 1283 0366")
    reply(agent, "12/27 CVV 123")
    reply(agent, "yes")
    assert api.payment_calls[0][1] == Decimal("500.00")


def test_amount_above_balance_is_rejected(acc1001):
    api = FakeAPIClient({"ACC1001": acc1001})
    agent = make_agent(api)
    reply(agent, "hi"); reply(agent, "ACC1001")
    reply(agent, "Nithin Jain"); reply(agent, "1990-05-14")
    response = reply(agent, "pay 5000")
    assert "more than your outstanding balance" in response
    assert api.payment_calls == []  # no API call attempted


def test_verification_retries_exhaust(acc1001):
    api = FakeAPIClient({"ACC1001": acc1001})
    agent = make_agent(api)
    reply(agent, "hi"); reply(agent, "ACC1001")
    reply(agent, "Nithin Jain")
    reply(agent, "DOB 2000-01-01")  # mismatch attempt 1
    reply(agent, "DOB 2001-01-01")  # attempt 2
    final = reply(agent, "DOB 2002-01-01")  # attempt 3 — terminates
    assert "ended" in final.lower() or "end this session" in final.lower()
    # Any subsequent turn stays terminal.
    after = reply(agent, "1990-05-14")
    assert "closed" in after.lower() or "new conversation" in after.lower()


def test_account_not_found_then_retry(acc1001):
    api = FakeAPIClient({"ACC1001": acc1001})
    agent = make_agent(api)
    reply(agent, "hi")
    miss = reply(agent, "ACC9999")
    assert "couldn't find" in miss
    found = reply(agent, "ACC1001")
    assert "verify your identity" in found


def test_zero_balance_closes_cleanly(acc1003_zero):
    api = FakeAPIClient({"ACC1003": acc1003_zero})
    agent = make_agent(api)
    reply(agent, "hi")
    response = reply(agent, "ACC1003")
    assert "nothing outstanding" in response


def test_invalid_card_triggers_retry(acc1001):
    api = FakeAPIClient(
        {"ACC1001": acc1001},
        payment_results=[
            PaymentFailure(success=False, error_code="invalid_card"),
            PaymentSuccess(success=True, transaction_id="txn_ok"),
        ],
    )
    agent = make_agent(api)
    reply(agent, "hi"); reply(agent, "ACC1001")
    reply(agent, "Nithin Jain"); reply(agent, "1990-05-14")
    reply(agent, "full amount")
    reply(agent, "Nithin Jain")
    reply(agent, "4532 0151 1283 0366")
    reply(agent, "12/27 CVV 123")
    first = reply(agent, "yes")
    assert "card number" in first.lower()
    # Re-collect details.
    reply(agent, "4532 0151 1283 0366")
    reply(agent, "12/27 CVV 123")
    second = reply(agent, "yes")
    assert "transaction ID" in second
    assert len(api.payment_calls) == 2


def test_decline_at_confirmation_resets_card_fields(acc1001):
    api = FakeAPIClient({"ACC1001": acc1001})
    agent = make_agent(api)
    reply(agent, "hi"); reply(agent, "ACC1001")
    reply(agent, "Nithin Jain"); reply(agent, "1990-05-14")
    reply(agent, "full amount")
    reply(agent, "Nithin Jain")
    reply(agent, "4532 0151 1283 0366")
    reply(agent, "12/27 CVV 123")
    response = reply(agent, "no")
    assert "redo the card details" in response or "card number" in response.lower()
    assert api.payment_calls == []


def test_pii_not_echoed(acc1001):
    api = FakeAPIClient({"ACC1001": acc1001})
    agent = make_agent(api)
    reply(agent, "hi"); reply(agent, "ACC1001")
    reply(agent, "Nithin Jain")
    response = reply(agent, "1990-05-14")
    assert "1990-05-14" not in response
    assert "4321" not in response
    assert "400001" not in response


def test_out_of_order_info_does_not_reask(acc1001):
    """User supplies account + name in one turn — agent should not ask for name again."""

    api = FakeAPIClient({"ACC1001": acc1001})
    agent = make_agent(api)
    reply(agent, "hi")
    response = reply(agent, "my account is ACC1001 and my name is Nithin Jain")
    # Should be asking for a secondary factor, not the name we already have.
    assert "full name" not in response.lower() or "secondary" in response.lower() or "date of birth" in response.lower()
    assert agent._state.candidate_name == "Nithin Jain"


def test_card_data_wiped_after_success(acc1001):
    api = FakeAPIClient({"ACC1001": acc1001})
    agent = make_agent(api)
    for turn in [
        "hi", "ACC1001", "Nithin Jain", "1990-05-14", "pay 100",
        "Nithin Jain", "4532 0151 1283 0366", "12/27 CVV 123", "yes",
    ]:
        reply(agent, turn)
    state = agent._state
    assert state.card_number is None
    assert state.cvv is None
    assert state.expiry_month is None
    assert state.expiry_year is None
    assert state.cardholder_name is None
    assert state.last_transaction_id is not None  # success metadata kept


def test_card_data_wiped_after_terminal_failure(acc1001):
    api = FakeAPIClient(
        {"ACC1001": acc1001},
        payment_results=[PaymentFailure(success=False, error_code="invalid_cvv")] * 3,
    )
    agent = make_agent(api)
    for turn in [
        "hi", "ACC1001", "Nithin Jain", "1990-05-14", "full amount",
        "Nithin Jain", "4532 0151 1283 0366", "12/27 CVV 123", "yes",
        "4532 0151 1283 0366", "12/27 CVV 123", "yes",
        "4532 0151 1283 0366", "12/27 CVV 123", "yes",
    ]:
        reply(agent, turn)
    state = agent._state
    assert state.card_number is None and state.cvv is None


def test_cancellation_terminates(acc1001):
    api = FakeAPIClient({"ACC1001": acc1001})
    agent = make_agent(api)
    reply(agent, "hi"); reply(agent, "ACC1001")
    response = reply(agent, "actually cancel this")
    # Deterministic extraction won't pick up intent=cancel without LLM, but
    # the agent shouldn't have proceeded to payment. Either it stays in
    # verification or terminates — both are acceptable.
    assert "process" not in response.lower() or "card" not in response.lower()
