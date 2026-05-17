"""Tests for strict identity verification."""

from __future__ import annotations

from decimal import Decimal

import pytest

from payment_agent.models import Account
from payment_agent.verification import VerificationOutcome, verify


@pytest.fixture
def account() -> Account:
    return Account(
        account_id="ACC1001",
        full_name="Nithin Jain",
        dob="1990-05-14",
        aadhaar_last4="4321",
        pincode="400001",
        balance=Decimal("1250.75"),
    )


def test_verified_with_dob(account):
    result = verify(account, full_name="Nithin Jain", dob="1990-05-14", aadhaar_last4=None, pincode=None)
    assert result.outcome == VerificationOutcome.VERIFIED
    assert result.secondary_used == "dob"


def test_verified_with_aadhaar(account):
    result = verify(account, full_name="Nithin Jain", dob=None, aadhaar_last4="4321", pincode=None)
    assert result.outcome == VerificationOutcome.VERIFIED
    assert result.secondary_used == "aadhaar_last4"


def test_verified_with_pincode(account):
    result = verify(account, full_name="Nithin Jain", dob=None, aadhaar_last4=None, pincode="400001")
    assert result.outcome == VerificationOutcome.VERIFIED


def test_name_mismatch_blocks(account):
    result = verify(account, full_name="nithin jain", dob="1990-05-14", aadhaar_last4=None, pincode=None)
    assert result.outcome == VerificationOutcome.NAME_MISMATCH


def test_missing_name(account):
    result = verify(account, full_name=None, dob="1990-05-14", aadhaar_last4=None, pincode=None)
    assert result.outcome == VerificationOutcome.MISSING_NAME


def test_need_secondary(account):
    result = verify(account, full_name="Nithin Jain", dob=None, aadhaar_last4=None, pincode=None)
    assert result.outcome == VerificationOutcome.NEED_SECONDARY


def test_secondary_mismatch_records_failed(account):
    result = verify(
        account,
        full_name="Nithin Jain",
        dob="1991-05-14",
        aadhaar_last4="0000",
        pincode=None,
    )
    assert result.outcome == VerificationOutcome.SECONDARY_MISMATCH
    assert set(result.failed_secondary) == {"dob", "aadhaar_last4"}


def test_partial_correct_secondary_verifies(account):
    # Even if one secondary is wrong, ANY correct match verifies.
    result = verify(
        account,
        full_name="Nithin Jain",
        dob="1990-05-14",
        aadhaar_last4="0000",
        pincode=None,
    )
    assert result.outcome == VerificationOutcome.VERIFIED


def test_whitespace_in_name_is_collapsed(account):
    result = verify(account, full_name="Nithin   Jain", dob="1990-05-14", aadhaar_last4=None, pincode=None)
    assert result.outcome == VerificationOutcome.VERIFIED
