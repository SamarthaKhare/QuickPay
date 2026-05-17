"""Tests for the deterministic extraction layer.

LLM extraction is exercised in the eval harness (which can be run with a real
API key) — these unit tests cover only what regex/heuristics should catch
reliably, so they run hermetically in CI.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from payment_agent.extractors import DeterministicExtractor


@pytest.fixture
def extractor() -> DeterministicExtractor:
    return DeterministicExtractor()


class TestAccountId:
    @pytest.mark.parametrize(
        "raw",
        [
            "yeah my account number is ACC1001 I think",
            "it's ACC 1001",
            "account id: acc1001",
            "acc-1001",
        ],
    )
    def test_extracts(self, extractor, raw):
        assert extractor.extract(raw).account_id == "ACC1001"

    def test_no_account_in_other_text(self, extractor):
        assert extractor.extract("just paying my bill").account_id is None


class TestDOB:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1990-05-14", "1990-05-14"),
            ("DOB is 1990-05-14", "1990-05-14"),
            ("14-05-1990", "1990-05-14"),
            ("14th May 1990", "1990-05-14"),
        ],
    )
    def test_extracts(self, extractor, raw, expected):
        assert extractor.extract(raw).dob == expected


class TestCard:
    def test_extracts_spaced_pan(self, extractor):
        fields = extractor.extract("the card number is 4532 0151 1283 0366")
        assert fields.card_number == "4532015112830366"

    def test_expiry_slash_short_year(self, extractor):
        fields = extractor.extract("expires 12/27")
        assert (fields.expiry_month, fields.expiry_year) == (12, 2027)

    def test_expiry_word(self, extractor):
        fields = extractor.extract("expires December 2027")
        assert (fields.expiry_month, fields.expiry_year) == (12, 2027)

    def test_cvv_with_label(self, extractor):
        fields = extractor.extract("CVV is 123")
        assert fields.cvv == "123"

    def test_cvv_spoken(self, extractor):
        fields = extractor.extract("CVV is one two three")
        assert fields.cvv == "123"


class TestAadhaarPincode:
    def test_aadhaar_last4(self, extractor):
        fields = extractor.extract("last four of my Aadhaar is 4321")
        assert fields.aadhaar_last4 == "4321"

    def test_pincode_spaced(self, extractor):
        fields = extractor.extract("pincode? it's 4 0 0 0 0 1")
        assert fields.pincode == "400001"


class TestAmount:
    def test_word_amount_thousand(self, extractor):
        fields = extractor.extract("I want to pay a thousand rupees")
        assert fields.payment_amount == Decimal("1000.00")

    def test_explicit_amount(self, extractor):
        fields = extractor.extract("can I do 500 for now?")
        assert fields.payment_amount == Decimal("500.00")

    def test_full_balance(self, extractor):
        fields = extractor.extract("just clear the full amount")
        assert fields.pay_full_balance is True
