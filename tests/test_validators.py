"""Unit tests for the deterministic validation layer."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from payment_agent.errors import ValidationError
from payment_agent.validators import (
    is_amex_pan,
    luhn_check,
    normalise_aadhaar_last4,
    normalise_account_id,
    normalise_amount,
    normalise_card_number,
    normalise_cvv,
    normalise_dob,
    normalise_expiry,
    normalise_pincode,
)


class TestAccountId:
    def test_canonical(self):
        assert normalise_account_id("ACC1001") == "ACC1001"

    @pytest.mark.parametrize("raw", ["acc1001", "ACC 1001", "acc-1001", " acc_1001 "])
    def test_messy_forms(self, raw):
        assert normalise_account_id(raw) == "ACC1001"

    @pytest.mark.parametrize("raw", ["", "1001", "ABC1001", "ACC", "ACCXYZ"])
    def test_rejects(self, raw):
        with pytest.raises(ValidationError):
            normalise_account_id(raw)


class TestLuhnAndCardNumber:
    def test_known_good_pan(self):
        assert luhn_check("4532015112830366")

    def test_known_bad_pan(self):
        assert not luhn_check("4532015112830367")

    def test_normalises_spaces_and_dashes(self):
        assert normalise_card_number("4532-0151 1283 0366") == "4532015112830366"

    @pytest.mark.parametrize("raw", ["abc", "1234", "1234567890123", "453201511283036X"])
    def test_rejects(self, raw):
        with pytest.raises(ValidationError):
            normalise_card_number(raw)

    def test_amex_detection(self):
        assert is_amex_pan("378282246310005")
        assert not is_amex_pan("4532015112830366")


class TestCVV:
    def test_three_digit_default(self):
        assert normalise_cvv("123") == "123"

    def test_four_digit_amex(self):
        assert normalise_cvv("1234", is_amex=True) == "1234"

    @pytest.mark.parametrize("raw", ["12", "12345", "abc", ""])
    def test_rejects(self, raw):
        with pytest.raises(ValidationError):
            normalise_cvv(raw)


class TestExpiry:
    today = date(2026, 5, 17)

    def test_future_two_digit_year(self):
        assert normalise_expiry(12, 27, today=self.today) == (12, 2027)

    def test_future_four_digit_year(self):
        assert normalise_expiry("12", "2027", today=self.today) == (12, 2027)

    def test_past_year_rejected(self):
        with pytest.raises(ValidationError):
            normalise_expiry(12, 2020, today=self.today)

    def test_invalid_month(self):
        with pytest.raises(ValidationError):
            normalise_expiry(13, 2027, today=self.today)


class TestDOB:
    def test_iso(self):
        assert normalise_dob("1990-05-14") == "1990-05-14"

    def test_slashed(self):
        assert normalise_dob("14/05/1990") == "1990-05-14"

    def test_dotted(self):
        assert normalise_dob("14.05.1990") == "1990-05-14"

    def test_spoken_full(self):
        assert normalise_dob("14th May 1990") == "1990-05-14"

    def test_spoken_short_year(self):
        assert normalise_dob("May 14, 90") == "1990-05-14"

    def test_short_year_2k(self):
        assert normalise_dob("14-05-05") == "2005-05-14"

    def test_leap_year_valid(self):
        assert normalise_dob("1988-02-29") == "1988-02-29"

    def test_leap_year_invalid(self):
        with pytest.raises(ValidationError):
            normalise_dob("1989-02-29")

    def test_unparseable(self):
        with pytest.raises(ValidationError):
            normalise_dob("yesterday")


class TestAadhaarAndPincode:
    def test_last4_strips_spaces(self):
        assert normalise_aadhaar_last4("4 3 2 1") == "4321"

    def test_last4_from_full_aadhaar(self):
        assert normalise_aadhaar_last4("1234 5678 4321") == "4321"

    def test_pincode_six_digits(self):
        assert normalise_pincode("4 0 0 0 0 1") == "400001"

    def test_pincode_wrong_length(self):
        with pytest.raises(ValidationError):
            normalise_pincode("40001")


class TestAmount:
    def test_plain(self):
        assert normalise_amount("100") == Decimal("100.00")

    def test_decimal_two_places(self):
        assert normalise_amount("540.00") == Decimal("540.00")

    def test_strips_currency_and_commas(self):
        assert normalise_amount("₹1,250.75") == Decimal("1250.75")

    def test_zero_rejected(self):
        with pytest.raises(ValidationError):
            normalise_amount("0")

    def test_negative_rejected(self):
        with pytest.raises(ValidationError):
            normalise_amount("-50")

    def test_too_many_decimals(self):
        with pytest.raises(ValidationError):
            normalise_amount("10.123")
