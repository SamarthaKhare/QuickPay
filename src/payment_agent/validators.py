"""Deterministic validators for fields the agent collects.

The state machine calls these before any API request so we can give the user a
fast, friendly error rather than relying on the server to bounce a bad payload.
The server is still the source of truth — these checks just shorten the loop
for obvious problems and let us match the spec's documented error codes
(`invalid_amount`, `invalid_card`, `invalid_cvv`, `invalid_expiry`) instead of
whatever the live API decides to return for malformed input.
"""

from __future__ import annotations

import calendar
import re
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from .errors import ValidationError

# Public API ------------------------------------------------------------------

ACCOUNT_ID_RE = re.compile(r"^ACC\d{4,}$")


def normalise_account_id(raw: str) -> str:
    """Canonicalise an account ID string.

    Accepts inputs like `"ACC 1001"`, `"acc1001"`, or `"acc-1001"` and returns
    the canonical `ACC1001` form. Raises `ValidationError` if the result does
    not match the documented shape.
    """

    stripped = re.sub(r"[\s\-_]", "", raw or "").upper()
    if not ACCOUNT_ID_RE.match(stripped):
        raise ValidationError(
            "invalid_account_id",
            "That doesn't look like a valid account ID. It should start with ACC followed by digits, e.g. ACC1001.",
        )
    return stripped


def luhn_check(card_number: str) -> bool:
    """Return True if `card_number` (digits only) passes the Luhn algorithm."""

    digits = [int(c) for c in card_number if c.isdigit()]
    if len(digits) != len(card_number):
        return False
    if not 12 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for idx, digit in enumerate(digits):
        if idx % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def normalise_card_number(raw: str) -> str:
    """Strip whitespace/dashes from a PAN and validate length + Luhn.

    Raises `ValidationError("invalid_card", ...)` on failure.
    """

    digits = re.sub(r"[\s\-]", "", raw or "")
    if not digits.isdigit():
        raise ValidationError(
            "invalid_card",
            "The card number should be 13–19 digits with no letters or symbols.",
        )
    if not 13 <= len(digits) <= 19:
        raise ValidationError(
            "invalid_card",
            "That card number length doesn't look right — most cards are 16 digits.",
        )
    if not luhn_check(digits):
        raise ValidationError(
            "invalid_card",
            "That card number didn't pass our format check. Could you re-read it for me?",
        )
    return digits


def normalise_cvv(raw: str, *, is_amex: bool = False) -> str:
    """Validate a CVV. Amex cards use 4 digits; everything else uses 3."""

    digits = re.sub(r"\s", "", raw or "")
    expected = 4 if is_amex else 3
    if not digits.isdigit() or len(digits) != expected:
        raise ValidationError(
            "invalid_cvv",
            f"The CVV should be {expected} digits — the small number on the back of the card.",
        )
    return digits


def is_amex_pan(card_number: str) -> bool:
    return card_number.startswith(("34", "37"))


def normalise_expiry(month: int | str, year: int | str, *, today: date | None = None) -> tuple[int, int]:
    """Validate an expiry month/year pair.

    Accepts 2- or 4-digit years (`27` → `2027`, `2027` → `2027`). Raises
    `ValidationError("invalid_expiry", ...)` for non-numeric, out-of-range, or
    past-due values.
    """

    try:
        m = int(month)
        y = int(year)
    except (TypeError, ValueError) as exc:
        raise ValidationError("invalid_expiry", "Expiry month/year must be numbers.") from exc

    if y < 100:
        y += 2000
    if not 1 <= m <= 12:
        raise ValidationError("invalid_expiry", "Expiry month must be between 1 and 12.")
    if not 2000 <= y <= 2100:
        raise ValidationError("invalid_expiry", "Expiry year looks out of range.")

    last_day = calendar.monthrange(y, m)[1]
    expires_on = date(y, m, last_day)
    today = today or date.today()
    if expires_on < today:
        raise ValidationError("invalid_expiry", "That card has already expired.")
    return m, y


def normalise_dob(raw: str) -> str:
    """Normalise a date-of-birth string to `YYYY-MM-DD`.

    Accepts many user-friendly forms — ISO, slashes, words like `14th May 1990`
    or `May 14, 90`. Falls back to raising `ValidationError("invalid_dob", ...)`
    only if we genuinely cannot extract a valid calendar date. Leap-year dates
    like `1988-02-29` parse successfully; `1989-02-29` does not.
    """

    text = (raw or "").strip()
    if not text:
        raise ValidationError("invalid_dob", "I didn't catch a date there.")

    iso = _parse_iso_like(text)
    if iso is not None:
        return iso
    spoken = _parse_spoken_date(text)
    if spoken is not None:
        return spoken
    raise ValidationError(
        "invalid_dob",
        "I couldn't parse that as a date. Please share it like 1990-05-14 or 14 May 1990.",
    )


def normalise_aadhaar_last4(raw: str) -> str:
    digits = re.sub(r"\s", "", raw or "")
    digits = re.sub(r"\D", "", digits)
    if len(digits) < 4:
        raise ValidationError(
            "invalid_aadhaar_last4",
            "Please share the last 4 digits of your Aadhaar.",
        )
    return digits[-4:]


def normalise_pincode(raw: str) -> str:
    digits = re.sub(r"\s", "", raw or "")
    digits = re.sub(r"\D", "", digits)
    if len(digits) != 6:
        raise ValidationError(
            "invalid_pincode",
            "Indian pincodes are 6 digits — could you double-check?",
        )
    return digits


def normalise_amount(raw: object) -> Decimal:
    """Parse an amount into a `Decimal` with two-decimal precision.

    Rejects zero/negative values and amounts with more than two decimal places
    (the API would otherwise return `invalid_amount`).
    """

    if isinstance(raw, Decimal):
        amount = raw
    else:
        text = re.sub(r"[₹$,\s]", "", str(raw or ""))
        if not text:
            raise ValidationError("invalid_amount", "I didn't catch an amount.")
        try:
            amount = Decimal(text)
        except InvalidOperation as exc:
            raise ValidationError("invalid_amount", "That doesn't look like a valid amount.") from exc

    if amount <= 0:
        raise ValidationError("invalid_amount", "The amount needs to be greater than zero.")
    quantised = amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if quantised != amount:
        raise ValidationError(
            "invalid_amount",
            "Amounts can have at most two decimal places.",
        )
    return quantised


# Internal helpers ------------------------------------------------------------

_ISO_RE = re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$")
_DMY_RE = re.compile(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})$")
_MONTHS = {
    name.lower(): idx
    for idx, name in enumerate(calendar.month_name)
    if name
}
_MONTHS.update({name.lower(): idx for idx, name in enumerate(calendar.month_abbr) if name})


def _parse_iso_like(text: str) -> str | None:
    m = _ISO_RE.match(text)
    if m:
        y, mo, d = (int(p) for p in m.groups())
        return _format_if_valid(y, mo, d)
    m = _DMY_RE.match(text)
    if m:
        a, b, c = (int(p) for p in m.groups())
        # Disambiguate DD-MM-YY(YY) vs MM-DD-YY(YY): prefer DD-MM as the
        # assignment examples use Indian convention; if first is > 12, that
        # forces day-first anyway.
        if a > 12:
            d, mo = a, b
        elif b > 12:
            mo, d = a, b
        else:
            d, mo = a, b
        year = c if c >= 100 else (1900 + c if c >= 30 else 2000 + c)
        return _format_if_valid(year, mo, d)
    return None


def _parse_spoken_date(text: str) -> str | None:
    cleaned = re.sub(r"(?i)(\d+)(st|nd|rd|th)", r"\1", text)
    cleaned = re.sub(r"[,]", " ", cleaned).strip()
    tokens = [t for t in re.split(r"\s+", cleaned) if t]

    month: int | None = None
    numbers: list[tuple[int, int]] = []  # (value, token_length)
    for token in tokens:
        lower = token.lower()
        if lower in _MONTHS and month is None:
            month = _MONTHS[lower]
        elif token.isdigit():
            numbers.append((int(token), len(token)))

    if month is None or not numbers:
        return None

    # Prefer the 4-digit number as year; otherwise treat the larger value as year.
    year = next((val for val, length in numbers if length == 4), None)
    if year is None and len(numbers) >= 2:
        # Pick the "year-shaped" number: 2-digit values that don't fit as a day,
        # or the larger of two candidates.
        candidates = sorted(numbers, key=lambda nv: (-nv[1], -nv[0]))
        year_value, year_len = candidates[0]
        if year_len == 2:
            year = 1900 + year_value if year_value >= 30 else 2000 + year_value
        else:
            year = year_value
    elif year is None and len(numbers) == 1:
        return None

    day = next(
        (val for val, length in numbers if val != (year % 100 if year else None) and 1 <= val <= 31 and length <= 2),
        None,
    )
    if day is None:
        return None
    return _format_if_valid(year, month, day)


def _format_if_valid(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None
