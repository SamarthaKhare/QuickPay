"""Natural-language extraction layer.

The agent's state machine never parses raw user text directly. Instead, each
turn passes through this extractor which produces an `ExtractedFields` object.
That separation has two big benefits:

1. The state machine stays simple and testable — it operates on structured
   data, not strings.
2. The LLM's role is narrowly scoped to "turn messy text into JSON". It does
   not decide what to ask next, what to validate, or when to call APIs.

We layer two extractors:

- `DeterministicExtractor`: regex/heuristic pass that handles obvious cases
  (account IDs, ISO dates, plain card numbers). Cheap, deterministic, and
  works without an API key — useful for tests.
- `LLMExtractor`: Claude-powered extraction via a forced tool call. Handles
  the messy phrasing real users produce. Falls back to deterministic output
  when the API key is missing or the call fails.

The `CompositeExtractor` runs both and merges them: deterministic results take
precedence for fields where regex is sufficient and clearly correct, LLM
results fill in the rest. Every field is re-validated by `validators.py`
before being stored in conversation state.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from typing import Iterable

from .config import Config
from .errors import ValidationError
from .logging_utils import get_logger
from .models import ExtractedFields
from .prompts import EXTRACTION_TOOL, SYSTEM_PROMPT, build_user_message
from .validators import (
    is_amex_pan,
    normalise_aadhaar_last4,
    normalise_account_id,
    normalise_amount,
    normalise_card_number,
    normalise_cvv,
    normalise_dob,
    normalise_expiry,
    normalise_pincode,
)

_LOG = get_logger(__name__)


# Deterministic patterns ------------------------------------------------------

_ACCOUNT_ID_RE = re.compile(r"(?i)\bACC[\s\-_]*(\d{4,})\b")
_CARD_RE = re.compile(r"(?:\d[ -]?){12,18}\d")
_AMOUNT_RE = re.compile(r"(?i)(?:₹|rs\.?|inr)?\s*(\d[\d,]*(?:\.\d{1,2})?)\s*(?:rupees|rs|inr)?")
_FULL_PAY_RE = re.compile(r"(?i)\b(full|entire|outstanding|clear (?:the|my) balance|everything)\b")
_PINCODE_RE = re.compile(r"(?i)\bpin\s*(?:code)?[^\d]{0,10}((?:\d\s?){6})")
_AADHAAR_LAST4_RE = re.compile(r"(?i)aadhaa?r[^\d]{0,20}(\d[\d ]{2,})")
_EXPIRY_SLASH_RE = re.compile(r"(?<!\d)(\d{1,2})\s*/\s*(\d{2,4})(?!\d)")
_EXPIRY_WORD_RE = re.compile(
    r"(?i)(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)[ ,/]+((?:19|20)?\d{2})"
)
_CVV_RE = re.compile(r"(?i)\b(?:cvv2?|cvc|security\s*code)\b[^\d]{0,10}(\d{3,4})\b")
_SPOKEN_DIGITS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}
_MONTH_NUM = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _spoken_to_digits(text: str) -> str:
    """Collapse runs of spoken digits into contiguous numerals.

    "CVV is one two three" → "CVV is 123". Single isolated digit words (e.g.
    "one ticket please") are left alone to avoid stealing semantics.
    """

    tokens = re.split(r"(\s+|[,])", text)
    out: list[str] = []
    buf_digits: list[str] = []
    buf_seps: list[str] = []  # separators consumed while in a digit run

    def flush() -> None:
        if len(buf_digits) >= 2:
            out.append("".join(buf_digits))
        else:
            out.extend(buf_digits)
            out.extend(buf_seps)
        buf_digits.clear()
        buf_seps.clear()

    for tok in tokens:
        word = tok.lower().strip()
        if word in _SPOKEN_DIGITS:
            # Carrying separators belongs only when a run is forming.
            buf_digits.append(_SPOKEN_DIGITS[word])
        elif tok.isspace() or tok == ",":
            if buf_digits:
                buf_seps.append(tok)
            else:
                out.append(tok)
        else:
            flush()
            out.append(tok)
    flush()
    return "".join(out)


class DeterministicExtractor:
    """Regex-driven extraction for the unambiguous cases."""

    def extract(self, user_input: str) -> ExtractedFields:
        text = _spoken_to_digits(user_input)
        fields: dict[str, object] = {}

        if m := _ACCOUNT_ID_RE.search(text):
            try:
                fields["account_id"] = normalise_account_id(f"ACC{m.group(1)}")
            except ValidationError:
                pass

        if _FULL_PAY_RE.search(text):
            fields["pay_full_balance"] = True

        if m := _PINCODE_RE.search(text):
            try:
                fields["pincode"] = normalise_pincode(m.group(1))
            except ValidationError:
                pass

        if m := _AADHAAR_LAST4_RE.search(text):
            try:
                fields["aadhaar_last4"] = normalise_aadhaar_last4(m.group(1))
            except ValidationError:
                pass

        if m := _CVV_RE.search(text):
            fields["cvv"] = m.group(1)

        if m := _CARD_RE.search(text):
            try:
                fields["card_number"] = normalise_card_number(m.group(0))
            except ValidationError:
                pass

        # Expiry: prefer MM/YY-style; fall back to "Month YYYY".
        if m := _EXPIRY_SLASH_RE.search(text):
            try:
                month, year = normalise_expiry(m.group(1), m.group(2))
                fields["expiry_month"] = month
                fields["expiry_year"] = year
            except ValidationError:
                pass
        elif m := _EXPIRY_WORD_RE.search(text):
            try:
                month = _MONTH_NUM[m.group(1)[:3].lower()]
                month, year = normalise_expiry(month, m.group(2))
                fields["expiry_month"] = month
                fields["expiry_year"] = year
            except (KeyError, ValidationError):
                pass

        # DOB: only try if the message clearly looks date-shaped.
        dob_candidate = self._extract_dob_candidate(text)
        if dob_candidate is not None:
            try:
                fields["dob"] = normalise_dob(dob_candidate)
            except ValidationError:
                pass

        amount = self._extract_amount(text, has_card=bool(fields.get("card_number")))
        if amount is not None:
            fields["payment_amount"] = amount

        return ExtractedFields(**fields)

    @staticmethod
    def _extract_dob_candidate(text: str) -> str | None:
        # ISO-like or slashed dates.
        for pat in (
            r"\b(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})\b",
            r"\b(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b",
        ):
            if m := re.search(pat, text):
                return m.group(1)
        # Spoken dates with a month name nearby.
        if re.search(r"(?i)\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", text):
            return text
        return None

    @staticmethod
    def _extract_amount(text: str, *, has_card: bool) -> Decimal | None:
        # If a card number is in the message, the regex below would catch its
        # digits — skip amount extraction in that case.
        if has_card:
            return None
        if _FULL_PAY_RE.search(text):
            return None
        # Word-numbers: "a thousand", "five hundred".
        word_amounts = {
            "hundred": 100,
            "thousand": 1000,
            "lakh": 100_000,
            "lac": 100_000,
            "crore": 10_000_000,
        }
        lower = text.lower()
        for word, value in word_amounts.items():
            if word in lower:
                # Look for an optional leading multiplier.
                m = re.search(rf"(\d+|a|one|two|three|four|five|six|seven|eight|nine|ten)\s+{word}", lower)
                if m:
                    multiplier_text = m.group(1)
                    multiplier_map = {
                        "a": 1, "one": 1, "two": 2, "three": 3, "four": 4,
                        "five": 5, "six": 6, "seven": 7, "eight": 8,
                        "nine": 9, "ten": 10,
                    }
                    multiplier = multiplier_map.get(
                        multiplier_text, int(multiplier_text) if multiplier_text.isdigit() else 1
                    )
                    try:
                        return normalise_amount(multiplier * value)
                    except ValidationError:
                        return None
        match = _AMOUNT_RE.search(text)
        if match:
            try:
                return normalise_amount(match.group(1))
            except ValidationError:
                return None
        return None


# LLM extractor ---------------------------------------------------------------


class LLMExtractor:
    """Uses Claude with forced tool use to extract `ExtractedFields`."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = None
        if config.anthropic_api_key:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)
            except Exception as exc:  # pragma: no cover - import failure
                _LOG.warning("Anthropic client unavailable: %r", exc)

    @property
    def available(self) -> bool:
        return self._client is not None

    def extract(
        self,
        user_input: str,
        *,
        awaiting: str,
        already_known: Iterable[str],
    ) -> ExtractedFields | None:
        if self._client is None:
            return None
        try:
            response = self._client.messages.create(
                model=self._config.model,
                max_tokens=512,
                temperature=0,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=[EXTRACTION_TOOL],
                tool_choice={"type": "tool", "name": EXTRACTION_TOOL["name"]},
                messages=[
                    {
                        "role": "user",
                        "content": build_user_message(
                            user_input,
                            awaiting=awaiting,
                            already_known=list(already_known),
                        ),
                    }
                ],
            )
        except Exception as exc:
            _LOG.warning("LLM extraction failed: %r", exc)
            return None

        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == EXTRACTION_TOOL["name"]:
                raw = block.input if isinstance(block.input, dict) else {}
                return self._coerce(raw)
        _LOG.warning("LLM extraction returned no tool_use block")
        return None

    @staticmethod
    def _coerce(raw: dict[str, object]) -> ExtractedFields | None:
        cleaned: dict[str, object] = {}
        for key, value in raw.items():
            if value in (None, "", "null"):
                continue
            cleaned[key] = value
        if "payment_amount" in cleaned:
            try:
                cleaned["payment_amount"] = Decimal(str(cleaned["payment_amount"]))
            except Exception:
                cleaned.pop("payment_amount", None)
        try:
            return ExtractedFields(**cleaned)
        except Exception as exc:
            _LOG.warning("LLM produced unparseable fields: %r", exc)
            return None


# Composite -------------------------------------------------------------------


class CompositeExtractor:
    """Run the LLM extractor when available, deterministic as fallback/backup."""

    def __init__(self, config: Config) -> None:
        self._deterministic = DeterministicExtractor()
        self._llm = LLMExtractor(config)

    def extract(
        self,
        user_input: str,
        *,
        awaiting: str,
        already_known: Iterable[str],
    ) -> ExtractedFields:
        deterministic = self._deterministic.extract(user_input)
        llm = self._llm.extract(
            user_input, awaiting=awaiting, already_known=already_known
        )
        if llm is None:
            return deterministic
        return _merge(deterministic, llm)


def _merge(primary: ExtractedFields, secondary: ExtractedFields) -> ExtractedFields:
    """Prefer secondary (LLM) values, fall back to primary (deterministic).

    The LLM is generally better at messy NL; deterministic is only stronger for
    extremely regular formats. Where they disagree we re-validate the LLM
    output downstream, so this preference is safe.
    """

    merged: dict[str, object] = {}
    for field_name in ExtractedFields.model_fields:
        primary_value = getattr(primary, field_name)
        secondary_value = getattr(secondary, field_name)
        merged[field_name] = secondary_value if secondary_value is not None else primary_value
    return ExtractedFields(**merged)
