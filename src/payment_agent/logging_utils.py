"""Logger factory that redacts sensitive fields before emission.

The agent handles regulated data (full card numbers, CVV, DOB, Aadhaar,
pincode). We never want any of that landing in stdout, log files, or crash
reports. This module wraps Python's `logging` with a filter that masks anything
matching the sensitive patterns regardless of how the call site formatted it.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

_REDACTED = "[REDACTED]"

# Patterns are deliberately broad: better to over-redact than to leak.
_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # 13–19 digit runs (PAN). Matches with or without spaces/dashes.
    (re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), _REDACTED),
    # 3–4 digit CVV when adjacent to the word cvv/cvc/cvv2.
    (re.compile(r"(?i)\b(cvv2?|cvc)\b[^\d]{0,5}\d{3,4}\b"), r"\1 " + _REDACTED),
    # Aadhaar 12-digit and "last 4" mentions.
    (re.compile(r"(?i)\baadhaa?r[^\d]{0,12}\d{3,12}\b"), "aadhaar " + _REDACTED),
    # DOB-shaped strings.
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), _REDACTED),
    (re.compile(r"\b\d{2}[/-]\d{2}[/-]\d{2,4}\b"), _REDACTED),
)


def _redact(text: str) -> str:
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class _RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = _redact(str(record.msg))
            if record.args:
                record.args = _redact_args(record.args)
        except Exception:  # pragma: no cover - never break logging
            pass
        return True


def _redact_args(args: object) -> object:
    if isinstance(args, dict):
        return {k: _redact(str(v)) for k, v in args.items()}
    if isinstance(args, tuple):
        return tuple(_redact(str(a)) for a in args)
    if isinstance(args, Iterable) and not isinstance(args, (str, bytes)):
        return tuple(_redact(str(a)) for a in args)
    return _redact(str(args))


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger configured with PII redaction and a single stream handler."""

    logger = logging.getLogger(name)
    if getattr(logger, "_payment_agent_configured", False):
        logger.setLevel(level)
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.addFilter(_RedactingFilter())
    logger.addHandler(handler)
    logger.propagate = False
    logger._payment_agent_configured = True  # type: ignore[attr-defined]
    return logger


def redact(text: str) -> str:
    """Public helper for callers that need to redact ad-hoc strings."""

    return _redact(text)
