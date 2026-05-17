"""Typed errors raised by the API client and validators."""

from __future__ import annotations


class PaymentAgentError(Exception):
    """Base class for agent-internal errors."""


class AccountNotFoundError(PaymentAgentError):
    """Raised when the lookup API reports the account does not exist."""

    error_code = "account_not_found"


class APIError(PaymentAgentError):
    """Raised for any payment-API failure that returns a known error_code."""

    def __init__(self, error_code: str, message: str, status_code: int | None = None):
        super().__init__(message)
        self.error_code = error_code
        self.status_code = status_code


class APITransportError(PaymentAgentError):
    """Raised for network-level failures, timeouts, or malformed responses."""


class ValidationError(PaymentAgentError):
    """Raised when a payload fails local validation before an API call."""

    def __init__(self, error_code: str, message: str):
        super().__init__(message)
        self.error_code = error_code
