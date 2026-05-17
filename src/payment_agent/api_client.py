"""Typed HTTP client for the payment verification API.

We deliberately keep this layer thin: it owns transport, retries on transient
network errors, and the mapping from HTTP response bodies to typed exceptions.
Business logic (verification, amount validation, state transitions) lives
elsewhere.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import httpx

from .errors import AccountNotFoundError, APIError, APITransportError
from .logging_utils import get_logger
from .models import (
    Account,
    CardDetails,
    PaymentFailure,
    PaymentMethod,
    PaymentSuccess,
    ProcessPaymentRequest,
)

_LOG = get_logger(__name__)

# Error codes the API may return. We surface them verbatim so the state machine
# can branch on them; new codes from the server will still raise APIError.
KNOWN_PAYMENT_ERROR_CODES = frozenset(
    {
        "account_not_found",
        "invalid_amount",
        "insufficient_balance",
        "invalid_card",
        "invalid_cvv",
        "invalid_expiry",
    }
)


class PaymentAPIClient:
    """Synchronous client for the two endpoints we need."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 15.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=timeout_seconds,
            transport=transport,
            headers={"Content-Type": "application/json"},
        )

    def __enter__(self) -> "PaymentAPIClient":  # pragma: no cover - convenience
        return self

    def __exit__(self, *exc: object) -> None:  # pragma: no cover - convenience
        self.close()

    def close(self) -> None:
        self._client.close()

    def lookup_account(self, account_id: str) -> Account:
        """Fetch an account by id.

        Raises:
            AccountNotFoundError: API returned 404 / account_not_found.
            APIError: API returned a different known error_code.
            APITransportError: network failure, timeout, or malformed JSON.
        """

        _LOG.info("lookup_account account_id=%s", account_id)
        body = self._post("/api/lookup-account", {"account_id": account_id})
        if "error_code" in body:
            code = body.get("error_code", "unknown")
            message = body.get("message", "Account lookup failed.")
            if code == "account_not_found":
                raise AccountNotFoundError(message)
            raise APIError(code, message)
        try:
            return Account(**body)
        except Exception as exc:  # pragma: no cover - schema drift
            raise APITransportError(f"Malformed account payload: {exc!r}") from exc

    def process_payment(
        self,
        account_id: str,
        amount: Decimal,
        card: CardDetails,
    ) -> PaymentSuccess | PaymentFailure:
        """Submit a card payment.

        Returns either `PaymentSuccess` (with `transaction_id`) or
        `PaymentFailure` (with `error_code`). Network-level issues raise
        `APITransportError` so the caller can distinguish "we never heard back"
        from "the server said no".
        """

        request = ProcessPaymentRequest(
            account_id=account_id,
            amount=amount,
            payment_method=PaymentMethod(card=card),
        )
        # Avoid logging the full payload — `request.model_dump()` would include
        # PAN and CVV. The redaction filter is a backstop, not an excuse.
        _LOG.info(
            "process_payment account_id=%s amount=%s last4=%s",
            account_id,
            amount,
            card.card_number[-4:],
        )
        payload = request.model_dump(mode="json")
        # Decimal serialises to a JSON string by default; the API requires a
        # JSON number. Coerce here rather than weakening the typed model.
        payload["amount"] = float(amount)
        body = self._post("/api/process-payment", payload)
        if body.get("success") is True:
            return PaymentSuccess(**body)
        if body.get("success") is False:
            return PaymentFailure(**body)
        raise APITransportError(f"Unexpected payment response shape: {set(body)}")

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.post(path, json=payload)
        except httpx.HTTPError as exc:
            raise APITransportError(f"HTTP error contacting payment API: {exc!r}") from exc

        try:
            body = response.json()
        except ValueError as exc:
            raise APITransportError(
                f"Non-JSON response from {path} (status={response.status_code})"
            ) from exc

        if not isinstance(body, dict):
            raise APITransportError(f"Unexpected JSON shape from {path}: {type(body).__name__}")

        if response.status_code >= 500:
            raise APITransportError(
                f"Server error {response.status_code} from {path}: {body!r}"
            )
        return body
