"""The conversational payment-collection agent.

Architecture in one paragraph: each call to `Agent.next()` runs an extraction
pass (LLM tool-use + deterministic regex) to turn the user's free-form message
into a structured `ExtractedFields` object. That object is merged into
`ConversationState`, the state machine advances based on the current phase
plus the fields now populated, and the agent emits a templated response.
APIs are called only at the moments the spec demands: once for account
lookup, once per payment attempt. Verification is performed locally against
the lookup result.

Why a state machine over a fully LLM-driven agent: the assignment requires
strict gating ("do not proceed to payment without verification", "do not skip
steps", "no fuzzy matching"). Encoding these as state transitions makes them
mechanically enforceable and testable. The LLM is intentionally confined to
parsing, where it adds the most value.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

from .api_client import PaymentAPIClient
from .config import Config
from .errors import (
    AccountNotFoundError,
    APIError,
    APITransportError,
    ValidationError,
)
from .extractors import CompositeExtractor
from .logging_utils import get_logger
from .models import Account, CardDetails, ExtractedFields, PaymentFailure, PaymentSuccess
from .state import ConversationState, Phase
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
from .verification import VerificationOutcome, verify

_LOG = get_logger(__name__)

# Awaiting-field metadata sent to the extractor so it can prioritise. The
# values are stable string identifiers, not user-facing prose.
_AWAITING_BY_PHASE = {
    Phase.GREETING: "account_id",
    Phase.AWAITING_ACCOUNT: "account_id",
    Phase.AWAITING_VERIFICATION: "verification_factors",
    Phase.AWAITING_AMOUNT: "payment_amount",
    Phase.AWAITING_CARD: "card_details",
    Phase.CONFIRM_PAYMENT: "confirmation",
    Phase.DONE_SUCCESS: "none",
    Phase.DONE_FAILED: "none",
    Phase.DONE_TERMINATED: "none",
}


class Agent:
    """Implements the required `Agent.next(user_input: str) -> dict` contract.

    Each instance maintains its own `ConversationState` so multiple sessions
    can run concurrently. State persists across calls within an instance and
    does not require any external setup between turns.
    """

    def __init__(
        self,
        *,
        config: Optional[Config] = None,
        api_client: Optional[PaymentAPIClient] = None,
        extractor: Optional[CompositeExtractor] = None,
    ) -> None:
        self._config = config or Config.from_env()
        self._api_client = api_client or PaymentAPIClient(
            self._config.api_base_url,
            timeout_seconds=self._config.http_timeout_seconds,
        )
        self._extractor = extractor or CompositeExtractor(self._config)
        self._state = ConversationState()
        # Track whether we've ever emitted the opening line so a bare "hi" gets
        # a greeting before the account prompt without short-circuiting.
        self._greeted = False

    # ------------------------------------------------------------------ API --

    def next(self, user_input: str) -> dict:
        """Process one turn of the conversation and return a response dict."""

        user_input = (user_input or "").strip()
        self._state.push_history("user", user_input)
        try:
            message = self._dispatch(user_input)
        except APITransportError as exc:
            _LOG.exception("transport failure")
            message = (
                "I'm having trouble reaching our payment system right now. "
                "Could we try again in a moment?"
            )
        except Exception:  # last-resort safety net
            _LOG.exception("unhandled error in agent.next")
            message = (
                "Something unexpected went wrong on my end. "
                "Let's pause here — please contact support if this keeps happening."
            )
            self._state.phase = Phase.DONE_TERMINATED
        self._state.push_history("agent", message)
        return {"message": message}

    # -------------------------------------------------------------- dispatch --

    def _dispatch(self, user_input: str) -> str:
        if self._state.is_terminal():
            return self._terminal_followup()

        if not user_input:
            return self._reprompt_for_current_phase()

        extracted = self._extractor.extract(
            user_input,
            awaiting=_AWAITING_BY_PHASE[self._state.phase],
            already_known=self._known_field_names(),
        )

        if extracted.user_intent == "cancel":
            self._state.phase = Phase.DONE_TERMINATED
            return (
                "No problem — I'll close out this session. "
                "Feel free to come back any time to make a payment."
            )

        # Capture any reusable fields regardless of phase. We never *skip* a
        # step on the strength of an early-volunteered value, but we do
        # remember it so we don't ask again.
        early_capture_messages = self._capture_fields(extracted)

        # Phase-specific handlers do the actual progress work.
        handler = {
            Phase.GREETING: self._handle_greeting,
            Phase.AWAITING_ACCOUNT: self._handle_awaiting_account,
            Phase.AWAITING_VERIFICATION: self._handle_awaiting_verification,
            Phase.AWAITING_AMOUNT: self._handle_awaiting_amount,
            Phase.AWAITING_CARD: self._handle_awaiting_card,
            Phase.CONFIRM_PAYMENT: self._handle_confirm_payment,
        }[self._state.phase]
        response = handler(user_input, extracted)

        # Soft validation feedback (e.g. an invalid DOB the user just supplied)
        # is appended only when we don't already have a more specific prompt.
        if early_capture_messages and not response.startswith(early_capture_messages[0][:20]):
            return f"{early_capture_messages[0]} {response}".strip()
        return response

    # ----------------------------------------------------------- field capture

    def _capture_fields(self, extracted: ExtractedFields) -> list[str]:
        """Validate and merge extracted fields into state.

        Returns a list of *soft* error messages the agent might want to relay
        before its main response (e.g. "That date didn't look valid — ...").
        Field-by-field validation errors during proactive capture are stored as
        soft errors so the user gets immediate, specific feedback. Validation
        of the current-step input is handled by the phase handler itself.
        """

        messages: list[str] = []
        state = self._state

        # Account ID — only meaningful before lookup.
        if extracted.account_id and state.account is None:
            try:
                state.account_id = normalise_account_id(extracted.account_id)
            except ValidationError as exc:
                messages.append(str(exc))

        # Identity factors are stored as candidates regardless of phase.
        if extracted.full_name:
            state.candidate_name = " ".join(extracted.full_name.split())
        if extracted.dob:
            try:
                state.candidate_dob = normalise_dob(extracted.dob)
            except ValidationError as exc:
                messages.append(str(exc))
        if extracted.aadhaar_last4:
            try:
                state.candidate_aadhaar_last4 = normalise_aadhaar_last4(extracted.aadhaar_last4)
            except ValidationError as exc:
                messages.append(str(exc))
        if extracted.pincode:
            try:
                state.candidate_pincode = normalise_pincode(extracted.pincode)
            except ValidationError as exc:
                messages.append(str(exc))

        # Payment fields. Captured only once we have a verified account so we
        # don't preemptively reject a thrown-out card number during greetings.
        if state.verified:
            # Amount is only collected during AWAITING_AMOUNT. Once set, leave
            # it alone so spurious numbers from later turns (e.g. CVV digits)
            # don't overwrite it.
            if state.phase == Phase.AWAITING_AMOUNT:
                if extracted.pay_full_balance:
                    state.pay_full_balance_intent = True
                    assert state.account is not None
                    state.payment_amount = state.account.balance
                if extracted.payment_amount is not None and not state.pay_full_balance_intent:
                    try:
                        state.payment_amount = normalise_amount(extracted.payment_amount)
                    except ValidationError as exc:
                        messages.append(str(exc))
                        state.payment_amount = None
            if extracted.cardholder_name:
                state.cardholder_name = extracted.cardholder_name.strip()
            elif (
                state.phase in (Phase.AWAITING_CARD, Phase.CONFIRM_PAYMENT)
                and extracted.full_name
                and not state.cardholder_name
            ):
                # Heuristic: while collecting card details, a "name" the user
                # supplies is the cardholder name unless we've already captured
                # one. Without this the deterministic-only path stalls.
                state.cardholder_name = " ".join(extracted.full_name.split())
            if extracted.card_number:
                try:
                    state.card_number = normalise_card_number(extracted.card_number)
                except ValidationError as exc:
                    messages.append(str(exc))
                    state.card_number = None
            if extracted.cvv:
                is_amex = bool(state.card_number and is_amex_pan(state.card_number))
                try:
                    state.cvv = normalise_cvv(extracted.cvv, is_amex=is_amex)
                except ValidationError as exc:
                    messages.append(str(exc))
                    state.cvv = None
            if extracted.expiry_month is not None and extracted.expiry_year is not None:
                try:
                    month, year = normalise_expiry(extracted.expiry_month, extracted.expiry_year)
                    state.expiry_month = month
                    state.expiry_year = year
                except ValidationError as exc:
                    messages.append(str(exc))
                    state.expiry_month = state.expiry_year = None
        return messages

    def _known_field_names(self) -> list[str]:
        names: list[str] = []
        state = self._state
        if state.account_id:
            names.append("account_id")
        if state.candidate_name:
            names.append("full_name")
        if state.candidate_dob:
            names.append("dob")
        if state.candidate_aadhaar_last4:
            names.append("aadhaar_last4")
        if state.candidate_pincode:
            names.append("pincode")
        if state.payment_amount is not None:
            names.append("payment_amount")
        if state.cardholder_name:
            names.append("cardholder_name")
        if state.card_number:
            names.append("card_number")
        if state.cvv:
            names.append("cvv")
        if state.expiry_month and state.expiry_year:
            names.append("expiry")
        return names

    # --------------------------------------------------- phase: greeting/acct

    def _handle_greeting(self, user_input: str, extracted: ExtractedFields) -> str:
        self._greeted = True
        if self._state.account_id:
            return self._begin_account_lookup()
        return (
            "Hi there! I'm QuickPay, here to help you make a payment on your account. "
            "Could you share your account ID to get started? It usually looks like ACC followed by digits."
        )

    def _handle_awaiting_account(self, user_input: str, extracted: ExtractedFields) -> str:
        if not self._state.account_id:
            return (
                "I didn't catch your account ID. It should start with ACC followed by digits, "
                "for example ACC1001."
            )
        return self._begin_account_lookup()

    def _begin_account_lookup(self) -> str:
        assert self._state.account_id is not None
        try:
            account = self._api_client.lookup_account(self._state.account_id)
        except AccountNotFoundError:
            failed_id = self._state.account_id
            self._state.account_id = None
            self._state.phase = Phase.AWAITING_ACCOUNT
            return (
                f"I couldn't find an account with the ID {failed_id}. "
                "Could you double-check and share it again?"
            )
        except APIError as exc:
            self._state.account_id = None
            self._state.phase = Phase.AWAITING_ACCOUNT
            return (
                "Our account lookup ran into an issue "
                f"({exc.error_code}). Could you share your account ID again?"
            )
        self._state.account = account

        if account.balance <= 0:
            self._state.phase = Phase.DONE_SUCCESS
            return (
                "Good news — there's nothing outstanding on this account. "
                "No payment is needed today."
            )

        self._state.phase = Phase.AWAITING_VERIFICATION
        return self._verification_prompt(first_time=True)

    # -------------------------------------------------- phase: verification --

    def _handle_awaiting_verification(self, user_input: str, extracted: ExtractedFields) -> str:
        state = self._state
        assert state.account is not None

        # Need a name first. If we don't have one, ask for it.
        if not state.candidate_name:
            return (
                "To get started with verification, could you share your full name as it appears "
                "on your account?"
            )

        result = verify(
            state.account,
            full_name=state.candidate_name,
            dob=state.candidate_dob,
            aadhaar_last4=state.candidate_aadhaar_last4,
            pincode=state.candidate_pincode,
        )

        if result.outcome == VerificationOutcome.VERIFIED:
            state.verified = True
            state.phase = Phase.AWAITING_AMOUNT
            balance_str = _format_inr(state.account.balance)
            return (
                f"Thanks — your identity is verified. "
                f"Your outstanding balance is {balance_str}. "
                "How much would you like to pay today? You can pay the full amount or a partial amount."
            )

        if result.outcome == VerificationOutcome.NEED_SECONDARY:
            return (
                "I have your name. To finish verifying, could you also share your "
                "date of birth, the last 4 digits of your Aadhaar, or your pincode? "
                "Any one of those is enough."
            )

        # Anything else is a mismatch — counts as a verification attempt.
        state.verification_attempts += 1
        remaining = self._config.max_verification_attempts - state.verification_attempts

        if remaining <= 0:
            state.phase = Phase.DONE_TERMINATED
            return (
                "I'm sorry, but I wasn't able to verify your identity after several attempts. "
                "For security, I'll need to end this session here. "
                "Please contact our support team if you'd like help recovering access."
            )

        # Reset only the factors that the user supplied for *this* attempt; if
        # they want to try a different secondary, they should be able to.
        if result.outcome == VerificationOutcome.NAME_MISMATCH:
            state.candidate_name = None
            attempts_word = "attempt" if remaining == 1 else "attempts"
            return (
                f"That name didn't match what we have on file. Could you share the full name "
                f"exactly as on your account? ({remaining} {attempts_word} remaining)"
            )

        # Secondary mismatch — clear the failed ones so a fresh value replaces them.
        for label in result.failed_secondary:
            if label == "dob":
                state.candidate_dob = None
            elif label == "aadhaar_last4":
                state.candidate_aadhaar_last4 = None
            elif label == "pincode":
                state.candidate_pincode = None
        attempts_word = "attempt" if remaining == 1 else "attempts"
        return (
            "Those details didn't match what we have on file. "
            "Could you try again with your date of birth, last 4 of Aadhaar, or pincode? "
            f"({remaining} {attempts_word} remaining)"
        )

    def _verification_prompt(self, *, first_time: bool) -> str:
        if first_time:
            return (
                "Thanks — I've pulled up your account. "
                "Before I can share any details, I need to verify your identity. "
                "Could you confirm your full name to start?"
            )
        return self._handle_awaiting_verification("", ExtractedFields())

    # ------------------------------------------------------ phase: amount --

    def _handle_awaiting_amount(self, user_input: str, extracted: ExtractedFields) -> str:
        state = self._state
        assert state.account is not None

        if state.payment_amount is None:
            balance_str = _format_inr(state.account.balance)
            return (
                f"Your outstanding balance is {balance_str}. "
                "How much would you like to pay today?"
            )

        if state.payment_amount > state.account.balance:
            amount = state.payment_amount
            state.payment_amount = None
            state.pay_full_balance_intent = False
            return (
                f"That amount ({_format_inr(amount)}) is more than your outstanding balance of "
                f"{_format_inr(state.account.balance)}. Could you share an amount up to that?"
            )

        # Got a valid amount; move on to card collection.
        state.phase = Phase.AWAITING_CARD
        return self._next_card_prompt(initial=True)

    # --------------------------------------------------------- phase: card --

    def _handle_awaiting_card(self, user_input: str, extracted: ExtractedFields) -> str:
        # _capture_fields already validated and stored what it could.
        if not self._state.card_fields_complete():
            return self._next_card_prompt(initial=False)
        return self._begin_confirmation()

    def _next_card_prompt(self, *, initial: bool) -> str:
        state = self._state
        missing: list[str] = []
        if not state.cardholder_name:
            missing.append("the name printed on the card")
        if not state.card_number:
            missing.append("the card number")
        if state.expiry_month is None or state.expiry_year is None:
            missing.append("the expiry month and year")
        if not state.cvv:
            missing.append("the CVV (the 3- or 4-digit number on the back)")

        if initial:
            return (
                f"Great. To process the payment of {_format_inr(state.payment_amount)}, "
                f"I'll need a few card details: " + ", ".join(missing) + "."
            )
        return "Thanks. Could you also share " + _join_natural(missing) + "?"

    def _begin_confirmation(self) -> str:
        state = self._state
        assert state.payment_amount is not None and state.card_number is not None
        last4 = state.card_number[-4:]
        state.phase = Phase.CONFIRM_PAYMENT
        return (
            f"Just to confirm: I'll charge {_format_inr(state.payment_amount)} to the card "
            f"ending {last4}, expiring {state.expiry_month:02d}/{state.expiry_year}. "
            "Shall I go ahead? (yes / no)"
        )

    # ----------------------------------------------------- phase: confirm --

    def _handle_confirm_payment(self, user_input: str, extracted: ExtractedFields) -> str:
        intent = extracted.user_intent
        if intent == "decline" or _looks_like_no(user_input):
            self._state.phase = Phase.AWAITING_CARD
            self._state.reset_card_fields()
            return (
                "No problem — let's redo the card details. "
                "Could you share the card number, expiry, and CVV again?"
            )
        if intent != "confirm" and not _looks_like_yes(user_input):
            return "Could you confirm with a yes or no so I can proceed?"

        return self._submit_payment()

    def _submit_payment(self) -> str:
        state = self._state
        assert state.account is not None
        assert state.account_id is not None
        assert state.payment_amount is not None
        assert state.cardholder_name and state.card_number and state.cvv
        assert state.expiry_month is not None and state.expiry_year is not None

        state.payment_attempts += 1
        card = CardDetails(
            cardholder_name=state.cardholder_name,
            card_number=state.card_number,
            cvv=state.cvv,
            expiry_month=state.expiry_month,
            expiry_year=state.expiry_year,
        )
        try:
            result = self._api_client.process_payment(
                state.account_id, state.payment_amount, card
            )
        except APITransportError:
            # Bubble out — top-level handler in next() will render a friendly
            # message and leave us in CONFIRM_PAYMENT so user can retry.
            raise

        if isinstance(result, PaymentSuccess):
            state.last_transaction_id = result.transaction_id
            state.phase = Phase.DONE_SUCCESS
            amount = _format_inr(state.payment_amount)
            return (
                f"Payment of {amount} processed successfully. "
                f"Your transaction ID is {result.transaction_id}. "
                "Thanks for using QuickPay — have a great day!"
            )

        assert isinstance(result, PaymentFailure)
        return self._handle_payment_failure(result)

    def _handle_payment_failure(self, failure: PaymentFailure) -> str:
        state = self._state
        state.last_payment_error = failure.error_code
        code = failure.error_code

        remaining = self._config.max_payment_attempts - state.payment_attempts
        terminal = remaining <= 0

        if code == "insufficient_balance":
            # Spec-wise this means the amount exceeds balance — terminal-ish but
            # user can choose a smaller amount, so we redirect to amount entry.
            assert state.account is not None
            state.payment_amount = None
            state.pay_full_balance_intent = False
            state.phase = Phase.AWAITING_AMOUNT
            return (
                f"That amount is more than your outstanding balance "
                f"({_format_inr(state.account.balance)}). "
                "Could you share a smaller amount?"
            )

        if code == "invalid_amount":
            state.payment_amount = None
            state.pay_full_balance_intent = False
            state.phase = Phase.AWAITING_AMOUNT
            if terminal:
                state.phase = Phase.DONE_FAILED
                return (
                    "We weren't able to process the payment after several attempts. "
                    "Please try again later or contact support."
                )
            return (
                "The amount didn't pass validation. "
                "Could you share an amount above zero with up to two decimal places?"
            )

        retryable = {
            "invalid_card": (
                "That card number didn't go through. Could you double-check the digits and try again?"
            ),
            "invalid_cvv": (
                "The CVV didn't match — it's the 3-digit number on the back of the card "
                "(4 digits on Amex). Could you re-enter the CVV?"
            ),
            "invalid_expiry": (
                "The expiry didn't work. Could you confirm the month and year on the card?"
            ),
        }
        if code in retryable:
            state.reset_card_fields()
            state.phase = Phase.AWAITING_CARD
            if terminal:
                state.phase = Phase.DONE_FAILED
                return (
                    "We've tried a few times and the payment still isn't going through. "
                    "Please double-check your card with your bank and try again later."
                )
            return retryable[code] + f" ({remaining} attempts remaining)"

        # Unknown error code from the server — log and close gracefully.
        _LOG.warning("Unknown payment error_code: %s", code)
        state.phase = Phase.DONE_FAILED
        return (
            "The payment didn't go through, and the issue isn't one I can fix automatically. "
            "Please contact support to complete this payment."
        )

    # ---------------------------------------------------- terminal handling --

    def _terminal_followup(self) -> str:
        state = self._state
        if state.phase == Phase.DONE_SUCCESS:
            if state.last_transaction_id:
                return (
                    "Your payment is already complete. Transaction ID: "
                    f"{state.last_transaction_id}. Is there anything else I can help with?"
                )
            return "We're all set. Is there anything else I can help with?"
        if state.phase == Phase.DONE_FAILED:
            return (
                "This session has been closed because the payment couldn't be completed. "
                "Please contact support if you need help."
            )
        return (
            "This session has been closed. Please start a new conversation when you're ready to try again."
        )

    def _reprompt_for_current_phase(self) -> str:
        prompts = {
            Phase.GREETING: "Hi! Could you share your account ID to get started?",
            Phase.AWAITING_ACCOUNT: "Could you share your account ID? It usually looks like ACC1001.",
            Phase.AWAITING_VERIFICATION: "Could you share the details I asked for so we can verify your identity?",
            Phase.AWAITING_AMOUNT: "How much would you like to pay today?",
            Phase.AWAITING_CARD: "Could you share the card details so we can process the payment?",
            Phase.CONFIRM_PAYMENT: "Shall I go ahead and process this payment? (yes / no)",
        }
        return prompts.get(self._state.phase, "Could you say that again?")


# -------------------------------------------------------------- helpers ------


def _format_inr(amount: Decimal | None) -> str:
    if amount is None:
        return "₹0.00"
    return f"₹{amount:,.2f}"


def _join_natural(parts: list[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


_YES_TOKENS = {"y", "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "go", "go ahead", "confirm", "proceed", "do it"}
_NO_TOKENS = {"n", "no", "nope", "nah", "stop", "cancel", "wait", "hold on", "don't"}


def _looks_like_yes(text: str) -> bool:
    normalised = text.strip().lower()
    if normalised in _YES_TOKENS:
        return True
    first = normalised.split()[0] if normalised else ""
    return first in _YES_TOKENS


def _looks_like_no(text: str) -> bool:
    normalised = text.strip().lower()
    if normalised in _NO_TOKENS:
        return True
    first = normalised.split()[0] if normalised else ""
    return first in _NO_TOKENS
