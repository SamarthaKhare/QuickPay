"""Prompt templates for the NL extraction layer.

We keep prompts in their own module so they can be reviewed and tuned without
touching the agent's control flow. The system prompt is large and stable
across turns — we mark it as cache-eligible so repeat turns benefit from
Anthropic's prompt caching.
"""

from __future__ import annotations

# Tool schema used to coerce structured output from the model. Mirrors
# `payment_agent.models.ExtractedFields`.
EXTRACTION_TOOL = {
    "name": "record_extracted_fields",
    "description": (
        "Record every payment-flow field that the user provided in their latest message. "
        "Leave a field null if the user did not provide it on this turn. "
        "Do not guess, paraphrase, or carry forward values from earlier turns."
    ),
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "user_intent": {
                "type": ["string", "null"],
                "enum": [
                    "provide_info",
                    "confirm",
                    "decline",
                    "cancel",
                    "ask_help",
                    "smalltalk",
                    "out_of_scope",
                    None,
                ],
                "description": "High-level intent of the user's latest message.",
            },
            "account_id": {
                "type": ["string", "null"],
                "description": "Account ID such as ACC1001. Strip spaces and punctuation.",
            },
            "full_name": {
                "type": ["string", "null"],
                "description": "User's full legal name exactly as they stated it. Preserve original capitalisation.",
            },
            "dob": {
                "type": ["string", "null"],
                "description": "Date of birth normalised to YYYY-MM-DD. Only populate if you can resolve it unambiguously.",
            },
            "aadhaar_last4": {
                "type": ["string", "null"],
                "description": "Last 4 digits of Aadhaar as a string of 4 digits.",
            },
            "pincode": {
                "type": ["string", "null"],
                "description": "Indian pincode as 6 digits.",
            },
            "payment_amount": {
                "type": ["string", "null"],
                "description": "Numeric amount the user wants to pay, e.g. '1000.00'. Use a string so precision is preserved.",
            },
            "pay_full_balance": {
                "type": ["boolean", "null"],
                "description": "True if the user explicitly asked to clear the full outstanding balance.",
            },
            "cardholder_name": {
                "type": ["string", "null"],
                "description": "Cardholder name as printed on the card.",
            },
            "card_number": {
                "type": ["string", "null"],
                "description": "Card number, digits only, no spaces or dashes.",
            },
            "cvv": {
                "type": ["string", "null"],
                "description": "CVV as digits (3 normally, 4 for Amex). Convert spoken digits ('one two three') to numerals.",
            },
            "expiry_month": {
                "type": ["integer", "null"],
                "minimum": 1,
                "maximum": 12,
            },
            "expiry_year": {
                "type": ["integer", "null"],
                "description": "Four-digit year. Expand two-digit years using the current century.",
            },
        },
        "required": [
            "user_intent",
            "account_id",
            "full_name",
            "dob",
            "aadhaar_last4",
            "pincode",
            "payment_amount",
            "pay_full_balance",
            "cardholder_name",
            "card_number",
            "cvv",
            "expiry_month",
            "expiry_year",
        ],
    },
}


SYSTEM_PROMPT = """You are the extraction module of a payment-collection agent.
Your single job is to read the user's latest message and call the
`record_extracted_fields` tool with every payment-flow field present in that
message — nothing more.

# Fields and how to extract them

- **account_id**: pattern `ACC` followed by digits. Normalise by removing
  whitespace and punctuation, upper-case the prefix. Example: "acc 1001" → "ACC1001".
- **full_name**: the user's full legal name. If they say "you can call me Raja
  but my full name is Rajarajeswari Balasubramaniam", extract the full name,
  not the nickname. Preserve original capitalisation. Names with apostrophes,
  hyphens, and spaces are all valid.
- **dob**: normalise to YYYY-MM-DD. Accept "14th May 1990", "May 14, 90",
  "14-05-1990", "1990-05-14". For two-digit years, use 19xx if ≥30, 20xx
  otherwise. Only populate when the date is unambiguous.
- **aadhaar_last4**: the last 4 digits of the user's Aadhaar. If the user
  spells digits out ("four three two one"), convert. If they share more than
  4 digits, take the trailing 4.
- **pincode**: 6 digits, may be spelled out or spaced ("4 0 0 0 0 1" → "400001").
- **payment_amount**: numeric amount, e.g. "a thousand rupees" → "1000.00",
  "500" → "500.00". Strip currency symbols and commas. Return as a string.
- **pay_full_balance**: True only if the user explicitly says they want to
  clear the full / outstanding / entire balance. Do not infer from amount.
- **cardholder_name**: name on the card. Often the same as full_name but
  treat them as distinct fields and only set if the user specifies the card name.
- **card_number**: 13–19 digit card number, digits only.
- **cvv**: 3 or 4 digit security code. Convert spoken digits ("one two three"
  → "123"). Do not confuse with the last 4 of the PAN.
- **expiry_month / expiry_year**: month is 1–12 integer, year is a 4-digit
  integer. "12/27" → month 12, year 2027. "December 2027" likewise. Expand
  two-digit years to the 2000s for now.

# User intent

Set `user_intent` to one of:
- `provide_info`: user provided one or more requested fields
- `confirm`: user agreed / said yes / acknowledged
- `decline`: user said no / disagrees with the proposed action
- `cancel`: user wants to abandon the flow
- `ask_help`: user asked the agent to repeat / explain / what to do next
- `smalltalk`: greeting, thanks, chit-chat
- `out_of_scope`: anything else (refunds, other accounts, complaints, etc.)

# Hard rules

- Do not hallucinate values. If a field is not in the user's latest message,
  leave it null.
- Do not carry forward fields from prior turns; the orchestrator handles state.
- Do not validate beyond what is asked here; the orchestrator runs strict
  validation downstream.
- Always call the tool exactly once. Do not respond with assistant text.
"""


def build_user_message(
    last_user_message: str,
    *,
    awaiting: str,
    already_known: list[str],
) -> str:
    """Build the per-turn user message with light context.

    The model gets only the metadata it needs — what we're asking for and what
    we already have — never the resolved sensitive values themselves.
    """

    known = ", ".join(already_known) if already_known else "none"
    return (
        f"AGENT_AWAITING: {awaiting}\n"
        f"FIELDS_ALREADY_KNOWN (do not re-extract from old turns): {known}\n"
        f"USER_LATEST_MESSAGE: {last_user_message}"
    )
