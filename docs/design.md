# QuickPay — Design Document

## 1. Goal

Build a conversational AI agent that collects card payments end-to-end:
greet → look up account → verify identity → share balance → collect card →
process → close. The agent must handle free-form natural language at every
step, enforce strict verification, never leak PII, and behave deterministically
across runs.

## 2. Architecture

```
                         ┌────────────────────┐
   user_input ─────────► │      Agent.next    │ ──────► {"message": ...}
                         └─────────┬──────────┘
                                   │
              ┌────────────────────┼─────────────────────┐
              ▼                    ▼                     ▼
        Extractor           ConversationState        PaymentAPIClient
   (LLM tool-use +          (Phase, candidate         (lookup-account,
    deterministic           name/DOB/etc,             process-payment;
    regex fallback)         card fields,              typed errors)
                            attempt counters)
              │                    │
              └──────► validators ─┘
              (Luhn, expiry, DOB normalisation, amount parsing)
```

### Module map

| Module | Responsibility |
| --- | --- |
| `agent.py` | State-machine driver. Owns one `ConversationState`, dispatches per-phase handlers, calls API client at the right moments. |
| `state.py` | `Phase` enum + `ConversationState` dataclass. |
| `extractors.py` | Composite extractor: `LLMExtractor` (Anthropic forced tool-use) + `DeterministicExtractor` (regex). Returns an `ExtractedFields` Pydantic model. |
| `prompts.py` | System prompt + tool schema for extraction. Cache-eligible. |
| `validators.py` | Pure functions: Luhn check, expiry, DOB normalisation, amount parsing. Raise `ValidationError(error_code, …)` aligned with the spec. |
| `verification.py` | Strict identity check (exact name + ≥1 secondary). Returns enum outcome. |
| `api_client.py` | `httpx`-based client. Wraps the two endpoints and surfaces typed errors. |
| `models.py` | Pydantic schemas for API payloads and `ExtractedFields`. |
| `logging_utils.py` | Logger with redaction of PAN, CVV, Aadhaar, DOB. |
| `config.py` | Environment-driven runtime config. |

## 3. Key design decisions

### 3.1 State machine in code, parsing in the LLM

The assignment explicitly invites this question: *"what role should the LLM
play in that process versus deterministic code?"* My answer:

- **LLM owns extraction.** Free-form user input is highly variable
  ("DOB is May 14, 90", "card number is 4532 0151 1283 0366"). Writing
  regex for every variant would be brittle. Claude with forced tool use
  reliably maps messy text → structured fields.
- **Deterministic code owns control flow, validation, and verification.**
  These are the rules the spec is strictest about: "do not skip steps",
  "do not proceed to payment without verification", "no fuzzy matching".
  Encoding them as a `Phase` enum + per-phase handlers means a hallucinating
  LLM cannot bypass them — it can only fail to extract a field, which the
  validators catch.

### 3.2 Composite extractor with deterministic fallback

`CompositeExtractor` runs both layers and merges results, preferring the LLM
output. Two reasons:

1. The agent stays usable without an API key (for evaluation in restricted
   environments and for unit tests).
2. The deterministic layer is a sanity check: if LLM and regex disagree on
   a clean input like `ACC1001`, we have signal that something went wrong.

Every extracted field is re-validated downstream by `validators.py` before
being stored in conversation state, so a bad LLM extraction cannot land
invalid data in the system.

### 3.3 Strict verification, encoded as a single function

`verify(account, full_name, dob, aadhaar_last4, pincode)` returns one of:
`VERIFIED`, `MISSING_NAME`, `NAME_MISMATCH`, `NEED_SECONDARY`,
`SECONDARY_MISMATCH`. The agent's state machine branches on the enum. This
makes the rule mechanically auditable: it is a single function with 9 unit
tests covering each outcome.

Whitespace inside a name is collapsed (`"Nithin   Jain"` → `"Nithin Jain"`) —
this is the only normalisation. Case is preserved; the spec explicitly
forbids case-insensitive workarounds.

### 3.4 Retry policy

Two independent counters live on `ConversationState`:

- `verification_attempts` (cap 3). Counts both name mismatches and
  secondary-factor mismatches. Exceeding the cap moves to
  `DONE_TERMINATED`. Even after termination, the agent responds politely
  to further turns rather than crashing.
- `payment_attempts` (cap 3). Counts every `process_payment` call. After
  three retryable failures (`invalid_card`, `invalid_cvv`, `invalid_expiry`)
  the agent moves to `DONE_FAILED` and asks the user to contact support.

`insufficient_balance` is special-cased: rather than counting it as a
payment attempt-eating failure, we treat it as a recoverable amount error
and route back to amount entry. We also block "amount > balance" *locally*
before the API call so the user gets immediate feedback.

### 3.5 PII handling

- The `Account` record returned by the lookup API is kept in memory only.
  Its sensitive fields (DOB, Aadhaar last 4, pincode) are **never** echoed
  back to the user. Even after verification, the agent talks about the
  balance only.
- `logging_utils.py` installs a redaction filter on every logger that masks
  any 13–19 digit run (PAN), CVV-shaped neighbouring digits, Aadhaar
  mentions, and YYYY-MM-DD / DD-MM-YYYY strings. The redactor is a backstop
  — the API client and agent already avoid logging sensitive payloads.
- `cardholder_name`, `card_number`, `cvv`, and `expiry_*` live in
  `ConversationState` only as long as needed for the payment call; on a
  retryable error the card fields are wiped before reprompting.

### 3.6 Determinism

The interface contract requires consistent behaviour across runs. We get it
by:

- Calling Claude at `temperature=0` with `tool_choice` forced to the
  extraction tool.
- Routing every state transition through deterministic enum/handler logic.
- Using templated response messages rather than LLM-generated prose, so
  the same state produces the same wording.

The LLM may still surface micro-variations in extraction on truly ambiguous
input, but the state machine smooths these into the same downstream
behaviour.

## 4. Tradeoffs

| Choice | Tradeoff |
| --- | --- |
| Templated responses, not LLM-generated | Cheaper, deterministic, easier to evaluate. Slightly less natural-sounding than open-ended generation. |
| Single forced tool call per turn | Latency stays at ~1 LLM call/turn. We give up the ability for the LLM to ask clarifying questions on its own. |
| Strict regex-only deterministic fallback | Works without an API key, but can't handle every messy phrasing from the spec. The LLM extractor closes that gap when present. |
| Memory-only session state | Suitable for the evaluator interface. A real production deployment would need a session store (Redis) keyed by user id; current shape makes that easy to add. |
| Card data lives in process memory between turns | Lowest moving parts. For a real system I would tokenise on intake and forward only the token. |

## 5. Edge cases handled

- **ACC1003 zero balance**: agent confirms there is nothing to pay and
  closes the session without prompting for a card.
- **ACC1004 leap-year DOB (1988-02-29)**: `normalise_dob` validates the
  date through `datetime.date`, so `1988-02-29` parses but `1989-02-29`
  raises `invalid_dob`. Strict equality against the account DOB then
  matches.
- **Account ID retry**: a `404 / account_not_found` rewinds to
  `AWAITING_ACCOUNT` so the user can correct the ID without losing the
  session.
- **Amount above balance**: caught locally before the API call so the user
  gets a friendly reprompt instead of a server error.
- **PAN, CVV, expiry collected across multiple turns**: the field-by-field
  capture logic merges partial information turn-over-turn so users can
  share details in any order.
- **Decline at confirmation**: wipes only the card fields (keeps amount
  and cardholder name) and re-enters the card collection sub-phase.
- **Out-of-order info early in the flow**: the extractor captures any
  field it sees but the state machine never skips a step.
- **Conversation continues after a terminal state**: handled by
  `_terminal_followup` so the agent stays polite rather than crashing.

## 6. Evaluation

Two complementary evaluators:

- **Scripted scenarios** (`evals.run_eval`): 11 deterministic scenarios that
  replay a hardcoded sequence of user turns against an in-memory API
  client. Each scenario asserts on:
  - final phase,
  - number of lookup / payment API calls,
  - payment amount on the final call,
  - presence of required phrases,
  - absence of any PII string anywhere in the transcript.
  These run hermetically in CI and currently all pass.
- **LLM persona evaluator** (`evals.persona_eval`): an Anthropic model
  plays each of four personas (clean happy path, distracted partial payer,
  identity-theft attempts, mind-changer at confirm) and a separate judge
  model scores the transcript on goal completion, PII safety, and
  state-machine fidelity. Run with an API key for richer exploration.

### Metrics

- **Success rate** = scenarios passed / total. Currently 11/11.
- **Tool-call correctness** = `(min_payment_calls ≤ actual ≤ max_payment_calls)`
  and the final-turn amount matches `expect.payment_amount`. All 7
  scenarios that should call payment hit the exact expected amount.
- **PII safety** = no message contains DOB, Aadhaar last 4, or pincode
  values. Verified via substring search in the eval runner.

### Where the agent struggles

- The deterministic-only path (no API key) cannot parse "you can call me
  Raja but my full name is Rajarajeswari Balasubramaniam"-style
  introductions reliably. With the LLM enabled, it parses fine.
- Names that resemble common phrases (a user actually named "Sure
  Sharma", "Pay Patel") could in principle confuse the affirmation
  heuristic. In practice the LLM resolves the ambiguity from context.
- Long-tail conversational maneuvers — switching accounts mid-flow, asking
  for refunds, asking to be transferred — are routed to a generic "let's
  stay on track" response. A production deployment would need explicit
  intent handling for these.

## 7. What I would add with more time

- **Session persistence** — externalise `ConversationState` to Redis so a
  process restart doesn't drop in-flight payments.
- **Tokenised card capture** — push card details to a PCI-compliant vault
  inside the API client so they never sit in process memory.
- **Streaming responses** for interactive use; the current code returns
  the full message per turn.
- **Audit logging** to a structured sink (DataDog/Splunk) with the
  redaction filter applied, plus a per-turn trace ID for debugging.
- **More personas** in `persona_eval.py`, including adversarial ones
  (prompt injection in the user channel, social-engineering attempts).
- **A small finetune** of an open-weight model for the extraction
  prompt would reduce latency and cost vs. forced tool use on a frontier
  model.
