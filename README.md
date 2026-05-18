# QuickPay — Conversational Payment Collection Agent

A production-grade conversational AI agent that runs an end-to-end payment
collection flow over chat:

1. Greets the user and asks for an account ID
2. Looks up the account via the payment API
3. Verifies identity (strict name match + at least one secondary factor)
4. Shares the outstanding balance with the verified user
5. Collects card details
6. Processes the payment
7. Communicates the outcome clearly
8. Recaps and closes

The agent uses an **LLM for natural-language extraction** (handling the messy
phrasing real users actually produce) and a **deterministic state machine** for
flow control, verification, and validation. Sensitive fields (DOB, Aadhaar,
pincode, PAN, CVV) are never echoed back to the user and are redacted from logs.

> Built against the assignment spec in
> `Agent_Engineer_-_payment_agent_assignment.pdf` (not committed). See
> [docs/design.md](docs/design.md) for architecture and tradeoffs.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your ANTHROPIC_API_KEY
python cli.py
```

Programmatic use matches the required interface exactly:

```python
from agent import Agent

agent = Agent()
print(agent.next("hi")["message"])
print(agent.next("my account is ACC1001")["message"])
```

## Project layout

```
agent.py                 # Re-exports the Agent class for evaluators
cli.py                   # Interactive REPL
src/payment_agent/       # Implementation
    agent.py             # Agent.next() — state machine driver
    state.py             # Conversation state and transitions
    api_client.py        # Typed client for the payment API
    extractors.py        # LLM-based + deterministic extraction
    validators.py        # Luhn, expiry, DOB normalization
    verification.py      # Strict identity verification
    models.py            # Pydantic schemas
    prompts.py           # Extraction prompts
    logging_utils.py     # PII-redacting logger
    config.py
    errors.py
evals/                   # Evaluation harness, scenarios, and metrics
docs/                    # Design doc and sample conversations
tests/                   # Unit tests
```

## Running tests and evaluation

```bash
python -m pytest -q                  # unit tests
python -m evals.run_eval             # full evaluation suite + report
```

## Architecture

### High-level

```
                          ┌──────────────────────┐
   user_input ──────────► │     Agent.next       │ ────► {"message": str}
                          └──────────┬───────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                            ▼
  CompositeExtractor          ConversationState           PaymentAPIClient
  ┌─────────────────┐         ┌───────────────┐           ┌────────────────┐
  │ LLMExtractor    │         │ Phase enum    │           │ lookup-account │
  │ (Claude tool-   │         │ candidate ID  │           │ process-       │
  │  use, temp=0)   │         │ factors       │           │  payment       │
  │ + Determin-     │         │ card fields   │           │ typed errors   │
  │  isticExtractor │         │ retry counters│           └────────────────┘
  │ (regex)         │         └───────────────┘
  └─────────────────┘
        │
        ▼
   validators.py
   (Luhn, expiry, DOB, amount, Aadhaar, pincode)
        │
        ▼
   verification.py
   (strict: name == account.name AND ≥1 secondary match)
```

The agent is a **deterministic state machine** that delegates only the
unbounded part of the problem — turning messy natural language into
structured fields — to an LLM. Control flow, validation, verification, and
retry policy are all in code so the LLM cannot bypass the spec's hard rules
("no fuzzy matching", "do not skip steps", "do not proceed without
verification") even when it hallucinates.

### Request lifecycle (one call to `Agent.next`)

1. **Extract.** `CompositeExtractor` produces an `ExtractedFields` Pydantic
   object from the user's message. The LLM extractor uses Anthropic forced
   tool-use at temperature 0 with prompt caching; the deterministic
   extractor uses regex + word-number heuristics. Their outputs are merged,
   LLM-preferred.
2. **Capture.** `_capture_fields` validates every extracted value through
   `validators.py` (Luhn, expiry, DOB calendar check, amount parsing) and
   merges valid values into `ConversationState`. Invalid values become soft
   error messages the agent surfaces to the user.
3. **Dispatch.** Based on `state.phase`, the agent calls one of seven
   phase handlers (`_handle_greeting`, `_handle_awaiting_account`,
   `_handle_awaiting_verification`, `_handle_awaiting_amount`,
   `_handle_awaiting_card`, `_handle_confirm_payment`, or the terminal
   follow-up).
4. **Side effects.** The handler decides whether to call the payment API —
   `lookup_account` once after the greeting, `process_payment` once per
   `yes` at the confirm step.
5. **Respond.** Each handler returns a templated string; the agent appends
   it to history and returns `{"message": ...}`.

### State machine

```
GREETING ─► AWAITING_ACCOUNT ─► AWAITING_VERIFICATION ─► AWAITING_AMOUNT
                                       │                       │
                                       ▼ (3 retries exceeded)  ▼
                                DONE_TERMINATED         AWAITING_CARD
                                                               │
                                                               ▼
                                                       CONFIRM_PAYMENT
                                                          │     │
                                                          ▼     ▼
                                                 DONE_SUCCESS  retryable
                                                               failure ─► loops back
                                                               │
                                                               ▼ (3 retries exceeded)
                                                         DONE_FAILED
```

Sub-states (e.g. which card field is missing) are *derived* from which
fields on `ConversationState` are still null, rather than enumerated. This
keeps the phase set small and makes out-of-order info trivially supported:
if the user volunteers their card expiry while we're still collecting the
number, we just store it and the missing-fields prompt reflects what's
left.

### LLM vs deterministic split

| Concern | Owner | Why |
|---|---|---|
| Parsing messy NL → structured fields | LLM | Open-ended, brittle to regex |
| Deciding next question | Code | Spec's "do not skip" rule must be mechanically enforced |
| Verification check | Code | "No fuzzy matching" rule; strict equality |
| Field validation (Luhn, expiry, etc.) | Code | Deterministic; we want exact spec-aligned error codes |
| Response wording | Code (templates) | Reproducibility for evaluators |
| Retry policy | Code | Counters live on `ConversationState` |

Without an `ANTHROPIC_API_KEY`, the agent still runs — the deterministic
extractor handles clean inputs, and the test suite + eval scenarios run
hermetically in that mode. With the key, the LLM extractor adds robustness
to messy phrasing.

### Key invariants

- **Hard payment gate.** `process_payment` is only callable from
  `CONFIRM_PAYMENT`, which is only reachable from `AWAITING_CARD` once
  `state.verified is True`. There is no shortcut.
- **No PII echo.** Account-side identifiers (DOB, Aadhaar last 4, pincode)
  are stored on `ConversationState.account` but never interpolated into
  response strings. The eval runner asserts this by substring search.
- **Log redaction.** `logging_utils.py` installs a filter that masks any
  PAN-shaped run, CVV-adjacent digits, Aadhaar/DOB mentions in every log
  record. The API client and agent already avoid logging sensitive
  payloads; the filter is a backstop.
- **Card data lifecycle.** Card number, CVV, expiry, and cardholder name
  are wiped from `ConversationState` immediately after the API call
  completes (success or terminal failure), so they live in memory only
  long enough to make the one request.
- **Determinism.** Temperature 0 on the LLM, forced tool use to lock the
  output schema, and templated responses on the agent side keep behaviour
  reproducible across runs of the evaluator.

### Failure-mode matrix

| Source | What happens |
|---|---|
| `account_not_found` (404) | Stays in `AWAITING_ACCOUNT`; user re-enters the ID |
| Lookup transport error | Friendly "trouble reaching" message; user can retry the turn |
| Name mismatch | Counts against the 3-retry verification cap; user reprompted |
| Secondary-factor mismatch | Same retry counter; the specific failed factor is cleared so the user can supply a fresh value |
| Verification cap exhausted | `DONE_TERMINATED`; subsequent turns receive a closing message |
| Amount > balance (local) | Reprompt without an API call |
| `insufficient_balance` (server) | Loop back to `AWAITING_AMOUNT` |
| `invalid_card` / `invalid_cvv` / `invalid_expiry` | Card fields wiped, reprompt; counts against the 3-retry payment cap |
| Payment cap exhausted | `DONE_FAILED`; card data wiped; user told to contact support |
| Unknown `error_code` | Logged, session closed cleanly |

See [docs/design.md](docs/design.md) for the full design document
(tradeoffs, edge cases, what I'd do with more time) and
[docs/sample_conversations.md](docs/sample_conversations.md) for runnable
transcripts.

## Sample conversations

See [docs/sample_conversations.md](docs/sample_conversations.md) for end-to-end
transcripts covering success, verification exhaustion, payment failure, and an
edge case (leap-year DOB on ACC1004).
