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

## Design highlights

- **State machine in code, parsing in the LLM.** The LLM extracts structured
  fields from free-form user turns; the state machine decides what to ask next.
  See [docs/design.md](docs/design.md).
- **Strict verification.** Exact full-name match plus at least one of DOB,
  last-4 of Aadhaar, or pincode. Three attempts, then the session terminates.
- **Hard payment gate.** Payment APIs are unreachable until verification flips
  to `verified`; the state machine enforces this independent of the LLM.
- **No PII leakage.** Account-side identifiers fetched from the lookup API are
  kept in private memory only; logs scrub card numbers, CVV, DOB, Aadhaar.
- **Graceful failures.** API errors map to user-facing messages distinguishing
  retryable (`invalid_card`, `invalid_cvv`, `invalid_expiry`,
  `insufficient_balance`) from terminal failures.

## Sample conversations

See [docs/sample_conversations.md](docs/sample_conversations.md) for end-to-end
transcripts covering success, verification exhaustion, payment failure, and an
edge case (leap-year DOB on ACC1004).
