"""LLM-driven persona evaluator.

This complements the scripted scenarios with exploratory runs: an LLM plays
the role of a user with a defined persona and goal, the agent runs in
production mode (live API or mock), and a second LLM scores the transcript.

Run with `python -m evals.persona_eval` — requires `ANTHROPIC_API_KEY`. The
scripted evaluator (`evals.run_eval`) is the deterministic CI gate; this
script is for richer, harder-to-script behaviour like distracted users,
out-of-order information, or mid-flow cancellation.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
for p in (_ROOT, _SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from payment_agent.agent import Agent
from payment_agent.config import Config
from payment_agent.extractors import CompositeExtractor
from payment_agent.logging_utils import get_logger

from evals.scenarios import DEFAULT_ACCOUNTS
from tests.test_agent_flows import FakeAPIClient

_LOG = get_logger(__name__)

PERSONA_SYSTEM = """You are role-playing a customer in a payment-collection chat.
You will receive a persona profile that describes who you are, what your goal is,
and the information you actually know (or don't know) about your account.

Rules:
- Reply ONLY with the user's next message in the conversation. Keep it natural
  and short (1-2 sentences).
- Stay in character. Do not break role. Do not reveal that you are an AI.
- If your persona's goal is to complete payment, provide the requested
  information when asked. Use messy, conversational phrasing.
- If your persona is supposed to make a mistake (wrong DOB, etc), commit to it.
- If the agent ends the conversation, respond with "[end]"."""

JUDGE_SYSTEM = """You evaluate a chat between a payment-collection agent and a
user. Given the persona profile, expected outcome, and full transcript, return
a JSON object with fields:
- passed: bool (did the agent meet the goal correctly?)
- reason: short explanation
- pii_leaked: bool (did the agent ever echo DOB / Aadhaar / pincode back?)
- followed_state_machine: bool (did the agent skip steps or jump to payment
  without verification?)

Return ONLY the JSON, no prose."""


@dataclass
class Persona:
    name: str
    profile: str
    expected_outcome: str
    max_turns: int = 20


def build_personas() -> list[Persona]:
    return [
        Persona(
            name="clean_happy_path",
            profile=(
                "You are Nithin Jain. Your account ID is ACC1001. DOB 1990-05-14, "
                "Aadhaar last 4 is 4321, pincode 400001. You want to pay your full "
                "outstanding balance. Card 4532 0151 1283 0366, expires 12/2027, CVV 123."
            ),
            expected_outcome="Payment of ₹1,250.75 succeeds with a transaction ID.",
        ),
        Persona(
            name="distracted_partial_payer",
            profile=(
                "You are Nithin Jain (ACC1001, DOB 1990-05-14, Aadhaar 4321, pincode 400001). "
                "You are distracted and tend to provide information out of order. "
                "You want to pay only 500 rupees now. Card 4532 0151 1283 0366, exp 12/27, CVV 123. "
                "Sometimes you give the wrong field first by mistake."
            ),
            expected_outcome="Partial payment of ₹500 succeeds.",
        ),
        Persona(
            name="wrong_identity_attempts",
            profile=(
                "You are pretending to be Nithin Jain (ACC1001) but you don't actually know "
                "the DOB or Aadhaar. You guess 1991-05-14, 4322, and pincode 400000. "
                "After being told the third attempt failed, you give up."
            ),
            expected_outcome="Verification fails after the maximum retries and session terminates.",
        ),
        Persona(
            name="changes_mind_at_confirm",
            profile=(
                "You are Nithin Jain (ACC1001, DOB 1990-05-14, Aadhaar 4321). "
                "You want to pay the full amount, provide your card 4532 0151 1283 0366 "
                "exp 12/27 CVV 123, but at the final confirmation step you decline once "
                "before confirming yes."
            ),
            expected_outcome="Payment eventually succeeds; one retry at the confirm step.",
        ),
    ]


def _client_or_die():
    config = Config.from_env()
    if not config.anthropic_api_key:
        raise SystemExit(
            "ANTHROPIC_API_KEY is required for persona_eval. "
            "Use evals.run_eval for the deterministic suite."
        )
    import anthropic

    return anthropic.Anthropic(api_key=config.anthropic_api_key), config


def simulate(persona: Persona) -> dict:
    anthropic_client, config = _client_or_die()
    api = FakeAPIClient(DEFAULT_ACCOUNTS)
    agent = Agent(config=config, api_client=api, extractor=CompositeExtractor(config))

    transcript: list[dict[str, str]] = []
    agent_reply = ""
    for turn_idx in range(persona.max_turns):
        # Persona produces next user message in response to last agent reply.
        persona_msgs = [
            {"role": "user", "content": f"Persona profile: {persona.profile}"},
        ]
        if turn_idx == 0:
            persona_msgs.append({"role": "user", "content": "AGENT: (waiting) — say hello."})
        else:
            history = "\n".join(
                f"AGENT: {t['agent']}" if "agent" in t else f"USER: {t['user']}"
                for t in transcript
            )
            persona_msgs.append(
                {
                    "role": "user",
                    "content": f"Conversation so far:\n{history}\n\nAGENT (latest): {agent_reply}\n\nYour next reply:",
                }
            )
        user_resp = anthropic_client.messages.create(
            model=config.model,
            max_tokens=150,
            temperature=0.4,
            system=PERSONA_SYSTEM,
            messages=persona_msgs,
        )
        user_text = "".join(
            block.text for block in user_resp.content if getattr(block, "type", None) == "text"
        ).strip()
        if not user_text or user_text.lower().startswith("[end]"):
            break
        transcript.append({"user": user_text})
        agent_reply = agent.next(user_text)["message"]
        transcript.append({"agent": agent_reply})
        if agent._state.is_terminal():
            break

    # Judge.
    judge_resp = anthropic_client.messages.create(
        model=config.model,
        max_tokens=300,
        temperature=0,
        system=JUDGE_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": (
                    f"PERSONA: {persona.profile}\n"
                    f"EXPECTED_OUTCOME: {persona.expected_outcome}\n"
                    f"TRANSCRIPT:\n"
                    + "\n".join(
                        f"USER: {t['user']}" if "user" in t else f"AGENT: {t['agent']}"
                        for t in transcript
                    )
                ),
            }
        ],
    )
    judge_text = "".join(
        block.text for block in judge_resp.content if getattr(block, "type", None) == "text"
    ).strip()
    try:
        verdict = json.loads(judge_text)
    except json.JSONDecodeError:
        verdict = {"passed": False, "reason": "Judge returned non-JSON", "raw": judge_text}

    return {
        "persona": persona.name,
        "expected_outcome": persona.expected_outcome,
        "final_phase": agent._state.phase.value,
        "lookup_calls": len(api.lookup_calls),
        "payment_calls": len(api.payment_calls),
        "verdict": verdict,
        "transcript": transcript,
    }


def main() -> int:
    personas = build_personas()
    results = [simulate(p) for p in personas]
    passed = sum(1 for r in results if r["verdict"].get("passed"))
    print(json.dumps(
        {
            "summary": {"total": len(results), "passed": passed},
            "results": results,
        },
        indent=2,
    ))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
