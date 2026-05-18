"""Evaluation harness entry point.

Run with `python -m evals.run_eval`. Each scenario is replayed against a fresh
`Agent` instance backed by an in-memory API client; the harness then grades the
run against the scenario's expectations and prints a per-scenario report plus
aggregate metrics.

Metrics reported:
- success rate (scenarios passing all assertions)
- tool-call correctness (every payment scenario reaches the expected number of
  payment-API calls with the expected amount)
- PII safety (sensitive values never leaked in any agent message)
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
for p in (_ROOT, _SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from payment_agent.agent import Agent
from payment_agent.config import Config
from payment_agent.extractors import CompositeExtractor

from evals.scenarios import Scenario, all_scenarios
from tests.test_agent_flows import FakeAPIClient


@dataclass
class ScenarioResult:
    name: str
    description: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    transcript: list[tuple[str, str]] = field(default_factory=list)
    lookup_calls: int = 0
    payment_calls: int = 0


def run_scenario(scenario: Scenario) -> ScenarioResult:
    api = FakeAPIClient(scenario.accounts, payment_results=scenario.payment_results)
    config = Config.from_env()
    agent = Agent(config=config, api_client=api, extractor=CompositeExtractor(config))

    transcript: list[tuple[str, str]] = []
    for turn in scenario.user_turns:
        message = agent.next(turn)["message"]
        transcript.append((turn, message))

    failures: list[str] = []
    state = agent._state  # accessing internal state for assertions
    expect = scenario.expect

    if state.phase != expect.final_phase:
        failures.append(f"final_phase: expected {expect.final_phase}, got {state.phase}")
    if not (expect.min_payment_calls <= len(api.payment_calls) <= expect.max_payment_calls):
        failures.append(
            f"payment_calls: expected {expect.min_payment_calls}..{expect.max_payment_calls}, "
            f"got {len(api.payment_calls)}"
        )
    if not (expect.min_lookup_calls <= len(api.lookup_calls) <= expect.max_lookup_calls):
        failures.append(
            f"lookup_calls: expected {expect.min_lookup_calls}..{expect.max_lookup_calls}, "
            f"got {len(api.lookup_calls)}"
        )
    if expect.payment_amount is not None and api.payment_calls:
        actual = api.payment_calls[-1][1]
        if actual != expect.payment_amount:
            failures.append(f"payment_amount: expected {expect.payment_amount}, got {actual}")
    if expect.last_transaction_id_present and not state.last_transaction_id:
        failures.append("expected a transaction_id on success but none was recorded")

    full_text = "\n".join(reply for _, reply in transcript)
    for needle in expect.must_contain:
        if needle not in full_text:
            failures.append(f"missing required phrase: {needle!r}")
    for forbidden in expect.must_not_contain_in_any_message:
        if forbidden in full_text:
            failures.append(f"PII leak: {forbidden!r} appeared in an agent message")

    if expect.custom is not None:
        result = expect.custom(state, transcript, api)
        if result:
            failures.append(result)

    return ScenarioResult(
        name=scenario.name,
        description=scenario.description,
        passed=not failures,
        failures=failures,
        transcript=transcript,
        lookup_calls=len(api.lookup_calls),
        payment_calls=len(api.payment_calls),
    )


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    verbose = "--verbose" in argv or "-v" in argv
    json_only = "--json" in argv

    scenarios = all_scenarios()
    start = time.time()
    results = [run_scenario(s) for s in scenarios]
    elapsed = time.time() - start

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    if json_only:
        out = {
            "summary": {
                "total": total,
                "passed": passed,
                "success_rate": round(passed / total, 4) if total else 0.0,
                "elapsed_seconds": round(elapsed, 3),
            },
            "scenarios": [
                {**asdict(r), "transcript": r.transcript if verbose else []}
                for r in results
            ],
        }
        print(json.dumps(out, indent=2, default=str))
        return 0 if passed == total else 1

    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"[{mark}] {r.name:42s} lookups={r.lookup_calls} payments={r.payment_calls}")
        if not r.passed:
            for f in r.failures:
                print(f"        - {f}")
        if verbose:
            print(f"        — {r.description}")
            for u, b in r.transcript:
                print(f"        USER: {u}")
                print(f"         BOT: {b}")

    print()
    print(f"Summary: {passed}/{total} scenarios passed in {elapsed:.2f}s")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
