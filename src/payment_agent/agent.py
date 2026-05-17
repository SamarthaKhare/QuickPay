"""Stub Agent used during scaffolding.

This module is replaced in a follow-up commit with the real state-machine
implementation. We keep a minimal `Agent` here so that the import surface in
`agent.py` is valid from the very first commit.
"""

from __future__ import annotations


class Agent:
    """Placeholder agent — real implementation lands in subsequent commits."""

    def next(self, user_input: str) -> dict:  # noqa: D401, ARG002
        return {
            "message": (
                "QuickPay agent is being initialised. The full payment flow "
                "will be available once setup completes."
            )
        }
