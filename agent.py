"""Top-level entry point exposing the required `Agent` class.

The assignment specifies that evaluators will import an `Agent` class with a
`.next(user_input: str) -> dict` method. We keep the real implementation in the
`src/payment_agent` package and just re-export it here so the import path is
stable regardless of how the project is invoked.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from payment_agent.agent import Agent  # noqa: E402

__all__ = ["Agent"]
