"""Runtime configuration for the agent.

Loads from environment variables (with `.env` support) so the agent can be run
in tests with deterministic settings and in production without code changes.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

try:  # pragma: no cover - import side effect
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional at runtime
    pass


DEFAULT_API_BASE_URL = (
    "https://se-payment-verification-api.service.external.usea2.aws.prodigaltech.com"
)
DEFAULT_MODEL = "claude-sonnet-4-6"


@dataclass(frozen=True)
class Config:
    """Immutable runtime configuration."""

    api_base_url: str
    model: str
    anthropic_api_key: str | None
    log_level: int
    max_verification_attempts: int
    max_payment_attempts: int
    http_timeout_seconds: float

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            api_base_url=os.getenv("PAYMENT_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/"),
            model=os.getenv("PAYMENT_AGENT_MODEL", DEFAULT_MODEL),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
            log_level=_parse_log_level(os.getenv("PAYMENT_AGENT_LOG_LEVEL", "INFO")),
            max_verification_attempts=int(os.getenv("PAYMENT_AGENT_MAX_VERIFY_ATTEMPTS", "3")),
            max_payment_attempts=int(os.getenv("PAYMENT_AGENT_MAX_PAYMENT_ATTEMPTS", "3")),
            http_timeout_seconds=float(os.getenv("PAYMENT_AGENT_HTTP_TIMEOUT", "15")),
        )


def _parse_log_level(raw: str) -> int:
    level = getattr(logging, raw.upper(), None)
    return level if isinstance(level, int) else logging.INFO
