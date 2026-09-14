"""Integration registry for external coding tools."""

from omlx.integrations.base import Integration, IntegrationContext
from omlx.integrations.claude import ClaudeCodeIntegration
from omlx.integrations.codex import CodexIntegration
from omlx.integrations.codex_app import CodexAppIntegration
from omlx.integrations.copilot import CopilotIntegration
from omlx.integrations.hermes import HermesIntegration
from omlx.integrations.openclaw import OpenClawIntegration
from omlx.integrations.opencode import OpenCodeIntegration
from omlx.integrations.pi import PiIntegration
from omlx.integrations.swival import SwivalIntegration

INTEGRATIONS: dict[str, Integration] = {
    "claude": ClaudeCodeIntegration(),
    "codex": CodexIntegration(),
    "codex_app": CodexAppIntegration(),
    "opencode": OpenCodeIntegration(),
    "openclaw": OpenClawIntegration(),
    "hermes": HermesIntegration(),
    "pi": PiIntegration(),
    "copilot": CopilotIntegration(),
    "swival": SwivalIntegration(),
}


def get_integration(name: str) -> Integration | None:
    """Get an integration by name."""
    return INTEGRATIONS.get(name)


def list_integrations() -> list[Integration]:
    """List all available integrations."""
    return list(INTEGRATIONS.values())


__all__ = [
    "Integration",
    "IntegrationContext",
    "ClaudeCodeIntegration",
    "CodexAppIntegration",
    "CopilotIntegration",
    "HermesIntegration",
    "SwivalIntegration",
    "INTEGRATIONS",
    "get_integration",
    "list_integrations",
]
