"""Integration registry for external coding tools."""

from omlx.integrations.base import Integration
from omlx.integrations.claude import ClaudeCodeIntegration
from omlx.integrations.codex import CodexIntegration
from omlx.integrations.copilot import CopilotIntegration
from omlx.integrations.openclaw import OpenClawIntegration
from omlx.integrations.opencode import OpenCodeIntegration
from omlx.integrations.pi import PiIntegration

INTEGRATIONS: dict[str, Integration] = {
    "claude": ClaudeCodeIntegration(),
    "codex": CodexIntegration(),
    "opencode": OpenCodeIntegration(),
    "openclaw": OpenClawIntegration(),
    "pi": PiIntegration(),
    "copilot": CopilotIntegration(),
}


def get_integration(name: str) -> Integration | None:
    """Get an integration by name."""
    return INTEGRATIONS.get(name)


def list_integrations() -> list[Integration]:
    """List all available integrations."""
    return list(INTEGRATIONS.values())


__all__ = [
    "Integration",
    "ClaudeCodeIntegration",
    "CopilotIntegration",
    "INTEGRATIONS",
    "get_integration",
    "list_integrations",
]
