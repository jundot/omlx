# SPDX-License-Identifier: Apache-2.0
"""Error budget for client-driven retry loops.

The server provides configurable defaults serialized in
``x_omlx_validation.budget``. The client tracks actual retry and
tool-error counts locally and enforces the budget.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorBudget:
    """Advisory retry/tool-error limits for client-side retry loops.

    The server does NOT track per-session counts. It serializes these
    defaults via :meth:`to_dict` into the ``x_omlx_validation.budget``
    extension. The client calls :meth:`should_retry` with its own
    tracked counts.
    """

    max_retries: int = 3
    max_tool_errors: int = 2

    def should_retry(self, retry_count: int, tool_error_count: int) -> bool:
        """Return True if both counts are within budget (inclusive)."""
        return retry_count <= self.max_retries and tool_error_count <= self.max_tool_errors

    def recommended_action(self, retry_count: int = 0, tool_error_count: int = 0) -> str:
        """Return ``"retry"`` if under budget, ``"give_up"`` if exhausted."""
        return "retry" if self.should_retry(retry_count, tool_error_count) else "give_up"

    def to_dict(self) -> dict:
        """Serialize for ``x_omlx_validation.budget`` response extension.

        ``recommended_action`` is always ``"retry"`` in the serialized
        form because the server does not know client-side counts. The
        client calls :meth:`recommended_action` locally for dynamic
        decisions.
        """
        return {
            "max_retries": self.max_retries,
            "max_tool_errors": self.max_tool_errors,
            "recommended_action": "retry",
        }

    @classmethod
    def from_dict(cls, data: dict) -> ErrorBudget:
        """Deserialize from dict (ignores ``recommended_action``)."""
        return cls(
            max_retries=data.get("max_retries", 3),
            max_tool_errors=data.get("max_tool_errors", 2),
        )
