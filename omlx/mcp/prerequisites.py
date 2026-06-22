# SPDX-License-Identifier: Apache-2.0
"""MCP prerequisite enforcement — stateless per-request.

Builds executed-tool state from prior message history and checks
whether each tool call's declared prerequisites are satisfied.

Two declaration modes (adapted from Forge's StepTracker):
  - Name-only:  "read_file"  — any prior read_file call satisfies
  - Arg-matched: {"tool": "read_file", "match_arg": "path"}
                  — prior call must have matching arg value
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from omlx.api.guardrails.types import CheckResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrerequisiteCheck:
    """Result of checking prerequisites for a single tool call."""

    satisfied: bool
    missing: list[str]


class PrerequisiteChecker:
    """Check tool-call prerequisites against prior message history.

    Stateless per-request: ``executed_tools`` is rebuilt from
    ``prior_messages`` on every ``check()`` call. No server-side
    session state.
    """

    def __init__(self, prerequisites: dict[str, list]) -> None:
        self._prereqs = prerequisites

    def check(
        self,
        tool_calls: list[Any],
        prior_messages: list[dict],
    ) -> list[CheckResult]:
        """Check each tool call's prerequisites.

        Returns a CheckResult per tool call that has declared
        prerequisites. Tool calls with no declared prereqs produce
        no CheckResult (omitted, not passed).
        """
        if not self._prereqs or not tool_calls:
            return []

        executed = self._build_executed_set(prior_messages)
        results: list[CheckResult] = []

        for tc in tool_calls:
            name = self._tool_name(tc)
            spec = self._prereqs.get(name)
            if not spec:
                continue
            prereqs = spec.get("requires", spec) if isinstance(spec, dict) else spec
            if not prereqs:
                continue
            missing = self._evaluate(name, tc, prereqs, executed)
            results.append(
                CheckResult(
                    check="prerequisite",
                    passed=len(missing) == 0,
                    detail=(
                        None
                        if not missing
                        else f"Tool '{name}' missing prerequisites: {', '.join(missing)}"
                    ),
                )
            )
        return results

    def _build_executed_set(
        self, prior_messages: list[dict]
    ) -> dict[str, list[dict]]:
        """Scan prior assistant messages for tool_calls.

        Returns a mapping of tool_name -> list of arg dicts.
        """
        executed: dict[str, list[dict]] = {}
        for msg in prior_messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "assistant":
                continue
            calls = msg.get("tool_calls")
            if not calls:
                continue
            for call in calls:
                func = call.get("function") if isinstance(call, dict) else None
                if not func:
                    continue
                name = func.get("name", "")
                if not name:
                    continue
                raw_args = func.get("arguments", "{}")
                args = self._parse_args(raw_args)
                executed.setdefault(name, []).append(args)
        return executed

    def _evaluate(
        self,
        tool_name: str,
        tc: Any,
        prereqs: list,
        executed: dict[str, list[dict]],
    ) -> list[str]:
        """Return list of unsatisfied prerequisite tool names."""
        missing: list[str] = []
        tc_args = self._tc_args(tc)
        for prereq in prereqs:
            if isinstance(prereq, str):
                if prereq not in executed:
                    missing.append(prereq)
            elif isinstance(prereq, dict):
                prereq_tool = prereq.get("tool", "")
                match_arg = prereq.get("match_arg", "")
                if not prereq_tool:
                    continue
                required_value = tc_args.get(match_arg) if tc_args else None
                prior_calls = executed.get(prereq_tool, [])
                if not prior_calls:
                    missing.append(prereq_tool)
                    continue
                if not any(
                    c.get(match_arg) == required_value for c in prior_calls
                ):
                    missing.append(prereq_tool)
        return missing

    @staticmethod
    def _tool_name(tc: Any) -> str:
        """Extract tool name from a ToolCall or dict."""
        func = getattr(getattr(tc, "function", None), "name", None)
        if func:
            return func
        if isinstance(tc, dict):
            f = tc.get("function", {})
            return f.get("name", "") if isinstance(f, dict) else ""
        return ""

    @staticmethod
    def _tc_args(tc: Any) -> dict:
        """Extract args dict from a ToolCall or dict."""
        raw = None
        func = getattr(tc, "function", None)
        if func is not None:
            raw = getattr(func, "arguments", None)
        elif isinstance(tc, dict):
            f = tc.get("function", {})
            raw = f.get("arguments") if isinstance(f, dict) else None
        return PrerequisiteChecker._parse_args(raw)

    @staticmethod
    def _parse_args(raw: Any) -> dict:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else {}
            except (json.JSONDecodeError, ValueError):
                return {}
        return {}
