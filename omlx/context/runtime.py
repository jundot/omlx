# SPDX-License-Identifier: Apache-2.0
"""Agent runtime helpers for context and tool-policy resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bootstrap import extract_context_selector
from .external_agents import get_external_agents_overview
from .skills import list_skills, resolve_skills
from ..settings import GlobalSettings

_TOOL_MODE_VALUES = {"all", "mcp_only", "user_only", "none"}


def _coerce_string_list(value: Any) -> list[str]:
    """Normalize a value into a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",")]
    elif isinstance(value, list):
        items = [item.strip() for item in value if isinstance(item, str)]
    else:
        return []
    return [item for item in items if item]


@dataclass
class ToolPolicy:
    """Request-level tool visibility policy."""

    mode: str = "all"
    allowed_tools: list[str] = field(default_factory=list)
    blocked_tools: list[str] = field(default_factory=list)
    prefer_parallel_tools: bool | None = None


@dataclass
class AgentRuntimeRequest:
    """Resolved request-scoped runtime options."""

    selector: dict[str, str] = field(default_factory=dict)
    tool_policy: ToolPolicy = field(default_factory=ToolPolicy)
    skills: list[str] = field(default_factory=list)


def resolve_agent_runtime_request(payload: Any) -> AgentRuntimeRequest:
    """Extract runtime selector and tool policy from a request payload."""
    selector = extract_context_selector(payload)

    if payload is None:
        source: dict[str, Any] = {}
    elif isinstance(payload, dict):
        source = payload
    else:
        model_dump = getattr(payload, "model_dump", None)
        source = model_dump() if callable(model_dump) else {}

    context = source.get("omlx_context") or {}
    if not isinstance(context, dict):
        context = {}

    metadata = source.get("metadata") or {}
    if isinstance(metadata, dict):
        meta_context = metadata.get("omlx_context")
        if isinstance(meta_context, dict):
            context = {**meta_context, **context}

    mode = context.get("tool_mode")
    if not isinstance(mode, str) or mode not in _TOOL_MODE_VALUES:
        mode = "all"

    parallel = context.get("prefer_parallel_tools")
    if not isinstance(parallel, bool):
        parallel = None

    skills = _coerce_string_list(context.get("skills"))

    return AgentRuntimeRequest(
        selector=selector,
        tool_policy=ToolPolicy(
            mode=mode,
            allowed_tools=_coerce_string_list(context.get("allowed_tools")),
            blocked_tools=_coerce_string_list(context.get("blocked_tools")),
            prefer_parallel_tools=parallel,
        ),
        skills=skills,
    )


def apply_tool_policy(
    user_tools: list[dict[str, Any]] | None,
    mcp_tools: list[dict[str, Any]] | None,
    runtime: AgentRuntimeRequest,
) -> list[dict[str, Any]] | None:
    """Resolve the effective tool list for a request."""
    if runtime.tool_policy.mode == "none":
        return None

    merged: list[dict[str, Any]] = []
    if runtime.tool_policy.mode in {"all", "mcp_only"}:
        merged.extend(mcp_tools or [])
    if runtime.tool_policy.mode in {"all", "user_only"}:
        merged.extend(user_tools or [])

    if not merged:
        return None

    deduped: dict[str, dict[str, Any]] = {}
    for tool in merged:
        name = tool.get("function", {}).get("name", "")
        if name:
            deduped[name] = tool

    allowed = set(runtime.tool_policy.allowed_tools)
    blocked = set(runtime.tool_policy.blocked_tools)

    filtered = []
    for name, tool in deduped.items():
        if allowed and name not in allowed:
            continue
        if name in blocked:
            continue
        filtered.append(tool)

    return filtered or None


def apply_skill_defaults(
    settings: GlobalSettings,
    runtime: AgentRuntimeRequest,
) -> AgentRuntimeRequest:
    """Merge skill-defined tool defaults into the runtime request."""
    bundles = resolve_skills(settings.base_path, runtime.skills)
    if not bundles:
        return runtime

    mode = runtime.tool_policy.mode
    allowed = list(runtime.tool_policy.allowed_tools)
    blocked = list(runtime.tool_policy.blocked_tools)
    for bundle in bundles:
        if mode == "all" and bundle.tool_mode:
            mode = bundle.tool_mode
        allowed.extend(bundle.allowed_tools)
        blocked.extend(bundle.blocked_tools)

    runtime.tool_policy.mode = mode
    runtime.tool_policy.allowed_tools = list(dict.fromkeys(allowed))
    runtime.tool_policy.blocked_tools = list(dict.fromkeys(blocked))
    return runtime


def list_agent_profiles(base_path: Path) -> list[str]:
    """List available agent profiles under the base path."""
    root = base_path / "profiles"
    if not root.exists():
        return []
    return sorted(
        path.name for path in root.iterdir() if path.is_dir() and (path / "agent").is_dir()
    )


def list_agent_workspaces(base_path: Path) -> list[str]:
    """List available agent workspaces under the base path."""
    root = base_path / "workspaces"
    if not root.exists():
        return []
    return sorted(
        path.name for path in root.iterdir() if path.is_dir() and (path / "agent").is_dir()
    )


def get_runtime_overview(
    settings: GlobalSettings,
    mcp_tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a summary of the local agent runtime state."""
    base_path = settings.base_path
    return {
        "base_path": str(base_path),
        "default_agent_dir": str(base_path / "agent"),
        "profiles": list_agent_profiles(base_path),
        "workspaces": list_agent_workspaces(base_path),
        "skills": list_skills(base_path),
        "mcp_tools": [tool.get("name", "") for tool in (mcp_tools or []) if tool.get("name")],
        "external_agents": get_external_agents_overview(settings),
    }
