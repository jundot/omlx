# SPDX-License-Identifier: Apache-2.0
"""Context bootstrap helpers for agent-style request enrichment."""

from .bootstrap import (
    build_bootstrap_system_message,
    extract_context_selector,
    prepend_bootstrap_message,
    set_tool_catalog_getter,
)
from .external_agents import (
    build_external_agent_tools,
    execute_external_agent_tool,
    external_agent_tool_name,
    get_external_agents_overview,
)
from .runtime import (
    apply_skill_defaults,
    apply_tool_policy,
    get_runtime_overview,
    list_agent_profiles,
    list_agent_workspaces,
    resolve_agent_runtime_request,
)
from .skills import list_skills, load_skill, resolve_skills

__all__ = [
    "build_bootstrap_system_message",
    "extract_context_selector",
    "prepend_bootstrap_message",
    "set_tool_catalog_getter",
    "build_external_agent_tools",
    "execute_external_agent_tool",
    "external_agent_tool_name",
    "get_external_agents_overview",
    "apply_skill_defaults",
    "apply_tool_policy",
    "get_runtime_overview",
    "list_agent_profiles",
    "list_agent_workspaces",
    "resolve_agent_runtime_request",
    "list_skills",
    "load_skill",
    "resolve_skills",
]
