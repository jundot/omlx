# SPDX-License-Identifier: Apache-2.0
"""Tests for agent runtime selection and tool policies."""

from pathlib import Path

from omlx.context.runtime import (
    apply_tool_policy,
    list_agent_profiles,
    list_agent_workspaces,
    resolve_agent_runtime_request,
)


def test_resolve_agent_runtime_request_reads_context_fields():
    runtime = resolve_agent_runtime_request({
        "metadata": {"omlx_context": {"profile": "meta"}},
        "omlx_context": {
            "profile": "focused",
            "workspace": "ops",
            "tool_mode": "mcp_only",
            "allowed_tools": ["fs.read", "web.search"],
            "blocked_tools": ["fs.write"],
            "prefer_parallel_tools": True,
        },
    })

    assert runtime.selector == {"profile": "focused", "workspace": "ops"}
    assert runtime.tool_policy.mode == "mcp_only"
    assert runtime.tool_policy.allowed_tools == ["fs.read", "web.search"]
    assert runtime.tool_policy.blocked_tools == ["fs.write"]
    assert runtime.tool_policy.prefer_parallel_tools is True


def test_apply_tool_policy_filters_and_deduplicates():
    runtime = resolve_agent_runtime_request({
        "omlx_context": {
            "tool_mode": "all",
            "allowed_tools": ["mcp.search", "user.calc"],
            "blocked_tools": ["user.calc"],
        },
    })
    tools = apply_tool_policy(
        user_tools=[
            {"function": {"name": "user.calc"}},
            {"function": {"name": "shared.tool"}},
        ],
        mcp_tools=[
            {"function": {"name": "mcp.search"}},
            {"function": {"name": "shared.tool"}},
        ],
        runtime=runtime,
    )

    assert tools == [{"function": {"name": "mcp.search"}}]


def test_apply_tool_policy_supports_none_mode():
    runtime = resolve_agent_runtime_request({
        "omlx_context": {"tool_mode": "none"},
    })

    assert apply_tool_policy(
        user_tools=[{"function": {"name": "user.calc"}}],
        mcp_tools=[{"function": {"name": "mcp.search"}}],
        runtime=runtime,
    ) is None


def test_list_agent_profiles_and_workspaces(tmp_path: Path):
    (tmp_path / "profiles" / "focused" / "agent").mkdir(parents=True)
    (tmp_path / "profiles" / "ops" / "agent").mkdir(parents=True)
    (tmp_path / "workspaces" / "client-a" / "agent").mkdir(parents=True)

    assert list_agent_profiles(tmp_path) == ["focused", "ops"]
    assert list_agent_workspaces(tmp_path) == ["client-a"]
