# SPDX-License-Identifier: Apache-2.0
"""Tests for external-agent runtime support."""

from omlx.context.external_agents import (
    build_external_agent_tools,
    execute_external_agent_tool,
    external_agent_tool_name,
)
from omlx.settings import ExternalAgentDefinition, GlobalSettings


def test_build_external_agent_tools_exposes_registered_agent():
    settings = GlobalSettings()
    settings.integrations.external_agents = [
        ExternalAgentDefinition(
            name="Codex Remote",
            endpoint="http://127.0.0.1:9000/bridge",
            description="Delegate coding tasks.",
        )
    ]

    tools = build_external_agent_tools(settings)

    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "agent__codex_remote"
    assert "prompt" in tools[0]["function"]["parameters"]["properties"]


async def _fake_post_json(url, payload, headers, timeout):
    return {
        "id": "resp_123",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": f"remote: {payload['input']}"}],
            }
        ],
    }


def test_execute_external_agent_tool_supports_openai_responses(monkeypatch):
    settings = GlobalSettings()
    settings.integrations.external_agents = [
        ExternalAgentDefinition(
            name="planner",
            endpoint="http://127.0.0.1:9000/v1/responses",
            protocol="openai_responses",
            model="gpt-test",
            default_skills=["plan"],
        )
    ]

    monkeypatch.setattr(
        "omlx.context.external_agents._post_json",
        _fake_post_json,
    )

    result = __import__("asyncio").run(
        execute_external_agent_tool(
            settings,
            external_agent_tool_name("planner"),
            {
                "prompt": "build roadmap",
                "workspace": "ops",
                "skills": ["review"],
            },
            caller_metadata={"agent_id": "workspace:lead"},
        )
    )

    assert result["agent"] == "planner"
    assert result["tool_name"] == "agent__planner"
    assert result["output_text"] == "remote: build roadmap"
