# SPDX-License-Identifier: Apache-2.0
"""Tests for markdown bootstrap context injection."""

from pathlib import Path

from omlx.api.adapters.anthropic import AnthropicAdapter
from omlx.api.adapters.openai import OpenAIAdapter
from omlx.api.anthropic_models import AnthropicMessage, MessagesRequest
from omlx.api.openai_models import ChatCompletionRequest, Message
from omlx.context.bootstrap import (
    build_bootstrap_system_message,
    extract_context_selector,
    prepend_bootstrap_message,
)
from omlx.api.adapters.base import InternalMessage
from omlx.settings import BootstrapSettings, GlobalSettings, reset_settings, init_settings


def test_build_bootstrap_system_message_reads_markdown_files(tmp_path: Path):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "IDENTITY.md").write_text("You are concise.", encoding="utf-8")
    (agent_dir / "MEMORY.md").write_text("User prefers Chinese.", encoding="utf-8")

    settings = GlobalSettings(base_path=tmp_path)
    settings.bootstrap = BootstrapSettings(enabled=True)

    message = build_bootstrap_system_message(settings)

    assert message is not None
    assert "[Identity]" in message
    assert "You are concise." in message
    assert "[Memory]" in message
    assert "User prefers Chinese." in message


def test_prepend_bootstrap_message_skips_when_disabled(tmp_path: Path):
    settings = GlobalSettings(base_path=tmp_path)
    settings.bootstrap = BootstrapSettings(enabled=False)

    messages = [InternalMessage(role="user", content="hello")]
    result = prepend_bootstrap_message(messages, settings)

    assert result == messages


def test_build_bootstrap_system_message_uses_profile_selector(tmp_path: Path):
    profile_dir = tmp_path / "profiles" / "operator" / "agent"
    profile_dir.mkdir(parents=True)
    (profile_dir / "IDENTITY.md").write_text("Profile operator.", encoding="utf-8")

    settings = GlobalSettings(base_path=tmp_path)
    settings.bootstrap = BootstrapSettings(enabled=True)

    message = build_bootstrap_system_message(
        settings,
        selector={"profile": "operator"},
    )

    assert message is not None
    assert "Profile operator." in message


def test_build_bootstrap_system_message_uses_workspace_selector(tmp_path: Path):
    workspace_dir = tmp_path / "workspaces" / "client-a" / "agent"
    workspace_dir.mkdir(parents=True)
    (workspace_dir / "USER.md").write_text("Workspace client A.", encoding="utf-8")

    settings = GlobalSettings(base_path=tmp_path)
    settings.bootstrap = BootstrapSettings(enabled=True)

    message = build_bootstrap_system_message(
        settings,
        selector={"workspace": "client-a"},
    )

    assert message is not None
    assert "Workspace client A." in message


def test_build_bootstrap_system_message_rejects_unsafe_selector(tmp_path: Path):
    default_dir = tmp_path / "agent"
    default_dir.mkdir()
    (default_dir / "IDENTITY.md").write_text("Default context.", encoding="utf-8")

    settings = GlobalSettings(base_path=tmp_path)
    settings.bootstrap = BootstrapSettings(enabled=True)

    message = build_bootstrap_system_message(
        settings,
        selector={"workspace": "../escape"},
    )

    assert message is not None
    assert "Default context." in message
    assert "escape" not in message


def test_openai_adapter_injects_bootstrap_system_message(tmp_path: Path):
    reset_settings()
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "IDENTITY.md").write_text("Act like an operator.", encoding="utf-8")
    init_settings(base_path=tmp_path)

    adapter = OpenAIAdapter()
    request = ChatCompletionRequest(
        model="test-model",
        messages=[Message(role="user", content="hello")],
    )

    internal = adapter.parse_request(request)

    assert internal.messages[0].role == "system"
    assert "Act like an operator." in internal.messages[0].content
    assert internal.messages[1].role == "user"
    reset_settings()


def test_openai_adapter_supports_profile_selector(tmp_path: Path):
    reset_settings()
    profile_dir = tmp_path / "profiles" / "focused" / "agent"
    profile_dir.mkdir(parents=True)
    (profile_dir / "IDENTITY.md").write_text("Focused profile.", encoding="utf-8")
    init_settings(base_path=tmp_path)

    adapter = OpenAIAdapter()
    request = ChatCompletionRequest(
        model="test-model",
        messages=[Message(role="user", content="hello")],
        metadata={"omlx_context": {"profile": "focused"}},
    )

    internal = adapter.parse_request(request)

    assert internal.messages[0].role == "system"
    assert "Focused profile." in internal.messages[0].content
    reset_settings()


def test_anthropic_adapter_injects_bootstrap_system_message(tmp_path: Path):
    reset_settings()
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "USER.md").write_text("Reply in Chinese.", encoding="utf-8")
    init_settings(base_path=tmp_path)

    adapter = AnthropicAdapter()
    request = MessagesRequest(
        model="test-model",
        max_tokens=32,
        messages=[AnthropicMessage(role="user", content="hello")],
    )

    internal = adapter.parse_request(request)

    assert internal.messages[0].role == "system"
    assert "Reply in Chinese." in internal.messages[0].content
    assert internal.messages[1].role == "user"
    reset_settings()


def test_anthropic_adapter_supports_workspace_selector(tmp_path: Path):
    reset_settings()
    workspace_dir = tmp_path / "workspaces" / "ops" / "agent"
    workspace_dir.mkdir(parents=True)
    (workspace_dir / "USER.md").write_text("Ops workspace.", encoding="utf-8")
    init_settings(base_path=tmp_path)

    adapter = AnthropicAdapter()
    request = MessagesRequest(
        model="test-model",
        max_tokens=32,
        messages=[AnthropicMessage(role="user", content="hello")],
        omlx_context={"workspace": "ops"},
    )

    internal = adapter.parse_request(request)

    assert internal.messages[0].role == "system"
    assert "Ops workspace." in internal.messages[0].content
    reset_settings()


def test_extract_context_selector_prefers_explicit_field():
    request = ChatCompletionRequest(
        model="test-model",
        messages=[Message(role="user", content="hello")],
        metadata={"omlx_context": {"profile": "from-meta"}},
        omlx_context={"profile": "explicit", "workspace": "team-a"},
    )

    selector = extract_context_selector(request)

    assert selector == {"profile": "explicit", "workspace": "team-a"}
