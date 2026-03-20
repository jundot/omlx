# SPDX-License-Identifier: Apache-2.0
"""Tests for local skill bundles."""

from pathlib import Path

from omlx.context.bootstrap import build_bootstrap_system_message
from omlx.context.runtime import apply_skill_defaults, resolve_agent_runtime_request
from omlx.settings import GlobalSettings, BootstrapSettings


def test_skill_bundle_affects_runtime_and_bootstrap(tmp_path: Path):
    skill_dir = tmp_path / "skills" / "planner"
    skill_dir.mkdir(parents=True)
    (skill_dir / "IDENTITY.md").write_text("Plan carefully.", encoding="utf-8")
    (skill_dir / "RULES.md").write_text("Always update TODO.", encoding="utf-8")
    (skill_dir / "TOOLS.json").write_text(
        '{"tool_mode":"mcp_only","allowed_tools":["mcp.search"]}',
        encoding="utf-8",
    )

    settings = GlobalSettings(base_path=tmp_path)
    settings.bootstrap = BootstrapSettings(enabled=True, include_tool_catalog=False)

    runtime = resolve_agent_runtime_request({"omlx_context": {"skills": ["planner"]}})
    runtime = apply_skill_defaults(settings, runtime)

    assert runtime.tool_policy.mode == "mcp_only"
    assert runtime.tool_policy.allowed_tools == ["mcp.search"]

    message = build_bootstrap_system_message(
        settings,
        selector={},
        skill_names=["planner"],
    )
    assert message is not None
    assert "[Skills]" in message
    assert "Plan carefully." in message
