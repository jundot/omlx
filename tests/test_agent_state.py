# SPDX-License-Identifier: Apache-2.0
"""Tests for structured agent state."""

from pathlib import Path

from omlx.context.memory import (
    AgentState,
    build_agent_state_system_appendix,
    load_agent_state,
    save_agent_state,
)
from omlx.settings import GlobalSettings


def test_save_and_load_agent_state(tmp_path: Path):
    settings = GlobalSettings(base_path=tmp_path)
    state = save_agent_state(
        settings,
        AgentState(
            mission="Ship ACPX runtime.",
            facts="KV cache is not shared understanding.",
            decisions="Use Obsidian for long-term memory.",
            todo="Build session board.",
        ),
        selector={"profile": "planner"},
    )

    assert state.mission == "Ship ACPX runtime."
    loaded = load_agent_state(settings, selector={"profile": "planner"})
    assert loaded.decisions == "Use Obsidian for long-term memory."


def test_build_agent_state_system_appendix(tmp_path: Path):
    settings = GlobalSettings(base_path=tmp_path)
    save_agent_state(
        settings,
        AgentState(
            mission="Coordinate multiple agents.",
            facts="Workspace is shared.",
        ),
        selector={"workspace": "ops"},
    )

    appendix = build_agent_state_system_appendix(
        settings,
        selector={"workspace": "ops"},
    )

    assert appendix is not None
    assert "[Mission]" in appendix
    assert "Coordinate multiple agents." in appendix
