# SPDX-License-Identifier: Apache-2.0
"""Tests for automatic structured state updates."""

from pathlib import Path

from omlx.context.state_updater import update_agent_state_from_exchange
from omlx.context.memory import load_agent_state
from omlx.settings import GlobalSettings


def test_update_agent_state_from_exchange_populates_fields(tmp_path: Path):
    settings = GlobalSettings(base_path=tmp_path)

    update_agent_state_from_exchange(
        settings,
        selector={"profile": "planner"},
        input_messages=[
            {"role": "user", "content": "Build a long-running agent runtime. Next add retrieval."}
        ],
        output_messages=[
            {
                "role": "assistant",
                "content": "We will use Obsidian for durable memory. Decision: keep KV cache separate from memory.",
                "tool_calls": [{"function": {"name": "fs.read"}}],
            }
        ],
    )

    state = load_agent_state(settings, selector={"profile": "planner"})
    assert "Build a long-running agent runtime" in state.mission
    assert "Used tool: fs.read" in state.facts
    assert "Obsidian" in state.decisions
    assert "Next add retrieval" in state.todo
