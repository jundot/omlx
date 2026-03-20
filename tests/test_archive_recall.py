# SPDX-License-Identifier: Apache-2.0
"""Tests for archive recall injection."""

from pathlib import Path

from omlx.context.bootstrap import build_bootstrap_system_message
from omlx.context.session_store import append_session_record, rollover_session
from omlx.settings import GlobalSettings, BootstrapSettings


def test_build_bootstrap_system_message_includes_archive_recall(tmp_path: Path):
    settings = GlobalSettings(base_path=tmp_path)
    settings.bootstrap = BootstrapSettings(enabled=True, include_tool_catalog=False)
    settings.agent_memory.retrieval_enabled = True

    metadata = {"agent_id": "profile:planner", "omlx_context": {"profile": "planner"}}
    append_session_record(
        settings,
        public_response={"id": "resp_1", "metadata": metadata, "output": []},
        input_messages=[{"role": "user", "content": "Need Obsidian recall and task index."}],
        output_messages=[{"role": "assistant", "content": "Stored the note."}],
    )
    rollover_session(settings, metadata)

    message = build_bootstrap_system_message(
        settings,
        selector={"profile": "planner"},
        retrieval_query="Can you find the Obsidian recall notes?",
    )

    assert message is not None
    assert "[Archive Recall]" in message
    assert "Obsidian" in message
