# SPDX-License-Identifier: Apache-2.0
"""Message phase on Responses output items.

Codex folds reasoning and tool blocks under the narration only when the
message item names its phase, so the field has to be present, correctly named,
and absent for shapes where it cannot be known.
"""

import ast
from pathlib import Path

from omlx.api.responses_models import OutputItem
from omlx.api.responses_utils import (
    assistant_message_phase,
    build_message_output_item,
    build_response_store_record,
    normalize_response_output_to_messages,
)


def _server_function(name: str):
    source = (Path(__file__).resolve().parents[1] / "omlx" / "server.py").read_text()
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return node
    raise AssertionError(f"{name} not found in server.py")


class TestMessagePhase:
    def test_tool_turn_ending_in_a_call_is_commentary(self):
        item = build_message_output_item("Checking the file.", phase="commentary")
        assert item.phase == "commentary"
        assert item.model_dump()["phase"] == "commentary"

    def test_tool_turn_that_stops_is_final_answer(self):
        item = build_message_output_item("Done.", phase="final_answer")
        assert item.model_dump()["phase"] == "final_answer"

    def test_phase_is_omitted_when_unknown(self):
        """Tool-less requests cannot tell narration from the answer."""
        item = build_message_output_item("Hello!")
        assert "phase" not in item.model_dump(), (
            "an unknown phase must stay off the wire rather than be guessed"
        )

    def test_phase_helper_only_names_phases_for_tool_requests(self):
        assert assistant_message_phase(has_tools=True, has_tool_calls=True) == (
            "commentary"
        )
        assert assistant_message_phase(has_tools=True, has_tool_calls=False) == (
            "final_answer"
        )
        assert assistant_message_phase(has_tools=False, has_tool_calls=False) is None
        assert assistant_message_phase(has_tools=False, has_tool_calls=True) is None

    def test_non_message_items_have_no_phase(self):
        call = OutputItem(type="function_call", id="fc_1", call_id="call_1")
        reasoning = OutputItem(type="reasoning", id="rs_1")
        assert "phase" not in call.model_dump()
        assert "phase" not in reasoning.model_dump()

    def test_history_conversion_ignores_phase(self):
        """The phase is presentation metadata; chat history must not gain a field."""
        messages = normalize_response_output_to_messages(
            [
                {
                    "type": "message",
                    "role": "assistant",
                    "phase": "commentary",
                    "content": [{"type": "output_text", "text": "looking"}],
                }
            ]
        )
        assert messages == [{"role": "assistant", "content": "looking"}]

    def test_store_record_keeps_phase_in_public_response(self):
        public = {
            "id": "resp_1",
            "created_at": 1,
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": "answer"}],
                }
            ],
        }
        record = build_response_store_record(
            public, input_messages=[], output_messages=[{"role": "assistant", "content": "answer"}]
        )
        assert record["public_response"]["output"][0]["phase"] == "final_answer"


class TestServerWiring:
    """Static guards: both Responses paths must name the message phase."""

    def test_nonstream_response_builds_phase(self):
        node = _server_function("create_response")
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "build_message_output_item"
            ):
                assert any(
                    keyword.arg == "phase" for keyword in call.keywords
                ), "create_response must name the message phase"
                break
        else:
            raise AssertionError("create_response must build a message output item")

    def test_stream_response_builds_phase(self):
        node = _server_function("stream_responses_api")
        called = {
            call.func.id
            for call in ast.walk(node)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        }
        assert "assistant_message_phase" in called, (
            "stream_responses_api must name the message phase"
        )
