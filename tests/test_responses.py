# SPDX-License-Identifier: Apache-2.0
"""Tests for OpenAI Responses API models and utilities."""

import json

import pytest

from omlx.api.responses_models import (
    InputItem,
    ResponseObject,
    ResponsesRequest,
    ResponsesTool,
    ResponseUsage,
    TextConfig,
    TextFormatConfig,
)
from omlx.api.responses_utils import (
    ResponseStateCorruptError,
    ResponseStateNotFoundError,
    ResponseStore,
    build_function_call_output_item,
    build_message_output_item,
    build_reasoning_output_item,
    build_response_object,
    build_response_store_record,
    build_response_usage,
    convert_responses_input_to_messages,
    convert_responses_tools,
    format_sse_event,
    normalize_response_output_to_messages,
    split_namespace_tool_name,
    validate_responses_request,
)
from omlx.api.shared_models import IDPrefix, generate_id
from omlx.api.tool_bindings import ToolBindingRegistry, ensure_call_id
from omlx.exceptions import InvalidRequestError

# =============================================================================
# ID Generation Tests
# =============================================================================


class TestIDGeneration:
    def test_response_id_prefix(self):
        rid = generate_id(IDPrefix.RESPONSE)
        assert rid.startswith("resp_")
        assert len(rid) == 29  # "resp_" + 24 hex chars

    def test_function_call_id_prefix(self):
        fid = generate_id(IDPrefix.FUNCTION_CALL)
        assert fid.startswith("fc_")
        assert len(fid) == 11  # "fc_" + 8 hex chars

    def test_reasoning_id_prefix(self):
        rid = generate_id(IDPrefix.REASONING)
        assert rid.startswith("rs_")
        assert len(rid) == 27  # "rs_" + 24 hex chars

    def test_unique_ids(self):
        ids = {generate_id(IDPrefix.RESPONSE) for _ in range(100)}
        assert len(ids) == 100


# =============================================================================
# Input Conversion Tests
# =============================================================================


class TestConvertResponsesInput:
    def test_string_input(self):
        messages = convert_responses_input_to_messages("Hello world")
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Hello world"}

    def test_none_input(self):
        messages = convert_responses_input_to_messages(None)
        assert messages == []

    def test_instructions_prepended(self):
        messages = convert_responses_input_to_messages(
            "Hi", instructions="You are helpful"
        )
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "You are helpful"}
        assert messages[1] == {"role": "user", "content": "Hi"}

    def test_message_items(self):
        items = [
            InputItem(type="message", role="user", content="Hello"),
            InputItem(type="message", role="assistant", content="Hi there"),
            InputItem(type="message", role="user", content="How are you?"),
        ]
        messages = convert_responses_input_to_messages(items)
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["content"] == "How are you?"

    def test_easy_input_message_no_type(self):
        """EasyInputMessage has no type field — just role and content."""
        items = [
            InputItem(role="user", content="Hello"),
            InputItem(role="assistant", content="Hi"),
        ]
        assert items[0].type is None  # No type set
        messages = convert_responses_input_to_messages(items)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"

    def test_easy_input_message_from_dict(self):
        """Verify Pydantic parses EasyInputMessage dict (no type field)."""
        raw = {"role": "user", "content": "Hello from dict"}
        item = InputItem(**raw)
        assert item.type is None
        assert item.role == "user"
        messages = convert_responses_input_to_messages([item])
        assert messages[0]["content"] == "Hello from dict"

    def test_developer_role_maps_to_system(self):
        items = [
            InputItem(type="message", role="developer", content="Be concise"),
            InputItem(type="message", role="user", content="Hi"),
        ]
        messages = convert_responses_input_to_messages(items)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be concise"
        assert messages[1]["role"] == "user"

    def test_developer_role_no_type(self):
        """EasyInputMessage with developer role (no type field)."""
        items = [InputItem(role="developer", content="Be brief")]
        messages = convert_responses_input_to_messages(items)
        assert messages[0]["role"] == "system"

    def test_instructions_and_developer_merged(self):
        """instructions + developer role in input → single system message."""
        items = [
            InputItem(role="developer", content="Working dir: /foo"),
            InputItem(role="user", content="Hello"),
        ]
        messages = convert_responses_input_to_messages(
            items, instructions="You are helpful"
        )
        # Should be exactly 2: one merged system + one user
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "You are helpful" in messages[0]["content"]
        assert "Working dir: /foo" in messages[0]["content"]
        assert messages[1]["role"] == "user"

    def test_multiple_developer_messages_merged(self):
        """Multiple developer messages all merge into one system message."""
        items = [
            InputItem(role="developer", content="Rule 1"),
            InputItem(role="developer", content="Rule 2"),
            InputItem(role="user", content="Hi"),
        ]
        messages = convert_responses_input_to_messages(items)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "Rule 1" in messages[0]["content"]
        assert "Rule 2" in messages[0]["content"]

    def test_developer_position_can_be_deferred(self):
        """Server path can preserve developer/system position until template probe."""
        items = [
            InputItem(role="user", content="Hi"),
            InputItem(role="developer", content="Plan mode"),
        ]
        messages = convert_responses_input_to_messages(
            items,
            consolidate_system_messages=False,
        )
        assert [m["role"] for m in messages] == ["user", "system"]
        assert messages[1]["content"] == "Plan mode"

    def test_function_call_items(self):
        items = [
            InputItem(type="message", role="user", content="What's the weather?"),
            InputItem(
                type="function_call",
                call_id="call_abc",
                name="get_weather",
                arguments='{"location": "Paris"}',
            ),
            InputItem(
                type="function_call_output",
                call_id="call_abc",
                output='{"temp": 22}',
            ),
        ]
        messages = convert_responses_input_to_messages(items)
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        # function_call → assistant with tool_calls
        assert messages[1]["role"] == "assistant"
        assert len(messages[1]["tool_calls"]) == 1
        assert messages[1]["tool_calls"][0]["function"]["name"] == "get_weather"
        # arguments should be parsed as dict (not string) for Jinja2 chat templates
        assert messages[1]["tool_calls"][0]["function"]["arguments"] == {
            "location": "Paris"
        }
        # function_call_output → tool message
        assert messages[2]["role"] == "tool"
        assert messages[2]["tool_call_id"] == "call_abc"
        assert messages[2]["content"] == '{"temp": 22}'

    def test_multiple_function_calls_grouped(self):
        """Multiple consecutive function_calls should be grouped into one assistant message."""
        items = [
            InputItem(type="message", role="user", content="Do two things"),
            InputItem(
                type="function_call",
                call_id="call_1",
                name="fn_a",
                arguments="{}",
            ),
            InputItem(
                type="function_call",
                call_id="call_2",
                name="fn_b",
                arguments="{}",
            ),
            InputItem(
                type="function_call_output",
                call_id="call_1",
                output="result_a",
            ),
        ]
        messages = convert_responses_input_to_messages(items)
        # user, assistant (2 tool_calls), tool (output for call_1)
        assert len(messages) == 3
        assert messages[1]["role"] == "assistant"
        assert len(messages[1]["tool_calls"]) == 2

    def test_assistant_message_merged_with_function_call(self):
        """Assistant message followed by function_call should produce a single
        assistant message (not two), preventing duplicate assistant turns that
        confuse chat templates."""
        items = [
            InputItem(type="message", role="user", content="Do something"),
            InputItem(
                type="message",
                role="assistant",
                content=[{"type": "output_text", "text": ""}],
            ),
            InputItem(
                type="function_call",
                call_id="call_abc",
                name="exec_command",
                arguments='{"cmd": "ls"}',
            ),
            InputItem(
                type="function_call_output",
                call_id="call_abc",
                output="file1.txt\nfile2.txt",
            ),
        ]
        messages = convert_responses_input_to_messages(items)
        # user, assistant (content + tool_calls merged), tool
        assert len(messages) == 3
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == ""
        assert len(messages[1]["tool_calls"]) == 1
        assert messages[1]["tool_calls"][0]["function"]["name"] == "exec_command"
        assert messages[2]["role"] == "tool"

    def test_assistant_with_content_merged_with_function_call(self):
        """Assistant message with non-empty content merges with tool_calls."""
        items = [
            InputItem(type="message", role="user", content="Check weather"),
            InputItem(
                type="message",
                role="assistant",
                content=[{"type": "output_text", "text": "Let me check."}],
            ),
            InputItem(
                type="function_call",
                call_id="call_1",
                name="get_weather",
                arguments='{"city": "Seoul"}',
            ),
            InputItem(
                type="function_call_output",
                call_id="call_1",
                output='{"temp": 20}',
            ),
        ]
        messages = convert_responses_input_to_messages(items)
        assert len(messages) == 3
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Let me check."
        assert len(messages[1]["tool_calls"]) == 1

    def test_multi_round_tool_calls_no_duplicate_assistant(self):
        """Multiple rounds of tool calls should not create duplicate assistant
        messages (the pattern that causes model EOS after many rounds)."""
        items = [
            InputItem(type="message", role="user", content="Explore the code"),
        ]
        # Simulate 3 rounds of tool calls (Codex pattern)
        for i in range(3):
            items.extend(
                [
                    InputItem(
                        type="message",
                        role="assistant",
                        content=[{"type": "output_text", "text": ""}],
                    ),
                    InputItem(
                        type="function_call",
                        call_id=f"call_{i}",
                        name="exec_command",
                        arguments=f'{{"cmd": "cmd_{i}"}}',
                    ),
                    InputItem(
                        type="function_call_output",
                        call_id=f"call_{i}",
                        output=f"result_{i}",
                    ),
                ]
            )
        messages = convert_responses_input_to_messages(items)
        # Count assistant messages — should be exactly 3 (one per round)
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 3
        # Each assistant message should have tool_calls
        for msg in assistant_msgs:
            assert "tool_calls" in msg
            assert len(msg["tool_calls"]) == 1

    def test_output_text_content_type(self):
        """output_text content type should be parsed correctly."""
        items = [
            InputItem(
                type="message",
                role="assistant",
                content=[{"type": "output_text", "text": "Hello world"}],
            ),
        ]
        messages = convert_responses_input_to_messages(items)
        assert messages[0]["content"] == "Hello world"

    def test_content_parts_array(self):
        items = [
            InputItem(
                type="message",
                role="user",
                content=[
                    {"type": "input_text", "text": "Hello"},
                    {"type": "input_text", "text": "World"},
                ],
            ),
        ]
        messages = convert_responses_input_to_messages(items)
        assert messages[0]["content"] == "Hello\nWorld"

    def test_input_image_preserved_as_content_list(self):
        """input_image parts should be preserved as list for VLM processing."""
        items = [
            InputItem(
                type="message",
                role="user",
                content=[
                    {"type": "input_text", "text": "What is in this image?"},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,abc123",
                        "detail": "high",
                    },
                ],
            ),
        ]
        messages = convert_responses_input_to_messages(items)
        content = messages[0]["content"]
        # Content should be a list (not flattened to string) when images present
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "What is in this image?"}
        assert content[1] == {
            "type": "input_image",
            "image_url": "data:image/png;base64,abc123",
            "detail": "high",
        }

    def test_input_image_url_field_fallback(self):
        """input_image with 'url' field instead of 'image_url' should also work."""
        items = [
            InputItem(
                type="message",
                role="user",
                content=[
                    {"type": "input_text", "text": "Describe"},
                    {"type": "input_image", "url": "https://example.com/img.png"},
                ],
            ),
        ]
        messages = convert_responses_input_to_messages(items)
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert content[1]["image_url"] == "https://example.com/img.png"
        assert content[1]["detail"] == "auto"  # default

    def test_text_only_content_parts_flattened(self):
        """Content with only text parts should still be flattened to string."""
        items = [
            InputItem(
                type="message",
                role="user",
                content=[
                    {"type": "input_text", "text": "Hello"},
                    {"type": "input_text", "text": "World"},
                ],
            ),
        ]
        messages = convert_responses_input_to_messages(items)
        assert isinstance(messages[0]["content"], str)
        assert messages[0]["content"] == "Hello\nWorld"

    def test_previous_messages_prepended(self):
        prev = [{"role": "assistant", "content": "Previous answer"}]
        messages = convert_responses_input_to_messages(
            "Follow up", previous_messages=prev
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[1]["role"] == "user"

    def test_instructions_with_previous(self):
        prev = [{"role": "assistant", "content": "Previous"}]
        messages = convert_responses_input_to_messages(
            "New question",
            instructions="System prompt",
            previous_messages=prev,
        )
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

    def test_previous_system_messages_merge_with_current_instructions(self):
        prev = [{"role": "system", "content": "Previous instruction"}]
        messages = convert_responses_input_to_messages(
            "New question",
            instructions="Current instruction",
            previous_messages=prev,
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "Previous instruction" in messages[0]["content"]
        assert "Current instruction" in messages[0]["content"]

    def test_current_function_call_does_not_merge_into_previous_assistant(self):
        prev = [{"role": "assistant", "content": "Previous answer"}]
        items = [
            InputItem(
                type="function_call",
                call_id="call_next",
                name="exec_command",
                arguments='{"cmd":"pwd"}',
            )
        ]
        messages = convert_responses_input_to_messages(items, previous_messages=prev)
        assert len(messages) == 2
        assert messages[0]["content"] == "Previous answer"
        assert messages[1]["tool_calls"][0]["id"] == "call_next"

    def test_reasoning_then_message_attaches_reasoning_content(self):
        items = [
            InputItem(
                type="reasoning",
                summary=[{"type": "summary_text", "text": "step-by-step"}],
            ),
            InputItem(type="message", role="assistant", content="Final answer"),
        ]
        messages = convert_responses_input_to_messages(items)
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "Final answer"
        assert messages[0]["reasoning_content"] == "step-by-step"

    def test_reasoning_then_function_call_attaches_reasoning_content(self):
        # Tool-call sequence: reasoning → function_call → function_call_output.
        # Reasoning must survive into the synthesized assistant tool_calls
        # message instead of being dropped on the floor.
        items = [
            InputItem(
                type="reasoning",
                summary=[{"type": "summary_text", "text": "need weather"}],
            ),
            InputItem(
                type="function_call",
                call_id="call_w",
                name="get_weather",
                arguments='{"city":"Paris"}',
            ),
            InputItem(type="function_call_output", call_id="call_w", output="sunny"),
        ]
        messages = convert_responses_input_to_messages(items)
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[0]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert messages[0]["reasoning_content"] == "need weather"
        assert messages[1]["role"] == "tool"
        assert messages[1]["content"] == "sunny"

    def test_reasoning_with_multi_part_summary_joined(self):
        items = [
            InputItem(
                type="reasoning",
                summary=[
                    {"type": "summary_text", "text": "line1"},
                    {"type": "summary_text", "text": "line2"},
                ],
            ),
            InputItem(type="message", role="assistant", content="ok"),
        ]
        messages = convert_responses_input_to_messages(items)
        assert messages[0]["reasoning_content"] == "line1\nline2"

    def test_reasoning_orphan_then_tool_calls_only(self):
        # reasoning followed by function_call but no function_call_output
        # in the same input — reasoning still lands on the tool_calls message
        # when the final flush runs.
        items = [
            InputItem(
                type="reasoning",
                summary=[{"type": "summary_text", "text": "thinking"}],
            ),
            InputItem(
                type="function_call",
                call_id="call_x",
                name="lookup",
                arguments="{}",
            ),
        ]
        messages = convert_responses_input_to_messages(items)
        assert len(messages) == 1
        assert messages[0]["tool_calls"][0]["function"]["name"] == "lookup"
        assert messages[0]["reasoning_content"] == "thinking"

    def test_reasoning_content_only_item_is_read(self):
        """Responses-dialect clients send reasoning as content[].reasoning_text."""
        items = [
            InputItem(
                type="reasoning",
                content=[{"type": "reasoning_text", "text": "raw cot"}],
            ),
            InputItem(type="message", role="assistant", content="answer"),
        ]
        messages = convert_responses_input_to_messages(items)
        assert messages[0]["reasoning_content"] == "raw cot"
        assert messages[0]["content"] == "answer"

    def test_reasoning_summary_wins_over_duplicate_content(self):
        """Both shapes carry the same upstream string; do not concatenate it."""
        items = [
            InputItem(
                type="reasoning",
                summary=[{"type": "summary_text", "text": "cot"}],
                content=[{"type": "reasoning_text", "text": "cot"}],
            ),
            InputItem(type="message", role="assistant", content="answer"),
        ]
        messages = convert_responses_input_to_messages(items)
        assert messages[0]["reasoning_content"] == "cot"

    @pytest.mark.parametrize(
        "item_type",
        [
            "computer_call",
            "web_search_call",
            "file_search_call",
            "mcp_call",
            "image_generation_call",
            "code_interpreter_call",
            "local_shell_call",
            "custom_tool_call",
            "custom_tool_call_output",
            "item_reference",
        ],
    )
    def test_unsupported_input_item_types_are_rejected(self, item_type):
        with pytest.raises(InvalidRequestError) as excinfo:
            convert_responses_input_to_messages([InputItem(type=item_type)])
        assert item_type in str(excinfo.value)
        assert excinfo.value.field == "input"

    def test_unknown_input_item_type_is_rejected(self):
        with pytest.raises(InvalidRequestError) as excinfo:
            convert_responses_input_to_messages([InputItem(type="mystery")])
        assert "mystery" in str(excinfo.value)

    def test_type_less_item_without_role_is_rejected(self):
        with pytest.raises(InvalidRequestError) as excinfo:
            convert_responses_input_to_messages([InputItem(content="hi")])
        assert excinfo.value.field == "input"

    def test_unknown_message_role_is_rejected(self):
        with pytest.raises(InvalidRequestError) as excinfo:
            convert_responses_input_to_messages(
                [InputItem(type="message", role="wizard", content="hi")]
            )
        assert "wizard" in str(excinfo.value)

    def test_unsupported_content_part_is_rejected(self):
        with pytest.raises(InvalidRequestError) as excinfo:
            convert_responses_input_to_messages(
                [
                    InputItem(
                        type="message",
                        role="user",
                        content=[{"type": "input_file", "file_id": "file_1"}],
                    )
                ]
            )
        assert "input_file" in str(excinfo.value)
        assert excinfo.value.field == "input"

    def test_omitted_call_ids_still_pair(self):
        """function_call and its output generate the same fallback id."""
        items = [
            InputItem(type="function_call", name="lookup", arguments="{}"),
            InputItem(type="function_call_output", output="result"),
        ]
        messages = convert_responses_input_to_messages(items)
        call_id = messages[0]["tool_calls"][0]["id"]
        assert call_id.startswith("call_")
        assert messages[1]["tool_call_id"] == call_id

    def test_ensure_call_id_is_stable_and_nonempty(self):
        assert ensure_call_id("call_abc") == "call_abc"
        assert ensure_call_id(None).startswith("call_")
        assert ensure_call_id("").startswith("call_")
        assert ensure_call_id("  ").startswith("call_")


IMAGE_URI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="


class TestFunctionCallOutputMultimodal:
    """Images in function_call_output lists route to VLM, not the prompt (#2989)."""

    def _tool_round(self, outputs):
        items = [InputItem(type="message", role="user", content="take screenshots")]
        for i in range(len(outputs)):
            items.append(
                InputItem(
                    type="function_call",
                    call_id=f"call_{i}",
                    name="screenshot",
                    arguments="{}",
                )
            )
        for i, output in enumerate(outputs):
            items.append(
                InputItem(
                    type="function_call_output",
                    call_id=f"call_{i}",
                    output=output,
                )
            )
        return items

    def test_image_extracted_to_user_message(self):
        output = [
            {"type": "input_text", "text": "screenshot result"},
            {"type": "input_image", "detail": "auto", "image_url": IMAGE_URI},
        ]
        messages = convert_responses_input_to_messages(
            self._tool_round([output]), preserve_images=True
        )
        assert [m["role"] for m in messages] == ["user", "assistant", "tool", "user"]
        assert messages[2]["content"] == "screenshot result"
        assert messages[3]["content"] == [
            {"type": "input_image", "image_url": IMAGE_URI, "detail": "auto"}
        ]

    def test_parallel_outputs_keep_tool_messages_contiguous(self):
        output = [
            {"type": "input_text", "text": "shot"},
            {"type": "input_image", "detail": "auto", "image_url": IMAGE_URI},
        ]
        messages = convert_responses_input_to_messages(
            self._tool_round([output, output]), preserve_images=True
        )
        # Images flush once after the tool run, never between tool messages
        assert [m["role"] for m in messages] == [
            "user",
            "assistant",
            "tool",
            "tool",
            "user",
        ]
        assert len(messages[4]["content"]) == 2
        for msg in messages[2:4]:
            assert IMAGE_URI not in msg["content"]

    def test_images_flush_before_next_tool_round(self):
        items = self._tool_round(
            [[{"type": "input_image", "detail": "auto", "image_url": IMAGE_URI}]]
        )
        items.extend(
            [
                InputItem(
                    type="function_call",
                    call_id="call_next",
                    name="lookup",
                    arguments="{}",
                ),
                InputItem(
                    type="function_call_output",
                    call_id="call_next",
                    output="done",
                ),
            ]
        )
        messages = convert_responses_input_to_messages(items, preserve_images=True)
        assert [m["role"] for m in messages] == [
            "user",
            "assistant",
            "tool",
            "user",
            "assistant",
            "tool",
        ]
        assert messages[3]["content"][0]["type"] == "input_image"
        assert messages[5]["content"] == "done"

    def test_placeholder_without_preserve_images(self):
        output = [
            {"type": "input_text", "text": "screenshot result"},
            {"type": "input_image", "detail": "auto", "image_url": IMAGE_URI},
        ]
        messages = convert_responses_input_to_messages(self._tool_round([output]))
        assert [m["role"] for m in messages] == ["user", "assistant", "tool"]
        assert messages[2]["content"] == "screenshot result\n(see attached image)"
        assert IMAGE_URI not in json.dumps(messages)

    def test_plain_json_list_falls_back_to_dumps(self):
        messages = convert_responses_input_to_messages(
            self._tool_round([[1, 2, 3]]), preserve_images=True
        )
        assert messages[2]["content"] == "[1, 2, 3]"
        messages = convert_responses_input_to_messages(
            self._tool_round([["a", "b"]]), preserve_images=True
        )
        assert messages[2]["content"] == '["a", "b"]'

    def test_text_only_typed_list_extracted(self):
        messages = convert_responses_input_to_messages(
            self._tool_round([[{"type": "input_text", "text": "hello"}]]),
            preserve_images=True,
        )
        assert messages[2]["content"] == "hello"


# =============================================================================
# Tool Conversion Tests
# =============================================================================


class TestConvertResponsesTools:
    def test_none_tools(self):
        assert convert_responses_tools(None) is None

    def test_empty_tools(self):
        assert convert_responses_tools([]) is None

    def test_single_tool(self):
        tools = [
            ResponsesTool(
                type="function",
                name="get_weather",
                description="Get weather info",
                parameters={
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            )
        ]
        result = convert_responses_tools(tools)
        assert result is not None
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get weather info"
        assert "parameters" in result[0]["function"]

    def test_strict_field(self):
        tools = [ResponsesTool(name="fn", strict=True)]
        result = convert_responses_tools(tools)
        assert result[0]["function"]["strict"] is True

    def test_minimal_tool(self):
        tools = [ResponsesTool(name="simple_fn")]
        result = convert_responses_tools(tools)
        assert result[0]["function"]["name"] == "simple_fn"
        assert "description" not in result[0]["function"]
        assert "parameters" not in result[0]["function"]

    @pytest.mark.parametrize(
        "tool_type,label",
        [
            ("local_shell", "local_shell (hosted)"),
            ("mcp", "mcp (hosted)"),
            ("web_search", "web_search (hosted)"),
            ("web_search_preview", "web_search_preview (hosted)"),
            ("file_search", "file_search (hosted)"),
            ("computer_use_preview", "computer_use_preview (hosted)"),
            ("code_interpreter", "code_interpreter (hosted)"),
            ("image_generation", "image_generation (hosted)"),
            ("custom", "custom (hosted)"),
            ("not_a_real_tool_type", "not_a_real_tool_type (unknown type)"),
        ],
    )
    def test_unsupported_tool_types_are_accepted_but_not_exposed(
        self, tool_type, label
    ):
        """A declaration the model cannot use is accepted and named, not 400.

        Declaring a capability is separate from using it: the model never sees
        the tool, so an unused declaration cannot change the response. Codex
        declares ``web_search`` unconditionally, so rejecting it made the
        endpoint unusable. The degradation is reported instead of silent.
        """
        unexposed = []
        result = convert_responses_tools(
            [ResponsesTool(type=tool_type)], unexposed=unexposed
        )
        assert result is None
        assert unexposed == [label]

    def test_unexposed_label_cannot_inject_a_header(self):
        """A client-supplied type is constrained to header-safe characters."""
        unexposed = []
        convert_responses_tools(
            [ResponsesTool(type='web"search\r\nX-Evil: 1')], unexposed=unexposed
        )
        assert unexposed == ["web_search__X-Evil:_1 (unknown type)"]
        for label in unexposed:
            assert '"' not in label and "\r" not in label and "\n" not in label

    def test_supported_and_unsupported_tools_mixed_keeps_supported(self):
        """A supported tool is still exposed alongside an unexposed declaration."""
        unexposed = []
        tools = [
            ResponsesTool(type="function", name="fn_a"),
            ResponsesTool(type="web_search"),
        ]
        result = convert_responses_tools(tools, unexposed=unexposed)
        assert [t["function"]["name"] for t in result] == ["fn_a"]
        assert unexposed == ["web_search (hosted)"]

    def test_function_tool_without_a_name_is_rejected(self):
        with pytest.raises(InvalidRequestError) as excinfo:
            convert_responses_tools([ResponsesTool(type="function")])
        assert "name" in str(excinfo.value)
        assert excinfo.value.field == "tools"

    def test_tool_search_is_a_noop(self):
        """tool_search only lazy-loads namespace members, which oMLX exposes
        eagerly, so accepting it does not change which tools the model sees."""
        tools = [
            ResponsesTool(type="tool_search"),
            ResponsesTool(type="function", name="fn_a"),
        ]
        result = convert_responses_tools(tools)
        assert [t["function"]["name"] for t in result] == ["fn_a"]

    def test_namespace_tools_are_expanded(self):
        """Namespace groups hold client-executed function tools (#3371)."""
        registry = ToolBindingRegistry()
        tools = [
            ResponsesTool(
                type="namespace",
                name="mcp__demo__",
                description="Demo MCP server",
                tools=[
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get the current weather for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                    {"type": "function", "name": "get_time"},
                ],
            )
        ]
        result = convert_responses_tools(tools, registry)
        assert result is not None
        # Wire names join the namespace like Codex's join_tool_name().
        assert [t["function"]["name"] for t in result] == [
            "mcp__demo__get_weather",
            "mcp__demo__get_time",
        ]
        assert result[0]["type"] == "function"
        assert result[0]["function"]["description"] == (
            "Get the current weather for a city."
        )
        assert result[0]["function"]["parameters"]["required"] == ["city"]
        assert registry.aliases() == {
            "mcp__demo__get_weather": ("mcp__demo__", "get_weather"),
            "mcp__demo__get_time": ("mcp__demo__", "get_time"),
        }

    def test_namespace_wire_name_survives_collision(self):
        """A flat tool already holding the joined name must not be shadowed."""
        registry = ToolBindingRegistry()
        tools = [
            ResponsesTool(type="function", name="mcp__demo__get_weather"),
            ResponsesTool(
                type="namespace",
                name="mcp__demo__",
                tools=[{"type": "function", "name": "get_weather"}],
            ),
        ]
        result = convert_responses_tools(tools, registry)
        assert [t["function"]["name"] for t in result] == [
            "mcp__demo__get_weather",
            "mcp__demo__get_weather_2",
        ]
        assert registry.aliases() == {
            "mcp__demo__get_weather_2": ("mcp__demo__", "get_weather")
        }

    def test_namespace_member_without_a_name_is_rejected(self):
        tools = [
            ResponsesTool(
                type="namespace",
                name="mcp__demo__",
                tools=[{"type": "function"}],
            )
        ]
        with pytest.raises(InvalidRequestError) as excinfo:
            convert_responses_tools(tools)
        assert "mcp__demo__" in str(excinfo.value)
        assert excinfo.value.field == "tools"

    def test_namespace_with_hosted_member_is_dropped_and_reported(self):
        """A hosted namespace member follows the same declared-vs-used rule."""
        unexposed = []
        tools = [
            ResponsesTool(
                type="namespace",
                name="mcp__demo__",
                tools=[{"type": "web_search"}],
            )
        ]
        result = convert_responses_tools(tools, unexposed=unexposed)
        assert result is None
        assert unexposed == ["web_search (hosted) in namespace mcp__demo__"]

    def test_nested_namespace_is_dropped_and_reported(self):
        """A nested namespace is never exposed, so it too is accepted."""
        unexposed = []
        tools = [
            ResponsesTool(
                type="namespace",
                name="mcp__demo__",
                tools=[{"type": "namespace", "name": "inner__"}],
            )
        ]
        result = convert_responses_tools(tools, unexposed=unexposed)
        assert result is None
        assert unexposed == ["namespace (nested) in namespace mcp__demo__"]

    def test_split_namespace_tool_name(self):
        registry = ToolBindingRegistry()
        convert_responses_tools(
            [
                ResponsesTool(
                    type="namespace",
                    name="mcp__demo__",
                    tools=[{"type": "function", "name": "get_weather"}],
                )
            ],
            registry,
        )
        assert split_namespace_tool_name("mcp__demo__get_weather", registry) == (
            "mcp__demo__",
            "get_weather",
        )
        # Flat calls, and calls the model invented, pass through unchanged.
        assert split_namespace_tool_name("get_weather", registry) == (
            None,
            "get_weather",
        )
        assert split_namespace_tool_name("get_weather") == (None, "get_weather")

    def test_registry_rebinds_history_to_current_wire_name(self):
        """A preserved namespace call follows a collision rename in this turn."""
        registry = ToolBindingRegistry()
        convert_responses_tools(
            [
                ResponsesTool(type="function", name="mcp__demo__get_weather"),
                ResponsesTool(
                    type="namespace",
                    name="mcp__demo__",
                    tools=[{"type": "function", "name": "get_weather"}],
                ),
            ],
            registry,
        )
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {},
                            "namespace": "mcp__demo__",
                        },
                    }
                ],
            }
        ]
        registry.apply_to_messages(messages)
        function = messages[0]["tool_calls"][0]["function"]
        assert function["name"] == "mcp__demo__get_weather_2"
        assert "namespace" not in function

    def test_namespace_serializes_only_when_set(self):
        namespaced = build_function_call_output_item(
            name="get_weather",
            arguments='{"city": "Paris"}',
            call_id="call_1",
            namespace="mcp__demo__",
        ).model_dump()
        assert namespaced["name"] == "get_weather"
        assert namespaced["namespace"] == "mcp__demo__"
        flat = build_function_call_output_item(
            name="get_weather", arguments="{}", call_id="call_2"
        ).model_dump()
        assert "namespace" not in flat
        # Other optional fields keep serializing as null, as before.
        assert flat["role"] is None and flat["summary"] is None

    def test_empty_namespace_contributes_nothing(self):
        tools = [
            ResponsesTool(type="namespace", name="empty__"),
            ResponsesTool(type="function", name="fn_a"),
        ]
        result = convert_responses_tools(tools)
        assert [t["function"]["name"] for t in result] == ["fn_a"]

    def test_namespace_member_not_a_tool_object_is_rejected(self):
        tools = [ResponsesTool(type="namespace", name="junk__", tools=["nope"])]
        with pytest.raises(InvalidRequestError) as excinfo:
            convert_responses_tools(tools)
        assert "junk__" in str(excinfo.value)


class TestValidateResponsesRequest:
    """Unimplemented capabilities are refused by field, never echoed back."""

    def _request(self, **kwargs):
        return ResponsesRequest(model="m", input="hi", **kwargs)

    def test_auto_and_none_are_accepted(self):
        validate_responses_request(self._request(tool_choice="auto"))
        validate_responses_request(self._request(tool_choice="none"))
        validate_responses_request(self._request())

    def test_required_tool_choice_is_rejected(self):
        with pytest.raises(InvalidRequestError) as excinfo:
            validate_responses_request(self._request(tool_choice="required"))
        assert excinfo.value.field == "tool_choice"

    def test_named_function_tool_choice_is_rejected(self):
        with pytest.raises(InvalidRequestError) as excinfo:
            validate_responses_request(
                self._request(
                    tool_choice={"type": "function", "name": "get_weather"}
                )
            )
        assert excinfo.value.field == "tool_choice"

    def test_unknown_tool_choice_is_rejected(self):
        with pytest.raises(InvalidRequestError) as excinfo:
            validate_responses_request(self._request(tool_choice="sometimes"))
        assert excinfo.value.field == "tool_choice"

    def test_truncation_disabled_is_accepted(self):
        validate_responses_request(self._request(truncation="disabled"))

    def test_truncation_auto_is_rejected(self):
        with pytest.raises(InvalidRequestError) as excinfo:
            validate_responses_request(self._request(truncation="auto"))
        assert excinfo.value.field == "truncation"

    def test_invalid_truncation_is_rejected(self):
        with pytest.raises(InvalidRequestError) as excinfo:
            validate_responses_request(self._request(truncation="maybe"))
        assert excinfo.value.field == "truncation"

    def test_parallel_tool_calls_true_is_accepted(self):
        validate_responses_request(self._request(parallel_tool_calls=True))

    def test_parallel_tool_calls_false_is_rejected(self):
        with pytest.raises(InvalidRequestError) as excinfo:
            validate_responses_request(
                self._request(parallel_tool_calls=False)
            )
        assert excinfo.value.field == "parallel_tool_calls"

    def test_background_is_rejected(self):
        with pytest.raises(InvalidRequestError) as excinfo:
            validate_responses_request(self._request(background=True))
        assert excinfo.value.field == "background"

    def test_conversation_is_rejected(self):
        with pytest.raises(InvalidRequestError) as excinfo:
            validate_responses_request(
                self._request(conversation={"id": "conv_1"})
            )
        assert excinfo.value.field == "conversation"

    def test_max_tool_calls_is_rejected(self):
        with pytest.raises(InvalidRequestError) as excinfo:
            validate_responses_request(self._request(max_tool_calls=3))
        assert excinfo.value.field == "max_tool_calls"

    def test_top_logprobs_is_rejected(self):
        with pytest.raises(InvalidRequestError) as excinfo:
            validate_responses_request(self._request(top_logprobs=5))
        assert excinfo.value.field == "top_logprobs"

    def test_encrypted_reasoning_include_is_a_documented_noop(self):
        validate_responses_request(
            self._request(include=["reasoning.encrypted_content"])
        )

    def test_other_include_values_are_rejected(self):
        with pytest.raises(InvalidRequestError) as excinfo:
            validate_responses_request(
                self._request(include=["file_search_call.results"])
            )
        assert excinfo.value.field == "include"

    def test_json_schema_format_requires_a_schema(self):
        with pytest.raises(InvalidRequestError) as excinfo:
            validate_responses_request(
                self._request(
                    text=TextConfig(format=TextFormatConfig(type="json_schema"))
                )
            )
        assert excinfo.value.field == "text.format"

    def test_unknown_text_format_is_rejected(self):
        with pytest.raises(InvalidRequestError) as excinfo:
            validate_responses_request(
                self._request(
                    text=TextConfig(format=TextFormatConfig(type="xml"))
                )
            )
        assert excinfo.value.field == "text.format"

    def test_known_text_formats_are_accepted(self):
        validate_responses_request(
            self._request(text=TextConfig(format=TextFormatConfig(type="text")))
        )
        validate_responses_request(
            self._request(
                text=TextConfig(format=TextFormatConfig(type="json_object"))
            )
        )
        validate_responses_request(
            self._request(
                text=TextConfig(
                    format=TextFormatConfig(
                        type="json_schema",
                        name="out",
                        schema={"type": "object"},
                    )
                )
            )
        )


# =============================================================================
# InputItem Validation Tests
# =============================================================================


class TestInputItemOutputSerialization:
    """InputItem should accept list/dict output; dict serializes, list survives."""

    def test_list_output_preserved(self):
        # Lists pass through so multimodal parts stay extractable (#2989)
        item = InputItem(
            type="function_call_output",
            call_id="call_123",
            output=[{"type": "input_image", "image_url": "data:image/jpeg;base64,abc"}],
        )
        assert isinstance(item.output, list)
        assert item.output[0]["type"] == "input_image"

    def test_dict_output_serialized_to_json(self):
        item = InputItem(
            type="function_call_output",
            call_id="call_456",
            output={"result": "success", "data": [1, 2, 3]},
        )
        assert isinstance(item.output, str)
        parsed = json.loads(item.output)
        assert parsed["result"] == "success"

    def test_string_output_unchanged(self):
        item = InputItem(
            type="function_call_output",
            call_id="call_789",
            output="plain text result",
        )
        assert item.output == "plain text result"

    def test_none_output_unchanged(self):
        item = InputItem(type="function_call_output", call_id="call_000")
        assert item.output is None


# =============================================================================
# Response Building Tests
# =============================================================================


class TestBuildOutputItems:
    def test_message_output_item(self):
        item = build_message_output_item("Hello!")
        assert item.type == "message"
        assert item.role == "assistant"
        assert item.content[0].text == "Hello!"
        assert item.content[0].type == "output_text"
        assert item.status == "completed"
        assert item.id.startswith("msg_")

    def test_function_call_output_item(self):
        item = build_function_call_output_item(
            name="get_weather",
            arguments='{"location": "Paris"}',
            call_id="call_abc",
        )
        assert item.type == "function_call"
        assert item.name == "get_weather"
        assert item.call_id == "call_abc"
        assert item.arguments == '{"location": "Paris"}'
        assert item.id.startswith("fc_")

    def test_response_usage(self):
        usage = build_response_usage(100, 50)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.input_tokens_details.cached_tokens == 0
        assert usage.output_tokens_details.reasoning_tokens == 0

    def test_response_usage_with_cached_tokens(self):
        usage = build_response_usage(100, 50, cached_tokens=30)
        assert usage.input_tokens == 100
        assert usage.input_tokens_details.cached_tokens == 30

    def test_response_usage_with_reasoning_tokens(self):
        usage = build_response_usage(100, 50, reasoning_tokens=20, cached_tokens=30)
        assert usage.output_tokens == 50
        assert usage.output_tokens_details.reasoning_tokens == 20
        assert usage.input_tokens_details.cached_tokens == 30

    def test_build_reasoning_output_item(self):
        item = build_reasoning_output_item("Step 1: foo. Step 2: bar.")
        assert item.type == "reasoning"
        assert item.status == "completed"
        assert item.id.startswith("rs_")
        assert len(item.summary) == 1
        assert item.summary[0].type == "summary_text"
        assert item.summary[0].text == "Step 1: foo. Step 2: bar."

    def test_build_reasoning_output_item_publishes_both_shapes(self):
        """Clients disagree on where the CoT lives, so it goes in both places.

        OpenAI's hosts publish ``summary``; Responses-dialect clients read
        ``content[]`` with a ``reasoning_text`` part and show nothing without it.
        """
        text = "Step 1: foo. Step 2: bar."
        item = build_reasoning_output_item(text)
        assert [part.text for part in item.content] == [text]
        assert item.content[0].type == "reasoning_text"

    def test_build_reasoning_output_item_empty_text(self):
        item = build_reasoning_output_item("")
        assert item.type == "reasoning"
        assert item.summary == []
        assert item.content == []

    def test_build_response_object_is_a_complete_envelope(self):
        """Both response paths serialize this, so it must carry every field.

        Nullable spec fields stay present as null rather than being dropped;
        a caller that serialized with exclude_none would otherwise emit a
        different key set from the other path.
        """
        env = build_response_object(
            ResponsesRequest(model="m", input="hi"),
            response_id="resp_test",
            created_at=1,
            output_items=[],
            usage=None,
            truncated=False,
            temperature=None,
            top_p=None,
        ).model_dump()
        for key in (
            "id",
            "object",
            "created_at",
            "model",
            "status",
            "output",
            "usage",
            "text",
            "truncation",
            "error",
            "incomplete_details",
            "instructions",
            "store",
            "parallel_tool_calls",
            "reasoning",
            "metadata",
            "previous_response_id",
        ):
            assert key in env, key
        assert env["status"] == "completed"

    def test_build_response_object_marks_truncation(self):
        env = build_response_object(
            ResponsesRequest(model="m", input="hi", max_output_tokens=8),
            response_id="resp_test",
            created_at=1,
            output_items=[],
            usage=None,
            truncated=True,
            temperature=None,
            top_p=None,
        )
        assert env.status == "incomplete"
        assert env.incomplete_details == {"reason": "max_output_tokens"}

    def test_build_response_object_echoes_truncation_mode(self):
        """The request's truncation mode must round-trip, not just be null."""
        default = build_response_object(
            ResponsesRequest(model="m", input="hi"),
            response_id="resp_test",
            created_at=1,
            output_items=[],
            usage=None,
            truncated=False,
            temperature=None,
            top_p=None,
        )
        assert default.truncation == "disabled"
        disabled = build_response_object(
            ResponsesRequest(model="m", input="hi", truncation="disabled"),
            response_id="resp_test",
            created_at=1,
            output_items=[],
            usage=None,
            truncated=False,
            temperature=None,
            top_p=None,
        )
        assert disabled.truncation == "disabled"

    def test_build_response_object_status_override_for_stream_open(self):
        env = build_response_object(
            ResponsesRequest(model="m", input="hi"),
            response_id="resp_test",
            created_at=1,
            output_items=[],
            usage=None,
            truncated=False,
            temperature=None,
            top_p=None,
            status="in_progress",
        )
        assert env.status == "in_progress"


class TestResponseObject:
    def test_defaults(self):
        resp = ResponseObject(model="test-model")
        assert resp.object == "response"
        assert resp.id.startswith("resp_")
        assert resp.status == "completed"
        assert resp.output == []

    def test_with_output(self):
        msg = build_message_output_item("Hi")
        resp = ResponseObject(
            model="test-model",
            output=[msg],
            usage=build_response_usage(10, 5),
        )
        assert len(resp.output) == 1
        assert resp.usage.total_tokens == 15

    def test_incomplete_details_on_truncation(self):
        # A max_output_tokens truncation must surface as status="incomplete"
        # with incomplete_details.reason, so clients can distinguish an
        # incomplete turn from a natural stop (the Responses API has no
        # finish_reason field).
        resp = ResponseObject(
            model="test-model",
            status="incomplete",
            incomplete_details={"reason": "max_output_tokens"},
        )
        assert resp.status == "incomplete"
        assert resp.incomplete_details == {"reason": "max_output_tokens"}
        dumped = resp.model_dump(exclude_none=True)
        assert dumped["incomplete_details"] == {"reason": "max_output_tokens"}

    def test_completed_response_omits_incomplete_details(self):
        resp = ResponseObject(model="test-model")
        dumped = resp.model_dump(exclude_none=True)
        assert "incomplete_details" not in dumped


# =============================================================================
# SSE Event Formatting Tests
# =============================================================================


class TestFormatSSEEvent:
    def test_dict_data(self):
        result = format_sse_event("response.created", {"id": "resp_123"})
        assert result.startswith("event: response.created\n")
        assert "data: " in result
        assert result.endswith("\n\n")
        data_line = result.split("\n")[1]
        parsed = json.loads(data_line.removeprefix("data: "))
        assert parsed["id"] == "resp_123"

    def test_string_data(self):
        result = format_sse_event("response.output_text.delta", '{"delta": "hi"}')
        assert "event: response.output_text.delta\n" in result
        assert 'data: {"delta": "hi"}\n' in result

    def test_pydantic_model(self):
        usage = ResponseUsage(input_tokens=10, output_tokens=5)
        result = format_sse_event("test", usage)
        assert "event: test\n" in result
        parsed = json.loads(result.split("\n")[1].removeprefix("data: "))
        assert parsed["input_tokens"] == 10


# =============================================================================
# Response Store Tests
# =============================================================================


class TestResponseStore:
    def test_put_and_get(self):
        store = ResponseStore(max_size=10)
        store.put("resp_1", {"id": "resp_1", "output": []})
        assert store.get("resp_1") is not None
        assert store.get("resp_1")["id"] == "resp_1"

    def test_get_nonexistent(self):
        store = ResponseStore()
        assert store.get("nonexistent") is None

    def test_delete(self):
        store = ResponseStore()
        store.put("resp_1", {"id": "resp_1"})
        assert store.delete("resp_1") is True
        assert store.get("resp_1") is None

    def test_delete_nonexistent(self):
        store = ResponseStore()
        assert store.delete("nonexistent") is False

    def test_lru_eviction(self):
        store = ResponseStore(max_size=3)
        store.put("resp_1", {"id": "resp_1"})
        store.put("resp_2", {"id": "resp_2"})
        store.put("resp_3", {"id": "resp_3"})
        # This should evict resp_1 (oldest)
        store.put("resp_4", {"id": "resp_4"})
        assert store.get("resp_1") is None
        assert store.get("resp_2") is not None
        assert store.get("resp_4") is not None
        assert len(store) == 3

    def test_lru_access_refreshes(self):
        store = ResponseStore(max_size=3)
        store.put("resp_1", {"id": "resp_1"})
        store.put("resp_2", {"id": "resp_2"})
        store.put("resp_3", {"id": "resp_3"})
        # Access resp_1 to refresh it
        store.get("resp_1")
        # Now resp_2 should be evicted (oldest after refresh)
        store.put("resp_4", {"id": "resp_4"})
        assert store.get("resp_1") is not None  # Refreshed, still here
        assert store.get("resp_2") is None  # Evicted
        assert store.get("resp_4") is not None

    def test_update_existing(self):
        store = ResponseStore(max_size=3)
        store.put("resp_1", {"id": "resp_1", "v": 1})
        store.put("resp_1", {"id": "resp_1", "v": 2})
        assert store.get("resp_1")["v"] == 2
        assert len(store) == 1

    def test_disk_persistence_round_trip(self, tmp_path):
        state_dir = tmp_path / "response-state"
        store = ResponseStore(max_size=10, state_dir=state_dir)
        public = {"id": "resp_1", "created_at": 1, "output": []}
        record = build_response_store_record(
            public, [{"role": "user", "content": "Hi"}], []
        )
        store.put("resp_1", record)

        reloaded = ResponseStore(max_size=10, state_dir=state_dir)
        assert reloaded.get("resp_1")["id"] == "resp_1"
        assert reloaded.get_record("resp_1")["input_messages"][0]["content"] == "Hi"

    def test_resolve_chain_messages(self, tmp_path):
        state_dir = tmp_path / "response-state"
        store = ResponseStore(max_size=10, state_dir=state_dir)
        store.put(
            "resp_1",
            build_response_store_record(
                {"id": "resp_1", "created_at": 1, "output": []},
                [{"role": "user", "content": "First"}],
                [{"role": "assistant", "content": "One"}],
            ),
        )
        store.put(
            "resp_2",
            build_response_store_record(
                {
                    "id": "resp_2",
                    "created_at": 2,
                    "previous_response_id": "resp_1",
                    "output": [],
                },
                [{"role": "tool", "tool_call_id": "call_1", "content": "result"}],
                [{"role": "assistant", "content": "Two"}],
            ),
        )

        messages = store.resolve_chain_messages("resp_2")
        assert [m["content"] for m in messages if "content" in m] == [
            "First",
            "One",
            "result",
            "Two",
        ]

    def test_missing_response_chain_raises_not_found(self):
        store = ResponseStore(max_size=10)
        with pytest.raises(ResponseStateNotFoundError):
            store.resolve_chain_messages("resp_missing")

    def test_missing_ancestor_raises_corrupt(self):
        store = ResponseStore(max_size=10)
        store.put(
            "resp_2",
            build_response_store_record(
                {
                    "id": "resp_2",
                    "created_at": 2,
                    "previous_response_id": "resp_missing",
                    "output": [],
                },
                [],
                [],
            ),
        )
        with pytest.raises(ResponseStateCorruptError):
            store.resolve_chain_messages("resp_2")


# =============================================================================
# Previous Response Conversion Tests
# =============================================================================


class TestConvertStoredResponse:
    def test_normalize_response_output_merges_assistant_tool_call_turn(self):
        output_items = [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Let me check."}],
            },
            {
                "type": "function_call",
                "call_id": "call_abc",
                "name": "get_weather",
                "arguments": '{"location":"Paris"}',
            },
        ]
        messages = normalize_response_output_to_messages(output_items)
        assert len(messages) == 1
        assert messages[0]["content"] == "Let me check."
        assert messages[0]["tool_calls"][0]["id"] == "call_abc"

    def test_normalize_reasoning_then_message(self):
        output_items = [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "thinking"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "answer"}],
            },
        ]
        messages = normalize_response_output_to_messages(output_items)
        assert len(messages) == 1
        assert messages[0]["content"] == "answer"
        assert messages[0]["reasoning_content"] == "thinking"

    def test_normalize_reasoning_then_function_call(self):
        # Tool-call output sequence — reasoning must attach to the
        # synthesized assistant tool_calls message.
        output_items = [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "deciding"}],
            },
            {
                "type": "function_call",
                "call_id": "call_xyz",
                "name": "lookup",
                "arguments": '{"q":"foo"}',
            },
        ]
        messages = normalize_response_output_to_messages(output_items)
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["tool_calls"][0]["function"]["name"] == "lookup"
        assert messages[0]["reasoning_content"] == "deciding"

    def test_normalize_reasoning_content_only_item(self):
        output_items = [
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "raw cot"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "answer"}],
            },
        ]
        messages = normalize_response_output_to_messages(output_items)
        assert messages[0]["reasoning_content"] == "raw cot"

    def test_normalize_unknown_output_item_raises_corrupt(self):
        """A stored item we cannot replay must not be dropped silently."""
        with pytest.raises(ResponseStateCorruptError) as excinfo:
            normalize_response_output_to_messages([{"type": "web_search_call"}])
        assert "web_search_call" in str(excinfo.value)

    def test_normalize_function_call_without_call_id_gets_one(self):
        messages = normalize_response_output_to_messages(
            [
                {
                    "type": "function_call",
                    "name": "lookup",
                    "arguments": "{}",
                }
            ]
        )
        assert messages[0]["tool_calls"][0]["id"].startswith("call_")


class TestPreviousResponseToolRoundTrip:
    """The reasoning → tool → result → final sequence survives the store."""

    def test_reasoning_tool_result_final_round_trip(self, tmp_path):
        store = ResponseStore(max_size=10, state_dir=tmp_path)
        output = [
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "need weather"}],
                "content": [{"type": "reasoning_text", "text": "need weather"}],
            },
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Let me check."}],
            },
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_abc",
                "name": "get_weather",
                "arguments": '{"city": "Paris"}',
            },
        ]
        public = {"id": "resp_1", "created_at": 1, "output": output}
        store.put(
            "resp_1",
            build_response_store_record(
                public,
                [{"role": "user", "content": "weather?"}],
                normalize_response_output_to_messages(output),
            ),
        )

        history = store.resolve_chain_messages("resp_1")
        messages = convert_responses_input_to_messages(
            [
                InputItem(
                    type="function_call_output",
                    call_id="call_abc",
                    output="sunny",
                )
            ],
            previous_messages=history,
        )

        assistant = [m for m in messages if m.get("tool_calls")]
        tool = [m for m in messages if m.get("role") == "tool"]
        assert assistant[0]["tool_calls"][0]["id"] == "call_abc"
        assert assistant[0]["reasoning_content"] == "need weather"
        assert tool[0]["tool_call_id"] == "call_abc"
        assert tool[0]["content"] == "sunny"

    def test_instructions_are_not_reinherited_from_chain(self, tmp_path):
        """A new turn's instructions stand alone; the prior turn's are gone."""
        store = ResponseStore(max_size=10, state_dir=tmp_path)
        store.put(
            "resp_1",
            build_response_store_record(
                {"id": "resp_1", "created_at": 1, "output": []},
                [{"role": "user", "content": "one"}],
                [{"role": "assistant", "content": "first"}],
            ),
        )
        history = store.resolve_chain_messages("resp_1")
        messages = convert_responses_input_to_messages(
            "two",
            instructions="Only this turn's instruction",
            previous_messages=history,
        )
        system = [m for m in messages if m["role"] == "system"]
        assert len(system) == 1
        assert system[0]["content"] == "Only this turn's instruction"

    def test_call_id_stays_stable_across_chain(self, tmp_path):
        """A tool call's id is the same in the stored response and next turn."""
        store = ResponseStore(max_size=10, state_dir=tmp_path)
        output = [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_stable",
                "name": "lookup",
                "arguments": "{}",
            }
        ]
        public = {"id": "resp_1", "created_at": 1, "output": output}
        store.put(
            "resp_1",
            build_response_store_record(
                public,
                [],
                normalize_response_output_to_messages(output),
            ),
        )
        assert store.get("resp_1")["output"][0]["call_id"] == "call_stable"
        history = store.resolve_chain_messages("resp_1")
        replayed = [
            call
            for message in history
            for call in message.get("tool_calls", [])
        ]
        assert replayed[0]["id"] == "call_stable"


# =============================================================================
# Request Model Tests
# =============================================================================


class TestResponsesRequest:
    def test_minimal_request(self):
        req = ResponsesRequest(model="test-model")
        assert req.model == "test-model"
        assert req.input is None
        assert req.stream is False
        assert req.store is None

    def test_string_input(self):
        req = ResponsesRequest(model="test", input="Hello")
        assert req.input == "Hello"

    def test_item_input(self):
        req = ResponsesRequest(
            model="test",
            input=[InputItem(type="message", role="user", content="Hi")],
        )
        assert len(req.input) == 1
        assert req.input[0].type == "message"

    def test_with_tools(self):
        req = ResponsesRequest(
            model="test",
            input="Hi",
            tools=[ResponsesTool(name="fn_a", description="Does A")],
        )
        assert len(req.tools) == 1
        assert req.tools[0].name == "fn_a"

    def test_text_format(self):
        req = ResponsesRequest(
            model="test",
            input="Hi",
            text=TextConfig(format=TextFormatConfig(type="json_object")),
        )
        assert req.text.format.type == "json_object"

    def test_codex_fields(self):
        """Codex CLI sends include, service_tier, prompt_cache_key, etc."""
        req = ResponsesRequest(
            model="test",
            input=[InputItem(role="user", content="Hi")],
            stream=True,
            store=False,
            include=["reasoning.encrypted_content"],
            service_tier="auto",
            prompt_cache_key="test-key",
            reasoning={"effort": "high", "summary": "auto"},
            parallel_tool_calls=True,
        )
        assert req.include == ["reasoning.encrypted_content"]
        assert req.service_tier == "auto"
        assert req.prompt_cache_key == "test-key"
        assert req.reasoning["effort"] == "high"

    def test_chat_template_kwargs(self):
        req = ResponsesRequest(
            model="test",
            input="Hi",
            chat_template_kwargs={"enable_thinking": False},
        )
        assert req.chat_template_kwargs == {"enable_thinking": False}

    def test_extra_fields_allowed(self):
        """Unknown fields should not cause validation errors."""
        req = ResponsesRequest(
            model="test",
            input="Hi",
            some_future_field="value",
            another_field=42,
        )
        assert req.model == "test"

    def test_input_item_extra_fields(self):
        """InputItem should accept unknown fields for forward compatibility."""
        item = InputItem(
            type="message",
            role="user",
            content="Hi",
            phase="final_answer",
        )
        assert item.role == "user"

    def test_request_from_codex_like_json(self):
        """Parse a request body similar to what Codex CLI sends."""
        raw = {
            "model": "Qwen3.5-122B-A10B-4bit",
            "instructions": "You are a helpful assistant.",
            "input": [
                {"role": "user", "content": "Hello!"},
            ],
            "tools": [
                {"type": "local_shell"},
                {
                    "type": "function",
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                    "strict": True,
                },
            ],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "reasoning": {"effort": "xhigh", "summary": "auto"},
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
        }
        req = ResponsesRequest(**raw)
        assert req.model == "Qwen3.5-122B-A10B-4bit"
        assert req.stream is True
        assert len(req.input) == 1
        assert req.input[0].role == "user"
        assert req.input[0].type is None  # EasyInputMessage
        assert len(req.tools) == 2
        assert req.tools[0].type == "local_shell"
        assert req.tools[1].name == "read_file"
        # local_shell has no local executor: its declaration is accepted but
        # never exposed, and is named so the caller can be warned.
        unexposed = []
        result = convert_responses_tools(req.tools, unexposed=unexposed)
        assert [t["function"]["name"] for t in result] == ["read_file"]
        assert unexposed == ["local_shell (hosted)"]
