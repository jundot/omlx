# SPDX-License-Identifier: Apache-2.0
"""Static guards for OpenAI chat streaming thinking separation."""

import ast
from pathlib import Path


def _server_stream_node(name: str):
    source = (Path(__file__).resolve().parents[1] / "omlx" / "server.py").read_text()
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return node
    raise AssertionError(f"{name} not found in server.py")


def test_stream_chat_completion_starts_parser_when_prompt_opens_thinking():
    """Chat streaming must not leak prompt-opened thinking as content."""
    node = _server_stream_node("stream_chat_completion")
    called = {
        call.func.id
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    assert "prompt_opens_thinking" in called, (
        "stream_chat_completion must detect when the rendered chat prompt "
        "already opens the thinking block; otherwise initial reasoning deltas "
        "are emitted as public content."
    )

    thinking_parser_calls = [
        call
        for call in ast.walk(node)
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "ThinkingParser"
        )
    ]
    assert any(
        keyword.arg == "start_in_thinking"
        for call in thinking_parser_calls
        for keyword in call.keywords
    ), (
        "stream_chat_completion must pass start_in_thinking into "
        "ThinkingParser so prompt-opened reasoning streams as "
        "reasoning_content, not content."
    )


def test_stream_responses_api_starts_parser_when_prompt_opens_thinking():
    """Responses streaming must not leak prompt-opened thinking as output_text."""
    node = _server_stream_node("stream_responses_api")
    called = {
        call.func.id
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    assert "prompt_opens_thinking" in called, (
        "stream_responses_api must detect when the rendered chat prompt "
        "already opens the thinking block; otherwise initial reasoning deltas "
        "are emitted as output_text instead of a reasoning item."
    )

    thinking_parser_calls = [
        call
        for call in ast.walk(node)
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "ThinkingParser"
        )
    ]
    assert any(
        keyword.arg == "start_in_thinking"
        for call in thinking_parser_calls
        for keyword in call.keywords
    ), (
        "stream_responses_api must pass start_in_thinking into "
        "ThinkingParser so prompt-opened reasoning streams as "
        "reasoning summary deltas, not output_text."
    )


def _called_names(node) -> set[str]:
    return {
        call.func.id
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }


def test_chat_prompt_detector_uses_scheduler_prompt_ids():
    """The shared detector must mirror the scheduler's token-level decision."""
    node = _server_stream_node("_chat_prompt_opens_thinking")
    assert "_render_chat_prompt_for_thinking_detection" in _called_names(node)
    assert any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "prompt_opens_thinking"
        and any(keyword.arg == "prompt_token_ids" for keyword in call.keywords)
        for call in ast.walk(node)
    ), "_chat_prompt_opens_thinking must pass the rendered prompt ids through."


def test_nonstream_builders_split_prompt_opened_thinking():
    """Complete replies must classify prompt-opened reasoning like streams,
    whatever the finish reason, so the start flag is the detector itself."""
    for name in (
        "_build_chat_completion",
        "_build_anthropic_message",
        "_build_responses_api",
    ):
        node = _server_stream_node(name)
        split_calls = [
            call
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "extract_thinking"
        ]
        assert split_calls, name
        for call in split_calls:
            flags = [kw.value for kw in call.keywords if kw.arg == "start_in_thinking"]
            assert len(flags) == 1, f"{name} must pass start_in_thinking"
            assert (
                isinstance(flags[0], ast.Call)
                and isinstance(flags[0].func, ast.Name)
                and flags[0].func.id == "_chat_prompt_opens_thinking"
            ), f"{name} must not gate start_in_thinking on the finish reason"


def test_responses_stream_start_state_follows_rendered_prompt_only():
    """The Responses stream must not start in thinking on a template flag."""
    node = _server_stream_node("stream_responses_api")
    assert "native_reasoning" not in {
        arg.arg for arg in node.args.args + node.args.kwonlyargs
    }, "stream_responses_api must not take a native_reasoning start flag"
    assert not any(
        isinstance(name, ast.Name) and name.id == "native_reasoning"
        for name in ast.walk(node)
    ), "stream_responses_api must decide its start state from the prompt alone"
