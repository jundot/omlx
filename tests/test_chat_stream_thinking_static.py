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
