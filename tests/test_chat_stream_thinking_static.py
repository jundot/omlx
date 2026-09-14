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


def test_stream_chat_completion_rejected_envelope_is_flushed_not_dropped():
    """An envelope rejected by the early-stream gate must be flushed as content.

    The ToolCallStreamFilter suppresses envelopes from content deltas, so a
    bare "continue" in the registered-name/parse-fail branch discards the
    turn's only payload and the client gets finish=stop with empty content and
    no tool_calls (jundot/omlx#3660). The reject branch must therefore yield
    the captured envelope as a visible content delta.
    """
    node = _server_stream_node("stream_chat_completion")
    rejects = [
        stmt
        for stmt in ast.walk(node)
        if isinstance(stmt, ast.If) and "registered_tool_names" in ast.dump(stmt.test)
    ]
    assert rejects, "early-stream registered-name gate not found"

    def _flushes_segment(stmt):
        dumped = ast.dump(stmt)
        return (
            any(isinstance(y, ast.Yield) for y in ast.walk(stmt))
            and "id='segment'" in dumped
            and "attr='text'" in dumped
        )

    for stmt in rejects:
        assert _flushes_segment(stmt), (
            "rejected qwen3_coder envelope must be yielded back as a content "
            "delta instead of being silently dropped"
        )
    flushed = any(_flushes_segment(stmt) for stmt in rejects)
    assert flushed, (
        "no reject-path yield flushes segment.text; a rejected tool-call "
        "envelope can still vanish from the stream (#3660)"
    )
