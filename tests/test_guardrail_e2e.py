# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests: full request → mocked model → verify response extension.

This file covers three layers:

1. ``TestE2EValidation`` — full-pipeline E2E tests that require a loaded
   model engine.  Marked ``slow`` so they are skipped in normal CI runs.

2. ``TestRescueParserIntegration`` — verify rescue parsers integrate with
   the existing parse chain (no model needed).

3. ``TestBackwardCompatibility`` — verify all behaviour is identical when
   guardrails are disabled.

4. ``TestValidationPayloadInjection`` — unit tests for the response
   injection pattern used at all 6 call sites in ``server.py``
   (non-streaming + streaming × 3 API formats).  These verify the
   ``x_omlx_validation`` field appears in responses when validation is
   enabled and is absent when disabled.
"""
import json
from unittest.mock import MagicMock

import pytest

from omlx.api.guardrail_wiring import guardrail_validation_payload
from omlx.api.guardrails.types import CheckResult, Nudge, ValidationResult
from omlx.api.openai_models import FunctionCall, ToolCall
from omlx.api.tool_calling import (
    ToolCallExtraction,
    _parse_mistral_bracket_tool_calls,
    _parse_rehearsal_tool_calls,
    _serialize_tool_call_arguments,
    extract_and_validate_tool_calls,
    extract_tool_calls_with_thinking,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


def _make_validation_result(passed=False, kind="unknown_tool"):
    """Build a ValidationResult with a single check for testing."""
    nudge = Nudge(role="tool", content="bad tool", kind=kind) if not passed else None
    return ValidationResult(
        checks=[
            CheckResult(
                check="unknown_tool",
                passed=passed,
                detail="ok" if passed else "tool not found",
            )
        ],
        nudge=nudge,
        passed=passed,
    )


def _make_extraction(validation_result=None):
    """Build a ToolCallExtraction with (optional) validation result."""
    return ToolCallExtraction(
        cleaned_text="",
        tool_calls=None,
        cleaned_thinking="",
        validation_result=validation_result,
    )


# ---------------------------------------------------------------------------
# Layer 1: Full E2E (skipped — require loaded model)
# ---------------------------------------------------------------------------


class TestE2EValidation:
    """Full pipeline tests requiring a loaded model engine.

    Marked 'slow' so they don't run in CI without explicit opt-in.
    """

    @pytest.mark.slow
    def test_unknown_tool_e2e(self):
        """Enable validation, mock model to emit unknown tool, verify response."""
        pytest.skip("Requires loaded test model engine")

    @pytest.mark.slow
    def test_malformed_args_e2e(self):
        """Mock model to emit non-dict args, verify malformed_args flag."""
        pytest.skip("Requires loaded test model engine")

    @pytest.mark.slow
    def test_disabled_by_default_e2e(self):
        """With default settings, response has no x_omlx_validation field."""
        pytest.skip("Requires loaded test model engine")


# ---------------------------------------------------------------------------
# Layer 2: Rescue parser integration
# ---------------------------------------------------------------------------


class TestRescueParserIntegration:
    """Verify rescue parsers integrate with the existing chain."""

    def test_rehearsal_parsed_when_no_native_match(self):
        """When native parsers fail, rehearsal parser should extract calls."""
        text = 'search[ARGS]{"query": "hello"}'
        result = _parse_rehearsal_tool_calls(text)
        assert result is not None
        assert len(result) == 1
        assert result[0].function.name == "search"

    def test_mistral_parsed_with_nested_json(self):
        """Mistral parser handles nested JSON objects."""
        text = '[TOOL_CALLS]search{"config": {"depth": {"nested": true}}}'
        result = _parse_mistral_bracket_tool_calls(text)
        assert result is not None
        assert len(result) == 1
        assert "config" in result[0].function.arguments


# ---------------------------------------------------------------------------
# Layer 3: Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Verify all behavior is identical when guardrails are disabled."""

    def test_wrapper_passthrough_when_disabled(self):
        """extract_and_validate_tool_calls with validation_enabled=False
        should return identical results to extract_tool_calls_with_thinking."""
        tokenizer = MagicMock()
        tokenizer.has_tool_calling = False
        tokenizer.tool_call_start = None
        tokenizer.tool_call_end = None
        tokenizer.tool_parser = None

        thinking = ""
        regular = '<tool_call>{"name": "search", "arguments": {"q": "test"}}</tool_call>'
        tools = [{"type": "function", "function": {"name": "search"}}]

        baseline = extract_tool_calls_with_thinking(thinking, regular, tokenizer, tools)
        wrapper = extract_and_validate_tool_calls(
            thinking, regular, tokenizer, tools=tools, validation_enabled=False
        )

        assert baseline.cleaned_text == wrapper.cleaned_text
        assert len(baseline.tool_calls) == len(wrapper.tool_calls)
        for bt, wt in zip(baseline.tool_calls or [], wrapper.tool_calls or []):
            assert bt.function.name == wt.function.name
            assert bt.function.arguments == wt.function.arguments
        assert wrapper.validation_result is None

    def test_serialize_args_default_unchanged(self):
        """Default _serialize_tool_call_arguments behavior is unchanged."""
        assert _serialize_tool_call_arguments({"a": 1}) == '{"a": 1}'
        assert _serialize_tool_call_arguments("bad") == "{}"
        assert _serialize_tool_call_arguments(None) == "{}"


# ---------------------------------------------------------------------------
# Layer 4: Validation payload injection (Task 8.5 wiring tests)
# ---------------------------------------------------------------------------


class TestValidationPayloadInjection:
    """Unit tests for the ``x_omlx_validation`` injection pattern used at
    all 6 call sites in ``server.py``.

    These tests verify the wiring contract:
    - When ``include_validation_metadata=True`` and a validation_result
      exists, the payload is built and injectable into any JSON response.
    - When disabled, no payload is produced (backward compatible).
    - Streaming endpoints emit the payload as an SSE chunk before the
      terminal event.
    """

    # --- payload building -------------------------------------------------

    def test_payload_none_when_metadata_disabled(self):
        """No payload when include_validation_metadata=False."""
        extraction = _make_extraction(_make_validation_result())
        payload = guardrail_validation_payload(
            extraction, include_validation_metadata=False
        )
        assert payload is None

    def test_payload_none_when_no_validation_result(self):
        """No payload when extraction has no validation_result."""
        extraction = _make_extraction(None)
        payload = guardrail_validation_payload(
            extraction, include_validation_metadata=True
        )
        assert payload is None

    def test_payload_built_when_enabled_and_result_present(self):
        """Payload is built with correct structure."""
        vr = _make_validation_result(passed=False, kind="unknown_tool")
        extraction = _make_extraction(vr)
        payload = guardrail_validation_payload(
            extraction, include_validation_metadata=True
        )
        assert payload is not None
        assert "x_omlx_validation" in payload
        assert payload["x_omlx_validation"]["passed"] is False
        assert payload["x_omlx_validation"]["nudge"]["kind"] == "unknown_tool"

    # --- non-streaming injection (all 3 endpoints) ------------------------

    def test_non_streaming_injection_adds_field(self):
        """Simulate the non-streaming injection pattern used at all 3
        non-streaming endpoints (/v1/chat/completions, /v1/messages,
        /v1/responses).

        Pattern: json.loads → update(payload) → json.dumps
        """
        vr = _make_validation_result(passed=False, kind="unknown_tool")
        extraction = _make_extraction(vr)

        # Simulate a response body from any of the 3 non-streaming endpoints
        mock_response = json.dumps(
            {"id": "resp_123", "object": "chat.completion", "choices": []}
        )

        payload = guardrail_validation_payload(
            extraction, include_validation_metadata=True
        )
        assert payload is not None

        # This is the exact pattern from server.py non-streaming endpoints
        result = json.loads(mock_response)
        result.update(payload)
        injected = json.dumps(result, ensure_ascii=False)

        parsed = json.loads(injected)
        assert "x_omlx_validation" in parsed
        assert parsed["x_omlx_validation"]["passed"] is False
        # Original fields preserved
        assert parsed["id"] == "resp_123"
        assert parsed["object"] == "chat.completion"

    def test_non_streaming_skips_when_disabled(self):
        """When include_validation_metadata=False, response is unchanged."""
        extraction = _make_extraction(_make_validation_result())

        mock_response = json.dumps({"id": "resp_123", "choices": []})
        payload = guardrail_validation_payload(
            extraction, include_validation_metadata=False
        )
        # Pattern: only inject if payload is not None
        if payload:
            result = json.loads(mock_response)
            result.update(payload)
            mock_response = json.dumps(result, ensure_ascii=False)

        parsed = json.loads(mock_response)
        assert "x_omlx_validation" not in parsed

    # --- streaming injection (all 3 endpoints) ----------------------------

    def test_streaming_openai_sse_format(self):
        """Verify the SSE chunk format for OpenAI streaming (/v1/chat/completions).

        Pattern: ``data: {payload_json}\\n\\n`` emitted before ``data: [DONE]``.
        """
        vr = _make_validation_result(passed=False, kind="unknown_tool")
        extraction = _make_extraction(vr)

        payload = guardrail_validation_payload(
            extraction, include_validation_metadata=True
        )
        assert payload is not None

        # SSE chunk format for OpenAI streaming
        sse_chunk = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        # Verify it parses as valid SSE data
        assert sse_chunk.startswith("data: ")
        assert sse_chunk.endswith("\n\n")
        json_part = sse_chunk[len("data: "):].rstrip("\n\n")
        parsed = json.loads(json_part)
        assert "x_omlx_validation" in parsed

    def test_streaming_anthropic_sse_format(self):
        """Verify the SSE event format for Anthropic streaming (/v1/messages).

        Pattern: ``event: x_omlx_validation\\ndata: {json}\\n\\n`` emitted
        before ``message_stop``.
        """
        from omlx.api.anthropic_utils import format_sse_event

        vr = _make_validation_result(passed=False, kind="unknown_tool")
        extraction = _make_extraction(vr)

        payload = guardrail_validation_payload(
            extraction, include_validation_metadata=True
        )
        assert payload is not None

        # SSE event format for Anthropic streaming
        sse_event = format_sse_event("x_omlx_validation", payload)

        assert "event: x_omlx_validation" in sse_event
        assert "data: " in sse_event
        # Extract and verify the data payload
        data_line = [
            line for line in sse_event.strip().split("\n") if line.startswith("data: ")
        ][0]
        parsed = json.loads(data_line[len("data: "):])
        assert "x_omlx_validation" in parsed

    def test_streaming_responses_sse_format(self):
        """Verify the SSE event format for Responses API streaming (/v1/responses).

        Pattern: ``event: response.x_omlx_validation\\ndata: {json}\\n\\n``
        emitted before ``response.completed``.
        """
        from omlx.api.responses_utils import format_sse_event

        vr = _make_validation_result(passed=False, kind="unknown_tool")
        extraction = _make_extraction(vr)

        payload = guardrail_validation_payload(
            extraction, include_validation_metadata=True
        )
        assert payload is not None

        # SSE event format for Responses API streaming
        sse_event = format_sse_event("response.x_omlx_validation", payload)

        assert "event: response.x_omlx_validation" in sse_event
        assert "data: " in sse_event
        data_line = [
            line for line in sse_event.strip().split("\n") if line.startswith("data: ")
        ][0]
        parsed = json.loads(data_line[len("data: "):])
        assert "x_omlx_validation" in parsed

    def test_streaming_skips_when_disabled(self):
        """No SSE chunk emitted when include_validation_metadata=False."""
        extraction = _make_extraction(_make_validation_result())
        payload = guardrail_validation_payload(
            extraction, include_validation_metadata=False
        )
        # Streaming endpoints only emit when payload is not None
        assert payload is None

    # --- full integration: extraction → validation → payload --------------

    def test_extraction_to_payload_unknown_tool(self):
        """Full chain: extract_and_validate → payload for unknown tool."""
        tokenizer = MagicMock()
        tokenizer.has_tool_calling = False
        tokenizer.tool_call_start = None
        tokenizer.tool_call_end = None
        tokenizer.tool_parser = None

        regular = '<tool_call>{"name": "bad_tool", "arguments": {}}</tool_call>'
        tools = [SEARCH_TOOL]

        extraction = extract_and_validate_tool_calls(
            "", regular, tokenizer, tools=tools, validation_enabled=True
        )

        assert extraction.validation_result is not None
        assert extraction.validation_result.passed is False

        payload = guardrail_validation_payload(
            extraction, include_validation_metadata=True
        )
        assert payload is not None
        assert payload["x_omlx_validation"]["passed"] is False
        assert payload["x_omlx_validation"]["nudge"]["kind"] == "unknown_tool"

    def test_extraction_to_payload_valid_call(self):
        """Full chain: extract_and_validate → payload for valid call."""
        tokenizer = MagicMock()
        tokenizer.has_tool_calling = False
        tokenizer.tool_call_start = None
        tokenizer.tool_call_end = None
        tokenizer.tool_parser = None

        regular = '<tool_call>{"name": "search", "arguments": {"query": "hello"}}</tool_call>'
        tools = [SEARCH_TOOL]

        extraction = extract_and_validate_tool_calls(
            "", regular, tokenizer, tools=tools, validation_enabled=True
        )

        assert extraction.validation_result is not None
        assert extraction.validation_result.passed is True

        payload = guardrail_validation_payload(
            extraction, include_validation_metadata=True
        )
        assert payload is not None
        assert payload["x_omlx_validation"]["passed"] is True
        assert "nudge" not in payload["x_omlx_validation"]
