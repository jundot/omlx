"""Unit tests for strict args mode and the extract_and_validate wrapper."""
from omlx.api.tool_calling import (
    _serialize_tool_call_arguments,
    ToolCallExtraction,
)


class TestSerializeToolCallArguments:
    def test_dict_args_serialized(self):
        result = _serialize_tool_call_arguments({"key": "val"})
        assert result == '{"key": "val"}'

    def test_non_dict_coerced_to_empty_default(self):
        result = _serialize_tool_call_arguments("not a dict")
        assert result == "{}"

    def test_non_dict_strict_preserves(self):
        result = _serialize_tool_call_arguments("not a dict", strict=True)
        assert result == "not a dict"

    def test_dict_strict_still_serializes(self):
        result = _serialize_tool_call_arguments({"key": "val"}, strict=True)
        assert result == '{"key": "val"}'

    def test_json_string_dict_parsed_default(self):
        result = _serialize_tool_call_arguments('{"key": "val"}')
        assert result == '{"key": "val"}'

    def test_json_string_dict_strict_preserves(self):
        result = _serialize_tool_call_arguments('{"key": "val"}', strict=True)
        assert result == '{"key": "val"}'

    def test_none_default_coerced(self):
        result = _serialize_tool_call_arguments(None)
        assert result == "{}"

    def test_none_strict_preserves(self):
        result = _serialize_tool_call_arguments(None, strict=True)
        # Strict mode preserves — None becomes "null" or the raw repr
        assert result != "{}"


class TestToolCallExtractionValidationField:
    def test_has_validation_result_field(self):
        ext = ToolCallExtraction(
            cleaned_text="test", tool_calls=None, cleaned_thinking=""
        )
        assert hasattr(ext, "validation_result")
        assert ext.validation_result is None  # default


class TestExtractAndValidateWrapper:
    def test_wrapper_passthrough_when_validation_disabled(self):
        from omlx.api.tool_calling import extract_and_validate_tool_calls

        class FakeTokenizer:
            pass

        result = extract_and_validate_tool_calls(
            thinking_content="",
            regular_content="hello world",
            tokenizer=FakeTokenizer(),
            tools=None,
            validation_enabled=False,
        )
        assert isinstance(result, ToolCallExtraction)
        assert result.validation_result is None
        assert "hello world" in result.cleaned_text

    def test_wrapper_validation_enabled_no_tools_skips_validation(self):
        from omlx.api.tool_calling import extract_and_validate_tool_calls

        class FakeTokenizer:
            pass

        # validation_enabled=True but no tools → should skip validation
        result = extract_and_validate_tool_calls(
            thinking_content="",
            regular_content="hi",
            tokenizer=FakeTokenizer(),
            tools=None,
            validation_enabled=True,
        )
        assert result.validation_result is None
