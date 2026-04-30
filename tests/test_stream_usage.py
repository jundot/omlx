# SPDX-License-Identifier: Apache-2.0
"""Tests for streaming usage (stream_options.include_usage) support."""

import json

import pytest

from omlx.api.openai_models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    CompletionRequest,
    PromptTokensDetails,
    StreamOptions,
    Usage,
)


class TestStreamOptions:
    """Tests for StreamOptions model."""

    def test_default_include_usage_false(self):
        opts = StreamOptions()
        assert opts.include_usage is False

    def test_include_usage_true(self):
        opts = StreamOptions(include_usage=True)
        assert opts.include_usage is True

    def test_from_dict(self):
        opts = StreamOptions(**{"include_usage": True})
        assert opts.include_usage is True


class TestStreamOptionsInRequest:
    """Tests for stream_options field in request models."""

    def test_chat_request_no_stream_options(self):
        req = ChatCompletionRequest(
            model="test", messages=[{"role": "user", "content": "hi"}]
        )
        assert req.stream_options is None

    def test_chat_request_with_stream_options(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        assert req.stream_options is not None
        assert req.stream_options.include_usage is True

    def test_completion_request_with_stream_options(self):
        req = CompletionRequest(
            model="test",
            prompt="hello",
            stream=True,
            stream_options={"include_usage": True},
        )
        assert req.stream_options is not None
        assert req.stream_options.include_usage is True


class TestUsageExtendedFields:
    """Tests for extended timing fields in Usage model."""

    def test_basic_usage_unchanged(self):
        usage = Usage(prompt_tokens=10, completion_tokens=5)
        assert usage.total_tokens == 15
        assert usage.prompt_tokens_details is None
        assert usage.time_to_first_token is None

    def test_usage_with_timing(self):
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=20),
            time_to_first_token=0.5,
            total_time=2.0,
            prompt_eval_duration=0.5,
            generation_duration=1.5,
            prompt_tokens_per_second=200.0,
            generation_tokens_per_second=33.33,
        )
        assert usage.total_tokens == 150
        assert usage.prompt_tokens_details.cached_tokens == 20
        assert usage.time_to_first_token == 0.5
        assert usage.generation_tokens_per_second == 33.33

    def test_usage_none_fields_excluded(self):
        """None timing fields should be excluded with exclude_none."""
        usage = Usage(prompt_tokens=10, completion_tokens=5)
        dumped = usage.model_dump(exclude_none=True)
        assert "prompt_tokens_details" not in dumped
        assert "time_to_first_token" not in dumped
        assert "model_load_duration" not in dumped
        # Standard fields should still be present
        assert dumped["prompt_tokens"] == 10
        assert dumped["completion_tokens"] == 5
        assert dumped["total_tokens"] == 15

    def test_usage_with_model_load(self):
        usage = Usage(
            prompt_tokens=10,
            completion_tokens=5,
            model_load_duration=55.93,
        )
        dumped = usage.model_dump(exclude_none=True)
        assert dumped["model_load_duration"] == 55.93
        assert "prompt_tokens_details" not in dumped


class TestUsageChunkFormat:
    """Tests for usage chunk structure (OpenAI spec: choices=[], usage present)."""

    def test_usage_chunk_empty_choices(self):
        chunk = ChatCompletionChunk(
            id="chatcmpl-test",
            model="test-model",
            choices=[],
            usage=Usage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                time_to_first_token=0.12,
                total_time=1.5,
                prompt_eval_duration=0.12,
                generation_duration=1.38,
                prompt_tokens_per_second=833.33,
                generation_tokens_per_second=36.23,
            ),
        )
        data = json.loads(chunk.model_dump_json(exclude_none=True))
        assert data["choices"] == []
        assert data["usage"]["prompt_tokens"] == 100
        assert data["usage"]["completion_tokens"] == 50
        assert data["usage"]["total_tokens"] == 150
        assert data["usage"]["time_to_first_token"] == 0.12
        assert data["usage"]["generation_tokens_per_second"] == 36.23
        assert "model_load_duration" not in data["usage"]

    def test_usage_chunk_with_all_fields(self):
        chunk = ChatCompletionChunk(
            id="chatcmpl-test",
            model="test-model",
            choices=[],
            usage=Usage(
                prompt_tokens=9752,
                completion_tokens=554,
                total_tokens=10306,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=0),
                model_load_duration=55.93,
                time_to_first_token=115.05,
                total_time=182.47,
                prompt_eval_duration=59.13,
                generation_duration=67.42,
                prompt_tokens_per_second=164.93,
                generation_tokens_per_second=8.22,
            ),
        )
        data = json.loads(chunk.model_dump_json(exclude_none=True))
        usage = data["usage"]
        assert usage["prompt_tokens"] == 9752
        assert usage["completion_tokens"] == 554
        assert usage["total_tokens"] == 10306
        assert usage["prompt_tokens_details"]["cached_tokens"] == 0
        assert usage["model_load_duration"] == 55.93
        assert usage["time_to_first_token"] == 115.05
        assert usage["total_time"] == 182.47
        assert usage["prompt_eval_duration"] == 59.13
        assert usage["generation_duration"] == 67.42
        assert usage["prompt_tokens_per_second"] == 164.93
        assert usage["generation_tokens_per_second"] == 8.22
