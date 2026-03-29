# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.request module."""

import time

import pytest

from omlx.request import (
    RequestStatus,
    SamplingParams,
    Request,
    RequestOutput,
)


class TestRequestStatus:
    """Test cases for RequestStatus enum."""

    def test_status_values(self):
        """Test that status enum has expected values."""
        assert RequestStatus.WAITING is not None
        assert RequestStatus.RUNNING is not None
        assert RequestStatus.PREEMPTED is not None
        assert RequestStatus.FINISHED_STOPPED is not None
        assert RequestStatus.FINISHED_LENGTH_CAPPED is not None
        assert RequestStatus.FINISHED_ABORTED is not None

    def test_status_ordering(self):
        """Test that finished statuses are greater than active statuses."""
        assert RequestStatus.WAITING < RequestStatus.FINISHED_STOPPED
        assert RequestStatus.RUNNING < RequestStatus.FINISHED_STOPPED
        assert RequestStatus.PREEMPTED < RequestStatus.FINISHED_STOPPED

    def test_is_finished_active_states(self):
        """Test is_finished returns False for active states."""
        assert RequestStatus.is_finished(RequestStatus.WAITING) is False
        assert RequestStatus.is_finished(RequestStatus.RUNNING) is False
        assert RequestStatus.is_finished(RequestStatus.PREEMPTED) is False

    def test_is_finished_finished_states(self):
        """Test is_finished returns True for finished states."""
        assert RequestStatus.is_finished(RequestStatus.FINISHED_STOPPED) is True
        assert RequestStatus.is_finished(RequestStatus.FINISHED_LENGTH_CAPPED) is True
        assert RequestStatus.is_finished(RequestStatus.FINISHED_ABORTED) is True

    def test_get_finish_reason_stopped(self):
        """Test get_finish_reason for FINISHED_STOPPED."""
        reason = RequestStatus.get_finish_reason(RequestStatus.FINISHED_STOPPED)
        assert reason == "stop"

    def test_get_finish_reason_length_capped(self):
        """Test get_finish_reason for FINISHED_LENGTH_CAPPED."""
        reason = RequestStatus.get_finish_reason(RequestStatus.FINISHED_LENGTH_CAPPED)
        assert reason == "length"

    def test_get_finish_reason_aborted(self):
        """Test get_finish_reason for FINISHED_ABORTED."""
        reason = RequestStatus.get_finish_reason(RequestStatus.FINISHED_ABORTED)
        assert reason == "abort"

    def test_get_finish_reason_active_states(self):
        """Test get_finish_reason returns None for active states."""
        assert RequestStatus.get_finish_reason(RequestStatus.WAITING) is None
        assert RequestStatus.get_finish_reason(RequestStatus.RUNNING) is None
        assert RequestStatus.get_finish_reason(RequestStatus.PREEMPTED) is None


class TestSamplingParams:
    """Test cases for SamplingParams dataclass."""

    def test_default_values(self):
        """Test default sampling parameter values."""
        params = SamplingParams()
        assert params.max_tokens == 256
        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.top_k == 0
        assert params.min_p == 0.0
        assert params.xtc_probability == 0.0
        assert params.xtc_threshold == 0.1
        assert params.repetition_penalty == 1.0
        assert params.presence_penalty == 0.0
        assert params.stop == []
        assert params.stop_token_ids == []
        assert params.logprobs is False
        assert params.top_logprobs is None

    def test_custom_values(self):
        """Test custom sampling parameter values."""
        params = SamplingParams(
            max_tokens=1024,
            temperature=0.5,
            top_p=0.95,
            top_k=40,
            min_p=0.05,
            xtc_probability=0.5,
            xtc_threshold=0.1,
            repetition_penalty=1.1,
            presence_penalty=0.5,
            stop=["###", "END"],
            stop_token_ids=[2, 100],
            logprobs=True,
            top_logprobs=5,
        )
        assert params.max_tokens == 1024
        assert params.temperature == 0.5
        assert params.top_p == 0.95
        assert params.top_k == 40
        assert params.min_p == 0.05
        assert params.xtc_probability == 0.5
        assert params.xtc_threshold == 0.1
        assert params.repetition_penalty == 1.1
        assert params.presence_penalty == 0.5
        assert params.stop == ["###", "END"]
        assert params.stop_token_ids == [2, 100]
        assert params.logprobs is True
        assert params.top_logprobs == 5

    def test_post_init_none_stop(self):
        """Test that None stop sequences are converted to empty lists."""
        params = SamplingParams(stop=None, stop_token_ids=None)
        assert params.stop == []
        assert params.stop_token_ids == []

    def test_greedy_sampling(self):
        """Test parameters for greedy sampling (temperature=0)."""
        params = SamplingParams(temperature=0.0, top_k=1)
        assert params.temperature == 0.0
        assert params.top_k == 1


class TestRequest:
    """Test cases for Request dataclass."""

    def test_basic_creation(self):
        """Test basic request creation."""
        request = Request(
            request_id="test-001",
            prompt="Hello, world!",
            sampling_params=SamplingParams(),
        )
        assert request.request_id == "test-001"
        assert request.prompt == "Hello, world!"
        assert request.status == RequestStatus.WAITING
        assert request.output_token_ids == []
        assert request.output_text == ""

    def test_creation_with_token_ids(self):
        """Test request creation with token IDs as prompt."""
        request = Request(
            request_id="test-002",
            prompt=[1, 2, 3, 4, 5],
            sampling_params=SamplingParams(),
        )
        assert request.prompt == [1, 2, 3, 4, 5]

    def test_arrival_time_auto_set(self):
        """Test that arrival_time is automatically set."""
        before = time.time()
        request = Request(
            request_id="test-003",
            prompt="Test",
            sampling_params=SamplingParams(),
        )
        after = time.time()
        assert before <= request.arrival_time <= after

    def test_num_output_tokens_property(self):
        """Test num_output_tokens property."""
        request = Request(
            request_id="test-004",
            prompt="Test",
            sampling_params=SamplingParams(),
        )
        assert request.num_output_tokens == 0

        request.output_token_ids = [100, 200, 300]
        assert request.num_output_tokens == 3

    def test_num_tokens_property(self):
        """Test num_tokens property (prompt + output)."""
        request = Request(
            request_id="test-005",
            prompt="Test",
            sampling_params=SamplingParams(),
        )
        request.num_prompt_tokens = 10
        request.output_token_ids = [100, 200, 300]
        assert request.num_tokens == 13

    def test_max_tokens_property(self):
        """Test max_tokens property from sampling_params."""
        request = Request(
            request_id="test-006",
            prompt="Test",
            sampling_params=SamplingParams(max_tokens=512),
        )
        assert request.max_tokens == 512

    def test_is_finished_method(self):
        """Test is_finished method."""
        request = Request(
            request_id="test-007",
            prompt="Test",
            sampling_params=SamplingParams(),
        )
        assert request.is_finished() is False

        request.status = RequestStatus.FINISHED_STOPPED
        assert request.is_finished() is True

    def test_get_finish_reason_method(self):
        """Test get_finish_reason method."""
        request = Request(
            request_id="test-008",
            prompt="Test",
            sampling_params=SamplingParams(),
        )

        # No finish reason while active
        assert request.get_finish_reason() is None

        # From status
        request.status = RequestStatus.FINISHED_STOPPED
        assert request.get_finish_reason() == "stop"

        # Override with explicit finish_reason
        request.finish_reason = "custom_reason"
        assert request.get_finish_reason() == "custom_reason"

    def test_append_output_token(self):
        """Test append_output_token method."""
        request = Request(
            request_id="test-009",
            prompt="Test",
            sampling_params=SamplingParams(),
        )
        request.append_output_token(100)
        request.append_output_token(200)

        assert request.output_token_ids == [100, 200]
        assert request.num_computed_tokens == 2

    def test_set_finished(self):
        """Test set_finished method."""
        request = Request(
            request_id="test-010",
            prompt="Test",
            sampling_params=SamplingParams(),
        )
        request.set_finished(RequestStatus.FINISHED_STOPPED)

        assert request.status == RequestStatus.FINISHED_STOPPED
        assert request.finish_reason == "stop"

    def test_set_finished_with_reason(self):
        """Test set_finished with custom reason."""
        request = Request(
            request_id="test-011",
            prompt="Test",
            sampling_params=SamplingParams(),
        )
        request.set_finished(RequestStatus.FINISHED_ABORTED, reason="user_cancelled")

        assert request.status == RequestStatus.FINISHED_ABORTED
        assert request.finish_reason == "user_cancelled"

    def test_comparison_by_priority(self):
        """Test request comparison by priority."""
        request1 = Request(
            request_id="test-012",
            prompt="Test",
            sampling_params=SamplingParams(),
            priority=1,
        )
        request2 = Request(
            request_id="test-013",
            prompt="Test",
            sampling_params=SamplingParams(),
            priority=2,
        )
        # Lower priority value = higher priority
        assert request1 < request2

    def test_comparison_by_arrival_time(self):
        """Test request comparison by arrival time (same priority)."""
        request1 = Request(
            request_id="test-014",
            prompt="Test",
            sampling_params=SamplingParams(),
            arrival_time=100.0,
        )
        request2 = Request(
            request_id="test-015",
            prompt="Test",
            sampling_params=SamplingParams(),
            arrival_time=200.0,
        )
        # Earlier arrival time = higher priority
        assert request1 < request2

    def test_hash(self):
        """Test request hash is based on request_id."""
        request1 = Request(
            request_id="test-016",
            prompt="Test",
            sampling_params=SamplingParams(),
        )
        request2 = Request(
            request_id="test-016",
            prompt="Different prompt",
            sampling_params=SamplingParams(),
        )
        assert hash(request1) == hash(request2)

    def test_equality(self):
        """Test request equality is based on request_id."""
        request1 = Request(
            request_id="test-017",
            prompt="Test",
            sampling_params=SamplingParams(),
        )
        request2 = Request(
            request_id="test-017",
            prompt="Different",
            sampling_params=SamplingParams(),
        )
        request3 = Request(
            request_id="test-018",
            prompt="Test",
            sampling_params=SamplingParams(),
        )
        assert request1 == request2
        assert request1 != request3

    def test_equality_with_non_request(self):
        """Test equality with non-Request objects."""
        request = Request(
            request_id="test-019",
            prompt="Test",
            sampling_params=SamplingParams(),
        )
        assert request != "test-019"
        assert request != 123
        assert request != None

    def test_reasoning_model_fields(self):
        """Test reasoning model support fields."""
        request = Request(
            request_id="test-020",
            prompt="Test",
            sampling_params=SamplingParams(),
            needs_think_prefix=True,
        )
        assert request.needs_think_prefix is True
        assert request.think_prefix_sent is False

    def test_harmony_model_field(self):
        """Test Harmony model field."""
        request = Request(
            request_id="test-021",
            prompt="Test",
            sampling_params=SamplingParams(),
            is_harmony_model=True,
        )
        assert request.is_harmony_model is True

    def test_multimodal_fields(self):
        """Test multimodal content fields."""
        request = Request(
            request_id="test-022",
            prompt="Describe this image",
            sampling_params=SamplingParams(),
            images=["image_data_1", "image_data_2"],
            videos=["video_data_1"],
        )
        assert request.images == ["image_data_1", "image_data_2"]
        assert request.videos == ["video_data_1"]


class TestRequestOutput:
    """Test cases for RequestOutput dataclass."""

    def test_basic_creation(self):
        """Test basic RequestOutput creation."""
        output = RequestOutput(request_id="test-001")
        assert output.request_id == "test-001"
        assert output.new_token_ids == []
        assert output.new_text == ""
        assert output.output_token_ids == []
        assert output.output_text == ""
        assert output.finished is False
        assert output.finish_reason is None

    def test_with_tokens(self):
        """Test RequestOutput with tokens."""
        output = RequestOutput(
            request_id="test-002",
            new_token_ids=[100, 200],
            new_text="Hello",
            output_token_ids=[100, 200, 300, 400],
            output_text="Hello world",
        )
        assert output.new_token_ids == [100, 200]
        assert output.new_text == "Hello"
        assert output.output_token_ids == [100, 200, 300, 400]
        assert output.output_text == "Hello world"

    def test_finished_output(self):
        """Test finished RequestOutput."""
        output = RequestOutput(
            request_id="test-003",
            finished=True,
            finish_reason="stop",
        )
        assert output.finished is True
        assert output.finish_reason == "stop"

    def test_usage_property(self):
        """Test usage property."""
        output = RequestOutput(
            request_id="test-004",
            prompt_tokens=10,
            completion_tokens=20,
        )
        usage = output.usage
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 20
        assert usage["total_tokens"] == 30

    def test_usage_property_zero(self):
        """Test usage property with zero tokens."""
        output = RequestOutput(request_id="test-005")
        usage = output.usage
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["total_tokens"] == 0

    def test_tool_calls(self):
        """Test RequestOutput with tool calls."""
        tool_calls = [
            {"id": "call_1", "function": {"name": "get_weather", "arguments": "{}"}},
            {"id": "call_2", "function": {"name": "search", "arguments": "{}"}},
        ]
        output = RequestOutput(
            request_id="test-006",
            tool_calls=tool_calls,
        )
        assert output.tool_calls == tool_calls
        assert len(output.tool_calls) == 2
