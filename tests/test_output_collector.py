# SPDX-License-Identifier: Apache-2.0
"""
Tests for OutputCollector module.

Tests cover:
- RequestStreamState: should_send(), mark_sent()
- RequestOutputCollector initialization
- put(): adding outputs
- get_nowait(): non-blocking get
- get(): async get
- clear(): cleanup
- _merge_outputs(): output merging
- has_waiting_consumers(): class method

Note: Uses pytest-asyncio for async tests.
"""

import asyncio

import pytest

from omlx.output_collector import RequestOutputCollector, RequestStreamState
from omlx.request import RequestOutput


class TestRequestStreamState:
    """Tests for RequestStreamState dataclass."""

    def test_default_values(self):
        """Test RequestStreamState has correct defaults."""
        state = RequestStreamState()

        assert state.stream_interval == 1
        assert state.sent_tokens == 0

    def test_custom_values(self):
        """Test RequestStreamState with custom values."""
        state = RequestStreamState(stream_interval=5, sent_tokens=10)

        assert state.stream_interval == 5
        assert state.sent_tokens == 10


class TestRequestStreamStateShouldSend:
    """Tests for RequestStreamState.should_send()."""

    def test_should_send_on_finish(self):
        """Test should_send returns True when finished."""
        state = RequestStreamState(stream_interval=10)
        state.sent_tokens = 5

        # Should always send on finish regardless of token count
        assert state.should_send(total_tokens=6, finished=True) is True
        assert state.should_send(total_tokens=5, finished=True) is True

    def test_should_send_first_token(self):
        """Test should_send returns True for first token."""
        state = RequestStreamState(stream_interval=10)

        # First token (sent_tokens == 0) should always send
        assert state.should_send(total_tokens=1, finished=False) is True

    def test_should_send_interval_reached(self):
        """Test should_send returns True when interval reached."""
        state = RequestStreamState(stream_interval=5)
        state.sent_tokens = 5

        # Not enough tokens accumulated
        assert state.should_send(total_tokens=6, finished=False) is False
        assert state.should_send(total_tokens=9, finished=False) is False

        # Exactly at interval
        assert state.should_send(total_tokens=10, finished=False) is True

        # Past interval
        assert state.should_send(total_tokens=15, finished=False) is True

    def test_should_send_interval_one(self):
        """Test should_send with interval=1 (every token)."""
        state = RequestStreamState(stream_interval=1)

        # Should send every token
        state.sent_tokens = 0
        assert state.should_send(total_tokens=1, finished=False) is True

        state.sent_tokens = 1
        assert state.should_send(total_tokens=2, finished=False) is True

        state.sent_tokens = 5
        assert state.should_send(total_tokens=6, finished=False) is True


class TestRequestStreamStateMarkSent:
    """Tests for RequestStreamState.mark_sent()."""

    def test_mark_sent_updates_count(self):
        """Test mark_sent updates sent_tokens."""
        state = RequestStreamState()

        state.mark_sent(total_tokens=5)
        assert state.sent_tokens == 5

        state.mark_sent(total_tokens=10)
        assert state.sent_tokens == 10

    def test_mark_sent_sequence(self):
        """Test mark_sent in a typical streaming sequence."""
        state = RequestStreamState(stream_interval=3)

        # Stream interval = 3: send at 0, 3, 6, 9...
        assert state.should_send(1, False)  # First token
        state.mark_sent(1)
        assert state.sent_tokens == 1

        assert not state.should_send(2, False)
        assert not state.should_send(3, False)
        assert state.should_send(4, False)  # 4 - 1 = 3 >= interval
        state.mark_sent(4)
        assert state.sent_tokens == 4


class TestRequestOutputCollectorInit:
    """Tests for RequestOutputCollector initialization."""

    def test_default_init(self):
        """Test default initialization."""
        collector = RequestOutputCollector()

        assert collector.output is None
        assert collector.aggregate is True
        assert collector._is_waiting is False

    def test_init_without_aggregation(self):
        """Test initialization without aggregation."""
        collector = RequestOutputCollector(aggregate=False)

        assert collector.aggregate is False

    def test_ready_event_initial_state(self):
        """Test ready event is initially not set."""
        collector = RequestOutputCollector()

        assert not collector.ready.is_set()


class TestRequestOutputCollectorPut:
    """Tests for RequestOutputCollector.put()."""

    def test_put_first_output(self):
        """Test putting first output."""
        collector = RequestOutputCollector()

        output = RequestOutput(
            request_id="test-001",
            new_token_ids=[100],
            new_text="Hello",
        )
        collector.put(output)

        assert collector.output is output
        assert collector.ready.is_set()

    def test_put_multiple_with_aggregation(self):
        """Test putting multiple outputs aggregates them."""
        collector = RequestOutputCollector(aggregate=True)

        output1 = RequestOutput(
            request_id="test-001",
            new_token_ids=[100],
            new_text="Hello",
            completion_tokens=1,
        )
        output2 = RequestOutput(
            request_id="test-001",
            new_token_ids=[101],
            new_text=" world",
            completion_tokens=2,
        )

        collector.put(output1)
        collector.put(output2)

        # Should be merged
        assert collector.output is not None
        assert collector.output.new_token_ids == [100, 101]
        assert collector.output.new_text == "Hello world"
        assert collector.output.completion_tokens == 2

    def test_put_multiple_without_aggregation(self):
        """Test putting multiple outputs replaces without aggregation."""
        collector = RequestOutputCollector(aggregate=False)

        output1 = RequestOutput(
            request_id="test-001",
            new_token_ids=[100],
            new_text="Hello",
        )
        output2 = RequestOutput(
            request_id="test-001",
            new_token_ids=[101],
            new_text=" world",
        )

        collector.put(output1)
        collector.put(output2)

        # Should be replaced, not merged
        assert collector.output is not None
        assert collector.output.new_token_ids == [101]
        assert collector.output.new_text == " world"


class TestRequestOutputCollectorGetNowait:
    """Tests for RequestOutputCollector.get_nowait()."""

    def test_get_nowait_empty(self):
        """Test get_nowait returns None when empty."""
        collector = RequestOutputCollector()

        result = collector.get_nowait()

        assert result is None

    def test_get_nowait_returns_output(self):
        """Test get_nowait returns and clears output."""
        collector = RequestOutputCollector()

        output = RequestOutput(
            request_id="test-001",
            new_token_ids=[100],
            new_text="Hello",
        )
        collector.put(output)

        result = collector.get_nowait()

        assert result is output
        assert collector.output is None
        assert not collector.ready.is_set()

    def test_get_nowait_clears_ready(self):
        """Test get_nowait clears ready event."""
        collector = RequestOutputCollector()

        output = RequestOutput(request_id="test-001")
        collector.put(output)
        assert collector.ready.is_set()

        collector.get_nowait()
        assert not collector.ready.is_set()


class TestRequestOutputCollectorGet:
    """Tests for RequestOutputCollector.get() async method."""

    @pytest.mark.asyncio
    async def test_get_with_output_ready(self):
        """Test get() returns immediately when output available."""
        collector = RequestOutputCollector()

        output = RequestOutput(request_id="test-001", new_text="Hello")
        collector.put(output)

        result = await collector.get()

        assert result is output

    @pytest.mark.asyncio
    async def test_get_waits_for_output(self):
        """Test get() waits until output is available."""
        collector = RequestOutputCollector()

        async def delayed_put():
            await asyncio.sleep(0.05)
            collector.put(RequestOutput(request_id="test-001", new_text="Delayed"))

        # Start delayed put
        asyncio.create_task(delayed_put())

        # get() should wait
        result = await asyncio.wait_for(collector.get(), timeout=1.0)

        assert result is not None
        assert result.new_text == "Delayed"

    @pytest.mark.asyncio
    async def test_get_tracks_waiting_consumer(self):
        """Test get() tracks waiting consumers."""
        collector = RequestOutputCollector()

        # Reset global counter
        initial_count = RequestOutputCollector._waiting_consumers

        async def get_with_delay():
            return await collector.get()

        # Start get task
        task = asyncio.create_task(get_with_delay())

        # Give time for task to start waiting
        await asyncio.sleep(0.01)

        # Check counter incremented
        assert RequestOutputCollector._waiting_consumers > initial_count

        # Provide output
        collector.put(RequestOutput(request_id="test-001"))

        # Wait for task to complete
        await task

        # Counter should be back to initial
        assert RequestOutputCollector._waiting_consumers == initial_count


class TestRequestOutputCollectorClear:
    """Tests for RequestOutputCollector.clear()."""

    def test_clear_removes_output(self):
        """Test clear() removes pending output."""
        collector = RequestOutputCollector()

        collector.put(RequestOutput(request_id="test-001"))
        collector.clear()

        assert collector.output is None
        assert not collector.ready.is_set()

    def test_clear_resets_waiting_flag(self):
        """Test clear() resets waiting flag."""
        collector = RequestOutputCollector()
        collector._is_waiting = True

        initial_count = RequestOutputCollector._waiting_consumers
        RequestOutputCollector._waiting_consumers += 1

        collector.clear()

        assert collector._is_waiting is False


class TestRequestOutputCollectorMergeOutputs:
    """Tests for RequestOutputCollector._merge_outputs()."""

    def test_merge_combines_tokens(self):
        """Test _merge_outputs combines token lists."""
        collector = RequestOutputCollector()

        existing = RequestOutput(
            request_id="test-001",
            new_token_ids=[100, 101],
            new_text="Hello ",
            output_token_ids=[100, 101],
            completion_tokens=2,
        )
        new = RequestOutput(
            request_id="test-001",
            new_token_ids=[102],
            new_text="world",
            output_token_ids=[100, 101, 102],
            completion_tokens=3,
        )

        result = collector._merge_outputs(existing, new)

        assert result.new_token_ids == [100, 101, 102]
        assert result.new_text == "Hello world"
        assert result.output_token_ids == [100, 101, 102]  # Uses latest
        assert result.completion_tokens == 3  # Uses latest

    def test_merge_preserves_finished_status(self):
        """Test _merge_outputs preserves finished status."""
        collector = RequestOutputCollector()

        existing = RequestOutput(
            request_id="test-001",
            new_token_ids=[100],
            new_text="Hello",
            finished=False,
        )
        new = RequestOutput(
            request_id="test-001",
            new_token_ids=[101],
            new_text=" world",
            finished=True,
            finish_reason="stop",
        )

        result = collector._merge_outputs(existing, new)

        assert result.finished is True
        assert result.finish_reason == "stop"

    def test_merge_preserves_tool_calls(self):
        """Test _merge_outputs preserves tool_calls."""
        collector = RequestOutputCollector()

        existing = RequestOutput(
            request_id="test-001",
            new_token_ids=[100],
            new_text="",
        )
        new = RequestOutput(
            request_id="test-001",
            new_token_ids=[101],
            new_text="",
            tool_calls=[{"name": "test_tool", "arguments": "{}"}],
        )

        result = collector._merge_outputs(existing, new)

        assert result.tool_calls == [{"name": "test_tool", "arguments": "{}"}]


class TestRequestOutputCollectorHasWaitingConsumers:
    """Tests for RequestOutputCollector.has_waiting_consumers()."""

    def test_has_waiting_consumers_false_initially(self):
        """Test has_waiting_consumers returns False when none waiting."""
        # Reset counter for clean test
        original = RequestOutputCollector._waiting_consumers
        RequestOutputCollector._waiting_consumers = 0

        try:
            assert RequestOutputCollector.has_waiting_consumers() is False
        finally:
            RequestOutputCollector._waiting_consumers = original

    def test_has_waiting_consumers_true_when_waiting(self):
        """Test has_waiting_consumers returns True when consumers waiting."""
        original = RequestOutputCollector._waiting_consumers
        RequestOutputCollector._waiting_consumers = 1

        try:
            assert RequestOutputCollector.has_waiting_consumers() is True
        finally:
            RequestOutputCollector._waiting_consumers = original

    @pytest.mark.asyncio
    async def test_has_waiting_consumers_integration(self):
        """Test has_waiting_consumers in realistic scenario."""
        collector = RequestOutputCollector()

        # Reset counter
        original = RequestOutputCollector._waiting_consumers
        RequestOutputCollector._waiting_consumers = 0

        try:
            assert RequestOutputCollector.has_waiting_consumers() is False

            # Start waiting task
            async def wait_for_output():
                return await collector.get()

            task = asyncio.create_task(wait_for_output())
            await asyncio.sleep(0.01)  # Let task start waiting

            assert RequestOutputCollector.has_waiting_consumers() is True

            # Provide output
            collector.put(RequestOutput(request_id="test-001"))
            await task

            # After task completes, should be back to original
            assert RequestOutputCollector._waiting_consumers == 0

        finally:
            RequestOutputCollector._waiting_consumers = original


class TestRequestOutputCollectorEdgeCases:
    """Edge case tests for RequestOutputCollector."""

    def test_put_get_sequence(self):
        """Test typical put/get sequence."""
        collector = RequestOutputCollector()

        # Simulate streaming sequence
        for i in range(3):
            output = RequestOutput(
                request_id="test-001",
                new_token_ids=[100 + i],
                new_text=f"token{i}",
                completion_tokens=i + 1,
            )
            collector.put(output)

            result = collector.get_nowait()
            assert result is not None
            assert result.new_text == f"token{i}"

    def test_empty_output_handling(self):
        """Test handling empty outputs."""
        collector = RequestOutputCollector()

        output = RequestOutput(
            request_id="test-001",
            new_token_ids=[],
            new_text="",
        )
        collector.put(output)

        result = collector.get_nowait()
        assert result is not None
        assert result.new_token_ids == []
        assert result.new_text == ""

    def test_merge_empty_with_content(self):
        """Test merging empty output with content."""
        collector = RequestOutputCollector()

        existing = RequestOutput(
            request_id="test-001",
            new_token_ids=[],
            new_text="",
        )
        new = RequestOutput(
            request_id="test-001",
            new_token_ids=[100],
            new_text="Hello",
        )

        result = collector._merge_outputs(existing, new)

        assert result.new_token_ids == [100]
        assert result.new_text == "Hello"

    def test_merge_preserves_error_from_new(self):
        """Test _merge_outputs preserves error field from new output."""
        collector = RequestOutputCollector()

        existing = RequestOutput(
            request_id="test-001",
            new_token_ids=[100],
            new_text="Hello",
        )
        new = RequestOutput(
            request_id="test-001",
            new_token_ids=[101],
            new_text=" world",
            finished=True,
            error="Memory limit exceeded during prefill",
        )

        result = collector._merge_outputs(existing, new)

        assert result.error == "Memory limit exceeded during prefill"
        assert result.finished is True

    def test_merge_preserves_error_from_existing(self):
        """Test _merge_outputs preserves error field from existing output."""
        collector = RequestOutputCollector()

        existing = RequestOutput(
            request_id="test-001",
            new_token_ids=[100],
            new_text="Hello",
            error="Some earlier error",
        )
        new = RequestOutput(
            request_id="test-001",
            new_token_ids=[101],
            new_text=" world",
            finished=True,
        )

        result = collector._merge_outputs(existing, new)

        assert result.error == "Some earlier error"

    def test_error_output_put_and_get(self):
        """Test putting and getting an error output."""
        collector = RequestOutputCollector()

        error_output = RequestOutput(
            request_id="test-001",
            finished=True,
            finish_reason="error",
            error="Memory limit exceeded during prefill",
        )
        collector.put(error_output)

        result = collector.get_nowait()
        assert result is not None
        assert result.error == "Memory limit exceeded during prefill"
        assert result.finished is True
        assert result.finish_reason == "error"
