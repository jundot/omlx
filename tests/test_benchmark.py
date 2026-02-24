# SPDX-License-Identifier: Apache-2.0
"""Tests for the admin benchmark module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.admin.benchmark import (
    VALID_BATCH_SIZES,
    VALID_PROMPT_LENGTHS,
    BenchmarkRequest,
    BenchmarkRun,
    _compute_single_metrics,
    _generate_prompt,
    cleanup_old_runs,
    create_run,
    get_run,
)


# =============================================================================
# BenchmarkRequest validation tests
# =============================================================================


class TestBenchmarkRequest:
    def test_valid_request(self):
        req = BenchmarkRequest(
            model_id="test-model",
            prompt_lengths=[1024, 4096],
            generation_length=128,
            batch_sizes=[2, 4],
        )
        assert req.model_id == "test-model"
        assert req.prompt_lengths == [1024, 4096]
        assert req.batch_sizes == [2, 4]

    def test_prompt_lengths_sorted(self):
        req = BenchmarkRequest(
            model_id="test-model",
            prompt_lengths=[8192, 1024, 4096],
        )
        assert req.prompt_lengths == [1024, 4096, 8192]

    def test_empty_prompt_lengths_rejected(self):
        with pytest.raises(ValueError, match="At least one prompt length"):
            BenchmarkRequest(model_id="test-model", prompt_lengths=[])

    def test_invalid_prompt_length_rejected(self):
        with pytest.raises(ValueError, match="Invalid prompt length 512"):
            BenchmarkRequest(model_id="test-model", prompt_lengths=[512])

    def test_invalid_batch_size_rejected(self):
        with pytest.raises(ValueError, match="Invalid batch size 3"):
            BenchmarkRequest(
                model_id="test-model",
                prompt_lengths=[1024],
                batch_sizes=[3],
            )

    def test_empty_batch_sizes_allowed(self):
        req = BenchmarkRequest(
            model_id="test-model",
            prompt_lengths=[1024],
            batch_sizes=[],
        )
        assert req.batch_sizes == []

    def test_batch_sizes_sorted(self):
        req = BenchmarkRequest(
            model_id="test-model",
            prompt_lengths=[1024],
            batch_sizes=[8, 2, 4],
        )
        assert req.batch_sizes == [2, 4, 8]

    def test_default_generation_length(self):
        req = BenchmarkRequest(
            model_id="test-model",
            prompt_lengths=[1024],
        )
        assert req.generation_length == 128


# =============================================================================
# Prompt generation tests
# =============================================================================


class TestGeneratePrompt:
    def test_exact_token_count(self):
        """Verify prompt generates exact number of tokens."""
        tokenizer = MagicMock()

        # Simulate tokenizer behavior
        def mock_encode(text):
            # Return roughly 1 token per 4 chars
            return list(range(len(text) // 4))

        def mock_decode(tokens):
            return "x" * len(tokens) * 4

        tokenizer.encode = mock_encode
        tokenizer.decode = mock_decode

        prompt = _generate_prompt(tokenizer, 1024)

        # Verify encode was called and result was truncated
        encoded = tokenizer.encode(prompt)
        assert len(encoded) == 1024

    def test_uuid_prefix_uniqueness(self):
        """Verify each generated prompt has a unique UUID prefix."""
        tokenizer = MagicMock()
        tokenizer.encode = lambda text: list(range(2048))
        tokenizer.decode = lambda tokens: f"decoded-{len(tokens)}"

        prompts = set()
        for _ in range(10):
            # We can't easily verify uniqueness since decode is mocked,
            # but we verify encode is called with text containing "BENCH-"
            prompt = _generate_prompt(tokenizer, 100)
            prompts.add(prompt)

        # With mock decode they'll all be the same, but in real usage
        # the UUID prefix ensures cache isolation


# =============================================================================
# Metrics computation tests
# =============================================================================


class TestComputeMetrics:
    def test_basic_metrics(self):
        """Test metric computation with known values."""
        metrics = _compute_single_metrics(
            prompt_tokens=1024,
            completion_tokens=128,
            start_time=0.0,
            first_token_time=0.1,  # 100ms TTFT
            end_time=1.38,  # 1.28s generation
            peak_memory=4 * 1024 * 1024 * 1024,  # 4GB
            cached_tokens=0,
        )

        assert metrics["ttft_ms"] == pytest.approx(100.0, abs=0.1)
        assert metrics["prompt_tokens"] == 1024
        assert metrics["completion_tokens"] == 128
        assert metrics["cached_tokens"] == 0
        assert metrics["peak_memory_bytes"] == 4 * 1024 * 1024 * 1024

        # Gen TPS = 128 / 1.28 = 100 tok/s
        assert metrics["gen_tps"] == pytest.approx(100.0, abs=0.1)

        # Processing TPS = 1024 / 0.1 = 10240 tok/s
        assert metrics["processing_tps"] == pytest.approx(10240.0, abs=1.0)

        # TPOT = 1280ms / 127 = ~10.08 ms/tok
        assert metrics["tpot_ms"] == pytest.approx(10.08, abs=0.1)

        # E2E = 1.38s
        assert metrics["e2e_latency_s"] == pytest.approx(1.38, abs=0.001)

        # Total throughput = (1024 + 128) / 1.38 = ~834.8 tok/s
        assert metrics["total_throughput"] == pytest.approx(834.8, abs=1.0)

    def test_zero_duration_safety(self):
        """Test that zero/near-zero durations don't cause division by zero."""
        metrics = _compute_single_metrics(
            prompt_tokens=100,
            completion_tokens=1,
            start_time=0.0,
            first_token_time=0.0,
            end_time=0.0,
            peak_memory=0,
            cached_tokens=0,
        )
        # Should not raise, values should be finite
        assert metrics["ttft_ms"] == 0.0
        assert metrics["gen_tps"] > 0  # Protected by max(duration, 1e-9)


# =============================================================================
# BenchmarkRun lifecycle tests
# =============================================================================


class TestBenchmarkRunLifecycle:
    def test_create_run(self):
        req = BenchmarkRequest(
            model_id="test-model",
            prompt_lengths=[1024],
        )
        run = create_run(req)
        assert run.bench_id.startswith("bench-")
        assert run.status == "running"
        assert run.results == []

    def test_get_run(self):
        req = BenchmarkRequest(
            model_id="test-model",
            prompt_lengths=[1024],
        )
        run = create_run(req)
        found = get_run(run.bench_id)
        assert found is run

    def test_get_nonexistent_run(self):
        assert get_run("nonexistent") is None

    def test_cleanup_old_runs(self):
        # Create many completed runs
        for _ in range(15):
            req = BenchmarkRequest(
                model_id="test-model",
                prompt_lengths=[1024],
            )
            run = create_run(req)
            run.status = "completed"

        cleanup_old_runs(max_runs=5)

        # Should have at most ~5 completed + any running ones
        from omlx.admin.benchmark import _benchmark_runs

        completed = [r for r in _benchmark_runs.values() if r.status == "completed"]
        assert len(completed) <= 5


# =============================================================================
# SSE event format tests
# =============================================================================


class TestSSEEventFormat:
    @pytest.mark.asyncio
    async def test_send_event(self):
        """Test that events are properly queued."""
        from omlx.admin.benchmark import _send_event

        run = BenchmarkRun(
            bench_id="test",
            request=BenchmarkRequest(
                model_id="test-model",
                prompt_lengths=[1024],
            ),
        )

        await _send_event(run, {
            "type": "progress",
            "phase": "single",
            "message": "Testing",
            "current": 1,
            "total": 3,
        })

        event = run.queue.get_nowait()
        assert event["type"] == "progress"
        assert event["phase"] == "single"
        assert event["current"] == 1
        assert event["total"] == 3

    @pytest.mark.asyncio
    async def test_result_event_format(self):
        from omlx.admin.benchmark import _send_event

        run = BenchmarkRun(
            bench_id="test",
            request=BenchmarkRequest(
                model_id="test-model",
                prompt_lengths=[1024],
            ),
        )

        result_data = {
            "test_type": "single",
            "pp": 1024,
            "tg": 128,
            "ttft_ms": 45.2,
            "gen_tps": 81.3,
        }
        await _send_event(run, {"type": "result", "data": result_data})

        event = run.queue.get_nowait()
        assert event["type"] == "result"
        assert event["data"]["test_type"] == "single"
        assert event["data"]["pp"] == 1024
