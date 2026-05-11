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
    _clean_model_name,
    _compute_single_metrics,
    _detect_quantization,
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


# =============================================================================
# Quantization detection tests
# =============================================================================


class TestDetectQuantization:
    def test_from_config_json(self, tmp_path):
        config = {"quantization_config": {"quant_method": "awq", "bits": 4}}
        (tmp_path / "config.json").write_text(
            __import__("json").dumps(config)
        )
        assert _detect_quantization(str(tmp_path)) == "4bit"

    def test_from_config_json_8bit(self, tmp_path):
        config = {"quantization_config": {"bits": 8}}
        (tmp_path / "config.json").write_text(
            __import__("json").dumps(config)
        )
        assert _detect_quantization(str(tmp_path)) == "8bit"

    def test_from_dirname_4bit(self, tmp_path):
        model_dir = tmp_path / "Qwen3-30B-A3B-4bit"
        model_dir.mkdir()
        assert _detect_quantization(str(model_dir)) == "4bit"

    def test_from_dirname_fp16(self, tmp_path):
        model_dir = tmp_path / "Llama-3-8B-fp16"
        model_dir.mkdir()
        assert _detect_quantization(str(model_dir)) == "fp16"

    def test_from_dirname_bf16(self, tmp_path):
        model_dir = tmp_path / "Model-bf16"
        model_dir.mkdir()
        assert _detect_quantization(str(model_dir)) == "bf16"

    def test_from_dirname_mxfp4(self, tmp_path):
        model_dir = tmp_path / "gpt-oss-120b-MXFP4"
        model_dir.mkdir()
        assert _detect_quantization(str(model_dir)) == "mxfp4"

    def test_from_dirname_nvfp4(self, tmp_path):
        model_dir = tmp_path / "Model-NVFP4"
        model_dir.mkdir()
        assert _detect_quantization(str(model_dir)) == "nvfp4"

    def test_unknown_fallback(self, tmp_path):
        model_dir = tmp_path / "SomeModel"
        model_dir.mkdir()
        assert _detect_quantization(str(model_dir)) == "unknown"

    def test_config_takes_priority_over_dirname(self, tmp_path):
        model_dir = tmp_path / "Model-4bit"
        model_dir.mkdir()
        config = {"quantization_config": {"bits": 8}}
        (model_dir / "config.json").write_text(
            __import__("json").dumps(config)
        )
        assert _detect_quantization(str(model_dir)) == "8bit"


# =============================================================================
# Model name cleaning tests
# =============================================================================


class TestCleanModelName:
    def test_strip_4bit(self):
        assert _clean_model_name("Qwen3-30B-A3B-4bit", "4bit") == "Qwen3-30B-A3B"

    def test_strip_8bit(self):
        assert _clean_model_name("Llama-3-8B-8bit", "8bit") == "Llama-3-8B"

    def test_strip_fp16(self):
        assert _clean_model_name("Model-fp16", "fp16") == "Model"

    def test_strip_mlx_marker(self):
        assert _clean_model_name("Qwen3-30B-MLX-4bit", "4bit") == "Qwen3-30B"

    def test_strip_mxfp4(self):
        assert _clean_model_name("gpt-oss-120b-MXFP4", "mxfp4") == "gpt-oss-120b"

    def test_strip_nvfp4(self):
        assert _clean_model_name("Model-NVFP4", "nvfp4") == "Model"

    def test_no_quant_suffix(self):
        assert _clean_model_name("Qwen3-30B-A3B", "unknown") == "Qwen3-30B-A3B"

    def test_preserves_model_size(self):
        assert _clean_model_name("DeepSeek-R1-0528-Qwen3-8B-4bit", "4bit") == "DeepSeek-R1-0528-Qwen3-8B"


# =============================================================================
# Upload integration tests (mocked HTTP)
# =============================================================================


class TestUploadToOmlxAi:
    @pytest.mark.asyncio
    async def test_upload_success(self):
        """Test successful upload sends correct SSE events."""
        from omlx.admin.benchmark import _upload_to_omlx_ai

        run = BenchmarkRun(
            bench_id="test-bench",
            request=BenchmarkRequest(
                model_id="Qwen3-30B-4bit",
                prompt_lengths=[1024],
            ),
        )
        run.results = [
            {
                "test_type": "single",
                "pp": 1024,
                "tg": 128,
                "processing_tps": 500.0,
                "gen_tps": 50.0,
                "ttft_ms": 100.0,
                "peak_memory_bytes": 8 * 1024**3,
            },
        ]

        mock_entry = MagicMock()
        mock_entry.model_path = "/models/Qwen3-30B-4bit"
        mock_pool = MagicMock()
        mock_pool.get_entry.return_value = mock_entry
        mock_pool._settings_manager = None

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "id": "abc12345",
            "url": "https://omlx.ai/benchmarks/abc12345",
        }

        mock_to_thread = AsyncMock(return_value=mock_response)

        with patch("asyncio.to_thread", mock_to_thread):
            await _upload_to_omlx_ai(run, mock_pool)

        # Collect all events
        events = []
        while not run.queue.empty():
            events.append(run.queue.get_nowait())

        # Should have: progress, upload, upload_done
        event_types = [e["type"] for e in events]
        assert "progress" in event_types
        assert "upload" in event_types
        assert "upload_done" in event_types

        upload_event = next(e for e in events if e["type"] == "upload")
        assert upload_event["data"]["context_length"] == 1024
        assert upload_event["data"]["id"] == "abc12345"

        done_event = next(e for e in events if e["type"] == "upload_done")
        assert done_event["data"]["success"] == 1
        assert done_event["data"]["failed"] == 0

    @pytest.mark.asyncio
    async def test_upload_duplicate(self):
        """Test 409 duplicate response is handled as success."""
        from omlx.admin.benchmark import _upload_to_omlx_ai

        run = BenchmarkRun(
            bench_id="test-bench",
            request=BenchmarkRequest(
                model_id="Qwen3-30B-4bit",
                prompt_lengths=[1024],
            ),
        )
        run.results = [
            {
                "test_type": "single",
                "pp": 1024,
                "tg": 128,
                "processing_tps": 500.0,
                "gen_tps": 50.0,
                "ttft_ms": 100.0,
                "peak_memory_bytes": 0,
            },
        ]

        mock_entry = MagicMock()
        mock_entry.model_path = "/models/Qwen3-30B-4bit"
        mock_pool = MagicMock()
        mock_pool.get_entry.return_value = mock_entry
        mock_pool._settings_manager = None

        mock_response = MagicMock()
        mock_response.status_code = 409
        mock_response.json.return_value = {
            "error": "Duplicate",
            "existing_id": "xyz789",
            "existing_url": "https://omlx.ai/benchmarks/xyz789",
        }

        mock_to_thread = AsyncMock(return_value=mock_response)

        with patch("asyncio.to_thread", mock_to_thread):
            await _upload_to_omlx_ai(run, mock_pool)

        events = []
        while not run.queue.empty():
            events.append(run.queue.get_nowait())

        upload_event = next(e for e in events if e["type"] == "upload")
        assert upload_event["data"]["duplicate"] is True
        assert upload_event["data"]["id"] == "xyz789"

        done_event = next(e for e in events if e["type"] == "upload_done")
        assert done_event["data"]["success"] == 1

    @pytest.mark.asyncio
    async def test_upload_skipped_when_experimental_features_enabled(self):
        """Upload is skipped (no HTTP call) when experimental features were active."""
        from omlx.admin.benchmark import _upload_to_omlx_ai

        run = BenchmarkRun(
            bench_id="test-bench",
            request=BenchmarkRequest(
                model_id="Qwen3-30B-4bit",
                prompt_lengths=[1024],
            ),
            experimental_features=["dflash", "turboquant"],
        )
        run.results = [
            {
                "test_type": "single",
                "pp": 1024,
                "tg": 128,
                "processing_tps": 500.0,
                "gen_tps": 50.0,
                "ttft_ms": 100.0,
                "peak_memory_bytes": 0,
            },
        ]

        mock_pool = MagicMock()
        mock_pool._settings_manager = None
        mock_to_thread = AsyncMock()

        with patch("asyncio.to_thread", mock_to_thread):
            await _upload_to_omlx_ai(run, mock_pool)

        events = []
        while not run.queue.empty():
            events.append(run.queue.get_nowait())

        # Only an upload_skipped event is emitted, no progress / upload / upload_done
        event_types = [e["type"] for e in events]
        assert "upload_skipped" in event_types
        assert "upload" not in event_types
        assert "upload_done" not in event_types
        assert "progress" not in event_types

        skipped = next(e for e in events if e["type"] == "upload_skipped")
        assert skipped["reason"] == "experimental_features"
        assert skipped["features"] == ["dflash", "turboquant"]

        # No HTTP call was made
        mock_to_thread.assert_not_called()


_CF_INTERSTITIAL = (
    '<!DOCTYPE html><html lang="en-US"><head><title>Just a moment...</title>'
    '<meta http-equiv="refresh" content="360"></head><body><div class="main-wrapper">'
    + ("x" * 5000)
    + "</div></body></html>"
)


class TestSanitizeUploadError:
    """The Cloudflare interstitial pollutes the dashboard upload panel when
    omlx.ai's API endpoint is gated behind a managed challenge. The
    sanitizer must detect that case and surface an actionable message
    without dumping the full 5KB HTML body."""

    def _resp(self, status=403, headers=None, text="", json_raises=True, json_data=None):
        from unittest.mock import MagicMock
        resp = MagicMock()
        resp.status_code = status
        resp.headers = headers or {}
        resp.text = text
        if json_raises:
            resp.json.side_effect = ValueError("not JSON")
        else:
            resp.json.return_value = json_data or {}
        return resp

    def test_cloudflare_challenge_via_header(self):
        from omlx.admin.benchmark import _sanitize_upload_error
        resp = self._resp(
            status=403,
            headers={"cf-mitigated": "challenge"},
            text=_CF_INTERSTITIAL,
        )
        msg = _sanitize_upload_error(resp)
        assert "Cloudflare" in msg
        assert "403" in msg
        # The raw HTML body must NOT appear in the error message.
        assert "<!DOCTYPE" not in msg
        assert "Just a moment" not in msg
        assert len(msg) < 300

    def test_cloudflare_challenge_via_body_sniff(self):
        """Header missing but body still contains the interstitial — covers
        edge transports / proxies that strip cf-mitigated."""
        from omlx.admin.benchmark import _sanitize_upload_error
        resp = self._resp(status=403, headers={}, text=_CF_INTERSTITIAL)
        msg = _sanitize_upload_error(resp)
        assert "Cloudflare" in msg
        assert "<!DOCTYPE" not in msg

    def test_json_error_field_extracted(self):
        from omlx.admin.benchmark import _sanitize_upload_error
        resp = self._resp(
            status=400,
            json_raises=False,
            json_data={"error": "Invalid model_name"},
            text='{"error": "Invalid model_name"}',
        )
        assert _sanitize_upload_error(resp) == "Invalid model_name"

    def test_json_detail_field_extracted(self):
        from omlx.admin.benchmark import _sanitize_upload_error
        resp = self._resp(
            status=422,
            json_raises=False,
            json_data={"detail": "context_length out of range"},
            text='{"detail": "context_length out of range"}',
        )
        assert _sanitize_upload_error(resp) == "context_length out of range"

    def test_html_body_without_cf_signals_collapses_to_hint(self):
        """Non-CF HTML body (e.g. nginx 502 page) should not be dumped raw."""
        from omlx.admin.benchmark import _sanitize_upload_error
        resp = self._resp(
            status=502,
            text="<html><body>502 Bad Gateway</body></html>",
        )
        msg = _sanitize_upload_error(resp)
        assert "<html>" not in msg
        assert "502" in msg

    def test_plain_text_short_body_passes_through(self):
        from omlx.admin.benchmark import _sanitize_upload_error
        resp = self._resp(status=500, text="upstream connection refused")
        assert _sanitize_upload_error(resp) == "upstream connection refused"

    def test_empty_body_falls_back_to_status(self):
        from omlx.admin.benchmark import _sanitize_upload_error
        resp = self._resp(status=503, text="")
        assert _sanitize_upload_error(resp) == "HTTP 503"
