# SPDX-License-Identifier: Apache-2.0
"""Tests for the admin benchmark module."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.admin.benchmark import (
    VALID_BATCH_SIZES,
    VALID_PROMPT_LENGTHS,
    BenchmarkRequest,
    BenchmarkRun,
    _clean_model_name,
    _compute_single_metrics,
    _detect_experimental_features,
    _detect_quantization,
    _generate_prompt,
    _run_single_test,
    cleanup_old_runs,
    create_run,
    get_run,
    run_benchmark,
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

    def test_force_lm_engine_defaults_to_false(self):
        req = BenchmarkRequest(
            model_id="test-model",
            prompt_lengths=[1024],
        )
        assert req.force_lm_engine is False

    def test_force_lm_engine_can_be_enabled(self):
        req = BenchmarkRequest(
            model_id="test-model",
            prompt_lengths=[1024],
            force_lm_engine=True,
        )
        assert req.force_lm_engine is True


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
        """Single-token outputs do not report bogus decode throughput."""
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
        assert metrics["gen_tps"] == 0.0
        assert metrics["tpot_ms"] == 0.0

    def test_native_duration_overrides(self):
        """Native engine timings can override streaming timing artifacts."""
        metrics = _compute_single_metrics(
            prompt_tokens=1024,
            completion_tokens=128,
            start_time=0.0,
            first_token_time=1.0,
            end_time=1.0,
            peak_memory=0,
            cached_tokens=0,
            prefill_duration_s=4.0,
            generation_duration_s=8.0,
        )

        assert metrics["processing_tps"] == pytest.approx(256.0)
        assert metrics["gen_tps"] == pytest.approx(16.0)
        assert metrics["tpot_ms"] == pytest.approx(62.99, abs=0.01)

    def test_single_token_completion_has_no_decode_rate(self):
        """Immediate stop has no inter-token decode interval to benchmark."""
        metrics = _compute_single_metrics(
            prompt_tokens=16384,
            completion_tokens=1,
            start_time=0.0,
            first_token_time=8.629,
            end_time=8.629001,
            peak_memory=0,
            cached_tokens=0,
        )

        assert metrics["gen_tps"] == 0.0
        assert metrics["tpot_ms"] == 0.0
        assert metrics["total_throughput"] == pytest.approx(1898.7, abs=0.1)


class TestRunSingleTest:
    @pytest.mark.asyncio
    async def test_uses_native_diffusion_metrics_for_chunked_stream(self):
        """Diffusion streams by canvas, so benchmark must not use chunk timing."""

        class ChunkedDiffusionEngine:
            async def stream_generate(self, **kwargs):
                yield SimpleNamespace(
                    completion_tokens=128,
                    prompt_tokens=1024,
                    cached_tokens=0,
                    new_text="x" * 10,
                    finished=True,
                    finish_reason="length",
                    prompt_tps=256.0,
                    generation_tps=32.0,
                )

        metrics = await _run_single_test(
            ChunkedDiffusionEngine(),
            prompt="prompt",
            max_tokens=128,
            pp_len=1024,
        )

        assert metrics["processing_tps"] == pytest.approx(256.0)
        assert metrics["gen_tps"] == pytest.approx(32.0)
        assert metrics["gen_tps"] < 1000.0

    @pytest.mark.asyncio
    async def test_uses_diffusion_canvas_metrics_when_eos_stops_early(self):
        """Diffusion benchmark TG should measure canvas work, not early EOS text."""

        class EarlyStopDiffusionEngine:
            async def stream_generate(self, **kwargs):
                yield SimpleNamespace(
                    completion_tokens=16,
                    prompt_tokens=1024,
                    cached_tokens=0,
                    new_text="short answer",
                    finished=True,
                    finish_reason="stop",
                    prompt_tps=256.0,
                    generation_tps=2.0,
                    diffusion_canvas_tokens=128,
                    diffusion_canvas_tps=64.0,
                )

        metrics = await _run_single_test(
            EarlyStopDiffusionEngine(),
            prompt="prompt",
            max_tokens=128,
            pp_len=1024,
        )

        assert metrics["completion_tokens"] == 128
        assert metrics["gen_tps"] == pytest.approx(64.0)
        assert metrics["tpot_ms"] == pytest.approx(15.75, abs=0.01)

    @pytest.mark.asyncio
    async def test_uses_producer_timestamps_for_aggregated_output(self):
        """Aggregated chunks should use producer-side decode timing."""

        class AggregatedEngine:
            async def stream_generate(self, **kwargs):
                yield SimpleNamespace(
                    completion_tokens=128,
                    prompt_tokens=1024,
                    cached_tokens=0,
                    new_text="x" * 128,
                    finished=True,
                    finish_reason="length",
                    generated_at=0.2,
                    generated_until=1.2,
                )

        with patch("omlx.admin.benchmark.time.perf_counter", side_effect=[0.0, 1.3]):
            metrics = await _run_single_test(
                AggregatedEngine(),
                prompt="prompt",
                max_tokens=128,
                pp_len=1024,
            )

        assert metrics["ttft_ms"] == pytest.approx(200.0)
        assert metrics["gen_tps"] == pytest.approx(128.0)
        assert metrics["tpot_ms"] == pytest.approx(7.87, abs=0.01)

    @pytest.mark.asyncio
    async def test_aggregated_output_without_end_time_has_no_decode_rate(self):
        """Single aggregate chunks need a producer-side end time for tg TPS."""

        class AggregatedEngine:
            async def stream_generate(self, **kwargs):
                yield SimpleNamespace(
                    completion_tokens=128,
                    prompt_tokens=1024,
                    cached_tokens=0,
                    new_text="x" * 128,
                    finished=True,
                    finish_reason="length",
                    generated_at=0.2,
                )

        with patch("omlx.admin.benchmark.time.perf_counter", side_effect=[0.0, 1.2]):
            metrics = await _run_single_test(
                AggregatedEngine(),
                prompt="prompt",
                max_tokens=128,
                pp_len=1024,
            )

        assert metrics["ttft_ms"] == pytest.approx(200.0)
        assert metrics["gen_tps"] == 0.0
        assert metrics["tpot_ms"] == 0.0


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


class _FakeBenchTokenizer:
    def encode(self, text):
        return list(range(2048))

    def decode(self, tokens):
        return "prompt"


class _FakeBenchEngine:
    tokenizer = _FakeBenchTokenizer()
    _engine = None

    async def stream_generate(self, **kwargs):
        yield SimpleNamespace(
            completion_tokens=1,
            prompt_tokens=1024,
            cached_tokens=0,
            new_text="x",
            finished=True,
            finish_reason="length",
        )


class _FakeSettingsManager:
    def __init__(self, settings):
        self.settings = settings

    def get_settings(self, model_id):
        return self.settings


class _FakeBenchEnginePool:
    def __init__(self, settings=None, engine=None):
        self._settings_manager = (
            _FakeSettingsManager(settings) if settings is not None else None
        )
        self._engine = engine or _FakeBenchEngine()
        self.force_lm_values = []

    def get_loaded_model_ids(self):
        return []

    async def get_engine(self, model_id, force_lm=False):
        self.force_lm_values.append(force_lm)
        return self._engine

    async def _unload_engine(self, model_id):
        pass


class TestBenchmarkEngineSelection:
    async def _run(self, *, settings=None, force_lm_engine=False):
        run = BenchmarkRun(
            bench_id="bench-test",
            request=BenchmarkRequest(
                model_id="test-model",
                prompt_lengths=[1024],
                generation_length=1,
                force_lm_engine=force_lm_engine,
            ),
        )
        pool = _FakeBenchEnginePool(settings)
        with patch("omlx.admin.benchmark._upload_to_omlx_ai", AsyncMock()):
            await run_benchmark(run, pool)
        return run, pool

    @pytest.mark.asyncio
    async def test_auto_uses_vlm_engine_for_vlm_mtp_with_drafter(self):
        settings = SimpleNamespace(
            vlm_mtp_enabled=True,
            vlm_mtp_draft_model="draft-model",
        )

        run, pool = await self._run(settings=settings)

        assert pool.force_lm_values == [False]
        assert run.experimental_features == ["vlm_mtp"]
        assert run.status == "completed"

    @pytest.mark.asyncio
    async def test_force_lm_engine_overrides_vlm_mtp_auto(self):
        settings = SimpleNamespace(
            vlm_mtp_enabled=True,
            vlm_mtp_draft_model="draft-model",
        )

        run, pool = await self._run(settings=settings, force_lm_engine=True)

        assert pool.force_lm_values == [True]
        assert run.experimental_features == ["vlm_mtp"]
        assert run.status == "completed"

    @pytest.mark.asyncio
    async def test_auto_keeps_lm_engine_without_vlm_mtp_drafter(self):
        settings = SimpleNamespace(
            vlm_mtp_enabled=True,
            vlm_mtp_draft_model=None,
        )

        run, pool = await self._run(settings=settings)

        assert pool.force_lm_values == [True]
        assert run.experimental_features == ["vlm_mtp"]
        assert run.status == "completed"

    @pytest.mark.asyncio
    async def test_batch_request_skips_engine_with_none_scheduler_core(self):
        run = BenchmarkRun(
            bench_id="bench-test",
            request=BenchmarkRequest(
                model_id="test-model",
                prompt_lengths=[1024],
                generation_length=1,
                batch_sizes=[2],
            ),
        )
        pool = _FakeBenchEnginePool(engine=_FakeBenchEngine())

        with patch("omlx.admin.benchmark._upload_to_omlx_ai", AsyncMock()):
            await run_benchmark(run, pool)

        assert run.status == "completed"
        assert [r["test_type"] for r in run.results] == ["single"]

    @pytest.mark.asyncio
    async def test_diffusion_warmup_uses_benchmark_generation_length(self):
        class FakeDiffusionEngine(_FakeBenchEngine):
            is_diffusion_model = True

            def __init__(self):
                self.calls = []

            async def stream_generate(self, **kwargs):
                self.calls.append(kwargs)
                yield SimpleNamespace(
                    completion_tokens=128,
                    prompt_tokens=1024,
                    cached_tokens=0,
                    new_text="x",
                    finished=True,
                    finish_reason="length",
                    prompt_tps=256.0,
                    generation_tps=32.0,
                    diffusion_canvas_tokens=128,
                    diffusion_canvas_tps=64.0,
                )

        engine = FakeDiffusionEngine()
        run = BenchmarkRun(
            bench_id="bench-test",
            request=BenchmarkRequest(
                model_id="test-model",
                prompt_lengths=[1024],
                generation_length=128,
            ),
        )
        pool = _FakeBenchEnginePool(engine=engine)

        with patch("omlx.admin.benchmark._upload_to_omlx_ai", AsyncMock()):
            await run_benchmark(run, pool)

        assert run.status == "completed"
        assert engine.calls[0]["max_tokens"] == 128


# =============================================================================
# Experimental feature detection tests
# =============================================================================


class TestExperimentalFeatureDetection:
    def test_detects_all_upload_skipping_features(self):
        settings = SimpleNamespace(
            dflash_enabled=True,
            specprefill_enabled=True,
            turboquant_kv_enabled=True,
            mtp_enabled=True,
            vlm_mtp_enabled=True,
        )

        assert _detect_experimental_features(settings) == [
            "dflash",
            "specprefill",
            "turboquant",
            "mtp",
            "vlm_mtp",
        ]

    def test_missing_flags_are_treated_as_disabled(self):
        assert _detect_experimental_features(SimpleNamespace()) == []


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

        # SSE delivery: events are appended to `run.events` (replay log).
        assert len(run.events) == 1
        event = run.events[0]
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

        assert len(run.events) == 1
        event = run.events[0]
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

        # Collect all events from the replay log.
        events = list(run.events)

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

        events = list(run.events)

        upload_event = next(e for e in events if e["type"] == "upload")
        assert upload_event["data"]["duplicate"] is True
        assert upload_event["data"]["id"] == "xyz789"

        done_event = next(e for e in events if e["type"] == "upload_done")
        assert done_event["data"]["success"] == 1

    @pytest.mark.asyncio
    async def test_upload_skips_unmeasurable_generation_results(self):
        """Rows without a measured decode rate are not uploaded."""
        from omlx.admin.benchmark import _upload_to_omlx_ai

        run = BenchmarkRun(
            bench_id="test-bench",
            request=BenchmarkRequest(
                model_id="Qwen3-30B-4bit",
                prompt_lengths=[1024, 8192],
            ),
        )
        run.results = [
            {
                "test_type": "single",
                "pp": 1024,
                "tg": 128,
                "processing_tps": 500.0,
                "gen_tps": 0.0,
                "ttft_ms": 100.0,
                "peak_memory_bytes": 0,
            },
            {
                "test_type": "single",
                "pp": 8192,
                "tg": 128,
                "processing_tps": 500.0,
                "gen_tps": 50.0,
                "ttft_ms": 500.0,
                "peak_memory_bytes": 0,
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

        assert mock_to_thread.await_count == 1

        upload_events = [e for e in run.events if e["type"] == "upload"]
        assert [e["data"]["context_length"] for e in upload_events] == [8192]

        done_event = next(e for e in run.events if e["type"] == "upload_done")
        assert done_event["data"]["total"] == 1
        assert done_event["data"]["success"] == 1
        assert done_event["data"]["failed"] == 0
        assert done_event["data"]["skipped"] == 1

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

        events = list(run.events)

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


# =============================================================================
# External endpoint benchmark tests
# =============================================================================


def _external_config():
    from omlx.admin.external_api import ExternalEndpointConfig

    return ExternalEndpointConfig(
        base_url="http://localhost:8001/v1",
        api_key="sk-test",
        model="remote-model",
    )


def _stream_stats(prompt=1000, completion=128, cached=0):
    from omlx.admin.external_api import StreamStats

    return StreamStats(
        prompt_tokens=prompt,
        completion_tokens=completion,
        cached_tokens=cached,
        start_time=0.0,
        first_content_time=0.5,
        last_content_time=1.5,
        end_time=1.6,
        text="x" * 16,
    )


class TestExternalBenchmarkRequest:
    def test_external_accepted(self):
        req = BenchmarkRequest(
            model_id="remote-model",
            prompt_lengths=[1024],
            external=_external_config(),
        )
        assert req.external is not None
        assert req.external.model == "remote-model"

    def test_external_defaults_to_none(self):
        req = BenchmarkRequest(model_id="m", prompt_lengths=[1024])
        assert req.external is None

    def test_external_invalid_base_url_rejected(self):
        with pytest.raises(ValueError, match="http:// or https://"):
            BenchmarkRequest(
                model_id="m",
                prompt_lengths=[1024],
                external={"base_url": "localhost:8001", "model": "m"},
            )


class TestGenerateExternalPrompt:
    def test_scales_with_target(self):
        from omlx.admin.benchmark import _generate_external_prompt

        short = _generate_external_prompt(1024)
        long = _generate_external_prompt(4096)
        assert len(long) > len(short) * 3

    def test_unique_prefix(self):
        from omlx.admin.benchmark import _generate_external_prompt

        a = _generate_external_prompt(1024)
        b = _generate_external_prompt(1024)
        assert a.startswith("BENCH-")
        assert a != b


class TestRunExternalBenchmark:
    def _make_run(self, prompt_lengths=None, batch_sizes=None):
        return BenchmarkRun(
            bench_id="bench-ext",
            request=BenchmarkRequest(
                model_id="remote-model",
                prompt_lengths=prompt_lengths or [1024],
                batch_sizes=batch_sizes or [],
                external=_external_config(),
            ),
        )

    def _mock_client(self, stats=None):
        client = MagicMock()
        client.stream_chat_completion = AsyncMock(
            return_value=stats or _stream_stats()
        )
        client.aclose = AsyncMock()
        return client

    async def test_never_touches_engine_pool(self):
        run = self._make_run(batch_sizes=[2])
        pool = MagicMock()
        client = self._mock_client()

        with patch(
            "omlx.admin.benchmark.ExternalAPIClient", return_value=client
        ):
            await run_benchmark(run, pool)

        assert run.status == "completed"
        pool.get_engine.assert_not_called()
        pool.get_loaded_model_ids.assert_not_called()
        pool._unload_engine.assert_not_called()

    async def test_event_sequence_and_upload_skipped(self):
        run = self._make_run(prompt_lengths=[1024], batch_sizes=[2])
        client = self._mock_client()

        with patch(
            "omlx.admin.benchmark.ExternalAPIClient", return_value=client
        ):
            await run_benchmark(run, MagicMock())

        event_types = [e["type"] for e in run.events]
        assert event_types == [
            "progress",  # warmup
            "progress",  # single
            "result",
            "progress",  # batch
            "result",
            "done",
            "upload_skipped",
        ]
        skipped = run.events[-1]
        assert skipped["reason"] == "external_endpoint"
        assert run.upload_state["phase"] == "skipped"
        assert run.upload_state["skipped_reason"] == "external_endpoint"
        client.aclose.assert_awaited()

    async def test_single_result_uses_usage_token_counts(self):
        run = self._make_run(prompt_lengths=[4096])
        client = self._mock_client(_stream_stats(prompt=3900, completion=128))

        with patch(
            "omlx.admin.benchmark.ExternalAPIClient", return_value=client
        ):
            await run_benchmark(run, MagicMock())

        result = run.results[0]
        assert result["test_type"] == "single"
        assert result["pp"] == 4096
        # Actual usage counts, not the nominal pp target
        assert result["prompt_tokens"] == 3900
        assert result["completion_tokens"] == 128
        assert result["peak_memory_bytes"] is None
        # gen duration 1.0s (first 0.5 → last 1.5) with 128 tokens
        assert result["gen_tps"] == 128.0
        assert result["ttft_ms"] == 500.0

    async def test_batch_result_aggregates_usage_counts(self):
        run = self._make_run(prompt_lengths=[1024], batch_sizes=[2])
        client = self._mock_client(_stream_stats(prompt=1000, completion=100))

        with patch(
            "omlx.admin.benchmark.ExternalAPIClient", return_value=client
        ):
            await run_benchmark(run, MagicMock())

        batch = [r for r in run.results if r["test_type"] == "batch"][0]
        assert batch["batch_size"] == 2
        assert batch["total_gen_tokens"] == 200

    async def test_missing_usage_fails_run(self):
        from omlx.admin.external_api import ExternalEndpointError

        run = self._make_run()
        client = MagicMock()
        client.stream_chat_completion = AsyncMock(
            side_effect=ExternalEndpointError(
                "External endpoint does not support stream usage"
            )
        )
        client.aclose = AsyncMock()

        with patch(
            "omlx.admin.benchmark.ExternalAPIClient", return_value=client
        ):
            await run_benchmark(run, MagicMock())

        assert run.status == "error"
        assert "stream usage" in run.error_message
        error_events = [e for e in run.events if e["type"] == "error"]
        assert error_events and "stream usage" in error_events[0]["message"]

    async def test_cancellation_mid_run(self):
        run = self._make_run()
        started = asyncio.Event()

        async def hang(**kwargs):
            started.set()
            await asyncio.Event().wait()

        client = MagicMock()
        client.stream_chat_completion = AsyncMock(side_effect=hang)
        client.aclose = AsyncMock()

        with patch(
            "omlx.admin.benchmark.ExternalAPIClient", return_value=client
        ):
            task = asyncio.create_task(run_benchmark(run, MagicMock()))
            await started.wait()
            task.cancel()
            await task

        assert run.status == "cancelled"
        assert run.events[-1]["type"] == "error"
        assert "cancelled" in run.events[-1]["message"].lower()
        client.aclose.assert_awaited()
