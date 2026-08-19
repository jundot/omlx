# SPDX-License-Identifier: Apache-2.0
"""Tests for the admin benchmark module."""

import asyncio
import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.admin.benchmark import (
    BENCHMARK_CONTEXT_PROFILES,
    BenchmarkContextProfile,
    BenchmarkRequest,
    BenchmarkRun,
    BenchmarkWarmupMode,
    _compute_single_metrics,
    _derive_feature_flags,
    _detect_experimental_features,
    _detect_quantization,
    _filter_uploaded_settings,
    _generate_prompt,
    _load_bench_corpus,
    _log_ane_benchmark_trace,
    _run_batch_test,
    _run_single_test,
    _upload_model_name,
    _upload_model_repo,
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

    def test_context_profile_defaults_to_python_code(self):
        req = BenchmarkRequest(model_id="test-model", prompt_lengths=[1024])
        assert req.context_profile is BenchmarkContextProfile.CODE_PYTHON

    def test_warmup_mode_defaults_to_quick(self):
        req = BenchmarkRequest(model_id="test-model", prompt_lengths=[1024])
        assert req.warmup_mode is BenchmarkWarmupMode.QUICK

    def test_ane_warmup_mode_is_accepted(self):
        req = BenchmarkRequest(
            model_id="test-model",
            prompt_lengths=[1024],
            warmup_mode="ane_2048",
        )
        assert req.warmup_mode is BenchmarkWarmupMode.ANE_2048

    def test_ane_aligned_prompt_adds_one_to_single_trials(self):
        from omlx.admin.benchmark import _single_prompt_lengths

        req = BenchmarkRequest(
            model_id="test-model",
            prompt_lengths=[4096, 8192],
            align_prompt_to_ane=True,
        )

        assert _single_prompt_lengths(req) == [4097, 8193]

    def test_standard_prompt_lengths_remain_unchanged(self):
        from omlx.admin.benchmark import _single_prompt_lengths

        req = BenchmarkRequest(
            model_id="test-model",
            prompt_lengths=[4096, 8192],
        )

        assert _single_prompt_lengths(req) == [4096, 8192]

    def test_unknown_warmup_mode_is_rejected(self):
        with pytest.raises(ValueError, match="warmup_mode"):
            BenchmarkRequest(
                model_id="test-model",
                prompt_lengths=[1024],
                warmup_mode="unknown",
            )

    @pytest.mark.parametrize("profile", list(BenchmarkContextProfile))
    def test_all_context_profiles_are_accepted(self, profile):
        req = BenchmarkRequest(
            model_id="test-model",
            prompt_lengths=[1024],
            context_profile=profile.value,
        )
        assert req.context_profile is profile

    def test_unknown_context_profile_is_rejected(self):
        with pytest.raises(ValueError, match="context_profile"):
            BenchmarkRequest(
                model_id="test-model",
                prompt_lengths=[1024],
                context_profile="unknown",
            )


# =============================================================================
# Prompt generation tests
# =============================================================================


class TestGeneratePrompt:
    def test_exact_token_count(self):
        """Prompt IDs are exact without a decode/re-encode round trip."""
        tokenizer = MagicMock()

        # Simulate tokenizer behavior
        def mock_encode(text):
            # Return roughly 1 token per 4 chars
            return list(range(len(text) // 4))

        tokenizer.encode = mock_encode

        prompt = _generate_prompt(tokenizer, 1024)

        assert isinstance(prompt, list)
        assert len(prompt) == 1024
        tokenizer.decode.assert_not_called()

    def test_rejects_non_positive_target(self):
        with pytest.raises(ValueError, match="must be positive"):
            _generate_prompt(MagicMock(), 0)

    def test_uuid_prefix_uniqueness(self):
        """Verify each generated prompt has a unique UUID prefix."""
        tokenizer = _WordTokenizer()

        prompts = set()
        for _ in range(10):
            prompt = _generate_prompt(tokenizer, 100)
            prompts.add(tuple(prompt))

        assert len(prompts) == 10


class _WordTokenizer:
    """Word-level round-trip tokenizer for content-sensitive prompt tests."""

    def __init__(self):
        self.vocab = {}
        self.words = []

    def encode(self, text):
        ids = []
        for word in text.split(" "):
            if word not in self.vocab:
                self.vocab[word] = len(self.words)
                self.words.append(word)
            ids.append(self.vocab[word])
        return ids

    def decode(self, ids):
        return " ".join(self.words[i] for i in ids)


class TestPromptRealism:
    @pytest.mark.parametrize("profile", list(BenchmarkContextProfile))
    def test_all_corpus_files_ship_with_long_form_content(self, profile):
        corpus = _load_bench_corpus(profile)
        assert len(corpus) > 400_000
        assert "quick brown fox jumps" not in corpus

    def test_english_corpus_keeps_moby_dick_content(self):
        corpus = _load_bench_corpus(BenchmarkContextProfile.NOVEL_EN)
        assert corpus.startswith("Call me Ishmael.")

    def test_manifest_records_representative_250k_token_headroom(self):
        manifest_path = (
            Path(__file__).parents[1]
            / "omlx"
            / "admin"
            / "bench_corpora"
            / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for counts in manifest["representative_token_counts"].values():
            assert set(counts) == {
                spec.filename for spec in BENCHMARK_CONTEXT_PROFILES.values()
            }
            assert min(counts.values()) >= 250_000

    def test_prompt_is_not_repetitive(self):
        """Speculative decoders (MTP/DFlash) hit ~99% draft acceptance on
        repeated filler, inflating benchmark tg ~2x over real prompts; the
        prompt must be natural, non-looping text."""
        tokenizer = _WordTokenizer()
        ids = _generate_prompt(tokenizer, 2048, BenchmarkContextProfile.NOVEL_EN)
        window = 16
        windows = {tuple(ids[i : i + window]) for i in range(len(ids) - window)}
        distinct_ratio = len(windows) / (len(ids) - window)
        assert distinct_ratio > 0.5

    def test_prompt_deterministic_apart_from_uuid_prefix(self):
        tokenizer = _WordTokenizer()
        first = _generate_prompt(tokenizer, 512, BenchmarkContextProfile.NOVEL_EN)
        second = _generate_prompt(tokenizer, 512, BenchmarkContextProfile.NOVEL_EN)
        assert first[0] != second[0]
        assert first[1:] == second[1:]

    def test_prefix_and_corpus_are_encoded_together(self):
        tokenizer = _WordTokenizer()
        encoded_texts = []
        original_encode = tokenizer.encode

        def recording_encode(text):
            encoded_texts.append(text)
            return original_encode(text)

        tokenizer.encode = recording_encode
        _generate_prompt(tokenizer, 256, BenchmarkContextProfile.NOVEL_EN)
        assert encoded_texts
        assert all(text.startswith("BENCH-") for text in encoded_texts)
        assert all("Call me Ishmael." in text for text in encoded_texts)

    def test_external_prompt_is_not_repetitive(self):
        from omlx.admin.benchmark import _generate_external_prompt

        prompt = _generate_external_prompt(1024, BenchmarkContextProfile.NOVEL_EN)
        assert "Call me Ishmael." in prompt[:100]
        words = prompt.split(" ")
        window = 12
        windows = {tuple(words[i : i + window]) for i in range(len(words) - window)}
        distinct_ratio = len(windows) / (len(words) - window)
        assert distinct_ratio > 0.5

    def test_whole_corpus_repeats_when_tokenizer_needs_more_content(self):
        tokenizer = MagicMock()
        encoded_texts = []

        def encode(text):
            encoded_texts.append(text)
            return list(range(len(text)))

        tokenizer.encode = encode
        with patch("omlx.admin.benchmark._load_bench_corpus", return_value="abcdef"):
            ids = _generate_prompt(tokenizer, 100)

        assert len(ids) == 100
        assert encoded_texts[-1].count("abcdef") > 1


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
        # Should not raise, values should be finite. Decode rate is
        # unmeasurable (not 0.0 tok/s, which would misleadingly read as
        # a real stall) so it reports None.
        assert metrics["ttft_ms"] == 0.0
        assert metrics["gen_tps"] is None
        assert metrics["tpot_ms"] is None

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

        assert metrics["gen_tps"] is None
        assert metrics["tpot_ms"] is None
        assert metrics["total_throughput"] == pytest.approx(1898.7, abs=0.1)


class TestRunSingleTest:
    @pytest.mark.asyncio
    async def test_rejects_prompt_length_before_submission(self):
        engine = MagicMock()

        with pytest.raises(RuntimeError, match="before pp1024"):
            await _run_single_test(
                engine,
                prompt=[0] * 1023,
                max_tokens=1,
                pp_len=1024,
            )

        engine.stream_generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_cache_store(self):
        class RecordingEngine:
            def __init__(self):
                self.kwargs = None

            async def stream_generate(self, **kwargs):
                self.kwargs = kwargs
                yield SimpleNamespace(
                    completion_tokens=1,
                    prompt_tokens=1024,
                    cached_tokens=0,
                    new_text="x",
                    finished=True,
                    finish_reason="length",
                )

        engine = RecordingEngine()
        await _run_single_test(
            engine,
            prompt=[0] * 1024,
            max_tokens=1,
            pp_len=1024,
        )

        assert engine.kwargs["skip_cache_store"] is True

    @pytest.mark.asyncio
    async def test_rejects_engine_reported_prompt_length(self):
        class MismatchedEngine:
            async def stream_generate(self, **kwargs):
                yield SimpleNamespace(
                    completion_tokens=1,
                    prompt_tokens=1023,
                    cached_tokens=0,
                    new_text="x",
                    finished=True,
                    finish_reason="length",
                )

        with pytest.raises(RuntimeError, match="engine reported 1023 tokens"):
            await _run_single_test(
                MismatchedEngine(),
                prompt=[0] * 1024,
                max_tokens=1,
                pp_len=1024,
            )

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
            prompt=[0] * 1024,
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
            prompt=[0] * 1024,
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
                prompt=[0] * 1024,
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
                prompt=[0] * 1024,
                max_tokens=128,
                pp_len=1024,
            )

        assert metrics["ttft_ms"] == pytest.approx(200.0)
        assert metrics["gen_tps"] is None
        assert metrics["tpot_ms"] is None


class TestRunBatchTest:
    @pytest.mark.asyncio
    async def test_submits_exact_token_ids(self):
        class Core:
            def __init__(self):
                self.prompts = {}
                self.skip_cache_store = []

            async def add_request(
                self, prompt, sampling_params, skip_cache_store=False
            ):
                request_id = f"request-{len(self.prompts)}"
                self.prompts[request_id] = prompt
                self.skip_cache_store.append(skip_cache_store)
                return request_id

            async def stream_outputs(self, request_id):
                prompt = self.prompts[request_id]
                yield SimpleNamespace(
                    completion_tokens=2,
                    prompt_tokens=len(prompt),
                    finished=True,
                )

        core = Core()
        engine = SimpleNamespace(_engine=core)
        prompts = [[row] * 1024 for row in (1, 2)]

        metrics = await _run_batch_test(
            engine=engine,
            prompts=prompts,
            prompt_tokens=1024,
            max_tokens=2,
            batch_size=2,
        )

        assert list(core.prompts.values()) == prompts
        assert core.skip_cache_store == [True, True]
        assert metrics["batch_size"] == 2

    @pytest.mark.asyncio
    async def test_rejects_mismatched_prompt_before_submission(self):
        core = MagicMock()
        engine = SimpleNamespace(_engine=core)

        with pytest.raises(RuntimeError, match="expected 1024"):
            await _run_batch_test(
                engine=engine,
                prompts=[[0] * 1023, [1] * 1024],
                prompt_tokens=1024,
                max_tokens=1,
                batch_size=2,
            )

        core.add_request.assert_not_called()


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
    async def test_auto_uses_vlm_engine_for_lightning_mtp(self):
        settings = SimpleNamespace(mtp_enabled=True)

        run, pool = await self._run(settings=settings)

        assert pool.force_lm_values == [False]
        assert run.experimental_features == ["mtp"]
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
    async def test_force_lm_engine_overrides_lightning_mtp_auto(self):
        settings = SimpleNamespace(mtp_enabled=True)

        run, pool = await self._run(settings=settings, force_lm_engine=True)

        assert pool.force_lm_values == [True]
        assert run.experimental_features == ["mtp"]
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
    async def test_run_benchmark_pins_speed_priority_and_restores(self):
        """Throughput runs force prefill speed priority (no throttle
        shrinking mid-measurement) and restore the previous mode after."""
        pool = _FakeBenchEnginePool()
        pool._scheduler_config = SimpleNamespace(prefill_speed_priority=False)
        seen = []

        class _Recorder(_FakeBenchEngine):
            async def stream_generate(self, **kwargs):
                seen.append(pool._scheduler_config.prefill_speed_priority)
                yield SimpleNamespace(
                    completion_tokens=1,
                    prompt_tokens=1024,
                    cached_tokens=0,
                    new_text="x",
                    finished=True,
                    finish_reason="length",
                )

        pool._engine = _Recorder()
        run = BenchmarkRun(
            bench_id="bench-pin",
            request=BenchmarkRequest(
                model_id="m", prompt_lengths=[1024], generation_length=1
            ),
        )
        with patch("omlx.admin.benchmark._upload_to_omlx_ai", AsyncMock()):
            await run_benchmark(run, pool)

        assert run.status == "completed"
        assert seen and all(seen), "speed priority must be pinned during the run"
        assert pool._scheduler_config.prefill_speed_priority is False

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

    @pytest.mark.asyncio
    async def test_ane_warmup_executes_a_full_2048_token_prefill(self):
        class LongTokenizer:
            def encode(self, text):
                return list(range(10000))

        class RecordingEngine(_FakeBenchEngine):
            tokenizer = LongTokenizer()

            def __init__(self):
                self.calls = []

            async def stream_generate(self, **kwargs):
                self.calls.append(kwargs)
                yield SimpleNamespace(
                    completion_tokens=1,
                    prompt_tokens=len(kwargs["prompt"]),
                    cached_tokens=0,
                    new_text="x",
                    finished=True,
                    finish_reason="length",
                )

        engine = RecordingEngine()
        run = BenchmarkRun(
            bench_id="bench-ane-warmup",
            request=BenchmarkRequest(
                model_id="test-model",
                prompt_lengths=[1024],
                generation_length=1,
                warmup_mode="ane_2048",
            ),
        )

        with patch("omlx.admin.benchmark._upload_to_omlx_ai", AsyncMock()):
            await run_benchmark(run, _FakeBenchEnginePool(engine=engine))

        assert run.status == "completed"
        # stream_generate reserves the final token for first decode, leaving
        # exactly 2,048 tokens for the ANE-eligible prefill invocation.
        assert len(engine.calls[0]["prompt"]) == 2049
        assert len(engine.calls[1]["prompt"]) == 1024


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
            qwen35_ane_prefill_enabled=True,
        )

        assert _detect_experimental_features(settings) == [
            "dflash",
            "specprefill",
            "turboquant",
            "mtp",
            "vlm_mtp",
            "qwen35_ane_prefill",
        ]

    def test_missing_flags_are_treated_as_disabled(self):
        assert _detect_experimental_features(SimpleNamespace()) == []


class TestDeriveFeatureFlags:
    """The upload projection: [{key, label}] for the active features only."""

    def test_shape_is_key_and_label(self):
        settings = SimpleNamespace(mtp_enabled=True)
        assert _derive_feature_flags(settings) == [
            {"key": "lightning_mtp", "label": "Lightning MTP"}
        ]

    def test_only_active_features_appear(self):
        settings = SimpleNamespace(mtp_enabled=True, dflash_enabled=False)
        keys = [f["key"] for f in _derive_feature_flags(settings)]
        assert keys == ["lightning_mtp"]

    def test_turboquant_carries_its_bit_width(self):
        settings = SimpleNamespace(turboquant_kv_enabled=True, turboquant_kv_bits=4)
        assert _derive_feature_flags(settings) == [
            {"key": "turboquant_kv_4bit", "label": "TurboQuant KV 4-bit"}
        ]

    def test_qwen_ane_prefill_is_reported_as_acceleration(self):
        settings = SimpleNamespace(qwen35_ane_prefill_enabled=True)
        assert _derive_feature_flags(settings) == [
            {"key": "qwen35_ane_prefill", "label": "Qwen ANE Prefill"}
        ]

    def test_fractional_bit_width_stays_key_safe(self):
        # Keys must match [a-z0-9_] for the leaderboard, so 2.5 becomes 2_5.
        settings = SimpleNamespace(turboquant_kv_enabled=True, turboquant_kv_bits=2.5)
        flag = _derive_feature_flags(settings)[0]
        assert flag["key"] == "turboquant_kv_2_5bit"
        assert flag["label"] == "TurboQuant KV 2.5-bit"

    def test_float_valued_whole_bits_do_not_render_a_decimal(self):
        settings = SimpleNamespace(turboquant_kv_enabled=True, turboquant_kv_bits=4.0)
        assert _derive_feature_flags(settings)[0]["key"] == "turboquant_kv_4bit"

    def test_turboquant_without_a_bit_width_falls_back_to_the_bare_key(self):
        settings = SimpleNamespace(turboquant_kv_enabled=True, turboquant_kv_bits=None)
        assert _derive_feature_flags(settings) == [
            {"key": "turboquant_kv", "label": "TurboQuant KV"}
        ]

    def test_index_cache_freq_is_not_a_flag(self):
        # It is a layer stride, not an on/off accelerator — tagging a run
        # "accelerated" for it would mislead the leaderboard.
        settings = SimpleNamespace(index_cache_freq=4)
        assert _derive_feature_flags(settings) == []

    def test_no_settings_yields_no_flags(self):
        assert _derive_feature_flags(SimpleNamespace()) == []


class TestFilterUploadedSettings:
    """Allowlist projection — a denylist would ship every future field."""

    def _settings(self, **overrides):
        from omlx.model_settings import ModelSettings

        return ModelSettings(**overrides)

    def test_performance_fields_are_kept(self):
        out = _filter_uploaded_settings(
            self._settings(
                turboquant_kv_enabled=True,
                turboquant_kv_bits=4,
                mtp_enabled=True,
                mtp_num_draft_tokens=3,
                index_cache_freq=4,
                guided_grammar_enabled=True,
                qwen35_ane_prefill_enabled=True,
                qwen35_ane_prefill_sequence_length=2048,
                qwen35_ane_prefill_fraction=0.53,
                qwen35_ane_prefill_max_layers=64,
                qwen35_ane_prefill_dual_ane=True,
                qwen35_ane_prefill_gdn=True,
                qwen35_ane_prefill_gdn_fraction=0.5,
                qwen35_ane_prefill_gdn_max_layers=48,
            )
        )
        assert out["turboquant_kv_bits"] == 4
        assert out["mtp_num_draft_tokens"] == 3
        assert out["index_cache_freq"] == 4
        assert out["guided_grammar_enabled"] is True
        assert out["qwen35_ane_prefill_enabled"] is True
        assert out["qwen35_ane_prefill_sequence_length"] == 2048
        assert out["qwen35_ane_prefill_fraction"] == 0.53
        assert out["qwen35_ane_prefill_max_layers"] == 64
        assert out["qwen35_ane_prefill_dual_ane"] is True
        assert out["qwen35_ane_prefill_gdn"] is True
        assert out["qwen35_ane_prefill_gdn_fraction"] == 0.5
        assert out["qwen35_ane_prefill_gdn_max_layers"] == 48

    def test_free_text_and_organization_fields_are_dropped(self):
        out = _filter_uploaded_settings(
            self._settings(
                display_name="my private name",
                description="notes only I should see",
                model_alias="alias",
                is_favorite=True,
                is_pinned=True,
                is_default=True,
                is_hidden=True,
                active_profile_name="profile",
                trust_remote_code=True,
                ttl_seconds=60,
            )
        )
        for key in (
            "display_name",
            "description",
            "model_alias",
            "is_favorite",
            "is_pinned",
            "is_default",
            "is_hidden",
            "active_profile_name",
            "trust_remote_code",
            "ttl_seconds",
        ):
            assert key not in out

    def test_grammar_body_is_dropped_but_the_toggle_is_kept(self):
        out = _filter_uploaded_settings(
            self._settings(
                guided_grammar_enabled=True,
                guided_grammar="root ::= " + "x" * 5000,
            )
        )
        assert "guided_grammar" not in out
        assert out["guided_grammar_enabled"] is True

    def test_draft_model_paths_are_reduced_to_a_basename(self):
        # The drafter's identity explains an MTP/DFlash result; the full path
        # would leak the local filesystem layout and the OS username.
        out = _filter_uploaded_settings(
            self._settings(
                dflash_enabled=True,
                dflash_draft_model="/Users/someone/Workspace/models/Qwen3-0.6B-4bit",
            )
        )
        assert out["dflash_draft_model"] == "Qwen3-0.6B-4bit"

    def test_bare_draft_model_name_is_unchanged(self):
        out = _filter_uploaded_settings(
            self._settings(
                dflash_enabled=True,
                dflash_draft_model="Qwen3-0.6B-4bit",
            )
        )
        assert out["dflash_draft_model"] == "Qwen3-0.6B-4bit"

    def test_non_settings_object_returns_none(self):
        assert _filter_uploaded_settings(SimpleNamespace()) is None

    def test_oversized_snapshot_falls_back_to_accelerator_flags(self):
        # The fallback keeps every accelerator toggle, disabled ones included:
        # knowing a feature was off is as useful as knowing it was on.
        settings = self._settings(mtp_enabled=True, turboquant_kv_enabled=True)
        with patch("omlx.admin.benchmark._MAX_UPLOADED_SETTINGS_BYTES", 10):
            out = _filter_uploaded_settings(settings)
        assert out == {
            "dflash_enabled": False,
            "specprefill_enabled": False,
            "turboquant_kv_enabled": True,
            "mtp_enabled": True,
            "vlm_mtp_enabled": False,
            "qwen35_ane_prefill_enabled": False,
        }
        # Everything else is gone.
        assert "temperature" not in out
        assert "max_context_window" not in out


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

        await _send_event(
            run,
            {
                "type": "progress",
                "phase": "single",
                "message": "Testing",
                "current": 1,
                "total": 3,
            },
        )

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
        (tmp_path / "config.json").write_text(__import__("json").dumps(config))
        assert _detect_quantization(str(tmp_path)) == "4bit"

    def test_from_config_json_8bit(self, tmp_path):
        config = {"quantization_config": {"bits": 8}}
        (tmp_path / "config.json").write_text(__import__("json").dumps(config))
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
        (model_dir / "config.json").write_text(__import__("json").dumps(config))
        assert _detect_quantization(str(model_dir)) == "8bit"


# =============================================================================
# Model name cleaning tests
# =============================================================================


class TestUploadModelName:
    """The published name is what oMLX shows and its copy button copies.

    Quantization and MLX suffixes used to be stripped, which erased the one
    detail distinguishing two builds of the same model on the leaderboard.
    """

    def test_keeps_quantization_suffix(self):
        assert _upload_model_name("Qwen3-30B-A3B-4bit") == "Qwen3-30B-A3B-4bit"

    def test_keeps_mlx_marker(self):
        assert (
            _upload_model_name("qwen3.6-35b-a3b-8bit-mlx") == "qwen3.6-35b-a3b-8bit-mlx"
        )

    def test_keeps_mixed_case_and_dwq_style_suffixes(self):
        assert _upload_model_name("Llama-3-8B-4bit-DWQ") == "Llama-3-8B-4bit-DWQ"

    def test_plain_name_is_unchanged(self):
        assert _upload_model_name("Qwen3-30B-A3B") == "Qwen3-30B-A3B"

    def test_takes_last_path_component(self):
        assert (
            _upload_model_name("mlx-community/Qwen3-30B-A3B-4bit")
            == "Qwen3-30B-A3B-4bit"
        )

    def test_truncates_to_the_leaderboard_limit(self):
        assert len(_upload_model_name("x" * 300)) == 150

    def test_org_repo_from_two_level_layout(self):
        # Organized layout: the org directory qualifies the repo id, matching
        # what the local models UI shows (#1808).
        entry = MagicMock(spec=["model_path", "source_repo_id"])
        entry.model_path = "/models/mlx-community/Qwen3-30B-A3B-4bit"
        entry.source_repo_id = None
        assert (
            _upload_model_repo(
                "Qwen3-30B-A3B-4bit", entry=entry, model_dirs=[Path("/models")]
            )
            == "mlx-community/Qwen3-30B-A3B-4bit"
        )

    def test_flat_layout_has_no_repo(self):
        # Flat layout: the parent is the scan root, never treated as an org.
        entry = MagicMock(spec=["model_path", "source_repo_id"])
        entry.model_path = "/models/Qwen3-30B-A3B-4bit"
        entry.source_repo_id = None
        assert (
            _upload_model_repo(
                "Qwen3-30B-A3B-4bit", entry=entry, model_dirs=[Path("/models")]
            )
            is None
        )

    def test_hf_cache_repo_uses_source_repo_id(self):
        entry = MagicMock(spec=["model_path", "source_repo_id"])
        entry.model_path = "/cache/models--mlx-community--Qwen3-4bit/snapshots/ab"
        entry.source_repo_id = "mlx-community/Qwen3-4bit"
        assert (
            _upload_model_repo("mlx-community--Qwen3-4bit", entry=entry, model_dirs=[])
            == "mlx-community/Qwen3-4bit"
        )

    def test_repo_none_without_entry(self):
        assert _upload_model_repo("Qwen3-30B-A3B") is None


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
    async def test_upload_proceeds_when_acceleration_is_active(self):
        """Accelerated runs upload too, carrying their flags."""
        from omlx.admin.benchmark import _upload_to_omlx_ai

        run = BenchmarkRun(
            bench_id="test-bench",
            request=BenchmarkRequest(
                model_id="Qwen3-30B-4bit",
                prompt_lengths=[1024],
            ),
            experimental_features=["dflash", "turboquant"],
            feature_flags=[
                {"key": "dflash", "label": "DFlash"},
                {"key": "turboquant_kv_4bit", "label": "TurboQuant KV 4-bit"},
            ],
            model_settings_snapshot={"dflash_enabled": True},
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
                "system_metrics": {"sample_count": 4, "interval_s": 1.0},
            },
        ]

        mock_entry = MagicMock()
        mock_entry.model_path = "/models/Qwen3-30B-4bit"
        mock_pool = MagicMock()
        mock_pool.get_entry.return_value = mock_entry
        mock_pool._settings_manager = None

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "abc", "url": "https://omlx.ai/b/abc"}
        mock_to_thread = AsyncMock(return_value=mock_response)

        with patch("asyncio.to_thread", mock_to_thread):
            await _upload_to_omlx_ai(run, mock_pool)

        mock_to_thread.assert_awaited_once()
        payload = mock_to_thread.await_args.kwargs["json"]
        assert payload["feature_flags"] == [
            {"key": "dflash", "label": "DFlash"},
            {"key": "turboquant_kv_4bit", "label": "TurboQuant KV 4-bit"},
        ]
        assert payload["context_profile"] == "code_python"
        assert payload["model_settings"] == {
            "benchmark_context": "Code (Python)",
            "dflash_enabled": True,
        }
        assert payload["system_metrics"] == {"sample_count": 4, "interval_s": 1.0}

        event_types = [e["type"] for e in run.events]
        assert "upload_done" in event_types
        assert "upload_skipped" not in event_types
        assert run.upload_state["phase"] == "done"
        assert run.upload_state["feature_flags"] == run.feature_flags

    @pytest.mark.asyncio
    async def test_payload_carries_new_fields_when_unaccelerated(self):
        """A plain run still sends the new keys, with empty/None values."""
        from omlx.admin.benchmark import _upload_to_omlx_ai

        run = BenchmarkRun(
            bench_id="test-bench",
            request=BenchmarkRequest(
                model_id="qwen3.6-35b-a3b-8bit-mlx",
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
        mock_entry.model_path = "/models/qwen3.6-35b-a3b-8bit-mlx"
        mock_pool = MagicMock()
        mock_pool.get_entry.return_value = mock_entry
        mock_pool._settings_manager = None

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "abc", "url": "https://omlx.ai/b/abc"}
        mock_to_thread = AsyncMock(return_value=mock_response)

        with patch("asyncio.to_thread", mock_to_thread):
            await _upload_to_omlx_ai(run, mock_pool)

        payload = mock_to_thread.await_args.kwargs["json"]
        # The full id reaches the leaderboard, suffixes intact.
        assert payload["model_name"] == "qwen3.6-35b-a3b-8bit-mlx"
        assert payload["feature_flags"] == []
        assert payload["context_profile"] == "code_python"
        assert payload["model_settings"] == {"benchmark_context": "Code (Python)"}
        # Null rather than a zero-filled object, so the site does not average
        # fabricated measurements.
        assert payload["system_metrics"] is None
        assert "peak_footprint_gb" in payload


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

    def _resp(
        self, status=403, headers=None, text="", json_raises=True, json_data=None
    ):
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
        assert len(short) == 1024 * 4
        assert len(long) == 4096 * 4
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
        client.stream_chat_completion = AsyncMock(return_value=stats or _stream_stats())
        client.aclose = AsyncMock()
        return client

    async def test_never_touches_engine_pool(self):
        run = self._make_run(batch_sizes=[2])
        pool = MagicMock()
        client = self._mock_client()

        with patch("omlx.admin.benchmark.ExternalAPIClient", return_value=client):
            await run_benchmark(run, pool)

        assert run.status == "completed"
        pool.get_engine.assert_not_called()
        pool.get_loaded_model_ids.assert_not_called()
        pool._unload_engine.assert_not_called()

    async def test_event_sequence_and_upload_skipped(self):
        run = self._make_run(prompt_lengths=[1024], batch_sizes=[2])
        client = self._mock_client()

        with patch("omlx.admin.benchmark.ExternalAPIClient", return_value=client):
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

        with patch("omlx.admin.benchmark.ExternalAPIClient", return_value=client):
            await run_benchmark(run, MagicMock())

        result = run.results[0]
        assert result["test_type"] == "single"
        assert result["requested_pp"] == 4096
        assert result["pp"] == 3900
        assert result["prompt_tokens"] == 3900
        assert result["completion_tokens"] == 128
        assert result["peak_memory_bytes"] is None
        # gen duration 1.0s (first 0.5 → last 1.5) with 128 tokens
        assert result["gen_tps"] == 128.0
        assert result["ttft_ms"] == 500.0

    async def test_batch_result_aggregates_usage_counts(self):
        run = self._make_run(prompt_lengths=[1024], batch_sizes=[2])
        client = self._mock_client(_stream_stats(prompt=1000, completion=100))

        with patch("omlx.admin.benchmark.ExternalAPIClient", return_value=client):
            await run_benchmark(run, MagicMock())

        batch = [r for r in run.results if r["test_type"] == "batch"][0]
        assert batch["batch_size"] == 2
        assert batch["requested_pp"] == 1024
        assert batch["pp"] == 1000
        assert batch["prompt_tokens_min"] == 1000
        assert batch["prompt_tokens_max"] == 1000
        assert batch["total_prompt_tokens"] == 2000
        assert batch["total_gen_tokens"] == 200

    async def test_single_result_unmeasurable_span_reports_none(self):
        """A true single-chunk dump (first_content_time == last_content_time
        == end_time) must report tg TPS as unmeasured, not 0.0.

        Content was still observed, so TTFT and the prefill rate derived from
        it remain real measurements and must keep their values.
        """
        from omlx.admin.external_api import StreamStats

        collapsed_stats = StreamStats(
            prompt_tokens=1000,
            completion_tokens=128,
            cached_tokens=0,
            start_time=0.0,
            first_content_time=0.5,
            last_content_time=0.5,
            end_time=0.5,
            text="x" * 16,
            content_observed=True,
        )
        run = self._make_run(prompt_lengths=[1024])
        client = self._mock_client(collapsed_stats)

        with patch("omlx.admin.benchmark.ExternalAPIClient", return_value=client):
            await run_benchmark(run, MagicMock())

        result = run.results[0]
        assert result["gen_tps"] is None
        assert result["tpot_ms"] is None
        assert result["ttft_ms"] == pytest.approx(500.0)
        assert result["processing_tps"] == pytest.approx(2000.0)

    async def test_single_result_without_content_reports_ttft_none(self):
        """When no content or reasoning delta was ever seen, the fallback
        timestamps make TTFT and pp TPS describe the whole response, so both
        have to report unmeasured alongside the decode metrics."""
        from omlx.admin.external_api import StreamStats

        unobserved_stats = StreamStats(
            prompt_tokens=1000,
            completion_tokens=128,
            cached_tokens=0,
            start_time=0.0,
            # stream_chat_completion falls both timestamps back to end_time
            first_content_time=1.2,
            last_content_time=1.2,
            end_time=1.2,
            text="",
            content_observed=False,
        )
        run = self._make_run(prompt_lengths=[1024])
        client = self._mock_client(unobserved_stats)

        with patch("omlx.admin.benchmark.ExternalAPIClient", return_value=client):
            await run_benchmark(run, MagicMock())

        result = run.results[0]
        assert result["ttft_ms"] is None
        assert result["processing_tps"] is None
        assert result["gen_tps"] is None
        assert result["tpot_ms"] is None
        # e2e still spans a real interval and stays measured
        assert result["e2e_latency_s"] == pytest.approx(1.2)

    async def test_batch_without_content_reports_none_not_inflated(self):
        """Models the real collapse on the real clock: no field matched, so
        every timestamp fell back to end_time, which stream_chat_completion
        samples inside the request coroutine and therefore always before the
        wall_end sampled after asyncio.gather.

        The aggregate window stays a small positive number in that state, so
        a sign check on it cannot detect the collapse and the batch reports
        millions of tok/s. Measurability has to come from the stats.
        """
        from omlx.admin.benchmark import _run_external_batch_test
        from omlx.admin.external_api import StreamStats

        async def collapsed_stream(**kwargs):
            await asyncio.sleep(0.01)
            now = time.perf_counter()
            return StreamStats(
                prompt_tokens=1000,
                completion_tokens=128,
                cached_tokens=0,
                start_time=now - 0.01,
                first_content_time=now,
                last_content_time=now,
                end_time=now,
                text="",
                content_observed=False,
            )

        client = MagicMock()
        client.stream_chat_completion = collapsed_stream

        result = await _run_external_batch_test(
            client=client,
            prompts=["p1", "p2"],
            max_tokens=128,
            batch_size=2,
        )

        assert result["tg_tps"] is None
        assert result["pp_tps"] is None
        assert result["avg_ttft_ms"] is None
        assert result["total_gen_tokens"] == 256

    async def test_batch_single_chunk_dump_keeps_ttft_but_drops_tg(self):
        """Content arrived, but all of it in one chunk: the decode span is
        unmeasurable while TTFT and pp TPS are real measurements."""
        from omlx.admin.benchmark import _run_external_batch_test
        from omlx.admin.external_api import StreamStats

        async def single_chunk_stream(**kwargs):
            await asyncio.sleep(0.01)
            now = time.perf_counter()
            return StreamStats(
                prompt_tokens=1000,
                completion_tokens=128,
                cached_tokens=0,
                start_time=now - 0.01,
                first_content_time=now,
                last_content_time=now,
                end_time=now,
                text="x" * 16,
                content_observed=True,
            )

        client = MagicMock()
        client.stream_chat_completion = single_chunk_stream

        result = await _run_external_batch_test(
            client=client,
            prompts=["p1", "p2"],
            max_tokens=128,
            batch_size=2,
        )

        assert result["tg_tps"] is None
        assert result["pp_tps"] is not None
        assert result["avg_ttft_ms"] is not None

    async def test_batch_measurable_window_computes_tg_tps(self):
        """A real, observed decode span still computes a normal tg_tps, with
        no regression from the None guard."""
        from omlx.admin.benchmark import _run_external_batch_test
        from omlx.admin.external_api import StreamStats

        measured = StreamStats(
            prompt_tokens=1000,
            completion_tokens=128,
            cached_tokens=0,
            start_time=0.0,
            first_content_time=0.5,
            last_content_time=1.5,
            end_time=1.6,
            text="x" * 16,
            content_observed=True,
        )
        client = MagicMock()
        client.stream_chat_completion = AsyncMock(return_value=measured)

        # A plain two-element side_effect would raise StopIteration on any
        # other perf_counter call in the process during this block, so fall
        # back to the last value instead.
        clock = iter([0.0, 1.6])  # wall_start, wall_end
        with patch(
            "omlx.admin.benchmark.time.perf_counter",
            side_effect=lambda: next(clock, 1.6),
        ):
            result = await _run_external_batch_test(
                client=client,
                prompts=["p1", "p2"],
                max_tokens=128,
                batch_size=2,
            )

        # total_gen_tokens=256, gen window = wall_end(1.6) - max_first_token(0.5) = 1.1s
        assert result["tg_tps"] == pytest.approx(256 / 1.1, abs=0.1)
        assert result["pp_tps"] == pytest.approx(2000 / 0.5, abs=0.1)

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

        with patch("omlx.admin.benchmark.ExternalAPIClient", return_value=client):
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

        with patch("omlx.admin.benchmark.ExternalAPIClient", return_value=client):
            task = asyncio.create_task(run_benchmark(run, MagicMock()))
            await started.wait()
            task.cancel()
            await task

        assert run.status == "cancelled"
        assert run.events[-1]["type"] == "error"
        assert "cancelled" in run.events[-1]["message"].lower()
        client.aclose.assert_awaited()


# =============================================================================
# ANE benchmark trace tests
# =============================================================================


class TestAneBenchmarkTrace:
    def test_summary_reports_observed_scheduler_calls_not_implied_prompt_width(
        self, caplog
    ):
        with caplog.at_level(logging.INFO):
            _log_ane_benchmark_trace(
                pp_len=4097,
                prefill_duration_s=1.0,
                config={"sequence_length": 2048, "mlp_layers": 2},
                profile={"mlp": {"operations": 4}},
                scheduler_trace={
                    "chunk_tokens": [2048, 2048],
                    "requested_steps": [4096, 4096],
                    "boundary_enabled": True,
                    "cache_block_size": 2048,
                },
            )

        messages = [record.getMessage() for record in caplog.records]
        summary = next(m for m in messages if "[benchmark-ane-summary]" in m)
        assert "model_calls=2" in summary
        assert "model_call_widths=2048x2" in summary
        assert "requested_steps=4096x2" in summary
        assert "boundary_enabled=True" in summary
        assert "cache_block_size=2048" in summary
        assert "accounting=observed" in summary
        assert "full_ane_tiles=2" in summary
        assert "gpu_tail_tokens=0" in summary

    def test_summary_distinguishes_one_wide_call_from_two_tiles(self, caplog):
        with caplog.at_level(logging.INFO):
            _log_ane_benchmark_trace(
                pp_len=4097,
                prefill_duration_s=1.0,
                config={"sequence_length": 2048, "mlp_layers": 2},
                profile={"mlp": {"operations": 4}},
                scheduler_trace={
                    "chunk_tokens": [4096],
                    "requested_steps": [4096],
                    "boundary_enabled": False,
                    "cache_block_size": 0,
                },
            )

        summary = next(
            record.getMessage()
            for record in caplog.records
            if "[benchmark-ane-summary]" in record.getMessage()
        )
        assert "model_calls=1" in summary
        assert "model_call_widths=4096x1" in summary
        assert "full_ane_tiles=2" in summary

    def test_expectations_follow_compiled_layers(self, caplog):
        with caplog.at_level(logging.INFO):
            _log_ane_benchmark_trace(
                pp_len=16385,
                prefill_duration_s=1.0,
                config={
                    "sequence_length": 2048,
                    "mlp_layers": 64,
                    "gdn_layers": 48,
                    "compiled_mlp_layers": 60,
                    "compiled_gdn_layers": 0,
                    "active": True,
                },
                profile={"mlp": {"operations": 480}, "gdn": {}},
            )

        messages = [record.getMessage() for record in caplog.records]
        mlp_line = next(m for m in messages if "category=mlp" in m)
        assert "configured_layers=64" in mlp_line
        assert "compiled_layers=60" in mlp_line
        assert "expected_operations=480" in mlp_line
        gdn_line = next(m for m in messages if "category=gdn" in m)
        assert "compiled_layers=0" in gdn_line
        assert "expected_operations=0" in gdn_line

    def test_settings_layers_remain_the_fallback_when_unknown(self, caplog):
        with caplog.at_level(logging.INFO):
            _log_ane_benchmark_trace(
                pp_len=4096,
                prefill_duration_s=None,
                config={"sequence_length": 2048, "mlp_layers": 64},
                profile={},
            )

        mlp_line = next(
            record.getMessage()
            for record in caplog.records
            if "category=mlp" in record.getMessage()
        )
        assert "configured_layers=64" in mlp_line
        assert "compiled_layers=unknown" in mlp_line
        assert "expected_operations=64" in mlp_line
