"""Tests for per-engine thread isolation (issue #1248)."""

import concurrent.futures
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

from omlx.engine_core import EngineCore
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig


class TestSchedulerStreamParam:
    """Scheduler must accept an explicit stream and use it instead of the
    module-level generation_stream."""

    def test_scheduler_stores_explicit_stream(self):
        mock_model = MagicMock()
        mock_model.model_type = "test"
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 0

        stream = mx.new_thread_local_stream(mx.default_device())
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            stream=stream,
        )
        assert scheduler._stream is stream

    def test_scheduler_defaults_to_generation_stream(self):
        from omlx.scheduler import _default_generation_stream

        mock_model = MagicMock()
        mock_model.model_type = "test"
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 0

        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )
        assert scheduler._stream is _default_generation_stream


class TestSchedulerStreamIsolation:
    """Scheduler must use self._stream in all GPU stream operations,
    never the module-level generation_stream."""

    def test_no_module_level_generation_stream_in_hot_path(self):
        """After migration, scheduler.py should not reference the module-level
        generation_stream anywhere in the Scheduler class body except the
        __init__ default fallback and comments/docstrings."""
        import inspect
        import re

        import omlx.scheduler as sched_mod
        source = inspect.getsource(sched_mod.Scheduler)

        # Find bare generation_stream references that aren't:
        # - _default_generation_stream (the import alias)
        # - Part of a larger word
        bare_refs = re.findall(
            r'(?<!_default_)(?<!self\._)(?<!\w)generation_stream(?!\w)',
            source,
        )
        # Filter out string literals and comments by checking lines
        code_refs = []
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('"') or stripped.startswith("'"):
                continue
            matches = re.findall(
                r'(?<!_default_)(?<!self\._)(?<!\w)generation_stream(?!\w)',
                line,
            )
            code_refs.extend(matches)

        assert len(code_refs) == 0, (
            f"Found {len(code_refs)} bare generation_stream references in "
            f"Scheduler class body. All should be self._stream."
        )

    @pytest.mark.skipif(not mx.metal.is_available(), reason="Metal is required")
    def test_prefill_paths_use_engine_stream_on_worker_thread(self):
        """Direct prefill paths must bind native lazy ops to the engine stream.

        Qwen3.5/3.6's native MoE weighted-sum path is activated at 1024
        tokens. Without an explicit stream context, it binds its lazy
        primitive to the executor thread's unrelated default stream and M5
        macOS 27 builds fail during mx.eval (issue #2170).
        """

        observed_streams = []

        class RecordingModel:
            model_type = "test"
            config = SimpleNamespace(model_type="test")

            def parameters(self):
                return {}

            def __call__(self, inputs, **kwargs):
                observed_streams.append(mx.default_stream(mx.gpu))

        class Tokenizer:
            eos_token_id = 2
            pad_token_id = 0
            bos_token_id = 1

            def encode(self, text, add_special_tokens=True):
                return [1]

            def decode(self, token_ids, skip_special_tokens=True):
                return ""

        stream = mx.new_thread_local_stream(mx.default_device())
        scheduler = Scheduler(
            model=RecordingModel(),
            tokenizer=Tokenizer(),
            config=SchedulerConfig(prefill_step_size=2048),
            stream=stream,
        )
        request = Request(
            request_id="external-prefill-stream",
            prompt=[1, 2, 3],
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [1, 2, 3]
        request.num_prompt_tokens = 3
        cache = [SimpleNamespace(state=mx.array([0]))]
        spec_request = Request(
            request_id="specprefill-stream",
            prompt=[1, 2, 3, 4],
            sampling_params=SamplingParams(),
        )
        spec_request.prompt_token_ids = [1, 2, 3, 4]
        spec_request.remaining_tokens = [1, 2, 3, 4]
        spec_request.num_prompt_tokens = 4
        spec_request._specprefill_enabled = True
        spec_request._specprefill_threshold = 1
        spec_request._specprefill_keep_pct = 0.5
        scheduler._specprefill_draft_model = object()
        scheduler._draft_prefix_cache = None

        def record_specprefill_stream(*args, **kwargs):
            observed_streams.append(mx.default_stream(mx.gpu))
            return mx.ones(4), []

        def run_prefill():
            worker_default = mx.default_stream(mx.gpu)
            with mx.stream(stream):
                expected_engine_stream = mx.default_stream(mx.gpu)
            scheduler._do_external_prefill(
                request,
                request.prompt_token_ids,
                cache,
            )
            state = scheduler._begin_prefill(
                request,
                request.prompt_token_ids,
                cache,
            )
            scheduler._step_prefill_chunk(state)
            scheduler._try_specprefill_scoring(spec_request)
            return worker_default, expected_engine_stream, mx.default_stream(mx.gpu)

        with (
            patch(
                "omlx.patches.specprefill.score_tokens",
                side_effect=record_specprefill_stream,
            ),
            concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor,
        ):
            worker_default, expected_engine_stream, restored_default = executor.submit(
                run_prefill
            ).result()

        assert observed_streams == [
            expected_engine_stream,
            expected_engine_stream,
            expected_engine_stream,
        ]
        assert expected_engine_stream != worker_default
        assert restored_default == worker_default


class TestMtpStreamIsolation:
    """MTP patch must not use the module-level generation_stream directly."""

    def test_mtp_patch_no_get_generation_stream(self):
        """_get_generation_stream must not exist — MTP inherits the stream
        from the enclosing BatchGenerator context."""
        import omlx.patches.mlx_lm_mtp.batch_generator as mtp_mod

        assert not hasattr(mtp_mod, "_get_generation_stream"), (
            "_get_generation_stream still exists in MTP patch; "
            "MTP should inherit the per-engine stream from BatchGenerator"
        )

    def test_mtp_source_no_module_level_stream_read(self):
        """MTP patch source must not read sys.modules generation_stream."""
        import inspect
        import omlx.patches.mlx_lm_mtp.batch_generator as mtp_mod

        source = inspect.getsource(mtp_mod)
        assert "generation_stream" not in source, (
            "MTP patch references generation_stream — all stream context "
            "should be inherited from the enclosing BatchGenerator"
        )


class TestPerEngineExecutor:
    """Each EngineCore must create its own ThreadPoolExecutor, not share
    a global singleton."""

    def test_two_engines_have_different_executors(self):
        mock_model_a = MagicMock()
        mock_model_a.model_type = "test"
        mock_model_b = MagicMock()
        mock_model_b.model_type = "test"
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 0

        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine_a = EngineCore(mock_model_a, mock_tokenizer)
            engine_b = EngineCore(mock_model_b, mock_tokenizer)

            assert engine_a._mlx_executor is not engine_b._mlx_executor
            assert engine_a._mlx_stream is not engine_b._mlx_stream

            engine_a.close()
            engine_b.close()

    def test_engine_passes_stream_to_scheduler(self):
        mock_model = MagicMock()
        mock_model.model_type = "test"
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 0

        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(mock_model, mock_tokenizer)
            assert engine.scheduler._stream is engine._mlx_stream

            engine.close()

    def test_close_clears_compile_cache_then_shuts_down(self):
        """Normal path (compile-cache clear available): close() clears the
        worker thread's MLX thread_local compile cache (so ~CompilerCache is a
        no-op at thread exit) and then shuts the executor down normally."""
        import omlx.engine_core as ec

        mock_model = MagicMock()
        mock_model.model_type = "test"
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 0

        with patch("omlx.engine_core.get_registry") as mock_registry, patch(
            "omlx.engine_core.compile_cache_clear_available", return_value=True
        ), patch("omlx.engine_core.clear_thread_compile_cache") as mock_clear, patch(
            "omlx.engine_core._final_engine_thread_reclaim"
        ) as mock_reclaim:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(mock_model, mock_tokenizer)
            executor = engine._mlx_executor
            events = []

            def reclaim_side_effect(_stream):
                events.append("reclaim")
                assert engine.model is None
                assert engine.tokenizer is None
                assert engine.scheduler is None

            mock_model.release_resources.side_effect = lambda: events.append("release")
            mock_reclaim.side_effect = reclaim_side_effect
            mock_clear.side_effect = lambda: events.append("compile")
            engine.close()

            # Memory is reclaimed after dropping engine refs, then the compile
            # cache is cleared on the worker thread before thread shutdown.
            mock_model.release_resources.assert_called_once_with()
            mock_reclaim.assert_called_once_with(engine._mlx_stream)
            mock_clear.assert_called()
            assert events == ["release", "reclaim", "compile"]
            assert engine._mlx_executor is None
            assert executor._shutdown
            assert executor not in ec._immortal_mlx_executors

    def test_close_keeps_executor_alive_when_clear_unavailable(self):
        """Fallback (clear symbol unresolvable, e.g. a future MLX rename):
        close() must NOT exit the worker thread, since that would run MLX's
        thread_local ~CompilerCache and crash for @mx.compile models. The
        executor + stream are pinned immortal for the process lifetime."""
        import omlx.engine_core as ec

        mock_model = MagicMock()
        mock_model.model_type = "test"
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 0

        with patch("omlx.engine_core.get_registry") as mock_registry, patch(
            "omlx.engine_core.compile_cache_clear_available", return_value=False
        ):
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(mock_model, mock_tokenizer)
            executor = engine._mlx_executor
            stream = engine._mlx_stream
            engine.close()

            assert engine._mlx_executor is None
            assert not executor._shutdown
            assert executor in ec._immortal_mlx_executors
            assert stream in ec._immortal_mlx_streams


class TestConcurrentStreamIsolation:
    """Verify that per-engine streams don't leak across engines during
    concurrent execution."""

    def test_concurrent_schedulers_use_own_streams(self):
        """Two schedulers running step() concurrently must each use their
        own stream, not cross-contaminate."""
        mock_model_a = MagicMock()
        mock_model_a.model_type = "test"
        mock_model_b = MagicMock()
        mock_model_b.model_type = "test"
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 0

        stream_a = mx.new_thread_local_stream(mx.default_device())
        stream_b = mx.new_thread_local_stream(mx.default_device())
        assert stream_a is not stream_b

        sched_a = Scheduler(
            model=mock_model_a,
            tokenizer=mock_tokenizer,
            stream=stream_a,
        )
        sched_b = Scheduler(
            model=mock_model_b,
            tokenizer=mock_tokenizer,
            stream=stream_b,
        )

        assert sched_a._stream is stream_a
        assert sched_b._stream is stream_b
        assert sched_a._stream is not sched_b._stream

    def test_module_level_generation_stream_unchanged(self):
        """Creating schedulers with explicit streams must not modify the
        module-level _default_generation_stream."""
        from omlx.scheduler import _default_generation_stream

        original_id = id(_default_generation_stream)
        stream = mx.new_thread_local_stream(mx.default_device())

        mock_model = MagicMock()
        mock_model.model_type = "test"
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 0

        _ = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            stream=stream,
        )

        from omlx.scheduler import _default_generation_stream as current
        assert id(current) == original_id


class TestPrefixCacheReconstructStream:
    """Prefix cache reconstruction must run on the per-engine stream.

    Unwrapped, the reconstruct chain binds its lazy mx ops to the worker
    thread's default stream (gpu,0) while the prefill forward consumes them
    inside mx.stream(self._stream) — a cross-stream fence that can wedge
    permanently under MLX_METAL_FAST_SYNCH=1 when the async store-cache
    worker keeps gpu,0 busy (issue #2330).
    """

    def test_prepare_prefix_cache_runs_on_engine_stream(self):
        observed_streams = []

        model = MagicMock()
        model.model_type = "test"
        model.layers = []

        tokenizer = MagicMock()
        tokenizer.eos_token_id = 2

        stream = mx.new_thread_local_stream(mx.default_device())
        scheduler = Scheduler(
            model=model,
            tokenizer=tokenizer,
            config=SchedulerConfig(
                max_num_seqs=8,
                prefill_step_size=2048,
                paged_cache_block_size=0,
            ),
            stream=stream,
        )
        mock_bg = MagicMock()
        mock_bg.insert.return_value = [42]
        scheduler.batch_generator = mock_bg
        scheduler._current_sampler_params = ()

        class RecordingBlockAwareCache:
            def fetch_cache(self, request_id, prompt_token_ids, **kwargs):
                observed_streams.append(mx.default_stream(mx.gpu))
                return None, list(prompt_token_ids)

        scheduler.block_aware_cache = RecordingBlockAwareCache()

        request = Request(
            request_id="reconstruct-stream",
            prompt=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=8),
        )
        request.prompt_token_ids = [1, 2, 3]
        request.num_prompt_tokens = 3
        scheduler.add_request(request)

        def run_admission():
            worker_default = mx.default_stream(mx.gpu)
            with mx.stream(stream):
                expected_engine_stream = mx.default_stream(mx.gpu)
            with patch.object(
                scheduler, "_do_external_prefill", return_value=([], [0])
            ):
                scheduler._schedule_waiting()
            return worker_default, expected_engine_stream

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            worker_default, expected_engine_stream = executor.submit(
                run_admission
            ).result()

        assert observed_streams == [expected_engine_stream]
        assert expected_engine_stream != worker_default
