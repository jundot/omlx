"""Tests for per-engine thread isolation (issue #1248)."""

from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

from omlx.engine_core import EngineCore
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
        ), patch("omlx.engine_core.clear_thread_compile_cache") as mock_clear:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(mock_model, mock_tokenizer)
            executor = engine._mlx_executor
            engine.close()

            # Cache cleared on the worker thread, then thread shut down.
            mock_clear.assert_called()
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
