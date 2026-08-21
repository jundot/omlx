# SPDX-License-Identifier: Apache-2.0
"""Tests for requested-versus-effective DFlash state (issue #2896).

A DFlash load fails soft: oMLX logs the failure, falls back to the model's
natural engine, and serves normally with 200. Before this, an API client had
no way to tell a real DFlash load from that fallback. `/v1/models/status` now
carries `effective_engine`, `dflash_requested`, `dflash_active`, and a
redacted `dflash_fallback_reason`.
"""

import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.engine_pool import (
    _MAX_FALLBACK_REASON_CHARS,
    EnginePool,
    _sanitized_fallback_reason,
)
from omlx.model_settings import ModelSettings


def _make_pool(model_dir) -> EnginePool:
    pool = EnginePool()
    pool._get_final_ceiling = lambda: 0
    pool.discover_models(str(model_dir))
    return pool


@pytest.fixture
def model_dir(tmp_path):
    """One tiny LLM model directory; loading is always mocked."""
    model_a = tmp_path / "model-a"
    model_a.mkdir()
    (model_a / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (model_a / "model.safetensors").write_bytes(b"0" * 1024)
    return tmp_path


def _dflash_class(*, init_error=None, start_error=None):
    """Stand-in for DFlashEngine.

    The pool routes on `type(engine).__name__ == "DFlashEngine"`, so the class
    name is load-bearing and a bare MagicMock will not do.
    """

    class DFlashEngine:
        def __init__(self, **kwargs):
            if init_error is not None:
                raise init_error
            self.kwargs = kwargs
            self.stopped = False
            # Mirrors the real engine's public accessors, which start out
            # reporting the DFlash path and flip on a runtime fallback.
            self.in_fallback_mode = False
            self.fallback_engine_type = kwargs.get("fallback_engine_type", "batched")
            self.fallback_reason = None

        async def start(self):
            if start_error is not None:
                raise start_error

        async def stop(self):
            self.stopped = True

        def has_active_requests(self):
            return False

    return DFlashEngine


def _dflash_settings(**overrides) -> ModelSettings:
    values = {"dflash_enabled": True, "dflash_draft_model": "draft-model"}
    values.update(overrides)
    return ModelSettings(**values)


def _status_for(pool: EnginePool, model_id: str) -> dict:
    return next(m for m in pool.get_status()["models"] if m["id"] == model_id)


class TestDFlashSucceeded:
    @pytest.mark.asyncio
    async def test_active_dflash_is_reported_as_effective_engine(self, model_dir):
        pool = _make_pool(model_dir)

        with patch("omlx.engine.dflash.DFlashEngine", _dflash_class()):
            await pool._load_engine("model-a", runtime_settings=_dflash_settings())

        status = _status_for(pool, "model-a")
        assert status["engine_type"] == "batched"
        assert status["effective_engine"] == "dflash"
        assert status["dflash_requested"] is True
        assert status["dflash_active"] is True
        assert status["dflash_fallback_reason"] is None


class TestRuntimeFallbackAfterLoad:
    """DFlash drops its runtime after load on long context or multimodal
    content and never returns to it, so the status must follow the live
    engine instead of the attach-time record.
    """

    @staticmethod
    async def _load_dflash(pool, engine_type: str = "batched"):
        if engine_type == "vlm":
            entry = pool.get_entry("model-a")
            entry.model_type = "vlm"
            entry.engine_type = "vlm"
        with patch("omlx.engine.dflash.DFlashEngine", _dflash_class()):
            await pool._load_engine("model-a", runtime_settings=_dflash_settings())
        return pool.get_entry("model-a").engine

    @pytest.mark.asyncio
    async def test_context_fallback_reports_the_natural_engine(self, model_dir):
        pool = _make_pool(model_dir)
        engine = await self._load_dflash(pool)
        assert _status_for(pool, "model-a")["dflash_active"] is True

        engine.in_fallback_mode = True
        engine.fallback_reason = (
            "prompt of 14000 tokens reached the DFlash context limit of 8192"
        )

        status = _status_for(pool, "model-a")
        assert status["loaded"] is True
        assert status["effective_engine"] == "batched"
        assert status["dflash_requested"] is True
        assert status["dflash_active"] is False
        assert status["dflash_fallback_reason"] == (
            "DFlash runtime fallback: prompt of 14000 tokens reached the "
            "DFlash context limit of 8192"
        )

    @pytest.mark.asyncio
    async def test_multimodal_fallback_reports_the_vlm_engine(self, model_dir):
        pool = _make_pool(model_dir)
        engine = await self._load_dflash(pool, engine_type="vlm")
        assert engine.fallback_engine_type == "vlm"

        engine.in_fallback_mode = True
        engine.fallback_reason = "image content requires the VLM engine"

        status = _status_for(pool, "model-a")
        assert status["effective_engine"] == "vlm"
        assert status["dflash_active"] is False
        assert status["dflash_fallback_reason"] == (
            "DFlash runtime fallback: image content requires the VLM engine"
        )

    @pytest.mark.asyncio
    async def test_runtime_reason_is_bounded_and_redacted(self, model_dir):
        """The pool never forwards an engine-supplied reason verbatim."""
        pool = _make_pool(model_dir)
        engine = await self._load_dflash(pool)

        engine.in_fallback_mode = True
        engine.fallback_reason = (
            "evicted /Users/someone/models/draft/model.safetensors\n" + "detail " * 200
        )

        reason = _status_for(pool, "model-a")["dflash_fallback_reason"]
        assert len(reason) <= _MAX_FALLBACK_REASON_CHARS
        assert "\n" not in reason
        assert "/Users/someone" not in reason
        assert "<path>" in reason

    @pytest.mark.asyncio
    async def test_unrecorded_trigger_still_reports_the_fallback(self, model_dir):
        pool = _make_pool(model_dir)
        engine = await self._load_dflash(pool)

        engine.in_fallback_mode = True
        engine.fallback_reason = None

        status = _status_for(pool, "model-a")
        assert status["dflash_active"] is False
        assert status["dflash_fallback_reason"] == (
            "DFlash runtime fallback: trigger not recorded"
        )


class TestDFlashEngineFallbackAccessors:
    """The pool reads these public accessors; keep them wired to the state
    that `_evict_dflash_and_start_fallback` sets.
    """

    def test_accessors_expose_the_runtime_fallback_state(self):
        from omlx.engine.dflash import DFlashEngine

        # __new__ skips the loader-heavy __init__; these reads are pure.
        engine = DFlashEngine.__new__(DFlashEngine)
        engine._in_fallback_mode = True
        engine._fallback_engine_type = "vlm"
        engine._fallback_reason = "image content requires the VLM engine"
        engine._max_dflash_ctx = 8192

        assert engine.in_fallback_mode is True
        assert engine.fallback_engine_type == "vlm"
        assert engine.fallback_reason == "image content requires the VLM engine"
        assert engine._context_fallback_reason(14000) == (
            "prompt of 14000 tokens reached the DFlash context limit of 8192"
        )


class TestDFlashRequestedButFellBack:
    @pytest.mark.asyncio
    async def test_start_failure_reports_inactive_dflash_with_reason(self, model_dir):
        pool = _make_pool(model_dir)
        fallback = MagicMock()
        fallback.start = AsyncMock()

        with (
            patch(
                "omlx.engine.dflash.DFlashEngine",
                _dflash_class(
                    start_error=ValueError(
                        "Received parameters not in model: "
                        "candidate_selector.predecessor_codebook"
                    )
                ),
            ),
            patch("omlx.engine_pool.BatchedEngine", return_value=fallback),
        ):
            await pool._load_engine("model-a", runtime_settings=_dflash_settings())

        status = _status_for(pool, "model-a")
        assert status["loaded"] is True
        assert status["effective_engine"] == "batched"
        assert status["dflash_requested"] is True
        assert status["dflash_active"] is False
        assert status["dflash_fallback_reason"] == (
            "DFlash start failed: ValueError: Received parameters not in "
            "model: candidate_selector.predecessor_codebook"
        )

    @pytest.mark.asyncio
    async def test_init_failure_reason_hides_checkpoint_path(self, model_dir):
        pool = _make_pool(model_dir)
        fallback = MagicMock()
        fallback.start = AsyncMock()

        with (
            patch(
                "omlx.engine.dflash.DFlashEngine",
                _dflash_class(
                    init_error=RuntimeError(
                        "draft checkpoint /Users/someone/models/draft/model."
                        "safetensors is unreadable"
                    )
                ),
            ),
            patch("omlx.engine_pool.BatchedEngine", return_value=fallback),
        ):
            await pool._load_engine("model-a", runtime_settings=_dflash_settings())

        status = _status_for(pool, "model-a")
        assert status["dflash_requested"] is True
        assert status["dflash_active"] is False
        reason = status["dflash_fallback_reason"]
        assert reason.startswith("DFlash init failed: RuntimeError:")
        assert "<path>" in reason
        assert "/Users/someone" not in reason

    @pytest.mark.asyncio
    async def test_missing_dependency_is_reported(self, model_dir):
        pool = _make_pool(model_dir)
        fallback = MagicMock()
        fallback.start = AsyncMock()

        # A None entry in sys.modules makes the deferred import raise
        # ImportError, exactly as an uninstalled dflash-mlx does.
        with (
            patch.dict(sys.modules, {"omlx.engine.dflash": None}),
            patch("omlx.engine_pool.BatchedEngine", return_value=fallback),
        ):
            await pool._load_engine("model-a", runtime_settings=_dflash_settings())

        status = _status_for(pool, "model-a")
        assert status["effective_engine"] == "batched"
        assert status["dflash_requested"] is True
        assert status["dflash_active"] is False
        assert status["dflash_fallback_reason"] == "dflash-mlx is not installed"

    @pytest.mark.asyncio
    async def test_enabled_without_draft_model_is_reported(self, model_dir):
        pool = _make_pool(model_dir)
        fallback = MagicMock()
        fallback.start = AsyncMock()

        with patch("omlx.engine_pool.BatchedEngine", return_value=fallback):
            await pool._load_engine(
                "model-a",
                runtime_settings=_dflash_settings(dflash_draft_model=None),
            )

        status = _status_for(pool, "model-a")
        assert status["effective_engine"] == "batched"
        assert status["dflash_requested"] is True
        assert status["dflash_active"] is False
        assert status["dflash_fallback_reason"] == (
            "DFlash is enabled but no draft model is configured"
        )


class TestDFlashNotRequested:
    @pytest.mark.asyncio
    async def test_plain_load_leaves_dflash_fields_clear(self, model_dir):
        """A non-DFlash engine is never probed for fallback state. MagicMock
        auto-creates a truthy `in_fallback_mode`, so this also pins the
        `dflash_active and ...` short-circuit in `_live_dflash_status`."""
        pool = _make_pool(model_dir)
        engine = MagicMock()
        engine.start = AsyncMock()

        with patch("omlx.engine_pool.BatchedEngine", return_value=engine):
            await pool._load_engine("model-a", runtime_settings=ModelSettings())

        status = _status_for(pool, "model-a")
        assert status["effective_engine"] == "batched"
        assert status["dflash_requested"] is False
        assert status["dflash_active"] is False
        assert status["dflash_fallback_reason"] is None

    def test_unloaded_model_reports_no_effective_engine(self, model_dir):
        pool = _make_pool(model_dir)

        status = _status_for(pool, "model-a")
        assert status["loaded"] is False
        assert status["effective_engine"] is None
        assert status["dflash_requested"] is False
        assert status["dflash_active"] is False
        assert status["dflash_fallback_reason"] is None


class TestEffectiveEngineIdentity:
    @pytest.mark.asyncio
    async def test_effective_engine_differs_from_configured_engine(self, model_dir):
        """force_lm loads a VLM-configured model through BatchedEngine, so the
        effective engine must not simply echo `engine_type`."""
        pool = _make_pool(model_dir)
        entry = pool.get_entry("model-a")
        entry.model_type = "vlm"
        entry.engine_type = "vlm"

        engine = MagicMock()
        engine.start = AsyncMock()

        with patch("omlx.engine_pool.BatchedEngine", return_value=engine):
            await pool._load_engine("model-a", force_lm=True)

        status = _status_for(pool, "model-a")
        assert status["engine_type"] == "vlm"
        assert status["effective_engine"] == "batched"

    @pytest.mark.asyncio
    async def test_unload_clears_effective_state(self, model_dir):
        pool = _make_pool(model_dir)

        with patch("omlx.engine.dflash.DFlashEngine", _dflash_class()):
            await pool._load_engine("model-a", runtime_settings=_dflash_settings())
        assert _status_for(pool, "model-a")["dflash_active"] is True

        await pool._unload_engine("model-a")

        status = _status_for(pool, "model-a")
        assert status["loaded"] is False
        assert status["effective_engine"] is None
        assert status["dflash_requested"] is False
        assert status["dflash_active"] is False
        assert status["dflash_fallback_reason"] is None


class TestStatusPayload:
    """The new keys are additive: existing consumers keep every field they
    already read, and the payload stays JSON-serializable for /v1/models/status.
    """

    _EXISTING_KEYS = {
        "id",
        "model_path",
        "loaded",
        "is_loading",
        "loading_started_at",
        "estimated_size",
        "resident_estimated_size",
        "distributed",
        "actual_size",
        "pinned",
        "engine_type",
        "model_type",
        "config_model_type",
        "realtime_stt",
        "model_context_length",
        "is_helper",
        "thinking_default",
        "preserve_thinking_default",
        "source_type",
        "source_repo_id",
        "last_access",
    }
    _NEW_KEYS = {
        "effective_engine",
        "dflash_requested",
        "dflash_active",
        "dflash_fallback_reason",
    }

    @pytest.mark.asyncio
    async def test_payload_is_serializable_and_keeps_existing_keys(self, model_dir):
        pool = _make_pool(model_dir)
        fallback = MagicMock()
        fallback.start = AsyncMock()

        with (
            patch(
                "omlx.engine.dflash.DFlashEngine",
                _dflash_class(start_error=ValueError("draft rank mismatch")),
            ),
            patch("omlx.engine_pool.BatchedEngine", return_value=fallback),
        ):
            await pool._load_engine("model-a", runtime_settings=_dflash_settings())

        status = pool.get_status()
        assert json.loads(json.dumps(status)) == status

        keys = set(_status_for(pool, "model-a"))
        assert keys >= self._EXISTING_KEYS
        assert keys >= self._NEW_KEYS


class TestSanitizedFallbackReason:
    def test_filesystem_paths_are_redacted(self):
        reason = _sanitized_fallback_reason(
            "cannot open /Users/me/models/draft/model.safetensors"
        )
        assert reason == "cannot open <path>"

    def test_multiline_text_collapses_to_one_line(self):
        reason = _sanitized_fallback_reason(
            'Traceback (most recent call last):\n  File "engine.py", line 4\n'
            "    raise ValueError(msg)\nValueError: bad draft"
        )
        assert "\n" not in reason
        assert reason.startswith("Traceback (most recent call last):")

    def test_long_text_is_truncated_to_the_budget(self):
        reason = _sanitized_fallback_reason("mismatch: " + "weight, " * 200)
        assert len(reason) <= _MAX_FALLBACK_REASON_CHARS
        assert reason.endswith("...")
