# SPDX-License-Identifier: Apache-2.0
"""Tests for requested-versus-effective DFlash state (issue #2896).

A DFlash request fails soft: a declined or failed batched drafter leaves the
model's natural engine serving, and the request still succeeds with 200.
Before this, an API client had no way to tell a real DFlash load from that
fallback. `/v1/models/status` now carries `effective_engine`,
`dflash_requested`, `dflash_active`, and a redacted `dflash_fallback_reason`.
"""

import json
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
    """One tiny DFlash-capable LLM model directory; loading is always mocked."""
    model_a = tmp_path / "model-a"
    model_a.mkdir()
    (model_a / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (model_a / "model.safetensors").write_bytes(b"0" * 1024)
    return tmp_path


def _batched_engine() -> MagicMock:
    engine = MagicMock()
    engine.start = AsyncMock()
    return engine


def _dflash_settings(**overrides) -> ModelSettings:
    values = {"dflash_enabled": True, "dflash_draft_model": "draft-model"}
    values.update(overrides)
    return ModelSettings(**values)


def _status_for(pool: EnginePool, model_id: str) -> dict:
    return next(m for m in pool.get_status()["models"] if m["id"] == model_id)


class TestDFlashDrafterAttached:
    @pytest.mark.asyncio
    async def test_attached_drafter_reports_active_dflash(self, model_dir):
        pool = _make_pool(model_dir)

        with (
            patch("omlx.engine_pool.BatchedEngine", return_value=_batched_engine()),
            patch("omlx.engine_pool.load_dflash_drafter", return_value=object()),
        ):
            await pool._load_engine("model-a", runtime_settings=_dflash_settings())

        status = _status_for(pool, "model-a")
        assert status["engine_type"] == "batched"
        assert status["effective_engine"] == "batched"
        assert status["dflash_requested"] is True
        assert status["dflash_active"] is True
        assert status["dflash_fallback_reason"] is None


class TestDFlashRequestedButFellBack:
    @pytest.mark.asyncio
    async def test_drafter_load_failure_reports_inactive_dflash_with_reason(
        self, model_dir
    ):
        pool = _make_pool(model_dir)

        with (
            patch("omlx.engine_pool.BatchedEngine", return_value=_batched_engine()),
            patch(
                "omlx.engine_pool.load_dflash_drafter",
                side_effect=ValueError(
                    "Received parameters not in model: "
                    "candidate_selector.predecessor_codebook"
                ),
            ),
        ):
            await pool._load_engine("model-a", runtime_settings=_dflash_settings())

        status = _status_for(pool, "model-a")
        assert status["loaded"] is True
        assert status["effective_engine"] == "batched"
        assert status["dflash_requested"] is True
        assert status["dflash_active"] is False
        assert status["dflash_fallback_reason"] == (
            "DFlash drafter load failed: ValueError: Received parameters not in "
            "model: candidate_selector.predecessor_codebook"
        )

    @pytest.mark.asyncio
    async def test_drafter_failure_reason_hides_checkpoint_path(self, model_dir):
        pool = _make_pool(model_dir)

        with (
            patch("omlx.engine_pool.BatchedEngine", return_value=_batched_engine()),
            patch(
                "omlx.engine_pool.load_dflash_drafter",
                side_effect=RuntimeError(
                    "draft checkpoint /Users/someone/models/draft/model."
                    "safetensors is unreadable"
                ),
            ),
        ):
            await pool._load_engine("model-a", runtime_settings=_dflash_settings())

        status = _status_for(pool, "model-a")
        assert status["dflash_requested"] is True
        assert status["dflash_active"] is False
        reason = status["dflash_fallback_reason"]
        assert reason.startswith("DFlash drafter load failed: RuntimeError:")
        assert "<path>" in reason
        assert "/Users/someone" not in reason

    @pytest.mark.asyncio
    async def test_enabled_without_draft_model_is_reported(self, model_dir):
        pool = _make_pool(model_dir)

        with patch("omlx.engine_pool.BatchedEngine", return_value=_batched_engine()):
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

    @pytest.mark.asyncio
    async def test_unsupported_model_type_reports_reason(self, model_dir):
        pool = _make_pool(model_dir)
        pool.get_entry("model-a").config_model_type = "llama"

        with patch("omlx.engine_pool.BatchedEngine", return_value=_batched_engine()):
            await pool._load_engine("model-a", runtime_settings=_dflash_settings())

        status = _status_for(pool, "model-a")
        assert status["effective_engine"] == "batched"
        assert status["dflash_requested"] is True
        assert status["dflash_active"] is False
        assert status["dflash_fallback_reason"] == (
            "DFlash is not supported for this model type and engine"
        )

    @pytest.mark.asyncio
    async def test_decision_model_reports_declined_dflash(self, model_dir):
        pool = _make_pool(model_dir)
        pool.get_entry("model-a").engine_type = "decision"
        with patch("omlx.engine_pool.DecisionEngine", return_value=_batched_engine()):
            await pool._load_engine("model-a", runtime_settings=_dflash_settings())
        status = _status_for(pool, "model-a")
        assert status["effective_engine"] == "decision"
        assert status["dflash_requested"] is True
        assert status["dflash_active"] is False
        assert status["dflash_fallback_reason"] == (
            "DFlash is not supported for decision models"
        )


class TestDFlashNotRequested:
    @pytest.mark.asyncio
    async def test_plain_load_leaves_dflash_fields_clear(self, model_dir):
        pool = _make_pool(model_dir)

        with patch("omlx.engine_pool.BatchedEngine", return_value=_batched_engine()):
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

        with patch("omlx.engine_pool.BatchedEngine", return_value=_batched_engine()):
            await pool._load_engine("model-a", force_lm=True)

        status = _status_for(pool, "model-a")
        assert status["engine_type"] == "vlm"
        assert status["effective_engine"] == "batched"

    @pytest.mark.asyncio
    async def test_unload_clears_effective_state(self, model_dir):
        pool = _make_pool(model_dir)

        with (
            patch("omlx.engine_pool.BatchedEngine", return_value=_batched_engine()),
            patch("omlx.engine_pool.load_dflash_drafter", return_value=object()),
        ):
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

        with patch("omlx.engine_pool.BatchedEngine", return_value=_batched_engine()):
            await pool._load_engine("model-a", runtime_settings=ModelSettings())

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

    def test_paths_with_spaces_are_redacted(self):
        reason = _sanitized_fallback_reason(
            "cannot open /Users/Joshua Warren/models/draft/model.safetensors"
        )
        assert reason == "cannot open <path>"

    def test_quoted_path_is_redacted(self):
        reason = _sanitized_fallback_reason(
            "DFlash start failed: FileNotFoundError: [Errno 2] "
            "No such file or directory: '/opt/models/draft.safetensors'"
        )
        assert "/opt/models" not in reason
        assert "<path>" in reason

    def test_ratio_is_not_a_path(self):
        reason = _sanitized_fallback_reason("the ratio a/b is wrong")
        assert reason == "the ratio a/b is wrong"

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
