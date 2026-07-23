# SPDX-License-Identifier: Apache-2.0
"""Tests for DFlash engine integration."""

import json
from types import SimpleNamespace

import pytest

from omlx.model_settings import ModelSettings


class TestDFlashModelSettings:
    """Test DFlash fields in ModelSettings."""

    def test_default_values(self):
        settings = ModelSettings()
        assert settings.dflash_enabled is False
        assert settings.dflash_draft_model is None
        assert settings.dflash_draft_quant_enabled is None
        assert settings.dflash_draft_quant_weight_bits is None
        assert settings.dflash_draft_quant_activation_bits is None
        assert settings.dflash_draft_quant_group_size is None
        assert settings.dflash_max_ctx is None
        assert settings.dflash_in_memory_cache is True
        assert settings.dflash_in_memory_cache_max_entries == 4
        assert settings.dflash_in_memory_cache_max_bytes == 8 * 1024 * 1024 * 1024
        assert settings.dflash_ssd_cache is False
        # New long-context tuning knobs (issue #1276). None → dflash-mlx default.
        assert settings.dflash_draft_window_size is None
        assert settings.dflash_draft_sink_size is None
        assert settings.dflash_verify_mode is None

    def test_no_speculative_tokens_field(self):
        """dflash_speculative_tokens was removed in v2 and stays removed."""
        settings = ModelSettings()
        assert not hasattr(settings, "dflash_speculative_tokens")

    def test_to_dict_includes_dflash_fields(self):
        settings = ModelSettings(
            dflash_enabled=True,
            dflash_draft_model="z-lab/Qwen3.5-4B-DFlash",
        )
        d = settings.to_dict()
        assert d["dflash_enabled"] is True
        assert d["dflash_draft_model"] == "z-lab/Qwen3.5-4B-DFlash"

    def test_to_dict_excludes_none_dflash_fields(self):
        settings = ModelSettings(dflash_enabled=True)
        d = settings.to_dict()
        assert "dflash_draft_model" not in d
        assert "dflash_draft_quant_enabled" not in d
        assert "dflash_draft_quant_weight_bits" not in d
        assert "dflash_draft_quant_activation_bits" not in d
        assert "dflash_draft_quant_group_size" not in d
        assert "dflash_max_ctx" not in d
        # Tuning knobs default to None → omitted from on-disk JSON.
        assert "dflash_draft_window_size" not in d
        assert "dflash_draft_sink_size" not in d
        assert "dflash_verify_mode" not in d

    def test_from_dict_with_dflash_fields(self):
        data = {
            "dflash_enabled": True,
            "dflash_draft_model": "z-lab/Qwen3.5-4B-DFlash",
            "dflash_draft_quant_enabled": True,
            "dflash_draft_quant_weight_bits": 4,
            "dflash_draft_quant_activation_bits": 16,
            "dflash_draft_quant_group_size": 64,
            "dflash_max_ctx": 8192,
            "dflash_in_memory_cache": False,
            "dflash_in_memory_cache_max_entries": 16,
            "dflash_in_memory_cache_max_bytes": 4 * 1024 * 1024 * 1024,
            "dflash_ssd_cache": True,
        }
        settings = ModelSettings.from_dict(data)
        assert settings.dflash_enabled is True
        assert settings.dflash_draft_model == "z-lab/Qwen3.5-4B-DFlash"
        assert settings.dflash_draft_quant_enabled is True
        assert settings.dflash_draft_quant_weight_bits == 4
        assert settings.dflash_draft_quant_activation_bits == 16
        assert settings.dflash_draft_quant_group_size == 64
        assert settings.dflash_max_ctx == 8192
        assert settings.dflash_in_memory_cache is False
        assert settings.dflash_in_memory_cache_max_entries == 16
        assert settings.dflash_in_memory_cache_max_bytes == 4 * 1024 * 1024 * 1024
        assert settings.dflash_ssd_cache is True

    def test_from_dict_missing_new_fields_uses_defaults(self):
        """Old settings.json without new fields should fall back to dataclass defaults."""
        data = {
            "dflash_enabled": True,
            "dflash_draft_model": "z-lab/Qwen3.5-4B-DFlash",
        }
        settings = ModelSettings.from_dict(data)
        assert settings.dflash_max_ctx is None
        assert settings.dflash_in_memory_cache is True
        assert settings.dflash_in_memory_cache_max_entries == 4
        assert settings.dflash_in_memory_cache_max_bytes == 8 * 1024 * 1024 * 1024
        assert settings.dflash_ssd_cache is False

    def test_from_dict_ignores_removed_speculative_tokens(self):
        """dflash_speculative_tokens (removed in v2) is silently dropped."""
        data = {
            "dflash_enabled": True,
            "dflash_speculative_tokens": 16,
        }
        settings = ModelSettings.from_dict(data)
        assert settings.dflash_enabled is True
        assert not hasattr(settings, "dflash_speculative_tokens")

    def test_from_dict_accepts_new_tuning_fields(self):
        """Issue #1276 — draft window / sink / verify_mode round-trip from JSON."""
        data = {
            "dflash_enabled": True,
            "dflash_draft_window_size": 2048,
            "dflash_draft_sink_size": 32,
            "dflash_verify_mode": "adaptive",
        }
        settings = ModelSettings.from_dict(data)
        assert settings.dflash_draft_window_size == 2048
        assert settings.dflash_draft_sink_size == 32
        assert settings.dflash_verify_mode == "adaptive"

    def test_roundtrip_serialization(self):
        original = ModelSettings(
            dflash_enabled=True,
            dflash_draft_model="z-lab/Qwen3.5-4B-DFlash",
            dflash_draft_quant_enabled=True,
            dflash_draft_quant_weight_bits=4,
            dflash_draft_quant_activation_bits=16,
            dflash_draft_quant_group_size=64,
            dflash_max_ctx=16384,
            dflash_in_memory_cache=False,
            dflash_ssd_cache=False,
            dflash_ssd_cache_max_bytes=30 * 1024**3,
        )
        d = original.to_dict()
        restored = ModelSettings.from_dict(d)
        assert restored.dflash_enabled == original.dflash_enabled
        assert restored.dflash_draft_model == original.dflash_draft_model
        assert (
            restored.dflash_draft_quant_enabled == original.dflash_draft_quant_enabled
        )
        assert (
            restored.dflash_draft_quant_weight_bits
            == original.dflash_draft_quant_weight_bits
        )
        assert (
            restored.dflash_draft_quant_activation_bits
            == original.dflash_draft_quant_activation_bits
        )
        assert (
            restored.dflash_draft_quant_group_size
            == original.dflash_draft_quant_group_size
        )
        assert restored.dflash_max_ctx == original.dflash_max_ctx
        assert restored.dflash_in_memory_cache == original.dflash_in_memory_cache
        assert restored.dflash_ssd_cache == original.dflash_ssd_cache
        assert (
            restored.dflash_ssd_cache_max_bytes == original.dflash_ssd_cache_max_bytes
        )


class TestDFlashEngineInit:
    """Test DFlashEngine initialization and configuration."""

    def test_import_without_dflash_mlx(self):
        from omlx.engine import DFlashEngine  # noqa: F401

        # Should not raise even if dflash-mlx is not installed

    def test_engine_properties(self):
        try:
            from omlx.engine.dflash import DFlashEngine
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
            draft_quant_enabled=True,
            draft_quant_weight_bits=4,
            draft_quant_activation_bits=16,
            draft_quant_group_size=64,
        )
        assert engine.model_name == "test-model"
        assert engine.tokenizer is None
        assert engine.model_type is None
        assert engine.has_active_requests() is False
        assert engine.scheduler is None

    def test_scheduler_resolves_nested_fallback_scheduler(self):
        try:
            from omlx.engine.dflash import DFlashEngine
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        scheduler = object()
        fallback = SimpleNamespace(
            _engine=SimpleNamespace(engine=SimpleNamespace(scheduler=scheduler))
        )
        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
        )
        engine._fallback_engine = fallback

        assert engine.scheduler is scheduler

    def test_scheduler_config_snapshot_at_construction(self):
        """The engine pool mutates the shared scheduler config on every model
        load, so DFlashEngine must snapshot it at construction time for the
        lazily started fallback engine (PR #2178 follow-up)."""
        try:
            from omlx.engine.dflash import DFlashEngine
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        from omlx.scheduler import SchedulerConfig

        shared_config = SchedulerConfig(
            model_name="model-a", model_path="/models/model-a"
        )
        engine = DFlashEngine(
            model_name="/models/model-a",
            draft_model_path="test-draft",
            scheduler_config=shared_config,
        )

        # Simulate the pool loading another model afterwards
        shared_config.model_name = "model-b"
        shared_config.model_path = "/models/model-b"

        assert engine._scheduler_config.model_name == "model-a"
        assert engine._scheduler_config.model_path == "/models/model-a"

    def test_quant_disabled_keeps_none(self):
        try:
            from omlx.engine.dflash import DFlashEngine
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
        )
        assert engine._draft_quant_enabled is None
        assert engine._draft_quant_weight_bits is None
        assert engine._draft_quant_activation_bits is None
        assert engine._draft_quant_group_size is None

    def test_quant_enabled_true_uses_custom_values(self):
        try:
            from omlx.engine.dflash import DFlashEngine
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
            draft_quant_enabled=True,
            draft_quant_weight_bits=8,
            draft_quant_activation_bits=32,
            draft_quant_group_size=128,
        )
        assert engine._draft_quant_enabled is True
        assert engine._draft_quant_weight_bits == 8
        assert engine._draft_quant_activation_bits == 32
        assert engine._draft_quant_group_size == 128

    def test_get_stats_no_verify_mode(self):
        """Stats should not include verify_mode (removed in v2)."""
        try:
            from omlx.engine.dflash import DFlashEngine
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
        )
        stats = engine.get_stats()
        assert stats["engine_type"] == "dflash"
        assert stats["model_name"] == "test-model"
        assert stats["draft_model"] == "test-draft"
        assert stats["loaded"] is False
        assert "verify_mode" not in stats

    def test_cache_stats_returns_none(self):
        try:
            from omlx.engine.dflash import DFlashEngine
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
        )
        assert engine.get_cache_stats() is None

    def test_stream_events_passes_suppress_token_ids(self, monkeypatch):
        try:
            from dflash_mlx import runtime as dflash_runtime
            from dflash_mlx.server.prefix_cache_flow import PrefixCacheFlow

            from omlx.engine.dflash import DFlashEngine
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
        )
        target_model = object()
        target_ops = object()
        snapshot = object()
        engine._target_model = target_model
        engine._target_ops = target_ops
        engine._executor_tokenizer = object()
        engine._draft_model = object()
        engine._draft_backend = object()
        engine._runtime_context = object()
        engine._suppress_token_ids = {258883, 258882}

        fake_flow = SimpleNamespace(
            snapshot=snapshot,
            snapshot_service=None,
            stable_prefix_len=None,
            cache_active=False,
            publish_generation_snapshot=True,
            hit_kind="l2_prefix",
        )
        captured = {}
        prefix_kwargs = {}

        def fake_for_request(cls, **kwargs):
            prefix_kwargs.update(kwargs)
            return fake_flow

        monkeypatch.setattr(
            PrefixCacheFlow, "for_request", classmethod(fake_for_request)
        )
        monkeypatch.setattr(dflash_runtime, "get_stop_token_ids", lambda tokenizer: [2])

        def fake_stream_dflash_generate(**kwargs):
            captured.update(kwargs)
            return iter(())

        monkeypatch.setattr(
            dflash_runtime,
            "stream_dflash_generate",
            fake_stream_dflash_generate,
        )

        event_iter, _, stop_ids = engine._stream_dflash_events(
            prompt_tokens=[1, 2],
            max_tokens=3,
        )

        assert list(event_iter) == []
        assert stop_ids == [2]
        assert captured["suppress_token_ids"] == [258882, 258883]
        assert captured["prefix_snapshot"] is snapshot
        assert captured["prefix_hit_kind"] == "l2_prefix"
        assert fake_flow.snapshot is None
        assert prefix_kwargs["max_new_tokens"] == 3
        model_provider = prefix_kwargs["model_provider"]
        assert model_provider.model is target_model
        assert model_provider.target_ops is target_ops

    def test_runtime_cache_request_boundary_calls_supported_manager(self, monkeypatch):
        try:
            from dflash_mlx.cache import manager as cache_manager_mod

            from omlx.engine.dflash import DFlashEngine
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        calls = []

        class FakeManager:
            def begin_request(self):
                calls.append("begin")

            def end_request(self):
                calls.append("end")

        fake_manager = FakeManager()
        monkeypatch.setattr(
            cache_manager_mod,
            "current_runtime_cache_manager",
            lambda: fake_manager,
        )

        manager = DFlashEngine._begin_runtime_cache_request()
        DFlashEngine._end_runtime_cache_request(manager)

        assert manager is fake_manager
        assert calls == ["begin", "end"]

    def test_runtime_cache_request_boundary_is_noop_on_old_manager(self, monkeypatch):
        try:
            from dflash_mlx.cache import manager as cache_manager_mod

            from omlx.engine.dflash import DFlashEngine
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        monkeypatch.setattr(
            cache_manager_mod,
            "current_runtime_cache_manager",
            lambda: object(),
        )

        assert DFlashEngine._begin_runtime_cache_request() is None
        DFlashEngine._end_runtime_cache_request(object())

    @pytest.mark.asyncio
    async def test_start_passes_verify_config_to_target_load(self, monkeypatch):
        try:
            from dflash_mlx.runtime import loading as dflash_loading

            from omlx.engine import dflash as dflash_mod
            from omlx.engine.dflash import DFlashEngine
            from omlx.patches import dflash_lifecycle
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        captured = {}

        def fake_load_target_bundle(model_ref, **kwargs):
            captured["model_ref"] = model_ref
            captured.update(kwargs)
            return SimpleNamespace(
                model=SimpleNamespace(),
                tokenizer=SimpleNamespace(
                    name_or_path="fake-target",
                    eos_token_id=1,
                    eos_token_ids=[1],
                ),
                meta={"config": {"model_type": "gemma4"}},
                target_ops=SimpleNamespace(),
            )

        def fake_load_draft_bundle(model_ref, **kwargs):
            captured["draft_model_ref"] = model_ref
            captured["draft_kwargs"] = kwargs
            return SimpleNamespace(), {}

        monkeypatch.setattr(
            dflash_loading, "load_target_bundle", fake_load_target_bundle
        )
        monkeypatch.setattr(dflash_loading, "load_draft_bundle", fake_load_draft_bundle)
        monkeypatch.setattr(
            dflash_mod,
            "maybe_apply_pre_load_patches",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            dflash_lifecycle,
            "install_dflash_lifecycle_wrap",
            lambda: True,
        )
        monkeypatch.setattr(
            dflash_mod,
            "load_generation_config_token_ids",
            lambda *args, **kwargs: set(),
        )
        monkeypatch.setattr(
            dflash_mod, "detect_output_parser", lambda *args, **kwargs: None
        )
        monkeypatch.setattr(dflash_mod, "set_model_info_from_model", lambda *args: None)

        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
            model_settings=ModelSettings(dflash_verify_mode="off"),
        )

        await engine.start()
        try:
            verify_config = captured["verify_config"]
            assert captured["model_ref"] == "test-model"
            assert captured["draft_model_ref"] == "test-draft"
            assert verify_config.mode == "off"
            assert captured["quantize_kv_cache"] is False
        finally:
            await engine.stop()

    def test_should_fallback_unlimited_when_max_ctx_none(self):
        """A None threshold means dflash handles every prompt size."""
        try:
            from omlx.engine.dflash import DFlashEngine
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
            model_settings=ModelSettings(dflash_max_ctx=None),
        )
        assert engine._should_fallback([0] * 10_000) is False

    def test_should_fallback_triggers_at_threshold(self):
        try:
            from omlx.engine.dflash import DFlashEngine
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
            model_settings=ModelSettings(dflash_max_ctx=4096),
        )
        assert engine._should_fallback([0] * 4095) is False
        assert engine._should_fallback([0] * 4096) is True

    def test_build_quant_spec(self):
        try:
            from omlx.engine.dflash import DFlashEngine
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        assert DFlashEngine._build_quant_spec(4, 16, 64) == "w4a16:gs64"
        assert DFlashEngine._build_quant_spec(2, 32, 128) == "w2a32:gs128"
        assert DFlashEngine._build_quant_spec(8, 16, 64) == "w8a16:gs64"

    def test_build_quant_spec_none_fields_fall_back_to_dflash_defaults(self):
        """None bit values must coalesce to dflash 0.1.5 defaults so the spec
        stays parseable when a profile or external API sets enabled=True
        without populating every field."""
        try:
            from omlx.engine.dflash import DFlashEngine
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        assert DFlashEngine._build_quant_spec(None, None, None) == "w4a16:gs64"
        assert DFlashEngine._build_quant_spec(8, None, None) == "w8a16:gs64"
        assert DFlashEngine._build_quant_spec(None, 32, None) == "w4a32:gs64"
        assert DFlashEngine._build_quant_spec(None, None, 128) == "w4a16:gs128"

    def test_resolve_dflash_l2_dir_disabled_when_no_omlx_ssd(self, tmp_path):
        try:
            from omlx.engine.dflash import DFlashEngine
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
            model_settings=ModelSettings(dflash_ssd_cache=True),
            omlx_ssd_cache_dir=None,
        )
        assert engine._resolve_dflash_l2_dir() is None

    def test_resolve_dflash_l2_dir_uses_subdir(self, tmp_path):
        try:
            from omlx.engine.dflash import DFlashEngine
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
            model_settings=ModelSettings(
                dflash_ssd_cache=True,
                dflash_in_memory_cache=True,
            ),
            omlx_ssd_cache_dir=tmp_path,
        )
        resolved = engine._resolve_dflash_l2_dir()
        assert resolved == tmp_path / "dflash_l2"

    def test_resolve_dflash_l2_dir_disabled_when_l1_off(self, tmp_path):
        try:
            from omlx.engine.dflash import DFlashEngine
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
            model_settings=ModelSettings(
                dflash_ssd_cache=True,
                dflash_in_memory_cache=False,
            ),
            omlx_ssd_cache_dir=tmp_path,
        )
        assert engine._resolve_dflash_l2_dir() is None

    def test_long_context_knobs_default_to_none(self):
        """No settings → engine stores None → dflash-mlx fills DEFAULT_RUNTIME_CONFIG."""
        try:
            from omlx.engine.dflash import DFlashEngine
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
        )
        assert engine._draft_window_size is None
        assert engine._draft_sink_size is None
        assert engine._verify_mode is None

    def test_long_context_knobs_read_from_settings(self):
        """Issue #1276 — DFlashEngine picks up window/sink/verify_mode from ModelSettings."""
        try:
            from omlx.engine.dflash import DFlashEngine
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
            model_settings=ModelSettings(
                dflash_draft_window_size=2048,
                dflash_draft_sink_size=32,
                dflash_verify_mode="adaptive",
            ),
        )
        assert engine._draft_window_size == 2048
        assert engine._draft_sink_size == 32
        assert engine._verify_mode == "adaptive"

    def test_build_runtime_context_passes_knobs(self):
        """The new kwargs reach dflash-mlx and end up in RuntimeContext.runtime."""
        try:
            from omlx.engine.dflash import DFlashEngine
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
            model_settings=ModelSettings(
                dflash_draft_window_size=512,
                dflash_draft_sink_size=16,
                dflash_verify_mode="dflash",
            ),
        )
        ctx = engine._build_runtime_context()
        runtime = getattr(ctx, "runtime")
        assert runtime.draft_window_size == 512
        assert runtime.draft_sink_size == 16
        assert runtime.verify_mode == "dflash"

    def test_build_runtime_context_defaults_to_dflash_mlx_values(self):
        """None settings → dflash-mlx fills DEFAULT_RUNTIME_CONFIG (1024 / 64 / 'adaptive')."""
        try:
            from omlx.engine.dflash import DFlashEngine
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
        )
        ctx = engine._build_runtime_context()
        runtime = getattr(ctx, "runtime")
        assert runtime.draft_window_size == 1024
        assert runtime.draft_sink_size == 64
        assert runtime.verify_mode == "adaptive"

    def test_l2_max_bytes_from_settings(self, tmp_path):
        """Issue #1326 — dflash L2 disk budget comes from the per-model setting,
        not a hard-coded 1 TiB sentinel, so dflash_l2/ stays bounded."""
        try:
            from omlx.engine.dflash import DFlashEngine
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
            model_settings=ModelSettings(
                dflash_ssd_cache=True,
                dflash_in_memory_cache=True,
                dflash_ssd_cache_max_bytes=5 * 1024**3,
            ),
            omlx_ssd_cache_dir=tmp_path,
        )
        ctx = engine._build_runtime_context()
        runtime = getattr(ctx, "runtime")
        assert runtime.prefix_cache_l2_max_bytes == 5 * 1024**3

    def test_l2_max_bytes_defaults_to_20gib(self, tmp_path):
        """No explicit setting → engine falls back to the 20 GiB default budget."""
        try:
            from omlx.engine.dflash import DFlashEngine
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
            model_settings=ModelSettings(
                dflash_ssd_cache=True,
                dflash_in_memory_cache=True,
            ),
            omlx_ssd_cache_dir=tmp_path,
        )
        ctx = engine._build_runtime_context()
        runtime = getattr(ctx, "runtime")
        assert runtime.prefix_cache_l2_max_bytes == 20 * 1024**3


class TestDFlashCompatibility:
    """Test the model compatibility helper used to gate the admin UI toggle."""

    def _write_config(self, tmp_path, model_type: str):
        (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))

    def test_qwen_model_is_compatible(self, tmp_path):
        try:
            from omlx.engine.dflash import is_dflash_compatible
        except ImportError:
            pytest.skip("dflash-mlx not installed")
        self._write_config(tmp_path, "qwen3")
        compatible, reason = is_dflash_compatible(tmp_path)
        assert compatible is True
        assert reason == ""

    def test_qwen_moe_is_compatible(self, tmp_path):
        try:
            from omlx.engine.dflash import is_dflash_compatible
        except ImportError:
            pytest.skip("dflash-mlx not installed")
        self._write_config(tmp_path, "qwen3_moe")
        compatible, reason = is_dflash_compatible(tmp_path)
        assert compatible is True

    def test_llama_is_incompatible(self, tmp_path):
        try:
            from omlx.engine.dflash import is_dflash_compatible
        except ImportError:
            pytest.skip("dflash-mlx not installed")
        self._write_config(tmp_path, "llama")
        compatible, reason = is_dflash_compatible(tmp_path)
        assert compatible is False
        assert "Qwen" in reason

    def test_missing_config_is_incompatible(self, tmp_path):
        try:
            from omlx.engine.dflash import is_dflash_compatible
        except ImportError:
            pytest.skip("dflash-mlx not installed")
        compatible, reason = is_dflash_compatible(tmp_path)
        assert compatible is False
        assert "config.json" in reason

    def test_invalid_json_is_incompatible(self, tmp_path):
        try:
            from omlx.engine.dflash import is_dflash_compatible
        except ImportError:
            pytest.skip("dflash-mlx not installed")
        (tmp_path / "config.json").write_text("{not valid json")
        compatible, reason = is_dflash_compatible(tmp_path)
        assert compatible is False
        assert "config.json" in reason

    def test_gemma4_top_level_is_compatible(self, tmp_path):
        try:
            from omlx.engine.dflash import is_dflash_compatible
        except ImportError:
            pytest.skip("dflash-mlx not installed")
        self._write_config(tmp_path, "gemma4")
        compatible, reason = is_dflash_compatible(tmp_path)
        assert compatible is True
        assert reason == ""

    def test_gemma4_text_top_level_is_compatible(self, tmp_path):
        """Top-level model_type=gemma4_text is also accepted (text-only variant)."""
        try:
            from omlx.engine.dflash import is_dflash_compatible
        except ImportError:
            pytest.skip("dflash-mlx not installed")
        self._write_config(tmp_path, "gemma4_text")
        compatible, reason = is_dflash_compatible(tmp_path)
        assert compatible is True
        assert reason == ""

    def test_gemma4_assistant_is_incompatible(self, tmp_path):
        """MTP -assistant variants declare gemma4_assistant at the top level
        even though their text_config.model_type is gemma4_text. The toggle
        must read top-level only to keep these out of the DFlash gate."""
        try:
            from omlx.engine.dflash import is_dflash_compatible
        except ImportError:
            pytest.skip("dflash-mlx not installed")
        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "gemma4_assistant",
                    "text_config": {"model_type": "gemma4_text"},
                }
            )
        )
        compatible, reason = is_dflash_compatible(tmp_path)
        assert compatible is False
        assert "gemma4_assistant" in reason

    def test_gemma3_is_incompatible(self, tmp_path):
        """Gemma3 has no DFlash backend and must not pass the gate."""
        try:
            from omlx.engine.dflash import is_dflash_compatible
        except ImportError:
            pytest.skip("dflash-mlx not installed")
        self._write_config(tmp_path, "gemma3_text")
        compatible, reason = is_dflash_compatible(tmp_path)
        assert compatible is False
        assert "Gemma4" in reason

    def test_incompatible_reason_mentions_both_families(self, tmp_path):
        try:
            from omlx.engine.dflash import is_dflash_compatible
        except ImportError:
            pytest.skip("dflash-mlx not installed")
        self._write_config(tmp_path, "mistral")
        compatible, reason = is_dflash_compatible(tmp_path)
        assert compatible is False
        assert "Qwen" in reason
        assert "Gemma4" in reason


class TestDFlashEnginePoolRouting:
    """Test that EnginePool routes to DFlashEngine based on settings."""

    def test_dflash_disabled_uses_batched(self):
        settings = ModelSettings(dflash_enabled=False)
        assert not getattr(settings, "dflash_enabled", False)

    def test_dflash_enabled_without_draft_model(self):
        settings = ModelSettings(dflash_enabled=True)
        draft = getattr(settings, "dflash_draft_model", None)
        assert draft is None

    def test_dflash_enabled_with_draft_model(self):
        settings = ModelSettings(
            dflash_enabled=True,
            dflash_draft_model="z-lab/Qwen3.5-4B-DFlash",
        )
        assert settings.dflash_enabled is True
        assert settings.dflash_draft_model == "z-lab/Qwen3.5-4B-DFlash"


class TestDFlashThinkPrefix:
    """DFlash bypasses the scheduler, so it must replicate scheduler's
    needs_think_prefix detection. Otherwise reasoning models leak the
    whole thinking block into content (issue #1068)."""

    def _make_engine(self, tokenizer):
        from omlx.engine.dflash import DFlashEngine

        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
        )
        engine._tokenizer_obj = tokenizer
        return engine

    def _tokenizer(
        self, *, think_start_id=None, think_end_id=None, think_start_str="<think>"
    ):
        class _Tok:
            pass

        tok = _Tok()
        tok.unk_token_id = 999
        tok.think_start_id = think_start_id
        tok.think_end_id = think_end_id
        tok.think_start = think_start_str
        return tok

    def test_detect_returns_true_when_prompt_ends_with_think(self):
        try:
            from omlx.engine.dflash import DFlashEngine  # noqa: F401
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        engine = self._make_engine(
            self._tokenizer(
                think_start_id=151667,
                think_end_id=151668,
            )
        )
        # prompt ending: ..., <|im_start|>assistant\n, <think>\n
        assert engine._detect_needs_think_prefix([100, 200, 151667]) is True

    def test_detect_returns_false_when_close_follows_open(self):
        try:
            from omlx.engine.dflash import DFlashEngine  # noqa: F401
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        engine = self._make_engine(
            self._tokenizer(
                think_start_id=151667,
                think_end_id=151668,
            )
        )
        # disabled-thinking pattern: <think></think>
        assert engine._detect_needs_think_prefix([100, 151667, 151668]) is False

    def test_detect_returns_false_when_think_start_id_unavailable(self):
        try:
            from omlx.engine.dflash import DFlashEngine  # noqa: F401
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        # Tokenizer has neither think_start_id nor convert_tokens_to_ids
        tok = self._tokenizer(think_start_id=None)
        engine = self._make_engine(tok)
        assert engine._detect_needs_think_prefix([100, 200, 300]) is False

    def test_detect_returns_false_for_empty_prompt(self):
        try:
            from omlx.engine.dflash import DFlashEngine  # noqa: F401
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        engine = self._make_engine(self._tokenizer(think_start_id=151667))
        assert engine._detect_needs_think_prefix([]) is False

    def test_detect_returns_false_when_think_not_in_tail(self):
        try:
            from omlx.engine.dflash import DFlashEngine  # noqa: F401
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        engine = self._make_engine(self._tokenizer(think_start_id=151667))
        # <think> appears earlier but not in last 3 — already inside an
        # assistant turn, so a fresh prefix is not needed
        assert engine._detect_needs_think_prefix([151667, 1, 2, 3, 4, 5]) is False

    def test_think_prefix_text_uses_tokenizer_attr(self):
        try:
            from omlx.engine.dflash import DFlashEngine  # noqa: F401
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        engine = self._make_engine(
            self._tokenizer(
                think_start_str="<longcat_think>",
            )
        )
        assert engine._think_prefix_text() == "<longcat_think>\n"

    def test_think_prefix_text_default(self):
        try:
            from omlx.engine.dflash import DFlashEngine  # noqa: F401
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        # Tokenizer with no think_start attr falls back to <think>
        class _Tok:
            pass

        engine = self._make_engine(_Tok())
        assert engine._think_prefix_text() == "<think>\n"


class TestDFlashApplyChatTemplatePartialMode:
    """Regression tests for partial-mode is_partial plumbing on DFlashEngine.

    Mirrors TestApplyChatTemplatePartialMode in tests/test_batched_engine.py.
    Catches the gap that a sibling text engine wasn't updated alongside
    BatchedEngine when the API server began forwarding ``is_partial``.
    """

    def test_count_then_apply_chat_template_idempotent_under_partial_mode(self):
        """Server flow: count_chat_tokens then _apply_chat_template on the
        same messages list must render with identical partial-mode flags.

        Mirrors the BatchedEngine regression test.  Without is_partial
        plumbing on DFlashEngine, the API server's
        ``count_chat_tokens(messages, ..., is_partial=is_partial)`` would
        raise TypeError, and chat-path ``is_partial`` forwarding via
        ``**kwargs`` would never reach ``_apply_chat_template`` --
        re-introducing the in-place message mutation bug for any
        dflash-routed request.
        """
        try:
            from omlx.engine.dflash import DFlashEngine
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        from unittest.mock import MagicMock

        from omlx.api.utils import detect_and_strip_partial

        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "<formatted>"
        mock_tokenizer.encode.return_value = [1, 2, 3]
        engine._tokenizer_obj = mock_tokenizer

        messages = [
            {"role": "user", "content": "Generate JSON"},
            {"role": "assistant", "content": "{", "partial": True},
        ]

        # Server flow: detect_and_strip_partial once at the API boundary,
        # forward the resolved value to all engine methods.
        is_partial = detect_and_strip_partial(messages)
        assert is_partial is True

        # Phase 1: count.
        engine.count_chat_tokens(messages, is_partial=is_partial)
        count_kwargs = dict(mock_tokenizer.apply_chat_template.call_args.kwargs)

        # Phase 2: chat.  Operates on the same (now-stripped) messages list.
        engine._apply_chat_template(messages, is_partial=is_partial)
        chat_kwargs = dict(mock_tokenizer.apply_chat_template.call_args.kwargs)

        # Both phases must render with identical partial-mode flags.
        assert count_kwargs.get("continue_final_message") == chat_kwargs.get(
            "continue_final_message"
        ), (
            "continue_final_message diverged across phases: "
            f"count={count_kwargs.get('continue_final_message')}, "
            f"chat={chat_kwargs.get('continue_final_message')}"
        )
        assert (
            count_kwargs["add_generation_prompt"]
            == chat_kwargs["add_generation_prompt"]
        ), (
            "add_generation_prompt diverged across phases: "
            f"count={count_kwargs['add_generation_prompt']}, "
            f"chat={chat_kwargs['add_generation_prompt']}"
        )

        # Specific contract: with partial=True forwarded, both phases use
        # continue_final_message=True (not add_generation_prompt=True).
        assert count_kwargs["continue_final_message"] is True
        assert count_kwargs["add_generation_prompt"] is False


class TestDFlashOutputParserWiring:
    """Regression tests for OutputParserSession integration on DFlashEngine.

    Without this wiring gemma4's raw `<|channel>thought\\n` /  `<channel|>`
    protocol markers leak into the response body because dflash bypasses
    the scheduler that normally drives the parser. The tests below pin the
    factory plumbing without booting the full mlx model — full marker
    conversion is verified in the real-server smoke run.
    """

    def test_output_parser_factory_defaults_to_none(self):
        try:
            from omlx.engine.dflash import DFlashEngine
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
        )
        assert engine._output_parser_factory is None

    def test_detect_output_parser_returns_gemma4_factory(self):
        """``detect_output_parser`` (the helper dflash.start() uses) must
        recognise gemma4 by config and hand back a factory whose
        ``create_session`` produces a ``Gemma4OutputParserSession``."""
        from unittest.mock import MagicMock

        from omlx.adapter.gemma4 import Gemma4OutputParserSession
        from omlx.adapter.output_parser import detect_output_parser

        tokenizer = MagicMock()
        factory = detect_output_parser(
            "/some/path/gemma-4-26b-a4b-it-8bit",
            tokenizer,
            {"model_type": "gemma4_text"},
        )
        assert factory is not None
        assert factory.kind == "gemma4"
        session = factory.create_session(tokenizer)
        assert isinstance(session, Gemma4OutputParserSession)

    def test_detect_output_parser_returns_none_for_qwen(self):
        """Qwen models have no protocol parser — dflash should stay on the
        existing detokenizer / think_prefix path."""
        from unittest.mock import MagicMock

        from omlx.adapter.output_parser import detect_output_parser

        factory = detect_output_parser(
            "/some/path/Qwen3-4B-bf16",
            MagicMock(),
            {"model_type": "qwen3"},
        )
        assert factory is None


class TestDFlashCachedTokens:
    """#1441: DFlash must surface prefix-cache hits as cached_tokens.

    When a DFlash prefix snapshot hits, prefill is skipped for the matched
    tokens (PrefixCacheFlow.hit_tokens). The engine previously never set
    cached_tokens on its GenerationOutput, so the API always reported 0 with
    DFlash enabled (restored when disabled). These cover the pure mapping; the
    end-to-end wiring is exercised by the CI-gated integration test below.
    """

    def test_hit_reports_matched_tokens(self):
        from omlx.engine.dflash import DFlashEngine

        flow = SimpleNamespace(hit_tokens=4273)
        assert DFlashEngine._cached_tokens_from_flow(flow) == 4273

    def test_miss_is_zero(self):
        from omlx.engine.dflash import DFlashEngine

        assert DFlashEngine._cached_tokens_from_flow(SimpleNamespace(hit_tokens=0)) == 0

    def test_none_flow_is_zero(self):
        from omlx.engine.dflash import DFlashEngine

        assert DFlashEngine._cached_tokens_from_flow(None) == 0

    def test_missing_attr_is_zero(self):
        from omlx.engine.dflash import DFlashEngine

        assert DFlashEngine._cached_tokens_from_flow(SimpleNamespace()) == 0

    def test_negative_is_clamped(self):
        from omlx.engine.dflash import DFlashEngine

        assert (
            DFlashEngine._cached_tokens_from_flow(SimpleNamespace(hit_tokens=-5)) == 0
        )


class TestDFlashCachedTokensWiring:
    """End-to-end: a prefix hit reaches GenerationOutput.cached_tokens.

    Requires dflash-mlx (for SummaryEvent / the inner event loop); skipped where
    it is unavailable, runs in CI.
    """

    @pytest.mark.asyncio
    async def test_generate_sets_cached_tokens_from_hit(self, monkeypatch):
        try:
            from dflash_mlx.engine.events import SummaryEvent
        except ImportError:
            pytest.skip("dflash-mlx not installed")
        from omlx.engine.dflash import DFlashEngine

        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
            model_settings=ModelSettings(),
        )
        engine._loaded = True
        engine._tokenizer_obj = SimpleNamespace(
            encode=lambda s: [1, 2, 3],
            decode=lambda toks, **kw: "hi",
        )
        engine._output_parser_factory = None
        engine._should_fallback = lambda toks: False
        engine._detect_needs_think_prefix = lambda toks: False

        summary = SummaryEvent(
            elapsed_us=1000,
            prompt_token_count=4273,
            generated_token_ids=(5,),
            generation_tokens=1,
            accepted_from_draft=0,
            acceptance_ratio=0.0,
            cycles_completed=1,
            phase_timings_us={},
        )
        fake_flow = SimpleNamespace(hit_tokens=4273)

        def fake_stream_events(*, prompt_tokens, max_tokens):
            return iter([summary]), fake_flow, [2]

        monkeypatch.setattr(engine, "_stream_dflash_events", fake_stream_events)

        out = await engine.generate("hello", max_tokens=4)
        assert out.cached_tokens == 4273
