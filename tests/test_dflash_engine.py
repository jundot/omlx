# SPDX-License-Identifier: Apache-2.0
"""Tests for DFlash engine integration."""

import json
import sys
import types

import pytest

from omlx.model_settings import ModelSettings


class TestDFlashModelSettings:
    """Test DFlash fields in ModelSettings."""

    def test_default_values(self):
        settings = ModelSettings()
        assert settings.dflash_enabled is False
        assert settings.dflash_draft_model is None
        assert settings.dflash_draft_quant_bits is None
        assert settings.dflash_max_ctx is None
        assert settings.dflash_in_memory_cache is True
        assert settings.dflash_in_memory_cache_max_bytes == 8 * 1024 * 1024 * 1024
        assert settings.dflash_ssd_cache is False

    def test_no_verify_mode_field(self):
        """verify_mode and speculative_tokens were removed in v2."""
        settings = ModelSettings()
        assert not hasattr(settings, "dflash_verify_mode")
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
        assert "dflash_draft_quant_bits" not in d
        assert "dflash_max_ctx" not in d

    def test_from_dict_with_dflash_fields(self):
        data = {
            "dflash_enabled": True,
            "dflash_draft_model": "z-lab/Qwen3.5-4B-DFlash",
            "dflash_draft_quant_bits": 4,
            "dflash_max_ctx": 8192,
            "dflash_in_memory_cache": False,
            "dflash_in_memory_cache_max_bytes": 4 * 1024 * 1024 * 1024,
            "dflash_ssd_cache": True,
        }
        settings = ModelSettings.from_dict(data)
        assert settings.dflash_enabled is True
        assert settings.dflash_draft_model == "z-lab/Qwen3.5-4B-DFlash"
        assert settings.dflash_draft_quant_bits == 4
        assert settings.dflash_max_ctx == 8192
        assert settings.dflash_in_memory_cache is False
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
        assert settings.dflash_in_memory_cache_max_bytes == 8 * 1024 * 1024 * 1024
        assert settings.dflash_ssd_cache is False

    def test_from_dict_ignores_removed_fields(self):
        """Old settings with verify_mode/speculative_tokens should be ignored."""
        data = {
            "dflash_enabled": True,
            "dflash_verify_mode": "parallel-replay",
            "dflash_speculative_tokens": 16,
        }
        settings = ModelSettings.from_dict(data)
        assert settings.dflash_enabled is True

    def test_roundtrip_serialization(self):
        original = ModelSettings(
            dflash_enabled=True,
            dflash_draft_model="z-lab/Qwen3.5-4B-DFlash",
            dflash_draft_quant_bits=4,
            dflash_max_ctx=16384,
            dflash_in_memory_cache=False,
            dflash_ssd_cache=False,
        )
        d = original.to_dict()
        restored = ModelSettings.from_dict(d)
        assert restored.dflash_enabled == original.dflash_enabled
        assert restored.dflash_draft_model == original.dflash_draft_model
        assert restored.dflash_draft_quant_bits == original.dflash_draft_quant_bits
        assert restored.dflash_max_ctx == original.dflash_max_ctx
        assert restored.dflash_in_memory_cache == original.dflash_in_memory_cache
        assert restored.dflash_ssd_cache == original.dflash_ssd_cache


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
            draft_quant_bits=4,
        )
        assert engine.model_name == "test-model"
        assert engine.tokenizer is None
        assert engine.model_type is None
        assert engine.has_active_requests() is False

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

    def test_bits_to_quant_spec(self):
        try:
            from omlx.engine.dflash import DFlashEngine
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        assert DFlashEngine._bits_to_quant_spec(None) is None
        assert DFlashEngine._bits_to_quant_spec(4) == "w4"
        assert DFlashEngine._bits_to_quant_spec(8) == "w8"
        with pytest.raises(ValueError):
            DFlashEngine._bits_to_quant_spec(2)

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


class TestImageContentDetection:
    """Helpers that decide whether a chat request carries vision content.

    These gate DFlash's vision-fallback path: when any message contains an
    image content part, DFlash must delegate to its VLM fallback rather than
    flatten the messages through ``_apply_chat_template`` (which silently
    drops the image).
    """

    def _import_helpers(self):
        try:
            from omlx.engine.dflash import (
                _content_has_image_part,
                _messages_have_images,
            )
        except ImportError:
            pytest.skip("dflash-mlx not installed")
        return _content_has_image_part, _messages_have_images

    def test_text_only_string_content(self):
        _content_has_image_part, _ = self._import_helpers()
        assert _content_has_image_part("hello") is False

    def test_text_only_list_content(self):
        _content_has_image_part, _ = self._import_helpers()
        content = [{"type": "text", "text": "hi"}]
        assert _content_has_image_part(content) is False

    def test_image_url_part(self):
        _content_has_image_part, _ = self._import_helpers()
        content = [
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
        ]
        assert _content_has_image_part(content) is True

    def test_image_part_alias(self):
        _content_has_image_part, _ = self._import_helpers()
        assert _content_has_image_part([{"type": "image", "image": "..."}]) is True

    def test_input_image_part_alias(self):
        _content_has_image_part, _ = self._import_helpers()
        assert _content_has_image_part([{"type": "input_image", "image": "..."}]) is True

    def test_messages_have_images_false_for_text_only(self):
        _, _messages_have_images = self._import_helpers()
        messages = [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hi"},
        ]
        assert _messages_have_images(messages) is False

    def test_messages_have_images_true_when_any_message_has_image(self):
        _, _messages_have_images = self._import_helpers()
        messages = [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": [
                {"type": "text", "text": "what's here?"},
                {"type": "image_url", "image_url": {"url": "https://..."}},
            ]},
        ]
        assert _messages_have_images(messages) is True

    def test_messages_have_images_handles_non_dict_messages(self):
        _, _messages_have_images = self._import_helpers()
        # Defensive: callers occasionally pass strings or odd shapes
        assert _messages_have_images(["plain string", None]) is False  # type: ignore[list-item]

    def test_object_with_type_attr_treated_as_image(self):
        """Pydantic-style content parts use attribute access, not dict get."""
        _content_has_image_part, _ = self._import_helpers()

        class _Part:
            def __init__(self, t):
                self.type = t

        assert _content_has_image_part([_Part("image_url")]) is True
        assert _content_has_image_part([_Part("text")]) is False


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

    def _tokenizer(self, *, think_start_id=None, think_end_id=None,
                   think_start_str="<think>"):
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

        engine = self._make_engine(self._tokenizer(
            think_start_id=151667, think_end_id=151668,
        ))
        # prompt ending: ..., <|im_start|>assistant\n, <think>\n
        assert engine._detect_needs_think_prefix([100, 200, 151667]) is True

    def test_detect_returns_false_when_close_follows_open(self):
        try:
            from omlx.engine.dflash import DFlashEngine  # noqa: F401
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        engine = self._make_engine(self._tokenizer(
            think_start_id=151667, think_end_id=151668,
        ))
        # disabled-thinking pattern: <think></think>
        assert engine._detect_needs_think_prefix(
            [100, 151667, 151668]
        ) is False

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
        assert engine._detect_needs_think_prefix(
            [151667, 1, 2, 3, 4, 5]
        ) is False

    def test_think_prefix_text_uses_tokenizer_attr(self):
        try:
            from omlx.engine.dflash import DFlashEngine  # noqa: F401
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        engine = self._make_engine(self._tokenizer(
            think_start_str="<longcat_think>",
        ))
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


class TestDFlashFallbackCoexistence:
    """DFlash + sidecar fallback coexistence (vision routing).

    Vision-bearing chat requests must lazy-init a sidecar fallback engine and
    delegate to it WITHOUT releasing the DFlash models. Subsequent text-only
    requests can then continue to use DFlash speculation. This is distinct
    from the long-context overflow path, which deliberately evicts DFlash
    to free memory for a larger context window.
    """

    @staticmethod
    def _patch_dflash_mlx(monkeypatch):
        """Stub the dflash_mlx prefix-cache import the eviction path uses,
        so any accidental call into eviction during a coexistence test is
        observable via state changes rather than ImportError."""
        fake_mod = types.ModuleType("dflash_mlx.server.prefix_cache_flow")
        fake_mod.shutdown_dflash_prefix_cache = lambda: None
        monkeypatch.setitem(
            sys.modules, "dflash_mlx", types.ModuleType("dflash_mlx")
        )
        monkeypatch.setitem(
            sys.modules,
            "dflash_mlx.server",
            types.ModuleType("dflash_mlx.server"),
        )
        monkeypatch.setitem(
            sys.modules, "dflash_mlx.server.prefix_cache_flow", fake_mod
        )

    @staticmethod
    def _patch_vlm(monkeypatch, recorder):
        from omlx.engine import vlm as vlm_module
        from omlx.engine.base import GenerationOutput

        class _FakeVLM:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.started = False
                self.stopped = False
                self.chat_calls = []
                self.stream_chat_calls = []
                recorder.append(self)

            async def start(self):
                self.started = True

            async def stop(self):
                self.stopped = True

            async def chat(self, **kwargs):
                self.chat_calls.append(kwargs)
                return GenerationOutput(
                    text="ok", prompt_tokens=10, completion_tokens=1,
                    cached_tokens=0, finish_reason="stop",
                )

            async def stream_chat(self, **kwargs):
                self.stream_chat_calls.append(kwargs)
                yield GenerationOutput(
                    text="ok", prompt_tokens=10, completion_tokens=1,
                    cached_tokens=0, finish_reason="stop",
                )

            def has_active_requests(self):
                return False

        monkeypatch.setattr(vlm_module, "VLMBatchedEngine", _FakeVLM)

    @staticmethod
    def _make_engine():
        from omlx.engine.dflash import DFlashEngine

        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
            fallback_engine_type="vlm",
        )
        engine._loaded = True
        # Sentinel objects so we can verify the refs survive coexistence.
        engine._target_model = object()
        engine._draft_model = object()
        engine._dflash_prefix_cache = object()
        engine._runtime_context = object()
        engine._executor_tokenizer = object()
        return engine

    async def test_ensure_fallback_engine_started_lazy_inits_vlm(self, monkeypatch):
        try:
            from omlx.engine.dflash import DFlashEngine  # noqa: F401
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        recorder = []
        self._patch_vlm(monkeypatch, recorder)

        engine = self._make_engine()
        await engine._ensure_fallback_engine_started()

        assert len(recorder) == 1
        assert recorder[0].started is True
        assert engine._fallback_engine is recorder[0]

    async def test_ensure_fallback_engine_started_is_idempotent(self, monkeypatch):
        try:
            from omlx.engine.dflash import DFlashEngine  # noqa: F401
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        recorder = []
        self._patch_vlm(monkeypatch, recorder)

        engine = self._make_engine()
        await engine._ensure_fallback_engine_started()
        await engine._ensure_fallback_engine_started()
        await engine._ensure_fallback_engine_started()

        assert len(recorder) == 1

    async def test_ensure_fallback_engine_started_keeps_dflash_loaded(self, monkeypatch):
        try:
            from omlx.engine.dflash import DFlashEngine  # noqa: F401
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        recorder = []
        self._patch_vlm(monkeypatch, recorder)

        engine = self._make_engine()
        target_ref = engine._target_model
        draft_ref = engine._draft_model

        await engine._ensure_fallback_engine_started()

        assert engine._target_model is target_ref
        assert engine._draft_model is draft_ref
        assert engine._in_fallback_mode is False

    async def test_chat_with_image_keeps_dflash_loaded(self, monkeypatch):
        try:
            from omlx.engine.dflash import DFlashEngine  # noqa: F401
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        recorder = []
        self._patch_vlm(monkeypatch, recorder)
        self._patch_dflash_mlx(monkeypatch)

        engine = self._make_engine()
        target_ref = engine._target_model
        draft_ref = engine._draft_model

        messages = [{"role": "user", "content": [
            {"type": "text", "text": "what's here?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
        ]}]
        result = await engine.chat(messages=messages)

        assert len(recorder) == 1
        assert len(recorder[0].chat_calls) == 1
        assert recorder[0].chat_calls[0]["messages"] == messages
        # Coexistence: DFlash refs must NOT have been released.
        assert engine._target_model is target_ref
        assert engine._draft_model is draft_ref
        assert engine._in_fallback_mode is False
        assert result.text == "ok"

    async def test_stream_chat_with_image_keeps_dflash_loaded(self, monkeypatch):
        try:
            from omlx.engine.dflash import DFlashEngine  # noqa: F401
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        recorder = []
        self._patch_vlm(monkeypatch, recorder)
        self._patch_dflash_mlx(monkeypatch)

        engine = self._make_engine()
        target_ref = engine._target_model
        draft_ref = engine._draft_model

        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "..."}},
        ]}]
        outputs = []
        async for output in engine.stream_chat(messages=messages):
            outputs.append(output)

        assert len(outputs) == 1
        assert len(recorder[0].stream_chat_calls) == 1
        assert engine._target_model is target_ref
        assert engine._draft_model is draft_ref
        assert engine._in_fallback_mode is False

    async def test_two_image_requests_share_one_fallback_engine(self, monkeypatch):
        try:
            from omlx.engine.dflash import DFlashEngine  # noqa: F401
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        recorder = []
        self._patch_vlm(monkeypatch, recorder)
        self._patch_dflash_mlx(monkeypatch)

        engine = self._make_engine()
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "..."}},
        ]}]
        await engine.chat(messages=messages)
        await engine.chat(messages=messages)

        assert len(recorder) == 1
        assert len(recorder[0].chat_calls) == 2

    def test_supports_vision_true_when_fallback_is_vlm(self):
        try:
            from omlx.engine.dflash import DFlashEngine
        except ImportError:
            pytest.skip("dflash-mlx not installed")
        engine = DFlashEngine(
            model_name="m", draft_model_path="d", fallback_engine_type="vlm",
        )
        assert engine.supports_vision is True

    def test_supports_vision_false_when_fallback_is_batched(self):
        try:
            from omlx.engine.dflash import DFlashEngine
        except ImportError:
            pytest.skip("dflash-mlx not installed")
        engine = DFlashEngine(
            model_name="m", draft_model_path="d", fallback_engine_type="batched",
        )
        assert engine.supports_vision is False

    async def test_get_stats_reports_sidecar_loaded(self, monkeypatch):
        """get_stats() must distinguish 'sidecar started for vision' from
        'in fallback mode' (long-context eviction)."""
        try:
            from omlx.engine.dflash import DFlashEngine  # noqa: F401
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        recorder = []
        self._patch_vlm(monkeypatch, recorder)

        engine = self._make_engine()
        assert engine.get_stats()["sidecar_loaded"] is False
        assert engine.get_stats()["in_fallback_mode"] is False

        await engine._ensure_fallback_engine_started()
        assert engine.get_stats()["sidecar_loaded"] is True
        assert engine.get_stats()["in_fallback_mode"] is False

    async def test_evict_still_releases_dflash_when_called(self, monkeypatch):
        """Long-context eviction path must still release DFlash refs and flip
        the flag, even if the fallback engine was already started by a prior
        vision request (coexistence path)."""
        try:
            from omlx.engine.dflash import DFlashEngine  # noqa: F401
        except ImportError:
            pytest.skip("dflash-mlx not installed")

        recorder = []
        self._patch_vlm(monkeypatch, recorder)
        self._patch_dflash_mlx(monkeypatch)

        engine = self._make_engine()
        # Prime: vision request lazy-inits the sidecar.
        await engine._ensure_fallback_engine_started()
        assert engine._target_model is not None

        # Long-context overflow now arrives → eviction.
        await engine._evict_dflash_and_start_fallback()

        # DFlash refs released, flag flipped, fallback NOT recreated.
        assert engine._target_model is None
        assert engine._draft_model is None
        assert engine._in_fallback_mode is True
        assert len(recorder) == 1
