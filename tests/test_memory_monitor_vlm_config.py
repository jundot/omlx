# SPDX-License-Identifier: Apache-2.0
"""Tests for VLM nested-config probing in ``_set_model_info_for_monitor``.

VLM / multimodal models (Qwen3.6-VL, Gemma-4, etc.) nest the language-model
dimensions under ``text_config`` / ``language_config`` / ``llm_config``. The
top-level config may hold *vision tower* dimensions instead; reading the
wrong field underestimates KV+SDPA peak memory for the LM by a constant
factor and lets ``_preflight_memory_check`` approve prefills that go on to
crash Metal.

These tests pin the priority: prefer any sub-config that has the LM layer
count, else fall back to the top-level config.
"""

from unittest.mock import MagicMock

import mlx.core as mx

from omlx.memory_monitor import (
    _SDPA_FALLBACK_SCORE_DTYPE_SIZE,
    _SDPA_FULL_SUPPORTED_HEAD_DIMS,
    _SDPA_VECTOR_QUERY_TOKEN_THRESHOLD,
    _SDPA_VECTOR_SUPPORTED_HEAD_DIMS,
    MemoryMonitor,
)
from omlx.scheduler import Scheduler, SchedulerConfig


def _make_scheduler() -> Scheduler:
    """Return a Scheduler with a mocked model/tokenizer.

    The scheduler is constructed without a paged-SSD cache so
    ``_set_model_info_for_monitor`` runs against the simple init path.
    """
    model = MagicMock()
    model.layers = []
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    config = SchedulerConfig(paged_cache_block_size=0)
    return Scheduler(model=model, tokenizer=tokenizer, config=config)


class _LMConfig:
    """Minimal LM config: 40 layers, 8 KV heads, head_dim=128."""

    num_hidden_layers = 40
    num_key_value_heads = 8
    num_attention_heads = 32
    head_dim = 128
    hidden_size = 4096


class _VLMConfigWithTextConfig:
    """Top-level VLM config: vision-tower dims at the top, LM nested."""

    num_hidden_layers = 33  # vision tower — must be ignored
    num_attention_heads = 16  # vision tower
    text_config = _LMConfig()


class _VLMConfigWithLanguageConfig:
    """Variant using the ``language_config`` attribute name."""

    num_hidden_layers = 33
    language_config = _LMConfig()


class _VLMConfigWithLlmConfig:
    """Variant using the ``llm_config`` attribute name."""

    num_hidden_layers = 33
    llm_config = _LMConfig()


class _PlainLMConfig:
    """Top-level LM config with no nested sub-configs — fallback path."""

    num_hidden_layers = 40
    num_key_value_heads = 8
    num_attention_heads = 32
    head_dim = 128


class _VLMConfigEmptySubConfigs:
    """Sub-configs are present but expose no layer count — skip and fall
    back to the top-level config. Defends against accidentally walking
    into a useless sub-config."""

    num_hidden_layers = 40  # this is the LM at top-level
    num_key_value_heads = 8
    num_attention_heads = 32
    head_dim = 128
    text_config = MagicMock(spec=["something_else"])  # no layer count


class TestSetModelInfoForMonitorVLMWalk:
    """``_set_model_info_for_monitor`` must prefer LM dimensions from a
    nested sub-config when one exists, otherwise fall back to top-level."""

    def test_picks_text_config_over_top_level_vision_dims(self):
        sched = _make_scheduler()
        # Scheduler.__init__ now constructs a MemoryMonitor in
        # estimator-only mode (eviction_enabled=False) so preflight
        # estimation works without prior set_model_info. Replace with
        # a MagicMock so we can inspect the set_model_info call.
        sched.memory_monitor = MagicMock()
        sched.model = MagicMock()
        sched.model.config = _VLMConfigWithTextConfig()
        # ``hasattr`` on a MagicMock auto-creates ``args``; remove it so
        # the config branch picks ``config`` and not ``args``.
        del sched.model.args

        sched._set_model_info_for_monitor()

        sched.memory_monitor.set_model_info.assert_called_once()
        kwargs = sched.memory_monitor.set_model_info.call_args.kwargs
        assert kwargs["num_layers"] == 40, (
            "Should have read the 40-layer LM from text_config, not the "
            "33-layer vision tower at the top level"
        )
        assert kwargs["num_kv_heads"] == 8

    def test_picks_language_config_over_top_level(self):
        sched = _make_scheduler()
        sched.memory_monitor = MagicMock()
        sched.model = MagicMock()
        sched.model.config = _VLMConfigWithLanguageConfig()
        del sched.model.args

        sched._set_model_info_for_monitor()

        kwargs = sched.memory_monitor.set_model_info.call_args.kwargs
        assert kwargs["num_layers"] == 40

    def test_picks_llm_config_over_top_level(self):
        sched = _make_scheduler()
        sched.memory_monitor = MagicMock()
        sched.model = MagicMock()
        sched.model.config = _VLMConfigWithLlmConfig()
        del sched.model.args

        sched._set_model_info_for_monitor()

        kwargs = sched.memory_monitor.set_model_info.call_args.kwargs
        assert kwargs["num_layers"] == 40

    def test_falls_back_to_top_level_when_no_subconfig(self):
        """Plain LM (no sub-configs) must still work — regression guard
        against the walking helper accidentally requiring a sub-config."""
        sched = _make_scheduler()
        sched.memory_monitor = MagicMock()
        sched.model = MagicMock()
        sched.model.config = _PlainLMConfig()
        del sched.model.args

        sched._set_model_info_for_monitor()

        kwargs = sched.memory_monitor.set_model_info.call_args.kwargs
        assert kwargs["num_layers"] == 40

    def test_falls_back_when_subconfig_lacks_layer_count(self):
        """Sub-config without ``num_hidden_layers`` or ``n_layer`` must
        not be selected — the top-level LM dims win."""
        sched = _make_scheduler()
        sched.memory_monitor = MagicMock()
        sched.model = MagicMock()
        sched.model.config = _VLMConfigEmptySubConfigs()
        del sched.model.args

        sched._set_model_info_for_monitor()

        kwargs = sched.memory_monitor.set_model_info.call_args.kwargs
        assert kwargs["num_layers"] == 40

    def test_n_layer_alias_also_triggers_subconfig_selection(self):
        """GPT-style configs use ``n_layer`` instead of ``num_hidden_layers``.
        The walking helper must recognize both."""

        class _GPTStyleLM:
            n_layer = 24
            n_head = 16
            n_embd = 1024

        class _VLMWithGPTStyleSub:
            num_hidden_layers = 12  # vision tower
            text_config = _GPTStyleLM()

        sched = _make_scheduler()
        sched.memory_monitor = MagicMock()
        sched.model = MagicMock()
        sched.model.config = _VLMWithGPTStyleSub()
        del sched.model.args

        sched._set_model_info_for_monitor()

        kwargs = sched.memory_monitor.set_model_info.call_args.kwargs
        assert (
            kwargs["num_layers"] == 24
        ), "GPT-style ``n_layer`` in the sub-config should be recognized"


class TestSetModelInfoTurboQuantDtype:
    def _make_sched_with_config(self, config) -> Scheduler:
        from mlx_lm.models.cache import KVCache

        sched = _make_scheduler()
        sched.memory_monitor = MagicMock()
        sched.model = MagicMock()
        sched.model.config = config
        sched.model.make_cache.return_value = [KVCache() for _ in range(40)]
        del sched.model.args
        return sched

    def test_no_turboquant_uses_full_dtype(self):
        sched = self._make_sched_with_config(_PlainLMConfig())
        sched._turboquant_kv_bits = None

        sched._set_model_info_for_monitor()

        kwargs = sched.memory_monitor.set_model_info.call_args.kwargs
        assert kwargs["dtype_size"] == 2

    def test_turboquant_4bit_without_skip_last_uses_quantized_dtype(self):
        sched = self._make_sched_with_config(_PlainLMConfig())
        sched._turboquant_kv_bits = 4.0
        sched._turboquant_skip_last = False

        sched._set_model_info_for_monitor()

        kwargs = sched.memory_monitor.set_model_info.call_args.kwargs
        expected = 4.0 / 8.0 + 2.0 / 128
        assert abs(kwargs["dtype_size"] - expected) < 1e-9

    def test_turboquant_4bit_default_skip_last_keeps_one_full_dtype_layer(self):
        sched = self._make_sched_with_config(_PlainLMConfig())
        sched._turboquant_kv_bits = 4.0
        sched._turboquant_skip_last = True

        sched._set_model_info_for_monitor()

        kwargs = sched.memory_monitor.set_model_info.call_args.kwargs
        quantized = 4.0 / 8.0 + 2.0 / 128
        expected = (39 * quantized + 2.0) / 40
        assert abs(kwargs["dtype_size"] - expected) < 1e-9

    def test_turboquant_8bit_without_skip_last_uses_quantized_dtype(self):
        sched = self._make_sched_with_config(_PlainLMConfig())
        sched._turboquant_kv_bits = 8.0
        sched._turboquant_skip_last = False

        sched._set_model_info_for_monitor()

        kwargs = sched.memory_monitor.set_model_info.call_args.kwargs
        expected = 8.0 / 8.0 + 2.0 / 128
        assert abs(kwargs["dtype_size"] - expected) < 1e-9

    def test_turboquant_dtype_with_vlm_nested_config(self):
        sched = self._make_sched_with_config(_VLMConfigWithTextConfig())
        sched._turboquant_kv_bits = 4.0
        sched._turboquant_skip_last = False

        sched._set_model_info_for_monitor()

        kwargs = sched.memory_monitor.set_model_info.call_args.kwargs
        assert kwargs["num_layers"] == 40
        expected = 4.0 / 8.0 + 2.0 / 128
        assert abs(kwargs["dtype_size"] - expected) < 1e-9

    def test_turboquant_hybrid_arrays_cache_counts_only_kv_layers(self):
        from mlx_lm.models.cache import ArraysCache, KVCache

        sched = self._make_sched_with_config(_VLMConfigWithTextConfig())
        sched.model.make_cache.return_value = [
            KVCache() if (i + 1) % 4 == 0 else ArraysCache(size=2) for i in range(40)
        ]
        sched._turboquant_kv_bits = 4.0
        sched._turboquant_skip_last = True

        sched._set_model_info_for_monitor()

        kwargs = sched.memory_monitor.set_model_info.call_args.kwargs
        quantized = 4.0 / 8.0 + 2.0 / 128
        expected = (9 * quantized + 2.0) / 10
        assert kwargs["num_kv_cache_layers"] == 10
        assert abs(kwargs["dtype_size"] - expected) < 1e-9

    def test_turboquant_arrays_cache_only_uses_full_dtype(self):
        from mlx_lm.models.cache import ArraysCache

        sched = self._make_sched_with_config(_PlainLMConfig())
        sched.model.make_cache.return_value = [ArraysCache(size=2) for _ in range(40)]
        sched._turboquant_kv_bits = 4.0
        sched._turboquant_skip_last = False

        sched._set_model_info_for_monitor()

        kwargs = sched.memory_monitor.set_model_info.call_args.kwargs
        assert kwargs["dtype_size"] == 2

    def test_turboquant_ineligible_cache_uses_full_dtype(self):
        sched = self._make_sched_with_config(_PlainLMConfig())
        sched.model.make_cache.return_value = [object()]
        sched._turboquant_kv_bits = 4.0
        sched._turboquant_skip_last = False

        sched._set_model_info_for_monitor()

        kwargs = sched.memory_monitor.set_model_info.call_args.kwargs
        assert kwargs["dtype_size"] == 2

    def test_turboquant_mla_model_uses_full_dtype(self):
        class _MLAConfig(_PlainLMConfig):
            kv_lora_rank = 512

        sched = self._make_sched_with_config(_MLAConfig())
        sched._turboquant_kv_bits = 4.0
        sched._turboquant_skip_last = False

        sched._set_model_info_for_monitor()

        kwargs = sched.memory_monitor.set_model_info.call_args.kwargs
        assert kwargs["dtype_size"] == 2

    def test_turboquant_attention_sink_model_uses_full_dtype(self):
        sched = self._make_sched_with_config(_PlainLMConfig())
        sched.model.modules = lambda: [{"sinks": mx.zeros((8,))}]
        sched._turboquant_kv_bits = 4.0
        sched._turboquant_skip_last = True

        sched._set_model_info_for_monitor()

        kwargs = sched.memory_monitor.set_model_info.call_args.kwargs
        assert kwargs["dtype_size"] == 2

    def test_reported_scale_fits_after_turboquant_skip_last_accounting(self):
        tokens = 327_872
        ceiling = 44.0 * 1024**3
        current = 27.17 * 1024**3
        headroom = ceiling - current

        monitor = MemoryMonitor(max_kv_cache_memory=None, eviction_enabled=False)
        monitor.set_model_info(
            num_layers=40,
            num_kv_heads=8,
            head_dim=128,
            dtype_size=2,
            num_attention_heads=32,
            num_kv_cache_layers=40,
        )
        full_dtype_peak = monitor.estimate_prefill_peak_bytes(
            tokens, 2048, cached_tokens=0
        )

        quantized = 4.0 / 8.0 + 2.0 / 128
        skip_last_dtype = (39 * quantized + 2.0) / 40
        monitor.set_model_info(
            num_layers=40,
            num_kv_heads=8,
            head_dim=128,
            dtype_size=skip_last_dtype,
            num_attention_heads=32,
            num_kv_cache_layers=40,
        )
        turboquant_peak = monitor.estimate_prefill_peak_bytes(
            tokens, 2048, cached_tokens=0
        )

        assert full_dtype_peak > headroom
        assert turboquant_peak < headroom


class TestSdpaDispatchEstimate:
    """MemoryMonitor mirrors MLX SDPA full/vector dispatch support."""

    def test_sdpa_dispatch_constants_match_mlx_031(self):
        assert _SDPA_VECTOR_QUERY_TOKEN_THRESHOLD == 8
        assert frozenset({64, 80, 128}) == _SDPA_FULL_SUPPORTED_HEAD_DIMS
        assert frozenset({64, 96, 128, 256}) == _SDPA_VECTOR_SUPPORTED_HEAD_DIMS

    def test_estimate_prefill_uses_full_fallback_for_head_dim_256(self):
        """head_dim=256 is not supported by MLX fused full prefill."""
        monitor = MemoryMonitor(max_kv_cache_memory=None, eviction_enabled=False)
        monitor.set_model_info(
            num_layers=28,
            num_kv_heads=4,
            num_attention_heads=28,
            head_dim=256,
            dtype_size=2,
        )
        n_q = 28
        hd = 256
        chunk = 512
        new_tokens = 327872
        full_kv_len = new_tokens

        eff_chunk = min(chunk, new_tokens)
        output_only = n_q * eff_chunk * hd * 4
        expected_attn = n_q * eff_chunk * full_kv_len * _SDPA_FALLBACK_SCORE_DTYPE_SIZE
        expected_attn += output_only
        kv = monitor.estimate_prompt_kv_bytes(new_tokens)
        expected_peak = expected_attn + kv

        actual = monitor.estimate_prefill_peak_bytes(new_tokens, chunk, cached_tokens=0)
        assert actual == expected_peak, (
            f"head_dim=256 should use full-score fallback formula "
            f"({expected_peak:,} bytes), "
            f"got {actual:,} bytes"
        )
        assert output_only < expected_attn

    def test_estimate_chunk_transient_uses_full_fallback_for_head_dim_256(self):
        """head_dim=256 full prefill transient must include fp32 scores."""
        monitor = MemoryMonitor(max_kv_cache_memory=None, eviction_enabled=False)
        monitor.set_model_info(
            num_layers=28,
            num_kv_heads=4,
            num_attention_heads=28,
            head_dim=256,
            dtype_size=2,
        )
        n_q = 28
        hd = 256
        n_tokens = 512
        kv_len = 327872

        output_only = n_q * n_tokens * hd * 4
        expected = n_q * n_tokens * kv_len * _SDPA_FALLBACK_SCORE_DTYPE_SIZE
        expected += output_only
        actual = monitor.estimate_chunk_transient_bytes(n_tokens, kv_len)
        assert actual == expected, (
            f"head_dim=256 chunk transient should be {expected:,} bytes, "
            f"got {actual:,} bytes"
        )
        assert actual > output_only
