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
        assert sched.memory_monitor is None  # init path is paged-SSD-only mode
        # Inject a memory monitor so the method has somewhere to write.
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
        assert kwargs["num_layers"] == 24, (
            "GPT-style ``n_layer`` in the sub-config should be recognized"
        )
