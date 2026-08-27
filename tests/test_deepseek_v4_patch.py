# SPDX-License-Identifier: Apache-2.0
"""Tests for the DeepSeek V4 monkey-patch (PR 1192 port)."""

import importlib
import inspect
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture(scope="module")
def applied_patch():
    """Apply the patch once for the whole module. The patch itself is
    idempotent so repeated calls are safe."""
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()
    return True


class TestPatchOrchestration:
    """Top-level apply / idempotency / module registration checks."""

    def test_apply_returns_true_first_time(self):
        from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch, is_applied

        # The patch may have been applied by a previous test run in the
        # same process; force-reset is_applied to validate the flow.
        # The module-level _APPLIED guard means we cannot un-apply, so
        # this test is informational about the *current* state.
        if is_applied():
            assert apply_deepseek_v4_patch() is False
        else:
            assert apply_deepseek_v4_patch() is True
            assert is_applied() is True

    def test_apply_is_idempotent(self, applied_patch):
        from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

        # After fixture has applied the patch, a second call must return False.
        assert apply_deepseek_v4_patch() is False

    def test_hyper_connection_registered(self, applied_patch):
        assert "mlx_lm.models.hyper_connection" in sys.modules

    def test_deepseek_v4_registered(self, applied_patch):
        assert "mlx_lm.models.deepseek_v4" in sys.modules

    def test_deepseek_v4_module_package(self, applied_patch):
        mod = sys.modules["mlx_lm.models.deepseek_v4"]
        # __package__ must be mlx_lm.models so relative imports inside
        # the loaded file resolve through the real mlx_lm package.
        assert mod.__package__ == "mlx_lm.models"

    def test_deepseek_v4_mtp_alias_registered(self, applied_patch):
        assert (
            sys.modules["mlx_lm.models.deepseek_v4_mtp"]
            is sys.modules["mlx_lm.models.deepseek_v4"]
        )


class TestCacheInjection:
    """PoolingCache / BatchPoolingCache injected into mlx_lm.models.cache."""

    def test_pooling_cache_attribute(self, applied_patch):
        import mlx_lm.models.cache as cache_mod

        assert hasattr(cache_mod, "PoolingCache")
        assert hasattr(cache_mod, "BatchPoolingCache")

    def test_pooling_cache_module_attribute(self, applied_patch):
        from mlx_lm.models.cache import BatchPoolingCache, PoolingCache

        # The injected classes claim to live in mlx_lm.models.cache so
        # any introspection (e.g. type(c).__module__) sees the right name.
        assert PoolingCache.__module__ == "mlx_lm.models.cache"
        assert BatchPoolingCache.__module__ == "mlx_lm.models.cache"

    def test_pooling_cache_instantiation(self, applied_patch):
        from mlx_lm.models.cache import PoolingCache

        cache = PoolingCache(ratio=4)
        assert cache.ratio == 4
        assert cache.empty()
        assert cache.size() == 0
        assert cache.offset == 0


def test_b1_cache_offset_uses_host_scalar_without_sync(applied_patch):
    import mlx.core as mx

    dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]
    cache = SimpleNamespace(offset=mx.array([15360]), _offset=15360)

    old = dsv4._DEEPSEEK_V4_B1_SCALAR_OFFSET
    dsv4._DEEPSEEK_V4_B1_SCALAR_OFFSET = True
    assert dsv4._b1_cache_offset(cache, 1) == 15360
    assert dsv4._b1_cache_offset(cache, 2) is cache.offset
    dsv4._DEEPSEEK_V4_B1_SCALAR_OFFSET = False
    assert dsv4._b1_cache_offset(cache, 1) is cache.offset
    dsv4._DEEPSEEK_V4_B1_SCALAR_OFFSET = old


def test_restored_singleton_merge_preserves_absolute_rotating_offset(applied_patch):
    import mlx.core as mx
    from mlx_lm.models.cache import BatchRotatingKVCache, RotatingKVCache

    cache = RotatingKVCache(max_size=128)
    cache.keys = mx.zeros((1, 1, 1151, 8), dtype=mx.bfloat16)
    cache.values = mx.zeros((1, 1, 1151, 0), dtype=mx.bfloat16)
    cache.offset = 4096
    cache._idx = 1151

    merged = BatchRotatingKVCache.merge([cache])
    mx.eval(merged.offset)

    assert merged.offset.tolist() == [4096]
    assert merged._offset == 4096
    assert merged.keys.shape[2] == 128


class TestUtilsPatch:
    """mlx_lm.utils.load_model + _load_safetensors + SAFETENSORS_DTYPE_FALLBACKS."""

    def test_load_model_replaced(self, applied_patch):
        import mlx_lm.utils as utils_mod

        # The replaced function carries our docstring marker via its
        # bound name; just check it's not the upstream one by virtue of
        # the new attributes around it.
        assert hasattr(utils_mod, "_load_safetensors")
        assert hasattr(utils_mod, "SAFETENSORS_DTYPE_FALLBACKS")

    def test_dtype_fallback_map(self, applied_patch):
        import mlx_lm.utils as utils_mod

        assert utils_mod.SAFETENSORS_DTYPE_FALLBACKS == {"F8_E8M0": "U8"}

    def test_load_safetensors_passthrough_for_normal_dtype(
        self, applied_patch, tmp_path
    ):
        """A safetensors file with a standard dtype must round-trip
        through _load_safetensors unchanged (no header rewrite)."""
        import mlx.core as mx
        from mlx_lm.utils import _load_safetensors

        path = tmp_path / "model.safetensors"
        data = {"x": mx.zeros((4, 4), dtype=mx.float32)}
        mx.save_safetensors(str(path), data)
        loaded = _load_safetensors(str(path))
        assert "x" in loaded
        assert loaded["x"].shape == (4, 4)

    def test_sub4_v4_disables_compressed_native_attention(self):
        from omlx.patches.deepseek_v4.utils_patch import (
            _native_ratio128_attention_enabled,
        )

        assert (
            _native_ratio128_attention_enabled(
                {"model_type": "deepseek_v4", "quantization": {"bits": 2}}
            )
            is False
        )
        assert (
            _native_ratio128_attention_enabled(
                {
                    "model_type": "deepseek_v4",
                    "text_config": {"quantization_config": {"bits": 3.5}},
                }
            )
            is False
        )
        assert (
            _native_ratio128_attention_enabled(
                {
                    "model_type": "deepseek_v4",
                    "quantization": {"bits": 4},
                    "text_config": {"quantization_config": {"bits": 3.5}},
                }
            )
            is False
        )
        assert (
            _native_ratio128_attention_enabled(
                {"model_type": "deepseek_v4", "quantization": {"bits": 4}}
            )
            is True
        )

    @pytest.mark.parametrize(
        ("bits", "expected_enabled"),
        ((4, True), (2, False)),
        ids=("four-bit-native", "sub-four-bit-reference"),
    )
    def test_load_model_propagates_ratio128_attention_policy_to_model_args(
        self, tmp_path, applied_patch, bits, expected_enabled
    ):
        import mlx.nn as nn
        from mlx_lm.utils import load_model

        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]
        config = {
            "model_type": "deepseek_v4",
            "num_hidden_layers": 1,
            "compress_ratios": [128],
            "quantization": {
                "bits": bits,
                "group_size": 8,
                "mode": "affine",
            },
        }
        (tmp_path / "config.json").write_text(json.dumps(config))

        class CapturingModel(nn.Module):
            def __init__(self, args):
                super().__init__()
                self.args = args

        model, loaded_config = load_model(
            tmp_path,
            strict=False,
            lazy=True,
            get_model_classes=lambda config: (CapturingModel, dsv4.ModelArgs),
        )

        assert model.args.use_native_ratio128_attention is expected_enabled
        assert loaded_config["use_native_ratio128_attention"] is expected_enabled


class TestGeneratePatch:
    """mlx_lm.generate._make_cache replaced."""

    def test_make_cache_replaced(self, applied_patch):
        gen_mod = importlib.import_module("mlx_lm.generate")

        assert hasattr(gen_mod, "_make_cache")
        # Source must include PoolingCache → BatchPoolingCache branch.
        # We can't easily compare functions, so just verify the new
        # behavior: passing a model with a PoolingCache in make_cache
        # produces a BatchPoolingCache.
        from mlx_lm.models.cache import BatchPoolingCache, PoolingCache

        class FakeModel:
            def __init__(self):
                self.layers = [None]

            def make_cache(self):
                return [PoolingCache(ratio=4)]

        result = gen_mod._make_cache(FakeModel(), [0], None)
        assert len(result) == 1
        assert isinstance(result[0], BatchPoolingCache)


class TestTokenizerPatch:
    """mlx_lm.tokenizer_utils.AutoTokenizer wrapped with deepseek_v4 fallback."""

    def test_autotokenizer_wrapped(self, applied_patch):
        import mlx_lm.tokenizer_utils as tu

        # Wrapped class still exposes from_pretrained.
        assert hasattr(tu.AutoTokenizer, "from_pretrained")
        # Class name preserved for any introspection.
        assert tu.AutoTokenizer.__name__ == "AutoTokenizer"

    def test_passthrough_on_success(self, applied_patch):
        """When upstream AutoTokenizer.from_pretrained succeeds, the wrapper
        must return its result unmodified — no fallback path taken."""
        from unittest.mock import patch as mock_patch

        from omlx.patches.deepseek_v4 import tokenizer_patch

        sentinel = object()

        class _FakeUpstream:
            calls = []

            @staticmethod
            def from_pretrained(model_path, *args, **kwargs):
                _FakeUpstream.calls.append((model_path, args, kwargs))
                return sentinel

        with mock_patch("transformers.AutoTokenizer", _FakeUpstream):
            wrapper = tokenizer_patch._build_wrapper()
            result = wrapper.from_pretrained("/fake/path", trust_remote_code=True)

        assert result is sentinel
        assert len(_FakeUpstream.calls) == 1
        # Fallback never injected its own config kwarg.
        assert "config" not in _FakeUpstream.calls[0][2]

    def test_fallback_on_max_position_embeddings_error(self, applied_patch):
        """The exact AttributeError that transformers raises when it cannot
        recognize deepseek_v4 must trigger a retry with PreTrainedConfig()."""
        import pytest as _pytest
        from unittest.mock import patch as mock_patch

        from omlx.patches.deepseek_v4 import tokenizer_patch

        class _FakeUpstream:
            calls = []

            @staticmethod
            def from_pretrained(model_path, *args, **kwargs):
                _FakeUpstream.calls.append((model_path, args, kwargs))
                if "config" in kwargs:
                    return "FALLBACK_OK"
                raise AttributeError(
                    "'PreTrainedConfig' object has no attribute "
                    "'max_position_embeddings'"
                )

        with mock_patch("transformers.AutoTokenizer", _FakeUpstream):
            wrapper = tokenizer_patch._build_wrapper()
            with _pytest.warns(
                RuntimeWarning, match="Falling back to generic tokenizer config"
            ):
                result = wrapper.from_pretrained("/fake/path")

        assert result == "FALLBACK_OK"
        assert len(_FakeUpstream.calls) == 2
        # Second call must inject config=PreTrainedConfig().
        assert "config" in _FakeUpstream.calls[1][2]

    def test_fallback_on_deepseek_v4_value_error(self, applied_patch):
        """ValueError mentioning deepseek_v4 also triggers fallback."""
        import pytest as _pytest
        from unittest.mock import patch as mock_patch

        from omlx.patches.deepseek_v4 import tokenizer_patch

        class _FakeUpstream:
            calls = []

            @staticmethod
            def from_pretrained(model_path, *args, **kwargs):
                _FakeUpstream.calls.append((model_path, args, kwargs))
                if "config" in kwargs:
                    return "FALLBACK_OK"
                raise ValueError("Unrecognized configuration class for deepseek_v4")

        with mock_patch("transformers.AutoTokenizer", _FakeUpstream):
            wrapper = tokenizer_patch._build_wrapper()
            with _pytest.warns(
                RuntimeWarning, match="Falling back to generic tokenizer config"
            ):
                result = wrapper.from_pretrained("/fake/path")

        assert result == "FALLBACK_OK"
        assert len(_FakeUpstream.calls) == 2

    def test_unrelated_error_reraises(self, applied_patch):
        """Errors outside the deepseek_v4 / max_position_embeddings signature
        must NOT be swallowed."""
        from unittest.mock import patch as mock_patch

        import pytest as _pytest

        from omlx.patches.deepseek_v4 import tokenizer_patch

        class _FakeUpstream:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                raise ValueError("totally unrelated error")

        with mock_patch("transformers.AutoTokenizer", _FakeUpstream):
            wrapper = tokenizer_patch._build_wrapper()
            with _pytest.raises(ValueError, match="totally unrelated"):
                wrapper.from_pretrained("/fake/path")

    def test_explicit_config_skips_fallback(self, applied_patch):
        """If the caller already passed config=, we must not override it
        even when the inner call raises a matching error."""
        from unittest.mock import patch as mock_patch

        import pytest as _pytest

        from omlx.patches.deepseek_v4 import tokenizer_patch

        class _FakeUpstream:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                # Caller-provided config is in kwargs; we still raise the
                # max_position_embeddings error to verify the wrapper does
                # not silently retry.
                raise AttributeError(
                    "'PreTrainedConfig' object has no attribute "
                    "'max_position_embeddings'"
                )

        with mock_patch("transformers.AutoTokenizer", _FakeUpstream):
            wrapper = tokenizer_patch._build_wrapper()
            with _pytest.raises(AttributeError, match="max_position_embeddings"):
                wrapper.from_pretrained("/fake/path", config="caller_supplied")

    def test_class_attribute_forwarding(self, applied_patch):
        """Class-level attribute access (e.g. AutoTokenizer.register) must
        forward to the upstream class so mlx-lm's NewlineTokenizer
        registration still works."""
        import mlx_lm.tokenizer_utils as tu
        from transformers import AutoTokenizer as upstream_at

        # register is an upstream classmethod — wrapped class must expose it.
        assert tu.AutoTokenizer.register is upstream_at.register


class TestDSMLToolParser:
    """tool_parser_v4 — DSML invoke / parameter grammar parsing."""

    def test_single_invoke_typed_args(self, applied_patch):
        from omlx.patches.deepseek_v4 import tool_parser_v4 as tp

        text = (
            '<｜DSML｜invoke name="get_weather">\n'
            '<｜DSML｜parameter name="city" string="true">Seoul</｜DSML｜parameter>\n'
            '<｜DSML｜parameter name="days" string="false">7</｜DSML｜parameter>\n'
            '<｜DSML｜parameter name="imperial" string="false">false</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>"
        )
        result = tp.parse_tool_call(text)
        assert result["name"] == "get_weather"
        assert result["arguments"] == {
            "city": "Seoul",
            "days": 7,
            "imperial": False,
        }

    def test_multiple_invokes_returns_list(self, applied_patch):
        from omlx.patches.deepseek_v4 import tool_parser_v4 as tp

        text = (
            '<｜DSML｜invoke name="a">'
            '<｜DSML｜parameter name="x" string="false">1</｜DSML｜parameter>'
            "</｜DSML｜invoke>\n"
            '<｜DSML｜invoke name="b">'
            '<｜DSML｜parameter name="y" string="true">hello</｜DSML｜parameter>'
            "</｜DSML｜invoke>"
        )
        result = tp.parse_tool_call(text)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == {"name": "a", "arguments": {"x": 1}}
        assert result[1] == {"name": "b", "arguments": {"y": "hello"}}

    def test_object_and_array_parameters(self, applied_patch):
        from omlx.patches.deepseek_v4 import tool_parser_v4 as tp

        text = (
            '<｜DSML｜invoke name="search">\n'
            '<｜DSML｜parameter name="filters" string="false">'
            '{"category": "books", "min_price": 10}'
            "</｜DSML｜parameter>\n"
            '<｜DSML｜parameter name="ids" string="false">[1, 2, 3]</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>"
        )
        result = tp.parse_tool_call(text)
        assert result["arguments"]["filters"] == {"category": "books", "min_price": 10}
        assert result["arguments"]["ids"] == [1, 2, 3]

    def test_no_invoke_raises(self, applied_patch):
        import pytest as _pytest

        from omlx.patches.deepseek_v4 import tool_parser_v4 as tp

        with _pytest.raises(ValueError, match="No.*invoke.*block"):
            tp.parse_tool_call("just some plain text without DSML markup")

    def test_outer_markers_exposed(self, applied_patch):
        from omlx.patches.deepseek_v4 import tool_parser_v4 as tp

        # mlx-lm reads these as module attributes for stream detection.
        assert tp.tool_call_start == "<｜DSML｜tool_calls>"
        assert tp.tool_call_end == "</｜DSML｜tool_calls>"


class TestChatTemplateV4:
    """Official DeepSeek V4 0731 encoding plus the mlx-lm adapter."""

    def test_outer_marker_uses_tool_calls_not_function_calls(self, applied_patch):
        from omlx.patches.deepseek_v4 import chat_template_v4 as ct

        assert "function_calls" not in ct.tool_calls_template
        assert "tool_calls" in ct.tool_calls_template
        assert "function_calls" not in ct.TOOLS_TEMPLATE

    def test_inner_grammar_unchanged_from_v32(self, applied_patch):
        from omlx.patches.deepseek_v4 import chat_template_v4 as ct

        # Inner markers must still be invoke / parameter — V4 reuses V3.2's
        # invoke/parameter grammar.
        assert "invoke" in ct.tool_call_template
        assert "parameter" in ct.encode_arguments_to_dsml(
            {"name": "x", "arguments": '{"k": "v"}'}
        )

    def test_round_trip_encode_then_parse(self, applied_patch):
        from omlx.patches.deepseek_v4 import chat_template_v4 as ct
        from omlx.patches.deepseek_v4 import tool_parser_v4 as tp

        encoded_args = ct.encode_arguments_to_dsml(
            {"name": "f", "arguments": '{"a": 1, "b": "hi", "c": [1, 2]}'}
        )
        invoke = ct.tool_call_template.format(
            dsml_token=ct.dsml_token, name="f", arguments=encoded_args
        )
        block = ct.tool_calls_template.format(
            dsml_token=ct.dsml_token,
            tool_calls=invoke,
            tc_block_name=ct.tool_calls_block_name,
        )
        # Strip the outer markers as TokenizerWrapper would.
        inner = (
            block.replace(tp.tool_call_start, "").replace(tp.tool_call_end, "").strip()
        )
        parsed = tp.parse_tool_call(inner)
        assert parsed == {"name": "f", "arguments": {"a": 1, "b": "hi", "c": [1, 2]}}

    def test_user_only_request_with_tools_injects_dsml(self, applied_patch):
        """User-only message + tools must still emit the DSML tools block.

        Regression guard for the case where a Claude Code or OpenAI client
        passes ``tools`` without a system message. ``render_message`` only
        injects tools on system / developer roles, so ``encode_messages``
        synthesises an empty system message up front when the first
        message is a plain user. Without this fix the rendered prompt
        omits the ``<functions>`` schema entirely and the model never
        emits a tool_calls block.
        """
        from omlx.patches.deepseek_v4 import chat_template_v4 as ct

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ]
        prompt = ct.apply_chat_template(
            [{"role": "user", "content": "Weather in Seoul?"}],
            tools=tools,
            add_generation_prompt=True,
        )
        assert "### Available Tool Schemas" in prompt
        assert "get_weather" in prompt
        assert ct.dsml_token in prompt

    def test_system_user_request_with_tools_unchanged(self, applied_patch):
        """When a system message is already present, the synthetic prepend
        path must not fire — the rendered prompt keeps the original system
        content verbatim and only injects the tools schema once.
        """
        from omlx.patches.deepseek_v4 import chat_template_v4 as ct

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ]
        prompt = ct.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Weather in Seoul?"},
            ],
            tools=tools,
            add_generation_prompt=True,
        )
        assert "You are a helpful assistant." in prompt
        # Only one tools block — no double-injection from synthetic prepend.
        assert prompt.count("### Available Tool Schemas") == 1

    def test_user_only_no_tools_no_prepend(self, applied_patch):
        """No tools → no synthetic system. Plain user-only request renders
        with just the BOS + user wrapper, matching V3.2 baseline."""
        from omlx.patches.deepseek_v4 import chat_template_v4 as ct

        prompt = ct.apply_chat_template(
            [{"role": "user", "content": "Hi"}],
            add_generation_prompt=True,
        )
        assert "### Available Tool Schemas" not in prompt
        assert "## Tools" not in prompt

    def test_official_basic_thinking_prompt(self, applied_patch):
        from omlx.patches.deepseek_v4 import chat_template_v4 as ct

        prompt = ct.apply_chat_template(
            [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hello"},
            ],
            add_generation_prompt=True,
        )

        assert prompt == (
            "<｜begin▁of▁sentence｜>Be helpful." "<｜User｜>Hello<｜Assistant｜><think>"
        )

    def test_official_latest_reminder_before_user(self, applied_patch):
        from omlx.patches.deepseek_v4 import chat_template_v4 as ct

        prompt = ct.apply_chat_template(
            [
                {"role": "system", "content": "Be helpful."},
                {"role": "latest_reminder", "content": "2026-08-04,Seoul"},
                {"role": "user", "content": "Hello"},
            ],
            add_generation_prompt=True,
        )

        assert prompt == (
            "<｜begin▁of▁sentence｜>Be helpful."
            "<｜latest_reminder｜>2026-08-04,Seoul"
            "<｜User｜>Hello<｜Assistant｜><think>"
        )

    def test_official_tool_result_is_merged_into_user_turn(self, applied_patch):
        from omlx.patches.deepseek_v4 import chat_template_v4 as ct

        prompt = ct.apply_chat_template(
            [
                {"role": "user", "content": "Look it up"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": {"query": "oMLX"},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "result",
                },
            ],
            add_generation_prompt=True,
        )

        assert "<｜User｜><tool_result>result</tool_result>" in prompt
        assert prompt.endswith("<｜Assistant｜><think>")

    def test_declares_generic_mid_system_unsupported(self, applied_patch):
        from omlx.patches.deepseek_v4 import chat_template_v4 as ct

        assert ct.supports_mid_system_messages is False
        assert ct.apply_chat_template.supports_mid_system_messages is False

    def test_relocates_claude_tail_system_before_its_user(self, applied_patch):
        from omlx.patches.deepseek_v4 import chat_template_v4 as ct

        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "Plan mode"},
        ]

        relocated = ct.relocate_mid_system_messages(messages)

        assert relocated == [
            {"role": "system", "content": "Be helpful."},
            {"role": "latest_reminder", "content": "Plan mode"},
            {"role": "user", "content": "Hello"},
        ]
        assert messages[1]["role"] == "user"
        prompt = ct.apply_chat_template(relocated, add_generation_prompt=True)
        assert prompt == (
            "<｜begin▁of▁sentence｜>Be helpful."
            "<｜latest_reminder｜>Plan mode"
            "<｜User｜>Hello<｜Assistant｜><think>"
        )

    def test_relocation_merges_system_run_before_same_user(self, applied_patch):
        from omlx.patches.deepseek_v4 import chat_template_v4 as ct

        relocated = ct.relocate_mid_system_messages(
            [
                {"role": "user", "content": "Hello"},
                {"role": "system", "content": "Plan mode"},
                {"role": "system", "content": "Hook context"},
                {"role": "assistant", "content": "OK"},
            ]
        )

        assert relocated == [
            {
                "role": "latest_reminder",
                "content": "Plan mode\n\nHook context",
            },
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "OK"},
        ]

    def test_relocation_refuses_ambiguous_system_placement(self, applied_patch):
        from omlx.patches.deepseek_v4 import chat_template_v4 as ct

        assert (
            ct.relocate_mid_system_messages(
                [
                    {"role": "user", "content": "First"},
                    {"role": "system", "content": "Ambiguous"},
                    {"role": "user", "content": "Second"},
                ]
            )
            is None
        )

    def test_relocates_tool_adjacent_system_in_place(self, applied_patch):
        # Claude Code's periodic reminders arrive after the Anthropic
        # adapter split tool_result blocks, i.e. tool -> system -> assistant.
        from omlx.patches.deepseek_v4 import chat_template_v4 as ct

        tool_call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": {"query": "x"}},
        }
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Look it up"},
            {"role": "assistant", "content": "", "tool_calls": [tool_call]},
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            {"role": "system", "content": "Task reminder"},
            {"role": "assistant", "content": "Done"},
        ]

        relocated = ct.relocate_mid_system_messages(messages)

        assert relocated is not None
        assert relocated[3] == {
            "role": "latest_reminder",
            "content": "Task reminder",
        }
        assert relocated[4]["role"] == "tool"
        assert relocated[5]["role"] == "assistant"

        prompt = ct.apply_chat_template(relocated, add_generation_prompt=True)
        assert (
            "<｜latest_reminder｜>Task reminder"
            "<｜User｜><tool_result>result</tool_result>" in prompt
        )

    def test_relocates_tool_adjacent_system_at_tail(self, applied_patch):
        from omlx.patches.deepseek_v4 import chat_template_v4 as ct

        relocated = ct.relocate_mid_system_messages(
            [
                {"role": "user", "content": "Look it up"},
                {"role": "assistant", "content": "", "tool_calls": []},
                {"role": "tool", "tool_call_id": "c1", "content": "one"},
                {"role": "tool", "tool_call_id": "c2", "content": "two"},
                {"role": "system", "content": "Task reminder"},
            ]
        )

        assert relocated is not None
        assert relocated[2] == {
            "role": "latest_reminder",
            "content": "Task reminder",
        }
        assert [m["role"] for m in relocated[3:]] == ["tool", "tool"]

    def test_encode_arguments_accepts_dict(self, applied_patch):
        """Anthropic /v1/messages history stores tool_call arguments as
        a dict (anthropic_utils.py decodes the input before saving).
        encode_arguments_to_dsml must accept that shape — not just the
        OpenAI JSON-string convention — so multi-turn renders don't
        raise TypeError when the assistant history is from Claude Code.
        """
        from omlx.patches.deepseek_v4 import chat_template_v4 as ct

        encoded = ct.encode_arguments_to_dsml(
            {"name": "f", "arguments": {"location": "Seoul", "n": 3}}
        )
        assert 'name="location"' in encoded and "Seoul" in encoded
        assert 'name="n"' in encoded and ">3<" in encoded
        # string="true" for string params, "false" for non-string.
        assert 'string="true"' in encoded
        assert 'string="false"' in encoded

    def test_assistant_tool_call_dict_arguments_round_trip(self, applied_patch):
        """End-to-end multi-turn: assistant message history contains a
        tool_use whose arguments came in as dict (Anthropic shape). The
        rendered prompt must include the assistant's prior tool_call
        block in DSML form so the model can continue the conversation
        coherently.
        """
        from omlx.patches.deepseek_v4 import chat_template_v4 as ct

        messages = [
            {"role": "user", "content": "Weather in Seoul?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"location": "Seoul"},
                        },
                    }
                ],
            },
            {"role": "tool", "content": "sunny, 22C"},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ]
        prompt = ct.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True
        )
        assert "<｜DSML｜tool_calls>" in prompt
        assert 'invoke name="get_weather"' in prompt
        assert "Seoul" in prompt
        assert "sunny, 22C" in prompt

    def test_streamed_tool_call_turn_rerender_is_byte_stable(self, applied_patch):
        """Parser-to-rerender round trip: content accumulated from the
        streaming filter's deltas, re-rendered through the template, must
        reproduce the model's raw emission byte-for-byte.

        The reference decoder consumes "\\n\\n<｜DSML｜tool_calls" as one
        literal stop token, so the separator before a tool-call block
        belongs to the envelope, not to content. The streaming filter now
        mirrors that: a client that accumulates content deltas stores the
        turn without the separator, and the template's own canonical
        "\\n\\n" restores it on re-render -- prompts stay append-only
        across tool hops.
        """
        from omlx.api.tool_calling import ToolCallStreamFilter
        from omlx.patches.deepseek_v4 import chat_template_v4 as ct
        from omlx.patches.deepseek_v4 import tool_parser_v4 as tp

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Run a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ]
        turn_0_user = {"role": "user", "content": "List the files in /tmp for me."}

        # The exact prompt the model was fed to GENERATE the tool-call turn.
        prompt_for_turn = ct.apply_chat_template(
            [turn_0_user],
            tools=tools,
            thinking_mode="chat",
            add_generation_prompt=True,
        )

        # The model's raw, unparsed emission: prose, the trained "\n\n"
        # separator, then the DSML block built from the template's own
        # grammar pieces.
        tc_args = {"command": "ls -la /tmp"}
        raw_dsml_block = ct.tool_calls_template.format(
            dsml_token=ct.dsml_token,
            tool_calls=ct.tool_call_template.format(
                dsml_token=ct.dsml_token,
                name="bash",
                arguments=ct.encode_arguments_to_dsml(
                    {"name": "bash", "arguments": tc_args}
                ),
            ),
            tc_block_name=ct.tool_calls_block_name,
        )
        raw_emission = "I'll list the files in /tmp for you.\n\n" + raw_dsml_block

        # Accumulate content exactly as a streaming client does: feed the
        # raw emission character-by-character so every boundary -- inside
        # the separator, inside the DSML markers -- is split across feeds.
        stream_filter = ToolCallStreamFilter(
            SimpleNamespace(
                tool_call_start=tp.tool_call_start, tool_call_end=tp.tool_call_end
            )
        )
        accumulated_content = ""
        for ch in raw_emission:
            accumulated_content += stream_filter.feed(ch)
        accumulated_content += stream_filter.finish()
        assert accumulated_content == "I'll list the files in /tmp for you."

        turn_0_assistant = {
            "role": "assistant",
            "content": accumulated_content,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": tc_args},
                }
            ],
        }

        # The decode loop stops before the eos token, so a completed
        # turn's full text adds it back once.
        prev_full_text = prompt_for_turn + raw_emission + ct.eos_token

        # Byte-stability: re-rendering the completed turn from its
        # parser-derived structured form reproduces the raw emission.
        render_through_turn = ct.apply_chat_template(
            [turn_0_user, turn_0_assistant], tools=tools, thinking_mode="chat"
        )
        assert render_through_turn == prev_full_text
        assert "\n\n\n" not in render_through_turn

        # Append-only: the NEXT turn (tool result + new user message)
        # extends that exact text rather than re-deriving a different
        # rendering of the historical tool-call turn.
        turn_1_tool_result = {
            "role": "tool",
            "content": "foo.log (123 bytes)\nbar.txt (4096 bytes)",
        }
        turn_1_user = {"role": "user", "content": "Which file is bigger?"}
        render_through_next_turn = ct.apply_chat_template(
            [turn_0_user, turn_0_assistant, turn_1_tool_result, turn_1_user],
            tools=tools,
            thinking_mode="chat",
            add_generation_prompt=True,
        )
        assert render_through_next_turn.startswith(prev_full_text)


class TestChatTemplateModuleRegistration:
    """sys.modules registration so mlx-lm's importlib path picks up our types."""

    def test_chat_template_module_registered(self, applied_patch):
        import sys

        assert "mlx_lm.chat_templates.deepseek_v4" in sys.modules
        mod = sys.modules["mlx_lm.chat_templates.deepseek_v4"]
        assert hasattr(mod, "apply_chat_template")

    def test_tool_parser_module_registered(self, applied_patch):
        import sys

        assert "mlx_lm.tool_parsers.deepseek_v4" in sys.modules
        mod = sys.modules["mlx_lm.tool_parsers.deepseek_v4"]
        assert hasattr(mod, "parse_tool_call")
        assert mod.tool_call_start == "<｜DSML｜tool_calls>"
        assert mod.tool_call_end == "</｜DSML｜tool_calls>"


class TestModelClassResolution:
    """mlx_lm.utils._get_classes resolves deepseek_v4 to our injected classes."""

    def test_get_classes_returns_injected_module(self, applied_patch):
        from mlx_lm.utils import _get_classes

        model_class, args_class = _get_classes({"model_type": "deepseek_v4"})
        assert model_class.__module__ == "mlx_lm.models.deepseek_v4"
        assert args_class.__module__ == "mlx_lm.models.deepseek_v4"
        assert model_class.__name__ == "Model"
        assert args_class.__name__ == "ModelArgs"

    def test_get_classes_returns_injected_module_for_mtp_variant(self, applied_patch):
        from mlx_lm.utils import _get_classes

        model_class, args_class = _get_classes({"model_type": "deepseek_v4_mtp"})
        assert model_class.__module__ == "mlx_lm.models.deepseek_v4"
        assert args_class.__module__ == "mlx_lm.models.deepseek_v4"


class TestPatchedLoadModelTrustRemoteCode:
    """DeepSeek's patched load_model must mirror mlx-lm's custom-code gate."""

    def test_signature_accepts_trust_remote_code(self, applied_patch):
        from mlx_lm.utils import load_model

        assert "trust_remote_code" in inspect.signature(load_model).parameters

    def test_model_file_requires_trust_remote_code(self, tmp_path, applied_patch):
        config_path = tmp_path / "config.json"
        config_path.write_text(
            '{"model_type": "custom", "model_file": "custom_arch.py"}'
        )
        (tmp_path / "custom_arch.py").write_text(
            "\n".join(
                [
                    "from pathlib import Path",
                    "import mlx.nn as nn",
                    "Path(__file__).with_name('executed.txt').write_text('yes')",
                    "",
                    "class ModelArgs:",
                    "    @classmethod",
                    "    def from_dict(cls, config):",
                    "        return cls()",
                    "",
                    "class Model(nn.Module):",
                    "    def __init__(self, args):",
                    "        super().__init__()",
                ]
            )
        )

        from mlx_lm.utils import load_model

        with pytest.raises(ValueError, match="trust_remote_code=True"):
            load_model(tmp_path, strict=False, lazy=True)

        assert not (tmp_path / "executed.txt").exists()

        load_model(
            tmp_path,
            strict=False,
            lazy=True,
            trust_remote_code=True,
        )
        assert (tmp_path / "executed.txt").read_text() == "yes"

    def test_untrusted_model_file_is_rejected_before_safetensors_open(
        self, tmp_path, applied_patch, monkeypatch
    ):
        (tmp_path / "config.json").write_text(
            '{"model_type": "custom", "model_file": "custom_arch.py"}'
        )
        (tmp_path / "model.safetensors").write_bytes(b"not opened")

        from mlx_lm import utils

        from omlx.patches.deepseek_v4 import utils_patch

        load_weights = MagicMock(side_effect=AssertionError("weights were opened"))
        monkeypatch.setattr(utils_patch, "_load_safetensors", load_weights)

        with pytest.raises(ValueError, match="trust_remote_code=True"):
            utils.load_model(tmp_path, lazy=True)

        load_weights.assert_not_called()


class TestCacheHandlerRegistration:
    """omlx CacheTypeRegistry resolves the new cache types to their handlers."""

    def test_pooling_cache_resolves_to_handler(self, applied_patch):
        from omlx.cache.type_registry import CacheTypeRegistry

        handler = CacheTypeRegistry.get_handler_by_class_name("PoolingCache")
        assert type(handler).__name__ == "PoolingCacheHandler"

    def test_batch_pooling_cache_resolves_to_handler(self, applied_patch):
        from omlx.cache.type_registry import CacheTypeRegistry

        handler = CacheTypeRegistry.get_handler_by_class_name("BatchPoolingCache")
        assert type(handler).__name__ == "BatchPoolingCacheHandler"

    def test_pooling_cache_not_block_sliceable(self, applied_patch):
        from omlx.cache.type_registry import CacheTypeRegistry

        handler = CacheTypeRegistry.get_handler_by_class_name("PoolingCache")
        assert handler.supports_block_slicing is False

    def test_batch_pooling_cache_not_block_sliceable(self, applied_patch):
        from omlx.cache.type_registry import CacheTypeRegistry

        handler = CacheTypeRegistry.get_handler_by_class_name("BatchPoolingCache")
        assert handler.supports_block_slicing is False

    def test_detect_cache_type_pooling(self, applied_patch):
        from mlx_lm.models.cache import PoolingCache

        from omlx.cache.type_handlers import CacheType
        from omlx.cache.type_registry import CacheTypeRegistry

        cache = PoolingCache(ratio=4)
        assert CacheTypeRegistry.detect_cache_type(cache) == CacheType.POOLING_CACHE


class TestPoolingCacheStateRoundTrip:
    """Handler extract_state → reconstruct_cache must preserve the pool tensor."""

    def test_round_trip_with_pooled_tensor(self, applied_patch):
        import mlx.core as mx
        from mlx_lm.models.cache import PoolingCache

        from omlx.cache.type_registry import CacheTypeRegistry

        # Build a PoolingCache with a known pool.
        ratio = 4
        cache = PoolingCache(ratio=ratio)
        # Simulate update_and_fetch having stuffed the pool with 8
        # compressed tokens of dim 32.
        pooled = mx.arange(1 * 8 * 32, dtype=mx.float32).reshape(1, 8, 32)
        cache.pooled = pooled

        handler = CacheTypeRegistry.get_handler_by_class_name("PoolingCache")
        state = handler.extract_state(cache)
        assert state["pooled"] is not None
        assert state["pooled"].shape == (1, 8, 32)

        restored = handler.reconstruct_cache(state, meta_state=ratio)
        assert restored is not None
        assert restored.ratio == ratio
        assert restored.pooled.shape == (1, 8, 32)
        # Verify content matches.
        diff = mx.max(mx.abs(restored.pooled - pooled)).item()
        assert diff == 0.0

    def test_round_trip_empty_cache(self, applied_patch):
        from mlx_lm.models.cache import PoolingCache

        from omlx.cache.type_registry import CacheTypeRegistry

        cache = PoolingCache(ratio=8)
        handler = CacheTypeRegistry.get_handler_by_class_name("PoolingCache")
        state = handler.extract_state(cache)
        assert state["pooled"] is None
        assert state["buf_kv"] is None

        restored = handler.reconstruct_cache(state, meta_state=8)
        assert restored is not None
        assert restored.empty()
        assert restored.ratio == 8

    def test_seq_len_from_state(self, applied_patch):
        import mlx.core as mx
        from mlx_lm.models.cache import PoolingCache

        from omlx.cache.type_registry import CacheTypeRegistry

        cache = PoolingCache(ratio=4)
        cache.pooled = mx.zeros((1, 12, 16), dtype=mx.float32)
        handler = CacheTypeRegistry.get_handler_by_class_name("PoolingCache")
        state = handler.extract_state(cache)
        assert handler.get_seq_len(state) == 12


class TestCacheMaterialization:
    """DeepSeek-V4 cache arrays are materialized after forward updates."""

    def test_helper_collects_plain_and_cachelist_leaf_arrays(
        self, applied_patch, monkeypatch
    ):
        import mlx.core as mx
        from mlx_lm.models.cache import CacheList

        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]

        class Leaf:
            def __init__(self, arr):
                self.arr = arr
                self.none_value = None
                self.scalar = 7

        leaf_a = Leaf(mx.array([1], dtype=mx.int32))
        leaf_b = Leaf(mx.array([2], dtype=mx.int32))
        leaf_c = Leaf(mx.array([3], dtype=mx.int32))
        calls = []

        def fake_eval(*arrays):
            calls.append(arrays)

        monkeypatch.setattr(dsv4.mx, "eval", fake_eval)

        dsv4._materialize_cache_arrays([CacheList(leaf_a, leaf_b), leaf_c, None])

        assert len(calls) == 1
        assert calls[0] == (leaf_a.arr, leaf_b.arr, leaf_c.arr)

    def test_helper_can_defer_exact_arrays_and_restores_after_exception(
        self, applied_patch, monkeypatch
    ):
        import mlx.core as mx

        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]

        class Leaf:
            def __init__(self, arr):
                self.arr = arr

        leaf = Leaf(mx.array([7], dtype=mx.int32))
        calls = []
        monkeypatch.setattr(dsv4.mx, "eval", lambda *arrays: calls.append(arrays))

        captured = None
        with pytest.raises(RuntimeError, match="synthetic failure"):
            with dsv4._defer_cache_materialization() as captured:
                dsv4._materialize_cache_arrays([leaf])
                raise RuntimeError("synthetic failure")

        assert captured == [leaf.arr]
        assert calls == []

        # The thread-local capture is strictly scoped: a later decode-style
        # call immediately restores the stock materialization barrier.
        dsv4._materialize_cache_arrays([leaf])
        assert calls == [(leaf.arr,)]

    def test_model_call_materializes_cache_after_layer_loop(self, applied_patch):
        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]
        source = inspect.getsource(dsv4.DeepseekV4Model.__call__)

        loop_pos = max(
            source.find("for layer, layer_cache in zip"),
            source.find("for layer_idx, (layer, layer_cache) in enumerate"),
        )
        assert loop_pos >= 0
        materialize_pos = source.index("_materialize_cache_arrays(cache)")
        pipeline_send_pos = source.index("if pipeline_rank != 0")

        assert loop_pos < materialize_pos < pipeline_send_pos


class TestDeepseekV4SwitchGLU:
    """DeepSeek-V4 SwitchGLU execution guards."""

    def test_short_affine_route_restores_bfloat16_output(
        self, applied_patch, monkeypatch
    ):
        mx = pytest.importorskip("mlx.core")
        from omlx.patches.deepseek_v4 import switch_layers

        mx.random.seed(17)
        layer = switch_layers.SwitchGLU(
            input_dims=64,
            hidden_dims=64,
            num_experts=4,
            bias=False,
        )
        for name in ("up_proj", "gate_proj", "down_proj"):
            projection = getattr(layer, name).to_quantized(
                group_size=64,
                bits=2,
                mode="affine",
            )
            projection.scales = projection.scales.astype(mx.float16)
            projection.biases = projection.biases.astype(mx.float16)
            setattr(layer, name, projection)

        projection_input_dtypes = []
        original_call = switch_layers.QuantizedSwitchLinear.__call__

        def record_projection_input(projection, value, *args, **kwargs):
            projection_input_dtypes.append(value.dtype)
            return original_call(projection, value, *args, **kwargs)

        monkeypatch.setattr(
            switch_layers.QuantizedSwitchLinear,
            "__call__",
            record_projection_input,
        )

        x = mx.random.normal((1, 7, 64), dtype=mx.bfloat16)
        indices = mx.array(
            [[[0, 1, 2, 3, 0, 1]] * 7],
            dtype=mx.int32,
        )

        assert indices.size < 64
        y = layer(x, indices)
        mx.eval(y)

        assert y.shape == (1, 7, 6, 64)
        assert y.dtype == mx.bfloat16
        assert projection_input_dtypes == [mx.float16, mx.float16, mx.float16]

    def test_shared_expert_uses_configured_swiglu_limit(self, applied_patch):
        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]

        config = dsv4.ModelArgs(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            moe_intermediate_size=4,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            n_shared_experts=1,
            n_routed_experts=2,
            num_experts_per_tok=1,
            num_hash_layers=0,
            q_lora_rank=0,
            qk_rope_head_dim=4,
            head_dim=4,
            o_lora_rank=0,
            index_n_heads=2,
            index_head_dim=4,
            index_topk=2,
            swiglu_limit=10.0,
        )

        moe = dsv4.DeepseekV4MoE(config, layer_idx=0)

        assert moe.switch_mlp.activation.limit == config.swiglu_limit
        assert moe.shared_experts.swiglu_limit == config.swiglu_limit

    def test_skips_fused_weighted_sum_for_cache_stability(
        self, applied_patch, monkeypatch
    ):
        mx = pytest.importorskip("mlx.core")
        from omlx.patches.deepseek_v4 import switch_layers

        monkeypatch.setattr(
            switch_layers.glm_fast,
            "has_symbol",
            lambda name: name == "glm_moe_weighted_sum",
        )

        def fail_weighted_sum(*args, **kwargs):
            raise AssertionError("DeepSeek V4 must not use fused weighted sum")

        monkeypatch.setattr(
            switch_layers.glm_fast,
            "glm_moe_weighted_sum",
            fail_weighted_sum,
            raising=False,
        )

        mx.random.seed(11)
        layer = switch_layers.SwitchGLU(
            input_dims=16,
            hidden_dims=32,
            num_experts=4,
            bias=False,
        )
        x = mx.random.normal((1, 8, 16), dtype=mx.float32)
        indices = mx.array(
            [
                [
                    [0, 1, 2, 3, 0, 1, 2, 3],
                    [1, 2, 3, 0, 1, 2, 3, 0],
                    [2, 3, 0, 1, 2, 3, 0, 1],
                    [3, 0, 1, 2, 3, 0, 1, 2],
                    [0, 2, 1, 3, 0, 2, 1, 3],
                    [1, 3, 2, 0, 1, 3, 2, 0],
                    [2, 0, 3, 1, 2, 0, 3, 1],
                    [3, 1, 0, 2, 3, 1, 0, 2],
                ]
            ],
            dtype=mx.int32,
        )
        scores = mx.softmax(
            mx.random.normal((1, 8, 8), dtype=mx.float32),
            axis=-1,
        )

        y = layer(x, indices, scores=scores)
        mx.eval(y)

        assert y.shape == (1, 8, 8, 16)


def test_pooled_mask_trims_batch_cache_physical_tail(applied_patch):
    mx = pytest.importorskip("mlx.core")
    dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]

    local = mx.ones((1, 1, 8, 128), dtype=mx.bool_)
    # Restored BatchPoolingCache capacity can exceed this row's logical pooled
    # KV width. The invalid physical tail must not change the logical mask.
    pooled = mx.concatenate(
        [
            mx.ones((1, 8, 25), dtype=mx.bool_),
            mx.zeros((1, 8, 5), dtype=mx.bool_),
        ],
        axis=-1,
    )

    extended = dsv4._extend_mask(
        local,
        pooled,
        128 + 25,
        local_width=128,
        pooled_width=25,
    )
    mx.eval(extended)

    assert extended.shape == (1, 1, 8, 153)
    assert bool(mx.all(extended).item()) is True


def test_pooled_mask_rejects_missing_validity_columns(applied_patch):
    mx = pytest.importorskip("mlx.core")
    dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]

    with pytest.raises(ValueError, match="shorter than pooled KV"):
        dsv4._extend_mask(
            mx.ones((1, 1, 2, 4), dtype=mx.bool_),
            mx.ones((1, 2, 2), dtype=mx.bool_),
            7,
            local_width=4,
            pooled_width=3,
        )


def test_pooled_mask_keeps_rotating_local_suffix(applied_patch):
    mx = pytest.importorskip("mlx.core")
    dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]

    local = mx.array([[[[False, False, True, False, True, True]]]])
    pooled = mx.ones((1, 1, 2), dtype=mx.bool_)
    extended = dsv4._extend_mask(
        local,
        pooled,
        6,
        local_width=4,
        pooled_width=2,
    )
    mx.eval(extended)

    assert extended.tolist() == [[[[True, False, True, True, True, True]]]]


class TestDeepseekV4CompressedNativeAttention:
    @staticmethod
    def _attention_config(dsv4):
        return dsv4.ModelArgs(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            n_shared_experts=1,
            n_routed_experts=2,
            num_experts_per_tok=1,
            num_hash_layers=0,
            q_lora_rank=16,
            qk_rope_head_dim=4,
            head_dim=8,
            o_groups=1,
            o_lora_rank=8,
            index_n_heads=2,
            index_head_dim=4,
            index_topk=8,
            sliding_window=128,
            compress_ratios=[128],
        )

    def test_ratio128_dispatch_and_reference_fallbacks(
        self, applied_patch, monkeypatch
    ):
        import mlx.core as mx

        from omlx.custom_kernels.glm_moe_dsa import fast

        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]
        layer = dsv4.CompressedAttention(self._attention_config(dsv4), 0)
        sparse_calls = []
        dense_masks = []

        def sparse_spy(q, local_kv, pooled, pooled_indices, *args, **kwargs):
            sparse_calls.append((local_kv.shape, pooled.shape, pooled_indices))
            return mx.zeros(q.shape, dtype=q.dtype)

        def dense_spy(q, key, value, *args, **kwargs):
            dense_masks.append(kwargs["mask"])
            return mx.zeros(q.shape, dtype=q.dtype)

        monkeypatch.setattr(dsv4, "_sparse_pooled_attention", sparse_spy)
        monkeypatch.setattr(dsv4, "scaled_dot_product_attention", dense_spy)
        monkeypatch.setattr(fast, "has_symbol", lambda name: True)

        x = mx.random.normal((1, 129, 16), dtype=mx.bfloat16)
        y = layer(x, _standard_mask=True)
        mx.eval(y, sparse_calls[0][2])

        assert y.shape == (1, 129, 16)
        assert sparse_calls[0][0] == (1, 1, 129, 8)
        assert sparse_calls[0][1] == (1, 1, 8)
        assert sparse_calls[0][2].tolist() == [[[0]] * 129]
        assert dense_masks == []

        monkeypatch.setattr(fast, "has_symbol", lambda name: False)
        layer(x, _standard_mask=True)
        assert len(sparse_calls) == 1
        assert len(dense_masks) == 1

        monkeypatch.setattr(fast, "has_symbol", lambda name: True)
        monkeypatch.setattr(dsv4, "_sparse_pooled_attention", lambda *a, **k: None)
        layer(x, _standard_mask=True)
        assert len(dense_masks) == 2
        monkeypatch.setattr(dsv4, "_sparse_pooled_attention", sparse_spy)

        monkeypatch.setattr(
            dsv4.Compressor,
            "__call__",
            lambda self, x, pool_cache, offset: mx.zeros(
                (x.shape[0], 1, self.head_dim), dtype=x.dtype
            ),
        )
        layer(x[:, :1], _standard_mask=True)
        assert len(sparse_calls) == 1
        assert len(dense_masks) == 3

        custom_mask = mx.tri(129, 129, dtype=mx.bool_)[None, None]
        layer(x, mask=custom_mask)
        assert len(sparse_calls) == 1
        assert dense_masks[-1].shape == (1, 1, 129, 130)

        low_bit_config = self._attention_config(dsv4)
        low_bit_config.use_native_ratio128_attention = False
        low_bit_layer = dsv4.CompressedAttention(low_bit_config, 0)
        low_bit_layer(x, _standard_mask=True)
        assert len(sparse_calls) == 1
        assert len(dense_masks) == 5

    def test_native_only_sparse_attention_rejects_unsupported_shape_without_gather(
        self, applied_patch
    ):
        import mlx.core as mx

        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]
        result = dsv4._sparse_pooled_attention(
            mx.zeros((1, 2, 5, 8), dtype=mx.bfloat16),
            mx.zeros((1, 1, 5, 8), dtype=mx.bfloat16),
            mx.zeros((1, 1, 8), dtype=mx.bfloat16),
            mx.zeros((1, 5, 1), dtype=mx.uint32),
            None,
            None,
            8**-0.5,
            mx.zeros((2,), dtype=mx.bfloat16),
            q_offset=0,
            compress_ratio=128,
            local_window=128,
            native_only=True,
        )

        assert result is None

    @pytest.mark.parametrize("heads", (64, 40, 32, 24))
    def test_topk_wsdpa_dispatch_ignores_native_sparse_disable(
        self, applied_patch, monkeypatch, heads
    ):
        import mlx.core as mx

        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]
        expected = mx.zeros((1, heads, 5, 512), dtype=mx.bfloat16)
        calls = []

        def wsdpa_spy(*args):
            calls.append(args)
            return expected

        monkeypatch.setattr(dsv4, "wsdpa_topk_prefill", wsdpa_spy)
        monkeypatch.setattr(
            dsv4,
            "_DEEPSEEK_V4_SPARSE_ATTENTION_NATIVE_DISABLED",
            True,
        )
        actual = dsv4._sparse_pooled_attention(
            mx.zeros((1, heads, 5, 512), dtype=mx.bfloat16),
            mx.zeros((1, 1, 133, 512), dtype=mx.bfloat16),
            mx.zeros((1, 513, 512), dtype=mx.bfloat16),
            mx.zeros((1, 5, 512), dtype=mx.uint32),
            None,
            None,
            512**-0.5,
            mx.zeros((heads,), dtype=mx.bfloat16),
            q_offset=128,
            compress_ratio=4,
            local_window=128,
            _standard_mask=True,
        )

        assert actual is expected
        assert len(calls) == 1

    @pytest.mark.parametrize(
        ("dtype_name", "max_tolerance"),
        (("float16", 0.004), ("bfloat16", 0.032)),
    )
    @pytest.mark.parametrize(
        ("offset", "length"),
        ((255, 17), (32_895, 17)),
        ids=("two-pooled-rows", "257-pooled-rows"),
    )
    def test_ratio128_native_attention_matches_causal_reference_across_pool_tiles(
        self, applied_patch, dtype_name, max_tolerance, offset, length
    ):
        import mlx.core as mx

        from omlx.custom_kernels.glm_moe_dsa import fast

        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]
        mx.random.seed(41)
        dtype = getattr(mx, dtype_name)
        local_window, compress_ratio = 128, 128
        local_start = max(0, offset - local_window)
        local_length = offset - local_start + length
        pooled_length = (offset + length) // compress_ratio
        q = mx.random.normal((1, 64, length, 512), dtype=dtype)
        local_kv = mx.random.normal((1, 1, local_length, 512), dtype=dtype)
        pooled = mx.random.normal((1, pooled_length, 512), dtype=dtype)
        topk = mx.broadcast_to(
            mx.arange(pooled_length, dtype=mx.uint32)[None, None],
            (1, length, pooled_length),
        )
        sinks = mx.random.normal((64,), dtype=dtype)
        query_rows = mx.arange(length)[:, None]
        local_positions = mx.arange(local_length)[None]
        local_end = local_length - length + query_rows + 1
        local_start_rows = mx.maximum(0, local_end - local_window)
        pooled_positions = (mx.arange(pooled_length)[None] + 1) * compress_ratio - 1
        local_mask = (local_positions >= local_start_rows) & (
            local_positions < local_end
        )
        pooled_mask = pooled_positions <= offset + query_rows
        scale = 512**-0.5
        if not fast.has_symbol("deepseek_v4_sparse_attention"):
            pytest.skip("deepseek_v4_sparse_attention native kernel is unavailable")

        previous = dsv4._DEEPSEEK_V4_SPARSE_ATTENTION_NATIVE_DISABLED
        try:
            dsv4._DEEPSEEK_V4_SPARSE_ATTENTION_NATIVE_DISABLED = False
            actual = dsv4._sparse_pooled_attention(
                q,
                local_kv,
                pooled,
                topk,
                local_mask,
                pooled_mask,
                scale,
                sinks,
                q_offset=offset,
                compress_ratio=compress_ratio,
                local_window=local_window,
                native_only=True,
            )
            assert actual is not None
            mx.eval(actual)
            assert dsv4._DEEPSEEK_V4_SPARSE_ATTENTION_NATIVE_DISABLED is False

            dsv4._DEEPSEEK_V4_SPARSE_ATTENTION_NATIVE_DISABLED = True
            expected = dsv4._sparse_pooled_attention(
                q,
                local_kv,
                pooled,
                topk,
                local_mask,
                pooled_mask,
                scale,
                sinks,
            )
            mx.eval(expected)
        finally:
            dsv4._DEEPSEEK_V4_SPARSE_ATTENTION_NATIVE_DISABLED = previous

        max_abs = mx.max(
            mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))
        )
        assert mx.allclose(actual, expected, atol=0.02, rtol=0.02).item()
        assert float(max_abs.item()) <= max_tolerance


class TestDeepseekV4AttentionInputBackend:
    @staticmethod
    def _config(dsv4, ratio):
        return dsv4.ModelArgs(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            n_shared_experts=1,
            n_routed_experts=2,
            num_experts_per_tok=1,
            num_hash_layers=0,
            q_lora_rank=16,
            qk_rope_head_dim=4,
            head_dim=8,
            o_groups=1,
            o_lora_rank=8,
            index_n_heads=2,
            index_head_dim=4,
            index_topk=8,
            sliding_window=128,
            compress_ratios=[ratio],
        )

    @pytest.mark.parametrize("ratio,length", [(0, 16), (128, 129), (4, 40)])
    def test_attention_variants_preserve_stacked_input_projection_results(
        self, applied_patch, monkeypatch, ratio, length
    ):
        import mlx.core as mx
        import mlx.nn as nn

        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]
        layer = dsv4.v4_attention_factory(self._config(dsv4, ratio), 0)
        x = mx.random.normal((1, length, 16), dtype=mx.bfloat16)
        expected = layer(x)
        mx.eval(expected)
        projected = {
            "wq_a": layer.wq_a(x),
            "wkv": layer.wkv(x),
        }
        if ratio:
            projected.update(
                compressor_wkv=layer.compressor.wkv(x),
                compressor_wgate=layer.compressor.wgate(x),
            )
        if ratio == 4:
            projected.update(
                indexer_compressor_wkv=layer.indexer.compressor.wkv(x),
                indexer_compressor_wgate=layer.indexer.compressor.wgate(x),
                indexer_weights=layer.indexer.weights_proj(x),
            )
        mx.eval(*projected.values())

        calls = []

        def backend(owner, value):
            calls.append((owner, value.shape))
            return projected

        class FailLinear(nn.Module):
            def __call__(self, value):
                raise AssertionError("stacked projection fell back to its GPU linear")

        monkeypatch.setattr(dsv4, "_ANE_ATTENTION_INPUT_BACKEND", backend)
        layer.wq_a = FailLinear()
        layer.wkv = FailLinear()
        if ratio:
            layer.compressor.wkv = FailLinear()
            layer.compressor.wgate = FailLinear()
        if ratio == 4:
            layer.indexer.compressor.wkv = FailLinear()
            layer.indexer.compressor.wgate = FailLinear()
            layer.indexer.weights_proj = FailLinear()

        output = layer(x)
        mx.eval(output)

        assert output.shape == (1, length, 16)
        assert mx.array_equal(output, expected).item()
        assert calls == [(layer, x.shape)]


class TestPreLoadDispatch:
    """maybe_apply_pre_load_patches gates correctly on config.json model_type."""

    def test_no_dispatch_for_other_model_type(self, tmp_path):
        # Create a fake model dir with a non-deepseek config.
        config_path = tmp_path / "config.json"
        config_path.write_text('{"model_type": "llama"}')

        from omlx.utils.model_loading import maybe_apply_pre_load_patches

        # Should be a no-op (no exception). We can't easily assert that
        # apply_deepseek_v4_patch was NOT called because earlier tests
        # may have applied it already. Just verify no crash.
        maybe_apply_pre_load_patches(str(tmp_path))

    def test_no_dispatch_for_missing_config(self, tmp_path):
        # No config.json present.
        from omlx.utils.model_loading import maybe_apply_pre_load_patches

        maybe_apply_pre_load_patches(str(tmp_path))

    def test_dispatch_for_deepseek_v4(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text('{"model_type": "deepseek_v4"}')

        from omlx.patches.deepseek_v4 import is_applied
        from omlx.utils.model_loading import maybe_apply_pre_load_patches

        maybe_apply_pre_load_patches(str(tmp_path))
        # Patch must be applied after this dispatch (or already applied).
        assert is_applied() is True

    def test_dispatch_for_deepseek_v4_mtp_variant(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text('{"model_type": "deepseek_v4_mtp"}')

        from omlx.patches.deepseek_v4 import is_applied
        from omlx.utils.model_loading import maybe_apply_pre_load_patches

        maybe_apply_pre_load_patches(str(tmp_path))
        assert is_applied() is True


class TestMakeQuantizationConfigMtp:
    """make_quantization_config must cover the MTP fusion projections.

    Without explicit entries, mtp.<i>.e_proj / mtp.<i>.h_proj fall through
    to the affine default, whose QuantizedLinear expects a .biases tensor
    the fp8 checkpoint doesn't ship, and strict load fails."""

    def test_mtp_projections_get_mxfp8(self, applied_patch):
        import mlx.nn as nn

        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]

        class _MTPStub(nn.Module):
            def __init__(self):
                super().__init__()
                self.e_proj = nn.Linear(8, 8, bias=False)
                self.h_proj = nn.Linear(8, 8, bias=False)

        class _ModelStub(nn.Module):
            def __init__(self):
                super().__init__()
                self.mtp = [_MTPStub()]
                self.lm_head = nn.Linear(8, 8, bias=False)

        qcfg = dsv4.make_quantization_config(_ModelStub())
        mxfp8 = {"group_size": 32, "bits": 8, "mode": "mxfp8"}
        assert qcfg["mtp.0.e_proj"] == mxfp8
        assert qcfg["mtp.0.h_proj"] == mxfp8
        # Non-MTP paths keep the affine default (no per-path entry).
        assert "lm_head" not in qcfg

    def test_no_mtp_no_entries(self, applied_patch):
        import mlx.nn as nn

        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]

        class _ModelStub(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(8, 8, bias=False)

        qcfg = dsv4.make_quantization_config(_ModelStub())
        assert not any(k.startswith("mtp.") for k in qcfg)

    def test_dspark_main_projection_gets_mxfp8(self, applied_patch):
        import mlx.nn as nn

        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]

        class _DSparkStub(nn.Module):
            def __init__(self):
                super().__init__()
                self.main_proj = nn.Linear(24, 8, bias=False)

        class _ModelStub(nn.Module):
            def __init__(self):
                super().__init__()
                self.mtp = [_DSparkStub()]

        qcfg = dsv4.make_quantization_config(_ModelStub())
        assert qcfg["mtp.0.main_proj"] == {
            "group_size": 32,
            "bits": 8,
            "mode": "mxfp8",
        }


class TestDeepSeekV4SanitizeAffineSwitchMLP:
    """Sanitize should enable the FP16 affine routed-MoE fast path."""

    def test_affine_switch_mlp_scale_bias_cast_to_fp16(self, applied_patch):
        mx = pytest.importorskip("mlx.core")

        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]
        fake_model = SimpleNamespace(
            args=SimpleNamespace(
                num_hidden_layers=1,
                n_routed_experts=2,
                o_groups=1,
                o_lora_rank=1,
            )
        )
        weights = {
            "model.layers.0.ffn.switch_mlp.up_proj.weight": mx.zeros(
                (2, 4, 2), dtype=mx.uint32
            ),
            "model.layers.0.ffn.switch_mlp.up_proj.scales": mx.zeros(
                (2, 4, 1), dtype=mx.bfloat16
            ),
            "model.layers.0.ffn.switch_mlp.up_proj.biases": mx.zeros(
                (2, 4, 1), dtype=mx.bfloat16
            ),
            "model.layers.0.ffn.switch_mlp.down_proj.weight": mx.zeros(
                (2, 4, 2), dtype=mx.uint32
            ),
            "model.layers.0.ffn.switch_mlp.down_proj.scales": mx.zeros(
                (2, 4, 1), dtype=mx.bfloat16
            ),
            "model.layers.0.ffn.switch_mlp.down_proj.biases": mx.zeros(
                (2, 4, 1), dtype=mx.bfloat16
            ),
            "model.layers.0.ffn.shared_experts.up_proj.scales": mx.zeros(
                (4, 1), dtype=mx.bfloat16
            ),
        }

        out = dsv4.Model.sanitize(fake_model, dict(weights))

        assert out["model.layers.0.ffn.switch_mlp.up_proj.scales"].dtype == mx.float16
        assert out["model.layers.0.ffn.switch_mlp.up_proj.biases"].dtype == mx.float16
        assert out["model.layers.0.ffn.switch_mlp.down_proj.scales"].dtype == mx.float16
        assert out["model.layers.0.ffn.switch_mlp.down_proj.biases"].dtype == mx.float16
        assert (
            out["model.layers.0.ffn.shared_experts.up_proj.scales"].dtype == mx.bfloat16
        )


class TestDeepSeekV4SanitizeHcAliases:
    """Sanitize accepts both upstream HC key spellings for V4 checkpoints."""

    @staticmethod
    def _fake_model():
        return SimpleNamespace(
            args=SimpleNamespace(
                num_hidden_layers=1,
                n_routed_experts=0,
                o_groups=1,
                o_lora_rank=1,
            )
        )

    def test_dotted_hc_aliases_remap_to_model_modules(self, applied_patch):
        mx = pytest.importorskip("mlx.core")

        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]
        weights = {
            "model.layers.0.hc_attn.base": mx.zeros((1,), dtype=mx.float32),
            "model.layers.0.hc_attn.fn": mx.zeros((1, 1), dtype=mx.float32),
            "model.layers.0.hc_attn.scale": mx.zeros((3,), dtype=mx.float32),
            "model.layers.0.hc_ffn.base": mx.zeros((1,), dtype=mx.float32),
            "model.layers.0.hc_ffn.fn": mx.zeros((1, 1), dtype=mx.float32),
            "model.layers.0.hc_ffn.scale": mx.zeros((3,), dtype=mx.float32),
        }

        out = dsv4.Model.sanitize(self._fake_model(), dict(weights))

        assert "model.layers.0.attn_hc.base" in out
        assert "model.layers.0.attn_hc.fn" in out
        assert "model.layers.0.attn_hc.scale" in out
        assert "model.layers.0.ffn_hc.base" in out
        assert "model.layers.0.ffn_hc.fn" in out
        assert "model.layers.0.ffn_hc.scale" in out
        assert not any(".hc_attn." in key or ".hc_ffn." in key for key in out)

    def test_dotted_hc_alias_does_not_override_canonical_key(self, applied_patch):
        mx = pytest.importorskip("mlx.core")

        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]
        weights = {
            "model.layers.0.hc_attn.base": mx.zeros((1,), dtype=mx.float32),
            "model.layers.0.attn_hc.base": mx.zeros((2,), dtype=mx.float32),
        }

        out = dsv4.Model.sanitize(self._fake_model(), dict(weights))

        assert out["model.layers.0.attn_hc.base"].shape == (2,)
        assert "model.layers.0.hc_attn.base" not in out


class TestMtpSanitizeWoAReshape:
    """The MTP patch sanitize must reshape mtp.<i>.block.attn.wo_a from the
    2D nn.Linear layout to the 3D MultiLinear layout, like the backbone."""

    @pytest.fixture()
    def patched_sanitize(self, applied_patch):
        import omlx.patches.mlx_lm_mtp.deepseek_v4_model as mtp_dsv4

        mtp_dsv4.apply()
        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]
        return dsv4.Model.sanitize

    @staticmethod
    def _fake_model(with_mtp=True):
        class _Args:
            num_hidden_layers = 1
            num_nextn_predict_layers = 1
            o_groups = 2
            o_lora_rank = 4
            n_routed_experts = 2

        class _Fake:
            args = _Args()

        fake = _Fake()
        if with_mtp:
            fake.mtp = [object()]
        return fake

    def test_mtp_wo_a_2d_reshaped_to_3d(self, patched_sanitize):
        import mlx.core as mx

        weights = {
            "mtp.0.attn.wo_a.weight": mx.zeros((8, 16), dtype=mx.bfloat16),
        }
        out = patched_sanitize(self._fake_model(), weights)
        assert out["mtp.0.block.attn.wo_a.weight"].shape == (2, 4, 16)

    def test_mtp_wo_a_3d_unchanged(self, patched_sanitize):
        import mlx.core as mx

        weights = {
            "mtp.0.block.attn.wo_a.weight": mx.zeros((2, 4, 16), dtype=mx.bfloat16),
        }
        out = patched_sanitize(self._fake_model(), weights)
        assert out["mtp.0.block.attn.wo_a.weight"].shape == (2, 4, 16)

    def test_mtp_dotted_hc_alias_nested_under_block(self, patched_sanitize):
        import mlx.core as mx

        weights = {
            "mtp.0.hc_attn.base": mx.zeros((1,), dtype=mx.float32),
            "mtp.0.hc_ffn.scale": mx.zeros((3,), dtype=mx.float32),
        }

        out = patched_sanitize(self._fake_model(), weights)

        assert "mtp.0.block.attn_hc.base" in out
        assert "mtp.0.block.ffn_hc.scale" in out
        assert "mtp.0.hc_attn.base" not in out
        assert "mtp.0.hc_ffn.scale" not in out


class TestMtpBackboneInterface:
    """The patched DSv4 Model.__call__ must accept the full patched-backbone
    interface — batch_generator._call_backbone passes n_confirmed=1 during
    MTP verify cycles (crashed with TypeError before the fix)."""

    def test_call_accepts_n_confirmed(self, applied_patch):
        import omlx.patches.mlx_lm_mtp.deepseek_v4_model as mtp_dsv4

        mtp_dsv4.apply()
        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]
        sig = inspect.signature(dsv4.Model.__call__)
        assert "n_confirmed" in sig.parameters
        assert sig.parameters["n_confirmed"].default == 0
        assert "return_hidden" in sig.parameters


class TestPoolingCacheTrimRollback:
    """trim(1) must exactly undo the last (draft) token of an MTP verify
    update, including the pool-boundary case where the draft completed a
    compression window. Equivalence is checked behaviorally: a trimmed
    cache must evolve identically to a reference cache that never saw the
    rejected token."""

    @staticmethod
    def _push(cache, tokens, offset):
        """Feed raw per-token rows through the PoolingCache contract,
        compressing completed windows with a deterministic stand-in
        (mean over the window) like Compressor does."""
        import mlx.core as mx

        kv = tokens
        gate = tokens * 0.5
        r_kv, _r_gate, _ = cache.accumulate_windows(kv, gate, offset)
        if r_kv.size == 0:
            rows = mx.zeros((kv.shape[0], 0, kv.shape[-1]), dtype=kv.dtype)
        else:
            rows = mx.unflatten(r_kv, 1, (-1, cache.ratio)).mean(axis=2)
        return cache.update_and_fetch(rows)

    @staticmethod
    def _tok(values):
        import mlx.core as mx

        arr = mx.array(values, dtype=mx.float32)
        return mx.broadcast_to(arr[None, :, None], (1, len(values), 8))

    def _equivalence(self, cache_cls, prefix, verify, post, applied):
        """Drive cache through prefix + 2-token verify, trim the draft,
        push `post`; compare against a reference that never saw the draft."""
        import mlx.core as mx

        ratio = 4
        if cache_cls.__name__ == "BatchPoolingCache":
            cache = cache_cls(ratio, [0])
            ref = cache_cls(ratio, [0])
        else:
            cache = cache_cls(ratio)
            ref = cache_cls(ratio)

        pos = 0
        for chunk in prefix:
            self._push(cache, self._tok(chunk), pos)
            self._push(ref, self._tok(chunk), pos)
            pos += len(chunk)

        # Verify forward: [confirmed, draft] on cache; confirmed only on ref.
        self._push(cache, self._tok(verify), pos)
        assert cache.is_trimmable()
        assert cache.trim(1) == 1
        if cache_cls.__name__ == "BatchPoolingCache" and cache.pooled is not None:
            # A rejected speculative token may have completed a pooled
            # window.  The physical tail must be removed as well as hidden
            # from the logical length; DSpark's M=1 verify path consumes the
            # physical view directly.
            assert cache.pooled.shape[1] == max(cache._pool_lengths)
        self._push(ref, self._tok(verify[:-1]), pos)
        pos += len(verify) - 1

        out = self._push(cache, self._tok(post), pos)
        ref_out = self._push(ref, self._tok(post), pos)

        if out is None or getattr(out, "size", 0) == 0:
            assert ref_out is None or getattr(ref_out, "size", 0) == 0
        else:
            pl = getattr(cache, "_pool_lengths", None)
            n = pl[0] if pl is not None else out.shape[1]
            ref_n = ref._pool_lengths[0] if pl is not None else ref_out.shape[1]
            assert n == ref_n
            assert mx.allclose(out[:, :n], ref_out[:, :n]).item()
        assert (
            cache.remainder
            if isinstance(cache.remainder, int)
            else list(cache.remainder)
        ) == (ref.remainder if isinstance(ref.remainder, int) else list(ref.remainder))

    def test_easy_case_draft_in_buffer(self, applied_patch):
        from mlx_lm.models.cache import PoolingCache

        # After verify: remainder = (1 + 2) % 4 = 3 >= 1 -> buffer trim.
        self._equivalence(PoolingCache, [[1.0]], [2.0, 3.0], [4.0], applied_patch)

    def test_boundary_case_draft_completed_window(self, applied_patch):
        from mlx_lm.models.cache import PoolingCache

        # remainder before verify = 2; verify adds 2 -> window completes on
        # the draft token -> undo log path (drop pooled row, replay
        # confirmed into the buffer).
        self._equivalence(
            PoolingCache, [[1.0, 2.0]], [3.0, 4.0], [5.0, 6.0, 7.0], applied_patch
        )

    def test_boundary_case_with_existing_pool(self, applied_patch):
        from mlx_lm.models.cache import PoolingCache

        # One full window already pooled, then the boundary case again.
        self._equivalence(
            PoolingCache,
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0]],
            [7.0, 8.0],
            [9.0, 10.0, 11.0],
            applied_patch,
        )

    def test_batch_easy_case(self, applied_patch):
        from mlx_lm.models.cache import BatchPoolingCache

        self._equivalence(BatchPoolingCache, [[1.0]], [2.0, 3.0], [4.0], applied_patch)

    def test_batch_boundary_case(self, applied_patch):
        from mlx_lm.models.cache import BatchPoolingCache

        self._equivalence(
            BatchPoolingCache,
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0]],
            [7.0, 8.0],
            [9.0, 10.0, 11.0],
            applied_patch,
        )

    def test_untrimmable_when_no_undo_after_prompt(self, applied_patch):
        """Prompt-sized updates (L > 8) don't stash an undo log; a trim at
        a pool boundary right after one must report not-trimmable instead
        of corrupting state. (Updates up to L == 8 keep an undo so depth-k
        MTP verify windows can roll back.)"""
        from mlx_lm.models.cache import PoolingCache

        cache = PoolingCache(4)
        self._push(
            cache,
            self._tok([float(v) for v in range(1, 13)]),  # L = 12 > 8
            0,
        )
        assert cache.remainder == 0
        assert cache.pooled is not None
        assert not cache.is_trimmable()
        assert cache.trim(1) == 0

    def test_verify_sized_update_keeps_undo(self, applied_patch):
        """MTP verify windows (2 < L <= 8) stash an undo log: a trim right
        after one rolls back across the pool boundary instead of failing."""
        from mlx_lm.models.cache import PoolingCache

        cache = PoolingCache(4)
        self._push(cache, self._tok([1.0, 2.0, 3.0, 4.0]), 0)
        assert cache.remainder == 0
        assert cache.pooled is not None
        assert cache.is_trimmable()
        assert cache.trim(1) == 1
        # The completed window is undone: its 3 surviving tokens are back
        # in the remainder buffer and no pooled row remains visible.
        assert cache.remainder == 3
        assert cache.size() == 0

    def test_accepted_prefix_can_cross_pool_boundary(self, applied_patch):
        from mlx_lm.models.cache import PoolingCache

        # The four-row verify crosses the boundary on its first row. Keeping
        # three rows must retain that pooled result and restore the two raw
        # rows after it, exactly like three ordinary decode calls.
        self._equivalence(
            PoolingCache,
            [[1.0, 2.0, 3.0]],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0],
            applied_patch,
        )

    def test_singleton_batch_accepted_prefix_crosses_boundary(self, applied_patch):
        from mlx_lm.models.cache import BatchPoolingCache

        self._equivalence(
            BatchPoolingCache,
            [[1.0, 2.0, 3.0]],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0],
            applied_patch,
        )


class TestNaxMoEStockRouting:
    """NAX GPUs route prefill-sized MoE gemms to stock mx.gather_qmm."""

    @pytest.fixture(autouse=True)
    def _nax_off_by_default(self, monkeypatch):
        from omlx.patches.deepseek_v4 import switch_layers as sl

        # Pin detection off so the block-kernel tests behave identically on
        # M5-family machines; each test overrides what it needs.
        monkeypatch.setattr(sl, "is_nax_available", lambda: False)
        monkeypatch.setattr(sl, "_NAX_STOCK_MODE", "")
        yield

    def test_prefers_stock_for_prefill_route_counts_only(self, monkeypatch):
        from omlx.patches.deepseek_v4 import switch_layers as sl

        monkeypatch.setattr(sl, "is_nax_available", lambda: True)
        assert not sl._nax_prefers_stock(8)
        assert not sl._nax_prefers_stock(sl._NAX_STOCK_MIN_ROUTES - 1)
        assert sl._nax_prefers_stock(sl._NAX_STOCK_MIN_ROUTES)
        assert sl._nax_prefers_stock(1 << 20)

    def test_no_stock_routing_without_nax(self, monkeypatch):
        from omlx.patches.deepseek_v4 import switch_layers as sl

        assert not sl._nax_prefers_stock(1 << 20)

    def test_env_kill_switch_keeps_block_kernels(self, monkeypatch):
        from omlx.patches.deepseek_v4 import switch_layers as sl

        monkeypatch.setattr(sl, "is_nax_available", lambda: True)
        monkeypatch.setattr(sl, "_NAX_STOCK_MODE", "0")
        assert not sl._nax_prefers_stock(1 << 20)

    def test_env_force_routes_everything(self, monkeypatch):
        from omlx.patches.deepseek_v4 import switch_layers as sl

        monkeypatch.setattr(sl, "is_nax_available", lambda: True)
        monkeypatch.setattr(sl, "_NAX_STOCK_MODE", "1")
        assert sl._nax_prefers_stock(1)

    def test_native_block_kind_short_circuits_on_nax_prefill(self, monkeypatch):
        import mlx.core as mx

        from omlx.patches.deepseek_v4 import switch_layers as sl

        linear = sl.QuantizedSwitchLinear(
            64, 64, num_experts=2, bias=False, group_size=64, bits=4
        )
        monkeypatch.setattr(sl, "_nax_prefers_stock", lambda n: n >= 1024)
        prefill_x = mx.zeros((2048, 1, 64), dtype=mx.bfloat16)
        assert linear._native_block_kind(prefill_x, True) is None
        # Decode-sized calls fall through to the regular block-kernel gates:
        # the NAX gate must not change what they resolve to.
        decode_x = mx.zeros((8, 1, 64), dtype=mx.bfloat16)
        gated = linear._native_block_kind(decode_x, True)
        monkeypatch.setattr(sl, "_nax_prefers_stock", lambda n: False)
        assert gated == linear._native_block_kind(decode_x, True)


class TestSparseCompressedAttentionIndexerSkip:
    @staticmethod
    def _config(dsv4):
        return dsv4.ModelArgs(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            n_shared_experts=1,
            n_routed_experts=2,
            num_experts_per_tok=1,
            num_hash_layers=0,
            q_lora_rank=16,
            qk_rope_head_dim=4,
            head_dim=8,
            o_groups=1,
            o_lora_rank=8,
            index_n_heads=2,
            index_head_dim=4,
            index_topk=8,
            sliding_window=128,
            compress_ratios=[4],
        )

    def test_all_pooled_skips_scoring_and_preserves_indexer_cache(
        self, applied_patch, monkeypatch
    ):
        import mlx.core as mx
        from mlx_lm.models.cache import CacheList, PoolingCache, RotatingKVCache

        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]
        layer = dsv4.SparseCompressedAttention(self._config(dsv4), 0)
        x = mx.random.normal((1, 10, 16), dtype=mx.bfloat16)

        # The current full indexer is the reference for both selected rows and
        # every public part of its compressor cache after a non-aligned chunk.
        reference_cache = PoolingCache(4)
        q_residual = layer.q_norm(layer.wq_a(x))
        reference_topk = layer.indexer(
            x,
            q_residual,
            layer.rope,
            reference_cache,
            0,
        )
        mx.eval(reference_topk, *(v for v in reference_cache.state if v is not None))
        assert reference_topk.tolist() == [[[0, 1]] * 10]

        original_compressor_call = dsv4.Compressor.__call__
        indexer_compressor_calls = 0

        def compressor_spy(compressor, *args, **kwargs):
            nonlocal indexer_compressor_calls
            if compressor is layer.indexer.compressor:
                indexer_compressor_calls += 1
            return original_compressor_call(compressor, *args, **kwargs)

        def fail_full_indexer(*args, **kwargs):
            raise AssertionError("all-pooled attention must skip indexer scoring")

        monkeypatch.setattr(dsv4.Compressor, "__call__", compressor_spy)
        monkeypatch.setattr(dsv4.Indexer, "__call__", fail_full_indexer)

        comp_cache = PoolingCache(4)
        index_cache = PoolingCache(4)
        cache = CacheList(
            RotatingKVCache(max_size=128),
            comp_cache,
            index_cache,
        )
        output = layer(x, cache=cache)
        mx.eval(output, *(v for v in index_cache.state if v is not None))

        assert output.shape == (1, 10, 16)
        assert indexer_compressor_calls == 1
        assert comp_cache.offset == index_cache.offset == reference_cache.offset == 2
        assert index_cache.remainder == reference_cache.remainder == 2
        for actual, expected in zip(index_cache.state, reference_cache.state):
            assert (actual is None) == (expected is None)
            if actual is not None:
                assert mx.array_equal(actual, expected).item()


class TestIndexerFallbackTiling:
    """The MLX indexer fallback (used when the native glm_moe_dsa kernel is
    not built) tiles the pooled axis so its (B, heads, L, P) intermediate
    never crosses 2**31 elements — the boundary where mlx int32 kernel
    indexing silently zeroes the tail and corrupts top-k selection at
    >256k context — while keeping top-k selection identical to the untiled
    reduction."""

    def _reduce_and_ref(self):
        # The patch registers deepseek_v4_model.py as mlx_lm.models.deepseek_v4
        # (its relative `.base` import resolves there); import it by that name.
        import sys

        import mlx.core as mx

        dm = sys.modules["mlx_lm.models.deepseek_v4"]
        return mx, dm

    def test_head_reduce_matches_naive(self, applied_patch):
        mx, dm = self._reduce_and_ref()
        mx.random.seed(0)
        scores = mx.random.normal((1, 8, 16, 64))
        weights = mx.random.normal((1, 8, 16, 1))
        got = dm._indexer_head_reduce(scores, weights, 0.125)
        ref = (mx.maximum(scores, 0) * 0.125 * weights).sum(axis=1)
        assert float(mx.abs(got - ref).max()) < 1e-5

    def test_tensor_prefill_row_split_matches_full_indexer(
        self, applied_patch, monkeypatch
    ):
        mx, dm = self._reduce_and_ref()

        class Group:
            def __init__(self, rank):
                self._rank = rank

            def rank(self):
                return self._rank

            @staticmethod
            def size():
                return 2

        config = dm.ModelArgs(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            n_shared_experts=1,
            n_routed_experts=2,
            num_experts_per_tok=1,
            num_hash_layers=0,
            q_lora_rank=8,
            qk_rope_head_dim=4,
            head_dim=8,
            o_groups=1,
            o_lora_rank=8,
            index_n_heads=4,
            index_head_dim=8,
            index_topk=3,
            sliding_window=128,
            compress_ratios=[4],
        )
        indexer = dm.Indexer(config, compress_ratio=4)
        monkeypatch.setattr(dm, "_DEEPSEEK_V4_INDEXER_ROW_TP_MIN_POOL", 0)
        mx.random.seed(29)
        indexer.wq_b.weight = mx.random.normal(indexer.wq_b.weight.shape)
        indexer.weights_proj.weight = mx.random.normal(
            indexer.weights_proj.weight.shape
        )
        pooled = mx.random.normal((1, 7, 8), dtype=mx.bfloat16)
        monkeypatch.setattr(
            dm.Compressor,
            "__call__",
            lambda self, x, pool_cache, offset: pooled,
        )
        x = mx.random.normal((1, 5, 16), dtype=mx.bfloat16)
        q_residual = mx.random.normal((1, 5, 8), dtype=mx.bfloat16)
        rope = lambda value, offset: value

        full = indexer(x, q_residual, rope, None, 0)
        monkeypatch.setattr(
            dm,
            "_gather_indexer_rows",
            lambda local, total_rows, group, pooled_tokens=None: local,
        )
        indexer.row_sharding_group = Group(0)
        first = indexer(x, q_residual, rope, None, 0)
        indexer.row_sharding_group = Group(1)
        second = indexer(x, q_residual, rope, None, 0)
        reconstructed = mx.concatenate([first, second], axis=1)
        mx.eval(full, reconstructed)

        assert dm._balanced_row_ranges(5, 2) == ((0, 3), (3, 5))
        assert mx.array_equal(reconstructed, full).item()

    def test_tensor_prefill_uses_qualified_weighted_row_ranges(
        self, applied_patch, monkeypatch
    ):
        _mx, dm = self._reduce_and_ref()
        monkeypatch.setenv("OMLX_TP_SHARD_WEIGHTS", "3,5")
        monkeypatch.setattr(dm, "_DEEPSEEK_V4_WEIGHTED_INDEXER_ROWS", True)

        group = SimpleNamespace(size=lambda: 2)

        assert dm._weighted_row_ranges(1024, (3, 5)) == (
            (0, 384),
            (384, 1024),
        )
        assert dm._indexer_row_ranges(1024, group) == (
            (0, 384),
            (384, 1024),
        )
        assert dm._weighted_row_ranges(5, (3, 5)) == ((0, 1), (1, 5))

        monkeypatch.setattr(dm, "_DEEPSEEK_V4_WEIGHTED_INDEXER_ROWS", False)
        assert dm._indexer_row_ranges(5, group) == ((0, 3), (3, 5))

    def test_tensor_prefill_row_gather_removes_uneven_padding(
        self, applied_patch, monkeypatch
    ):
        mx, dm = self._reduce_and_ref()
        first = mx.array([[[1, 2], [3, 4], [5, 6]]], dtype=mx.uint32)
        second = mx.array([[[7, 8], [9, 10]]], dtype=mx.uint32)
        second_padded = mx.concatenate(
            [second.swapaxes(0, 1), mx.zeros((1, 1, 2), dtype=mx.uint32)],
            axis=0,
        )
        wire = mx.concatenate([first.swapaxes(0, 1), second_padded], axis=0)
        monkeypatch.setattr(
            mx.distributed,
            "all_gather",
            lambda value, group=None: wire,
        )
        group = SimpleNamespace(size=lambda: 2)

        gathered = dm._gather_indexer_rows(first, 5, group)

        assert gathered.tolist() == [
            [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        ]

    def test_tensor_prefill_explicit_rows_activate_only_after_pool_threshold(
        self, applied_patch, monkeypatch
    ):
        _mx, dm = self._reduce_and_ref()
        monkeypatch.setattr(dm, "_DEEPSEEK_V4_INDEXER_ROW_WEIGHTS", (9, 7))
        monkeypatch.setattr(
            dm,
            "_DEEPSEEK_V4_INDEXER_ROW_WEIGHTS_MIN_POOL",
            16000,
        )
        group = SimpleNamespace(size=lambda: 2)

        assert dm._indexer_row_ranges(1024, group, 15999) == (
            (0, 512),
            (512, 1024),
        )
        assert dm._indexer_row_ranges(1024, group, 16000) == (
            (0, 576),
            (576, 1024),
        )

    def test_tensor_prefill_invalid_explicit_rows_fail_closed_to_balanced(
        self, applied_patch, monkeypatch
    ):
        _mx, dm = self._reduce_and_ref()
        group = SimpleNamespace(size=lambda: 2)

        for weights in ((9,), (9, 0), (9, -1)):
            monkeypatch.setattr(dm, "_DEEPSEEK_V4_INDEXER_ROW_WEIGHTS", weights)
            assert dm._indexer_row_ranges(8, group, 20000) == ((0, 4), (4, 8))

    def test_tensor_prefill_weighted_row_gather_preserves_sequence_order(
        self, applied_patch, monkeypatch
    ):
        mx, dm = self._reduce_and_ref()
        monkeypatch.setenv("OMLX_TP_SHARD_WEIGHTS", "3,5")
        monkeypatch.setattr(dm, "_DEEPSEEK_V4_WEIGHTED_INDEXER_ROWS", True)
        first = mx.array([[[1], [2], [3]]], dtype=mx.uint32)
        second = mx.array([[[4], [5], [6], [7], [8]]], dtype=mx.uint32)
        first_padded = mx.concatenate(
            [first.swapaxes(0, 1), mx.zeros((2, 1, 1), dtype=mx.uint32)],
            axis=0,
        )
        wire = mx.concatenate([first_padded, second.swapaxes(0, 1)], axis=0)
        monkeypatch.setattr(
            mx.distributed,
            "all_gather",
            lambda value, group=None: wire,
        )
        group = SimpleNamespace(size=lambda: 2)

        gathered = dm._gather_indexer_rows(first, 8, group)

        assert gathered.tolist() == [[[1], [2], [3], [4], [5], [6], [7], [8]]]

    def test_tensor_prefill_equal_row_gather_reuses_collective_output(
        self, applied_patch, monkeypatch
    ):
        mx, dm = self._reduce_and_ref()
        first = mx.array([[[1, 2], [3, 4]]], dtype=mx.uint32)
        second = mx.array([[[5, 6], [7, 8]]], dtype=mx.uint32)
        wire = mx.concatenate([first.swapaxes(0, 1), second.swapaxes(0, 1)], axis=0)
        monkeypatch.setattr(
            mx.distributed,
            "all_gather",
            lambda value, group=None: wire,
        )
        monkeypatch.setattr(
            dm.mx,
            "concatenate",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("equal row shards must not be copied again")
            ),
        )

        gathered = dm._gather_indexer_rows(
            first,
            4,
            SimpleNamespace(size=lambda: 2),
        )

        assert gathered.tolist() == [[[1, 2], [3, 4], [5, 6], [7, 8]]]

    @pytest.mark.parametrize(
        "rank,local_values,peer_values,expected_ops",
        (
            (0, (1, 2, 3), (4, 5, 6, 7, 8), ("eval", "send", "recv")),
            (1, (4, 5, 6, 7, 8), (1, 2, 3), ("eval", "send", "recv")),
        ),
    )
    def test_tensor_prefill_tp2_p2p_gather_is_exact_and_ordered(
        self,
        applied_patch,
        monkeypatch,
        rank,
        local_values,
        peer_values,
        expected_ops,
    ):
        mx, dm = self._reduce_and_ref()
        monkeypatch.setenv("OMLX_TP_SHARD_WEIGHTS", "3,5")
        monkeypatch.setattr(dm, "_DEEPSEEK_V4_INDEXER_GATHER_P2P", True)
        monkeypatch.setattr(dm, "_DEEPSEEK_V4_WEIGHTED_INDEXER_ROWS", True)
        group = SimpleNamespace(size=lambda: 2, rank=lambda: rank)
        local = mx.array([[list(local_values)]], dtype=mx.uint32).swapaxes(1, 2)
        peer = mx.array([[list(peer_values)]], dtype=mx.uint32).swapaxes(1, 2)
        peer_rows_first = peer.swapaxes(0, 1)
        operations = []
        original_eval = mx.eval

        def traced_eval(*values):
            operations.append(("eval", tuple(values[0].shape)))
            return original_eval(*values)

        def send(value, destination, *, group=None):
            assert destination == 1 - rank
            assert group is not None
            operations.append(("send", tuple(value.shape)))
            return value

        def recv_like(template, source, *, group=None):
            assert source == 1 - rank
            assert group is not None
            assert tuple(template.shape) == tuple(peer_rows_first.shape)
            assert template.dtype == peer_rows_first.dtype
            operations.append(("recv", tuple(template.shape)))
            return peer_rows_first

        monkeypatch.setattr(dm.mx, "eval", traced_eval)
        monkeypatch.setattr(dm.mx.distributed, "send", send)
        monkeypatch.setattr(dm.mx.distributed, "recv_like", recv_like)
        monkeypatch.setattr(
            dm.mx.distributed,
            "all_gather",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("qualified TP2 must not enter all_gather")
            ),
        )

        gathered = dm._gather_indexer_rows(local, 8, group)
        original_eval(gathered)

        assert gathered.tolist() == [[[1], [2], [3], [4], [5], [6], [7], [8]]]
        assert tuple(op for op, _shape in operations) == expected_ops

    @pytest.mark.parametrize(
        "weighted,weights,total_rows",
        ((False, "3,5", 5), (True, "4,4", 4)),
    )
    def test_tensor_prefill_p2p_requires_weighted_uneven_rows(
        self, applied_patch, monkeypatch, weighted, weights, total_rows
    ):
        mx, dm = self._reduce_and_ref()
        monkeypatch.setenv("OMLX_TP_SHARD_WEIGHTS", weights)
        monkeypatch.setattr(dm, "_DEEPSEEK_V4_INDEXER_GATHER_P2P", True)
        monkeypatch.setattr(dm, "_DEEPSEEK_V4_WEIGHTED_INDEXER_ROWS", weighted)
        ranges = dm._indexer_row_ranges(total_rows, SimpleNamespace(size=lambda: 2))
        local_rows = ranges[0][1] - ranges[0][0]
        max_rows = max(stop - start for start, stop in ranges)
        local = mx.arange(local_rows, dtype=mx.uint32).reshape(1, local_rows, 1)
        wire = mx.arange(2 * max_rows, dtype=mx.uint32).reshape(
            2 * max_rows, 1, 1
        )
        group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
        calls = []

        def all_gather(value, *, group=None):
            calls.append(group)
            return wire

        monkeypatch.setattr(dm.mx.distributed, "all_gather", all_gather)
        monkeypatch.setattr(
            dm.mx.distributed,
            "send",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("equal or unweighted rows must use all_gather")
            ),
        )

        dm._gather_indexer_rows(local, total_rows, group)

        assert calls == [group]

    @pytest.mark.parametrize("rank", (0, 1))
    def test_tensor_prefill_tp2_p2p_preserves_weighted_row_order(
        self, applied_patch, monkeypatch, rank
    ):
        mx, dm = self._reduce_and_ref()
        monkeypatch.setenv("OMLX_TP_SHARD_WEIGHTS", "3,5")
        monkeypatch.setattr(dm, "_DEEPSEEK_V4_WEIGHTED_INDEXER_ROWS", True)
        monkeypatch.setattr(dm, "_DEEPSEEK_V4_INDEXER_GATHER_P2P", True)
        group = SimpleNamespace(size=lambda: 2, rank=lambda: rank)
        shards = (
            mx.array([[[1], [2], [3]]], dtype=mx.uint32),
            mx.array([[[4], [5], [6], [7], [8]]], dtype=mx.uint32),
        )
        local = shards[rank]
        peer_rows_first = shards[1 - rank].swapaxes(0, 1)

        monkeypatch.setattr(
            dm.mx.distributed,
            "send",
            lambda value, destination, *, group=None: value,
        )
        monkeypatch.setattr(
            dm.mx.distributed,
            "recv_like",
            lambda template, source, *, group=None: peer_rows_first,
        )
        gathered = dm._gather_indexer_rows(local, 8, group)
        mx.eval(gathered)

        assert gathered.tolist() == [[[1], [2], [3], [4], [5], [6], [7], [8]]]

    def test_tensor_prefill_tp2_p2p_has_collective_rollback(
        self, applied_patch, monkeypatch
    ):
        mx, dm = self._reduce_and_ref()
        monkeypatch.setattr(dm, "_DEEPSEEK_V4_INDEXER_GATHER_P2P", False)
        first = mx.array([[[1], [2]]], dtype=mx.uint32)
        second = mx.array([[[3], [4]]], dtype=mx.uint32)
        wire = mx.concatenate([first.swapaxes(0, 1), second.swapaxes(0, 1)], axis=0)
        group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
        calls = []

        def all_gather(value, *, group=None):
            calls.append((tuple(value.shape), group))
            return wire

        monkeypatch.setattr(dm.mx.distributed, "all_gather", all_gather)
        monkeypatch.setattr(
            dm.mx.distributed,
            "send",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("rollback must not enter send")
            ),
        )
        monkeypatch.setattr(
            dm.mx.distributed,
            "recv_like",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("rollback must not enter recv")
            ),
        )

        gathered = dm._gather_indexer_rows(first, 4, group)

        assert gathered.tolist() == [[[1], [2], [3], [4]]]
        assert calls == [((2, 1, 1), group)]

    def test_tensor_prefill_p2p_hard_gates_tp2(
        self, applied_patch, monkeypatch
    ):
        mx, dm = self._reduce_and_ref()
        monkeypatch.setattr(dm, "_DEEPSEEK_V4_INDEXER_GATHER_P2P", True)
        local = mx.array([[[1], [2]]], dtype=mx.uint32)
        wire = mx.concatenate(
            [
                local.swapaxes(0, 1),
                mx.array([[[3], [4]]], dtype=mx.uint32).swapaxes(0, 1),
                mx.array([[[5], [6]]], dtype=mx.uint32).swapaxes(0, 1),
            ],
            axis=0,
        )
        group = SimpleNamespace(size=lambda: 3, rank=lambda: 0)
        calls = []

        def all_gather(value, *, group=None):
            calls.append(group)
            return wire

        monkeypatch.setattr(dm.mx.distributed, "all_gather", all_gather)
        monkeypatch.setattr(
            dm.mx.distributed,
            "send",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("non-TP2 world must not enter send")
            ),
        )

        gathered = dm._gather_indexer_rows(local, 6, group)

        assert gathered.tolist() == [[[1], [2], [3], [4], [5], [6]]]
        assert calls == [group]

    def test_missing_native_warning_fires_once(
        self, applied_patch, caplog, monkeypatch
    ):
        import logging
        import sys

        import mlx.core as mx

        dm = sys.modules["mlx_lm.models.deepseek_v4"]
        config = dm.ModelArgs(
            hidden_size=16,
            q_lora_rank=16,
            qk_rope_head_dim=2,
            num_hidden_layers=1,
            compress_ratios=[4],
            index_n_heads=32,
            index_head_dim=128,
            index_topk=512,
        )
        indexer = dm.Indexer(config, compress_ratio=4)
        pooled = mx.zeros((1, 577, 128), dtype=mx.float16)
        monkeypatch.setattr(
            dm.Compressor,
            "__call__",
            lambda self, x, pool_cache, offset: pooled,
        )
        monkeypatch.setattr(dm, "native_indexer_available", lambda: False)
        monkeypatch.setattr(dm, "native_indexer_disabled", lambda: False)
        monkeypatch.setattr(dm, "_DEEPSEEK_V4_INDEXER_FALLBACK_WARNED", False)

        x = mx.zeros((1, 65, 16), dtype=mx.float16)
        projected_q = mx.zeros((1, 32, 65, 128), dtype=mx.float16)
        projected_weights = mx.zeros((1, 65, 32), dtype=mx.float16)
        with caplog.at_level(logging.WARNING, logger=dm.__name__):
            for _ in range(2):
                result = indexer(
                    x,
                    q_residual=x,
                    position_rope=None,
                    pool_cache=None,
                    offset=0,
                    projected_q=projected_q,
                    projected_weights=projected_weights,
                )
                mx.eval(result)

        fallback_warnings = [
            record
            for record in caplog.records
            if "native dsa_indexer_scores/dsa_topk_indices unavailable"
            in record.getMessage()
        ]
        assert len(fallback_warnings) == 1

    def test_tiling_selects_identical_topk(self, applied_patch):
        # Split a non-reduced axis: the top-k indices must be bit-stable
        # regardless of tile size, at pooled counts that straddle the tile.
        mx, dm = self._reduce_and_ref()
        mx.random.seed(1)
        heads, length, head_dim, topk = 64, 32, 128, 64
        scale = head_dim**-0.5

        def untiled(q, pooled, w):
            wf = (w.astype(mx.float32)).swapaxes(-1, -2)[..., None]
            s = q.astype(mx.float32) @ pooled[:, None].swapaxes(-1, -2).astype(
                mx.float32
            )
            return dm._indexer_head_reduce(s, wf, scale)

        def tiled(q, pooled, w, tile):
            wf = (w.astype(mx.float32)).swapaxes(-1, -2)[..., None]
            qf = q.astype(mx.float32)
            kf = pooled[:, None].swapaxes(-1, -2).astype(mx.float32)
            pool_count = pooled.shape[1]
            if pool_count <= tile:
                return dm._indexer_head_reduce(qf @ kf, wf, scale)
            return mx.concatenate(
                [
                    dm._indexer_head_reduce(qf @ kf[..., s : s + tile], wf, scale)
                    for s in range(0, pool_count, tile)
                ],
                axis=-1,
            )

        for pool_count in (255, 256, 257, 800):
            q = mx.random.normal((1, heads, length, head_dim))
            pooled = mx.random.normal((1, pool_count, head_dim))
            w = mx.random.normal((1, length, heads))
            a = untiled(q, pooled, w)
            b = tiled(q, pooled, w, tile=256)
            k = min(topk, pool_count)
            ia = mx.sort(mx.argpartition(-a, k - 1, axis=-1)[..., :k], axis=-1)
            ib = mx.sort(mx.argpartition(-b, k - 1, axis=-1)[..., :k], axis=-1)
            assert (
                int((ia != ib).sum()) == 0
            ), f"top-k differs at pool_count={pool_count}"

    def test_tile_stays_under_int32_index_limit(self, applied_patch):
        # The prefill chunk is 512 and index heads are 64; the tiled matmul
        # output must stay below 2**31 elements at any context length.
        _, dm = self._reduce_and_ref()
        assert 64 * 512 * dm._INDEXER_POOL_TILE < 2**31
        assert dm._INDEXER_MAX_ELEMS < 2**31


class TestDS4NAXIndexerScoreDispatch:
    @staticmethod
    def _exact_config(dm):
        return SimpleNamespace(
            model_type="deepseek_v4",
            vocab_size=129280,
            hidden_size=4096,
            moe_intermediate_size=2048,
            num_hidden_layers=43,
            num_attention_heads=64,
            n_routed_experts=256,
            num_experts_per_tok=6,
            max_position_embeddings=1048576,
            index_n_heads=64,
            index_head_dim=128,
            index_topk=512,
        )

    def test_startup_gate_requires_exact_ds4f_ratio4_and_nax(
        self, applied_patch, monkeypatch
    ):
        dm = sys.modules["mlx_lm.models.deepseek_v4"]
        cfg = self._exact_config(dm)
        monkeypatch.setattr(dm, "_DEEPSEEK_V4_NAX_INDEXER_SCORE", True)
        monkeypatch.setattr(dm, "is_nax_available", lambda: True)
        assert dm._dsv4f_nax_indexer_score_enabled(cfg, 4)
        assert not dm._dsv4f_nax_indexer_score_enabled(cfg, 128)

        monkeypatch.setattr(dm, "is_nax_available", lambda: False)
        assert not dm._dsv4f_nax_indexer_score_enabled(cfg, 4)
        monkeypatch.setattr(dm, "is_nax_available", lambda: True)
        monkeypatch.setattr(dm, "_DEEPSEEK_V4_NAX_INDEXER_SCORE", False)
        assert not dm._dsv4f_nax_indexer_score_enabled(cfg, 4)

        monkeypatch.setattr(dm, "_DEEPSEEK_V4_NAX_INDEXER_SCORE", True)
        bad = SimpleNamespace(**vars(cfg))
        bad.index_n_heads = 32
        assert not dm._dsv4f_nax_indexer_score_enabled(bad, 4)

    @pytest.mark.parametrize(
        "device_name", ("Apple M2 Ultra", "Apple M3 Ultra", "Apple M5 Max")
    )
    def test_lossless_mma_gate_accepts_qualified_apple_chips(
        self, applied_patch, monkeypatch, device_name
    ):
        dm = sys.modules["mlx_lm.models.deepseek_v4"]
        cfg = self._exact_config(dm)
        monkeypatch.setattr(dm.mx, "device_info", lambda: {"device_name": device_name})
        monkeypatch.setattr(dm, "_DEEPSEEK_V4_MMA_SCORE", True)

        assert dm._dsv4f_mma_score_enabled(cfg, 4)
        assert not dm._dsv4f_mma_score_enabled(cfg, 128)

    def test_lossless_mma_gate_rejects_unknown_chip_and_config(
        self, applied_patch, monkeypatch
    ):
        dm = sys.modules["mlx_lm.models.deepseek_v4"]
        cfg = self._exact_config(dm)
        monkeypatch.setattr(dm, "_DEEPSEEK_V4_MMA_SCORE", True)
        monkeypatch.setattr(
            dm.mx, "device_info", lambda: {"device_name": "Apple M6 Ultra"}
        )
        assert not dm._dsv4f_mma_score_enabled(cfg, 4)

        monkeypatch.setattr(
            dm.mx, "device_info", lambda: {"device_name": "Apple M3 Ultra"}
        )
        bad = SimpleNamespace(**vars(cfg))
        bad.index_topk = 256
        assert not dm._dsv4f_mma_score_enabled(bad, 4)

    @pytest.mark.parametrize(
        "architecture,expected",
        (("applegpu_g15d", True), ("applegpu_g17s", False)),
    )
    def test_mma_partition_is_forwarded_only_on_g15d(
        self, applied_patch, monkeypatch, architecture, expected
    ):
        import mlx.core as mx

        from omlx.custom_kernels.glm_moe_dsa import fast

        dm = sys.modules["mlx_lm.models.deepseek_v4"]
        B, L, H, D, N = 1, 64, 64, 128, 513
        pooled = mx.zeros((B, N, D), dtype=mx.bfloat16)
        cache = dm.PoolingCache(4)
        cache.pooled = pooled
        indexer = SimpleNamespace(
            n_heads=H,
            head_dim=D,
            index_topk=512,
            compressor=lambda x, pool_cache, offset: pooled,
            scale=D**-0.5,
            _m2_mma_score=True,
            _nax_indexer_score=False,
            row_sharding_group=None,
        )
        monkeypatch.setattr(dm, "native_indexer_available", lambda: True)
        monkeypatch.setattr(dm, "native_indexer_disabled", lambda: False)
        monkeypatch.setattr(fast, "_EXT_MMA_SCORE", True)
        monkeypatch.setattr(fast, "_EXT_MMA_WM4", True)
        monkeypatch.setattr(
            fast.mx,
            "device_info",
            lambda: {"architecture": architecture},
        )
        monkeypatch.setattr(fast, "has_symbol", lambda name: True)
        seen = []

        def scores(q, k, weights, **kwargs):
            seen.append(kwargs.get("use_wm4_wn1"))
            return mx.zeros((B, 1, q.shape[2], k.shape[2]), dtype=q.dtype)

        monkeypatch.setattr(fast, "dsa_indexer_scores_mma", scores)
        monkeypatch.setattr(
            fast,
            "dsa_topk_indices",
            lambda scores, topk, **kwargs: mx.broadcast_to(
                mx.arange(topk, dtype=mx.uint32)[None, None, None],
                (B, 1, scores.shape[2], topk),
            ),
        )
        x = mx.zeros((B, L, 8), dtype=mx.bfloat16)
        q = mx.zeros((B, H, L, D), dtype=mx.bfloat16)
        weights = mx.zeros((B, L, H), dtype=mx.bfloat16)
        out = dm.Indexer.__call__(
            indexer,
            x,
            q_residual=x,
            position_rope=None,
            pool_cache=cache,
            offset=2048,
            projected_q=q,
            projected_weights=weights,
        )
        assert out.shape == (B, L, 512)
        assert seen == [expected]

    @pytest.mark.parametrize("physical_tail,expected_ratio", ((False, 4), (True, 0)))
    def test_singleton_batch_mask_reaches_hierarchy_only_when_uniform(
        self, applied_patch, monkeypatch, physical_tail, expected_ratio
    ):
        import mlx.core as mx
        from mlx_lm.models.cache import BatchPoolingCache

        from omlx.custom_kernels.glm_moe_dsa import fast

        dm = sys.modules["mlx_lm.models.deepseek_v4"]
        B, L, H, D, N, offset = 1, 64, 64, 128, 577, 2304
        pooled = mx.zeros((B, N, D), dtype=mx.bfloat16)
        cache = BatchPoolingCache(4, [0])
        cache._pool_buf = pooled
        cache._pool_extent = N
        cache._pool_lengths = [N - 1 if physical_tail else N]
        indexer = SimpleNamespace(
            n_heads=H,
            head_dim=D,
            index_topk=512,
            compressor=lambda x, pool_cache, offset: pooled,
            scale=D**-0.5,
            _m2_mma_score=True,
            _nax_indexer_score=False,
            row_sharding_group=None,
        )
        monkeypatch.setattr(dm, "native_indexer_available", lambda: True)
        monkeypatch.setattr(dm, "native_indexer_disabled", lambda: False)
        monkeypatch.setattr(fast, "_EXT_MMA_SCORE", True)
        seen = {"hierarchy": [], "score": []}

        def hierarchy(*args, **kwargs):
            seen["hierarchy"].append((kwargs["ratio"], kwargs["query_offset"]))
            return None

        def scores(q, k, weights, **kwargs):
            seen["score"].append(
                (kwargs["mask_ratio"], kwargs["mask_q_offset"])
            )
            return mx.zeros((B, 1, q.shape[2], k.shape[2]), dtype=q.dtype)

        monkeypatch.setattr(dm, "hierarchical_topk", hierarchy)
        monkeypatch.setattr(fast, "dsa_indexer_scores_mma", scores)
        monkeypatch.setattr(
            fast,
            "dsa_topk_indices",
            lambda score_sheet, topk, **kwargs: mx.broadcast_to(
                mx.arange(topk, dtype=mx.uint32)[None, None, None],
                (B, 1, score_sheet.shape[2], topk),
            ),
        )
        x = mx.zeros((B, L, 8), dtype=mx.bfloat16)
        q = mx.zeros((B, H, L, D), dtype=mx.bfloat16)
        weights = mx.zeros((B, L, H), dtype=mx.bfloat16)

        out = dm.Indexer.__call__(
            indexer,
            x,
            q_residual=x,
            position_rope=None,
            pool_cache=cache,
            offset=offset,
            projected_q=q,
            projected_weights=weights,
        )

        assert out.shape == (B, L, 512)
        assert seen["hierarchy"] == [(expected_ratio, offset)]
        assert seen["score"] == [
            (expected_ratio, offset if expected_ratio else 0)
        ]

    def test_foldable_batch_mask_rejects_true_batches(self, applied_patch):
        import mlx.core as mx
        from mlx_lm.models.cache import BatchPoolingCache

        dm = sys.modules["mlx_lm.models.deepseek_v4"]
        cache = BatchPoolingCache(4, [0, 0])
        cache._pool_lengths = [8, 8]
        pmask = mx.ones((2, 16, 8), dtype=mx.bool_)

        assert (
            dm._foldable_pool_mask_ratio(
                cache,
                pmask,
                batch_size=2,
                pooled_tokens=8,
                query_offset=32,
            )
            == 0
        )

    @staticmethod
    def _run_projected_indexer(dm, monkeypatch, *, group=None):
        import mlx.core as mx

        from omlx.custom_kernels.glm_moe_dsa import fast

        B, L, H, D, N = 1, 16, 64, 128, 513
        pooled = mx.zeros((B, N, D), dtype=mx.bfloat16)
        cache = dm.PoolingCache(4)
        cache.pooled = pooled
        indexer = SimpleNamespace(
            n_heads=H,
            head_dim=D,
            index_topk=512,
            compressor=lambda x, pool_cache, offset: pooled,
            scale=D**-0.5,
            _m2_mma_score=False,
            _nax_indexer_score=True,
            row_sharding_group=group,
        )
        monkeypatch.setattr(dm, "native_indexer_available", lambda: True)
        monkeypatch.setattr(dm, "native_indexer_disabled", lambda: False)
        monkeypatch.setattr(fast, "_EXT_NAX_SCORE", True)
        monkeypatch.setattr(fast, "dsa_indexer_nax_kernels_built", lambda: True)
        monkeypatch.setattr(fast, "has_symbol", lambda name: True)
        seen = []

        def scores(q, k, weights, **kwargs):
            seen.append(kwargs.get("use_nax"))
            return mx.zeros((B, 1, q.shape[2], k.shape[2]), dtype=q.dtype)

        monkeypatch.setattr(fast, "dsa_indexer_scores", scores)
        monkeypatch.setattr(
            fast,
            "dsa_topk_indices",
            lambda scores, topk, **kwargs: mx.broadcast_to(
                mx.arange(topk, dtype=mx.uint32)[None, None, None],
                (B, 1, scores.shape[2], topk),
            ),
        )
        x = mx.zeros((B, L, 8), dtype=mx.bfloat16)
        q = mx.zeros((B, H, L, D), dtype=mx.bfloat16)
        weights = mx.zeros((B, L, H), dtype=mx.bfloat16)
        out = dm.Indexer.__call__(
            indexer,
            x,
            q_residual=x,
            position_rope=None,
            pool_cache=cache,
            offset=2048,
            projected_q=q,
            projected_weights=weights,
        )
        return out, seen

    def test_single_m5_path_forwards_nax_hint(self, applied_patch, monkeypatch):
        dm = sys.modules["mlx_lm.models.deepseek_v4"]
        out, seen = self._run_projected_indexer(dm, monkeypatch)
        assert out.shape == (1, 16, 512)
        assert seen == [True]

    def test_replicated_tp_rows_stay_on_steel(self, applied_patch, monkeypatch):
        dm = sys.modules["mlx_lm.models.deepseek_v4"]

        class Group:
            @staticmethod
            def size():
                return 2

            @staticmethod
            def rank():
                return 0

        out, seen = self._run_projected_indexer(dm, monkeypatch, group=Group())
        assert out.shape == (1, 16, 512)
        assert seen == [False]


class TestNativeKernelLatchSemantics:
    """ValueError shape rejections (std::invalid_argument, raised before any
    GPU work) must fall back per-call WITHOUT latching the process-wide
    native-disable flags; genuine runtime failures still latch.

    Regression test for the stale-binary incident: an extension built before
    the unaligned-tail fix raised ValueError on the first 327-token prefill
    chunk and latched the fp32 fallback for the whole process, multiplying
    resident memory at long context.
    """

    @staticmethod
    def _sparse_args():
        import mlx.core as mx

        q = mx.zeros((1, 64, 5, 512), dtype=mx.bfloat16)
        local_kv = mx.zeros((1, 1, 5, 512), dtype=mx.bfloat16)
        pooled = mx.zeros((1, 64, 512), dtype=mx.bfloat16)
        topk = mx.broadcast_to(
            mx.arange(64, dtype=mx.uint32)[None, None], (1, 5, 64)
        )
        sinks = mx.zeros((64,), dtype=mx.bfloat16)
        return q, local_kv, pooled, topk, sinks

    def test_sparse_attention_value_error_falls_back_without_latch(
        self, applied_patch, monkeypatch
    ):
        import mlx.core as mx

        from omlx.custom_kernels.glm_moe_dsa import fast

        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]
        monkeypatch.setattr(
            dsv4, "_DEEPSEEK_V4_SPARSE_ATTENTION_NATIVE_DISABLED", False
        )
        monkeypatch.setattr(dsv4, "_DEEPSEEK_V4_SPARSE_ATTN_SHAPE_WARNED", False)
        monkeypatch.setattr(fast, "has_symbol", lambda name: True)
        calls = []

        def reject(*args):
            calls.append(1)
            raise ValueError("unsupported M3 GLM shape.")

        monkeypatch.setattr(fast, "deepseek_v4_sparse_attention", reject)
        q, local_kv, pooled, topk, sinks = self._sparse_args()

        out = dsv4._sparse_pooled_attention(
            q, local_kv, pooled, topk, None, None, 512**-0.5, sinks,
            q_offset=0, compress_ratio=128, local_window=128,
        )
        mx.eval(out)
        assert out.shape == q.shape  # composed MLX fallback produced output
        assert dsv4._DEEPSEEK_V4_SPARSE_ATTENTION_NATIVE_DISABLED is False
        assert dsv4._DEEPSEEK_V4_SPARSE_ATTN_SHAPE_WARNED is True

        # Native path stays armed: a second call attempts the kernel again.
        dsv4._sparse_pooled_attention(
            q, local_kv, pooled, topk, None, None, 512**-0.5, sinks,
            q_offset=0, compress_ratio=128, local_window=128,
        )
        assert len(calls) == 2

    def test_sparse_attention_runtime_error_still_latches(
        self, applied_patch, monkeypatch
    ):
        from omlx.custom_kernels.glm_moe_dsa import fast

        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]
        monkeypatch.setattr(
            dsv4, "_DEEPSEEK_V4_SPARSE_ATTENTION_NATIVE_DISABLED", False
        )
        monkeypatch.setattr(dsv4, "_DEEPSEEK_V4_SPARSE_ATTN_SHAPE_WARNED", False)
        monkeypatch.setattr(fast, "has_symbol", lambda name: True)
        calls = []

        def boom(*args):
            calls.append(1)
            raise RuntimeError("metal command buffer failed")

        monkeypatch.setattr(fast, "deepseek_v4_sparse_attention", boom)
        q, local_kv, pooled, topk, sinks = self._sparse_args()

        out = dsv4._sparse_pooled_attention(
            q, local_kv, pooled, topk, None, None, 512**-0.5, sinks,
            q_offset=0, compress_ratio=128, local_window=128,
        )
        assert out is not None
        assert dsv4._DEEPSEEK_V4_SPARSE_ATTENTION_NATIVE_DISABLED is True

        # Latched: the kernel is not attempted again.
        dsv4._sparse_pooled_attention(
            q, local_kv, pooled, topk, None, None, 512**-0.5, sinks,
            q_offset=0, compress_ratio=128, local_window=128,
        )
        assert len(calls) == 1

    @staticmethod
    def _indexer_config(dsv4):
        return dsv4.ModelArgs(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            n_shared_experts=1,
            n_routed_experts=2,
            num_experts_per_tok=1,
            num_hash_layers=0,
            q_lora_rank=16,
            qk_rope_head_dim=4,
            head_dim=8,
            o_groups=1,
            o_lora_rank=8,
            index_n_heads=64,
            index_head_dim=128,
            index_topk=512,
            sliding_window=128,
            compress_ratios=[4],
        )

    def _run_indexer(self, dsv4, indexer):
        import mlx.core as mx

        x = mx.zeros((1, 2, 16), dtype=mx.bfloat16)
        q_residual = mx.zeros((1, 2, 16), dtype=mx.bfloat16)
        return indexer(x, q_residual, lambda q, offset: q, None, 0)

    def test_indexer_value_error_falls_back_without_latch(
        self, applied_patch, monkeypatch
    ):
        import mlx.core as mx

        from omlx.custom_kernels.glm_moe_dsa import fast
        from omlx.patches.deepseek_v4 import indexer_dispatch

        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]
        monkeypatch.setattr(indexer_dispatch, "_NATIVE_INDEXER_DISABLED", False)
        monkeypatch.setattr(dsv4, "_DEEPSEEK_V4_INDEXER_SHAPE_WARNED", False)
        monkeypatch.setattr(fast, "has_symbol", lambda name: True)
        monkeypatch.setattr(
            dsv4.Compressor,
            "__call__",
            lambda self, x, pool_cache, offset: mx.zeros(
                (x.shape[0], 600, self.head_dim), dtype=x.dtype
            ),
        )
        calls = []

        def reject(*args, **kwargs):
            calls.append(1)
            raise ValueError("unsupported M3 GLM shape.")

        monkeypatch.setattr(fast, "dsa_indexer_scores", reject)
        monkeypatch.setattr(fast, "dsa_indexer_scores_mma", reject)

        indexer = dsv4.Indexer(self._indexer_config(dsv4), 4)
        indexer.set_dtype(mx.bfloat16)
        out = self._run_indexer(dsv4, indexer)
        mx.eval(out)
        assert out.shape == (1, 2, 512)  # fp32 MLX fallback produced top-k
        assert indexer_dispatch.native_indexer_disabled() is False
        assert dsv4._DEEPSEEK_V4_INDEXER_SHAPE_WARNED is True

        # Native path stays armed: a second call attempts the kernel again.
        self._run_indexer(dsv4, indexer)
        assert len(calls) == 2

    def test_indexer_runtime_error_still_latches(
        self, applied_patch, monkeypatch
    ):
        import mlx.core as mx

        from omlx.custom_kernels.glm_moe_dsa import fast
        from omlx.patches.deepseek_v4 import indexer_dispatch

        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]
        monkeypatch.setattr(indexer_dispatch, "_NATIVE_INDEXER_DISABLED", False)
        monkeypatch.setattr(dsv4, "_DEEPSEEK_V4_INDEXER_SHAPE_WARNED", False)
        monkeypatch.setattr(fast, "has_symbol", lambda name: True)
        monkeypatch.setattr(
            dsv4.Compressor,
            "__call__",
            lambda self, x, pool_cache, offset: mx.zeros(
                (x.shape[0], 600, self.head_dim), dtype=x.dtype
            ),
        )
        calls = []

        def boom(*args, **kwargs):
            calls.append(1)
            raise RuntimeError("metal command buffer failed")

        monkeypatch.setattr(fast, "dsa_indexer_scores", boom)
        monkeypatch.setattr(fast, "dsa_indexer_scores_mma", boom)

        indexer = dsv4.Indexer(self._indexer_config(dsv4), 4)
        indexer.set_dtype(mx.bfloat16)
        out = self._run_indexer(dsv4, indexer)
        mx.eval(out)
        assert out.shape == (1, 2, 512)
        assert indexer_dispatch.native_indexer_disabled() is True

        # Latched: the kernel is not attempted again.
        self._run_indexer(dsv4, indexer)
        assert len(calls) == 1

    def test_dspark_topk_value_error_falls_back_without_latch(
        self, applied_patch, monkeypatch
    ):
        import mlx.core as mx

        from omlx.custom_kernels.glm_moe_dsa import fast

        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]
        monkeypatch.setattr(
            dsv4, "_DEEPSEEK_V4_DSPARK_TOPK_NATIVE_DISABLED", False
        )
        monkeypatch.setattr(dsv4, "_DEEPSEEK_V4_DSPARK_TOPK_SHAPE_WARNED", False)
        monkeypatch.setattr(fast, "has_symbol", lambda name: True)
        calls = []

        def reject(*args):
            calls.append(1)
            raise ValueError("unsupported M3 GLM shape.")

        monkeypatch.setattr(fast, "dspark_fp32_topk_indices", reject)

        # Batched path: >1 rows, every row longer than top-k.
        indexer = SimpleNamespace(index_topk=8, n_heads=2, scale=0.5)
        pooled_rows = [
            mx.zeros((1, 10, 4), dtype=mx.bfloat16),
            mx.zeros((1, 11, 4), dtype=mx.bfloat16),
        ]
        projected_q = mx.zeros((1, 2, 2, 4), dtype=mx.bfloat16)
        projected_weights = mx.zeros((1, 2, 2), dtype=mx.bfloat16)
        result = dsv4._batch_indexer_rows(
            indexer, pooled_rows, projected_q, projected_weights
        )
        mx.eval(*result)
        assert [r.shape for r in result] == [(1, 1, 8), (1, 1, 8)]
        assert dsv4._DEEPSEEK_V4_DSPARK_TOPK_NATIVE_DISABLED is False
        assert dsv4._DEEPSEEK_V4_DSPARK_TOPK_SHAPE_WARNED is True

        # Grouped path: same-length rows, k == 512 gate.
        indexer512 = SimpleNamespace(index_topk=512, n_heads=2, scale=0.5)
        result = dsv4._batch_indexer_rows(
            indexer512,
            [mx.zeros((1, 600, 4), dtype=mx.bfloat16)],
            mx.zeros((1, 2, 1, 4), dtype=mx.bfloat16),
            mx.zeros((1, 1, 2), dtype=mx.bfloat16),
        )
        mx.eval(*result)
        assert [r.shape for r in result] == [(1, 1, 512)]
        assert dsv4._DEEPSEEK_V4_DSPARK_TOPK_NATIVE_DISABLED is False
        # Both paths kept the native kernel armed.
        assert len(calls) == 2

    def test_dspark_topk_runtime_error_still_latches(
        self, applied_patch, monkeypatch
    ):
        import mlx.core as mx

        from omlx.custom_kernels.glm_moe_dsa import fast

        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]
        monkeypatch.setattr(
            dsv4, "_DEEPSEEK_V4_DSPARK_TOPK_NATIVE_DISABLED", False
        )
        monkeypatch.setattr(dsv4, "_DEEPSEEK_V4_DSPARK_TOPK_SHAPE_WARNED", False)
        monkeypatch.setattr(fast, "has_symbol", lambda name: True)

        def boom(*args):
            raise RuntimeError("metal command buffer failed")

        monkeypatch.setattr(fast, "dspark_fp32_topk_indices", boom)

        indexer = SimpleNamespace(index_topk=8, n_heads=2, scale=0.5)
        result = dsv4._batch_indexer_rows(
            indexer,
            [
                mx.zeros((1, 10, 4), dtype=mx.bfloat16),
                mx.zeros((1, 11, 4), dtype=mx.bfloat16),
            ],
            mx.zeros((1, 2, 2, 4), dtype=mx.bfloat16),
            mx.zeros((1, 2, 2), dtype=mx.bfloat16),
        )
        mx.eval(*result)
        assert [r.shape for r in result] == [(1, 1, 8), (1, 1, 8)]
        assert dsv4._DEEPSEEK_V4_DSPARK_TOPK_NATIVE_DISABLED is True
