# SPDX-License-Identifier: Apache-2.0
"""Tests for the DeepSeek V4 monkey-patch (PR 1192 port)."""

import importlib
import inspect
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
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

    @pytest.mark.parametrize("patch_order", ("scheduler-first", "deepseek-first"))
    def test_deepseek_patch_preserves_mixed_model_owned_conversion(
        self, patch_order
    ):
        if patch_order == "scheduler-first":
            patch_setup = (
                "import omlx.scheduler\n            apply_deepseek_v4_patch()"
            )
        else:
            patch_setup = (
                "apply_deepseek_v4_patch()\n            import omlx.scheduler"
            )
        script = textwrap.dedent(
            f"""
            import importlib

            from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch
            from omlx.patches.mlx_vlm_qwen4_exp_compat import (
                apply_mlx_vlm_qwen4_exp_compat_patch,
            )

            {patch_setup}
            apply_mlx_vlm_qwen4_exp_compat_patch()

            from mlx_lm.models.cache import BatchPoolingCache, CacheList, PoolingCache
            from mlx_vlm.models.qwen4_exp.language import BatchQSAKVCache, QSAKVCache

            class Model:
                layers = (object(),)

                def make_cache(self):
                    return [CacheList(PoolingCache(4), QSAKVCache())]

            generate = importlib.import_module("mlx_lm.generate")
            caches = generate._make_cache(Model(), [0], None)
            assert isinstance(caches[0].caches[0], BatchPoolingCache)
            assert isinstance(caches[0].caches[1], BatchQSAKVCache)
            """
        )
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr


class TestTokenizerPatch:
    """mlx_lm.tokenizer_utils.AutoTokenizer wrapped with deepseek_v4 fallback."""

    def test_autotokenizer_wrapped(self, applied_patch):
        import mlx_lm.tokenizer_utils as tu

        # Wrapped class still exposes from_pretrained.
        assert hasattr(tu.AutoTokenizer, "from_pretrained")
        # Class name preserved for any introspection.
        assert tu.AutoTokenizer.__name__ == "AutoTokenizer"

    @pytest.mark.parametrize(
        "preconfigured_parser",
        (False, True),
        ids=("injected-parser", "preconfigured-parser"),
    )
    def test_dsml_start_tokens_include_canonical_newline_context(
        self, tmp_path, preconfigured_parser
    ):
        """The direct mlx-lm server must enter its tool state for V4 DSML.

        DeepSeek's tokenizer can merge ``>\n`` into one token.  Caching an
        encoding of only the bare opening marker therefore does not match the
        canonical marker followed by the first invoke on the next line.
        """
        script = textwrap.dedent(
            """
            import json
            import sys
            from pathlib import Path
            from types import SimpleNamespace

            import mlx_lm.tokenizer_utils as tokenizer_utils

            model_path = Path(sys.argv[1])
            preconfigured_parser = sys.argv[2] == "true"
            (model_path / "config.json").write_text(
                json.dumps({"model_type": "deepseek_v4"})
            )

            start = "<｜DSML｜tool_calls>"
            end = "</｜DSML｜tool_calls>"

            class ContextSensitiveTokenizer:
                def encode(self, text, add_special_tokens=False):
                    assert add_special_tokens is False
                    if text == start:
                        return [10, 11]
                    if text == start + "\\n":
                        return [10, 12]
                    if text == end:
                        return [13, 14]
                    raise AssertionError(f"unexpected encoding request: {text!r}")

            class Wrapper(SimpleNamespace):
                eos_token_ids = (0,)
                has_thinking = False

                @property
                def has_tool_calling(self):
                    return self._tool_parser is not None

                @property
                def tool_call_start(self):
                    return self._tool_call_start

                @property
                def tool_call_end(self):
                    return self._tool_call_end

                @property
                def tool_call_start_tokens(self):
                    return self._tool_call_start_tokens

                @property
                def tool_call_end_tokens(self):
                    return self._tool_call_end_tokens

                def encode(self, text, add_special_tokens=False):
                    return self._tokenizer.encode(text, add_special_tokens)

                def convert_ids_to_tokens(self, token_id):
                    return str(token_id)

            parser = None
            if preconfigured_parser:
                from omlx.patches.deepseek_v4 import tool_parser_v4

                parser = tool_parser_v4.parse_tool_call

            wrapper = Wrapper(
                _tokenizer=ContextSensitiveTokenizer(),
                _chat_template=None,
                _tool_parser=parser,
                _tool_call_start=None,
                _tool_call_end=None,
                _tool_call_start_tokens=(10, 11),
                _tool_call_end_tokens=(13, 14),
            )
            tokenizer_utils.load = lambda *args, **kwargs: wrapper

            from omlx.patches.deepseek_v4.tokenizer_patch import apply_load_patch

            assert apply_load_patch() is True
            loaded = tokenizer_utils.load(model_path)
            assert loaded.tool_call_start == start
            assert loaded.tool_call_start_tokens == (10, 12)

            from mlx_lm.server import ResponseGenerator

            generator = object.__new__(ResponseGenerator)
            generator._state_machine_cache = {}
            machine, _ = generator._make_state_machine(
                "deepseek-v4-test", loaded, [], "normal"
            )

            state = machine.make_state()
            current = "normal"
            for token in (10, 12):
                state, _, current = machine.match(state, token)
            assert current == "tool"

            for token in (13, 14):
                state, _, current = machine.match(state, token)
            assert current == "normal"
            """
        )
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                script,
                str(tmp_path),
                str(preconfigured_parser).lower(),
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr

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

    def test_topk_wsdpa_dispatch_ignores_native_sparse_disable(
        self, applied_patch, monkeypatch
    ):
        import mlx.core as mx

        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]
        expected = mx.zeros((1, 64, 5, 512), dtype=mx.bfloat16)
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
            mx.zeros((1, 64, 5, 512), dtype=mx.bfloat16),
            mx.zeros((1, 1, 133, 512), dtype=mx.bfloat16),
            mx.zeros((1, 513, 512), dtype=mx.bfloat16),
            mx.zeros((1, 5, 512), dtype=mx.uint32),
            None,
            None,
            512**-0.5,
            mx.zeros((64,), dtype=mx.bfloat16),
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

    def test_vision_attention_is_not_forced_to_language_mxfp8(self, applied_patch):
        import mlx.nn as nn

        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]

        class _Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.wqkv = nn.Linear(8, 24)

        class _Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = _Attention()

        class _Vision(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = [_Block()]

        class _ModelStub(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision = _Vision()
                self.language = _Block()

        qcfg = dsv4.make_quantization_config(_ModelStub())
        assert qcfg["language.attn.wqkv"]["mode"] == "mxfp8"
        assert "vision.blocks.0.attn.wqkv" not in qcfg


class TestDeepSeekV4VisionSuffixes:
    def test_image_embedding_slices_cover_cache_offset_and_multiple_images(
        self, applied_patch
    ):
        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]
        first = SimpleNamespace(start=1, types=(0, 2, 4))
        second = SimpleNamespace(start=6, types=(0, 2, 2, 4))

        def normalized(offset, length):
            return [
                (
                    image.start,
                    destination.start,
                    destination.stop,
                    source.start,
                    source.stop,
                )
                for image, destination, source in dsv4._image_embedding_slices(
                    (first, second), offset, length
                )
            ]

        assert normalized(0, 1) == []
        assert normalized(2, 1) == [(1, 0, 1, 1, 2)]
        assert normalized(4, 2) == []
        assert normalized(3, 5) == [(1, 0, 1, 2, 3), (6, 3, 5, 0, 2)]
        assert normalized(10, 2) == []

    def test_cached_image_is_not_encoded_when_later_image_overlaps_suffix(
        self, applied_patch
    ):
        import mlx.core as mx

        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]

        class _Embedding:
            weight = mx.zeros((100, 2))

            def __call__(self, token_ids):
                return mx.broadcast_to(token_ids[..., None], (*token_ids.shape, 2))

        class _Vision:
            def __init__(self):
                self.calls = []

            def __call__(self, patches, _height, _width):
                marker = int(patches[0, 0, 0, 0].item())
                self.calls.append(marker)
                return mx.array([[marker]], dtype=mx.float32)

        class _Group:
            @staticmethod
            def size():
                return 1

        first = SimpleNamespace(
            start=0,
            types=(0, 2, 4),
            patches=[[[[1]], [[1]], [[1]]]],
            n_vit_h=1,
            n_vit_w=1,
            permutation=(0,),
        )
        second = SimpleNamespace(
            start=5,
            types=(0, 2, 4),
            patches=[[[[2]], [[2]], [[2]]]],
            n_vit_h=1,
            n_vit_w=1,
            permutation=(0,),
        )
        vision = _Vision()
        fake = SimpleNamespace(
            args=SimpleNamespace(vocab_size=100, hidden_size=2),
            model=SimpleNamespace(embed_tokens=_Embedding()),
            image_start=mx.full((2,), 10),
            image_pad=mx.full((2,), 20),
            image_newline=mx.full((2,), 30),
            image_end=mx.full((2,), 40),
            vision=vision,
            aligner=lambda features, _height, _width: mx.broadcast_to(
                features, (features.shape[0], 2)
            ),
            _vision_rank=0,
            _vision_group=_Group(),
            _vision_inputs=(first, second),
            _vision_blocks={},
        )

        embeddings = dsv4.Model._vision_embeddings(
            fake, mx.array([[9, 100, 102, 104, 8]]), offset=4
        )
        mx.eval(embeddings)

        assert vision.calls == [2]
        assert embeddings[0, 1:4].tolist() == [[10, 10], [2, 2], [40, 40]]
        assert fake._vision_inputs == ()

    def test_distributed_vision_embeddings_are_materialized_before_pipeline(
        self, applied_patch, monkeypatch
    ):
        import mlx.core as mx

        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]
        original_eval = mx.eval
        events = []

        class _Group:
            @staticmethod
            def size():
                return 2

        class _Embedding:
            weight = mx.zeros((100, 2), dtype=mx.bfloat16)

        def all_sum(value, *, group):
            assert isinstance(group, _Group)
            kind = "status" if value.dtype == mx.int32 else "embeddings"
            events.append(f"all_sum_{kind}")
            if kind == "status":
                return mx.array(1, dtype=mx.int32)
            return mx.ones_like(value)

        def tracked_eval(value):
            kind = "status" if value.dtype == mx.int32 else "embeddings"
            events.append(f"eval_{kind}")
            return original_eval(value)

        monkeypatch.setattr(mx.distributed, "all_sum", all_sum)
        monkeypatch.setattr(mx, "eval", tracked_eval)
        fake = SimpleNamespace(
            args=SimpleNamespace(vocab_size=100, hidden_size=2),
            model=SimpleNamespace(embed_tokens=_Embedding()),
            _vision_rank=1,
            _vision_group=_Group(),
            _vision_inputs=None,
            _vision_blocks={},
        )

        embeddings = dsv4.Model._vision_embeddings(fake, mx.array([[100]]))

        assert embeddings.tolist() == [[[1, 1]]]
        assert events == [
            "all_sum_status",
            "eval_status",
            "all_sum_embeddings",
            "eval_embeddings",
        ]

    def test_cached_visibility_mask_disables_standard_kernel_path(
        self, applied_patch
    ):
        import mlx.core as mx

        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]

        class _Embedding:
            def __call__(self, inputs):
                return mx.zeros((*inputs.shape, 2))

        class _Cache:
            offset = 3

            @staticmethod
            def make_mask(length, **_kwargs):
                rows = mx.arange(3, 3 + length)[:, None]
                columns = mx.arange(3 + length)[None]
                return rows >= columns

        class _Layer:
            def __call__(self, h, mask, _cache, _inputs, *, _standard_mask):
                self.mask = mask
                self.standard = _standard_mask
                return h

        layer = _Layer()
        fake = SimpleNamespace(
            args=SimpleNamespace(hc_mult=1, sliding_window=128),
            vocab_size=100,
            embed_tokens=_Embedding(),
            pipeline_rank=0,
            pipeline_size=1,
            pipeline_layers=[layer],
            hc_head=lambda h: h,
            norm=lambda h: h,
        )

        dsv4.DeepseekV4Model.__call__(
            fake,
            mx.array([[102, 103, 104]]),
            cache=[_Cache()],
            inputs_embeds=mx.zeros((1, 3, 2)),
        )

        assert layer.standard is False
        assert layer.mask.shape == (3, 6)
        assert bool(mx.all(layer.mask[:, 3:6]).item())

        dsv4.DeepseekV4Model.__call__(
            fake,
            mx.array([[1, 2, 3]]),
            cache=[_Cache()],
            inputs_embeds=mx.zeros((1, 3, 2)),
        )
        assert layer.standard is True

    def test_mtp_disabled_outer_model_preserves_vision_embedding_path(
        self, applied_patch
    ):
        import mlx.core as mx

        dsv4 = sys.modules["mlx_lm.models.deepseek_v4"]
        expected = mx.full((1, 3, 2), 7)

        class _Inner:
            def __call__(self, inputs, cache, *, inputs_embeds=None):
                self.inputs = inputs
                self.cache = cache
                self.inputs_embeds = inputs_embeds
                return inputs_embeds

        class _Cache:
            offset = 5

        inner = _Inner()
        seen = {}

        def vision_embeddings(inputs, offset):
            seen.update(inputs=inputs, offset=offset)
            return expected

        fake = SimpleNamespace(
            args=SimpleNamespace(vocab_size=100),
            is_vision_model=True,
            _omlx_mtp_decode_enabled=False,
            _vision_embeddings=vision_embeddings,
            model=inner,
            lm_head=lambda hidden: hidden,
        )
        inputs = mx.array([[102, 103, 104]])
        cache = [_Cache()]

        output = dsv4.Model.__call__(fake, inputs, cache)

        assert output is expected
        assert seen == {"inputs": inputs, "offset": 5}
        assert inner.inputs_embeds is expected


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
