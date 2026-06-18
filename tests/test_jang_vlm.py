# SPDX-License-Identifier: Apache-2.0
"""Tests for JANG-specific discovery and live JANGLoader behavior.

All tests in this file require mlx (Apple Silicon only).  The file is
skipped cleanly on Linux via pytest.importorskip at module level.

mlx-free tests (resolve_jang_engine_type helper, engine-type ordering,
clobber guards) live in tests/test_jang_engine_type.py and run on Linux.
"""

import pytest

# Skip the entire file on platforms without mlx (e.g. Linux CI).
# This must come before any mlx-dependent imports so collection succeeds.
pytest.importorskip("mlx")

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from omlx.engine.jang import JANGLoader
from omlx.model_discovery import detect_model_type, discover_models


class _FakeArray:
    """Small array-like test double for patched mlx calls."""

    def __init__(self, shape):
        self.shape = shape

    def astype(self, _dtype):
        return self


class TestDetectModelTypeJangVlm:
    """Tests for production JANG VLM detection in model_discovery."""

    def test_detect_vlm_via_jang_has_vision(self, tmp_path):
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"version": "1.0", "architecture": {"has_vision": True}})
        )
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "jang_vlm"}))

        assert detect_model_type(tmp_path) == "vlm"

    def test_detect_text_only_jang_as_llm(self, tmp_path):
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"version": "1.0", "architecture": {"has_vision": False}})
        )
        (tmp_path / "preprocessor_config.json").write_text(
            json.dumps({"processor_type": "AutoProcessor"})
        )
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))

        assert detect_model_type(tmp_path) == "llm"

    def test_detect_vlm_via_preprocessor_only_for_jang(self, tmp_path):
        (tmp_path / "jjqf_config.json").write_text(json.dumps({"quantization": {"bits": 2}}))
        (tmp_path / "preprocessor_config.json").write_text(
            json.dumps({"processor_type": "AutoProcessor"})
        )
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))

        assert detect_model_type(tmp_path) == "vlm"

    def test_discover_jang_vlm_uses_jang_engine(self, tmp_path):
        model_dir = tmp_path / "qwen3.5-vlm-jang"
        model_dir.mkdir()
        (model_dir / "jang_config.json").write_text(
            json.dumps(
                {
                    "format": "jang",
                    "format_version": "2.0",
                    "architecture": {"has_vision": True},
                }
            )
        )
        (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))
        (model_dir / "model.safetensors").write_bytes(b"0" * 1000)

        models = discover_models(tmp_path)
        assert models["qwen3.5-vlm-jang"].model_type == "vlm"
        assert models["qwen3.5-vlm-jang"].engine_type == "jang"


class TestJANGLoader:
    """Tests for JANGLoader behavior that is exercised in production."""

    def test_is_jang_v2_reads_format_version(self, tmp_path):
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"format": "jang", "format_version": "2.0"})
        )

        loader = JANGLoader(str(tmp_path))

        assert loader._is_jang_v2() is True

    def test_should_use_vlm_loader_when_discovery_detects_vlm(self, tmp_path):
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"format": "jang", "format_version": "2.0", "architecture": {}})
        )
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "mistral3", "vision_config": {"hidden_size": 1024}})
        )

        loader = JANGLoader(str(tmp_path))

        assert loader._should_use_vlm_loader() is True

    def test_should_not_use_vlm_loader_when_jang_explicitly_says_no_vision(self, tmp_path):
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"format": "jang", "format_version": "2.0", "architecture": {"has_vision": False}})
        )
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "mistral3", "vision_config": {"hidden_size": 1024}})
        )

        loader = JANGLoader(str(tmp_path))

        assert loader._should_use_vlm_loader() is False

    @pytest.mark.asyncio
    async def test_start_skips_nemotron_fixup_for_jang_v2(self, tmp_path):
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"format": "jang", "format_version": "2.0"})
        )
        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "nemotron_h",
                    "architectures": ["NemotronHForCausalLM"],
                    "hidden_size": 4096,
                    "num_attention_heads": 32,
                    "head_dim": 128,
                    "vocab_size": 131072,
                }
            )
        )

        loader = JANGLoader(str(tmp_path))
        fake_model = MagicMock()
        fake_model.config = {"architectures": ["NemotronHForCausalLM"]}
        fake_engine = MagicMock()
        fake_engine.engine.start = AsyncMock()

        import sys
        fake_jang_tools = MagicMock()
        fake_jang_tools.loader.load_jang_model = MagicMock(
            return_value=(fake_model, MagicMock())
        )
        fake_jang_tools.loader.load_jang_vlm_model = MagicMock()

        with patch.object(loader, "_check_jang_tools_available"), \
             patch.object(loader, "_should_use_vlm_loader", return_value=False), \
             patch.dict(sys.modules, {
                 "jang_tools": fake_jang_tools,
                 "jang_tools.loader": fake_jang_tools.loader,
             }), \
             patch.object(loader, "_fix_nemotron_h_weights") as fixup, \
             patch.object(loader, "_needs_bfloat16", return_value=False), \
             patch.object(loader, "_detect_nemotron_h", return_value=True), \
             patch("omlx.engine_core.AsyncEngineCore", return_value=fake_engine), \
             patch("omlx.engine_core.EngineConfig",
                   side_effect=lambda **kwargs: SimpleNamespace(**kwargs)), \
             patch("omlx.scheduler.SchedulerConfig",
                   return_value=SimpleNamespace(model_name=None)):
            await loader.start()

        fixup.assert_not_called()

    def test_fix_nemotron_h_weights_no_index_is_noop(self, tmp_path):
        loader = JANGLoader(str(tmp_path))
        loader._model = MagicMock()

        loader._fix_nemotron_h_weights()

        loader._model.load_weights.assert_not_called()

    def test_fix_nemotron_h_weights_dequantizes_gate_weights(self, tmp_path):
        loader = JANGLoader(str(tmp_path))
        loader._model = MagicMock()

        index = {
            "weight_map": {
                "backbone.layers.0.mixer.gate.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.0.mixer.gate.scales": "model-00001-of-00001.safetensors",
                "backbone.layers.0.mixer.gate.biases": "model-00001-of-00001.safetensors",
            }
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

        gate_weight = _FakeArray((2, 32))
        scales = _FakeArray((2, 1))
        biases = _FakeArray((2, 1))
        dequantized = _FakeArray((2, 128))

        with patch.object(loader, "_needs_bfloat16", return_value=False), patch(
            "omlx.engine.jang.mx.load",
            return_value={
                "backbone.layers.0.mixer.gate.weight": gate_weight,
                "backbone.layers.0.mixer.gate.scales": scales,
                "backbone.layers.0.mixer.gate.biases": biases,
            },
        ) as mock_load, patch(
            "omlx.engine.jang.mx.dequantize", return_value=dequantized
        ) as mock_dequantize, patch(
            "omlx.engine.jang.mx.float16", "float16"
        ):
            loader._fix_nemotron_h_weights()

        mock_load.assert_called_once()
        mock_dequantize.assert_called_once_with(gate_weight, scales, biases, 128, 8)
        loader._model.load_weights.assert_called_once_with(
            [("backbone.layers.0.mixer.gate.weight", dequantized)],
            strict=False,
        )

    @pytest.mark.asyncio
    async def test_generate_uses_engine_and_cleans_output(self):
        loader = JANGLoader("/tmp/jang-model")
        loader._loaded = True
        loader._engine = MagicMock()
        loader._engine.generate = AsyncMock(
            return_value=SimpleNamespace(
                output_text="hello<|end|>",
                prompt_tokens=11,
                completion_tokens=7,
                finish_reason="stop",
                tool_calls=[],
                cached_tokens=3,
            )
        )

        with patch("omlx.engine.jang.clean_special_tokens", return_value="hello") as clean:
            output = await loader.generate("prompt", max_tokens=9, temperature=0.2)

        clean.assert_called_once_with("hello<|end|>")
        assert output.text == "hello"
        assert output.prompt_tokens == 11
        assert output.completion_tokens == 7
        sampling_params = loader._engine.generate.await_args.kwargs["sampling_params"]
        assert sampling_params.max_tokens == 9
        assert sampling_params.temperature == 0.2


# ---------------------------------------------------------------------------
# Helpers for _fix_jang_switch_quantization tests
# ---------------------------------------------------------------------------

class _FakeScales:
    """Minimal scales array double."""
    def __init__(self, num_groups: int):
        self.shape = (4, num_groups)  # (num_experts, scale_groups)


class _FakeSwitchLinear:
    """
    Test double for QuantizedSwitchLinear.

    Constructed from the per-layer shape parameters used in
    _fix_jang_switch_quantization so tests can control every relevant attribute.

    Args:
        num_experts: First dim of the packed weight tensor.
        out_dim: Second dim (rows of unpacked weight); must equal intermediate_size
                 or hidden_size to be classifiable.
        packed_dim: Third dim of the packed weight tensor (in_dim * bits / 32).
        bits: Initial (wrong) bits metadata from config.json.
        group_size: Initial (wrong) group_size metadata.
        scale_groups: Number of groups in the scales tensor.
    """
    def __init__(
        self,
        num_experts: int,
        out_dim: int,
        packed_dim: int,
        bits: int,
        group_size: int,
        scale_groups: int,
    ):
        self.weight = _FakeArray((num_experts, out_dim, packed_dim))
        self.scales = _FakeScales(scale_groups)
        self.bits = bits
        self.group_size = group_size


def _make_loader_with_fake_model(
    tmp_path,
    config: dict,
    modules: list[tuple[str, Any]],
) -> JANGLoader:
    """
    Return a JANGLoader whose _model exposes named_modules() returning `modules`.
    `config` is written to tmp_path/config.json so the method can read it.
    """
    (tmp_path / "config.json").write_text(json.dumps(config))
    loader = JANGLoader(str(tmp_path))
    fake_model = MagicMock()
    fake_model.named_modules.return_value = modules
    loader._model = fake_model
    return loader


# ---------------------------------------------------------------------------
# Tests for _fix_jang_switch_quantization
# ---------------------------------------------------------------------------

class TestFixJangSwitchQuantization:
    """
    NOTE: These tests import mlx_lm (via the method under test) and require
    Apple Silicon + mlx. They are written-but-unexecuted on this Linux box
    and must be run on helium. Marked with the jang_mlx marker so CI can
    skip them on non-Apple hosts.
    """

    pytestmark = pytest.mark.jang_mlx

    def test_correct_bits_and_group_size_for_gate_proj(self, tmp_path):
        """
        A gate_proj layer (out_dim == intermediate_size, in_dim == hidden_size)
        with wrong bits=8 in metadata but 2-bit packed weight gets bits corrected
        to 2 and group_size corrected from the scales shape.

        Layout: hidden_size=256, intermediate_size=512, 2-bit packing.
          packed_dim = hidden_size * bits / 32 = 256 * 2 / 32 = 16
          scale_groups = hidden_size / group_size = 256 / 64 = 4
        """
        config = {
            "hidden_size": 256,
            "intermediate_size": 512,
            "quantization": {"bits": 8},
        }
        layer = _FakeSwitchLinear(
            num_experts=4,
            out_dim=512,    # == intermediate_size → gate_proj
            packed_dim=16,  # == hidden_size * 2 / 32 → 2-bit
            bits=8,
            group_size=64,
            scale_groups=4, # hidden_size(256) / 64 = 4
        )

        loader = _make_loader_with_fake_model(tmp_path, config, [("model.layer.0", layer)])

        from mlx_lm.models.switch_layers import QuantizedSwitchLinear  # noqa
        with patch("mlx_lm.models.switch_layers.QuantizedSwitchLinear", type(layer)):
            loader._fix_jang_switch_quantization()

        assert layer.bits == 2
        assert layer.group_size == 64   # 256 / 4

    def test_correct_bits_for_down_proj(self, tmp_path):
        """
        A down_proj layer (out_dim == hidden_size, in_dim == intermediate_size)
        with 4-bit packed weight gets bits corrected from 8 to 4.

        Layout: hidden_size=256, intermediate_size=512, 4-bit packing.
          packed_dim = intermediate_size * 4 / 32 = 512 * 4 / 32 = 64
        """
        config = {
            "hidden_size": 256,
            "intermediate_size": 512,
            "quantization": {"bits": 8},
        }
        layer = _FakeSwitchLinear(
            num_experts=4,
            out_dim=256,    # == hidden_size → down_proj
            packed_dim=64,  # == intermediate_size * 4 / 32 → 4-bit
            bits=8,
            group_size=64,
            scale_groups=8, # intermediate_size(512) / 64 = 8
        )

        loader = _make_loader_with_fake_model(tmp_path, config, [("model.layer.0", layer)])

        from mlx_lm.models.switch_layers import QuantizedSwitchLinear  # noqa
        with patch("mlx_lm.models.switch_layers.QuantizedSwitchLinear", type(layer)):
            loader._fix_jang_switch_quantization()

        assert layer.bits == 4
        assert layer.group_size == 64   # 512 / 8

    def test_unclassifiable_out_dim_raises(self, tmp_path):
        """
        A layer whose out_dim matches neither intermediate_size nor hidden_size
        must raise ValueError, not silently continue.

        Sabotage check: if the raise were replaced with `continue`, this test
        would see bits unchanged at 8 and NOT raise — it would silently pass
        without the ValueError, proving the guard is real.
        """
        config = {
            "hidden_size": 256,
            "intermediate_size": 512,
        }
        layer = _FakeSwitchLinear(
            num_experts=4,
            out_dim=999,    # matches neither
            packed_dim=16,
            bits=8,
            group_size=64,
            scale_groups=4,
        )

        loader = _make_loader_with_fake_model(tmp_path, config, [("model.layer.0", layer)])

        from mlx_lm.models.switch_layers import QuantizedSwitchLinear  # noqa
        with patch("mlx_lm.models.switch_layers.QuantizedSwitchLinear", type(layer)):
            with pytest.raises(ValueError, match="unclassifiable"):
                loader._fix_jang_switch_quantization()

    def test_non_exact_packed_dim_raises(self, tmp_path):
        """
        A packed_dim that is inconsistent with in_dim for all supported bit-widths
        (2/3/4/6/8) means a corrupt or mismatched shard.  Must raise ValueError, not
        silently write a wrong bits value.

        packed_dim=17, in_dim=256: for bits=2 recovered=272, range [256,272) → 272 is
        not strictly less than 272, so no candidate matches → raises.

        Sabotage check: removing the trial-loop and the final None-raise would let the
        floor-division path write a wrong bits value and silently accept corrupt metadata;
        the pytest.raises assertion would then fail.
        """
        config = {
            "hidden_size": 256,
            "intermediate_size": 512,
        }
        # packed_dim=17, in_dim=256: inconsistent with every supported bit-width
        layer = _FakeSwitchLinear(
            num_experts=4,
            out_dim=512,
            packed_dim=17,
            bits=8,
            group_size=64,
            scale_groups=4,
        )

        loader = _make_loader_with_fake_model(tmp_path, config, [("model.layer.0", layer)])

        from mlx_lm.models.switch_layers import QuantizedSwitchLinear  # noqa
        with patch("mlx_lm.models.switch_layers.QuantizedSwitchLinear", type(layer)):
            with pytest.raises(ValueError, match="not consistent with in_dim"):
                loader._fix_jang_switch_quantization()

    @pytest.mark.jang_mlx
    def test_group_size_corrected_when_differs_from_initial(self, tmp_path):
        """
        When the actual group_size (derived from scales/in_dim) differs from the
        initially-declared group_size, _fix_jang_switch_quantization must update
        module.group_size to the corrected value.

        Layout: hidden_size=256, intermediate_size=512, 2-bit gate_proj.
          in_dim = hidden_size = 256
          packed_dim = 256 * 2 / 32 = 16
          scale_groups = 4  → actual_gs = 256 // 4 = 64
          initial group_size = 128  ← intentionally wrong to make the mutation visible

        Guards jang.py line `module.group_size = actual_gs` (~line 491).
        Sabotage: deleting that line leaves group_size=128 and the final
        `assert layer.group_size == 64` fails, confirming the test is non-vacuous.
        """
        config = {
            "hidden_size": 256,
            "intermediate_size": 512,
            "quantization": {"bits": 8},
        }
        layer = _FakeSwitchLinear(
            num_experts=4,
            out_dim=512,     # == intermediate_size → gate_proj, in_dim = hidden_size = 256
            packed_dim=16,   # == 256 * 2 / 32 → 2-bit
            bits=8,
            group_size=128,  # wrong initial value — must be corrected to 64
            scale_groups=4,  # in_dim(256) / 4 = 64 → actual_gs = 64
        )

        loader = _make_loader_with_fake_model(tmp_path, config, [("model.layer.0", layer)])

        from mlx_lm.models.switch_layers import QuantizedSwitchLinear  # noqa
        with patch("mlx_lm.models.switch_layers.QuantizedSwitchLinear", type(layer)):
            loader._fix_jang_switch_quantization()

        assert layer.bits == 2
        assert layer.group_size == 64  # corrected from 128 → 256 // 4

    def test_non_exact_scale_groups_raises(self, tmp_path):
        """
        in_dim not divisible by scale_groups means an inconsistent scale tensor.
        Must raise ValueError.
        """
        config = {
            "hidden_size": 256,
            "intermediate_size": 512,
        }
        # 2-bit, gate_proj: in_dim=256, scale_groups=7 → 256 % 7 != 0
        layer = _FakeSwitchLinear(
            num_experts=4,
            out_dim=512,
            packed_dim=16,
            bits=8,
            group_size=64,
            scale_groups=7,  # 256 % 7 != 0
        )

        loader = _make_loader_with_fake_model(tmp_path, config, [("model.layer.0", layer)])

        from mlx_lm.models.switch_layers import QuantizedSwitchLinear  # noqa
        with patch("mlx_lm.models.switch_layers.QuantizedSwitchLinear", type(layer)):
            with pytest.raises(ValueError, match="not divisible"):
                loader._fix_jang_switch_quantization()

    @pytest.mark.jang_mlx
    @pytest.mark.asyncio
    async def test_failed_quant_fix_propagates_through_start(self, tmp_path):
        """
        If _fix_jang_switch_quantization raises, start() must NOT catch and warn;
        it must wrap in JANGLoadError and re-raise.

        This is the BLOCKER fix — previously the exception was swallowed with
        logger.warning(), leaving the model loaded with wrong quantization metadata.

        Sabotage check: reverting the BLOCKER fix (putting the call back inside
        try/except) would cause this test to NOT raise JANGLoadError, which would
        fail the pytest.raises assertion.  Additionally, the cause chain check
        (ei.value.__cause__) confirms the original ValueError is preserved.

        Guards: jang.py start() line that calls _fix_jang_switch_quantization()
        outside the except-block, and the `raise JANGLoadError(...) from e` at the
        outer except handler.
        """
        (tmp_path / "jang_config.json").write_text(json.dumps({"format_version": "1.0"}))
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "test_moe",
            "hidden_size": 256,
            "intermediate_size": 512,
        }))

        loader = JANGLoader(str(tmp_path))
        fake_model = MagicMock()
        fake_engine = MagicMock()
        fake_engine.engine.start = AsyncMock()

        import sys
        # Inject a minimal jang_tools stub so the `import jang_tools` inside
        # start() succeeds without the real package installed.
        fake_jang_tools = MagicMock()
        fake_jang_tools.loader.load_jang_model = MagicMock(
            return_value=(fake_model, MagicMock())
        )
        fake_jang_tools.loader.load_jang_vlm_model = MagicMock()

        from omlx.exceptions import JANGLoadError
        with patch.object(loader, "_check_jang_tools_available"), \
             patch.object(loader, "_should_use_vlm_loader", return_value=False), \
             patch.dict(sys.modules, {
                 "jang_tools": fake_jang_tools,
                 "jang_tools.loader": fake_jang_tools.loader,
             }), \
             patch.object(loader, "_fix_jang_switch_quantization",
                          side_effect=ValueError("corrupt shard")), \
             patch.object(loader, "_needs_bfloat16", return_value=False), \
             patch.object(loader, "_detect_nemotron_h", return_value=False), \
             patch("omlx.engine_core.AsyncEngineCore", return_value=fake_engine), \
             patch("omlx.engine_core.EngineConfig",
                   side_effect=lambda **kw: SimpleNamespace(**kw)), \
             patch("omlx.scheduler.SchedulerConfig",
                   return_value=SimpleNamespace(model_name=None)), \
             patch("omlx.engine.jang.mx.clear_cache"):
            with pytest.raises(JANGLoadError, match="corrupt shard") as ei:
                await loader.start()

        assert isinstance(ei.value.__cause__, ValueError)
        assert "corrupt shard" in str(ei.value.__cause__)


# ---------------------------------------------------------------------------
# Tests for _is_jangtq detection
# ---------------------------------------------------------------------------


class TestIsJangtq:
    """
    Tests for the _is_jangtq() detection method on JANGLoader.

    Marked jang_mlx because JANGLoader can only be imported on Apple Silicon
    (the module-level `import mlx.core` at the top of jang.py).  Run on
    helium; skipped by default on Linux CI via `not jang_mlx`.

    Sabotage mapping (per test):
      - test_detects_mxtq_weight_format: removing the `weight_format == "mxtq"`
        check returns False → the JANGTQ branch in start() is never entered →
        load_jangtq_model is never called → Metal OOM on TurboQuant models.
      - test_detects_jangtq_profile: removing the `"JANGTQ" in profile` check
        returns False for profile-only JANGTQ markers (e.g. "JANGTQ2") →
        same Metal OOM regression.
      - test_does_not_detect_standard_jang: if the method always returned True,
        standard JANG_2L models would be sent to load_jangtq_model (wrong loader).
      - test_returns_false_when_no_config: if always True, no-config dirs would
        also be misrouted.
    """

    pytestmark = pytest.mark.jang_mlx

    def test_detects_mxtq_weight_format(self, tmp_path):
        """weight_format == "mxtq" in jang_config.json → True."""
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"weight_format": "mxtq", "profile": "JANG_2L"})
        )
        loader = JANGLoader(str(tmp_path))
        assert loader._is_jangtq() is True

    def test_detects_jangtq_profile(self, tmp_path):
        """profile containing "JANGTQ" (e.g. "JANGTQ2") → True."""
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"weight_format": "mx.quantize", "profile": "JANGTQ2"})
        )
        loader = JANGLoader(str(tmp_path))
        assert loader._is_jangtq() is True

    def test_does_not_detect_standard_jang(self, tmp_path):
        """mx.quantize + non-JANGTQ profile → False (standard JANG_2L model)."""
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"weight_format": "mx.quantize", "profile": "JANG_2L"})
        )
        loader = JANGLoader(str(tmp_path))
        assert loader._is_jangtq() is False

    def test_returns_false_when_no_config(self, tmp_path):
        """No JANG config file in dir → False."""
        loader = JANGLoader(str(tmp_path))
        assert loader._is_jangtq() is False


# ---------------------------------------------------------------------------
# Tests for JANGTQ dispatch in start()
# ---------------------------------------------------------------------------


class TestStartJangtqDispatch:
    """
    Tests that start() routes JANGTQ models to load_jangtq_model and bypasses
    the standard load_jang_model path and the switch-quantization fix.

    Also tests the F1 guard: JANGTQ + VLM raises JANGLoadError immediately
    rather than silently loading the model with the vision encoder dropped.

    Marked jang_mlx because JANGLoader imports mlx at module level.
    Run on helium; skipped on Linux CI.

    Mirrors the mocking style of test_start_skips_nemotron_fixup_for_jang_v2.

    Sabotage mapping (per test):
      - test_start_dispatches_jangtq_to_load_jangtq_model:
          * Removing the `if is_jangtq:` branch → load_jangtq_model.assert_called_once
            fails (not called) AND load_jang_model.assert_not_called fails (called).
          * Swapping `is_jangtq` for `is_vlm` in the condition → same failure for
            a non-VLM JANGTQ model.
      - test_start_skips_switch_quant_fix_for_jangtq:
          * Removing the `if not is_jangtq:` guard → fix_mock.assert_not_called fails.
          * This guards against the Metal OOM path: _fix_jang_switch_quantization
            imports mlx_lm.models.switch_layers and walks the model; TurboQuantLinear
            layers have no `bits`/`group_size` and would raise AttributeError.
      - test_start_raises_on_jangtq_vlm_combination:
          * Removing the `if is_jangtq and is_vlm:` guard (or changing it to only
            check one condition) means the test no longer raises JANGLoadError →
            the pytest.raises assertion fails, and load_jangtq_model would be called
            (the loader_mock.assert_not_called() check would also catch this).
    """

    pytestmark = [pytest.mark.jang_mlx, pytest.mark.asyncio]

    async def test_start_raises_on_jangtq_vlm_combination(self, tmp_path):
        """
        When _is_jangtq() and _should_use_vlm_loader() are BOTH True, start()
        must raise JANGLoadError with "VLM" in the message before calling any
        loader function.

        This guards jang.py: the `if is_jangtq and is_vlm: raise JANGLoadError`
        guard added for F1.

        Sabotage: removing the guard (or weakening it to check only is_jangtq OR
        is_vlm) causes start() to proceed past the check; for is_jangtq=True it
        would attempt load_jangtq_model which silently drops the vision encoder.
        The pytest.raises assertion fails immediately, surfacing the regression.
        """
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"weight_format": "mxtq", "architecture": {"has_vision": True}})
        )
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "qwen3_vl", "hidden_size": 4096, "vocab_size": 131072})
        )

        loader = JANGLoader(str(tmp_path))

        import sys
        fake_jang_tools = MagicMock()
        fake_load_jangtq_mod = MagicMock()

        from omlx.exceptions import JANGLoadError
        with patch.object(loader, "_check_jang_tools_available"), \
             patch.object(loader, "_is_jangtq", return_value=True), \
             patch.object(loader, "_should_use_vlm_loader", return_value=True), \
             patch.dict(sys.modules, {
                 "jang_tools": fake_jang_tools,
                 "jang_tools.loader": fake_jang_tools.loader,
                 "jang_tools.load_jangtq": fake_load_jangtq_mod,
             }):
            with pytest.raises(JANGLoadError, match="VLM") as exc_info:
                await loader.start()

        # Neither loader path should have been called
        fake_load_jangtq_mod.load_jangtq_model.assert_not_called()
        fake_jang_tools.loader.load_jang_model.assert_not_called()
        fake_jang_tools.loader.load_jang_vlm_model.assert_not_called()
        # Error message must explain the issue
        assert "vision" in str(exc_info.value).lower() or "vlm" in str(exc_info.value).lower()

    async def test_start_dispatches_jangtq_to_load_jangtq_model(self, tmp_path):
        """
        When _is_jangtq() returns True, start() must call load_jangtq_model
        and must NOT call load_jang_model.
        """
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"weight_format": "mxtq"})
        )
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "gpt_oss", "hidden_size": 4096, "vocab_size": 131072})
        )

        loader = JANGLoader(str(tmp_path))
        fake_model = MagicMock()
        fake_tok = MagicMock()
        fake_tok.encode = MagicMock(return_value=[1, 2, 3])
        fake_engine = MagicMock()
        fake_engine.engine.start = AsyncMock()

        import sys
        fake_jang_tools = MagicMock()
        fake_jang_tools.loader.load_jang_model = MagicMock()
        fake_jang_tools.loader.load_jang_vlm_model = MagicMock()
        fake_load_jangtq = MagicMock(return_value=(fake_model, fake_tok))
        fake_load_jangtq_mod = MagicMock()
        fake_load_jangtq_mod.load_jangtq_model = fake_load_jangtq

        with patch.object(loader, "_check_jang_tools_available"), \
             patch.object(loader, "_is_jangtq", return_value=True), \
             patch.object(loader, "_should_use_vlm_loader", return_value=False), \
             patch.dict(sys.modules, {
                 "jang_tools": fake_jang_tools,
                 "jang_tools.loader": fake_jang_tools.loader,
                 "jang_tools.load_jangtq": fake_load_jangtq_mod,
             }), \
             patch.object(loader, "_fix_jang_switch_quantization"), \
             patch.object(loader, "_needs_bfloat16", return_value=False), \
             patch.object(loader, "_detect_nemotron_h", return_value=False), \
             patch("omlx.engine_core.AsyncEngineCore", return_value=fake_engine), \
             patch("omlx.engine_core.EngineConfig",
                   side_effect=lambda **kw: SimpleNamespace(**kw)), \
             patch("omlx.scheduler.SchedulerConfig",
                   return_value=SimpleNamespace(model_name=None)):
            await loader.start()

        fake_load_jangtq.assert_called_once_with(str(tmp_path))
        fake_jang_tools.loader.load_jang_model.assert_not_called()
        fake_jang_tools.loader.load_jang_vlm_model.assert_not_called()

    async def test_start_skips_switch_quant_fix_for_jangtq(self, tmp_path):
        """
        When _is_jangtq() returns True, _fix_jang_switch_quantization must
        NOT be called — TurboQuantLinear has no QuantizedSwitchLinear metadata.
        """
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"weight_format": "mxtq"})
        )
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "gpt_oss", "hidden_size": 4096, "vocab_size": 131072})
        )

        loader = JANGLoader(str(tmp_path))
        fake_model = MagicMock()
        fake_tok = MagicMock()
        fake_tok.encode = MagicMock(return_value=[1, 2, 3])
        fake_engine = MagicMock()
        fake_engine.engine.start = AsyncMock()

        import sys
        fake_jang_tools = MagicMock()
        fake_jang_tools.loader.load_jang_model = MagicMock()
        fake_jang_tools.loader.load_jang_vlm_model = MagicMock()
        fake_load_jangtq = MagicMock(return_value=(fake_model, fake_tok))
        fake_load_jangtq_mod = MagicMock()
        fake_load_jangtq_mod.load_jangtq_model = fake_load_jangtq

        with patch.object(loader, "_check_jang_tools_available"), \
             patch.object(loader, "_is_jangtq", return_value=True), \
             patch.object(loader, "_should_use_vlm_loader", return_value=False), \
             patch.dict(sys.modules, {
                 "jang_tools": fake_jang_tools,
                 "jang_tools.loader": fake_jang_tools.loader,
                 "jang_tools.load_jangtq": fake_load_jangtq_mod,
             }), \
             patch.object(loader, "_fix_jang_switch_quantization") as fix_mock, \
             patch.object(loader, "_needs_bfloat16", return_value=False), \
             patch.object(loader, "_detect_nemotron_h", return_value=False), \
             patch("omlx.engine_core.AsyncEngineCore", return_value=fake_engine), \
             patch("omlx.engine_core.EngineConfig",
                   side_effect=lambda **kw: SimpleNamespace(**kw)), \
             patch("omlx.scheduler.SchedulerConfig",
                   return_value=SimpleNamespace(model_name=None)):
            await loader.start()

        fix_mock.assert_not_called()
