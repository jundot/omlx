# SPDX-License-Identifier: Apache-2.0
"""LoRA parity on the decision path: a forward with ``apply_adapter``'s
wrappers equals a merged-weight forward.

Runs through ``DecisionBackbone.prefill``, the forward autotrust JEV decisions
use, on tiny Qwen3.5 models: float and 4-bit bases, mlx-lm and mlx-vlm model
trees, oMLX's quantized prefill patches active, and packed leaves.
"""

import importlib.util
import sys
from collections import Counter
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from omlx.lora_adapters import LoRALinear, apply_adapter
from omlx.models.decision import DecisionBackbone

# Every target projection is a multiple of 64 wide, so the quantized prefill
# patches can route all of them.
_TEXT_CONFIG = {
    "model_type": "qwen3_5_text",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 320,
    "linear_num_value_heads": 2,
    "linear_num_key_heads": 2,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 32,
    "linear_conv_kernel_dim": 3,
    "full_attention_interval": 2,
    "tie_word_embeddings": False,
    "rms_norm_eps": 1e-5,
    "head_dim": 32,
    "max_position_embeddings": 4096,
    "rope_parameters": {
        "rope_theta": 1000.0,
        "partial_rotary_factor": 0.5,
        "mrope_section": [2, 3, 3],
        "mrope_interleaved": True,
        "rope_type": "default",
    },
}
_VISION_CONFIG = {
    "model_type": "qwen3_5",
    "depth": 1,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_heads": 2,
    "out_hidden_size": 64,
    "patch_size": 4,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "in_channels": 3,
    "num_position_embeddings": 64,
    "deepstack_visual_indexes": [],
}
_TARGETS = {
    "linear_attn": ("in_proj_qkv", "in_proj_z", "out_proj"),
    "self_attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
}
_PEFT_CONFIG = {"peft_type": "LORA", "r": 4, "lora_alpha": 8, "bias": "none"}
_SCALE = 8 / 4
# Relative error of the last hidden state. Float runs differ by ~1e-6 (TF32
# is off in tests/conftest.py). A 4-bit model with bf16 activations differs
# from its float32 reference by 1.6-2.6e-2 without any adapter, and by
# 1.9-2.0e-2 with the wrapped adapter; losing the delta costs more than 1.0.
_FLOAT_TOLERANCE = 1e-5
_QUANTIZED_TOLERANCE = 5e-2
_IDS = np.arange(5, 45, dtype=np.int32)  # 40 tokens: above the lowered floor


def _has_native_qmm() -> bool:
    from omlx.patches.qwen35_q4_mlp import _has_native_qmm

    return _has_native_qmm()


def _has_nax() -> bool:
    try:
        from omlx.custom_kernels.nax import is_nax_available

        return bool(is_nax_available())
    except Exception:
        return False


needs_qmm = pytest.mark.skipif(
    not _has_native_qmm(), reason="needs the native quantized matmul kernels"
)


# --- models and adapters ---------------------------------------------------


def _text_backbone(module=None, seed: int = 0) -> DecisionBackbone:
    """A text-only backbone, the way DecisionBackbone loads an autotrust JEV bundle."""
    if module is None:
        from mlx_lm.models import qwen3_5 as module
    mx.random.seed(seed)
    model = module.Model(module.ModelArgs.from_dict(_TEXT_CONFIG))
    mx.eval(model.parameters())
    backbone = DecisionBackbone("tiny-text")
    backbone.model = model
    backbone._text_model = model.language_model.model
    backbone._lm_head = model.language_model.lm_head
    return backbone


def _vlm_backbone(seed: int = 0) -> DecisionBackbone:
    """A backbone with a vision tower, the shape of a Qwen3.8 chat checkpoint."""
    from types import SimpleNamespace

    from mlx_vlm.models.qwen3_5 import Model, ModelConfig

    mx.random.seed(seed)
    config = ModelConfig.from_dict(
        {
            "model_type": "qwen3_5",
            "text_config": _TEXT_CONFIG,
            "vision_config": _VISION_CONFIG,
            "image_token_id": 300,
            "video_token_id": 301,
            "vision_start_token_id": 302,
            "vision_end_token_id": 303,
            "vocab_size": 320,
        }
    )
    model = Model(config)
    mx.eval(model.parameters())
    backbone = DecisionBackbone("tiny-vlm")
    backbone.model = model
    backbone.image_processor = SimpleNamespace(merge_size=2)
    backbone._text_model = model.language_model.model
    backbone._lm_head = model.language_model.lm_head
    return backbone


def _target_paths(text_model) -> list[str]:
    paths = []
    for i, layer in enumerate(text_model.layers):
        for block, names in _TARGETS.items():
            if block in layer:
                paths += [f"layers.{i}.{block}.{name}" for name in names]
    return paths


def _leaf(module, path: str):
    for part in path.split("."):
        module = module[int(part)] if isinstance(module, list) else module[part]
    return module


def _write_adapter(directory, text_model, seed: int = 7):
    """A random PEFT adapter for every target, with autotrust JEV's key naming.

    Returns its folder and its ``(A, B)`` pair per module path, for merging.
    """
    import json

    mx.random.seed(seed)
    tensors, modules = {}, {}
    for path in _target_paths(text_model):
        out_dims, in_dims = _leaf(text_model, path).weight.shape
        a = mx.random.normal((4, in_dims))
        b = mx.random.normal((out_dims, 4)) * 0.05
        key = f"base_model.model.model.{path}"
        tensors[f"{key}.lora_A.weight"] = a
        tensors[f"{key}.lora_B.weight"] = b
        modules[path] = (a, b)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(json.dumps(_PEFT_CONFIG))
    mx.save_safetensors(str(directory / "adapter_model.safetensors"), tensors)
    return SimpleNamespace(path=directory, modules=modules)


def _count_wrapper_calls(monkeypatch) -> Counter:
    """Count the calls of each LoRA wrapper, by wrapper identity."""
    calls = Counter()
    original = LoRALinear.__call__

    def counting(self, x):
        calls[id(self)] += 1
        return original(self, x)

    monkeypatch.setattr(LoRALinear, "__call__", counting)
    return calls


def _quantize(backbone: DecisionBackbone) -> None:
    """bf16 model with 4-bit affine linears, group size 64, bf16 scales."""
    backbone.model.set_dtype(mx.bfloat16)
    nn.quantize(
        backbone._text_model,
        group_size=64,
        bits=4,
        class_predicate=lambda _, m: isinstance(m, nn.Linear),
    )
    mx.eval(backbone.model.parameters())


def _merge(backbone: DecisionBackbone, adapter, source=None) -> None:
    """Set every linear of ``backbone`` to ``source``'s weight, dequantized,
    plus the adapter delta on the targets. ``source`` defaults to itself."""
    source = backbone if source is None else source
    weights = {}
    for path, module in source._text_model.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            weight = mx.dequantize(
                module.weight,
                module.scales,
                module.biases,
                group_size=module.group_size,
                bits=module.bits,
            ).astype(mx.float32)
        elif isinstance(module, nn.Linear):
            weight = module.weight.astype(mx.float32)
        else:
            continue
        if path in adapter.modules:
            a, b = adapter.modules[path]
            weight = weight + _SCALE * (b @ a)
        weights[f"{path}.weight"] = weight
    backbone._text_model.load_weights(list(weights.items()), strict=False)
    mx.eval(backbone.model.parameters())


def _float_copy_of(backbone: DecisionBackbone, make) -> DecisionBackbone:
    """A float32 backbone with ``backbone``'s non-linear parameters."""
    copy = make()
    params = {
        k: v.astype(mx.float32)
        for k, v in tree_flatten(backbone.model.parameters())
        if not k.endswith((".scales", ".biases")) and v.dtype != mx.uint32
    }
    copy.model.load_weights(list(params.items()), strict=False)
    return copy


def _last_hidden(backbone: DecisionBackbone, chunk: int = 1 << 30) -> mx.array:
    embeds, positions = backbone.embed(_IDS)
    generator = backbone.prefill(
        _IDS, embeds, positions, backbone.make_cache(), lambda: chunk, keep_all=False
    )
    try:
        while True:
            next(generator)
    except StopIteration as stop:
        return stop.value.astype(mx.float32)


def _rel_error(actual: mx.array, expected: mx.array) -> float:
    return (mx.abs(actual - expected).max() / mx.abs(expected).max()).item()


# --- float base ------------------------------------------------------------


@pytest.mark.parametrize("make", [_text_backbone, _vlm_backbone])
def test_wrapped_float_model_matches_merged_weights(tmp_path, make, monkeypatch):
    wrapped, merged = make(), make()
    adapter = _write_adapter(tmp_path, wrapped._text_model)
    apply_adapter(wrapped._text_model, adapter.path)
    _merge(merged, adapter)
    wrapper_calls = _count_wrapper_calls(monkeypatch)

    expected = _last_hidden(merged)
    one_shot = _last_hidden(wrapped)
    chunked = _last_hidden(wrapped, chunk=7)

    assert _rel_error(one_shot, expected) < _FLOAT_TOLERANCE
    assert _rel_error(chunked, expected) < _FLOAT_TOLERANCE
    # One one-shot pass and six chunks: every wrapper ran once per chunk.
    calls = {wrapper_calls[id(_leaf(wrapped._text_model, p))] for p in adapter.modules}
    assert calls == {1 + 6}


# --- quantized base --------------------------------------------------------


@pytest.mark.parametrize("make", [_text_backbone, _vlm_backbone])
def test_wrapped_quantized_model_matches_dequantized_merge(tmp_path, make):
    wrapped = make()
    _quantize(wrapped)
    adapter = _write_adapter(tmp_path, make()._text_model)
    reference = _float_copy_of(wrapped, make)
    _merge(reference, adapter, source=wrapped)
    base_only = _last_hidden(wrapped)

    apply_adapter(wrapped._text_model, adapter.path)

    expected = _last_hidden(reference)
    assert _rel_error(_last_hidden(wrapped), expected) < _QUANTIZED_TOLERANCE
    # The delta is far larger than the tolerance, so a lost delta would fail.
    assert _rel_error(base_only, expected) > 10 * _QUANTIZED_TOLERANCE


# --- quantized prefill patches ---------------------------------------------


def _fresh_mlx_lm_qwen35(monkeypatch):
    """A private copy of mlx-lm's qwen3_5, so its classes can be patched."""
    import mlx_lm.models.qwen3_5 as stock

    name = "mlx_lm.models._omlx_lora_test_qwen35"
    spec = importlib.util.spec_from_file_location(name, stock.__file__)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "mlx_lm.models"
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _count_fused_calls(monkeypatch) -> dict:
    from omlx.custom_kernels.quant_matmul import fast

    calls = {"count": 0}
    original = fast.affine_q4_qmm_t

    def spy(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(fast, "affine_q4_qmm_t", spy)
    return calls


@needs_qmm
def test_mlx_lm_prefill_patches_fall_back_to_the_wrappers(tmp_path, monkeypatch):
    from omlx.patches import qwen35_q4_mlp as q4

    module = _fresh_mlx_lm_qwen35(monkeypatch)
    monkeypatch.setattr(q4, "_MIN_TOKENS", 16)
    monkeypatch.setattr(q4, "_LM_LINEAR_PATCHED", q4._LM_LINEAR_PATCHED)
    assert q4._patch_class(module.__name__, "MLP", q4._QMM_VARIANT, 16, 16384)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.qwen3_5", module)
    assert q4.apply_qwen35_q4_lm_prefill_linear_patch()
    assert module.GatedDeltaNet._omlx_q4_lm_gdn_patched

    make = lambda: _text_backbone(module)  # noqa: E731
    wrapped = make()
    _quantize(wrapped)
    adapter = _write_adapter(tmp_path, make()._text_model)
    reference = _float_copy_of(wrapped, make)
    _merge(reference, adapter, source=wrapped)
    calls = _count_fused_calls(monkeypatch)

    base_only = _last_hidden(wrapped)
    assert calls["count"] > 0, "the base model must take the fused path"
    apply_adapter(wrapped._text_model, adapter.path)
    calls["count"] = 0
    wrapper_calls = _count_wrapper_calls(monkeypatch)

    result = _last_hidden(wrapped)

    expected = _last_hidden(reference)
    assert calls["count"] == 0, "wrapped leaves must not be routed past the delta"
    assert {
        wrapper_calls[id(_leaf(wrapped._text_model, p))] for p in adapter.modules
    } == {1}
    assert _rel_error(result, expected) < _QUANTIZED_TOLERANCE
    assert _rel_error(base_only, expected) > 10 * _QUANTIZED_TOLERANCE


@needs_qmm
def test_mlx_vlm_prefill_patches_keep_the_delta(tmp_path, monkeypatch):
    from mlx_vlm.models.qwen3_5 import language

    from omlx.patches import qwen35_q4_mlp as q4

    mlp_cls = language.Qwen3_5MLP
    monkeypatch.setattr(
        mlp_cls,
        "__call__",
        q4._make_patched_mlp(mlp_cls.__call__, q4._QMM_VARIANT, 16, 16384),
    )
    routed = q4._VLMQuantizedPrefillLinear
    for attr in ("_route_min_tokens", "_route_q8_min_tokens", "_route_variant"):
        monkeypatch.setattr(routed, attr, getattr(routed, attr))
    monkeypatch.setattr(q4, "_MIN_TOKENS", 16)

    wrapped = _vlm_backbone()
    _quantize(wrapped)
    assert q4.apply_qwen35_q4_prefill_linear_patch(wrapped.model)
    adapter = _write_adapter(tmp_path, _vlm_backbone()._text_model)
    reference = _float_copy_of(wrapped, _vlm_backbone)
    _merge(reference, adapter, source=wrapped)
    calls = _count_fused_calls(monkeypatch)
    base_only = _last_hidden(wrapped)
    assert calls["count"] > 0, "the base model must take the fused path"

    # Wrap after the patch, as an attached adapter would: the routed leaves
    # become the wrappers' bases and keep their fused kernels.
    apply_adapter(wrapped._text_model, adapter.path)
    assert isinstance(wrapped._text_model.layers[1].self_attn.q_proj.base, routed)
    calls["count"] = 0

    result = _last_hidden(wrapped)

    expected = _last_hidden(reference)
    assert calls["count"] > 0, "the bases still use the fused kernels"
    assert _rel_error(result, expected) < _QUANTIZED_TOLERANCE
    assert _rel_error(base_only, expected) > 10 * _QUANTIZED_TOLERANCE


# --- packed leaves ---------------------------------------------------------


class _PackedLike(nn.Module):
    """Like PackedLinear: not a QuantizedLinear, and no ``weight``."""

    def __init__(self, weight: mx.array):
        super().__init__()
        self.input_dims = weight.shape[1]
        self.output_dims = weight.shape[0]
        self.packed = weight

    def __call__(self, x):
        return x @ self.packed.T


def test_a_weightless_leaf_is_wrapped_and_called(tmp_path):
    wrapped, merged = _text_backbone(), _text_backbone()
    adapter = _write_adapter(tmp_path, wrapped._text_model)
    mlp = wrapped._text_model.layers[0].mlp
    mlp.up_proj = _PackedLike(mlp.up_proj.weight)
    apply_adapter(wrapped._text_model, adapter.path)
    _merge(merged, adapter)

    assert isinstance(mlp.up_proj.base, _PackedLike)
    assert _rel_error(_last_hidden(wrapped), _last_hidden(merged)) < _FLOAT_TOLERANCE


@pytest.mark.skipif(not _has_nax(), reason="PackedLinear needs the M5 tensor unit")
def test_a_real_packed_linear_is_wrapped_and_called():
    from omlx.patches.qwen35_packed_linear import PackedLinear, _pack

    mx.random.seed(3)
    linear = nn.QuantizedLinear(512, 256, bias=False, group_size=64, bits=4)
    weight = (mx.random.normal((256, 512)) * 0.02).astype(mx.bfloat16)
    linear.weight, linear.scales, linear.biases = mx.quantize(
        weight, group_size=64, bits=4
    )
    packed = _pack([linear])[0]
    assert isinstance(packed, PackedLinear)
    a = mx.random.normal((4, 512))
    b = mx.random.normal((256, 4)) * 0.05
    # LoRALinear takes B as [rank, out], with the scale folded in.
    wrapper = LoRALinear(
        packed, a.astype(mx.bfloat16), mx.contiguous((_SCALE * b).T.astype(mx.bfloat16))
    )
    dense = mx.dequantize(
        linear.weight, linear.scales, linear.biases, group_size=64, bits=4
    ).astype(mx.float32)

    for rows in (1, 8, 70):
        x = (mx.random.normal((1, rows, 512)) * 0.5).astype(mx.bfloat16)
        expected = x.astype(mx.float32) @ (dense + _SCALE * b @ a).T
        assert _rel_error(wrapper(x).astype(mx.float32), expected) < 2e-2
