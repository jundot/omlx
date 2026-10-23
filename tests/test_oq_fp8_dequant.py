# SPDX-License-Identifier: Apache-2.0
"""Tests for _block_dequant_fp8 scale decoding.

MXFP8 checkpoints (e.g. MiniMax-M3) store their e8m0 block scales with
safetensors dtype U8. Those bytes are shared exponents and must decode as
2^(s - 127), the same as the F8_E8M0 branch. Treating them as linear
scales blows the weights up by orders of magnitude.

DeepSeek-style FP8 checkpoints also use weight_scale_inv keys but store
the scales as real floats (block 128). Those must keep multiplying
linearly, so the discriminator is the scale dtype, not the key name.
"""

import glob
import json
import os
import struct

import mlx.core as mx
import numpy as np
import pytest

from omlx.oq import _block_dequant_fp8, _LazyTensorIndex

M3_DIR = "/Volumes/Scratch/models/MiniMax-M3-MXFP8"


def _write_safetensors(path, tensors):
    """Minimal safetensors writer for dtypes numpy cannot represent.

    tensors: {name: (dtype_str, shape, raw_bytes)}
    """
    header = {}
    offset = 0
    for name, (dtype_str, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype_str,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
    header_json = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for _, (_, _, data) in tensors.items():
            f.write(data)


def test_u8_scale_decodes_as_e8m0_exponent():
    mx.random.seed(0)
    w = mx.random.normal((64, 128)).astype(mx.bfloat16)
    qw, scales = mx.quantize(w, group_size=32, bits=8, mode="mxfp8")
    ref = mx.dequantize(qw, scales, group_size=32, bits=8, mode="mxfp8")
    assert scales.dtype == mx.uint8

    # On-disk view of the same data: raw e4m3 bytes, one per element.
    raw_fp8 = qw.view(mx.uint8)
    assert raw_fp8.shape == (64, 128)

    # Sanity check the target first: the explicit from_fp8 * 2^(s-127)
    # formula must reproduce mx.dequantize exactly, otherwise ref is not
    # a valid oracle for the function under test.
    explicit = (
        mx.from_fp8(raw_fp8, dtype=mx.bfloat16).reshape(64, 4, 32).astype(mx.float32)
        * mx.power(mx.array(2.0), scales.astype(mx.float32) - 127.0)[:, :, None]
    ).reshape(64, 128)
    assert mx.array_equal(explicit.astype(mx.bfloat16), ref).item()

    got = _block_dequant_fp8(raw_fp8, scales, "F8_E4M3", "U8")
    assert got.shape == ref.shape
    assert mx.allclose(
        got.astype(mx.float32), ref.astype(mx.float32), atol=1e-2, rtol=1e-2
    ).item(), (
        f"mean|got|={mx.abs(got).mean().item():.4g} vs "
        f"mean|ref|={mx.abs(ref).mean().item():.4g}"
    )


def test_f32_scale_stays_linear():
    # DeepSeek-style pair: e4m3 weight with a float block scale
    # (block 128). The scale is a linear multiplier and must be applied
    # as-is, untouched by the U8 exponent decoding.
    mx.random.seed(1)
    w = mx.random.normal((256, 128)).astype(mx.bfloat16)
    qw, _ = mx.quantize(w, group_size=32, bits=8, mode="mxfp8")
    raw_fp8 = qw.view(mx.uint8)
    scale = mx.array([[0.5], [2.0]], dtype=mx.float32)

    got = _block_dequant_fp8(raw_fp8, scale, "F8_E4M3", "F32")

    wf = mx.from_fp8(raw_fp8, dtype=mx.bfloat16).astype(mx.float32)
    expected = mx.concatenate([wf[:128] * 0.5, wf[128:] * 2.0], axis=0)
    assert mx.allclose(got.astype(mx.float32), expected, atol=1e-2, rtol=1e-2).item()


def test_weight_scale_pair_discovery_and_dequant(tmp_path):
    # compressed-tensors float-quantized (Laguna FP8): X.weight (F8_E4M3)
    # + X.weight_scale (f32 block scales). The pair must be discovered,
    # the scale key hidden, and _dequant_one must fold the [128, 128]
    # blocks linearly. Attention k_scale/v_scale sidecars must not pair.
    mx.random.seed(2)
    w_true = mx.random.normal((128, 256)).astype(mx.float32)
    scale = mx.array([[0.5, 2.0]], dtype=mx.float32)
    scale_expand = mx.repeat(mx.repeat(scale, 128, axis=0), 128, axis=1)
    codes = mx.to_fp8(w_true / scale_expand)

    shard = str(tmp_path / "model.safetensors")
    _write_safetensors(
        shard,
        {
            "model.layers.0.mlp.down_proj.weight": (
                "F8_E4M3",
                codes.shape,
                np.array(codes).tobytes(),
            ),
            "model.layers.0.mlp.down_proj.weight_scale": (
                "F32",
                scale.shape,
                np.array(scale).tobytes(),
            ),
            "model.layers.0.self_attn.k_scale": (
                "F32",
                (1,),
                np.ones(1, dtype=np.float32).tobytes(),
            ),
        },
    )

    idx = _LazyTensorIndex([shard])
    wk = "model.layers.0.mlp.down_proj.weight"
    assert idx._fp8_pairs.get(wk) == f"{wk}_scale"
    assert idx.source_quant_info(wk) is None  # dequant path, not passthrough
    assert not idx._is_visible(f"{wk}_scale")
    assert idx._is_visible("model.layers.0.self_attn.k_scale")
    assert "model.layers.0.self_attn.k_scale" not in idx._fp8_pairs

    got = idx._dequant_one(wk)
    expected = mx.from_fp8(codes, dtype=mx.bfloat16).astype(mx.float32) * scale_expand
    assert got.shape == (128, 256)
    assert mx.allclose(
        got.astype(mx.float32), expected, atol=1e-2, rtol=1e-2
    ).item()


@pytest.mark.skipif(not os.path.isdir(M3_DIR), reason="M3 not present")
def test_minimax_m3_k_proj_magnitude():
    # Grounded check on a real MXFP8 checkpoint. Pre-fix this layer
    # dequantized to mean|w| ~13410; the correct value is ~0.03.
    shards = sorted(glob.glob(os.path.join(M3_DIR, "model-*.safetensors")))
    idx = _LazyTensorIndex(shards)
    key = "language_model.model.layers.3.self_attn.k_proj.weight"
    weight = idx._dequant_one(key)
    mean_abs = mx.abs(weight).mean().item()
    max_abs = mx.abs(weight).max().item()
    assert mean_abs < 1.0, f"mean|w|={mean_abs}"
    assert max_abs < 2.0, f"max|w|={max_abs}"


def test_mimo_mxfp4_index_preserves_packed_experts_through_stack(tmp_path):
    from omlx.oq import _discover_sanitize_plan, _DiscoveredPlan

    mx.random.seed(5)
    packed, scales = mx.quantize(
        mx.random.normal((2, 64, 128)), group_size=32, bits=4, mode="mxfp4"
    )
    tensors = {}
    for i in range(2):
        key = f"experts.{i}.weight"
        tensors[key] = packed[i].view(mx.uint8)
        tensors[key + "_scale"] = scales[i]
    path = tmp_path / "model.safetensors"
    mx.save_safetensors(str(path), tensors)
    config = {"model_type": "mimo_v2", "quantization_config": {"store_dtype": "mxfp4"}}
    index = _LazyTensorIndex([path], config=config)
    assert index.logical_metadata()["experts.0.weight"][0] == (64, 128)
    assert "experts.0.weight_scale" not in index

    def sanitize(weights):
        return {
            "switch_mlp.weight": mx.stack(
                [weights[f"experts.{i}.weight"] for i in range(2)]
            )
        }

    plan = _DiscoveredPlan(_discover_sanitize_plan(sanitize, index), index)
    assert plan.source_quant_info("switch_mlp.weight")["mode"] == "mxfp4"
    actual_weight, actual_scales = plan.pop_packed("switch_mlp.weight")
    assert mx.array_equal(actual_weight, packed).item()
    assert mx.array_equal(actual_scales, scales).item()
    other = _LazyTensorIndex([path])
    assert other.source_quant_info("experts.0.weight") is None


def test_minimax_nvfp4_experts_expose_real_weight_and_packed_source(tmp_path):
    from omlx.oq import _discover_sanitize_plan, _DiscoveredPlan
    from omlx.patches.mlx_vlm_minimax_m3_compat.virtual_nvfp4 import (
        routed_expert_scale_tensors,
    )

    mx.random.seed(7)
    prefix = "language_model.model.layers.0.block_sparse_moe"
    tensors, packed, scales, globals_, real = {}, {}, {}, {}, {}
    for expert in range(2):
        for proj, shape in (("w1", (32, 64)), ("w2", (64, 32)), ("w3", (32, 64))):
            w = mx.random.normal(shape) * 0.05
            global_scale = 2688.0 / mx.abs(w).max()
            wq, ws = mx.quantize(w * global_scale, group_size=16, bits=4, mode="nvfp4")
            base = f"{prefix}.experts.{expert}.{proj}"
            tensors[f"{base}.weight_packed"] = wq.view(mx.uint8)
            tensors[f"{base}.weight_scale"] = ws
            tensors[f"{base}.weight_global_scale"] = global_scale.reshape(())
            tensors[f"{base}.input_global_scale"] = mx.array(1.0)
            packed[base], scales[base], globals_[base] = wq, ws, global_scale
            real[base] = (
                mx.dequantize(wq, ws, group_size=16, bits=4, mode="nvfp4")
                / global_scale
            )
    path = tmp_path / "model.safetensors"
    mx.save_safetensors(str(path), tensors)
    config = {
        "model_type": "minimax_m3_vl",
        "text_config": {"num_hidden_layers": 1, "num_local_experts": 2},
        "quantization_config": {"quant_method": "mxfp8", "moe_quant_algo": "NVFP4"},
    }

    index = _LazyTensorIndex([path], config=config)

    w1 = f"{prefix}.experts.0.w1.weight"
    assert index.logical_metadata()[w1] == ((32, 64), "BF16")
    assert f"{prefix}.experts.0.w1.weight_packed" not in index
    assert f"{prefix}.experts.0.w1.weight_global_scale" not in index
    assert f"{prefix}.experts.0.w1.input_global_scale" not in index
    expected = real[f"{prefix}.experts.0.w1"].astype(mx.bfloat16)
    assert mx.array_equal(index[w1], expected).item()
    assert index.source_quant_info(w1)["mode"] == "nvfp4"

    def sanitize(weights):
        return {
            f"{prefix}.switch_mlp.gate_proj.weight": mx.stack(
                [weights[f"{prefix}.experts.{e}.w1.weight"] for e in range(2)]
            )
        }

    plan = _DiscoveredPlan(_discover_sanitize_plan(sanitize, index), index)
    gate = f"{prefix}.switch_mlp.gate_proj.weight"
    assert plan.source_quant_info(gate)["group_size"] == 16
    actual_weight, actual_scales = plan.pop_packed(gate)
    bases = [f"{prefix}.experts.{e}.w1" for e in range(2)]
    assert mx.array_equal(actual_weight, mx.stack([packed[b] for b in bases])).item()
    assert mx.array_equal(actual_scales, mx.stack([scales[b] for b in bases])).item()

    vectors = routed_expert_scale_tensors(
        index,
        config,
        {f"{prefix}.switch_mlp.gate_proj": {"mode": "nvfp4"}},
    )
    # mlx-lm layout: only the NVFP4 projection gets a vector, 2688 / G.
    assert set(vectors) == {f"{prefix}.switch_mlp.gate_proj.global_scale"}
    expected_gate = mx.stack([2688.0 / globals_[b] for b in bases])
    assert mx.array_equal(
        vectors[f"{prefix}.switch_mlp.gate_proj.global_scale"], expected_gate
    ).item()

    # The same tensors without the NVFP4 declaration stay untouched.
    plain = _LazyTensorIndex([path], config={"model_type": "minimax_m3_vl"})
    assert f"{prefix}.experts.0.w1.weight_packed" in plain
    assert w1 not in plain
