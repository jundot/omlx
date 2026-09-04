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
