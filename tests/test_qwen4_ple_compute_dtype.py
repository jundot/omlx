# SPDX-License-Identifier: Apache-2.0
"""PLE row decoding must honor an explicitly loaded shared scale dtype."""

import json

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("layout", ["dense", "affine", "mixed_affine"])
@pytest.mark.parametrize("prefetch", [False, True])
def test_mmap_ple_preserves_compute_dtype(tmp_path, dtype, layout, prefetch):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import DiskBackedShardedEmbedding

    prefix = "language_model.model.layers.0.ple.ple_embedding.ngram_embedding"
    tensors, expected = {}, []
    for shard in range(2):
        # Values distinguish FP16 from an intermediate BF16 rounding.
        dense = (mx.arange(4 * 128).reshape(4, 128) / 317 + shard).astype(dtype)
        name = f"{prefix}.shards.{shard}"
        if layout == "dense":
            tensors[f"{name}.weight"] = dense
            expected.append(dense)
        else:
            group = 64 if layout == "mixed_affine" and shard else 32
            weight, scales, biases = mx.quantize(dense, group_size=group, bits=4)
            tensors.update(
                {
                    f"{name}.weight": weight,
                    f"{name}.scales": scales,
                    f"{name}.biases": biases,
                }
            )
            expected.append(
                mx.dequantize(weight, scales, biases, group_size=group, bits=4)
            )
    filename = "model.safetensors"
    mx.save_safetensors(str(tmp_path / filename), tensors)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: filename for key in tensors}})
    )
    embedding = DiskBackedShardedEmbedding(tmp_path, prefix, 8, 128, 2)
    scale = mx.array([0.125], dtype=dtype)
    embedding.load_weights([("weight_scale", scale)])
    indices = mx.array([[7, 1, 7, 0, 4]])
    try:
        if prefetch:
            embedding.prefetch(indices)
        actual = embedding(indices)
        reference = mx.concatenate(expected)[indices] * scale
        assert actual.dtype == dtype
        assert mx.array_equal(actual, reference).item()
        assert embedding.last_touched_shards == (0, 1)
        assert embedding.last_prefetch_hit == (prefetch and layout != "mixed_affine")
        empty = embedding(mx.array([], dtype=mx.int32))
        assert empty.shape == (0, 128)
        assert empty.dtype == dtype
        # Neither gather path alters the source packed weights/metadata.
        stored = mx.load(str(tmp_path / filename))
        for key, original in tensors.items():
            assert mx.array_equal(stored[key], original).item()
    finally:
        embedding.close()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_resident_fp8_ple_decodes_at_shared_scale_dtype(dtype):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import ShardedEmbedding

    embedding = ShardedEmbedding(4, 8, 2)
    source = mx.arange(32).reshape(4, 8).astype(mx.float32) / 17
    packed = mx.to_fp8(source)
    embedding.shards[0].weight = packed[:2]
    embedding.shards[1].weight = packed[2:]
    embedding.weight_scale = mx.array([0.125], dtype=dtype)
    indices = mx.array([[3, 0, 3]])
    actual = embedding(indices)
    expected = mx.from_fp8(packed, dtype=dtype)[indices] * embedding.weight_scale
    assert actual.dtype == dtype
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_mmap_fp8_conversion_accepts_compute_dtype(dtype):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import _SafeTensorMMap

    packed = mx.to_fp8(mx.arange(32).astype(mx.float32) / 17)
    actual = _SafeTensorMMap.to_mx(np.asarray(packed), "F8_E4M3", fp8_dtype=dtype)
    expected = mx.from_fp8(packed, dtype=dtype)
    assert actual.dtype == dtype
    assert mx.array_equal(actual, expected).item()
