# SPDX-License-Identifier: Apache-2.0
"""Hot promotion batches buffer materialization without changing cache bytes."""

from unittest.mock import patch

import mlx.core as mx
import pytest

from omlx.cache import paged_ssd_cache as ssd


@pytest.mark.parametrize("layout", ["dense", "strided", "scalar", "empty"])
def test_promotion_materializes_once_and_keeps_exact_owned_bytes(tmp_path, layout):
    dtypes = [
        mx.bfloat16,
        mx.float16,
        mx.float32,
        mx.int8,
        mx.int16,
        mx.int32,
        mx.int64,
        mx.uint8,
        mx.uint16,
        mx.uint32,
        mx.uint64,
        mx.bool_,
    ]
    arrays = {}
    for dtype in dtypes:
        value = (mx.arange(24).reshape(2, 3, 4) % 7).astype(dtype)
        if layout == "strided":
            value = value.transpose(2, 0, 1)[::2, :, :]
        elif layout == "scalar":
            value = value[0, 0, 0]
        elif layout == "empty":
            value = value[:0]
        arrays[str(dtype)] = value
    expected = {name: ssd._extract_tensor_bytes(arr) for name, arr in arrays.items()}
    # Make new lazy views for promotion instead of passing only ready arrays.
    arrays = {name: mx.contiguous(arr) for name, arr in arrays.items()}
    manager = ssd.PagedSSDCacheManager(
        cache_dir=tmp_path, max_size_bytes=1024**3, hot_cache_max_bytes=1024**2
    )
    metadata = ssd.PagedSSDBlockMetadata(
        block_hash=b"x" * 32,
        file_path=tmp_path / "block.safetensors",
        file_size=0,
        created_at=0.0,
        last_access=0.0,
        token_count=1,
        num_layers=1,
        model_name="synthetic",
    )
    try:
        with patch.object(ssd.mx, "eval", wraps=mx.eval) as evaluate:
            assert manager._promote_to_hot_cache(
                b"x" * 32, arrays, {"synthetic": "yes"}, metadata
            )
            assert evaluate.call_count == 1
        entry = manager._hot_cache_get(b"x" * 32)
        assert entry["tensors_raw"] == expected
        assert all(type(tensor[0]) is bytes for tensor in entry["tensors_raw"].values())
        assert entry["dirty"] is False
        assert entry["file_metadata"] == {"synthetic": "yes"}
        assert manager.get_stats().hot_cache_promotions == 1
    finally:
        manager.close()


def test_cold_disk_load_promotes_with_one_materialization(tmp_path):
    values = mx.arange(256).reshape(1, 2, 8, 16).astype(mx.bfloat16)
    manager = ssd.PagedSSDCacheManager(cache_dir=tmp_path, max_size_bytes=1024**3)
    assert manager.save_block(b"k" * 32, [(values, values)] * 3, 8, "synthetic")
    manager.close()
    reader = ssd.PagedSSDCacheManager(
        cache_dir=tmp_path, max_size_bytes=1024**3, hot_cache_max_bytes=1024**2
    )
    try:
        with patch.object(ssd.mx, "eval", wraps=mx.eval) as evaluate:
            data, metadata = reader.load_block_with_metadata(b"k" * 32)
            assert data is not None and metadata is not None
            assert evaluate.call_count == 1
        mx.eval(data)
        for key, value in data:
            assert mx.array_equal(key, values).item()
            assert mx.array_equal(value, values).item()
        assert reader._hot_cache_get(b"k" * 32)["dirty"] is False
    finally:
        reader.close()


def test_failed_materialization_does_not_publish_partial_hot_entry(tmp_path):
    manager = ssd.PagedSSDCacheManager(
        cache_dir=tmp_path, max_size_bytes=1024**3, hot_cache_max_bytes=1024**2
    )
    try:
        with patch.object(
            ssd.mx, "eval", side_effect=RuntimeError("synthetic failure")
        ):
            assert not manager._promote_to_hot_cache(
                b"x" * 32, {"x": mx.ones(8)}, {}, None
            )
        assert manager._hot_cache_get(b"x" * 32) is None
        assert manager.get_stats().hot_cache_promotion_failures == 1
    finally:
        manager.close()


def test_promotion_preserves_bfloat16_special_value_bits(tmp_path):
    bits = mx.array(
        [0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 0x7FA5, 0xFFD3, 0x3F80],
        dtype=mx.uint16,
    )
    mx.eval(bits)
    expected = bytes(memoryview(bits))
    values = bits.view(mx.bfloat16)
    metadata = ssd.PagedSSDBlockMetadata(
        block_hash=b"b" * 32,
        file_path=tmp_path / "block.safetensors",
        file_size=0,
        token_count=1,
        created_at=0.0,
        last_access=0.0,
        num_layers=1,
    )
    manager = ssd.PagedSSDCacheManager(
        cache_dir=tmp_path, max_size_bytes=1024**3, hot_cache_max_bytes=1024**2
    )
    try:
        assert manager._promote_to_hot_cache(b"b" * 32, {"bits": values}, {}, metadata)
        assert manager._hot_cache_get(b"b" * 32)["tensors_raw"]["bits"] == (
            expected,
            "BF16",
            [8],
        )
    finally:
        manager.close()
