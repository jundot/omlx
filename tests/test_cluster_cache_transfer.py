from __future__ import annotations

import copy

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache

from omlx.cluster.cache_transfer import (
    _transfer_window,
    prepare_cache_transfer,
    restore_cache_transfer,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    ((None, 8), ("garbage", 8), ("0", 1), ("4", 4), ("99", 16)),
)
def test_cache_transfer_window_is_bounded(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("OMLX_CLUSTER_CACHE_TRANSFER_WINDOW", raising=False)
    else:
        monkeypatch.setenv("OMLX_CLUSTER_CACHE_TRANSFER_WINDOW", raw)
    assert _transfer_window() == expected


def _fixture_cache():
    arrays = ArraysCache.from_state(
        [
            mx.arange(24, dtype=mx.float32).reshape(1, 3, 8),
            mx.arange(16, dtype=mx.bfloat16).reshape(1, 2, 8),
        ],
        "",
    )
    kv = KVCache.from_state(
        (
            mx.arange(64, dtype=mx.bfloat16).reshape(1, 2, 4, 8),
            mx.arange(64, dtype=mx.bfloat16).reshape(1, 2, 4, 8) + 1,
        ),
        "",
    )
    mx.eval(arrays.state, kv.state)
    return [arrays, kv]


def _flatten_arrays(value):
    from mlx.utils import tree_flatten

    return [leaf for _key, leaf in tree_flatten(value)]


def test_cache_transfer_round_trip_preserves_classes_metadata_and_arrays():
    original = _fixture_cache()
    prepared = prepare_cache_transfer(
        original,
        model_identity="fixture-model",
        prompt_tokens=4,
    )
    restored = restore_cache_transfer(
        prepared.manifest,
        [mx.array(value) for value in prepared.arrays],
        expected_model_identity="fixture-model",
    )

    assert [type(value) for value in restored] == [ArraysCache, KVCache]
    assert [value.meta_state for value in restored] == ["", ""]
    assert prepared.nbytes == sum(value.nbytes for value in prepared.arrays)
    left = _flatten_arrays([value.state for value in original])
    right = _flatten_arrays([value.state for value in restored])
    assert len(left) == len(right)
    assert all(bool(mx.array_equal(a, b).item()) for a, b in zip(left, right))
    assert restored[1].offset == 4


def test_cache_transfer_rejects_model_or_tensor_contract_mismatch():
    prepared = prepare_cache_transfer(
        _fixture_cache(),
        model_identity="fixture-model",
        prompt_tokens=4,
    )
    with pytest.raises(ValueError, match="model identity"):
        restore_cache_transfer(
            prepared.manifest,
            prepared.arrays,
            expected_model_identity="other-model",
        )

    manifest = copy.deepcopy(prepared.manifest)
    manifest["tensors"][0]["shape"][-1] += 1
    with pytest.raises(ValueError, match="wrong shape"):
        restore_cache_transfer(manifest, prepared.arrays)


def test_cache_transfer_rejects_unknown_cache_class():
    prepared = prepare_cache_transfer(
        _fixture_cache(),
        model_identity="fixture-model",
        prompt_tokens=4,
    )
    manifest = copy.deepcopy(prepared.manifest)
    class_item = next(item for item in manifest["metadata"] if item[1] == "ArraysCache")
    class_item[1] = "ArbitraryCache"
    with pytest.raises(ValueError, match="does not admit class"):
        restore_cache_transfer(manifest, prepared.arrays)
