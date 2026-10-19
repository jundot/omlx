from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import pytest
from mlx_vlm.turboquant import TurboQuantKVCache

from omlx.affine4 import (
    Affine4KVCache,
    Affine8KVCache,
    BatchAffine4KVCache,
)
from omlx.cache.hybrid_cache import ModelCacheConfig
from omlx.cache.paged_cache import BlockTable, PagedCacheManager
from omlx.cache.paged_ssd_cache import (
    PagedSSDCacheManager,
    _cache_compat_signature,
    _canonicalize_layer_cache_types,
)
from omlx.cache.prefix_cache import BlockAwarePrefixCache
from omlx.cache.type_registry import CacheTypeRegistry
from omlx.turboquant_kv import _slice_state_range


def _cache(length=8, key_dim=13, value_dim=21, cls=Affine4KVCache):
    cache = cls(bits=getattr(cls, "bits", 4), seed=7)
    keys = mx.random.normal((1, 2, length, key_dim)).astype(mx.float16)
    values = mx.random.normal((1, 2, length, value_dim)).astype(mx.float16)
    cache.update_and_fetch(keys, values)
    mx.eval(cache.state)
    return cache


def _state(cache):
    return tuple(getattr(part, "_state", part) for part in cache.state)


def _data(cache):
    return {
        "state": _state(cache),
        "class_name": type(cache).__name__,
        "cache_type": "KVCache",
        "meta_state": cache.meta_state,
    }


def _payload(cache, start=0, end=4, marker="__affine4__"):
    return marker, tuple(_slice_state_range(part, start, end) for part in _state(cache))


def _assert_state_equal(expected, actual):
    for left, right in zip(expected, actual):
        assert type(left) is type(right)
        for field in left._fields:
            a, b = getattr(left, field), getattr(right, field)
            assert a.dtype == b.dtype
            assert a.shape == b.shape
            assert mx.array_equal(a, b).item()


@pytest.fixture
def manager_factory(tmp_path):
    managers = []

    def make(types=("Affine4KVCache",), hot=False):
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / str(len(managers)),
            max_size_bytes=10 * 1024**2,
            hot_cache_max_bytes=1024**2 if hot else 0,
            expected_model_name="affine-test",
            expected_num_layers=len(types),
            expected_block_size=4,
        )
        manager.set_expected_layer_signature(list(types), turboquant_kv_bits=4)
        managers.append(manager)
        return manager

    yield make
    for manager in managers:
        manager.close()


def _prefix(ssd, num_layers=1):
    paged = PagedCacheManager(
        block_size=4, max_blocks=32, initial_blocks=32, model_name="affine-test"
    )
    paged.set_paged_ssd_cache_manager(ssd)
    prefix = BlockAwarePrefixCache(
        SimpleNamespace(layers=[object()] * num_layers), paged, ssd
    )
    return prefix, paged


def _chain(payloads, metas, classes=None):
    ssd = MagicMock(spec=PagedSSDCacheManager)
    ssd._expected_layer_cache_types = classes
    prefix, paged = _prefix(ssd)
    blocks = []
    for i in range(len(payloads)):
        block = paged.allocate_block()
        block.block_hash = f"block-{i}".encode()
        block.token_count = 4
        block.ref_count = 2
        blocks.append(block.block_id)
    table = BlockTable(
        request_id="restore", block_ids=blocks, num_tokens=4 * len(blocks)
    )
    ssd.load_block_with_metadata.side_effect = [
        (
            [payload],
            {
                "model_name": "affine-test",
                "num_layers": 1,
                "block_size": 4,
                "layer_cache_types": classes,
                "layer_meta_states": [meta],
            },
        )
        for payload, meta in zip(payloads, metas)
    ]
    return prefix.reconstruct_cache(table)


@pytest.mark.parametrize("matched,batched", [(4, False), (8, True)])
def test_prefix_reuse_keeps_packed_state_and_independent_dimensions(
    manager_factory, matched, batched
):
    manager = manager_factory()
    prefix, _ = _prefix(manager)
    cache = _cache()
    if batched:
        cache = BatchAffine4KVCache.merge([cache])
    assert prefix.store_cache("store", list(range(8)), [_data(cache)]) is not None
    manager.close()
    prefix.release_cache("store")
    table, remaining = prefix.fetch_cache("reuse", list(range(matched)) + [99])
    assert table is not None
    assert table.num_tokens == matched
    assert remaining == [99]
    restored = prefix.reconstruct_cache(table)
    assert restored is not None
    result = restored[0]
    assert type(result) is Affine4KVCache
    assert result.offset == matched
    assert result.key_codec.dim == 13
    assert result.value_codec.dim == 21
    expected = _payload(cache, end=matched)[1]
    _assert_state_equal(expected, _state(result))
    for a, b in zip(cache.dequantize(*expected), result.dequantize(*_state(result))):
        assert mx.array_equal(a, b).item()


def test_affine_and_dense_layers_roundtrip_together(manager_factory):
    manager = manager_factory(types=["Affine4KVCache", "KVCache"])
    prefix, _ = _prefix(manager, num_layers=2)
    affine = _cache()
    dense_state = (mx.ones((1, 2, 8, 13)), mx.zeros((1, 2, 8, 21)))
    dense = {"state": dense_state, "cache_type": "KVCache", "meta_state": (8,)}
    assert (
        prefix.store_cache("store", list(range(8)), [_data(affine), dense]) is not None
    )
    manager.close()
    table, remaining = prefix.fetch_cache("warm", list(range(8)) + [99])
    assert table is not None and remaining == [99]
    restored = prefix.reconstruct_cache(table)
    assert restored is not None
    assert [type(layer).__name__ for layer in restored] == ["Affine4KVCache", "KVCache"]
    _assert_state_equal(_state(affine), _state(restored[0]))
    for expected, actual in zip(dense_state, restored[1].state):
        assert mx.array_equal(expected, actual).item()


@pytest.mark.parametrize("same_request,hot", [(False, False), (True, True)])
def test_final_affine_store_replaces_dense_prefill_blocks(
    manager_factory, same_request, hot
):
    manager = manager_factory(hot=hot)
    prefix, _ = _prefix(manager)
    affine = _cache()
    dense = {
        "state": (mx.zeros((1, 2, 4, 13)), mx.zeros((1, 2, 4, 21))),
        "cache_type": "KVCache",
        "class_name": "KVCache",
        "meta_state": (4,),
    }
    assert prefix.store_cache("prefill", list(range(4)), [dense]) is not None
    config = ModelCacheConfig.from_cache_list([affine])
    assert config.get_type_names() == ["Affine4KVCache"]
    assert (
        prefix.store_cache(
            "prefill" if same_request else "final",
            list(range(8)),
            [_data(affine)],
            model_cache_config=config,
            boundary_snapshots={4: [dense]},
        )
        is not None
    )
    manager.close()
    table, remaining = prefix.fetch_cache("warm", list(range(8)) + [99])
    assert table is not None and remaining == [99]
    restored = prefix.reconstruct_cache(table)
    assert restored is not None and type(restored[0]) is Affine4KVCache
    assert restored[0].offset == 8
    _assert_state_equal(_state(affine), _state(restored[0]))


def test_batch_names_canonicalize_without_collapsing_formats(manager_factory):
    assert _canonicalize_layer_cache_types(
        ["BatchAffine4KVCache", "BatchAffine8KVCache", "BatchTurboQuantKVCache"]
    ) == ["Affine4KVCache", "Affine8KVCache", "TurboQuantKVCache"]
    for name in ("BatchAffine4KVCache", "BatchAffine8KVCache"):
        assert CacheTypeRegistry.get_handler_by_class_name(name).supports_block_slicing
    for expected, stored in (
        ("Affine4KVCache", "TurboQuantKVCache"),
        ("TurboQuantKVCache", "Affine4KVCache"),
        ("Affine8KVCache", "Affine4KVCache"),
        ("Affine4KVCache", "KVCache"),
    ):
        manager = manager_factory(types=[expected])
        signature = _cache_compat_signature(
            model_name="affine-test",
            num_layers=1,
            block_size=4,
            layer_cache_types=[stored],
            turboquant_kv_bits=4,
        )
        assert not manager.is_signature_compatible(signature)


@pytest.mark.parametrize(
    "meta",
    [
        None,
        ("4", "4", "7", "turboquant", "13", "21"),
        ("4", "4", "7", "affine4", "0", "21"),
    ],
)
def test_malformed_metadata_rejects_save_and_restore(manager_factory, meta):
    cache = _cache(length=4)
    payload = _payload(cache)
    manager = manager_factory()
    assert not manager.save_block(
        b"bad-affine",
        [payload],
        4,
        layer_cache_types=["Affine4KVCache"],
        layer_meta_states=[meta],
    )
    assert _chain([payload], [meta], ["Affine4KVCache"]) is None


@pytest.mark.parametrize("mutation", ["dimensions", "short"])
def test_inconsistent_block_chain_is_rejected(mutation):
    cache = _cache()
    parts = [_payload(cache), _payload(cache, 4, 8)]
    metas = [cache.meta_state, cache.meta_state]
    if mutation == "dimensions":
        metas = [(*meta[:4], "40", meta[5]) for meta in metas]
    else:
        parts[1] = _payload(cache, 4, 7)
    assert _chain(parts, metas, ["Affine4KVCache"]) is None


@pytest.mark.parametrize("other,reverse", [("dense", False), ("tq4", True)])
def test_mixed_formats_are_rejected_without_tq_fallback(reverse, other):
    affine = _cache(length=4, key_dim=64, value_dim=64)
    if other == "dense":
        payload = (mx.zeros((1, 2, 4, 64)), mx.zeros((1, 2, 4, 64)))
        meta = ()
    else:
        tq = _cache(length=4, key_dim=64, value_dim=64, cls=TurboQuantKVCache)
        payload, meta = _payload(tq, marker="__turboquant_v2__"), tq.meta_state
    payloads, metas = [_payload(affine), payload], [affine.meta_state, meta]
    if reverse:
        payloads.reverse()
        metas.reverse()
    assert _chain(payloads, metas, ["Affine4KVCache"]) is None


@pytest.mark.parametrize(
    "mutation",
    ["missing_meta", "both_markers", "tq_relabel"],
)
def test_corrupt_ssd_payload_is_rejected(manager_factory, mutation):
    manager = manager_factory()
    cache = _cache(length=4)
    block_hash = b"corrupt-affine"
    assert manager.save_block(
        block_hash,
        [_payload(cache)],
        4,
        model_name="affine-test",
        layer_cache_types=["Affine4KVCache"],
        layer_meta_states=[cache.meta_state],
    )
    manager.close()
    path = manager._index.get(block_hash).file_path
    arrays, metadata = mx.load(str(path), return_metadata=True)
    mx.eval(arrays)
    if mutation == "missing_meta":
        del metadata["layer_meta_states"]
    elif mutation == "tq_relabel":
        del metadata["layer_0_affine4"]
        metadata["layer_0_turboquant_v2"] = "1"
    else:
        metadata["layer_0_turboquant_v2"] = "1"
    classes = ["TurboQuantKVCache" if mutation == "tq_relabel" else "Affine4KVCache"]
    mx.save_safetensors(str(path), arrays, metadata)
    arrays, metadata = mx.load(str(path), return_metadata=True)
    mx.eval(arrays)
    assert manager._reconstruct_cache_data(arrays, metadata, 1, classes) is None


def test_affine8_prefix_and_ssd_roundtrip(manager_factory):
    manager = manager_factory(types=["Affine8KVCache"])
    manager.set_expected_layer_signature(["Affine8KVCache"], turboquant_kv_bits=8)
    prefix, _ = _prefix(manager)
    cache = _cache(cls=Affine8KVCache)
    data = _data(cache)
    assert prefix.store_cache("store", list(range(8)), [data]) is not None
    manager.close()
    prefix.release_cache("store")
    table, remaining = prefix.fetch_cache("reuse", list(range(8)) + [99])
    assert table is not None and remaining == [99]
    restored = prefix.reconstruct_cache(table)
    assert restored is not None and type(restored[0]) is Affine8KVCache
    assert restored[0].meta_state == cache.meta_state
    _assert_state_equal(_state(cache), _state(restored[0]))


