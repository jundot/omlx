# SPDX-License-Identifier: Apache-2.0
"""The GDN state codec applied to state embedded in cache blocks.

The SSD-sidecar layout has had this codec since #2644. These tests cover the
other layout — the one used when ``gdn_snapshot_storage`` resolves to
``embedded``, which is what hot-cache-only and SSD-disabled deployments run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")

from omlx.cache import gdn_state_codec as gdn_codec  # noqa: E402
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager  # noqa: E402

LAYER_TYPES = ["KVCache", "ArraysCache"]


def _recurrent(scale: float = 1.0) -> mx.array:
    """A GDN-shaped recurrent state with a power-of-two last axis."""
    base = mx.arange(1, 1 + 2 * 4 * 16, dtype=mx.float32).reshape(1, 2, 4, 16)
    return (base - 64.5) * 0.125 * scale


def _kv_layer() -> tuple:
    return (
        mx.zeros((1, 2, 8, 4), dtype=mx.float32),
        mx.ones((1, 2, 8, 4), dtype=mx.float32),
    )


def _gdn_layer(scale: float = 1.0) -> tuple:
    """The shape prefix_cache stores for an ArraysCache layer: (conv, state)."""
    return (mx.zeros((1, 1, 8), dtype=mx.float32), _recurrent(scale))


def _placeholder_layer() -> tuple:
    return (mx.zeros((1,)), mx.zeros((1,)))


def _manager(tmp_path: Path, dtype: str, **kwargs) -> PagedSSDCacheManager:
    return PagedSSDCacheManager(
        cache_dir=tmp_path / "ssd_cache",
        max_size_bytes=1024**3,
        expected_model_name="model",
        expected_num_layers=2,
        expected_block_size=64,
        expected_layer_cache_types=LAYER_TYPES,
        gdn_snapshot_state_dtype=dtype,
        **kwargs,
    )


def _save(manager: PagedSSDCacheManager, block_hash: bytes, cache_data) -> bool:
    return manager.save_block(
        block_hash=block_hash,
        cache_data=cache_data,
        token_count=64,
        model_name="model",
        layer_cache_types=LAYER_TYPES,
    )


def _wait_for_block_file(manager: PagedSSDCacheManager, block_hash: bytes) -> Path:
    """Block until the background SSD writer has produced the block file."""
    file_path = manager._get_file_path(block_hash)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not file_path.exists():
        time.sleep(0.01)
    assert file_path.exists(), "background writer never produced the block file"
    return file_path


def _block_header(manager: PagedSSDCacheManager, block_hash: bytes) -> dict:
    """Read the safetensors header the background writer produced."""
    with open(_wait_for_block_file(manager, block_hash), "rb") as handle:
        header_len = int.from_bytes(handle.read(8), "little")
        return json.loads(handle.read(header_len))


def _rel_error(restored, source) -> float:
    a = np.asarray(restored, dtype=np.float64)
    b = np.asarray(source, dtype=np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize(
    ("dtype", "stored_dtype", "max_rel_error"),
    [
        ("bf16", "BF16", 5e-3),
        ("int8", "I8", 5e-2),
        ("rht_int8", "I8", 5e-2),
        ("rht_int16", "I16", 1e-4),
    ],
)
def test_embedded_state_round_trips_through_every_codec(
    tmp_path, dtype, stored_dtype, max_rel_error
):
    manager = _manager(tmp_path, dtype)
    try:
        source = _recurrent()
        assert _save(manager, b"h" * 20, [_kv_layer(), _gdn_layer()])

        # Stored narrow...
        header = _block_header(manager, b"h" * 20)
        assert header["layer_1_state_1__q"]["dtype"] == stored_dtype

        # ...restored as fp32 in the original basis.
        loaded = manager.load_block(b"h" * 20)
        assert loaded is not None
        restored = loaded[1][1]
        assert restored.dtype == mx.float32
        assert restored.shape == source.shape
        assert _rel_error(restored, source) < max_rel_error
        assert manager.gdn_state_encodes == 1
        assert manager.gdn_state_dequantizations == 1
        assert manager.gdn_state_decode_failures == 0
    finally:
        manager.close()


def test_fp32_blocks_are_byte_identical_to_the_pre_codec_format(tmp_path):
    """The default must not rewrite what existing caches already contain."""
    manager = _manager(tmp_path, "fp32")
    try:
        assert _save(manager, b"h" * 20, [_kv_layer(), _gdn_layer()])
        header = _block_header(manager, b"h" * 20)
        assert "layer_1_state_1" in header
        assert not [name for name in header if name.endswith("__q")]
        assert manager.gdn_state_encodes == 0
    finally:
        manager.close()


def test_encoded_element_omits_the_flat_key_an_older_reader_would_read(tmp_path):
    """Older-reader safety.

    An oMLX that predates this codec has no idea what ``_storage_codec``
    means, and would read an int16 payload under ``layer_1_state_1`` as fp32.
    So that key is not written at all: the old reader hits its existing
    ``Missing ... in arrays`` path, treats the block as a miss and re-prefills.
    """
    manager = _manager(tmp_path, "rht_int16")
    try:
        assert _save(manager, b"h" * 20, [_kv_layer(), _gdn_layer()])
        header = _block_header(manager, b"h" * 20)

        assert "layer_1_state_1" not in header
        assert "layer_1_state_1__q" in header
        assert "layer_1_state_1__scale" in header

        metadata = header["__metadata__"]
        assert metadata["layer_1_state_1_storage_codec"] == (
            gdn_codec.RHT_INT16_CODEC
        )
        assert metadata["layer_1_state_1_original_dtype"] == "float32"
        assert metadata["layer_1_state_1_rht_dim"] == "16"

        # An older reader ignores that metadata entirely; emulate it.
        arrays = {
            name: mx.zeros(tuple(spec["shape"]))
            for name, spec in header.items()
            if name != "__metadata__"
        }
        legacy_metadata = {
            key: value
            for key, value in metadata.items()
            if "_storage_codec" not in key
        }
        assert (
            manager._reconstruct_cache_data(arrays, legacy_metadata, 2, LAYER_TYPES)
            is None
        )
    finally:
        manager.close()


def test_codec_blocks_keep_the_same_cache_signature_as_fp32_blocks(tmp_path):
    """Mixed-codec directories stay valid; nothing already stored is orphaned.

    The block describes its own codec per element, so a precision change does
    not have to bump the signature — which would invalidate every KV block in
    the cache for a setting that only concerns one layer family.
    """
    fp32 = _manager(tmp_path / "a", "fp32")
    rht = _manager(tmp_path / "b", "rht_int16")
    try:
        kwargs = dict(
            model_name="model",
            num_layers=2,
            block_size=64,
            layer_cache_types=LAYER_TYPES,
        )
        assert fp32.cache_signature_for(**kwargs) == rht.cache_signature_for(**kwargs)
    finally:
        fp32.close()
        rht.close()


def test_a_reader_on_a_different_codec_still_reads_existing_blocks(tmp_path):
    """One cache directory, blocks written under two settings."""
    writer = _manager(tmp_path, "rht_int16")
    try:
        assert _save(writer, b"a" * 20, [_kv_layer(), _gdn_layer()])
    finally:
        writer.close()

    reader = _manager(tmp_path, "int8")
    try:
        assert _save(reader, b"b" * 20, [_kv_layer(), _gdn_layer(scale=2.0)])
        first = reader.load_block(b"a" * 20)
        second = reader.load_block(b"b" * 20)
        assert first is not None and second is not None
        assert _rel_error(first[1][1], _recurrent()) < 1e-4
        assert _rel_error(second[1][1], _recurrent(scale=2.0)) < 5e-2
    finally:
        reader.close()


def test_placeholder_blocks_are_not_encoded(tmp_path):
    """Every block but the last carries a rank-1 placeholder, not real state."""
    manager = _manager(tmp_path, "rht_int16")
    try:
        assert _save(manager, b"h" * 20, [_kv_layer(), _placeholder_layer()])
        header = _block_header(manager, b"h" * 20)
        assert "layer_1_state_1" in header
        assert "layer_1_state_1__q" not in header
        assert manager.gdn_state_encodes == 0

        # And the placeholder still reads back as a placeholder, so partial
        # prefix matches are still detected and rejected upstream.
        loaded = manager.load_block(b"h" * 20)
        assert loaded is not None
        assert loaded[1][0].shape == (1,)
    finally:
        manager.close()


def test_unsupported_width_falls_back_to_raw_storage(tmp_path):
    """A non-power-of-two state is stored raw rather than failing the block."""
    manager = _manager(tmp_path, "rht_int16")
    try:
        odd = mx.ones((1, 2, 4, 12), dtype=mx.float32)
        assert _save(
            manager,
            b"h" * 20,
            [_kv_layer(), (mx.zeros((1, 1, 8), dtype=mx.float32), odd)],
        )
        header = _block_header(manager, b"h" * 20)
        assert header["layer_1_state_1"]["dtype"] == "F32"
        assert manager.gdn_capability_fallbacks == 1

        loaded = manager.load_block(b"h" * 20)
        assert loaded is not None
        assert np.array_equal(np.asarray(loaded[1][1]), np.asarray(odd))
    finally:
        manager.close()


def test_non_finite_state_is_refused_rather_than_stored(tmp_path):
    manager = _manager(tmp_path, "rht_int16")
    try:
        broken = _recurrent()
        broken[0, 0, 0, 0] = float("nan")
        assert (
            _save(manager, b"h" * 20, [_kv_layer(), (mx.zeros((1, 1, 8)), broken)])
            is False
        )
        assert manager.gdn_state_encode_failures == 1
        assert manager.load_block(b"h" * 20) is None
    finally:
        manager.close()


@pytest.mark.parametrize(
    "corruption",
    [
        {"layer_1_state_1_storage_codec": "made_up_codec_v9"},
        {"layer_1_state_1_original_dtype": "bfloat16"},
        {"layer_1_state_1_rht_dim": "32"},
        {"layer_1_state_1_rht_seed": "7"},
    ],
)
def test_corrupt_codec_metadata_fails_closed(tmp_path, corruption):
    manager = _manager(tmp_path, "rht_int16")
    try:
        assert _save(manager, b"h" * 20, [_kv_layer(), _gdn_layer()])
        header = _block_header(manager, b"h" * 20)
        arrays = {
            name: mx.zeros(
                tuple(spec["shape"]),
                dtype=mx.int16 if spec["dtype"] == "I16" else mx.float32,
            )
            for name, spec in header.items()
            if name != "__metadata__"
        }
        metadata = dict(header["__metadata__"])
        metadata.update(corruption)
        assert (
            manager._reconstruct_cache_data(arrays, metadata, 2, LAYER_TYPES) is None
        )
    finally:
        manager.close()


def test_hot_cache_only_blocks_carry_the_encoded_payload(tmp_path):
    """The hot tier is where an embedded-layout user pays for fp32 state."""
    fp32 = _manager(tmp_path / "a", "fp32", hot_cache_max_bytes=1024**3)
    rht = _manager(tmp_path / "b", "rht_int16", hot_cache_max_bytes=1024**3)
    try:
        for manager in (fp32, rht):
            assert manager.save_block(
                block_hash=b"h" * 20,
                cache_data=[_kv_layer(), _gdn_layer()],
                token_count=64,
                model_name="model",
                layer_cache_types=LAYER_TYPES,
                hot_cache_write_back=True,
            )
        # Same block, and the recurrent half of it is half the size.
        assert rht._hot_cache_total_bytes < fp32._hot_cache_total_bytes

        loaded = rht.load_block(b"h" * 20)
        assert loaded is not None
        assert rht._stats["hot_cache_hits"] == 1
        assert _rel_error(loaded[1][1], _recurrent()) < 1e-4
    finally:
        fp32.close()
        rht.close()


def test_a_block_costs_one_validation_sync_however_many_layers(tmp_path):
    """The other half of the no-amplification contract.

    The scale-validity check reads a tensor, so testing it per element cost a
    GPU->CPU sync per layer per block — 22.5 ms for a 48-layer block against
    3.9 ms for the same block in fp32. Deferring the flags and evaluating them
    once per block brought that to 6.4 ms. This asserts the shape of that fix:
    one verification call per block, carrying every layer's flag.
    """
    manager = _manager(tmp_path, "rht_int16")
    calls: list[int] = []
    original = gdn_codec.verify_decode_checks

    def counting_verify(checks):
        calls.append(len(checks))
        return original(checks)

    try:
        layer_types = ["ArraysCache"] * 6
        cache_data = [_gdn_layer() for _ in range(6)]
        assert manager.save_block(
            block_hash=b"h" * 20,
            cache_data=cache_data,
            token_count=64,
            model_name="model",
            layer_cache_types=layer_types,
        )
        _wait_for_block_file(manager, b"h" * 20)
        manager.clear_hot_cache()

        monkeypatched = pytest.MonkeyPatch()
        monkeypatched.setattr(gdn_codec, "verify_decode_checks", counting_verify)
        try:
            loaded = manager.load_block(b"h" * 20)
        finally:
            monkeypatched.undo()

        assert loaded is not None
        assert calls == [6]
    finally:
        manager.close()


def test_restoring_a_chain_does_not_decode_every_block(tmp_path):
    """The maintainer's condition: no decode amplification over a chain.

    A chain restore reconstructs every matched block, but only one block's
    recurrent state is ever adopted. The decode returns an unevaluated MLX
    graph, so the blocks whose state is discarded are never computed. Measured
    by MLX's active memory: decoding all of them would materialize one fp32
    state per block.
    """
    manager = _manager(tmp_path, "rht_int16")
    try:
        hashes = [bytes([i]) * 20 for i in range(1, 13)]
        for block_hash in hashes:
            assert _save(manager, block_hash, [_kv_layer(), _gdn_layer()])

        state_bytes = _recurrent().nbytes
        mx.clear_cache()
        before = mx.get_active_memory()
        chain = [manager.load_block(block_hash) for block_hash in hashes]
        assert all(block is not None for block in chain)

        # Nothing forced the decode yet.
        undecoded = mx.get_active_memory() - before
        assert undecoded < state_bytes

        # Adopting one block's state is what materializes it, exactly once.
        adopted = chain[-1][1][1]
        mx.eval(adopted)
        assert _rel_error(adopted, _recurrent()) < 1e-4
    finally:
        manager.close()
