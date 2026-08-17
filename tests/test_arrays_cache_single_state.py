# SPDX-License-Identifier: Apache-2.0
"""Regression tests for single-state ArraysCache (state_count == 1).

LFM2.x liquid-style hybrid attention + short-conv architectures expose a
single recurrent state per layer (no conv/values pair). The oMLX prefix
cache previously assumed ``len(state) >= 2`` everywhere it walked an
ArraysCache boundary snapshot, so a snapshot produced by an LFM2.x model
was rejected at the GDN-sidecar validation step and replaced with a
placeholder by the non-sliceable extract path.

These tests pin the round-trip:
    BoundarySnapshotSSDStore.save()
        -> GDN-sidecar V3 metadata ``state_count == 1``
        -> BlockAwarePrefixCache._extract_block_tensor_slice() (non-last block)
        -> BlockAwarePrefixCache._validated_gdn_snapshot_layers() (last block)
        -> BlockAwarePrefixCache.reconstruct_cache()
        -> SizedArraysCache wrapping a 1-element ArraysCache

They also pin the existing 2-state (Mamba) and 3-state (N-tuple) round
trips as backward-compatibility guards.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from omlx.cache.boundary_snapshot_store import BoundarySnapshotSSDStore
from omlx.cache.paged_cache import PagedCacheManager
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


class _MockLayer(MagicMock):
    """Placeholder layer carrying no state."""


class _MockModel:
    """Minimal model stub for BlockAwarePrefixCache."""

    def __init__(self, num_layers: int = 1):
        self.layers = [_MockLayer() for _ in range(num_layers)]

    @property
    def args(self):
        mock_args = MagicMock()
        mock_args.num_hidden_layers = len(self.layers)
        return mock_args


def _single_state(value: float = 1.0) -> tuple[mx.array]:
    """LFM2.x-style ArraysCache payload (one recurrent tensor)."""
    arr = mx.full((1, 8, 16), value, dtype=mx.float32)
    mx.eval(arr)
    return (arr,)


def _two_state(value: float = 1.0) -> tuple[mx.array, mx.array]:
    """Legacy Mamba/SSM pair (conv + recurrent)."""
    conv = mx.full((1, 3, 16), value, dtype=mx.float32)
    ssm = mx.full((1, 4, 8, 8), value * 2, dtype=mx.float32)
    mx.eval(conv, ssm)
    return (conv, ssm)


def _three_state(value: float = 1.0) -> tuple[mx.array, mx.array, mx.array]:
    """Generic N-tuple state."""
    a = mx.full((1, 4), value, dtype=mx.float32)
    b = mx.full((1, 4), value * 2, dtype=mx.float32)
    c = mx.full((1, 4), value * 3, dtype=mx.float32)
    mx.eval(a, b, c)
    return (a, b, c)


def _extracted(
    state: tuple[mx.array, ...],
    cache_type: str = "ArraysCache",
    class_name: str = "ArraysCache",
) -> list[dict[str, Any]]:
    """Wrap a state tuple in the shape _extract_cache_states() emits."""
    return [
        {
            "state": state,
            "meta_state": (),
            "class_name": class_name,
            "cache_type": cache_type,
        }
    ]


def _extract_fn(extracted: list[dict[str, Any]]):
    """Build the snapshot extractor callback BoundarySnapshotSSDStore.save() expects."""
    return lambda _snapshot, extracted=extracted: (extracted, None)


@pytest.fixture
def mx_available():
    if not hasattr(mx, "array"):
        pytest.skip("MLX not available")
    return mx


# ---------------------------------------------------------------------------
# _validated_gdn_snapshot_layers: structural validation for GDN split layout
# ---------------------------------------------------------------------------


class TestValidatedGdnSnapshotLayers:
    """The GDN split-layout validator must accept state_count == 1."""

    def _validator(self):
        from omlx.cache.type_registry import CacheTypeRegistry

        return BlockAwarePrefixCache._validated_gdn_snapshot_layers

    def test_single_state_arrays_cache_accepted(self):
        """state_count == 1 must produce an __nstate__ marker (not a None rejection)."""
        validator = self._validator()
        snapshot = [
            {
                "state": _single_state(value=7.0),
                "meta_state": (),
                "class_name": "ArraysCache",
                "cache_type": "ArraysCache",
            }
        ]
        layer_types = ["ArraysCache"]

        payloads = validator(snapshot, layer_types)

        assert payloads is not None, "single-state ArraysCache must be accepted"
        assert 0 in payloads
        cache_data, meta_state = payloads[0]
        # The 1-tuple cannot be a plain (k, v) pair: there is no conv_state,
        # so it must surface as an N-state element marker that reconstruction
        # can consume.
        assert isinstance(cache_data, tuple)
        assert cache_data[0] == "__nstate__"
        assert cache_data[1] == "ArraysCache"
        assert isinstance(cache_data[2], list)
        assert len(cache_data[2]) == 1
        assert meta_state == ()

    def test_two_state_arrays_cache_still_uses_pair_format(self):
        """Regression guard: 2-tuple Mamba shape must keep the (k, v) path."""
        validator = self._validator()
        snapshot = [
            {
                "state": _two_state(value=3.0),
                "meta_state": (),
                "class_name": "ArraysCache",
                "cache_type": "ArraysCache",
            }
        ]
        layer_types = ["ArraysCache"]

        payloads = validator(snapshot, layer_types)

        assert payloads is not None
        cache_data, _ = payloads[0]
        # Existing 2-tuple shape must remain unpacked (not wrapped in __nstate__).
        assert not isinstance(cache_data, tuple) or cache_data[0] != "__nstate__"
        assert len(cache_data) == 2

    def test_three_state_arrays_cache_uses_marker(self):
        """Regression guard: > 2 elements stay on the __nstate__ path."""
        validator = self._validator()
        snapshot = [
            {
                "state": _three_state(value=2.0),
                "meta_state": (),
                "class_name": "ArraysCache",
                "cache_type": "ArraysCache",
            }
        ]
        layer_types = ["ArraysCache"]

        payloads = validator(snapshot, layer_types)

        assert payloads is not None
        cache_data, _ = payloads[0]
        assert cache_data[0] == "__nstate__"
        assert len(cache_data[2]) == 3

    def test_empty_state_still_rejected(self):
        """Empty state tuples must continue to be rejected (no false positives)."""
        validator = self._validator()
        snapshot = [
            {
                "state": (),
                "meta_state": (),
                "class_name": "ArraysCache",
                "cache_type": "ArraysCache",
            }
        ]
        layer_types = ["ArraysCache"]

        assert validator(snapshot, layer_types) is None

    def test_non_arrays_family_layer_is_skipped(self):
        """Non-Arrays layers in the type list must not be inspected for state."""
        validator = self._validator()
        snapshot = [
            {
                "state": (mx.zeros((1,)),),
                "meta_state": (),
                "class_name": "KVCache",
                "cache_type": "KVCache",
            }
        ]
        layer_types = ["KVCache"]

        payloads = validator(snapshot, layer_types)
        assert payloads == {}


# ---------------------------------------------------------------------------
# _extract_block_tensor_slice: non-sliceable extract path for the last block
# ---------------------------------------------------------------------------


class TestExtractBlockTensorSliceSingleState:
    """Non-last and last block extraction for single-state ArraysCache."""

    @pytest.fixture
    def prefix_cache(self, tmp_path):
        paged = PagedCacheManager(
            block_size=4,
            max_blocks=8,
            model_name="lfm2-test",
            initial_blocks=8,
        )
        return BlockAwarePrefixCache(
            model=_MockModel(num_layers=1),
            paged_cache_manager=paged,
        )

    def test_last_block_routes_through_nstate_marker(self, prefix_cache, mx_available):
        """Single-state ArraysCache on the last block must surface as
        ``("__nstate__", "ArraysCache", [tensor])`` so reconstruction can
        rebuild a SizedArraysCache of size 1."""
        state = _single_state(value=11.0)
        cache_data = [
            {
                "state": state,
                "cache_type": "ArraysCache",
                "class_name": "ArraysCache",
            }
        ]

        result = prefix_cache._extract_block_tensor_slice(
            cache_data,
            0,
            4,
            model_cache_config=None,
            is_last_block=True,
        )

        assert result is not None
        assert len(result) == 1
        marker = result[0]
        assert isinstance(marker, tuple)
        assert marker[0] == "__nstate__"
        assert marker[1] == "ArraysCache"
        elements = marker[2]
        assert len(elements) == 1
        assert elements[0].shape == (1, 8, 16)

    def test_non_last_block_with_snapshot_uses_snapshot_state(
        self, prefix_cache, mx_available
    ):
        """When a boundary snapshot is available, non-last blocks carry the
        snapshot's full state, not a placeholder."""
        snapshot_state = _single_state(value=5.0)
        layer_state = _single_state(value=5.0)
        cache_data = [
            {
                "state": layer_state,
                "cache_type": "ArraysCache",
                "class_name": "ArraysCache",
            }
        ]
        snapshot_cache_data = [
            {
                "state": snapshot_state,
                "cache_type": "ArraysCache",
                "class_name": "ArraysCache",
            }
        ]

        result = prefix_cache._extract_block_tensor_slice(
            cache_data,
            0,
            4,
            model_cache_config=None,
            is_last_block=False,
            snapshot_cache_data=snapshot_cache_data,
        )

        assert result is not None
        assert len(result) == 1
        marker = result[0]
        assert marker[0] == "__nstate__"
        assert len(marker[2]) == 1

    def test_two_state_unaffected(self, prefix_cache, mx_available):
        """Regression guard: the legacy 2-tuple Mamba path still emits a
        plain ``(conv, ssm)`` pair, not an N-state marker."""
        cache_data = [
            {
                "state": _two_state(value=4.0),
                "cache_type": "ArraysCache",
                "class_name": "ArraysCache",
            }
        ]

        result = prefix_cache._extract_block_tensor_slice(
            cache_data,
            0,
            4,
            model_cache_config=None,
            is_last_block=True,
        )

        assert result is not None
        keys, values = result[0]
        # The 2-tuple path produces a plain (k, v), not a marker.
        assert not (isinstance(keys, tuple) and len(keys) == 3 and keys[0] == "__nstate__")
        assert keys.shape == (1, 3, 16)
        assert values.shape == (1, 4, 8, 8)

    def test_three_state_unaffected(self, prefix_cache, mx_available):
        """Regression guard: 3-element N-tuple still produces the marker."""
        cache_data = [
            {
                "state": _three_state(value=2.0),
                "cache_type": "ArraysCache",
                "class_name": "ArraysCache",
            }
        ]

        result = prefix_cache._extract_block_tensor_slice(
            cache_data,
            0,
            4,
            model_cache_config=None,
            is_last_block=True,
        )

        assert result is not None
        marker = result[0]
        assert marker[0] == "__nstate__"
        assert len(marker[2]) == 3


# ---------------------------------------------------------------------------
# End-to-end round-trip through BoundarySnapshotSSDStore + reconstruct_cache
# ---------------------------------------------------------------------------


class TestSingleStateEndToEnd:
    """Full save -> load -> reconstruct_cache round-trip for state_count == 1."""

    @pytest.fixture
    def store(self, tmp_path):
        base = tmp_path / "ssd_cache"
        base.mkdir()
        s = BoundarySnapshotSSDStore(base_dir=base)
        yield s
        s.shutdown()

    def test_boundary_snapshot_serializes_single_state(self, store):
        """V3 metadata must record ``state_count == 1`` for LFM2.x layers."""
        extracted = _extracted(_single_state(value=9.0))

        ok = store.save("req-lfm2", 1024, [MagicMock()], _extract_fn(extracted))
        assert ok

        # Pending buffer gives us the raw bytes without touching disk.
        with store._pending_lock:
            pending = store._pending_writes.get(("req-lfm2", 1024))
        assert pending is not None
        metadata = pending["metadata"]
        import json

        layer_info = json.loads(metadata["layer_info"])
        assert layer_info[0]["has_state"] == "true"
        assert layer_info[0]["state_count"] == "1"

    def test_load_recovers_one_element_state_tuple(self, store):
        """load() must reconstruct the original 1-element state tuple."""
        extracted = _extracted(_single_state(value=9.0))
        store.save("req-lfm2", 1024, [MagicMock()], _extract_fn(extracted))

        loaded = store.load("req-lfm2", 1024)
        assert loaded is not None
        assert len(loaded) == 1
        state = loaded[0]["state"]
        # V3 round-trip must keep exactly one element (the recurrent member).
        assert isinstance(state, tuple)
        assert len(state) == 1
        assert state[0].shape == (1, 8, 16)
        assert mx.array_equal(state[0], _single_state(value=9.0)[0]).item()

    def test_reconstruct_cache_builds_sized_arrays_cache_of_size_one(
        self, tmp_path, store
    ):
        """Through reconstruct_cache(), a single-state marker produces a
        ``SizedArraysCache`` wrapping an ``ArraysCache(size=1)`` whose
        single state slot equals the original recurrent tensor.

        Avoids importing ``omlx.scheduler`` (which depends on a missing
        ``mlx_vlm.speculative`` module in this dev env) by using a
        minimal provider duck-typed against ``_BoundarySnapshotProvider``.
        """
        from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
        from omlx.cache.type_handlers import SizedArraysCache

        cache_dir = tmp_path / "ssd_cache"
        ssd = PagedSSDCacheManager(
            cache_dir=cache_dir,
            max_size_bytes=16 * 1024**2,
            expected_model_name="lfm2-test",
            expected_num_layers=1,
            expected_block_size=4,
            expected_layer_cache_types=["ArraysCache"],
        )
        paged = PagedCacheManager(
            block_size=4,
            max_blocks=8,
            model_name="lfm2-test",
            initial_blocks=8,
        )
        paged.set_paged_ssd_cache_manager(ssd)
        prefix_cache = BlockAwarePrefixCache(
            model=_MockModel(num_layers=1),
            paged_cache_manager=paged,
            paged_ssd_cache_manager=ssd,
        )

        class _Provider:
            """Minimal duck-typed boundary-snapshot provider.

            Mirrors the ``_BoundarySnapshotProvider`` interface used by
            ``BlockAwarePrefixCache.store_cache`` without importing
            ``omlx.scheduler``.
            """

            def __init__(self, store, request_id, tcs, in_memory, paged_ssd):
                self._store = store
                self._request_id = request_id
                self._valid_tcs = set(tcs)
                self._in_memory = in_memory
                self._paged_ssd_manager = paged_ssd

            def __contains__(self, tc):
                return tc in self._valid_tcs

            def __getitem__(self, tc):
                snap = self._in_memory.get(tc)
                if snap is not None:
                    return snap
                if self._store is not None:
                    return self._store.load(self._request_id, tc)
                return None

            def __len__(self):
                return len(self._valid_tcs)

            def __bool__(self):
                return bool(self._valid_tcs)

            def iter_in_memory_extracted(self):
                for tc in sorted(self._valid_tcs):
                    snap = self._in_memory.get(tc)
                    if snap is not None:
                        yield snap

            def commit_gdn_checkpoint(self, *args, **kwargs):
                return False

        try:
            original = _single_state(value=13.0)
            extracted = _extracted(original)

            # Stage the boundary snapshot through the store at the only
            # relevant boundary (4 tokens == 1 block of size 4).
            request_id = "req-lfm2"
            assert store.save(
                request_id, 4, [MagicMock()], _extract_fn(extracted)
            )

            provider = _Provider(store, request_id, [4], {}, ssd)

            # Drive the store path: prefix_cache.store_cache() extracts block
            # tensor slices using the snapshot data the provider returns.
            stored = prefix_cache.store_cache(
                request_id,
                list(range(4)),
                extracted,
                boundary_snapshots=provider,
                hot_cache_write_back=False,
            )
            assert stored is not None
            assert stored.num_tokens == 4

            # Round-trip back through reconstruct_cache: lookup by tokens
            # yields a hit, then reconstruction rebuilds the cache objects.
            hit_table, remaining = prefix_cache.fetch_cache(
                "restore-lfm2", list(range(4))
            )
            assert hit_table is not None
            assert remaining == []
            assert hit_table.num_tokens == 4

            restored = prefix_cache.reconstruct_cache(hit_table)
            assert restored is not None
            assert len(restored) == 1
            cache = restored[0]
            assert isinstance(cache, SizedArraysCache)
            assert cache.size() == 4
            assert len(cache.state) == 1
            assert mx.array_equal(cache.state[0], original[0]).item()

            prefix_cache.release_cache("restore-lfm2")
        finally:
            ssd.close()

    def test_two_state_roundtrip_still_works(self, store):
        """Regression guard: 2-state Mamba round-trip must keep working."""
        extracted = _extracted(_two_state(value=6.0))
        store.save("req-mamba", 4, [MagicMock()], _extract_fn(extracted))
        loaded = store.load("req-mamba", 4)
        assert loaded is not None
        state = loaded[0]["state"]
        assert isinstance(state, tuple)
        assert len(state) == 2
        # Sanity on shape: conv (1, 3, 16) and ssm (1, 4, 8, 8).
        assert state[0].shape == (1, 3, 16)
        assert state[1].shape == (1, 4, 8, 8)