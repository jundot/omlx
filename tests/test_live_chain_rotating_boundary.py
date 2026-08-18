# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the rotating/CacheList boundary snapshot source."""

from types import SimpleNamespace

from omlx.scheduler import Scheduler, _BoundarySnapshotProvider

BLOCK = 4


def _rotating_snapshot(tc):
    """A boundary snapshot as a rotating model captures it: real state for the
    non-sliceable layer, placeholders for the sliceable ones."""
    return [
        {"class_name": "RotatingKVCache", "state": (f"rk@{tc}", f"rv@{tc}")},
        {"class_name": "KVCache", "state": ()},
    ]


def _slice_sliceable(cname, k, v, end):
    """Mirror _slice_sliceable_layer's dispatch: only the two sliceable classes
    return a slice, everything else reports failure."""
    if cname in ("KVCache", "TurboQuantKVCache"):
        return f"{k}[:{end}]", f"{v}[:{end}]"
    return None, None


def _override_provider(tcs, latest_tc):
    provider = _BoundarySnapshotProvider(
        store=None,
        request_id="r1",
        valid_tcs=list(tcs),
        in_memory_snapshots={tc: _rotating_snapshot(tc) for tc in tcs},
    )
    return (
        list(range(latest_tc)),
        _rotating_snapshot(latest_tc),
        None,
        provider,
    )


def _scheduler(override, evaluated=None):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._retention_slot = None
    scheduler.config = SimpleNamespace(paged_cache_block_size=BLOCK)
    scheduler.block_aware_cache = object()
    scheduler._store_cache_executor = object()
    scheduler._boundary_snapshot_store = SimpleNamespace(
        cleanup_request=lambda _rid: None
    )
    # Rotating caches are stateful and non-sliceable, so without a snapshot
    # source the payload is refused and the ordinary store runs instead.
    scheduler._detect_boundary_snapshot_need = lambda: True
    scheduler._split_gdn_snapshot_provider = lambda _rid, _n: None
    scheduler._get_boundary_store_override = lambda _rid, _tokens: override
    scheduler._collect_arrays_from_extracted_cache = lambda cache: list(cache)
    scheduler._eval_boundary_provider_snapshots = (
        (lambda provider: evaluated.append(provider))
        if evaluated is not None
        else (lambda _provider: None)
    )
    return scheduler


def test_rotating_model_gets_a_spill_payload():
    """Without a second source, Gemma fell back to the ordinary store and lost
    the single-copy residency this path exists for."""
    scheduler = _scheduler(_override_provider(range(BLOCK, 17, BLOCK), 16))

    skip_store, pending = scheduler._build_deferred_spill_store(
        "r1", list(range(18)), [{"class_name": "KVCache"}], SimpleNamespace()
    )

    assert skip_store is True
    assert pending is not None
    assert pending["snapshots"] is not None
    # The hold keeps the boundary snapshot directory alive until the spill runs.
    assert "r1" in scheduler._retention_pending_snapshot_ids()


def test_topmost_snapshot_is_addressable_by_token_count():
    """store_cache prefers a snapshot for the terminal block too, and 34d
    adopts at the topmost boundary — both address it by token count."""
    scheduler = _scheduler(_override_provider(range(BLOCK, 13, BLOCK), 16))

    provider = scheduler._boundary_override_snapshot_provider("r1", list(range(18)))

    assert 16 in provider
    assert provider[16][0]["state"] == ("rk@16", "rv@16")


def test_no_boundary_snapshots_leaves_the_ordinary_store_alone():
    """A configuration that cannot spill must not skip the completion store."""
    scheduler = _scheduler(None)

    skip_store, pending = scheduler._build_deferred_spill_store(
        "r1", list(range(18)), [{"class_name": "KVCache"}], SimpleNamespace()
    )

    assert skip_store is False
    assert pending is None
    assert "r1" not in scheduler._retention_pending_snapshot_ids()


def test_in_memory_snapshots_are_materialized_before_the_handoff():
    """The spill worker slices these on another thread; MLX streams are
    thread-local, so a lazy leaf would re-dispatch to a stream it cannot see."""
    evaluated = []
    scheduler = _scheduler(
        _override_provider(range(BLOCK, 17, BLOCK), 16), evaluated=evaluated
    )

    _skip_store, pending = scheduler._build_deferred_spill_store(
        "r1", list(range(18)), [{"class_name": "KVCache"}], SimpleNamespace()
    )

    assert evaluated == [pending["snapshots"]]


def test_rotating_layer_no_longer_rejects_the_whole_slice():
    """_slice_sliceable_layer cannot slice a rotating layer, and failing on it
    used to reject partial adoption on every rotating model."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._slice_sliceable_layer = _slice_sliceable
    cache = [
        {"class_name": "RotatingKVCache", "state": ("rk@live", "rv@live")},
        {"class_name": "KVCache", "state": ("k", "v")},
    ]

    sliced = scheduler._slice_cache_to_boundary(cache, 8, _rotating_snapshot(8))

    assert sliced is not None
    # Passed through: the caller replaces it from the snapshot, whose state is
    # at the boundary rather than at the retained chain's own offset.
    assert sliced[0]["state"] == ("rk@live", "rv@live")
    assert sliced[1]["state"] == ("k[:8]", "v[:8]")


def test_merged_rotating_layer_comes_from_the_snapshot():
    """The live rotating state sits at the retained prompt boundary, so the
    adopted chain must take the snapshot's state for that layer."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._slice_sliceable_layer = _slice_sliceable
    snapshot = _rotating_snapshot(8)
    cache = [
        {"class_name": "RotatingKVCache", "state": ("rk@live", "rv@live")},
        {"class_name": "KVCache", "state": ("k", "v")},
    ]

    sliced = scheduler._slice_cache_to_boundary(cache, 8, snapshot)
    merged = scheduler._merge_boundary_with_full_cache(snapshot, sliced)

    assert merged[0]["state"] == ("rk@8", "rv@8")
    assert merged[1]["state"] == ("k[:8]", "v[:8]")


def test_member_filtered_cachelist_still_refuses_to_slice():
    """A blanked CacheList member has to be refilled from the live chain, and
    slicing it member-wise is not implemented — keep bailing out."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._slice_sliceable_layer = _slice_sliceable
    snapshot = [{"class_name": "CacheList", "state": [["gdn"], ()]}]
    cache = [{"class_name": "CacheList", "state": (("k", "v"), ("k2", "v2"))}]

    assert scheduler._slice_cache_to_boundary(cache, 8, snapshot) is None
