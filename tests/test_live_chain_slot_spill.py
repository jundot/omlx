# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the spill of a contended retention slot."""

from concurrent.futures import Future
from types import SimpleNamespace

from omlx.scheduler import Scheduler


class _RecordingExecutor:
    def __init__(self):
        self.submissions = []

    def submit(self, fn, *args):
        self.submissions.append(args)
        future = Future()
        future.set_result(None)
        return future


class _RecordingSnapshotStore:
    def __init__(self):
        self.cleaned = []

    def cleanup_request(self, request_id):
        self.cleaned.append(request_id)


_SENTINEL_PROVIDER = object()


def _scheduler(provider=_SENTINEL_PROVIDER, snapshot_need=True):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._retention_slot = None
    scheduler.config = SimpleNamespace(paged_cache_block_size=4)
    scheduler.block_aware_cache = object()
    scheduler._store_cache_executor = _RecordingExecutor()
    scheduler._boundary_snapshot_store = _RecordingSnapshotStore()
    scheduler._memory_hard_limit_bytes = 0
    scheduler.memory_monitor = None
    scheduler._split_gdn_snapshot_provider = lambda _rid, _n: provider
    scheduler._detect_boundary_snapshot_need = lambda: snapshot_need
    scheduler._collect_arrays_from_extracted_cache = lambda _cache: []
    scheduler._bypass_hot_cache_under_pressure = lambda: False
    return scheduler


def _request(request_id):
    return SimpleNamespace(
        request_id=request_id,
        _extracted_cache=[{"state": ()}],
        _model_cache_config=None,
    )


def test_contended_slot_is_stored_instead_of_dropped():
    """A second conversation taking the slot must not erase the first one."""
    scheduler = _scheduler()
    first = _request("r1")
    scheduler._install_retention_slot(
        "r1", first, list(range(10)), [{"class_name": "KVCache"}], "generic"
    )
    assert first._retention_chain_retained is True
    assert scheduler._store_cache_executor.submissions == []

    scheduler._install_retention_slot(
        "r2",
        _request("r2"),
        list(range(20, 30)),
        [{"class_name": "KVCache"}],
        "generic",
    )

    (submission,) = scheduler._store_cache_executor.submissions
    request_id, tokens, cache, _config, snapshots = submission[:5]
    assert request_id == "r1"
    # SSD write-through: a salvaged chain must not evict the hot-cache blocks
    # of the conversations that are still active.
    assert submission[-1] is False
    # Truncated to the block boundary; the trailing partial block is not
    # storable and the ordinary store drops it too.
    assert tokens == list(range(8))
    assert cache == [{"class_name": "KVCache"}]
    assert snapshots is not None
    assert scheduler._retention_slot["request_id"] == "r2"


def test_adopted_slot_is_not_stored(monkeypatch):
    """The sequential path still pays nothing: adoption stores nothing.

    Payload inheritance changed where the payload goes — it is handed to the adopting
    request rather than released — but the cost claim is unchanged: no store
    is submitted and no snapshot directory is reclaimed on the prepare path.
    """
    monkeypatch.setenv("OMLX_RETAIN_LIVE_CHAIN", "1")
    scheduler = _scheduler()
    scheduler._install_retention_slot(
        "r1", _request("r1"), list(range(10)), [{"class_name": "KVCache"}], "generic"
    )
    scheduler._construct_retained_layer = lambda _layer: object()

    request = SimpleNamespace(
        request_id="r2",
        prompt_token_ids=list(range(10)) + [99],
        vlm_extra_keys_for_cache=None,
    )
    assert scheduler._try_adopt_retained_chain(request) is True
    assert scheduler._store_cache_executor.submissions == []
    assert scheduler._boundary_snapshot_store.cleaned == []
    # The payload went to the adopting request, so the hold is still live.
    assert request._retention_inherited_store["request_id"] == "r1"
    scheduler._drain_spill_futures()
    assert scheduler._boundary_snapshot_store.cleaned == []


def test_an_unused_inheritance_is_released(monkeypatch):
    """A request that adopts but never retains must not pin the snapshots."""
    monkeypatch.setenv("OMLX_RETAIN_LIVE_CHAIN", "1")
    scheduler = _scheduler()
    scheduler._install_retention_slot(
        "r1", _request("r1"), list(range(10)), [{"class_name": "KVCache"}], "generic"
    )
    scheduler._construct_retained_layer = lambda _layer: object()
    request = SimpleNamespace(
        request_id="r2",
        prompt_token_ids=list(range(10)) + [99],
        vlm_extra_keys_for_cache=None,
    )
    assert scheduler._try_adopt_retained_chain(request) is True

    scheduler._release_inherited_payload(request)
    scheduler._drain_spill_futures()
    assert scheduler._boundary_snapshot_store.cleaned == ["r1"]
    assert request._retention_inherited_store is None


def test_unspillable_chain_leaves_the_completion_store_alone():
    """Without per-block snapshots the chain cannot be stored later, so the
    ordinary completion-time store must still run."""
    scheduler = _scheduler(provider=None, snapshot_need=True)
    request = _request("r1")
    scheduler._install_retention_slot(
        "r1", request, list(range(10)), [{"class_name": "ArraysCache"}], "generic"
    )
    assert getattr(request, "_retention_chain_retained", False) is False
    assert request._extracted_cache is not None
    assert scheduler._retention_slot["pending_store"] is None


def test_sub_block_chain_keeps_the_pre_34c_skip():
    """Shorter than one block: the ordinary store keeps nothing either."""
    scheduler = _scheduler()
    request = _request("r1")
    scheduler._install_retention_slot(
        "r1", request, [1, 2, 3], [{"class_name": "KVCache"}], "generic"
    )
    assert request._retention_chain_retained is True
    assert request._extracted_cache is None
    assert scheduler._retention_slot["pending_store"] is None


def test_snapshot_cleanup_waits_for_the_spill_worker():
    """The spilled payload reads per-block snapshots off disk, so the rmtree
    that _cleanup_finished would run has to wait for the worker."""
    scheduler = _scheduler()
    scheduler._install_retention_slot(
        "r1", _request("r1"), list(range(10)), [{"class_name": "KVCache"}], "generic"
    )
    scheduler._install_retention_slot(
        "r2",
        _request("r2"),
        list(range(20, 30)),
        [{"class_name": "KVCache"}],
        "generic",
    )

    scheduler._cleanup_retention_boundary_snapshots("r1")
    assert scheduler._boundary_snapshot_store.cleaned == []

    scheduler._drain_spill_futures()
    assert scheduler._boundary_snapshot_store.cleaned == ["r1"]
    # Unrelated requests are never held back.
    scheduler._cleanup_retention_boundary_snapshots("other")
    assert scheduler._boundary_snapshot_store.cleaned == ["r1", "other"]


def test_spill_skipped_when_it_would_breach_the_ceiling():
    """The store path's memory ceiling guard applies to the spill as well."""
    scheduler = _scheduler()
    scheduler._memory_hard_limit_bytes = 1_000
    scheduler.memory_monitor = SimpleNamespace(
        estimate_prompt_kv_bytes=lambda _n: 10_000
    )
    scheduler._current_usage_bytes = lambda refresh_mlx_active=False: 0
    scheduler._install_retention_slot(
        "r1", _request("r1"), list(range(10)), [{"class_name": "KVCache"}], "generic"
    )
    scheduler._install_retention_slot(
        "r2",
        _request("r2"),
        list(range(20, 30)),
        [{"class_name": "KVCache"}],
        "generic",
    )
    assert scheduler._store_cache_executor.submissions == []
    scheduler._drain_spill_futures()
    assert scheduler._boundary_snapshot_store.cleaned == ["r1"]


def _snapshot_scheduler(snapshots, split_active=True):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SimpleNamespace(paged_cache_block_size=4)
    scheduler._boundary_cache_snapshots = {"r1": snapshots}
    scheduler._boundary_snapshot_store = _RecordingSnapshotStore()
    scheduler.paged_ssd_cache_manager = object()
    scheduler._gdn_split_active = lambda: split_active
    return scheduler


def test_gdn_snapshot_provider_serves_durable_full_blocks():
    """Only marker-free, block-aligned counts inside the prompt are offered."""
    scheduler = _snapshot_scheduler(
        {
            4: None,  # durable, block-aligned
            8: None,  # durable, block-aligned
            6: None,  # not block-aligned
            12: "pending",  # still has a marker
            20: None,  # past the token count asked for
        }
    )
    provider = scheduler._split_gdn_snapshot_provider("r1", 16)
    assert provider is not None
    assert 4 in provider and 8 in provider
    assert 6 not in provider and 12 not in provider and 20 not in provider
    assert len(provider) == 2


def test_gdn_snapshot_provider_is_absent_without_a_split_layout():
    scheduler = _snapshot_scheduler({4: None}, split_active=False)
    assert scheduler._split_gdn_snapshot_provider("r1", 16) is None


def test_gdn_snapshot_provider_is_absent_without_durable_blocks():
    # A request whose only snapshot is mid-block has nothing sliceable to offer.
    scheduler = _snapshot_scheduler({6: None})
    assert scheduler._split_gdn_snapshot_provider("r1", 16) is None
    assert scheduler._split_gdn_snapshot_provider("unknown", 16) is None
