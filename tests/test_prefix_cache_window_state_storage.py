# SPDX-License-Identifier: Apache-2.0
"""Regression guards for turn-end sliding-window storage (window_state_at_turn_ends).

Inkling 2 keeps each sliding visit's window K/V in an ArraysCache. A boundary
snapshot of that state is the whole window (~170 MB for Inkling 2 Mini), so
storing one in every block fills a 5 GB hot cache near 48k tokens and the
prefix chain is evicted. With the setting on, window state is stored only in
a store's last block (the turn end) and in the full blocks
keep_window_checkpoint selects: the last two, then sparser further back
(about 2*log2(n) per conversation). The lookup restores from the longest
point that still has window state.

Tests:
- test_checkpoint_rule_bounds_edit_cost — the rule keeps an edit's
  re-prefill within 5/3 of stock and only removes checkpoints as a chain grows.
- test_single_store_keeps_rule_checkpoints — one long store: only the
  selected blocks keep the window, full-attention KV stays in every block.
- test_flag_off_keeps_stock_storage — without the setting every snapshot
  block keeps its window state.
- test_state_without_window_is_unaffected — conv-only and recurrent
  (GDN-style: 3D conv + 4D state) ArraysCache layers are stored as before.
- test_lookup_falls_back_to_block_with_window — a longer match ending on a
  stripped block restores from the newest block that has window state.
- test_lookup_reads_payload_when_marker_is_unknown — same, with the
  in-memory markers cleared (as after a restart).
- test_lookup_reads_ssd_file_shapes — same, from SSD files (no hot cache).
- test_growing_chain_is_thinned_to_rule — a conversation growing turn by
  turn ends with exactly the selected checkpoints, for turns ending inside a
  block and on a block boundary; the turn-end tail keeps its window.
- test_markers_follow_stored_payload — a block whose snapshot had no window
  (a lean snapshot) is marked as such.
- test_backfill_keeps_interior_blocks_lean — a longer store over the same
  prefix ends with the selected checkpoints only.
- test_unknown_block_is_not_remembered — a probe miss is not cached.
- test_lean_snapshots_store_and_restore — a store fed lean snapshots still
  restores at the turn end.
- test_lean_snapshot_* — mid-prefill boundary snapshots the rule does not
  select drop the window K/V (the store would strip it anyway); the tail
  snapshot and recurrent states are kept, in memory and on the SSD path.
- test_setting_reaches_scheduler_config — settings.json, the env var and the
  CLI flag all set it; old settings files default to off.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from omlx.cache.paged_cache import PagedCacheManager, keep_window_checkpoint
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")

BLOCK_SIZE = 4
WINDOW = 4


class MockModel:
    def __init__(self, num_layers: int = 2):
        self._num_layers = num_layers
        self.layers = [MagicMock() for _ in range(num_layers)]

    @property
    def args(self):
        a = MagicMock()
        a.num_hidden_layers = self._num_layers
        return a


def _make_cache(tmp_path, turn_ends=True, ssd_only=False):
    paged_cache = PagedCacheManager(
        block_size=BLOCK_SIZE,
        max_blocks=400,
        model_name="test-model",
        initial_blocks=400,
    )
    ssd = PagedSSDCacheManager(
        cache_dir=tmp_path / "ssd_cache",
        max_size_bytes=100 * 1024**2,
        hot_cache_max_bytes=0 if ssd_only else 20 * 1024**2,
        hot_cache_only=not ssd_only,
    )
    paged_cache.set_paged_ssd_cache_manager(ssd)  # as the scheduler does
    cache = BlockAwarePrefixCache(
        model=MockModel(num_layers=2),
        paged_cache_manager=paged_cache,
        paged_ssd_cache_manager=ssd,
        window_state_at_turn_ends=turn_ends,
    )
    return cache, ssd


def _layers(seq_len, window=True):
    """A sliding layer (ArraysCache: window K/V + conv, Inkling 2 style) and a
    full-attention KVCache layer. ``window=False`` gives a conv-only
    ArraysCache, ``window="gdn"`` a recurrent one (3D conv + 4D state)."""
    sliding = [mx.ones((1, 3, 8)), mx.ones((1, 3, 8))]
    if window == "lean":
        sliding = (mx.zeros((1,)), mx.zeros((1,)))
    elif window == "gdn":
        sliding = [mx.ones((1, 3, 8)), mx.ones((1, 2, 8, 8))]
    elif window:
        sliding = [mx.ones((1, 2, WINDOW, 8)), mx.ones((1, 2, WINDOW, 8))] + sliding
    return [
        {
            "state": sliding,
            "cache_type": "ArraysCache",
            "class_name": "ArraysCache",
            "meta_state": (),
        },
        {
            "state": (mx.ones((1, 2, seq_len, 8)), mx.ones((1, 2, seq_len, 8))),
            "cache_type": "KVCache",
            "class_name": "KVCache",
            "meta_state": (str(seq_len),),
        },
    ]


def _store(cache, rid, num_tokens, window=True, tail_terminal=False, lean_before=0):
    """``lean_before``: snapshots below this boundary carry a window
    placeholder, as the scheduler's lean snapshots do."""
    tokens = list(range(num_tokens))
    snapshots = {
        b: _layers(b, window if b >= lean_before else "lean")
        for b in range(BLOCK_SIZE, num_tokens, BLOCK_SIZE)
    }
    table = cache.store_cache(
        rid,
        tokens,
        _layers(num_tokens, window),
        boundary_snapshots=snapshots,
        _store_tail_terminal=tail_terminal,
    )
    assert table is not None
    return table


def _hash(cache, table, idx):
    return cache.paged_cache.allocated_blocks[table.block_ids[idx]].block_hash


def _sliding_has_window(ssd, block_hash):
    data, meta = ssd.load_block_with_metadata(block_hash)
    assert data is not None and meta is not None
    layer = data[0]
    if isinstance(layer, tuple) and len(layer) == 3 and layer[0] == "__nstate__":
        return any(getattr(e, "ndim", 0) == 4 for e in layer[2])
    return False


def _kv_len(ssd, block_hash):
    data, _ = ssd.load_block_with_metadata(block_hash)
    return int(data[1][0].shape[2])


def _rule(n):
    return [keep_window_checkpoint(i, n) for i in range(n)]


def test_checkpoint_rule_bounds_edit_cost():
    for n in range(1, 200):
        kept = [i + 1 for i in range(n) if keep_window_checkpoint(i, n)]
        assert kept[-1] == n
        if n > 1:  # a longer chain keeps a subset of a shorter one's
            shorter = {i + 1 for i in range(n - 1) if keep_window_checkpoint(i, n - 1)}
            assert set(kept) - {n} <= shorter
        for edit in range(n):  # first changed block; stock resumes at it
            resume = max((b for b in kept if b <= edit), default=0)
            assert 3 * (n - resume) <= 5 * (n - edit)


def test_single_store_keeps_rule_checkpoints(tmp_path):
    cache, ssd = _make_cache(tmp_path)
    table = _store(cache, "a", 20 * BLOCK_SIZE)
    has_window = [_sliding_has_window(ssd, _hash(cache, table, i)) for i in range(20)]
    assert has_window == _rule(20)
    assert not all(has_window)
    assert all(_kv_len(ssd, _hash(cache, table, i)) == BLOCK_SIZE for i in range(20))


def test_flag_off_keeps_stock_storage(tmp_path):
    cache, ssd = _make_cache(tmp_path, turn_ends=False)
    table = _store(cache, "a", 4 * BLOCK_SIZE)
    assert all(_sliding_has_window(ssd, _hash(cache, table, i)) for i in range(4))


@pytest.mark.parametrize("kind", [False, "gdn"], ids=["conv-only", "recurrent"])
def test_state_without_window_is_unaffected(tmp_path, kind):
    cache, ssd = _make_cache(tmp_path)
    table = _store(cache, "a", 4 * BLOCK_SIZE, window=kind)
    for i in range(4):
        data, _ = ssd.load_block_with_metadata(_hash(cache, table, i))
        assert tuple(data[0][0].shape) == (1, 3, 8)  # real conv state, no placeholder
    assert not getattr(cache.paged_cache, "_swa_active", False)


LONG = 20  # blocks in the second turn
MATCH = 11  # blocks a diverging prompt matches
RESUME = max(i + 1 for i in range(MATCH) if keep_window_checkpoint(i, LONG))


def _two_turns(cache):
    # Turn 2 extends turn 1 to LONG blocks; a prompt that diverges after
    # MATCH blocks resumes at the newest selected checkpoint before it.
    _store(cache, "turn-1", 2 * BLOCK_SIZE)
    return _store(cache, "turn-2", LONG * BLOCK_SIZE)


def test_lookup_falls_back_to_block_with_window(tmp_path):
    cache, _ = _make_cache(tmp_path)
    _two_turns(cache)
    assert RESUME < MATCH
    prompt = list(range(MATCH * BLOCK_SIZE)) + [-1]
    blocks, n = cache.paged_cache.get_computed_blocks(prompt)
    assert n == RESUME * BLOCK_SIZE
    assert len(blocks) == RESUME


def test_lookup_reads_payload_when_marker_is_unknown(tmp_path):
    cache, _ = _make_cache(tmp_path)
    _two_turns(cache)
    cache.paged_cache._swa_full_hashes.clear()
    cache.paged_cache._swa_no_window.clear()
    prompt = list(range(MATCH * BLOCK_SIZE)) + [-1]
    _, n = cache.paged_cache.get_computed_blocks(prompt)
    assert n == RESUME * BLOCK_SIZE


@pytest.mark.parametrize("extra", [2, 0], ids=["tail-turn-ends", "aligned-turn-ends"])
def test_growing_chain_is_thinned_to_rule(tmp_path, extra):
    cache, ssd = _make_cache(tmp_path)
    turns = 24
    for t in range(1, turns + 1):
        table = _store(
            cache, f"t{t}", t * BLOCK_SIZE + extra, tail_terminal=bool(extra)
        )
    full = [_sliding_has_window(ssd, _hash(cache, table, i)) for i in range(turns)]
    assert full == _rule(turns)
    if extra:
        assert _sliding_has_window(ssd, _hash(cache, table, turns))  # turn-end tail


def test_lookup_reads_ssd_file_shapes(tmp_path):
    cache, ssd = _make_cache(tmp_path, ssd_only=True)
    _two_turns(cache)
    assert ssd.wait_for_pending_writes(timeout=30)
    cache.paged_cache._swa_full_hashes.clear()
    cache.paged_cache._swa_no_window.clear()
    prompt = list(range(MATCH * BLOCK_SIZE)) + [-1]
    _, n = cache.paged_cache.get_computed_blocks(prompt)
    assert n == RESUME * BLOCK_SIZE


def test_markers_follow_stored_payload(tmp_path):
    # The last full block's snapshot has no window (lean): it must not be
    # marked as holding one, nor take a cap slot.
    cache, ssd = _make_cache(tmp_path)
    table = _store(
        cache, "a", 3 * BLOCK_SIZE + 2, tail_terminal=True, lean_before=4 * BLOCK_SIZE
    )
    last_full = _hash(cache, table, 2)
    assert not _sliding_has_window(ssd, last_full)
    assert last_full in cache.paged_cache._swa_no_window
    assert _sliding_has_window(ssd, _hash(cache, table, 3))  # turn-end tail


def test_backfill_keeps_interior_blocks_lean(tmp_path):
    cache, ssd = _make_cache(tmp_path)
    _store(cache, "a", 20 * BLOCK_SIZE)
    # Same prefix, longer: blocks 0-19 dedup and carry boundary snapshots.
    table = _store(cache, "b", 24 * BLOCK_SIZE)
    has_window = [_sliding_has_window(ssd, _hash(cache, table, i)) for i in range(24)]
    assert has_window == _rule(24)


def test_unknown_block_is_not_remembered(tmp_path):
    cache, _ = _make_cache(tmp_path)
    missing = b"\x01" * 32
    assert not cache.paged_cache._swa_has_window(missing)
    assert missing not in cache.paged_cache._swa_no_window


def test_lean_snapshots_store_and_restore(tmp_path):
    cache, ssd = _make_cache(tmp_path)
    _store(cache, "turn-1", 2 * BLOCK_SIZE + 2, tail_terminal=True)
    table = _store(
        cache,
        "turn-2",
        5 * BLOCK_SIZE + 2,
        tail_terminal=True,
        lean_before=4 * BLOCK_SIZE,
    )
    has_window = [_sliding_has_window(ssd, _hash(cache, table, i)) for i in range(6)]
    # The rule for 5 full blocks, except block 2 whose snapshot was lean.
    expected = _rule(5)
    expected[2] = False
    assert has_window == expected + [True]
    prompt = list(range(5 * BLOCK_SIZE + 2)) + [999]
    _, n = cache.paged_cache.get_computed_blocks(prompt)
    assert n == 5 * BLOCK_SIZE + 2


# --- Lean boundary snapshots (scheduler side) ---------------------------------

from types import SimpleNamespace  # noqa: E402

from omlx.scheduler import _TAIL_SNAPSHOT_SOURCE, Scheduler  # noqa: E402


def _snapshot(seq_len, window="window"):
    layers = _layers(seq_len, window=window if window != "window" else True)
    return (Scheduler._PREFILL_SNAPSHOT_MARKER, layers)


def _lean(token_count, prompt_len, source="prefill", window="window", on=True):
    stub = _stub(prompt_len, on)
    lean = Scheduler._swa_snapshot_is_lean(stub, "r", token_count, BLOCK_SIZE, source)
    out = Scheduler._swa_lean_value(stub, _snapshot(token_count, window), lean)
    return out[1][0]["state"]  # the sliding layer


def _stub(prompt_len, on=True):
    return SimpleNamespace(
        config=SimpleNamespace(window_state_at_turn_ends=on),
        requests={"r": SimpleNamespace(prompt_token_ids=list(range(prompt_len)))},
        _PREFILL_SNAPSHOT_MARKER=Scheduler._PREFILL_SNAPSHOT_MARKER,
        _swa_drop_window_states=Scheduler._swa_drop_window_states,
        _extract_snapshot_cache_states=lambda c: (_layers(2 * BLOCK_SIZE), None),
    )


def _is_placeholder(state):
    return isinstance(state, tuple) and tuple(state[0].shape) == (1,)


def test_lean_snapshot_drops_window_far_from_prompt_end():
    # 10 blocks of prompt: boundaries before the last two keep nothing.
    prompt = 10 * BLOCK_SIZE + 3
    assert _is_placeholder(_lean(2 * BLOCK_SIZE, prompt))


def test_lean_snapshot_keeps_rule_checkpoints():
    prompt = 10 * BLOCK_SIZE + 3
    for b in range(1, 11):
        kept = keep_window_checkpoint(b - 1, 10)
        assert _is_placeholder(_lean(b * BLOCK_SIZE, prompt)) != kept


def test_lean_snapshot_keeps_tail_and_recurrent_state():
    prompt = 10 * BLOCK_SIZE + 3
    assert not _is_placeholder(_lean(2 * BLOCK_SIZE, prompt, _TAIL_SNAPSHOT_SOURCE))
    assert not _is_placeholder(_lean(2 * BLOCK_SIZE, prompt, window="gdn"))


def test_lean_snapshot_flag_off_keeps_window():
    assert not _is_placeholder(_lean(2 * BLOCK_SIZE, 10 * BLOCK_SIZE + 3, on=False))


def test_lean_snapshot_ssd_path_drops_window():
    states, _ = Scheduler._extract_lean_snapshot_cache_states(
        _stub(10 * BLOCK_SIZE), []
    )
    assert _is_placeholder(states[0]["state"])
    assert tuple(states[1]["state"][0].shape)[2] == 2 * BLOCK_SIZE  # KV untouched


def test_setting_reaches_scheduler_config(tmp_path, monkeypatch):
    from omlx.settings import CacheSettings, GlobalSettings

    on = CacheSettings(window_state_at_turn_ends=True)
    assert CacheSettings.from_dict(on.to_dict()).window_state_at_turn_ends
    assert not CacheSettings.from_dict({}).window_state_at_turn_ends

    settings = GlobalSettings()
    settings._apply_cli_overrides(SimpleNamespace(window_state_at_turn_ends=True))
    assert settings.to_scheduler_config().window_state_at_turn_ends

    monkeypatch.setenv("OMLX_WINDOW_STATE_AT_TURN_ENDS", "1")
    loaded = GlobalSettings.load(base_path=tmp_path)
    assert loaded.to_scheduler_config().window_state_at_turn_ends
