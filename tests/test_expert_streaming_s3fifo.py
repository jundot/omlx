# SPDX-License-Identifier: Apache-2.0
"""S3FIFOExpertCache — per-queue layer indexes, drain/resize accounting,
and the non-counting peek probes.

The per-layer indexes keep victim eviction O(1) (no _small + _store
scan), resize() re-bounds the small FIFO to the new capacity, and peek()
sees the small queue so a staged small-queue row does not read as a
miss.
"""
from omlx.patches.expert_streaming.cache_policies import S3FIFOExpertCache


def _cache(capacity_slots=64, layers=4):
    return S3FIFOExpertCache(
        capacity_slots * 1024,
        per_slot_bytes=1024,
        num_layers=layers,
    )


def test_resize_drains_both_queues():
    c = _cache(64, 4)
    for i in range(64):
        c.put((i % 4, i, "w"), object())
    assert 0 < c.size <= 64
    c.resize(16, 4)
    assert c.size <= 16
    assert len(c._small) + len(c._store) <= 16
    # _layer_counts tracks the surviving residents exactly.
    assert sum(c._layer_counts.values()) == c.size


def test_resize_retargets_queue_bounds():
    c = _cache(64, 4)
    for i in range(64):
        c.put((i % 4, i, "w"), object())
    c.resize(16, 4)
    # The small FIFO's bound follows the new capacity — a stale bound
    # would let it refill to the OLD size and undo the shrink.
    small_bound = c._small_cap
    assert small_bound <= max(16, c.capacity // 2) or small_bound <= 16
    for i in range(64, 128):
        c.put((i % 4, i, "w"), object())
    assert c.size <= 16


def test_peek_many_counts_small_queue_without_promotion():
    c = _cache(64, 4)
    key = (0, 5, "w")
    c.put(key, "v5")
    assert key in c._small  # fresh insert lands in the small FIFO
    hits0 = c.stats.hits
    # peek sees the small queue and never counts/promotes.
    assert c.peek(key) == "v5"
    assert c.peek_many([key]) == ["v5"]
    assert c.stats.hits == hits0
    assert key in c._small  # still small — peek is not a re-reference


def test_layer_indexes_track_every_queue_mutation():
    c = _cache(64, 4)
    keys = [(i % 4, i, "w") for i in range(32)]
    for k in keys:
        c.put(k, object())
    for k in keys:
        if k in c._small:
            assert k in c._small_layers.get(k[0], {})
        elif k in c._store:
            assert k in c._main_layers.get(k[0], {})
        else:
            raise AssertionError(f"{k} vanished")
    # Promotion to main moves the index entry.
    hit_key = next(k for k in keys if k in c._small)
    assert c.get(hit_key) is not None
    assert hit_key in c._store
    assert hit_key in c._main_layers.get(hit_key[0], {})
    assert hit_key not in c._small_layers.get(hit_key[0], {})
    # Clear wipes the indexes too.
    c.clear()
    assert c._small_layers == {} and c._main_layers == {}




def test_decode_miss_attribution_matches_base():
    c = _cache(64, 4)
    c.note_phase(True)
    assert c.get((3, 7, "w")) is None
    assert c.stats.decode_misses_by_layer.get(3) == 1
    c.note_phase(False)
    assert c.get((3, 8, "w")) is None
    assert c.stats.decode_misses_by_layer.get(3) == 1  # prefill ignored


def test_layer_local_victim_is_o1_via_index():
    """_evict_layer_unlocked resolves through the index, not a scan —
    assert by index membership rather than timing."""
    c = _cache(64, 4)
    for i in range(32):
        c.put((i % 4, i, "w"), object())
    layer = 1
    before = c._layer_counts.get(layer, 0)
    assert c._evict_layer_unlocked(layer) is True
    assert c._layer_counts.get(layer, 0) == before - 1
    # The evicted key is gone from whichever queue held it.
    survivors = set(c._small) | set(c._store)
    idx_keys = set()
    for m in (c._small_layers, c._main_layers):
        for order in m.values():
            idx_keys.update(order)
    assert idx_keys == survivors


def test_ghost_second_chance_goes_main():
    c = _cache(8, 1)
    c.put((0, 1, "w"), "a")
    c.put((0, 2, "w"), "b")
    # Overflow the small queue so (0,1) lands in ghost.
    small_cap = c._small_cap
    for i in range(3, 3 + small_cap + 2):
        c.put((0, i, "w"), i)
    assert (0, 1, "w") in c._ghost
    c.put((0, 1, "w"), "a2")  # ghost hit -> main queue
    assert c._store.get((0, 1, "w")) == "a2"
    assert (0, 1, "w") in c._main_layers.get(0, {})


def test_admission_note_counts_small_queue_toward_global_full():
    """B3: global fullness must see _small + _store, not _store alone.

    Probationary small-queue rows are residents. A fullness check over
    ``_store`` alone admitted on the first sighting into a cache that was
    already at capacity once the small FIFO was counted.
    """
    c = _cache(8, 4)  # capacity 8, per-layer cap 2
    # Raise layer 0's cap so the probe key's LAYER is not the binding
    # constraint — only the global occupancy matters.
    c.set_layer_caps({0: 16})
    for i in range(8):
        c.put((i % 4, i, "w"), object())
    # Promote one probationary row into the main queue, then refill —
    # the small FIFO re-bounds to its cap and total occupancy reaches
    # the global capacity.
    assert c.get((1, 1, "w")) is not None
    c.put((0, 9, "w"), object())
    assert c.size == c.capacity == 8
    assert len(c._small) == 7 and len(c._store) == 1
    # Under the old _store-only check, fullness read 1 < 8 and the first
    # sighting admitted; counting both queues the cache is full, so the
    # 2nd-touch filter applies.
    assert c.admission_note((0, 50, "w")) is False
    assert c.admission_note((0, 50, "w")) is True
