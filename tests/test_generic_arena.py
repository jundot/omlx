"""V4-2b generic arena mode (OMLX_EXPERT_STREAMING_ARENA).

SlotArena wired into StreamingSwitchGLU: residents are bound into fixed
slot rows of a persistent per-projection bank at admission and
gather_qmm indexes them by slot id — the bundle path's per-call mx.stack
assembly disappears on a hit. These tests use genuinely quantized
stacked experts (mx.quantize) over a real ExpertBackingStore so both
paths execute the same gather_qmm kernel on identical bytes.
"""

import json

import mlx.core as mx
import numpy as np
import pytest
from streaming_fixtures import closer, write_safetensors

import omlx.patches.expert_streaming.streaming_layers as ss
from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore
from omlx.patches.expert_streaming.streaming_switch import (
    ExpertLRUCache,
    StreamingQuantizedSwitchLinear,
    StreamingSwitchGLU,
)

_E = 8
_IN = 64
_MID = 32
_GS = 32
_BITS = 4
_PREFIX = "language_model.layers.0.mlp.switch_mlp"
_PROJS = ("gate_proj", "up_proj", "down_proj")


def _quant_stacked(e, out_dims, in_dims, gs=_GS, bits=_BITS, seed=0):
    """E quantized experts stacked: (E, out, packed_in) + scales/biases."""
    w = mx.random.normal((e * out_dims, in_dims), dtype=mx.float16)
    qw, qs, qb = mx.quantize(w, group_size=gs, bits=bits)
    return (
        qw.reshape(e, out_dims, -1),
        qs.reshape(e, out_dims, -1),
        qb.reshape(e, out_dims, -1),
    )


def _make_store(tmp_path, e=_E):
    tensors = {}
    dims = {"gate_proj": (_MID, _IN), "up_proj": (_MID, _IN), "down_proj": (_IN, _MID)}
    for i, (proj, (o, n)) in enumerate(dims.items()):
        qw, qs, qb = _quant_stacked(e, o, n, seed=i)
        tensors[f"{_PREFIX}.{proj}.weight"] = qw
        tensors[f"{_PREFIX}.{proj}.scales"] = qs
        tensors[f"{_PREFIX}.{proj}.biases"] = qb
    write_safetensors(tmp_path / "model.safetensors", tensors)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}})
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "num_experts": e,
                "num_hidden_layers": 1,
                "num_experts_per_tok": 2,
            }
        )
    )
    return ExpertBackingStore(tmp_path)


def _make_glu(store, per_layer_cap=6, e=_E):
    cache = ExpertLRUCache(
        budget_bytes=per_layer_cap * 4096,
        per_expert_bytes=4096,
        num_layers=1,
    )
    glu = StreamingSwitchGLU(
        input_dims=_IN,
        hidden_dims=_MID,
        num_experts=e,
        layer_idx=0,
        backing=store,
        cache=cache,
        quantized=True,
        group_size=_GS,
        bits=_BITS,
    )
    dims = {"gate_proj": (_IN, _MID), "up_proj": (_IN, _MID), "down_proj": (_MID, _IN)}
    for proj, (i, o) in dims.items():
        lin = StreamingQuantizedSwitchLinear(
            layer_idx=0,
            proj_name=proj,
            stacked_weight_key=f"{_PREFIX}.{proj}.weight",
            stacked_scales_key=f"{_PREFIX}.{proj}.scales",
            stacked_biases_key=f"{_PREFIX}.{proj}.biases",
            num_experts=e,
            input_dims=i,
            output_dims=o,
            backing=store,
            cache=cache,
            group_size=_GS,
            bits=_BITS,
        )
        setattr(glu, proj, lin)
    return glu


@pytest.fixture()
def model(tmp_path, closer):
    return closer(_make_store(tmp_path))


@pytest.fixture()
def arena_on(monkeypatch):
    """Enable the arena path (OMLX_EXPERT_STREAMING_ARENA) for a test."""
    monkeypatch.setattr(ss, "_ARENA_ENV", True)


def test_arena_disabled_by_default(model, monkeypatch):
    monkeypatch.setattr(ss, "_ARENA_ENV", False)
    glu = _make_glu(model)
    x = mx.random.normal((2, _IN), dtype=mx.float16)
    idx = mx.array([[0, 1], [2, 3]], dtype=mx.int32)
    out = glu(x, idx)
    mx.eval(out)
    assert getattr(glu, "_arena", None) is None


def test_arena_engage_maps_rhs_to_slots(model, arena_on):
    glu = _make_glu(model)
    plan = ss._RemapPlan()
    # Single-token shape (seq_len==1): the phase gate reads
    # indices.shape[-2] — a 2x2 call is a multi-token (prefill) call and
    # engages under the smaller prefill caps only.
    idx = mx.array([[0, 1, 2, 3]], dtype=mx.int32)
    assert glu._arena_engage(plan, idx)
    assert plan.arena_rhs is not None
    arena = glu._arena
    # every demanded expert is resident, rhs = slots[remapped]
    slots = [arena.book.slot_of[e] for e in plan.uniq_list]
    expected = np.array(
        [[slots[int(r)] for r in row] for row in plan.remapped.tolist()],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(np.asarray(plan.arena_rhs), expected)
    # all three projections have banks bound
    for proj in _PROJS:
        lin = getattr(glu, proj)
        assert lin._arena_bank.weight is not None
        assert lin._arena_bank.weight.shape[0] == arena.book.rooms


def test_arena_row_contents_bitexact(model, arena_on):
    glu = _make_glu(model)
    plan = ss._RemapPlan()
    idx = mx.array([[1, 4, 4, 2]], dtype=mx.int32)
    assert glu._arena_engage(plan, idx)
    arena = glu._arena
    for proj in _PROJS:
        lin = getattr(glu, proj)
        for eid in plan.uniq_list:
            slot = arena.book.slot_of[int(eid)]
            mx.eval(lin._arena_bank.weight)
            got_w = np.asarray(lin._arena_bank.weight[slot])
            src_w = np.asarray(
                lin._promote_np(
                    model.load_expert_slice(lin.stacked_weight_key, int(eid))
                )
            )
            np.testing.assert_array_equal(got_w, src_w)
            got_s = np.asarray(lin._arena_bank.scales[slot])
            dt = lin._slice_dtypes_lazy()
            src_s = np.asarray(
                lin._promote_np(
                    model.load_expert_slice(lin.stacked_scales_key, int(eid)),
                    dt[0],
                )
            )
            np.testing.assert_array_equal(got_s, src_s)


def test_arena_glu_output_matches_bundle_path(model, monkeypatch):
    # seq_len==1 (decode shape): the arena engages. A multi-token call is
    # gated by the prefill caps and would take the bundle path instead.
    x = mx.random.normal((1, _IN), dtype=mx.float16)
    idx = mx.array([[0, 1, 2, 3, 4, 5, 4, 5]], dtype=mx.int32)

    monkeypatch.setattr(ss, "_ARENA_ENV", False)
    glu_ref = _make_glu(model)
    ref = glu_ref(x, idx)
    mx.eval(ref)

    monkeypatch.setattr(ss, "_ARENA_ENV", True)
    glu_a = _make_glu(model)
    got = glu_a(x, idx)
    mx.eval(got)

    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
    # repeat call: hits only — same output again
    got2 = glu_a(x, idx)
    mx.eval(got2)
    np.testing.assert_array_equal(np.asarray(got2), np.asarray(ref))


def test_arena_sorted_path_matches_bundle_path(model, monkeypatch):
    """indices.size >= 64 engages _gather_sort: rhs in slot space must
    follow the sorted position order exactly like remapped does."""
    rng = np.random.default_rng(7)
    x = mx.random.normal((1, _IN), dtype=mx.float16)
    # One token, 64 routed rows (top_k=64): decode phase + do_sort both
    # engaged — the demand set must stay inside the per-layer cap (6).
    idx = mx.array(rng.integers(0, 6, size=(1, 64)), dtype=mx.int32)
    assert idx.size >= 64  # do_sort engaged

    monkeypatch.setattr(ss, "_ARENA_ENV", False)
    ref = _make_glu(model)(x, idx)
    mx.eval(ref)

    monkeypatch.setattr(ss, "_ARENA_ENV", True)
    glu_a = _make_glu(model)
    got = glu_a(x, idx)
    mx.eval(got)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


def test_arena_hit_pays_zero_reads(model, monkeypatch, arena_on):
    glu = _make_glu(model)
    reads = [0]
    orig = model.read_expert_into

    def spy(*a, **k):
        reads[0] += 1
        return orig(*a, **k)

    monkeypatch.setattr(model, "read_expert_into", spy)
    x = mx.random.normal((1, _IN), dtype=mx.float16)
    idx = mx.array([[0, 1, 2, 3]], dtype=mx.int32)
    mx.eval(glu(x, idx))
    first = reads[0]
    assert first > 0
    mx.eval(glu(x, idx))
    assert reads[0] == first, "second identical call must be all hits"
    assert glu._arena.book.hits > 0


def test_arena_eviction_keeps_correctness(model, arena_on):
    glu = _make_glu(model, per_layer_cap=3)
    x = mx.random.normal((1, _IN), dtype=mx.float16)
    # cycle 5 experts through a 3-slot arena: forces evictions
    for eids in ([0, 1], [2, 3], [4, 5], [0, 6], [7, 1]):
        idx = mx.array([list(eids)], dtype=mx.int32)
        mx.eval(glu(x, idx))
    arena = glu._arena
    assert arena.book.evictions > 0
    assert len(arena.book.slot_of) <= 3
    # the last demanded pair must be resident (LRU tail)
    assert {7, 1}.issubset(set(arena.book.slot_of))


def _sabotage_dict(glu):
    """Swap every projection's backing store for a plain dict."""
    for proj in _PROJS:
        getattr(glu, proj).backing = {}


def _sabotage_split(glu):
    """Activate a hobbit split on one projection."""
    glu.gate_proj.set_hobbit_split({0}, 8, 32)


@pytest.mark.parametrize(
    ("cap", "sabotage", "demand"),
    [
        # 6 unique experts > cap 3 -> engage declines before allocating
        (3, None, [0, 1, 2, 3, 4, 5]),
        # a non-store backing cannot host arena banks
        (6, _sabotage_dict, [0, 1]),
        # a hobbit-split projection cannot bind an arena bank
        (6, _sabotage_split, [0, 1]),
    ],
    ids=["demand_over_cap", "dict_backing", "split_active"],
)
def test_arena_falls_back(model, arena_on, cap, sabotage, demand):
    """Engage declines before allocating when the demand cannot be
    served — demand over the working cap, a backing without a real
    store, or a hobbit-split projection."""
    glu = _make_glu(model, per_layer_cap=cap)
    if sabotage is not None:
        sabotage(glu)
    plan = ss._RemapPlan()
    idx = mx.array([demand], dtype=mx.int32)
    assert not glu._arena_engage(plan, idx)
    assert plan.arena_rhs is None
    assert getattr(glu, "_arena", None) is None


def test_arena_rooms_byte_bounded(tmp_path, monkeypatch, arena_on, closer):
    """The arena bound is a byte ceiling across projections — a governor
    -sized cache cap must never commit unbounded bank memory (measured
    Metal OOM on qwen3.8-flash when rooms tracked the cache cap)."""
    store = closer(_make_store(tmp_path, e=64))
    glu = _make_glu(store, per_layer_cap=64, e=64)
    # ~1 byte of arena budget -> floor-1 room: the ceiling is a hard
    # bound, so an arena that cannot hold the demand must not commit
    # eight rows it cannot pay for.
    monkeypatch.setattr(ss, "_ARENA_MAX_BYTES", 1)
    plan = ss._RemapPlan()
    idx = mx.array([[0]], dtype=mx.int32)
    assert glu._arena_engage(plan, idx)
    arena = glu._arena
    assert arena.book.rooms == 1 == arena.book.cap
    # demand above the arena bound falls back to the bundle path
    plan2 = ss._RemapPlan()
    idx2 = mx.array(np.arange(10, dtype=np.int32).reshape(1, 10))
    assert not glu._arena_engage(plan2, idx2)
    assert plan2.arena_rhs is None


def test_arena_static_bound_survives_governor_shrink(tmp_path, arena_on, closer):
    """A governor shrink lowers the cache cap (fewer calls engage) but the
    arena's physical bound is static — bounded memory, no mid-run churn."""
    store = closer(_make_store(tmp_path, e=64))
    glu = _make_glu(store, per_layer_cap=16, e=64)
    plan = ss._RemapPlan()
    idx = mx.array([[0, 1, 2, 3]], dtype=mx.int32)
    assert glu._arena_engage(plan, idx)
    arena = glu._arena
    assert arena.book.rooms == 16
    glu._cache.set_layer_caps({0: 4})
    # demand within the shrunken cache cap still engages the arena
    plan2 = ss._RemapPlan()
    idx2 = mx.array([[0, 5, 6, 7]], dtype=mx.int32)
    assert glu._arena_engage(plan2, idx2)
    assert plan2.arena_rhs is not None
    # demand above the shrunken cache cap -> bundle path
    plan3 = ss._RemapPlan()
    idx3 = mx.array(np.arange(10, dtype=np.int32).reshape(1, 10))
    assert not glu._arena_engage(plan3, idx3)
    assert plan3.arena_rhs is None


def test_arena_feeds_governor_stats(model, arena_on):
    glu = _make_glu(model)
    x = mx.random.normal((1, _IN), dtype=mx.float16)
    idx = mx.array([[0, 1, 2, 3]], dtype=mx.int32)
    mx.eval(glu(x, idx))
    st = glu._cache.stats
    assert st.decode_layers + st.prefill_layers == 1
    assert st.decode_layers_missed + st.prefill_layers_missed == 1
    # per-layer miss map feeds the governor's targeted growth
    if st.decode_layers_missed:
        assert st.decode_misses_by_layer.get(0, 0) >= 1


def test_arena_bytes_counted_in_cache_resident_bytes(model, arena_on):
    """B6: arena-bound rows never enter _store, so the cache's
    resident_bytes must add every registered arena's bound banks — else
    the scheduler's heap accounting loses the whole arena residency."""
    glu = _make_glu(model)
    cache = glu._cache
    plan = ss._RemapPlan()
    idx = mx.array([[0, 1, 2, 3]], dtype=mx.int32)
    assert glu._arena_engage(plan, idx)
    arena = glu._arena
    expected = sum(
        int(a.nbytes)
        for proj in _PROJS
        for a in (
            getattr(glu, proj)._arena_bank.weight,
            getattr(glu, proj)._arena_bank.scales,
            getattr(glu, proj)._arena_bank.biases,
        )
        if a is not None
    )
    assert expected > 0
    assert arena.resident_bytes() == expected
    # The cache total is store slots + every registered arena's bytes.
    store_bytes = int(cache.size) * int(cache.per_slot_bytes)
    assert cache.resident_bytes() == store_bytes + expected
