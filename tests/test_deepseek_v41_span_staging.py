"""V4.1 staged span reads — converted-checkpoint staging coverage.

Every other staging test (test_deepseek_v41_dspark_offload.py) runs on
SOURCE checkpoints, where ``stage_predicted`` submits one ``_stage_one``
future per expert. Converted checkpoints take the coalesced path:
``_span_groups`` merges the predicted set into runs, one ``_stage_span``
future covers each run, and every expert id's ``StagedReads`` entry
points at that shared future — a join fans the decoded rows out to the
sibling keys (``block[expert - lo]`` per row). These
tests exercise that fan-out — including a gap-merged run where an
off-by-one in the row offset would commit a different expert's bytes.
"""

import mlx.core as mx
import numpy as np
import pytest
from streaming_fixtures import closer, ensure_spy, v41_backing
from test_deepseek_v41 import write_checkpoint

from omlx.patches.deepseek_v41.convert import convert
from omlx.patches.deepseek_v41.loading import load
from omlx.patches.deepseek_v41.moe_offload import _stage_span


def _slot(disk, i):
    return disk.language_model.layers[i].ffn.experts.slots


def _converted_disk(tmp_path, fraction=0.25):
    """Tiny checkpoint -> convert() -> offload load (converted path)."""
    source, _ = write_checkpoint(
        tmp_path, vision=False, n_routed_experts=8, n_activated_experts=2
    )
    target = tmp_path / "converted"
    convert(source, target)
    disk, _ = load(target, moe_expert_offload_resident_fraction=fraction)
    assert disk._moe_offload_plan.converted is not None
    return disk


def _backing(disk):
    return v41_backing(disk, dynamic=False)


def test_stage_span_payload_rows_match_per_expert_fetch(tmp_path, closer):
    """``_stage_span`` row math directly: each expert's payload sliced
    out of the merged block must equal the per-expert fetch — expert 3
    reads block[0], expert 5 reads block[2] of the [3,6) span. The
    return is a list aligned with the requested ids (the StagedReads
    set-read contract)."""
    disk = closer(_converted_disk(tmp_path))
    plan = disk._moe_offload_plan
    prefix = "language_model.layers.1.ffn.experts"
    got = _stage_span(plan, prefix, [3, 5])
    assert len(got) == 2
    for expert, payload in zip((3, 5), got):
        for proj in ("w1", "w3", "w2"):
            want = plan.fetch(prefix, proj, expert)["weight"]
            np.testing.assert_array_equal(
                np.asarray(
                    payload[proj]["weight"].astype(mx.float32)
                ),
                np.asarray(want.astype(mx.float32)),
            )


def test_backing_staged_span_preserves_arithmetic(tmp_path, monkeypatch, closer):
    """Converted-checkpoint analogue of
    test_backing_staged_payload_preserves_arithmetic: rows committed
    from a gap-merged staged span are bit-equal to the resident model's
    expert weights."""
    monkeypatch.setenv("OMLX_V41_SPAN_GAP", "2")  # {3, 5} merges into one span
    disk = closer(_converted_disk(tmp_path))
    resident = closer(load(tmp_path / "converted")[0])
    backing = _backing(disk)
    slots0 = _slot(disk, 0)
    slots1 = _slot(disk, 1)
    # Prime layer-1 residency away from the staged set, then let a
    # layer-0 ensure stage {3,5} into layer 1: one merged _stage_span
    # over rows [3,6), both expert keys sharing a single future in the
    # StagedReads registry (entry = (fut, ids, key_of)).
    slots1.ensure(mx.array([[6, 7]]))
    backing.note_routing(1, {3, 5})
    backing.recall_ewma[1] = 1.0  # skip EWMA warm-up (recall gate)
    slots0.ensure(mx.array([[0, 1]]))
    assert set(slots1._staged) == {3, 5}
    assert slots1._staged[3][0] is slots1._staged[5][0]
    slots1.ensure(mx.array([[3, 5]]))
    assert slots1.staged_hits == 2
    resident_experts = resident.language_model.layers[1].ffn.experts
    for expert in (3, 5):
        slot = slots1.slot_of[expert]
        for proj in ("w1", "w3", "w2"):
            np.testing.assert_array_equal(
                np.array(getattr(slots1.expert, proj).weight[slot]),
                np.array(getattr(resident_experts, proj).weight[expert]),
            )


def test_staged_span_multi_run_drops_undemanded(tmp_path, monkeypatch, closer):
    """A predicted set spanning several merged runs maps each run to one
    future; experts never demanded drop and cancel like the per-expert
    path."""
    monkeypatch.setenv("OMLX_V41_SPAN_GAP", "1")  # strictly-adjacent merge
    disk = closer(_converted_disk(tmp_path))
    backing = _backing(disk)
    slots0 = _slot(disk, 0)
    slots1 = _slot(disk, 1)
    slots1.ensure(mx.array([[6, 7]]))
    # {0,1} and {4,5} are two separate runs under gap=1.
    backing.note_routing(1, {0, 1, 4, 5})
    backing.recall_ewma[1] = 1.0
    slots0.ensure(mx.array([[2, 3]]))
    assert set(slots1._staged) == {0, 1, 4, 5}
    # One future per merged run, shared by that run's registry entries.
    assert slots1._staged[0][0] is slots1._staged[1][0]
    assert slots1._staged[4][0] is slots1._staged[5][0]
    assert slots1._staged[0][0] is not slots1._staged[4][0]
    rows = slots1.ensure(mx.array([[0, 1]]))
    mx.eval(rows)
    assert slots1.staged_hits == 2
    assert slots1.staged_drops == 2  # {4, 5} predicted, not demanded
    assert list(slots1.slot_of) == [0, 1]


def test_advise_demand_filters_residents(tmp_path, closer):
    """F1: advise_demand readaheads only missing experts — residents and
    staged rows are filtered, counters track advised rows."""
    disk = closer(_converted_disk(tmp_path))
    slots = _slot(disk, 0)
    assert slots.plan.converted is not None
    got = slots.advise_demand({0, 1, 2, 3})
    assert got == 4
    assert slots.ra_calls == 1
    assert slots.ra_rows == 4
    slots.ensure(mx.array([[0, 1]]))
    # Residents are skipped entirely: nothing left to advise.
    assert slots.advise_demand({0, 1}) == 0
    assert slots.ra_calls == 1
    # Partial overlap: only the missing rows advise.
    assert slots.advise_demand({0, 1, 4}) == 1
    assert slots.ra_rows == 5


def test_advise_demand_noop_on_source_checkpoint(tmp_path, closer):
    """F1: the unconverted per-expert-file path has no stacked store —
    advise is a no-op, never an error."""
    source, _ = write_checkpoint(
        tmp_path, vision=False, n_routed_experts=8, n_activated_experts=2
    )
    disk = closer(load(source, moe_expert_offload_resident_fraction=0.25)[0])
    slots = _slot(disk, 0)
    assert slots.plan.converted is None
    assert slots.advise_demand({0, 1}) == 0
    assert slots.advise_bank() == 0
    assert slots.ra_calls == 0


def test_prefill_call_advises_upcoming_chunks(tmp_path, closer):
    """F1: a multi-row call advises the whole demand in the first
    window (tiny bank < RA_TOKENS rows of lookahead)."""
    disk = closer(_converted_disk(tmp_path))
    slots = _slot(disk, 0)
    ffn = disk.language_model.layers[0].ffn
    mx.random.seed(0)
    x = mx.random.normal((1, 4, 32))
    out = ffn(x, None)
    mx.eval(out)
    # cap=2, top_k=2 -> step=1 -> 4 chunks; first window covers all
    # 4 rows' demand, so at least one advisory fired.
    assert slots.ra_calls >= 1
    assert slots.ra_rows >= 1


def test_prefill_advises_next_layer_bank(tmp_path, closer):
    """F1b: saturated coverage advises the NEXT layer's whole bank at
    this layer's call entry; the last layer does not wrap to layer 0."""
    disk = closer(_converted_disk(tmp_path))
    backing = _backing(disk)
    slots1 = _slot(disk, 1)
    ffn0 = disk.language_model.layers[0].ffn
    mx.random.seed(0)
    # 4 tokens x top-2 over an 8-expert bank: coverage is likely
    # saturated, but even partial union >= 50% triggers it. Drive
    # advise_next_layer directly for determinism.
    backing.advise_next_layer(0)
    assert slots1.ra_calls == 1
    assert slots1.ra_rows == slots1.plan.count
    n_layers = len(disk.language_model.layers)
    slots_last = _slot(disk, n_layers - 1)
    backing.advise_next_layer(n_layers - 1)  # no wrap
    assert slots_last.ra_calls == 0
    # And the forward path itself still works end to end.
    x = mx.random.normal((1, 4, 32))
    mx.eval(ffn0(x, None))


def test_ra_disabled_skips_all_advise(tmp_path, monkeypatch, closer):
    """F1: OMLX_V41_RA=0 turns every advisory path into a no-op."""
    monkeypatch.setenv("OMLX_V41_RA", "0")
    disk = closer(_converted_disk(tmp_path))
    backing = _backing(disk)
    slots = _slot(disk, 0)
    ffn = disk.language_model.layers[0].ffn
    mx.random.seed(0)
    mx.eval(ffn(mx.random.normal((1, 4, 32)), None))
    backing.advise_next_layer(0)
    assert slots.ra_calls == 0
    assert slots.ra_bytes == 0


def test_prefill_transient_cap_widens_and_restores(tmp_path, monkeypatch, closer):
    """F2: OMLX_V41_PREFILL_CAP raises the working cap for the call's
    duration — fewer, wider chunks — then restores cap and compacts the
    grown rooms back to the entry footprint."""
    monkeypatch.setenv("OMLX_V41_PREFILL_CAP", "6")
    disk = closer(_converted_disk(tmp_path))
    slots = _slot(disk, 0)
    ffn = disk.language_model.layers[0].ffn
    with ensure_spy(slots) as calls:
        mx.random.seed(0)
        mx.eval(ffn(mx.random.normal((1, 4, 32)), None))
    # step = 6 // 2 = 3 tokens per ensure -> [3, 1], not [1]*4.
    assert calls == [3, 1]
    # Working ceiling AND physical rooms return to the entry values.
    assert slots.cap == 2
    assert slots.rooms == 2


def test_prefill_cap_restores_on_mid_call_failure(tmp_path, monkeypatch, closer):
    """F2: an ensure failure mid-prefill still restores the cap and
    compacts — no leaked transient capacity after the exception."""
    monkeypatch.setenv("OMLX_V41_PREFILL_CAP", "6")
    disk = closer(_converted_disk(tmp_path))
    slots = _slot(disk, 0)
    ffn = disk.language_model.layers[0].ffn
    calls = 0
    orig_set = slots.arena.ensure_set

    def boom(needed, frozen, produce, verify=False, scratch=0):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("forced mid-prefill failure")
        return orig_set(
            needed, frozen, produce, verify=verify, scratch=scratch
        )

    slots.arena.ensure_set = boom
    mx.random.seed(0)
    with pytest.raises(RuntimeError, match="forced"):
        mx.eval(ffn(mx.random.normal((1, 4, 32)), None))
    slots.arena.ensure_set = orig_set
    assert slots.cap == 2
    assert slots.rooms == 2


def test_chunk_lookahead_stages_exact_demand(tmp_path, closer):
    """F2: each iteration stages the NEXT chunk's exact demand — the
    following ensure consumes staged payloads (hits, not misses)."""
    disk = closer(_converted_disk(tmp_path))
    slots = _slot(disk, 0)
    ffn = disk.language_model.layers[0].ffn
    mx.random.seed(0)
    # 8 tokens x top-2 over an 8-expert bank: the union saturates,
    # so the lookahead always has non-resident demand to stage.
    mx.eval(ffn(mx.random.normal((1, 8, 32)), None))
    assert slots.staged_submits >= 1
    assert slots.staged_hits >= 1


def test_chunk_lookahead_disabled(tmp_path, monkeypatch, closer):
    """F2: OMLX_V41_CHUNK_STAGE=0 leaves the staged dict untouched."""
    monkeypatch.setenv("OMLX_V41_CHUNK_STAGE", "0")
    disk = closer(_converted_disk(tmp_path))
    slots = _slot(disk, 0)
    ffn = disk.language_model.layers[0].ffn
    mx.random.seed(0)
    mx.eval(ffn(mx.random.normal((1, 8, 32)), None))
    assert slots.staged_submits == 0


def test_verify_round_advises_next_layer_union(tmp_path, closer):
    """F3 (converted path): after a verify round, a layer's ensure
    advises the next layer's predicted union — F_RDADVISE only, filtered
    to non-residents, and counted in verify_ra_submits."""
    from omlx.patches.deepseek_v41.moe_offload import verify_scope

    disk = closer(_converted_disk(tmp_path))
    backing = _backing(disk)
    slots0 = _slot(disk, 0)
    slots1 = _slot(disk, 1)
    with verify_scope():
        # Round 1: layer1 churns {3,4} -> {5,6} (cap=2), union 4.
        slots0.ensure(mx.array([0, 1]))
        slots1.ensure(mx.array([3, 4]))
        slots1.ensure(mx.array([5, 6]))
        # Round 2 wraps: verify_uniq[1] = {3,4,5,6}.
        slots0.ensure(mx.array([0, 1]))
    backing.verify_recall[1] = 1.0
    with verify_scope():
        slots0.ensure(mx.array([0, 1]))
    assert backing.verify_ra_submits >= 1
    # {5,6} resident on layer1 -> advise covered only {3,4}.
    assert slots1.ra_rows == 2
    summary = backing.summary()
    assert summary["verify_ra_submits"] >= 1
    assert summary["verify_recall"] > 0
