"""DSpark verify under frozen expert residency (clean-split design).

Layer 3 only: verify_scope() suspends LRU reordering in _ExpertSlots
while the shared block kernels (layer 1) and plan builders (layer 2)
are untouched. Native MTP + offload validates for deepseek_v41 alone.
"""

import mlx.core as mx
import numpy as np
import pytest
from test_deepseek_v41 import write_checkpoint

from omlx.model_settings import validate_moe_expert_offload
from omlx.patches.deepseek_v41.loading import load
from omlx.patches.deepseek_v41.moe_offload import verify_scope
from omlx.patches.deepseek_v41.streaming_backing import V41StreamingBacking


def _backed_disk(tmp_path, **kwargs):
    disk = _offloaded_disk(tmp_path)
    layers = [
        (i, layer.ffn.experts.slots)
        for i, layer in enumerate(disk.language_model.layers)
    ]
    backing = V41StreamingBacking(disk._moe_offload_plan, layers, **kwargs)
    # Engine parity: the hook chain resolves the backing off the model.
    disk._expert_streaming_backing = backing
    return disk, backing


def _offloaded_disk(tmp_path, subdir=None):
    base = tmp_path / subdir if subdir else tmp_path
    base.mkdir(parents=True, exist_ok=True)
    source, _ = write_checkpoint(
        base, vision=False, n_routed_experts=8, n_activated_experts=2
    )
    disk, _ = load(source, moe_expert_offload_resident_fraction=0.25)
    assert disk._moe_offload_plan.capacity == 2
    return disk


def _prime_recall(backing, layer, value=1.0):
    """Seed the per-layer staging recall gate.

    The EWMA needs ~4 matching prev->now routing observations to cross
    the 0.3 floor; tests set it directly so staging expectations measure
    the path under test, not gate warm-up.
    """
    backing.recall_ewma[int(layer)] = value


def test_verify_scope_freezes_slot_recency(tmp_path, monkeypatch):
    """DSpark verify traffic must not disturb decode-hot LRU order
    (legacy eviction semantics — scratch rows have their own tests)."""
    monkeypatch.setenv("OMLX_V41_VERIFY_SCRATCH", "0")
    disk = _offloaded_disk(tmp_path)
    try:
        slots = disk.language_model.layers[0].ffn.experts.slots
        slots.ensure(mx.array([0, 1]))
        assert list(slots.slot_of) == [0, 1]
        with verify_scope():
            # Frozen hit: served, order untouched.
            slots.ensure(mx.array([1]))
            assert list(slots.slot_of) == [0, 1]
            # Frozen miss: draft-only expert lands oldest-first, evicting
            # the oldest non-needed entry without reordering the rest.
            slots.ensure(mx.array([2]))
            assert list(slots.slot_of) == [2, 1]
            # Capacity guard still applies inside the scope.
            with pytest.raises(ValueError, match="exceeds resident capacity"):
                slots.ensure(mx.array([0, 1, 2]))
        # Outside the scope, normal LRU resumes.
        slots.ensure(mx.array([2]))
        assert list(slots.slot_of) == [1, 2]
    finally:
        disk.close()


def test_verify_scope_preserves_forward(tmp_path):
    """Frozen recency changes slot assignment, never the arithmetic."""
    disk = _offloaded_disk(tmp_path)
    try:
        ffn = disk.language_model.layers[0].ffn
        mx.random.seed(0)
        x = mx.random.normal((1, 4, 32))
        plain = ffn(x, None)
        mx.eval(plain)
        with verify_scope():
            scoped = ffn(x, None)
            mx.eval(scoped)
        np.testing.assert_allclose(
            np.array(plain), np.array(scoped), rtol=1e-5, atol=1e-6
        )
    finally:
        disk.close()


def test_offload_dspark_validation_matrix():
    """Native MTP + offload is allowed only for deepseek_v41 (DSpark)."""
    base = {
        "moe_expert_offload_enabled": True,
        "moe_expert_offload_resident_fraction": 0.125,
    }
    validate_moe_expert_offload(
        {**base, "mtp_enabled": True}, model_type="deepseek_v41"
    )
    for extra in ("vlm_mtp_enabled", "dflash_enabled"):
        with pytest.raises(ValueError, match="cannot be combined"):
            validate_moe_expert_offload(
                {**base, "mtp_enabled": True, extra: True},
                model_type="deepseek_v41",
            )
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_moe_expert_offload(
            {**base, "mtp_enabled": True}, model_type="qwen3_8_next"
        )
    # Unknown type stays strict: the exception needs a known model.
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_moe_expert_offload({**base, "mtp_enabled": True})


def test_settings_construction_scopes_by_model_type():
    """Post-init enforces the same per-type rule as load/save."""
    from omlx.model_settings import ModelSettings

    ModelSettings(
        moe_expert_offload_enabled=True,
        moe_expert_offload_resident_fraction=0.125,
        mtp_enabled=True,
        model_type="deepseek_v41",
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        ModelSettings(
            moe_expert_offload_enabled=True,
            mtp_enabled=True,
            model_type="qwen3_8_next",
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        ModelSettings(moe_expert_offload_enabled=True, mtp_enabled=True)


def test_backing_resize_preserves_arithmetic(tmp_path):
    """Grow/shrink change rooms, never the math (bit-exact gate)."""
    disk, backing = _backed_disk(tmp_path, dynamic=False)
    try:
        assert backing.governor is None
        slots = disk.language_model.layers[0].ffn.experts.slots
        assert backing.capacity == 2
        backing.resize(6, 6)
        assert slots.cap == 6 and slots.rooms == 2  # lazy growth
        slots.ensure(mx.array([0, 1, 2, 3]))
        assert slots.rooms == 4  # grown to need, not to ceiling
        assert list(slots.slot_of) == [0, 1, 2, 3]
        backing.resize(2, 2)
        assert list(slots.slot_of) == [2, 3]  # newest survive
        assert slots.rooms == 2
        mx.random.seed(1)
        x = mx.random.normal((1, 4, 32))
        resident_out = None
        ffn = disk.language_model.layers[0].ffn
        with verify_scope():
            pass
        out = ffn(x, None)
        mx.eval(out)
        assert out.shape == (1, 4, 32)
    finally:
        disk.close()


def test_backing_stats_count_decode_only(tmp_path):
    """Hunger sees decode visits; prefill and verify are invisible."""
    disk, backing = _backed_disk(tmp_path, dynamic=False)
    try:
        slots = disk.language_model.layers[0].ffn.experts.slots
        slots.ensure(mx.array([[0, 1]]))  # decode-shaped: counted
        assert backing.stats.decode_layers == 1
        slots.ensure(mx.array([[0, 1], [0, 1]]))  # prefill: skipped
        assert backing.stats.decode_layers == 1
        with verify_scope():
            slots.ensure(mx.array([[4, 5]]))  # verify: skipped
        assert backing.stats.decode_layers == 1
        assert backing.stats.decode_layers_missed >= 1
    finally:
        disk.close()


def test_parallel_fetch_matches_serial(tmp_path, monkeypatch):
    """Worker fetch must be bit-identical: ordered join, same LRU."""
    import os

    from test_deepseek_v41 import write_checkpoint

    from omlx.patches.deepseek_v41.loading import load as _load

    base = tmp_path / "shared"
    base.mkdir(parents=True, exist_ok=True)
    source, _ = write_checkpoint(
        base, vision=False, n_routed_experts=8, n_activated_experts=2
    )

    def _run(name, threads):
        disk, _ = _load(source, moe_expert_offload_resident_fraction=0.25)
        try:
            if threads is None:
                monkeypatch.delenv("OMLX_V41_FETCH_THREADS", raising=False)
            else:
                monkeypatch.setenv("OMLX_V41_FETCH_THREADS", threads)
            mx.random.seed(7)
            x = mx.random.normal((1, 4, 32))
            out = disk.language_model.layers[0].ffn(x, None)
            mx.eval(out)
            slots = disk.language_model.layers[0].ffn.experts.slots
            return np.array(out), list(slots.slot_of.items()), slots.misses
        finally:
            disk.close()
            monkeypatch.delenv("OMLX_V41_FETCH_THREADS", raising=False)

    serial = _run("serial", None)
    parallel = _run("parallel", "4")
    assert np.array_equal(parallel[0], serial[0])
    assert parallel[1] == serial[1]
    assert parallel[2] == serial[2]
    # Degenerate env values fall back to serial without crashing.
    _run("degenerate", "banana")


def test_backing_observe_acts_end_to_end(tmp_path):
    """Regression: observe() must not idle-skip on the backing.

    The generic governor gates on cache.capacity; the backing used to
    expose only base_cap, so every observe returned '' (actions stayed 0
    on real runs despite 96% stall). With tiny thresholds, free memory is
    abundant and observe must grow the per-layer base."""
    disk, backing = _backed_disk(tmp_path)
    try:
        mx.random.seed(11)
        x = mx.random.normal((1, 4, 32))
        mx.eval(disk.language_model.layers[0].ffn(x, None))
        bk = disk._expert_streaming_backing
        if bk is None:
            bk = backing
        gov = bk.governor
        assert gov is not None
        gov.low_free_bytes = 1
        gov.target_free_bytes = 2
        gov.high_free_bytes = 3
        gov.cooldown_s = 0
        before = bk.base_cap
        action = gov.observe()
        assert action != "", "observe idle-skipped on V41 backing"
        assert gov.actions == 1
        assert bk.base_cap > before
    finally:
        disk.close()


def test_backing_governor_shrink_and_grow(tmp_path):
    """Forced observe steps retarget per-layer caps and stay exact."""
    disk, backing = _backed_disk(tmp_path)
    try:
        assert backing.governor is not None
        gov = backing.governor
        slots = disk.language_model.layers[0].ffn.experts.slots
        # Fixture-scale floors: drop the RAM-sized defaults so one slot
        # of pressure/hunger moves the needle deterministically. Zero the
        # clear band too — the host's real free memory would otherwise
        # route observe() to the clear branch on a loaded machine.
        gov.min_budget_bytes = 0
        gov.min_cap = 1
        gov.grow_add_frac = 1.0
        gov.low_free_bytes = 0
        # Pressure: shrink halves the per-layer base.
        gov.target_free_bytes = 10**18
        gov.observe(force=True)
        assert backing.capacity == 1
        assert "shrink" in gov.last_action
        # Hunger: decode misses + headroom grow it back.
        slots.ensure(mx.array([[4, 4]]))
        slots.ensure(mx.array([[5, 5]]))
        gov.target_free_bytes = 0
        gov.low_free_bytes = 0
        gov.min_window_layers = 1
        gov.stall_target = 0.0
        gov.observe(force=True)
        assert backing.capacity == 2
        assert "grow" in gov.last_action
        assert gov.actions == 2
        # Targeting overrides are per-layer ceilings.
        backing.set_layer_caps({0: 4})
        assert backing.layer_cap_overrides() == {0: 4}
        assert slots.cap == 4
        backing.clear()
        assert len(slots.slot_of) == 0
    finally:
        disk.close()


def test_backing_layer_cap_noop_dropped(tmp_path):
    """The governor's num_layers==1 retarget sends {layer: base_cap} — an
    override that lands on the applied base is not targeting; dropping it
    keeps layer_cap_overrides() truthful."""
    disk, backing = _backed_disk(tmp_path, dynamic=False)
    try:
        base = backing.base_cap
        backing.set_layer_caps({0: base})
        assert backing.layer_cap_overrides() == {}
        assert backing.slots_of[0].cap == base
        # A real per-layer ceiling still applies and reports.
        backing.set_layer_caps({1: base + 1})
        assert backing.layer_cap_overrides() == {1: base + 1}
        assert backing.slots_of[1].cap == base + 1
        assert backing.slots_of[0].cap == base
    finally:
        disk.close()


def test_backing_clear_cancels_staged(tmp_path):
    """clear() drops residency AND pending staged fetches — an in-flight
    mispredict cannot commit after a desperate-clear."""
    disk, backing = _backed_disk(tmp_path, dynamic=False)
    try:
        slots0 = disk.language_model.layers[0].ffn.experts.slots
        slots1 = disk.language_model.layers[1].ffn.experts.slots
        slots1.ensure(mx.array([[6, 7]]))
        backing.note_routing(1, {4, 5})
        _prime_recall(backing, 1)
        slots0.ensure(mx.array([[0, 1]]))
        assert set(slots1._staged) == {4, 5}
        backing.clear()
        assert not slots1._staged
        assert len(slots1.slot_of) == 0
    finally:
        disk.close()


def test_backing_stage_next_prefetches_and_drops(tmp_path):
    """P8 staging: predicted next-layer set reads ahead, joins via the
    same commit path, and mispredicts drop without touching numerics."""
    disk, backing = _backed_disk(tmp_path, dynamic=False)
    try:
        slots0 = disk.language_model.layers[0].ffn.experts.slots
        slots1 = disk.language_model.layers[1].ffn.experts.slots
        slots0.ensure(mx.array([[0, 1]]))  # decode-shaped: records + stages
        assert not slots1._staged  # no predictor yet
        # Inject a predictor for layer 1 whose set is NOT resident:
        # ensure() leaves prev_uniq == resident, so a realistic miss
        # needs a predictor seeded by routing history directly.
        slots1.ensure(mx.array([[6, 7]]))
        backing.note_routing(1, {4, 5})
        _prime_recall(backing, 1)
        slots0.ensure(mx.array([[2, 3]]))  # stages {4,5} on slots1
        assert set(slots1._staged) == {4, 5}
        # Demand ensure: misses join the staged futures (same payload,
        # same commit order — staged_hits counts the prefetch wins).
        rows = slots1.ensure(mx.array([[4, 5]]))
        mx.eval(rows)
        assert slots1.staged_hits == 2
        assert list(slots1.slot_of) == [4, 5]
        # Mispredict: stage {0,1} but demand {4,5}-evicting set {6,7}.
        backing.note_routing(1, {0, 1})
        slots0.ensure(mx.array([[0, 1]]))
        assert set(slots1._staged) == {0, 1}
        slots1.ensure(mx.array([[6, 7]]))
        assert slots1.staged_drops == 2
        assert list(slots1.slot_of) == [6, 7]
    finally:
        disk.close()


def test_backing_stage_disabled_env(tmp_path, monkeypatch):
    """OMLX_V41_STAGE=0 leaves the legacy demand-only path untouched."""
    monkeypatch.setenv("OMLX_V41_STAGE", "0")
    disk, backing = _backed_disk(tmp_path, dynamic=False)
    try:
        slots0 = disk.language_model.layers[0].ffn.experts.slots
        slots1 = disk.language_model.layers[1].ffn.experts.slots
        backing.note_routing(1, {4, 5})
        slots0.ensure(mx.array([[0, 1]]))
        assert not slots1._staged
        rows = slots1.ensure(mx.array([[4, 5]]))
        mx.eval(rows)
        assert slots1.staged_hits == 0
        assert list(slots1.slot_of) == [4, 5]
    finally:
        disk.close()


def test_backing_stage_suppressed_without_headroom(tmp_path):
    """Starved regime (measured: drops 270 > hits 200 on the 48GB/422GB
    run): at the capacity floor or inside the clear band, speculative
    reads only waste saturated disk — the gate skips them."""
    disk, backing = _backed_disk(tmp_path)
    try:
        gov = backing.governor
        assert gov is not None
        gov.min_budget_bytes = 0
        gov.min_cap = 2  # = plan capacity = the fixture working set
        # Keep the desperate-free band off until the explicit check below —
        # the host's real free memory would otherwise suppress staging
        # nondeterministically on a loaded machine.
        gov.low_free_bytes = 0
        slots0 = disk.language_model.layers[0].ffn.experts.slots
        slots1 = disk.language_model.layers[1].ffn.experts.slots
        # First ensure populates gov._last_free_gib via tick->observe.
        slots0.ensure(mx.array([[0, 1]]))
        backing.note_routing(1, {4, 5})
        _prime_recall(backing, 1)  # suppression below is headroom-only
        # At the floor: no slot can hold a misprediction -> suppressed.
        slots0.ensure(mx.array([[0, 1]]))
        assert not ({4, 5} & set(slots1._staged))
        assert backing.staged_skips == 1
        # Spare capacity: staging resumes.
        backing.resize(4, 4)
        slots0.ensure(mx.array([[2, 3]]))
        assert set(slots1._staged) == {4, 5}
        rows = slots1.ensure(mx.array([[4, 5]]))
        mx.eval(rows)
        assert slots1.staged_hits == 2
        # Desperate-free band suppresses even with spare capacity.
        gov.low_free_bytes = 10**18
        backing.note_routing(1, {0, 1})
        slots0.ensure(mx.array([[6, 7]]))
        assert not ({0, 1} & set(slots1._staged))
        assert backing.staged_skips == 2
    finally:
        disk.close()


def test_backing_staged_payload_preserves_arithmetic(tmp_path):
    """Rows committed from staged payloads are bit-equal to the
    resident model's expert weights — prefetch changes WHEN the read
    happens, never WHAT lands in the slot."""
    base = tmp_path / "arith"
    base.mkdir(parents=True, exist_ok=True)
    source, _ = write_checkpoint(
        base, vision=False, n_routed_experts=8,
        n_activated_experts=2,
    )
    resident, _ = load(source)
    disk, _ = load(source, moe_expert_offload_resident_fraction=0.25)
    try:
        layers = [
            (i, layer.ffn.experts.slots)
            for i, layer in enumerate(disk.language_model.layers)
        ]
        backing = V41StreamingBacking(
            disk._moe_offload_plan, layers, dynamic=False
        )
        slots0 = disk.language_model.layers[0].ffn.experts.slots
        slots1 = disk.language_model.layers[1].ffn.experts.slots
        # Prime layer-1 residency away from the staged set, then let a
        # layer-0 ensure stage {4,5} into layer 1.
        slots1.ensure(mx.array([[6, 7]]))
        backing.note_routing(1, {4, 5})
        _prime_recall(backing, 1)
        slots0.ensure(mx.array([[0, 1]]))
        assert set(slots1._staged) == {4, 5}
        slots1.ensure(mx.array([[4, 5]]))
        assert slots1.staged_hits == 2
        resident_experts = resident.language_model.layers[1].ffn.experts
        for expert in (4, 5):
            slot = slots1.slot_of[expert]
            for proj in ("w1", "w3", "w2"):
                np.testing.assert_array_equal(
                    np.array(getattr(slots1.expert, proj).weight[slot]),
                    np.array(getattr(resident_experts, proj).weight[expert]),
                )
    finally:
        resident.close()
        disk.close()


def test_backing_stage_gated_by_recall(tmp_path):
    """Per-layer recall gate (generic stage_gate parity): a layer whose
    prev-token prediction never proved itself does not earn speculative
    reads; once the EWMA crosses the floor staging resumes."""
    disk, backing = _backed_disk(tmp_path, dynamic=False)
    try:
        slots0 = disk.language_model.layers[0].ffn.experts.slots
        slots1 = disk.language_model.layers[1].ffn.experts.slots
        slots1.ensure(mx.array([[6, 7]]))
        backing.note_routing(1, {4, 5})  # predictor set, recall unproven
        slots0.ensure(mx.array([[0, 1]]))
        assert not slots1._staged
        assert backing.staged_skips == 1
        _prime_recall(backing, 1)
        slots0.ensure(mx.array([[2, 3]]))
        assert set(slots1._staged) == {4, 5}
    finally:
        disk.close()


def test_prefill_tail_chunks_are_not_decode(tmp_path):
    """Phase is decided once per call from the pre-chunk route count.

    Fixture capacity 2 / top_k 2 makes every prefill chunk a single row —
    the old per-chunk shape check scored each as a decode visit: governor
    stats, the prev_uniq predictor and staging all saw phantom decode."""
    disk, backing = _backed_disk(tmp_path, dynamic=False)
    try:
        mx.random.seed(3)
        x = mx.random.normal((1, 5, 32))
        mx.eval(disk.language_model.layers[0].ffn(x, None))
        assert backing.stats.decode_layers == 0
        assert not backing.prev_uniq
    finally:
        disk.close()


def test_offload_preserves_dspark_tensors_and_matches(tmp_path):
    """Offload + preserve_mtp loads the draft head and stays exact."""
    source, _ = write_checkpoint(
        tmp_path,
        vision=False,
        n_routed_experts=8,
        n_activated_experts=2,
        preserve_mtp=True,
        n_mtp_layers=3,
        dspark_block_size=3,
        dspark_noise_token_id=2,
        dspark_target_layer_ids=(2, 3, 4),
        dspark_n_routed_experts=2,
        dspark_n_activated_experts=1,
        dspark_markov_rank=32,
        compress_ratios=(0, 2, 2, 1, 1, 0, 0, 0),
        temperature=0,
    )
    resident, _ = load(source, preserve_mtp=True)
    try:
        disk, _ = load(
            source,
            preserve_mtp=True,
            moe_expert_offload_resident_fraction=0.25,
        )
        try:
            # Draft head survived the offloaded load (the fixture stores
            # bare `mtp.*` keys, so draft_bytes — which counts production
            # `language_model.mtp.*` keys — stays 0 here by design).
            assert getattr(disk.language_model, "mtp", None)
            mx.random.seed(0)
            x = mx.random.normal((1, 4, 32))
            for layer in range(2):
                out_r = resident.language_model.layers[layer].ffn(x, None)
                mx.eval(out_r)
                with verify_scope():
                    out_d = disk.language_model.layers[layer].ffn(x, None)
                    mx.eval(out_d)
                np.testing.assert_allclose(
                    np.array(out_r), np.array(out_d), rtol=2e-4, atol=2e-4
                )
        finally:
            disk.close()
    finally:
        resident.close()


def test_verify_visits_counted_separately(tmp_path):
    """F0 telemetry: verify ensures count verify_visits, never decode
    visits, and book-level verify_misses accumulate per expert."""
    disk, backing = _backed_disk(tmp_path, dynamic=False)
    try:
        slots = disk.language_model.layers[0].ffn.experts.slots
        with verify_scope():
            slots.ensure(mx.array([[0, 1]]))
            slots.ensure(mx.array([[0, 1]]))  # verify hit: visited, no miss
        assert backing.stats.verify_visits == 2
        assert backing.stats.verify_visits_missed == 1
        assert backing.stats.decode_layers == 0
        assert slots.book.verify_misses == 2
        assert slots.book.verify_installed == {0, 1}
        summary = backing.summary()
        assert summary["verify_visits"] == 2
        assert summary["verify_visits_missed"] == 1
        assert summary["verify_misses"] == 2
    finally:
        disk.close()


def test_verify_evict_resident_counted(tmp_path, monkeypatch):
    """F0 telemetry: evicting a pre-verify resident during verify counts
    as verify_evict_resident; evicting a verify-installed row does not.
    (scratch=0: verify still evicts, so the counter is exercised.)"""
    monkeypatch.setenv("OMLX_V41_VERIFY_SCRATCH", "0")
    disk, backing = _backed_disk(tmp_path, dynamic=False)
    try:
        slots = disk.language_model.layers[0].ffn.experts.slots
        slots.ensure(mx.array([0, 1]))  # decode residents, cap=2
        with verify_scope():
            slots.ensure(mx.array([2]))  # evicts 0 (decode resident)
            assert slots.book.verify_evict_resident == 1
            slots.ensure(mx.array([3]))  # evicts 2 (verify-installed)
            assert slots.book.verify_evict_resident == 1
            assert slots.book.verify_installed == {3}
        assert backing.summary()["verify_evict_resident"] == 1
    finally:
        disk.close()


def test_staged_skip_reasons_split(tmp_path):
    """F0 telemetry: staged_skips split by cause while the aggregate
    stays the sum of the gated skips (bench JSON compat)."""
    disk, backing = _backed_disk(tmp_path, dynamic=False)
    try:
        slots0 = disk.language_model.layers[0].ffn.experts.slots
        slots0.ensure(mx.array([[0, 1]]))
        # No predictor for layer 1 yet -> no_pred bucket (not a gate skip).
        assert backing.staged_skips_no_pred == 1
        assert backing.staged_skips == 0
        # Predictor present, recall unproven -> recall bucket.
        disk.language_model.layers[1].ffn.experts.slots.ensure(
            mx.array([[6, 7]])
        )
        backing.note_routing(1, {4, 5})
        slots0.ensure(mx.array([[2, 3]]))
        assert backing.staged_skips_recall == 1
        assert backing.staged_skips == 1
        # Recall primed, staging submits -> staged_submits counts entries.
        _prime_recall(backing, 1)
        slots0.ensure(mx.array([[0, 1]]))
        slots1 = disk.language_model.layers[1].ffn.experts.slots
        assert set(slots1._staged) == {4, 5}
        assert slots1.staged_submits == 2
        assert backing.summary()["staged_submits"] == 2
    finally:
        disk.close()


def test_verify_feeds_predictor_and_advises_next(tmp_path):
    """F3: verify traffic records per-round routed unions; a layer-index
    decrease wraps the round and promotes the accumulated unions —
    the staged_hits=0 bug was verify starving every predictor. (The
    advisory itself is converted-checkpoint only; the converted-path
    coverage lives in test_deepseek_v41_span_staging.py.)"""
    disk, backing = _backed_disk(tmp_path, dynamic=False)
    try:
        slots0 = disk.language_model.layers[0].ffn.experts.slots
        slots1 = disk.language_model.layers[1].ffn.experts.slots
        with verify_scope():
            # Round 1: layer0 {0,1}, layer1 {3,4} then {5,6} (cap=2
            # churns the verify-installed rows, union accumulates 4).
            slots0.ensure(mx.array([0, 1]))
            slots1.ensure(mx.array([3, 4]))
            slots1.ensure(mx.array([5, 6]))
            assert backing.verify_uniq == {}
            assert backing.verify_acc[1] == {3, 4, 5, 6}
            # Round 2 begins: layer index decrease wraps the round and
            # promotes acc -> uniq.
            slots0.ensure(mx.array([0, 1]))
            assert backing.verify_uniq[0] == {0, 1}
            assert backing.verify_uniq[1] == {3, 4, 5, 6}
            # Round-1 coverage of round-2's {0,1} is total -> recall 0.1.
            assert backing.verify_recall[0] == pytest.approx(0.1)
        # Source checkpoint: advise_demand is a no-op (no stacked
        # store), but the gating path still counts the attempt. Round 1
        # ran with an empty predictor (3 ensures -> 3 no_pred skips) and
        # the wrap ensure hit the recall gate once (recall[1] was 0).
        assert backing.verify_ra_no_pred == 3
        assert backing.verify_ra_recall == 1
        backing.verify_recall[1] = 1.0  # skip EWMA warm-up
        with verify_scope():
            slots0.ensure(mx.array([0, 1]))
        # Gate passed this time (no new skip) even though the source
        # path's advise itself is a no-op.
        assert backing.verify_ra_recall == 1
        assert backing.verify_ra_no_pred == 3
    finally:
        disk.close()


def test_verify_advise_gates_on_recall(tmp_path):
    """F3: a next-layer prediction with unproven recall is skipped and
    the cause is visible in the split counters."""
    disk, backing = _backed_disk(tmp_path, dynamic=False)
    try:
        slots0 = disk.language_model.layers[0].ffn.experts.slots
        slots1 = disk.language_model.layers[1].ffn.experts.slots
        with verify_scope():
            slots0.ensure(mx.array([0, 1]))
            slots1.ensure(mx.array([3, 4]))
            slots0.ensure(mx.array([0, 1]))  # wrap: uniq[1] = {3,4}
        # No recall history for layer1: the wrap ensure's advise and
        # this next one both stop at the recall gate.
        before = slots1.ra_calls
        with verify_scope():
            slots0.ensure(mx.array([0, 1]))
        assert slots1.ra_calls == before
        assert backing.verify_ra_recall == 2
        assert backing.verify_ra_submits == 0
    finally:
        disk.close()


def test_verify_advise_no_pred_and_disabled(tmp_path, monkeypatch):
    """F3: empty predictor counts no_pred skips; OMLX_V41_VERIFY_RA=0
    disables the advisory entirely."""
    disk, backing = _backed_disk(tmp_path, dynamic=False)
    try:
        slots0 = disk.language_model.layers[0].ffn.experts.slots
        with verify_scope():
            slots0.ensure(mx.array([0, 1]))
        assert backing.verify_ra_no_pred == 1  # verify_uniq still empty
        monkeypatch.setenv("OMLX_V41_VERIFY_RA", "0")
        backing.verify_uniq[1] = {3, 4}
        backing.verify_recall[1] = 1.0
        with verify_scope():
            slots0.ensure(mx.array([0, 1]))
        assert backing.verify_ra_submits == 0
    finally:
        disk.close()


def test_verify_scratch_holds_union_and_protects_residents(
    tmp_path, monkeypatch
):
    """F4: scratch rows admit the whole verify union in ONE ensure;
    victims come only from the verify-installed region — decode-hot
    residents are never consumed. The decode-phase trim reclaims the
    scratch occupancy afterwards."""
    monkeypatch.setenv("OMLX_V41_VERIFY_SCRATCH", "6")
    disk, backing = _backed_disk(tmp_path, dynamic=False)
    try:
        slots = disk.language_model.layers[0].ffn.experts.slots
        slots.ensure(mx.array([0, 1]))  # decode residents, cap=2
        rooms0 = slots.rooms
        with verify_scope():
            # union=6 > cap=2 — only servable through scratch rows.
            slots.ensure(mx.array([2, 3, 4, 5, 6, 7]))
        assert set(slots.slot_of) == {0, 1, 2, 3, 4, 5, 6, 7}
        assert slots.rooms == rooms0 + 6
        assert slots.book.verify_evict_resident == 0
        assert slots.book.verify_installed == {2, 3, 4, 5, 6, 7}
        # Cross-iteration reuse: the same union hits fully next round.
        with verify_scope():
            hits0 = slots.book.hits
            slots.ensure(mx.array([2, 3, 4, 5, 6, 7]))
            assert slots.book.hits - hits0 == 6
        # Ordinary decode traffic trims scratch occupancy back to cap —
        # the cold-end verify rows leave first, decode-hot stays.
        slots.ensure(mx.array([0, 1]))
        assert set(slots.slot_of) == {0, 1}
    finally:
        disk.close()


def test_verify_scratch_widens_chunk_step(tmp_path, monkeypatch):
    """F4: a verify block's whole union fits one ensure — the chunk
    loop stops re-splitting into one-token ensures."""
    monkeypatch.setenv("OMLX_V41_VERIFY_SCRATCH", "6")
    disk, _backing = _backed_disk(tmp_path, dynamic=False)
    try:
        ffn = disk.language_model.layers[0].ffn
        slots = disk.language_model.layers[0].ffn.experts.slots
        calls = []
        orig = slots.ensure

        def spy(idx, phase=None):
            calls.append(int(idx.shape[0]))
            return orig(idx, phase=phase)

        slots.ensure = spy
        mx.random.seed(0)
        x = mx.random.normal((1, 4, 32))
        with verify_scope():
            mx.eval(ffn(x, None))
        slots.ensure = orig
        assert calls == [4]  # one ensure, not [1]*4
    finally:
        disk.close()


def test_verify_scratch_bound_degrades_gracefully(tmp_path, monkeypatch):
    """F4: a scratch bound below the union still works — chunks of
    scratch//top_k rows each, victims still verify-installed first."""
    monkeypatch.setenv("OMLX_V41_VERIFY_SCRATCH", "2")  # limit = 4
    disk, _backing = _backed_disk(tmp_path, dynamic=False)
    try:
        ffn = disk.language_model.layers[0].ffn
        slots = disk.language_model.layers[0].ffn.experts.slots
        slots.ensure(mx.array([0, 1]))  # decode residents
        calls = []
        orig = slots.ensure

        def spy(idx, phase=None):
            calls.append(int(idx.shape[0]))
            return orig(idx, phase=phase)

        slots.ensure = spy
        mx.random.seed(0)
        x = mx.random.normal((1, 4, 32))
        with verify_scope():
            mx.eval(ffn(x, None))
        slots.ensure = orig
        assert calls == [2, 2]  # step = (2+2)//2 = 2 tokens per ensure
        # Decode residents survived: limit 4 < union ~8 churns only the
        # verify-installed rows after the first ensure.
        assert slots.book.verify_evict_resident <= 2
    finally:
        disk.close()


def test_verify_scratch_zero_keeps_chunked(tmp_path, monkeypatch):
    """F4: OMLX_V41_VERIFY_SCRATCH=0 preserves the legacy chunked
    verify behavior — the default keeps bit-exact parity with today."""
    monkeypatch.setenv("OMLX_V41_VERIFY_SCRATCH", "0")
    disk, _backing = _backed_disk(tmp_path, dynamic=False)
    try:
        ffn = disk.language_model.layers[0].ffn
        slots = disk.language_model.layers[0].ffn.experts.slots
        calls = []
        orig = slots.ensure

        def spy(idx, phase=None):
            calls.append(int(idx.shape[0]))
            return orig(idx, phase=phase)

        slots.ensure = spy
        mx.random.seed(0)
        x = mx.random.normal((1, 4, 32))
        with verify_scope():
            mx.eval(ffn(x, None))
        slots.ensure = orig
        assert calls == [1, 1, 1, 1]
    finally:
        disk.close()


_DRAFT = dict(
    preserve_mtp=True,
    n_mtp_layers=3,
    dspark_block_size=3,
    dspark_noise_token_id=2,
    dspark_target_layer_ids=(2, 3, 4),
    dspark_n_routed_experts=4,
    dspark_n_activated_experts=2,
    dspark_markov_rank=32,
    compress_ratios=(0, 2, 2, 1, 1, 0, 0, 0),
    temperature=0,
)


def test_draft_offload_resident_when_disabled(tmp_path, monkeypatch):
    """F5: OMLX_V41_DRAFT_OFFLOAD=0 keeps the DSpark head resident —
    mtp.*.ffn.experts stays a plain Expert module and no draft expert
    keys are excluded from the normal load."""
    monkeypatch.setenv("OMLX_V41_DRAFT_OFFLOAD", "0")
    source, _ = write_checkpoint(
        tmp_path,
        vision=False,
        n_routed_experts=8,
        n_activated_experts=2,
        **_DRAFT,
    )
    disk, _ = load(
        source, preserve_mtp=True, moe_expert_offload_resident_fraction=0.25
    )
    try:
        from omlx.patches.deepseek_v41.moe_offload import OffloadedExpert

        assert disk._moe_offload_plan.draft_rows == 0
        for stage in disk.language_model.mtp:
            assert not isinstance(stage.ffn.experts, OffloadedExpert)
    finally:
        disk.close()


def test_draft_offload_default_wraps_and_clamps(tmp_path, monkeypatch):
    """F5: the default (8 rows/stage) wraps mtp.*.ffn.experts in
    OffloadedExpert; a bank smaller than the default clamps to full
    residency instead of failing."""
    monkeypatch.delenv("OMLX_V41_DRAFT_OFFLOAD", raising=False)
    source, _ = write_checkpoint(
        tmp_path,
        vision=False,
        n_routed_experts=8,
        n_activated_experts=2,
        **_DRAFT,
    )
    disk, _ = load(
        source, preserve_mtp=True, moe_expert_offload_resident_fraction=0.25
    )
    try:
        from omlx.patches.deepseek_v41.moe_offload import OffloadedExpert

        # 4-expert fixture bank < default 8 -> clamps to 4 (resident).
        assert disk._moe_offload_plan.draft_rows == 4
        for stage in disk.language_model.mtp:
            experts = stage.ffn.experts
            assert isinstance(experts, OffloadedExpert)
            assert experts.slots.cap == 4
            assert experts.slots.backing is None
    finally:
        disk.close()


def test_draft_offload_wraps_stages_and_stays_exact(tmp_path, monkeypatch):
    """F5: OMLX_V41_DRAFT_OFFLOAD wraps mtp.*.ffn.experts in
    OffloadedExpert with its own row budget — outside the trunk backing
    — and the draft ffn output matches the fully resident model."""
    from omlx.patches.deepseek_v41.moe_offload import OffloadedExpert

    source, _ = write_checkpoint(
        tmp_path,
        vision=False,
        n_routed_experts=8,
        n_activated_experts=2,
        **_DRAFT,
    )
    resident, _ = load(source, preserve_mtp=True)
    try:
        monkeypatch.setenv("OMLX_V41_DRAFT_OFFLOAD", "2")
        disk, _ = load(
            source,
            preserve_mtp=True,
            moe_expert_offload_resident_fraction=0.25,
        )
        try:
            plan = disk._moe_offload_plan
            assert plan.draft_rows == 2
            assert plan.draft_full_bytes > 0
            assert plan.draft_resident_bytes < plan.draft_full_bytes
            for stage in disk.language_model.mtp:
                experts = stage.ffn.experts
                assert isinstance(experts, OffloadedExpert)
                assert experts.slots.cap == 2
                # Draft slots never join the trunk governor.
                assert experts.slots.backing is None
            mx.random.seed(0)
            x = mx.random.normal((1, 4, 32))
            for i in range(3):
                out_r = resident.language_model.mtp[i].ffn(x, None)
                out_d = disk.language_model.mtp[i].ffn(x, None)
                mx.eval(out_r)
                mx.eval(out_d)
                np.testing.assert_allclose(
                    np.array(out_r),
                    np.array(out_d),
                    rtol=2e-4,
                    atol=2e-4,
                )
        finally:
            disk.close()
    finally:
        resident.close()


def test_draft_offload_validates_rows(tmp_path, monkeypatch):
    """F5: the row budget must hold the activated set and fit the bank;
    without a preserved head the env is ignored entirely."""
    (tmp_path / "ckpt").mkdir()
    source, _ = write_checkpoint(
        tmp_path / "ckpt",
        vision=False,
        n_routed_experts=8,
        n_activated_experts=2,
        **_DRAFT,
    )
    monkeypatch.setenv("OMLX_V41_DRAFT_OFFLOAD", "1")  # below top-2
    with pytest.raises(ValueError, match="DRAFT_OFFLOAD"):
        load(
            source,
            preserve_mtp=True,
            moe_expert_offload_resident_fraction=0.25,
        )
    # Above the 4-expert bank clamps to full residency — never an error.
    monkeypatch.setenv("OMLX_V41_DRAFT_OFFLOAD", "5")
    disk, _ = load(
        source, preserve_mtp=True, moe_expert_offload_resident_fraction=0.25
    )
    try:
        assert disk._moe_offload_plan.draft_rows == 4
    finally:
        disk.close()
    # No preserved head: the knob is inert (nothing to offload).
    (tmp_path / "plain").mkdir()
    plain, _ = write_checkpoint(
        tmp_path / "plain",
        vision=False,
        n_routed_experts=8,
        n_activated_experts=2,
    )
    monkeypatch.setenv("OMLX_V41_DRAFT_OFFLOAD", "2")
    disk, _ = load(plain, moe_expert_offload_resident_fraction=0.25)
    try:
        assert disk._moe_offload_plan.draft_rows == 0
    finally:
        disk.close()


def test_draft_offload_converted_checkpoint(tmp_path, monkeypatch):
    """F5 converted path: stacked mtp.* expert rows ride the shared
    backing store like trunk experts — fetch returns the resident
    model's rows bit-for-bit."""
    from omlx.patches.deepseek_v41.convert import convert
    from omlx.patches.deepseek_v41.moe_offload import OffloadedExpert

    (tmp_path / "ckpt").mkdir()
    source, _ = write_checkpoint(
        tmp_path / "ckpt",
        vision=False,
        n_routed_experts=8,
        n_activated_experts=2,
        **_DRAFT,
    )
    target = tmp_path / "converted"
    convert(source, target, preserve_mtp=True)
    resident, _ = load(target, preserve_mtp=True)
    try:
        monkeypatch.setenv("OMLX_V41_DRAFT_OFFLOAD", "2")
        disk, _ = load(
            target,
            preserve_mtp=True,
            moe_expert_offload_resident_fraction=0.25,
        )
        try:
            experts = disk.language_model.mtp[0].ffn.experts
            assert isinstance(experts, OffloadedExpert)
            slots = experts.slots
            mx.random.seed(0)
            x = mx.random.normal((1, 4, 32))
            mx.eval(disk.language_model.mtp[0].ffn(x, None))
            assert slots.slot_of  # offloaded rows served the call
            resident_experts = resident.language_model.mtp[0].ffn.experts
            for expert, row in slots.slot_of.items():
                np.testing.assert_array_equal(
                    np.array(slots.expert.w1.weight[row]),
                    np.array(resident_experts.w1.weight[expert]),
                )
        finally:
            disk.close()
    finally:
        resident.close()


def test_verify_hunger_bridge_fraction(tmp_path, monkeypatch):
    """F6: OMLX_V41_VERIFY_HUNGER bridges a fraction of verify traffic
    into the governor's decode-visit stream — 0 keeps verify
    telemetry-only, 1.0 feeds every visit, 0.5 emits every other."""
    disk, backing = _backed_disk(tmp_path, dynamic=False)
    try:
        slots = disk.language_model.layers[0].ffn.experts.slots
        with verify_scope():
            slots.ensure(mx.array([[0, 1]]))
        assert backing.stats.decode_layers == 0  # default: no bridge
        monkeypatch.setenv("OMLX_V41_VERIFY_HUNGER", "0.5")
        with verify_scope():
            for _ in range(4):
                slots.ensure(mx.array([[0, 1]]))
        assert backing.stats.decode_layers == 2  # every other visit
        monkeypatch.setenv("OMLX_V41_VERIFY_HUNGER", "1.0")
        with verify_scope():
            slots.ensure(mx.array([[0, 1]]))
            slots.ensure(mx.array([[2, 3]]))  # missed -> hunger sees it
        assert backing.stats.decode_layers == 4
        assert backing.stats.decode_layers_missed >= 1
        # Verify counters keep counting independently of the bridge.
        assert backing.stats.verify_visits == 7
    finally:
        disk.close()


def test_verify_advise_headroom_gate(tmp_path):
    """F6: the verify advisory obeys the same headroom discipline as
    decode staging — at the capacity floor it skips and counts the
    cause; headroom restores it."""
    disk, backing = _backed_disk(tmp_path)
    try:
        gov = backing.governor
        assert gov is not None
        gov.min_budget_bytes = 0
        gov.min_cap = 2  # = plan capacity = floor -> no staging slack
        slots0 = disk.language_model.layers[0].ffn.experts.slots
        backing.verify_uniq[1] = {3, 4}
        backing.verify_recall[1] = 1.0
        with verify_scope():
            slots0.ensure(mx.array([[0, 1]]))
        assert backing.verify_ra_headroom == 1
        assert backing.verify_ra_submits == 0
        # Spare capacity: the gate opens again (source checkpoint makes
        # advise_demand itself a no-op, so no submit is counted — the
        # signal is that headroom stops accumulating).
        backing.resize(4, 4)
        with verify_scope():
            slots0.ensure(mx.array([[0, 1]]))
        assert backing.verify_ra_headroom == 1
    finally:
        disk.close()
