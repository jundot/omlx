# SPDX-License-Identifier: Apache-2.0
"""Robustness contracts: DSv4.1 backing rollback/atomicity/lock/floor/
close, legacy cache fetch-first install + serialization marker,
cold-reader cleanup, atomic transition-profile writes."""

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import mlx.core as mx
import pytest
from test_deepseek_v41 import write_checkpoint


def _loaded_v41(tmp_path, *, n_routed=16, n_activated=4, fraction=0.25):
    source, _ = write_checkpoint(
        tmp_path,
        vision=False,
        n_routed_experts=n_routed,
        n_activated_experts=n_activated,
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        model, _ = executor.submit(
            load_v41, source, fraction
        ).result()
    return model


def load_v41(source, fraction):
    from omlx.patches.deepseek_v41.loading import load

    return load(source, moe_expert_offload_resident_fraction=fraction)


def _slots(model, layer=0):
    return model.language_model.layers[layer].ffn.experts.slots


class TestV41FetchRollback:
    """A fetch failure must not orphan rows or drop the victim."""

    def test_failed_fetch_restores_victim(self, tmp_path):
        model = _loaded_v41(tmp_path)
        try:
            slots = _slots(model)
            # Fill the cache, then miss: the miss evicts the LRU victim
            # before fetching — the failing fetch must restore it.
            slots.ensure(mx.array([[0, 1]]))
            slots.ensure(mx.array([[2, 3]]))
            victim = next(iter(slots.slot_of))
            before = dict(slots.slot_of)

            orig = slots.plan.fetch

            def _boom(prefix, proj, expert):
                if expert == 9:
                    raise OSError("injected shard failure")
                return orig(prefix, proj, expert)

            slots.plan.fetch = _boom
            with pytest.raises(OSError, match="injected"):
                slots.ensure(mx.array([[9, 0]]))
            # Every row accounted for; the evicted victim kept residency
            # (its bytes were never overwritten).
            assert len(slots.slot_of) + len(slots.free) == slots.rooms
            assert victim in slots.slot_of
            assert set(slots.slot_of) == set(before)
        finally:
            model.close()

    def test_failed_fetch_keeps_free_rows_consistent(self, tmp_path):
        model = _loaded_v41(tmp_path)
        try:
            slots = _slots(model)
            slots.ensure(mx.array([[0, 1]]))  # fill 2 of cap rooms
            free_before = sorted(slots.free)

            def _boom(prefix, proj, expert):
                raise OSError("injected shard failure")

            slots.plan.fetch = _boom
            with pytest.raises(OSError):
                slots.ensure(mx.array([[7]]))
            assert len(slots.slot_of) + len(slots.free) == slots.rooms
            assert sorted(slots.free) == free_before  # fresh row returned
            assert 7 not in slots.slot_of
        finally:
            model.close()


class TestV41GovernorFloor:
    """The dynamic floor is one decode working set, not min(8, cap)."""

    def test_floor_tracks_n_activated(self, tmp_path):
        from omlx.patches.deepseek_v41.streaming_backing import (
            V41StreamingBacking,
        )
        from omlx.patches.deepseek_v41.moe_offload import OffloadedExpert

        model = _loaded_v41(tmp_path, n_routed=16, n_activated=10)
        try:
            layers = [
                (i, l.ffn.experts.slots)
                for i, l in enumerate(model.language_model.layers)
                if isinstance(getattr(l.ffn, "experts", None), OffloadedExpert)
            ]
            backing = V41StreamingBacking(
                model._moe_offload_plan, layers, dynamic=True
            )
            try:
                assert backing.governor is not None
                # Slot floor = one decode working set — a smaller floor
                # would shrink top-k 10 models below the working set and
                # _ensure_locked would raise mid-generation.
                assert backing.governor.min_cap == 10
                assert backing.governor._min_cap_slots() >= 10
            finally:
                backing.close()
        finally:
            model.close()

    def test_capacity_property_and_guard_info(self, tmp_path):
        from omlx.patches.deepseek_v41.streaming_backing import (
            V41StreamingBacking,
        )
        from omlx.patches.deepseek_v41.moe_offload import OffloadedExpert

        model = _loaded_v41(tmp_path)
        try:
            layers = [
                (i, l.ffn.experts.slots)
                for i, l in enumerate(model.language_model.layers)
                if isinstance(getattr(l.ffn, "experts", None), OffloadedExpert)
            ]
            backing = V41StreamingBacking(
                model._moe_offload_plan, layers, dynamic=False
            )
            # capacity resolves to base_cap through a single property.
            assert backing.capacity == backing.base_cap
            # Absent on purpose — V4.1 slots are persistent buffers, not
            # the generic path's lazy mini-banks; the scheduler reads the
            # missing attr as None.
            assert getattr(backing, "streaming_guard_info", None) is None
            backing.close()
            assert model._moe_offload_plan._closed
            backing.close()  # idempotent
        finally:
            model.close()


class TestV41CompactAtomic:
    """compact/_grow evaluate all projections before rebinding."""

    def test_compact_failure_leaves_consistent_state(self, tmp_path, monkeypatch):
        model = _loaded_v41(tmp_path)
        try:
            slots = _slots(model)
            slots.ensure(mx.array([[0, 1]]))
            slots.ensure(mx.array([[2, 3]]))
            before = dict(slots.slot_of)

            orig_stack = mx.stack

            def _boom(*a, **k):
                raise RuntimeError("injected compact failure")

            monkeypatch.setattr(mx, "stack", _boom)
            with pytest.raises(RuntimeError, match="injected compact"):
                slots.compact(1)
            monkeypatch.undo()
            # Nothing rebound: slot map untouched, module still serves.
            assert slots.slot_of == before
            rows = slots.ensure(mx.array([[0, 1]]))
            mx.eval(rows)
        finally:
            model.close()


class TestLegacyCacheAtomicity:
    """Fetch-first install + a real lock on the legacy cache."""

    def _cache(self, tmp_path, capacity=2, n=4):
        from mlx_lm.models.switch_layers import SwitchGLU
        import mlx.nn as nn

        from omlx.patches.moe_expert_offload import (
            CheckpointExpertStore,
            ExpertCache,
            _GLUStoreView,
        )

        glu = SwitchGLU(32, 32, n)
        nn.quantize(glu, group_size=32, bits=4)
        tensors = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(glu, proj)
            for field in ("weight", "scales", "biases"):
                if lin.get(field) is not None:
                    tensors[f"layers.0.mlp.switch_glu.{proj}.{field}"] = lin[field]
        mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)
        store = CheckpointExpertStore(tmp_path)
        view = _GLUStoreView(store, "layers.0.mlp.switch_glu")
        return ExpertCache(glu, capacity, view)

    def test_failed_install_mutates_nothing(self, tmp_path):
        cache = self._cache(tmp_path)
        cache.ensure(mx.array([0]))  # one resident
        before_slots = dict(cache.slot_of)
        before_free = list(cache.free)

        orig = cache.disk.fetch
        calls = {"n": 0}

        def _boom(proj, field, expert):
            calls["n"] += 1
            if calls["n"] > 2:
                raise OSError("injected")
            return orig(proj, field, expert)

        cache.disk.fetch = _boom
        with pytest.raises(OSError):
            cache._install(1)
        assert cache.slot_of == before_slots
        assert cache.free == before_free
        assert 1 not in cache.slot_of

    def test_legacy_apply_stamps_serialization_marker(self, tmp_path):
        import mlx.nn as nn
        from mlx_lm.models.switch_layers import SwitchGLU

        from omlx.patches.moe_expert_offload import _apply_legacy_adapter
        from omlx.scheduler import _model_uses_expert_streaming

        glu = SwitchGLU(32, 32, 4)
        nn.quantize(glu, group_size=32, bits=4)
        tensors = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(glu, proj)
            for field in ("weight", "scales", "biases"):
                if lin.get(field) is not None:
                    tensors[f"layers.0.mlp.switch_glu.{proj}.{field}"] = lin[field]
        mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)

        class _GLU(nn.Module):
            def __init__(self, g):
                super().__init__()
                self.switch_glu = g

        class _Layer(nn.Module):
            def __init__(self, g):
                super().__init__()
                self.mlp = _GLU(g)

        class _Model(nn.Module):
            def __init__(self, g):
                super().__init__()
                self.layers = [_Layer(g)]

        model = _Model(glu)
        wrapped = _apply_legacy_adapter(model, tmp_path, 0.5)
        assert wrapped == 1
        # Requests must serialize: the marker is what the scheduler checks.
        assert _model_uses_expert_streaming(model) is True


class TestShardBankClose:
    """The cold-tier key memo must not hand back closed readers."""

    def test_close_clears_cold_key_map(self, tmp_path):
        from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore

        # Minimal checkpoint: the store only needs a readable header.
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "x"}))
        mx.save_safetensors(
            str(tmp_path / "model.safetensors"), {"w": mx.zeros((2, 2))}
        )
        store = ExpertBackingStore(tmp_path)
        store._cold_key_to_reader["some.key"] = SimpleNamespace()
        store.close()
        assert store._cold_key_to_reader == {}


class TestTransitionProfileAtomic:
    """The profile write is tmp+replace, never a truncated dest."""

    def test_save_is_atomic(self, tmp_path):
        from omlx.patches.expert_streaming import save_transition_profile

        (tmp_path / "config.json").write_text(json.dumps({"model_type": "x"}))

        class _Spec:
            trans_updates = 3

            def to_payload(self):
                return {"regimes": {}, "trans_updates": 3}

        backing = SimpleNamespace(spec_state=_Spec(), model_path=str(tmp_path))
        save_transition_profile(backing)
        dest = tmp_path / ".omlx" / "expert_transition.json"
        assert dest.is_file()
        assert not (tmp_path / ".omlx" / "expert_transition.json.tmp").exists()
        payload = json.loads(dest.read_text())
        assert payload["trans_updates"] == 3


class TestLegacyCacheResize:
    """Governor-facing resize/clear on the legacy per-layer cache."""

    def _cache(self, tmp_path, capacity=4, n=8):
        import mlx.nn as nn
        from mlx_lm.models.switch_layers import SwitchGLU

        from omlx.patches.moe_expert_offload import (
            CheckpointExpertStore,
            ExpertCache,
            _GLUStoreView,
        )

        glu = SwitchGLU(32, 32, n)
        nn.quantize(glu, group_size=32, bits=4)
        tensors = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(glu, proj)
            for field in ("weight", "scales", "biases"):
                if lin.get(field) is not None:
                    tensors[f"layers.0.mlp.switch_glu.{proj}.{field}"] = lin[field]
        mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)
        store = CheckpointExpertStore(tmp_path)
        view = _GLUStoreView(store, "layers.0.mlp.switch_glu")
        return ExpertCache(glu, capacity, view)

    def test_shrink_evicts_lru_and_caps_installs(self, tmp_path):
        cache = self._cache(tmp_path)
        try:
            cache.ensure(mx.array([0, 1, 2, 3]))  # fill all 4 rooms
            cache.resize(2)
            assert cache.capacity == 2
            assert len(cache.slot_of) == 2
            assert 0 not in cache.slot_of and 1 not in cache.slot_of
            # Ceiling is a count, not a row range: a free row exists but
            # the install must still evict the LRU victim.
            cache.ensure(mx.array([4]))
            assert 4 in cache.slot_of
            assert len(cache.slot_of) == 2
            assert len(cache.slot_of) + len(cache.free) == cache.rooms
        finally:
            cache.disk._store.close()

    def test_grow_reallocs_and_installs_land(self, tmp_path):
        cache = self._cache(tmp_path, capacity=4, n=8)
        try:
            cache.ensure(mx.array([0, 1]))
            cache.resize(6)
            assert cache.rooms == 6
            assert cache.capacity == 6
            cache.ensure(mx.array([5, 6]))
            assert 5 in cache.slot_of and 6 in cache.slot_of
            assert len(cache.slot_of) + len(cache.free) == cache.rooms
            # Beyond n_experts clamps.
            cache.resize(64)
            assert cache.capacity == 8
            assert cache.rooms == 8
        finally:
            cache.disk._store.close()

    def test_clear_drops_residency(self, tmp_path):
        cache = self._cache(tmp_path)
        try:
            cache.ensure(mx.array([0, 1]))
            cache.clear()
            assert cache.slot_of == {}
            assert len(cache.free) == cache.rooms
            assert cache.warm is False
        finally:
            cache.disk._store.close()


class TestLegacyOffloadState:
    """The legacy aggregate presents the V4.1 governor duck-type."""

    def _module(self, tmp_path, capacity=4, n=8):
        import mlx.nn as nn
        from mlx_lm.models.switch_layers import SwitchGLU

        from omlx.patches.moe_expert_offload import (
            CheckpointExpertStore,
            OffloadSwitchGLU,
            _GLUStoreView,
        )

        glu = SwitchGLU(32, 32, n)
        nn.quantize(glu, group_size=32, bits=4)
        tensors = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(glu, proj)
            for field in ("weight", "scales", "biases"):
                if lin.get(field) is not None:
                    tensors[f"layers.0.mlp.switch_glu.{proj}.{field}"] = lin[field]
        mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)
        store = CheckpointExpertStore(tmp_path)
        view = _GLUStoreView(store, "layers.0.mlp.switch_glu")
        return OffloadSwitchGLU(glu, capacity, view), store

    def test_governor_duck_and_stats(self, tmp_path):
        from omlx.patches.moe_expert_offload import LegacyOffloadState

        mod, store = self._module(tmp_path)
        try:
            state = LegacyOffloadState([mod.cache], dynamic=True, min_cap=2)
            assert state.governor is not None
            assert state.governor.min_cap == 2
            state.note_visit(0, True)
            state.note_visit(0, False)
            assert state.stats.decode_layers == 2
            assert state.stats.decode_layers_missed == 1
            assert state.stats.decode_misses_by_layer == {0: 1}
            state.resize(2, None)
            assert mod.cache.capacity == 2
            state.set_layer_caps({0: 3})
            assert mod.cache.capacity == 3
            assert state.layer_cap_overrides() == {0: 3}
            state.clear()
            assert mod.cache.slot_of == {}
            summary = state.summary()
            assert summary["layers"] == 1
            # Persistent slot buffers: no mini-bank transient term.
            assert getattr(state, "streaming_guard_info", None) is None
        finally:
            store.close()

    def test_decode_visit_feeds_stats(self, tmp_path):
        from omlx.patches.moe_expert_offload import LegacyOffloadState

        mod, store = self._module(tmp_path)
        try:
            state = LegacyOffloadState([mod.cache], dynamic=False)
            mod._state = state
            mod._layer = 0
            out = mod(mx.zeros((1, 32)), mx.array([[0]]))
            mx.eval(out)
            assert state.stats.decode_layers == 1
            assert state.stats.decode_layers_missed == 1  # cold miss
            # Multi-token call is not a decode visit.
            out = mod(mx.zeros((3, 32)), mx.array([[0], [1], [0]]))
            mx.eval(out)
            assert state.stats.decode_layers == 1
        finally:
            store.close()


class TestResolveBudgetBytes:
    """Public budget resolver for the admission path."""

    def test_pin_auto_and_zero(self):
        from omlx.model_settings import ModelSettings
        from omlx.patches.expert_streaming import resolve_budget_bytes

        assert resolve_budget_bytes(
            ModelSettings(expert_streaming_budget_gib=1.5)
        ) == int(1.5 * 1024**3)
        assert (
            resolve_budget_bytes(
                ModelSettings(expert_streaming_budget_auto=False)
            )
            == 0
        )
        assert resolve_budget_bytes(None) == 0
        # MiB spellings exist only on dict-shaped/legacy settings objects.
        assert resolve_budget_bytes(
            SimpleNamespace(expert_streaming_budget_mib=512)
        ) == 512 * 1024 * 1024
        # Beyond the 64 GiB ceiling clamps.
        assert resolve_budget_bytes(
            ModelSettings(expert_streaming_budget_gib=1000)
        ) == 64 * 1024**3


class TestSummaryMergesBacking:
    """Cache-less backings (V4.1/legacy) must not log a permanent 0."""

    def test_backing_summary_folds_into_lru_slots(self):
        from omlx.patches.expert_streaming import expert_streaming_summary

        backing = SimpleNamespace(
            summary=lambda: {
                "hits": 3,
                "misses": 1,
                "evictions": 2,
                "resident": 4,
                "capacity_per_layer": 8,
                "layers": 2,
                "governor": {},
            }
        )
        out = expert_streaming_summary(None, backing)
        assert out["lru_hits"] == 3
        assert out["lru_misses"] == 1
        assert out["lru_evictions"] == 2
        assert out["lru_size"] == 4
        assert out["lru_capacity"] == 16
        assert out["lru_hit_rate"] == 0.75
        assert out["backing"]["layers"] == 2


class TestAliasPreservesSettings:
    """The alias path clones the user's settings instead of
    building a bare ModelSettings that drops every streaming tunable."""

    def test_tunables_survive_and_pin_wins(self, tmp_path, monkeypatch):
        import omlx.patches.expert_streaming as es
        import omlx.patches.expert_streaming.residency as res
        from omlx.model_settings import ModelSettings
        from omlx.patches.moe_expert_offload import _apply_via_streaming

        est = SimpleNamespace(
            supported=True, expert_bytes=4 * 1024**3, num_moe_layers=2
        )
        monkeypatch.setattr(
            res, "expert_streaming_estimate", lambda *a, **k: est
        )
        captured = {}

        def _conv(model, path, settings, **kw):
            captured["settings"] = settings
            return model, SimpleNamespace()

        monkeypatch.setattr(es, "convert_model_to_streaming", _conv)

        ms = ModelSettings(
            moe_expert_offload_enabled=True,
            expert_streaming_io_depth=24,
        )
        wrapped = _apply_via_streaming(SimpleNamespace(), tmp_path, 0.5, ms)
        assert wrapped == 2
        s = captured["settings"]
        assert s.expert_streaming_enabled is True
        assert s.expert_streaming_io_depth == 24  # tunable survived
        assert s.expert_streaming_dynamic is True  # alias default forced
        assert s.expert_streaming_budget_gib == pytest.approx(2.0)

        ms2 = ModelSettings(
            moe_expert_offload_enabled=True,
            expert_streaming_budget_gib=5.0,
            expert_streaming_dynamic=False,
        )
        _apply_via_streaming(SimpleNamespace(), tmp_path, 0.5, ms2)
        s2 = captured["settings"]
        assert s2.expert_streaming_budget_gib == 5.0  # pin wins over fraction
        assert s2.expert_streaming_dynamic is False  # explicit False kept


class TestSchedulerStreamingFloor:
    """The guard counts the LRU heap accounting + wired pin pages."""

    def _sched(self, info, cache, backing):
        from omlx.scheduler import Scheduler

        sched = Scheduler.__new__(Scheduler)
        sched._streaming_guard_info = info
        sched._streaming_lru_cache = cache
        sched._streaming_lru_bytes_last = None
        sched._streaming_backing = backing
        sched._last_mlx_active_memory_bytes = 0
        return sched

    def test_floor_covers_resident_plus_pins(self):
        sched = self._sched(
            {"x": 1},
            SimpleNamespace(resident_bytes=lambda: 3 * 1024**3),
            SimpleNamespace(pinned_bytes=512 * 1024**2),
        )
        used = sched._current_usage_bytes(refresh_mlx_active=False)
        assert used == 3 * 1024**3 + 512 * 1024**2

    def test_backing_without_guard_info_takes_streaming_branch(self):
        # V4.1/legacy report streaming_guard_info=None — resolved to {} —
        # but still stream through mmap; the branch must fire on the
        # backing, not the metadata dict (phys footprint would report the
        # page cache and return a real nonzero number here).
        sched = self._sched({}, None, SimpleNamespace(pinned_bytes=0))
        assert sched._current_usage_bytes(refresh_mlx_active=False) == 0
