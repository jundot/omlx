# SPDX-License-Identifier: Apache-2.0
"""Dynamic budget (auto default) + phase-aware caps."""
from types import SimpleNamespace

from omlx.patches.expert_streaming import (
    _auto_budget_bytes,
    _budget_is_pinned,
    _dynamic_armed,
    resolve_budget_bytes,
)
from omlx.patches.expert_streaming.governor import ExpertResidencyGovernor


class _Stats:
    def __init__(self):
        self.decode_layers = 0
        self.decode_layers_missed = 0
        self.decode_misses_by_layer = {}


class _FakeCache:
    """Duck-typed cache: capacity/clear/resize/set_layer_caps/stats."""

    def __init__(self, capacity=400, per_layer=100, layers=4):
        self.capacity = capacity
        self._per_layer_cap = per_layer
        self.num_layers = layers
        self.stats = _Stats()
        self.overrides = {}
        self.cleared = 0

    def clear(self):
        self.cleared += 1

    def resize(self, cap, per_layer=None):
        self.capacity = cap
        if per_layer is not None:
            self._per_layer_cap = per_layer
        self.overrides = {}

    def set_layer_caps(self, mapping):
        self.overrides = dict(mapping or {})


def _gov(cache=None, **kw):
    cache = cache or _FakeCache()
    args = dict(
        per_slot=1024 * 1024,
        num_layers=4,
        max_budget_bytes=6 * 1024**3,
        low_free_bytes=1 * 1024**3,
        target_free_bytes=2 * 1024**3,
        high_free_bytes=4 * 1024**3,
        cooldown_s=0,
    )
    args.update(kw)
    return ExpertResidencyGovernor(cache, **args)


def _feed(cache, layers, missed, by_layer=None):
    cache.stats.decode_layers += layers
    cache.stats.decode_layers_missed += missed
    for k, v in (by_layer or {}).items():
        d = cache.stats.decode_misses_by_layer
        d[k] = d.get(k, 0) + v


# -- hunger -------------------------------------------------------------

def test_hunger_grows_additively_with_headroom(monkeypatch):
    c = _FakeCache(capacity=400, per_layer=100)
    g = _gov(c, num_layers=12)
    import omlx.patches.expert_streaming.governor as G
    monkeypatch.setattr(G, "_free_bytes", lambda: 3 * 1024**3)
    _feed(c, 64, 32, {0: 20, 1: 12})  # stall 0.50 > 0.05
    action = g.observe(force=True)
    assert action.startswith("grow")
    assert c.capacity == 500  # +25% additive (400 * 1.25)
    assert set(c.overrides) == {0, 1}  # targeted top-2, not uniform
    assert all(v <= c.capacity for v in c.overrides.values())


def test_no_grow_small_window(monkeypatch):
    import omlx.patches.expert_streaming.governor as G
    monkeypatch.setattr(G, "_free_bytes", lambda: 3 * 1024**3)
    c = _FakeCache()
    g = _gov(c)
    _feed(c, 8, 8, {0: 8})  # stall 1.0 but window < 16
    assert g.observe(force=True) == ""
    assert c.capacity == 400


def test_no_grow_without_headroom(monkeypatch):
    import omlx.patches.expert_streaming.governor as G
    monkeypatch.setattr(G, "_free_bytes", lambda: int(2.5 * 1024**3))
    c = _FakeCache()
    g = _gov(c, target_free_bytes=3 * 1024**3)  # free < TGT -> shrink zone
    _feed(c, 64, 64, {0: 40})
    action = g.observe(force=True)
    assert action.startswith("shrink")  # pressure wins over hunger


def test_no_grow_below_target(monkeypatch):
    import omlx.patches.expert_streaming.governor as G
    monkeypatch.setattr(G, "_free_bytes", lambda: 3 * 1024**3)
    c = _FakeCache()
    g = _gov(c, stall_target=0.05)
    _feed(c, 100, 2)  # stall 0.02 < target
    assert g.observe(force=True) == ""


def test_pressure_shrink_respects_min_budget(monkeypatch):
    import omlx.patches.expert_streaming.governor as G
    monkeypatch.setattr(G, "_free_bytes", lambda: int(1.5 * 1024**3))
    c = _FakeCache(capacity=400, per_layer=100)
    g = _gov(c, min_budget_bytes=300 * 1024 * 1024)  # floor 300 slots
    action = g.observe(force=True)
    assert action.startswith("shrink")
    assert c.capacity == 300  # max(floor, half)=300, not 200


def test_desperate_clears(monkeypatch):
    import omlx.patches.expert_streaming.governor as G
    monkeypatch.setattr(G, "_free_bytes", lambda: int(0.1 * 1024**3))
    c = _FakeCache()
    g = _gov(c)
    assert g.observe(force=True).startswith("clear")
    assert c.cleared == 1


def test_abundance_doubles(monkeypatch):
    import omlx.patches.expert_streaming.governor as G
    monkeypatch.setattr(G, "_free_bytes", lambda: 5 * 1024**3)
    c = _FakeCache(capacity=400, per_layer=100)
    g = _gov(c)
    _feed(c, 100, 1)  # no hunger; abundance path
    action = g.observe(force=True)
    assert action.startswith("grow") and "stall" not in action
    assert c.capacity == 800


def test_cooldown_blocks_flap(monkeypatch):
    import omlx.patches.expert_streaming.governor as G
    monkeypatch.setattr(G, "_free_bytes", lambda: 5 * 1024**3)
    c = _FakeCache()
    g = _gov(c, cooldown_s=3600)
    _feed(c, 100, 1)
    g.observe(force=True)
    assert g.observe() == ""  # cooling down


def test_reset_rebaselines(monkeypatch):
    import omlx.patches.expert_streaming.governor as G
    monkeypatch.setattr(G, "_free_bytes", lambda: 3 * 1024**3)
    c = _FakeCache()
    g = _gov(c)
    _feed(c, 100, 50, {0: 30})
    g.observe(force=True)  # baseline; hunger grows (cap 400->500)
    assert c.capacity == 500
    # stats wiped (e.g. cache.clear) then tiny hunger: must not act on
    # negative deltas, just re-baseline.
    c.stats.decode_layers = 4
    c.stats.decode_layers_missed = 4
    c.stats.decode_misses_by_layer = {}
    assert g.observe(force=True) == ""
    assert c.capacity == 500


def test_duck_cache_without_targeting(monkeypatch):
    import omlx.patches.expert_streaming.governor as G
    monkeypatch.setattr(G, "_free_bytes", lambda: 3 * 1024**3)

    class _Bare:
        """Minimal contract: capacity/stats/clear/resize — no targeting."""

        def __init__(self):
            self.capacity = 400
            self.stats = _Stats()

        def clear(self):
            pass

        def resize(self, cap, per_layer=None):
            self.capacity = cap

    b = _Bare()
    g = _gov(b)
    _feed(b, 64, 40)
    # No set_layer_caps/_per_layer_cap: retarget is a no-op, never raises.
    action = g.observe(force=True)
    assert action.startswith("grow")
    assert b.capacity == 500


def _real_cache():
    from omlx.patches.expert_streaming.streaming_switch import make_expert_cache
    return make_expert_cache(64 * 1024 * 1024, 1024 * 1024, num_layers=4, policy="lru")


def test_phase_caps_bound_prefill_churn():
    c = _real_cache()  # 64 slots global, 16/layer decode
    assert c._cap_for(0) == 16
    for i in range(16):
        c.put((0, i, "w"), object())
    assert c._layer_counts.get(0) == 16
    c.note_phase(False)  # prefill: max(32, 64//4)=32 global, 4/layer
    assert c._global_cap_active() == 32
    assert c._cap_for(0) == 4
    for i in range(16, 24):
        c.put((1, i, "w"), object())
    assert c._layer_counts.get(1) <= 4
    assert len(c._store) <= 32
    c.note_phase(True)
    assert c._global_cap_active() == 64
    assert c._cap_for(0) == 16


def test_layer_overrides_target_decode():
    c = _real_cache()
    c.set_layer_caps({2: 32})
    assert c._cap_for(2) == 32
    assert c._cap_for(0) == 16
    c.note_phase(False)
    assert c._cap_for(2) == 4  # overrides never apply to prefill
    c.note_phase(True)
    c.resize(64, 16)
    assert c.layer_cap_overrides() == {}  # resize resets to uniform


def test_miss_attribution_decode_only():
    c = _real_cache()
    c.note_phase(True)
    assert c.get((3, 7, "w")) is None
    assert c.stats.decode_misses_by_layer.get(3) == 1
    c.note_phase(False)
    assert c.get((3, 8, "w")) is None
    assert c.stats.decode_misses_by_layer.get(3) == 1  # prefill miss ignored


def test_summary_reports_hunger():
    c = _FakeCache()
    g = _gov(c)
    s = g.summary()
    assert s["stall_target"] == 0.05
    assert "window_stall" in s and "layer_overrides" in s


# -- public governor accessors (out-of-tree consumers must not reach
# into _min_cap_slots / _last_free_gib) --------------------------------

def test_public_accessors():
    c = _FakeCache(capacity=400)
    g = _gov(c, min_cap=64, min_budget_bytes=None)
    assert g.min_capacity() == 64
    assert g.at_floor() is False          # 400 > 64
    c.capacity = 64
    assert g.at_floor() is True           # floor reached -> no slack
    c.capacity = 32
    assert g.at_floor() is True           # below floor also reports
    # desperate band: unknown free (0) -> False; below low_free -> True
    assert g.in_desperate_band() is False
    g._last_free_gib = g.low_free_bytes / 1024**3 * 0.5
    assert g.in_desperate_band() is True
    g._last_free_gib = g.low_free_bytes / 1024**3 * 2.0
    assert g.in_desperate_band() is False
    # min_budget_bytes raises the floor above the bare min_cap
    g2 = _gov(_FakeCache(), min_cap=64, min_budget_bytes=300 * 1024 * 1024)
    assert g2.min_capacity() == 300       # 300 MiB // 1 MiB slots


def test_accessors_safe_without_cache():
    g = ExpertResidencyGovernor(
        None,
        per_slot=1024 * 1024,
        num_layers=4,
        max_budget_bytes=6 * 1024**3,
        low_free_bytes=1 * 1024**3,
        target_free_bytes=2 * 1024**3,
        high_free_bytes=4 * 1024**3,
    )
    assert g.at_floor() is False
    assert g.in_desperate_band() is False
    assert g.min_capacity() >= 1


def test_tick_observe_serialized(monkeypatch):
    """tick() holds the governor lock across its throttle bookkeeping and
    observe() — a mid-request tick cannot interleave a boundary observe."""
    import omlx.patches.expert_streaming.governor as G
    monkeypatch.setattr(G, "_free_bytes", lambda: 3 * 1024**3)
    c = _FakeCache()
    g = _gov(c, cooldown_s=0)
    _feed(c, 100, 50, {0: 30})
    # tick() runs through observe() with the short action cooldown; the
    # governor lock serializes the whole step.
    assert g._lock.acquire(blocking=False)
    g._lock.release()
    action = g.tick()
    assert isinstance(action, str)
    # _last_tick_at advanced under the lock.
    assert g._last_tick_at > 0


def test_reconcile_per_slot_under_lock():
    c = _FakeCache()
    g = _gov(c)
    g.reconcile_per_slot(2 * 1024 * 1024)
    assert g.per_slot == 2 * 1024 * 1024
    # Invalid values are ignored, not latched.
    g.reconcile_per_slot("bogus")
    g.reconcile_per_slot(0)
    g.reconcile_per_slot(-1)
    assert g.per_slot == 2 * 1024 * 1024


# -- budget auto --------------------------------------------------------

def test_pinned_detection():
    assert _budget_is_pinned(SimpleNamespace(expert_streaming_budget_gib=1.0))
    assert _budget_is_pinned(SimpleNamespace(expert_streaming_budget_gib=0))
    assert not _budget_is_pinned(SimpleNamespace(expert_streaming_budget_gib=None))
    assert not _budget_is_pinned(SimpleNamespace())
    assert not _budget_is_pinned(None)


def test_auto_budget_bounds():
    b = _auto_budget_bytes()
    assert int(0.5 * 1024**3) <= b <= int(4.0 * 1024**3)


def test_explicit_budget_wins():
    s = SimpleNamespace(expert_streaming_budget_gib=1.5, expert_streaming_budget_auto=True)
    assert resolve_budget_bytes(s) == int(1.5 * 1024**3)


def test_auto_default_returns_scaled():
    s = SimpleNamespace(expert_streaming_budget_gib=None, expert_streaming_budget_auto=None)
    assert resolve_budget_bytes(s) == _auto_budget_bytes() > 0


def test_auto_false_is_page_cache_only():
    s = SimpleNamespace(expert_streaming_budget_gib=None, expert_streaming_budget_auto=False)
    assert resolve_budget_bytes(s) == 0


def test_dynamic_armed_matrix(monkeypatch):
    pinned = SimpleNamespace(expert_streaming_budget_gib=2.0)
    auto = SimpleNamespace(expert_streaming_budget_gib=None)
    # explicit setting always wins
    assert _dynamic_armed(True, pinned) is True
    assert _dynamic_armed(False, auto) is False
    # env forces on
    import omlx.patches.expert_streaming as P
    monkeypatch.setattr(P, "dynamic_residency_enabled", lambda: True)
    assert _dynamic_armed(None, pinned) is True
    monkeypatch.setattr(P, "dynamic_residency_enabled", lambda: False)
    # auto rule: on for auto budgets, off for pinned
    assert _dynamic_armed(None, auto) is True
    assert _dynamic_armed(None, pinned) is False
    assert _dynamic_armed(None, None) is True


# -- watermark env fractions + generic staging headroom ----------------

def test_watermark_envs_scale_with_ram(monkeypatch):
    import omlx.patches.expert_streaming.governor as G
    ram = 96 * 1024**3
    monkeypatch.setattr(G, "_total_ram_bytes", lambda: ram)
    monkeypatch.setenv("OMLX_GOV_LOW_FRAC", "0.05")
    monkeypatch.setenv("OMLX_GOV_TARGET_FRAC", "0.15")
    monkeypatch.setenv("OMLX_GOV_HIGH_FRAC", "0.30")
    g = _gov(cache=None, low_free_bytes=None,
             target_free_bytes=None, high_free_bytes=None)
    assert g.low_free_bytes == int(ram * 0.05)
    assert g.target_free_bytes == int(ram * 0.15)
    assert g.high_free_bytes == int(ram * 0.30)


def test_watermark_explicit_kwargs_beat_env(monkeypatch):
    import omlx.patches.expert_streaming.governor as G
    monkeypatch.setattr(G, "_total_ram_bytes", lambda: 96 * 1024**3)
    monkeypatch.setenv("OMLX_GOV_LOW_FRAC", "0.05")
    g = _gov(low_free_bytes=1 * 1024**3)
    assert g.low_free_bytes == 1 * 1024**3  # explicit wins


def test_watermark_invalid_env_keeps_default(monkeypatch):
    import omlx.patches.expert_streaming.governor as G
    ram = 64 * 1024**3
    monkeypatch.setattr(G, "_total_ram_bytes", lambda: ram)
    monkeypatch.setenv("OMLX_GOV_LOW_FRAC", "banana")
    g = _gov(cache=None, low_free_bytes=None)
    assert g.low_free_bytes == int(ram * 0.10)


def test_stage_headroom_generic(monkeypatch):
    """Generic path staging mirrors the V4.1 headroom gate."""
    from omlx.patches.expert_streaming.streaming_switch import (
        StreamingQuantizedSwitchLinear,
    )

    c = _FakeCache(capacity=64)
    g = _gov(c, min_cap=64)  # floor == capacity -> at floor
    c.governor = g
    lin = SimpleNamespace(cache=c)
    fn = StreamingQuantizedSwitchLinear._stage_headroom

    # no governor -> legacy behavior (always stage)
    assert fn(SimpleNamespace(cache=_FakeCache())) is True
    # capacity at the governor floor -> suppress
    assert fn(lin) is False
    # headroom above floor -> stage
    c.capacity = 200
    assert fn(lin) is True
    # last observed free inside the desperate band -> suppress
    g._last_free_gib = g.low_free_bytes / 1024**3 * 0.5
    assert fn(lin) is False
    # unknown free (0) -> stage
    g._last_free_gib = 0.0
    assert fn(lin) is True
