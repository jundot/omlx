# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the adaptive MTP draft-depth controller.

The controller scores each depth as expected committed tokens over measured
cycle cost and keeps every per-depth cost estimate FRESH via bidirectional,
staleness-directed, duty-bounded probes — no hand-tuned per-chip / per-model
decision constant. These tests exercise the host-side logic in isolation
(no MLX / GPU): warmup measurement, self-calibrated marginal cost, the
wall-time cost EMA and spike guard, bidirectional rival probing (the fix for
the stale-cost depth lock), staleness-directed exploration, and the probe
duty bound on heavy models.
"""

import math
import random

from omlx.patches.mlx_lm_mtp.batch_generator import _DepthController


def _simulate(controller, cycles, p_by_depth, ms_by_depth, seed=0):
    """Drive the controller like the real loop: draft ``cur``, observe outcome."""
    rng = random.Random(seed)
    for _ in range(cycles):
        depth = controller.cur
        accepted = 0
        for j in range(depth):
            if rng.random() < p_by_depth[j]:
                accepted += 1
            else:
                break
        controller.observe(depth, accepted, ms_by_depth[depth])
    return controller


def test_observe_signature_is_three_positional():
    # Guards the call site batch_generator.py: controller.observe(k, m, ms).
    c = _DepthController(2)
    c.observe(2, 1, 12.5)
    assert c.cycles == 1


def test_warmup_measures_every_depth_once():
    c = _DepthController(3)
    assert c.cur == 3  # sweep walks 3 -> 2 -> 1
    c.observe(3, 3, 30.0)
    assert c.cur == 2
    c.observe(2, 2, 20.0)
    assert c.cur == 1
    c.observe(1, 1, 10.0)
    assert c.t == {1: 10.0, 2: 20.0, 3: 30.0}
    assert c._warmup == []


def test_marginal_est_uses_measured_slope_not_prior():
    c = _DepthController(3, marginal_ms=7.0)
    assert c._marginal_est() == 7.0  # fallback prior before two depths measured
    c.t = {1: 10.0, 2: 40.0, 3: 70.0}
    assert math.isclose(c._marginal_est(), 30.0, rel_tol=1e-9)
    c.t = {1: 10.0, 3: 70.0}
    assert math.isclose(c._t_est(2), 10.0 + 30.0 * 1, rel_tol=1e-9)


def test_time_alpha_horizon_is_wall_clock():
    c = _DepthController(2)
    assert math.isclose(c._time_alpha(c.TAU_MS), 1.0 - math.exp(-1.0), rel_tol=1e-9)
    assert c._time_alpha(80.0) > c._time_alpha(8.0)
    assert c._time_alpha(0.0) == 0.0


def test_spike_guard_damps_one_off_outlier():
    c = _DepthController(2)
    c.t[2] = 20.0
    c._update_time(2, 200.0)  # a 10x spike must not drag the estimate near 200
    assert c.t[2] < 60.0


def test_expensive_extra_verify_settles_at_depth_1():
    # MoE on a bandwidth-limited chip (M4 Max analog): depth-2 nearly doubles
    # the cycle cost at low d2 acceptance, so even starting deep it drops to 1.
    c = _DepthController(2)
    c._warmup = []
    c.p = [0.8, 0.25]
    c.t = {1: 10.0, 2: 19.0}
    c.cur = 2
    assert c._score(1) > c._score(2)
    assert c._best() == 1


def test_cheap_extra_verify_keeps_depth_2():
    # High-bandwidth chip (M3 Ultra analog) with a genuine depth-2 win: cheap
    # extra verify and high d2 acceptance -> the measured score keeps depth 2
    # (no shallow bias suppressing a real deep win — the GLM case).
    c = _DepthController(2)
    c._warmup = []
    c.p = [0.85, 0.7]
    c.t = {1: 10.0, 2: 10.5}
    c.cur = 1
    assert c._best() == 2


def test_exact_tie_does_not_move_deeper():
    # On an exact score tie, hysteresis + the strict '>' shallow-to-deep scan
    # keep the current (shallow) depth: no churn, no drift deeper.
    c = _DepthController(2)
    c._warmup = []
    c.p = [0.5, 0.0]
    c.t = {1: 10.0, 2: 10.0}
    c.cur = 1
    assert math.isclose(c._score(1), c._score(2), rel_tol=1e-12)
    assert c._best() == 1


def test_best_rival_is_bidirectional():
    # Sitting DEEP with a shallower rival within PROBE_MARGIN: the rival probe
    # must target the shallower depth — this is what breaks the depth-2 lock
    # (stale-high t[1] can only be corrected by re-running depth 1).
    c = _DepthController(2)
    c._warmup = []
    c.cur = 2
    c.p = [0.8, 0.5]
    c.t = {1: 11.0, 2: 12.0}  # t[1] stale-high; scores land within the margin
    assert c._score(2) >= c._score(1)  # cur currently looks better...
    assert c._best_rival() == 1  # ...but depth 1 is worth re-measuring

    # And a clearly-worse rival is not probed (no probe tax).
    c2 = _DepthController(2)
    c2._warmup = []
    c2.cur = 1
    c2.p = [0.8, 0.1]
    c2.t = {1: 10.0, 2: 19.0}
    assert c2._best_rival() is None


def test_most_stale_prefers_unmeasured_then_oldest():
    c = _DepthController(3)
    c._warmup = []
    c.cur = 1
    c.t_age = {1: 0.0, 2: 500.0}  # depth 3 never measured -> infinitely stale
    assert c._most_stale() == 3
    c.t_age = {1: 0.0, 2: 900.0, 3: 200.0}
    assert c._most_stale() == 2


def test_stale_lock_is_broken_by_repeated_probes():
    # Reproduce the measured failure: warmup right after prefill measures t[1]
    # inflated (11ms vs true 10ms), the controller settles at depth 2, and
    # without bidirectional probes t[1] would never refresh (the depth-2 lock).
    # With rival probes re-running depth 1 every ~1s, the slow EMA converges
    # over a few bursts and the lock breaks.
    c = _DepthController(2)
    c._warmup = []
    c.cur = 2
    c.p = [0.8, 0.3]
    c.t = {1: 11.0, 2: 12.0}  # stale-high t[1]: hides depth 1's true advantage
    c.t_age = {1: 0.0, 2: 0.0}
    assert c._best() == 2  # locked on the stale estimate
    # Drive real cycles: depth 2 truly costs 12ms, depth 1 truly costs 10ms.
    _simulate(c, 1500, p_by_depth=[0.8, 0.3], ms_by_depth={1: 10.0, 2: 12.0})
    assert c.t[1] < 10.5  # repeated probes converged t[1] toward the truth
    assert c._best() == 1  # lock broken


def test_probe_duty_bound_scales_period_on_heavy_models():
    # On a 100ms-cycle model, a 1s cadence would spend ~40% of cycles probing
    # (4-cycle burst every 10 cycles). The duty bound stretches the period so
    # probes stay under ~PROBE_DUTY of cycles.
    c = _DepthController(2)
    c._warmup = []
    c.cur = 1
    c.p = [0.8, 0.25]  # rival within PROBE_MARGIN but below HYSTERESIS
    c.t = {1: 100.0, 2: 110.0}
    c.t_age = {1: 0.0, 2: 0.0}
    c._ms_probe = c.PROBE_PERIOD_MS + 1.0  # past the light-model cadence...
    c.observe(1, 1, 100.0)
    assert c.probe_left == 0  # ...but under the duty-bounded period: no probe
    assert c.cur == 1
    # Past the duty-bounded period the rival probe fires.
    c._ms_probe = c.PROBE_LEN * 100.0 / c.PROBE_DUTY + 1.0
    c.observe(1, 1, 100.0)
    assert c.probe_left == c.PROBE_LEN
    assert c.cur == 2


def test_uncertain_rival_gets_probed_after_wall_clock_period():
    c = _DepthController(2)
    c._warmup = []
    c.probe_left = 0
    c.p = [0.85, 0.5]
    c.t = {1: 10.0, 2: 13.0}
    c.t_age = {1: 0.0, 2: 0.0}
    c.cur = 1
    c._ms_probe = c.PROBE_PERIOD_MS - 100.0
    c.observe(1, 1, 10.0)  # under the period -> no probe yet
    assert c.probe_left == 0
    assert c.cur == 1
    c._ms_probe = c.PROBE_PERIOD_MS - 5.0
    c.observe(1, 1, 10.0)  # crosses the period while rival is close -> probe
    assert c.probe_left == c.PROBE_LEN
    assert c.cur == 2


def test_exploration_probe_targets_most_stale_depth():
    # When the exploration clock lapses, the probe goes to the most-stale
    # depth even if it is not a close rival (bounded staleness for all depths).
    c = _DepthController(3)
    c._warmup = []
    c.probe_left = 0
    c.cur = 1
    c.p = [0.9, 0.1, 0.1]  # depths 2/3 score far below depth 1
    c.t = {1: 10.0, 2: 30.0, 3: 50.0}
    c.t_age = {1: 0.0, 2: 100.0, 3: 9000.0}
    assert c._best_rival() is None  # no close rival
    c._ms_probe = c.PROBE_PERIOD_MS + 1.0
    c._ms_explore = c.PROBE_PERIOD_MAX_MS + 1.0
    c.observe(1, 1, 10.0)
    assert c.probe_left == c.PROBE_LEN
    assert c.cur == 3  # the never/least-recently measured depth


def test_probe_burst_completes_and_resets_cadence():
    c = _DepthController(2)
    c._warmup = []
    c.p = [0.85, 0.55]
    c.t = {1: 10.0, 2: 11.5}
    c.cur = 2
    c.probe_left = c.PROBE_LEN
    for _ in range(c.PROBE_LEN):
        c.observe(2, 1, 11.5)
    assert c.probe_left == 0
    assert c._ms_probe == 0.0


def test_expensive_extra_verify_settles_at_depth_1_end_to_end():
    c = _DepthController(2)
    _simulate(c, 200, p_by_depth=[0.8, 0.25], ms_by_depth={1: 10.0, 2: 19.0})
    assert c._best() == 1


def test_max_depth_one_is_inert():
    c = _DepthController(1)
    _simulate(c, 40, p_by_depth=[0.9], ms_by_depth={1: 10.0})
    assert c.cur == 1
    assert c._best() == 1
