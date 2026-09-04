# SPDX-License-Identifier: Apache-2.0
"""Tests for PrefillTransientTracker — per-scheduler EWMA used by the
adaptive prefill throttle (#1040 follow-up)."""

import time

from omlx.prefill_transient_tracker import PrefillTransientTracker


class TestUpdate:
    def test_first_sample_seeds_ewma(self):
        t = PrefillTransientTracker("m")
        t.update(n_tokens=1000, transient_bytes=200_000)
        assert t.samples == 1
        assert t.bytes_per_token == 200.0  # 200_000 / 1000
        assert t.last_n_tokens == 1000
        assert t.last_delta_bytes == 200_000

    def test_subsequent_samples_apply_ewma_alpha(self):
        t = PrefillTransientTracker("m")
        t.update(1000, 100_000)  # 100/token
        t.update(1000, 200_000)  # 200/token; ewma = 0.3*200 + 0.7*100 = 130
        assert t.samples == 2
        assert abs(t.bytes_per_token - 130.0) < 0.01

    def test_negative_delta_skipped(self):
        t = PrefillTransientTracker("m")
        t.update(1000, 100_000)
        baseline = t.bytes_per_token
        t.update(1000, -50_000)  # cache reclaim larger than alloc
        assert t.samples == 1, "negative delta must not be recorded"
        assert t.bytes_per_token == baseline

    def test_zero_delta_skipped(self):
        t = PrefillTransientTracker("m")
        t.update(1000, 0)
        assert t.samples == 0

    def test_zero_tokens_skipped(self):
        t = PrefillTransientTracker("m")
        t.update(0, 100_000)
        assert t.samples == 0


class TestEwmaOutlierGuard:
    """Regression coverage for the 2026-07-29 incident: a single noisy
    tail-chunk reading poisoned the EWMA (773.3 -> 3690.5 KB/token off one
    n=185 sample), which then bounded every later admission check for the
    rest of the process lifetime, independent of cache-credit accuracy."""

    def test_first_sample_never_rejected_even_if_extreme(self):
        # No prior EWMA to compare a ratio against — must seed unconditionally.
        t = PrefillTransientTracker("m")
        t.update(n_tokens=185, transient_bytes=int(10497.1 * 1024 * 185))
        assert t.samples == 1
        assert t.bytes_per_token == 10497.1 * 1024

    def test_incident_outlier_rejected_from_ewma(self):
        t = PrefillTransientTracker("m")
        # Baseline regime: per-token readings observed in
        # ~/.omlx/logs/server.log 16:08:48-16:09:39 (KB/token), replayed as
        # (n_tokens=2048, transient_bytes) pairs.
        baseline_kb_per_token = [
            1058.0, 1171.0, 1085.0, 1103.0, 1123.1, 1141.3, 991.2, 1131.2,
            1099.3, 1839.3, 1867.3, 1186.5, 1031.4, 1529.5, 1117.5,
        ]
        for kb in baseline_kb_per_token:
            t.update(n_tokens=2048, transient_bytes=int(kb * 1024 * 2048))
        ewma_before_outlier = t.bytes_per_token
        assert 900 * 1024 < ewma_before_outlier < 2000 * 1024

        # The actual outlier: n=185, per_token=10497.1KB (~13.6x the EWMA
        # just before it, matching the live 773.3 -> 3690.5 KB/token jump).
        t.update(n_tokens=185, transient_bytes=int(10497.1 * 1024 * 185))

        # EWMA must stay close to its pre-outlier value, not jump toward
        # the outlier's per-token rate.
        assert t.bytes_per_token < ewma_before_outlier * 2
        assert t.bytes_per_token < 3000 * 1024, (
            "EWMA must not reach the ~3690.5 KB/token value observed in "
            "production before this fix"
        )
        # The raw sample is still visible for diagnostics.
        assert t.last_n_tokens == 185
        assert t.last_delta_bytes == int(10497.1 * 1024 * 185)

    def test_legitimate_fluctuation_within_ratio_still_updates_ewma(self):
        t = PrefillTransientTracker("m")
        t.update(n_tokens=2048, transient_bytes=int(1097.3 * 1024 * 2048))
        ewma_before = t.bytes_per_token
        # Matches the live 1097.3 -> 1839.3 KB/token jump (~1.68x): well
        # under the outlier ratio, must be treated as a normal sample.
        t.update(n_tokens=2048, transient_bytes=int(1839.3 * 1024 * 2048))
        expected = 0.3 * (1839.3 * 1024) + 0.7 * ewma_before
        assert abs(t.bytes_per_token - expected) < 1.0

    def test_outlier_still_counts_as_a_sample(self):
        t = PrefillTransientTracker("m")
        t.update(n_tokens=2048, transient_bytes=int(1000 * 1024 * 2048))
        t.update(n_tokens=185, transient_bytes=int(20000 * 1024 * 185))
        assert t.samples == 2, "rejected-from-EWMA samples still count"


class TestPredict:
    def test_predict_zero_when_no_samples(self):
        t = PrefillTransientTracker("m")
        assert t.predict(2048) == 0

    def test_predict_uses_ewma_with_safety_factor(self):
        t = PrefillTransientTracker("m")
        t.update(1000, 100_000)  # 100 bytes/token
        # default safety_factor = 1.2
        assert t.predict(2000) == int(100 * 2000 * 1.2)
        assert t.predict(2000, safety_factor=1.0) == 100 * 2000

    def test_predict_zero_n(self):
        t = PrefillTransientTracker("m")
        t.update(1000, 100_000)
        assert t.predict(0) == 0


class TestObservedMax:
    def test_first_sample_excluded_from_max(self):
        t = PrefillTransientTracker("m")
        # Load-residue noise seeds EWMA only, even at floor size.
        t.update(32, 500_000_000, floor_sample=True)
        assert t.samples == 1
        assert t.observed_max_bytes == 0

    def test_max_tracks_largest_accepted_floor_sample(self):
        t = PrefillTransientTracker("m")
        t.update(32, 900_000_000, floor_sample=True)  # first sample, excluded
        t.update(32, 100_000_000, floor_sample=True)
        t.update(32, 700_000_000, floor_sample=True)
        t.update(32, 300_000_000, floor_sample=True)
        # #3109: floor samples only ever occur under the pressure regime
        # the max exists to protect, so a lower reading must NOT pull it
        # back down (that coupling was the bug) -- it stays pinned at the
        # highest floor sample seen this idle window. See
        # TestObservedMaxIdleReset for how it eventually gets forgotten.
        assert t.observed_max_bytes == 700_000_000

    def test_non_floor_samples_never_enter_max(self):
        # Qwen3.6 regression: a 3GB transient from an unthrottled
        # 2048-token chunk must not become the floor-chunk admission
        # charge, or every prompt at a tight ceiling gets rejected.
        t = PrefillTransientTracker("m")
        t.update(32, 100_000_000, floor_sample=True)
        t.update(2048, 3 * 1024**3)  # big chunk, EWMA only
        assert t.observed_max_bytes == 0 or t.observed_max_bytes < 1024**3
        t.update(32, 200_000_000, floor_sample=True)
        assert t.observed_max_bytes == 200_000_000

    def test_outlier_above_clamp_rejected_not_clamped(self):
        t = PrefillTransientTracker("m")
        t.update(32, 100_000_000, floor_sample=True)  # first sample, excluded
        t.update(32, 200_000_000, floor_sample=True)
        ewma_before = t.bytes_per_token
        t.update(32, 5 * 1024**3, floor_sample=True)  # above 4GiB clamp
        assert t.observed_max_bytes == 200_000_000, "outlier must not enter"
        assert t.samples == 3, "outlier still counts as a sample"
        # This 5GiB/32-token reading is also a >8x EWMA outlier (see
        # TestEwmaOutlierGuard), so it must not move the EWMA either —
        # it is excluded from both the observed-max and the EWMA now.
        assert t.bytes_per_token == ewma_before

    def test_skipped_samples_do_not_touch_max(self):
        t = PrefillTransientTracker("m")
        t.update(32, 100_000_000, floor_sample=True)
        t.update(32, 200_000_000, floor_sample=True)
        t.update(0, 900_000_000, floor_sample=True)  # zero tokens: skipped
        t.update(32, -1, floor_sample=True)  # negative delta: skipped
        assert t.observed_max_bytes == 200_000_000

    def test_max_does_not_affect_ewma_or_predict(self):
        t = PrefillTransientTracker("m")
        t.update(1000, 100_000)  # 100/token
        t.update(32, 6_400, floor_sample=True)  # 200/token; max = 6_400
        assert t.observed_max_bytes == 6_400
        ewma = 0.3 * 200.0 + 0.7 * 100.0
        assert abs(t.bytes_per_token - ewma) < 0.01
        assert t.predict(2000, safety_factor=1.0) == int(ewma * 2000)


class TestObservedMaxIdleReset:
    """#3109: an earlier fix decayed observed_max_bytes on every subsequent
    lower floor-size sample. Review caught the flaw: floor samples only
    occur once the adaptive throttle is already at its minimum chunk size
    -- the pressure regime -- so that decay eroded the floor fastest
    exactly when it was load-bearing. Forgetting now hangs off wall-clock
    idle time instead (unrelated to pressure/sample count), per the
    reviewer's suggested alternative: hang it off ``reset()``, which was
    already documented for "after a long idle" but nothing called."""

    def test_lower_sample_does_not_decay_the_max_within_one_session(self):
        """The specific coupling the review flagged: repeated lower floor
        samples with no elapsed time must NOT erode a stale spike -- that
        was the bug in the pressure-coupled decay this replaces."""
        t = PrefillTransientTracker("m")
        t.update(32, 100_000_000, floor_sample=True)  # first, excluded
        t.update(32, 1_000_000_000, floor_sample=True)  # spike -> max=1e9
        for _ in range(400):
            t.update(32, 100_000_000, floor_sample=True)  # spike never recurs
        assert t.observed_max_bytes == 1_000_000_000

    def test_new_higher_sample_still_raises_instantly(self):
        t = PrefillTransientTracker("m")
        t.update(32, 100_000_000, floor_sample=True)  # first, excluded
        t.update(32, 200_000_000, floor_sample=True)
        t.update(32, 900_000_000, floor_sample=True)  # fresh spike
        assert t.observed_max_bytes == 900_000_000

    def test_idle_gap_under_threshold_does_not_reset(self, monkeypatch):
        times = iter([1000.0, 1000.0 + 60.0, 1000.0 + 60.0 + 1700.0])
        monkeypatch.setattr(time, "monotonic", lambda: next(times))
        t = PrefillTransientTracker("m")
        t.update(32, 100_000_000, floor_sample=True)  # t=1000, first, excluded
        t.update(32, 1_000_000_000, floor_sample=True)  # t=1060, spike
        t.update(32, 50_000_000, floor_sample=True)  # t=2760, +1700s < 30min
        assert t.observed_max_bytes == 1_000_000_000
        assert t.samples == 3

    def test_idle_gap_over_threshold_resets_the_whole_tracker(self, monkeypatch):
        """The actual fix: a session that goes quiet for longer than
        ``_IDLE_RESET_SECONDS`` forgets a stale spike (and everything
        else) on its next sample, instead of staying pinned at the
        spike's value indefinitely -- with no dependence on how many
        floor samples occur afterward."""
        times = iter([1000.0, 1010.0, 1010.0 + 1900.0])  # +1900s > 30min
        monkeypatch.setattr(time, "monotonic", lambda: next(times))
        t = PrefillTransientTracker("m")
        t.update(32, 100_000_000, floor_sample=True)  # t=1000, first, excluded
        t.update(32, 1_000_000_000, floor_sample=True)  # t=1010, spike -> max=1e9
        assert t.observed_max_bytes == 1_000_000_000
        assert t.samples == 2

        t.update(32, 50_000_000, floor_sample=True)  # t=2910, past the idle threshold
        # Reset wipes everything, including samples: this new reading is
        # itself the tracker's first sample post-reset, so it seeds the
        # EWMA only and is excluded from the max (mirrors a fresh model
        # load) rather than becoming the new max outright.
        assert t.observed_max_bytes == 0
        assert t.samples == 1

    def test_idle_reset_does_not_fire_before_the_first_sample(self, monkeypatch):
        """No prior timestamp means no elapsed-time comparison to make --
        the very first call must seed normally, not reset."""
        monkeypatch.setattr(time, "monotonic", lambda: 5000.0)
        t = PrefillTransientTracker("m")
        t.update(32, 500_000_000, floor_sample=True)
        assert t.samples == 1
        assert t.observed_max_bytes == 0  # first sample still excluded

    def test_decay_does_not_apply_to_clamp_rejected_outliers(self):
        """An above-clamp reading is excluded entirely (existing
        behavior) -- it must not affect the max at all, since it was
        never accepted as a real observation."""
        t = PrefillTransientTracker("m")
        t.update(32, 100_000_000, floor_sample=True)  # first, excluded
        t.update(32, 500_000_000, floor_sample=True)  # max = 500M
        t.update(32, 5 * 1024**3, floor_sample=True)  # above 4GiB clamp
        assert t.observed_max_bytes == 500_000_000


class TestReset:
    def test_reset_clears_all(self):
        t = PrefillTransientTracker("m")
        t.update(1000, 100_000)
        t.update(1000, 20_000, gathered_core=True)
        t.update(2000, 300_000)
        t.reset()
        assert t.samples == 0
        assert t.bytes_per_token == 0.0
        assert t.last_n_tokens == 0
        assert t.last_delta_bytes == 0
        assert t.predict(2048) == 0
        assert t.observed_max_bytes == 0
        assert t.samples_for(True) == 0
        assert t.bytes_per_token_for(True) == 0.0
        assert t.last_n_tokens_for(True) == 0
        assert t.last_delta_bytes_for(True) == 0
        assert t.predict(2048, gathered_core=True) == 0
        assert t.observed_max_bytes_for(True) == 0


class TestExecutionRegimes:
    def test_dense_and_gathered_rates_are_independent(self):
        t = PrefillTransientTracker("m")
        t.update(4096, 64 * 1024**3, gathered_core=False)

        assert t.samples == 1
        assert t.samples_for(True) == 0
        assert t.bytes_per_token_for(True) == 0.0
        assert t.last_delta_bytes_for(True) == 0

        t.update(4096, 4 * 1024**3, gathered_core=True)

        assert t.samples == 1
        assert t.samples_for(True) == 1
        assert t.bytes_per_token == 16 * 1024**2
        assert t.bytes_per_token_for(True) == 1024**2
        assert t.last_delta_bytes == 64 * 1024**3
        assert t.last_delta_bytes_for(True) == 4 * 1024**3
        assert t.predict(4096, safety_factor=1.0) == 64 * 1024**3
        assert t.predict(4096, safety_factor=1.0, gathered_core=True) == 4 * 1024**3

    def test_dense_floor_max_does_not_bind_gathered_admission(self):
        t = PrefillTransientTracker("m")
        t.update(32, 100 * 1024**2, floor_sample=True)
        t.update(32, 900 * 1024**2, floor_sample=True)
        t.update(32, 20 * 1024**2, floor_sample=True, gathered_core=True)
        t.update(32, 40 * 1024**2, floor_sample=True, gathered_core=True)

        assert t.observed_max_bytes == 900 * 1024**2
        assert t.observed_max_bytes_for(True) == 40 * 1024**2
