# SPDX-License-Identifier: Apache-2.0
"""Tests for PrefillTransientTracker — per-scheduler EWMA used by the
adaptive prefill throttle (#1040 follow-up)."""

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
            1058.0,
            1171.0,
            1085.0,
            1103.0,
            1123.1,
            1141.3,
            991.2,
            1131.2,
            1099.3,
            1839.3,
            1867.3,
            1186.5,
            1031.4,
            1529.5,
            1117.5,
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


class TestFlatOverhead:
    MB = 1024**2

    def test_fixed_overhead_is_not_scaled_by_token_count(self):
        t = PrefillTransientTracker("qwen4")
        t.observe_flat_overhead(
            54,
            571 * self.MB,
            static_bytes=self.MB,
        )
        assert t.flat_overhead_bytes_for(False) == 570 * self.MB
        assert t.flat_overhead_charge_for(False) == 0

    def test_only_reclaimed_flat_overhead_is_charged_again(self):
        t = PrefillTransientTracker("qwen4")
        t.observe_flat_overhead(
            2048,
            28_536 * self.MB,
            static_bytes=1402 * self.MB,
        )

        t.observe_flat_overhead(
            2048,
            10_000 * self.MB,
            static_bytes=1402 * self.MB,
            gathered_core=True,
        )
        # The retained 27 GB is already included in current footprint.
        assert t.flat_overhead_charge_for(False) == 0
        t.record_flat_reclaim(12500 * self.MB)
        assert t.flat_overhead_charge_for(False) == 12_500 * self.MB
        assert t.flat_overhead_charge_for(True) == (10_000 - 1402) * self.MB

        # The next positive delta repays that one-shot reallocation risk.
        t.observe_flat_overhead(
            2048,
            13_902 * self.MB,
            static_bytes=1402 * self.MB,
        )
        assert t.flat_overhead_charge_for(False) == 0
        assert t.flat_overhead_bytes_for(False) == 27_134 * self.MB

    def test_reclaim_debt_tracks_net_release(self):
        t = PrefillTransientTracker("qwen4")
        t.observe_flat_overhead(
            2048,
            -600 * self.MB,
            static_bytes=100 * self.MB,
        )
        t.observe_flat_overhead(
            2048,
            -400 * self.MB,
            static_bytes=100 * self.MB,
        )
        assert t.reclaim_debt_bytes_for(False) == 1000 * self.MB
        t.observe_flat_overhead(
            2048,
            250 * self.MB,
            static_bytes=100 * self.MB,
        )
        assert t.reclaim_debt_bytes_for(False) == 750 * self.MB

    def test_reallocation_does_not_become_new_flat_overhead(self):
        t = PrefillTransientTracker("qwen4")
        t.observe_flat_overhead(
            2048,
            4 * 1024 * self.MB,
            static_bytes=1024 * self.MB,
        )
        t.observe_flat_overhead(
            2048,
            -16 * 1024 * self.MB,
            static_bytes=1024 * self.MB,
        )
        t.observe_flat_overhead(
            2048,
            17 * 1024 * self.MB,
            static_bytes=1024 * self.MB,
        )
        assert t.flat_overhead_bytes_for(False) == 3 * 1024 * self.MB
        assert t.reclaim_debt_bytes_for(False) == 0

    def test_execution_regimes_are_independent(self):
        t = PrefillTransientTracker("qwen4")
        t.observe_flat_overhead(
            2048,
            500 * self.MB,
            static_bytes=100 * self.MB,
        )
        assert t.flat_overhead_bytes_for(False) == 400 * self.MB
        assert t.flat_overhead_bytes_for(True) == 0


class TestPoolReleaseAccounting:
    def test_repeated_releases_accumulate_without_exceeding_observed_overhead(self):
        tracker = PrefillTransientTracker()
        tracker.observe_flat_overhead(512, 5 * 1024**3, static_bytes=1024**3)
        tracker.record_flat_reclaim(1024**3)
        tracker.record_flat_reclaim(1024**3)
        assert tracker.flat_overhead_charge_for(False) == 2 * 1024**3
        tracker.record_flat_reclaim(10 * 1024**3)
        assert tracker.flat_overhead_charge_for(False) == 4 * 1024**3
        assert tracker.flat_overhead_charge_for(True) == 0

    def test_pool_clear_covers_both_routes_without_sharing_their_overhead(self):
        tracker = PrefillTransientTracker()
        tracker.observe_flat_overhead(512, 5 * 1024**3, static_bytes=1024**3)
        tracker.observe_flat_overhead(
            512, 2 * 1024**3, static_bytes=1024**3, gathered_core=True
        )
        tracker.record_flat_reclaim(3 * 1024**3)
        assert tracker.flat_overhead_charge_for(False) == 3 * 1024**3
        assert tracker.flat_overhead_charge_for(True) == 1024**3
        tracker.observe_flat_overhead(
            512, 2 * 1024**3, static_bytes=1024**3, gathered_core=True
        )
        assert tracker.flat_overhead_charge_for(True) == 0
        assert tracker.flat_overhead_charge_for(False) == 3 * 1024**3
