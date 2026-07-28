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
        t.update(32, 5 * 1024**3, floor_sample=True)  # above 4GiB clamp
        assert t.observed_max_bytes == 200_000_000, "outlier must not enter"
        assert t.samples == 3, "outlier still feeds the EWMA"

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
        t.update(2000, 300_000)
        t.reset()
        assert t.samples == 0
        assert t.bytes_per_token == 0.0
        assert t.last_n_tokens == 0
        assert t.last_delta_bytes == 0
        assert t.predict(2048) == 0
        assert t.observed_max_bytes == 0
