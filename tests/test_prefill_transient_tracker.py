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
