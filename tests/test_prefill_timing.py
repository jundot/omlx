"""Pure forward-time estimates for chunk-shape-aware prefill planning."""

from dataclasses import asdict

import pytest

from omlx.prefill.timing import PrefillTiming


def test_unknown_and_invalid_queries_have_no_estimate():
    timing = PrefillTiming()
    assert timing.estimate(512) is None
    timing.observe(512, 0.5)
    for token_count in (None, True, 0, -1, 1.5, "512"):
        assert timing.estimate(token_count) is None


@pytest.mark.parametrize("token_count", [None, True, 0, -1, 1.5, "512"])
def test_invalid_token_observations_do_not_change_calibration(token_count):
    timing = PrefillTiming()
    timing.observe(512, 0.5)
    before = asdict(timing)
    timing.observe(token_count, 1.0)
    assert asdict(timing) == before


@pytest.mark.parametrize(
    "duration", [None, True, 0, -1, "0.5", float("nan"), float("inf")]
)
def test_invalid_durations_do_not_change_calibration(duration):
    timing = PrefillTiming()
    timing.observe(512, 0.5)
    before = asdict(timing)
    timing.observe(512, duration)
    assert asdict(timing) == before


def test_short_tails_do_not_rescale_bulk_forward_time():
    timing = PrefillTiming()
    timing.observe(4096, 2.0)
    for _ in range(100):
        timing.observe(4, 0.02)
    assert timing.estimate(4) == pytest.approx(0.02)
    assert timing.estimate(4096) == pytest.approx(2.0)
    assert timing.estimate(512) < 0.3
    assert timing.estimate(8192) == pytest.approx(4.0)


def test_interpolates_and_extrapolates_without_decreasing_at_bucket_boundaries():
    timing = PrefillTiming()
    timing.observe(4, 0.02)
    timing.observe(512, 0.4)
    timing.observe(4096, 2.0)
    assert timing.estimate(2) == pytest.approx(0.01)
    assert timing.estimate(258) == pytest.approx(0.21)
    assert timing.estimate(2304) == pytest.approx(1.2)
    assert timing.estimate(8192) == pytest.approx(4.0)
    durations = [timing.estimate(tokens) for tokens in range(1, 8193)]
    assert durations == sorted(durations)


def test_same_bucket_combines_smallest_shape_and_largest_cost_conservatively():
    timing = PrefillTiming()
    timing.observe(1023, 0.8)
    timing.observe(512, 0.4)
    timing.observe(700, 0.6)
    assert timing.estimate(512) == pytest.approx(0.8)
    assert timing.estimate(700) >= 0.6
    assert timing.estimate(1023) >= 0.8


def test_noisy_samples_produce_a_monotone_upper_envelope():
    timing = PrefillTiming()
    timing.observe(8, 0.04)
    timing.observe(64, 0.03)
    timing.observe(512, 0.4)
    assert timing.estimate(64) == pytest.approx(0.04)
    durations = [timing.estimate(tokens) for tokens in range(1, 1025)]
    assert durations == sorted(durations)


def test_calibration_storage_grows_by_shape_bucket_not_forward_count():
    timing = PrefillTiming()
    for tokens in range(1, 8193):
        timing.observe(tokens, 0.01 + tokens / 2000)
    assert len(asdict(timing)["_samples"]) == (8192).bit_length()
    assert all(len(samples) <= 8 for samples in asdict(timing)["_samples"].values())


def test_recent_observations_replace_a_transient_slow_tail():
    timing = PrefillTiming()
    timing.observe(4, 1.1, now=0.0)
    for index in range(8):
        timing.observe(4, 0.02, now=float(index + 1))
    assert timing.estimate(4, now=8.0) == pytest.approx(0.02)


def test_slow_tail_expires_even_without_any_successful_new_batches():
    timing = PrefillTiming()
    timing.observe(4, 1.1, now=0.0)
    assert timing.estimate(2, now=29.9) > 0.5
    assert timing.estimate(2, now=30.0) is None
    timing.observe(512, 0.25, now=30.0)
    assert timing.estimate(512, now=30.0) == pytest.approx(0.25)


def test_expiry_does_not_remove_recent_observations_in_the_same_bucket():
    timing = PrefillTiming()
    timing.observe(4, 1.1, now=0.0)
    timing.observe(4, 0.02, now=25.0)
    assert timing.estimate(4, now=30.0) == pytest.approx(0.02)


def test_supplied_planning_time_keeps_the_curve_stable_across_expiry_boundary():
    timing = PrefillTiming()
    timing.observe(4, 0.02, now=0.0)
    timing.observe(512, 0.4, now=10.0)
    durations = [timing.estimate(tokens, now=29.99) for tokens in range(1, 1025)]
    assert durations == sorted(durations)
    assert timing.estimate(4, now=30.0) < 0.02
