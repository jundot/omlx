# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.utils.system_sampler."""

import sys
import time
from unittest.mock import patch

import pytest

from omlx.utils.system_sampler import (
    SystemSampler,
    _cluster_map,
    _cluster_map_from_ioreg,
    _Sample,
    aggregate,
)


def make_sample(t=0.0, cpu_total=None, cpu_p=None, cpu_e=None, gpu=None,
                thermal=None, footprint=0, mlx_active=0, mlx_cache=0,
                sys_used=0, sys_wired=0):
    return _Sample(
        t=t,
        cpu_total=cpu_total,
        cpu_p=cpu_p,
        cpu_e=cpu_e,
        gpu_util=gpu,
        thermal=thermal,
        phys_footprint=footprint,
        mlx_active=mlx_active,
        mlx_cache=mlx_cache,
        sys_used=sys_used,
        sys_wired=sys_wired,
    )


GIB = 1024**3


class TestAggregate:
    """Pure reduction over synthetic samples — no OS calls, runs anywhere."""

    def test_empty_window_returns_none(self):
        # Uploading a zero-filled object instead of null would let the site
        # average fabricated zeros in as real measurements.
        assert aggregate([], 1.0) is None

    def test_single_sample_is_enough(self):
        # Short tests (small model, 1024-token prompt) finish in under a
        # second. Each sample already carries a CPU delta over the preceding
        # interval, so one is real data — discarding it left those runs with
        # no host metrics at all.
        out = aggregate([make_sample(cpu_total=0.4, footprint=GIB)], 0.25)
        assert out is not None
        assert out["sample_count"] == 1
        assert out["cpu"]["total_avg"] == 40.0
        assert out["cpu"]["total_max"] == 40.0

    def test_cpu_avg_and_max_as_percent(self):
        samples = [
            make_sample(cpu_total=0.2, cpu_p=0.3, cpu_e=0.1),
            make_sample(cpu_total=0.6, cpu_p=0.7, cpu_e=0.5),
        ]
        out = aggregate(samples, 1.0)
        assert out["cpu"]["total_avg"] == 40.0
        assert out["cpu"]["total_max"] == 60.0
        assert out["cpu"]["p_avg"] == 50.0
        assert out["cpu"]["e_avg"] == 30.0

    def test_missing_cpu_readings_do_not_poison_the_average(self):
        samples = [
            make_sample(cpu_total=None),
            make_sample(cpu_total=0.5),
            make_sample(cpu_total=0.5),
        ]
        out = aggregate(samples, 1.0)
        assert out["cpu"]["total_avg"] == 50.0

    def test_all_cpu_readings_missing_yields_none(self):
        out = aggregate([make_sample(), make_sample()], 1.0)
        assert out["cpu"]["total_avg"] is None
        assert out["cpu"]["total_max"] is None

    def test_memory_reports_peaks_in_gib(self):
        samples = [
            make_sample(footprint=10 * GIB, mlx_active=8 * GIB,
                        mlx_cache=1 * GIB, sys_used=20 * GIB, sys_wired=15 * GIB),
            make_sample(footprint=12 * GIB, mlx_active=9 * GIB,
                        mlx_cache=2 * GIB, sys_used=25 * GIB, sys_wired=18 * GIB),
        ]
        mem = aggregate(samples, 1.0)["memory"]
        assert mem["phys_footprint_peak"] == 12.0
        assert mem["mlx_active_peak"] == 9.0
        assert mem["mlx_cache_peak"] == 2.0
        assert mem["system_used_peak"] == 25.0
        assert mem["system_wired_peak"] == 18.0

    def test_thermal_reports_start_and_max_not_last(self):
        # A run that heated up and cooled back down still has to report that
        # it got hot, so max is what matters, not the final reading.
        samples = [
            make_sample(thermal=0),
            make_sample(thermal=3),
            make_sample(thermal=1),
        ]
        out = aggregate(samples, 1.0)
        assert out["thermal"] == {"start": 0, "max": 3}

    def test_thermal_omitted_when_unavailable(self):
        out = aggregate([make_sample(), make_sample()], 1.0)
        assert "thermal" not in out

    def test_gpu_omitted_when_unavailable(self):
        out = aggregate([make_sample(), make_sample()], 1.0)
        assert "gpu" not in out

    def test_gpu_avg_and_max(self):
        samples = [make_sample(gpu=0.5), make_sample(gpu=0.9)]
        out = aggregate(samples, 1.0)
        assert out["gpu"]["util_avg"] == 70.0
        assert out["gpu"]["util_max"] == 90.0

    def test_sample_count_and_interval_are_reported(self):
        out = aggregate([make_sample(), make_sample(), make_sample()], 0.5)
        assert out["sample_count"] == 3
        assert out["interval_s"] == 0.5


# Captured from a real M3 Ultra, abbreviated to 8 CPUs. Two things this
# encodes: the key order flips between entries (ioreg emits cluster-type first
# for some nodes, logical-cpu-id first for others), and the efficiency cores
# are split into two groups rather than occupying a low-index prefix — so the
# contiguous-prefix heuristic would mislabel indices 2-3 and 4-5 here.
_ULTRA_IOREG = """
    | |     "cluster-type" = <"E">
    | |     "logical-cpu-id" = 0
    | |     "cluster-type" = <"E">
    | |     "logical-cpu-id" = 1
    | |     "logical-cpu-id" = 2
    | |     "cluster-type" = <"P">
    | |     "logical-cpu-id" = 3
    | |     "cluster-type" = <"P">
    | |     "cluster-type" = <"E">
    | |     "logical-cpu-id" = 4
    | |     "cluster-type" = <"E">
    | |     "logical-cpu-id" = 5
    | |     "logical-cpu-id" = 6
    | |     "cluster-type" = <"P">
    | |     "logical-cpu-id" = 7
    | |     "cluster-type" = <"P">
"""

_CONTIGUOUS_IOREG = """
    | |     "cluster-type" = <"E">
    | |     "logical-cpu-id" = 0
    | |     "cluster-type" = <"E">
    | |     "logical-cpu-id" = 1
    | |     "logical-cpu-id" = 2
    | |     "cluster-type" = <"P">
"""


class TestClusterMap:
    def setup_method(self):
        _cluster_map.cache_clear()

    def teardown_method(self):
        _cluster_map.cache_clear()

    def _run_with_ioreg(self, stdout):
        class Result:
            def __init__(self, out):
                self.stdout = out

        return patch(
            "omlx.utils.system_sampler.subprocess.run",
            return_value=Result(stdout),
        )

    def test_parses_interleaved_layout(self):
        with self._run_with_ioreg(_ULTRA_IOREG), \
             patch("omlx.utils.system_sampler.IS_DARWIN", True):
            assert _cluster_map_from_ioreg() == (
                True, True, False, False, True, True, False, False
            )

    def test_interleaved_layout_disagrees_with_the_prefix_heuristic(self):
        # The bug this parser exists to avoid: hw.perflevel1.logicalcpu says
        # there are 4 efficiency cores, and assuming they are indices 0-3
        # mislabels half of them on a fused-die part.
        with self._run_with_ioreg(_ULTRA_IOREG), \
             patch("omlx.utils.system_sampler.IS_DARWIN", True):
            actual = _cluster_map_from_ioreg()
        prefix_guess = tuple(i < 4 for i in range(8))
        assert actual != prefix_guess

    def test_parses_contiguous_layout(self):
        with self._run_with_ioreg(_CONTIGUOUS_IOREG), \
             patch("omlx.utils.system_sampler.IS_DARWIN", True):
            assert _cluster_map_from_ioreg() == (True, True, False)

    def test_malformed_output_returns_none(self):
        with self._run_with_ioreg("no cpu entries here"), \
             patch("omlx.utils.system_sampler.IS_DARWIN", True):
            assert _cluster_map_from_ioreg() is None

    def test_gap_in_cpu_ids_returns_none(self):
        broken = """
    | |     "cluster-type" = <"E">
    | |     "logical-cpu-id" = 0
    | |     "cluster-type" = <"P">
    | |     "logical-cpu-id" = 5
"""
        with self._run_with_ioreg(broken), \
             patch("omlx.utils.system_sampler.IS_DARWIN", True):
            assert _cluster_map_from_ioreg() is None

    def test_falls_back_to_prefix_when_ioreg_unavailable(self):
        with patch(
            "omlx.utils.system_sampler._cluster_map_from_ioreg", return_value=None
        ), patch("omlx.utils.system_sampler._sysctl_int", return_value=4):
            assert _cluster_map(8) == (True, True, True, True,
                                       False, False, False, False)

    def test_falls_back_to_all_performance_when_nothing_is_known(self):
        with patch(
            "omlx.utils.system_sampler._cluster_map_from_ioreg", return_value=None
        ), patch("omlx.utils.system_sampler._sysctl_int", return_value=None):
            assert _cluster_map(4) == (False, False, False, False)

    def test_ioreg_map_ignored_when_length_disagrees_with_cpu_count(self):
        with patch(
            "omlx.utils.system_sampler._cluster_map_from_ioreg",
            return_value=(True, True),
        ), patch("omlx.utils.system_sampler._sysctl_int", return_value=2):
            assert _cluster_map(8) == (True, True, False, False,
                                       False, False, False, False)


class TestSamplerLifecycle:
    def test_stop_joins_promptly(self):
        # Guards against using time.sleep instead of Event.wait, which would
        # make stop() block for the remainder of the current tick.
        sampler = SystemSampler(interval_s=5.0)
        sampler.start()
        started = time.monotonic()
        sampler.stop()
        assert time.monotonic() - started < 2.0

    def test_stop_is_idempotent(self):
        sampler = SystemSampler(interval_s=0.1)
        sampler.start()
        sampler.stop()
        sampler.stop()

    def test_window_without_samples_returns_none(self):
        sampler = SystemSampler(interval_s=0.1)
        assert sampler.window(0.0, 1.0) is None

    def test_window_excludes_samples_outside_the_range(self):
        sampler = SystemSampler(interval_s=1.0)
        sampler._samples.extend([
            make_sample(t=1.0, cpu_total=1.0),
            make_sample(t=5.0, cpu_total=0.0),
            make_sample(t=6.0, cpu_total=0.0),
            make_sample(t=9.0, cpu_total=1.0),
        ])
        out = sampler.window(4.0, 7.0)
        assert out["sample_count"] == 2
        assert out["cpu"]["total_max"] == 0.0

    def test_run_peak_footprint_is_the_max_seen(self):
        sampler = SystemSampler(interval_s=1.0)
        sampler._samples.extend([
            make_sample(footprint=5 * GIB),
            make_sample(footprint=9 * GIB),
            make_sample(footprint=7 * GIB),
        ])
        assert sampler.run_peak_footprint() == 9 * GIB

    def test_run_peak_footprint_without_samples_is_zero(self):
        assert SystemSampler(interval_s=1.0).run_peak_footprint() == 0


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin-only API")
class TestDarwinSmoke:
    def test_collect_returns_plausible_values(self):
        sampler = SystemSampler(interval_s=0.1)
        sampler._cpu.sample()  # prime the tick delta
        time.sleep(0.2)
        s = sampler._collect()
        sampler.stop()
        if s.cpu_total is not None:
            assert 0.0 <= s.cpu_total <= 1.0
        if s.gpu_util is not None:
            assert 0.0 <= s.gpu_util <= 1.0
        if s.thermal is not None:
            assert s.thermal in range(5)
        assert s.phys_footprint > 0

    def test_end_to_end_window_has_expected_shape(self):
        sampler = SystemSampler(interval_s=0.2)
        sampler.start()
        t0 = time.monotonic()
        time.sleep(1.0)
        t1 = time.monotonic()
        sampler.stop()
        out = sampler.window(t0, t1)
        assert out is not None
        assert out["sample_count"] >= 2
        assert out["memory"]["phys_footprint_peak"] > 0
