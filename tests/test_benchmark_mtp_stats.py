# SPDX-License-Identifier: Apache-2.0
"""Acceptance telemetry from the Lightning MTP decode loop into benchmark results.

Covers the snapshot registry in ``patches.mlx_lm_mtp.batch_generator`` and the
drain/summarize helpers in ``admin.benchmark``. Mocked throughout — no model
weights, no engine; the registry is fed by calling the same ``_log_mtp_stats``
the decode loop calls.
"""

from types import SimpleNamespace

import pytest

from omlx.admin.benchmark import _drain_mtp_snapshots, _summarize_mtp
from omlx.patches.mlx_lm_mtp import batch_generator as bg


def _stats(**kw) -> bg._MtpStats:
    s = bg._MtpStats()
    for k, v in kw.items():
        setattr(s, k, v)
    return s


@pytest.fixture(autouse=True)
def _clean_registry():
    bg.drain_mtp_stats_snapshots()
    yield
    bg.drain_mtp_stats_snapshots()


class TestSnapshotRegistry:
    def test_log_appends_snapshot_with_derived_fields(self):
        stats = _stats(
            cycles=52,
            accepts=119,
            depth_drafted=[49, 45, 38],
            depth_accepted=[46, 39, 34],
            init_emits=2,
            draft_emits=118,
            bonus_emits=38,
            verify_emits=13,
            zero_cycles=3,
        )
        bg._log_mtp_stats("uid-7", stats, "stop")
        (snap,) = bg.drain_mtp_stats_snapshots()
        assert snap["uid"] == "uid-7"
        assert snap["finish"] == "stop"
        assert snap["tokens"] == 2 + 118 + 38 + 13
        assert snap["cycles"] == 52
        assert snap["accepts"] == 119
        assert snap["drafted"] == 49 + 45 + 38
        assert snap["zero_cycles"] == 3

    def test_drain_clears_and_orders_oldest_first(self):
        bg._log_mtp_stats("a", _stats(cycles=1), "stop")
        bg._log_mtp_stats("b", _stats(cycles=2), "length")
        drained = bg.drain_mtp_stats_snapshots()
        assert [d["uid"] for d in drained] == ["a", "b"]
        assert bg.drain_mtp_stats_snapshots() == []

    def test_registry_is_bounded(self):
        for i in range(bg._MTP_STATS_SNAPSHOTS.maxlen + 10):
            bg._log_mtp_stats(str(i), _stats(cycles=i), "stop")
        drained = bg.drain_mtp_stats_snapshots()
        assert len(drained) == bg._MTP_STATS_SNAPSHOTS.maxlen
        # Overflow evicts the oldest entries, never the newest.
        assert drained[-1]["uid"] == str(bg._MTP_STATS_SNAPSHOTS.maxlen + 9)

    def test_same_stats_object_snapshots_once(self):
        # Park-on-terminal logs the same stats object twice (park exit, then
        # finish). The second log call must not produce a second snapshot —
        # summed counters would double.
        stats = _stats(cycles=5, accepts=4, draft_emits=4, init_emits=2)
        bg._log_mtp_stats("uid", stats, "parked-at-depth-0")
        bg._log_mtp_stats("uid", stats, "length")
        drained = bg.drain_mtp_stats_snapshots()
        assert len(drained) == 1
        assert drained[0]["finish"] == "parked-at-depth-0"

    def test_append_failure_never_raises_out_of_log(self, monkeypatch):
        # The finish paths call _log_mtp_stats unguarded; a broken registry
        # must degrade to a dropped snapshot, never a failed sequence.
        class Broken:
            def append(self, item):
                raise RuntimeError("registry broken")

        monkeypatch.setattr(bg, "_MTP_STATS_SNAPSHOTS", Broken())
        bg._log_mtp_stats("uid", _stats(cycles=1), "stop")  # must not raise

    def test_empty_depth_lists_fall_back_to_cycles_denominator(self):
        # Depth-1 legacy path can finish with depth lists empty; the log's
        # accept denominator falls back to cycles, and the snapshot mirrors it.
        bg._log_mtp_stats("legacy", _stats(cycles=10, accepts=6), "stop")
        (snap,) = bg.drain_mtp_stats_snapshots()
        assert snap["drafted"] == 10


class TestSummarize:
    def test_none_for_no_entries(self):
        assert _summarize_mtp([], expected_sequences=1) is None

    def test_single_entry_rates(self):
        out = _summarize_mtp(
            [{"uid": "a", "tokens": 128, "cycles": 64, "accepts": 60, "drafted": 100}],
            expected_sequences=1,
        )
        assert out["sequences"] == 1
        assert out["accept_pct"] == 60.0
        assert out["tok_per_cycle"] == 2.0

    def test_rates_recomputed_from_summed_counters(self):
        # 90% on a long sequence + 0% on a 1-cycle straggler must not
        # average to 45% — the pooled rate is what the operator needs.
        out = _summarize_mtp(
            [
                {"uid": "a", "tokens": 200, "cycles": 100, "accepts": 90, "drafted": 100},
                {"uid": "b", "tokens": 1, "cycles": 1, "accepts": 0, "drafted": 1},
            ],
            expected_sequences=2,
        )
        assert out["sequences"] == 2
        assert out["accept_pct"] == round(90 / 101 * 100, 1)
        assert out["tok_per_cycle"] == round(201 / 101, 2)

    def test_zero_denominators_yield_none_fields(self):
        out = _summarize_mtp(
            [{"uid": "a", "tokens": 1, "cycles": 0, "accepts": 0, "drafted": 0}],
            expected_sequences=1,
        )
        assert out["accept_pct"] is None
        assert out["tok_per_cycle"] is None

    def test_withheld_when_sequence_count_mismatches(self, caplog):
        # Attribution rests on benchmark exclusivity; a foreign sequence in
        # the drain means the numbers can't be trusted for this point, so the
        # summary must be withheld loudly rather than published quietly.
        import logging as _logging

        with caplog.at_level(_logging.WARNING):
            out = _summarize_mtp(
                [
                    {"uid": "mine", "tokens": 10, "cycles": 5, "accepts": 4, "drafted": 5},
                    {"uid": "foreign", "tokens": 9, "cycles": 4, "accepts": 1, "drafted": 4},
                ],
                expected_sequences=1,
            )
        assert out is None
        assert any("MTP telemetry withheld" in r.message for r in caplog.records)

    def test_late_join_segments_counted_once_per_sequence(self):
        # Late-join handoff logs a segment mid-request and the rejoined run
        # logs another at finish: one sequence, two snapshots. That must not
        # trip the attribution guard, and both segments' counters pool.
        out = _summarize_mtp(
            [
                {"uid": "req", "tokens": 40, "cycles": 20, "accepts": 15, "drafted": 20},
                {"uid": "req", "tokens": 60, "cycles": 30, "accepts": 27, "drafted": 30},
            ],
            expected_sequences=1,
        )
        assert out["sequences"] == 1
        assert out["segments"] == 2
        assert out["accepts"] == 42


class TestDrainHelper:
    def test_drains_through_registry(self):
        bg._log_mtp_stats("x", _stats(cycles=3, accepts=2), "stop")
        entries = _drain_mtp_snapshots()
        assert len(entries) == 1
        assert entries[0]["uid"] == "x"

    def test_fail_soft_when_registry_unavailable(self, monkeypatch):
        import omlx.admin.benchmark as bench

        def boom():
            raise RuntimeError("patch module gone")

        monkeypatch.setattr(bg, "drain_mtp_stats_snapshots", boom)
        assert bench._drain_mtp_snapshots() == []


class TestSingleTestAttachment:
    @pytest.mark.asyncio
    async def test_single_test_attaches_mtp_summary(self):
        from omlx.admin.benchmark import _run_single_test

        class FakeEngine:
            async def stream_generate(self, **kwargs):
                # Simulate the decode loop finishing a sequence mid-stream.
                bg._log_mtp_stats(
                    "req",
                    _stats(
                        cycles=4,
                        accepts=3,
                        depth_drafted=[4],
                        depth_accepted=[3],
                        draft_emits=3,
                        init_emits=2,
                        bonus_emits=1,
                        verify_emits=1,
                    ),
                    "length",
                )
                yield SimpleNamespace(
                    completion_tokens=7,
                    prompt_tokens=4,
                    cached_tokens=0,
                    new_text="hello",
                )

        # Stale pre-test entries must not leak into this point's summary.
        bg._log_mtp_stats("stale", _stats(cycles=99, accepts=99), "stop")
        metrics = await _run_single_test(FakeEngine(), [1, 2, 3, 4], 7, 4)
        assert metrics["mtp"]["sequences"] == 1
        assert metrics["mtp"]["cycles"] == 4
        assert metrics["mtp"]["accepts"] == 3

    @pytest.mark.asyncio
    async def test_single_test_omits_mtp_when_no_sequences_finished(self):
        from omlx.admin.benchmark import _run_single_test

        class FakeEngine:
            async def stream_generate(self, **kwargs):
                yield SimpleNamespace(
                    completion_tokens=1,
                    prompt_tokens=4,
                    cached_tokens=0,
                    new_text="x",
                )

        metrics = await _run_single_test(FakeEngine(), [1, 2, 3, 4], 1, 4)
        assert "mtp" not in metrics


class TestBatchTestAttachment:
    @pytest.mark.asyncio
    async def test_batch_test_attaches_pooled_summary(self):
        from omlx.admin.benchmark import _run_batch_test

        class Core:
            def __init__(self):
                self.prompts = {}

            async def add_request(self, prompt, sampling_params, skip_cache_store=False):
                request_id = f"request-{len(self.prompts)}"
                self.prompts[request_id] = prompt
                return request_id

            async def stream_outputs(self, request_id):
                # Each sequence finishes and logs its stats, like the real
                # decode loop does at finish_reason time.
                bg._log_mtp_stats(
                    request_id,
                    _stats(cycles=10, accepts=8, depth_drafted=[10], depth_accepted=[8],
                           draft_emits=8, init_emits=2),
                    "length",
                )
                yield SimpleNamespace(
                    completion_tokens=2,
                    prompt_tokens=len(self.prompts[request_id]),
                    finished=True,
                )

        # Stale pre-test entries must be discarded by the pre-drain.
        bg._log_mtp_stats("stale", _stats(cycles=99, accepts=99), "stop")

        engine = SimpleNamespace(_engine=Core())
        metrics = await _run_batch_test(
            engine=engine,
            prompts=[[1] * 8, [2] * 8],
            prompt_tokens=8,
            max_tokens=2,
            batch_size=2,
        )
        assert metrics["mtp"]["sequences"] == 2
        assert metrics["mtp"]["accepts"] == 16

    @pytest.mark.asyncio
    async def test_batch_test_withholds_on_foreign_traffic(self):
        from omlx.admin.benchmark import _run_batch_test

        class Core:
            def __init__(self):
                self.count = 0

            async def add_request(self, prompt, sampling_params, skip_cache_store=False):
                self.count += 1
                return f"request-{self.count}"

            async def stream_outputs(self, request_id):
                bg._log_mtp_stats(
                    request_id, _stats(cycles=5, accepts=4), "length"
                )
                if request_id == "request-1":
                    # A foreign request finishing mid-benchmark.
                    bg._log_mtp_stats("someone-else", _stats(cycles=3, accepts=1), "stop")
                yield SimpleNamespace(completion_tokens=1, prompt_tokens=8, finished=True)

        engine = SimpleNamespace(_engine=Core())
        metrics = await _run_batch_test(
            engine=engine,
            prompts=[[1] * 8, [2] * 8],
            prompt_tokens=8,
            max_tokens=1,
            batch_size=2,
        )
        assert "mtp" not in metrics
