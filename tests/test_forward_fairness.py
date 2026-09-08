# SPDX-License-Identifier: Apache-2.0
"""
Tests for ForwardFairnessGate (non-scheduler decode fairness).

Embedding forwards bypass the Scheduler; the gate makes them honor the
same shared hold protocol scheduler prefills use: wait out the
process-global hold deadline before a forward, accrue chunk_time * share
after it, cap contended chunk sizes in time, and skip the process-global
cache flush while another engine decodes -- resuming the flush once
memory crosses the enforcer's propagated soft watermark (or while no
watermark is known at all).
"""

import asyncio
import time
import unittest
from unittest.mock import patch

import pytest

from omlx.decode_activity import get_decode_activity
from omlx.engine.forward_fairness import (
    _FALLBACK_CONTENDED_ITEMS,
    ForwardFairnessGate,
)
from omlx.scheduler import (
    _DECODE_FAIR_SHARE,
    _DECODE_STALL_TARGET_MS,
    SchedulerConfig,
)


@pytest.fixture(autouse=True)
def _quiet_decode_activity():
    get_decode_activity().clear()
    yield
    get_decode_activity().clear()


def _publish_other_decode():
    get_decode_activity().publish("chat-engine:deadbeef", 1)


class TestContention:
    def test_uncontended_when_registry_empty(self):
        gate = ForwardFairnessGate("embed:test")
        assert not gate.contended()

    def test_contended_when_other_engine_decodes(self):
        _publish_other_decode()
        gate = ForwardFairnessGate("embed:test")
        assert gate.contended()

    def test_own_key_does_not_count(self):
        gate = ForwardFairnessGate("embed:test")
        get_decode_activity().publish("embed:test", 1)
        assert not gate.contended()

    def test_disabled_via_live_config_toggle(self):
        config = SchedulerConfig()
        gate = ForwardFairnessGate("embed:test", config)
        _publish_other_decode()
        assert gate.contended()
        config.decode_fairness = False  # admin live-toggle mutates the object
        assert not gate.contended()
        config.decode_fairness = True
        assert gate.contended()


class TestWaitTurn:
    def test_no_wait_uncontended(self):
        gate = ForwardFairnessGate("embed:test")
        start = time.perf_counter()
        assert gate.wait_turn() is False
        assert time.perf_counter() - start < 0.05

    def test_waits_out_shared_hold(self):
        _publish_other_decode()
        gate = ForwardFairnessGate("embed:test")
        deadline = time.perf_counter() + 0.15
        get_decode_activity().extend_hold(deadline)
        assert gate.wait_turn() is True
        assert time.perf_counter() >= deadline - 0.01

    def test_contended_without_hold_returns_quickly(self):
        _publish_other_decode()
        gate = ForwardFairnessGate("embed:test")
        start = time.perf_counter()
        assert gate.wait_turn() is True
        assert time.perf_counter() - start < 0.05

    def test_exits_early_when_decode_finishes(self):
        _publish_other_decode()
        gate = ForwardFairnessGate("embed:test")
        get_decode_activity().extend_hold(time.perf_counter() + 10.0)
        import threading

        def _finish():
            time.sleep(0.15)
            get_decode_activity().publish("chat-engine:deadbeef", 0)

        t = threading.Thread(target=_finish)
        t.start()
        start = time.perf_counter()
        gate.wait_turn()
        elapsed = time.perf_counter() - start
        t.join()
        assert elapsed < 1.0  # nowhere near the 10 s deadline


class TestSettle:
    def test_accrues_shared_hold_when_contended(self):
        _publish_other_decode()
        gate = ForwardFairnessGate("embed:test")
        before = time.perf_counter()
        gate.settle(0.4, 8, contended=True)
        hold = get_decode_activity().hold_until()
        assert hold >= before + 0.4 * _DECODE_FAIR_SHARE - 0.05
        assert hold <= time.perf_counter() + 0.4 * _DECODE_FAIR_SHARE + 0.05

    def test_no_hold_when_uncontended(self):
        gate = ForwardFairnessGate("embed:test")
        gate.settle(0.4, 8, contended=False)
        assert get_decode_activity().hold_until() == 0.0

    def test_hold_when_decode_started_mid_forward(self):
        gate = ForwardFairnessGate("embed:test")
        _publish_other_decode()  # decode began while the forward ran
        gate.settle(0.4, 8, contended=False)
        assert get_decode_activity().hold_until() > time.perf_counter()


class TestChunkCap:
    def test_uncapped_without_contention(self):
        gate = ForwardFairnessGate("embed:test")
        assert gate.chunk_cap() is None

    def test_fallback_before_first_measurement(self):
        _publish_other_decode()
        gate = ForwardFairnessGate("embed:test")
        assert gate.chunk_cap() == _FALLBACK_CONTENDED_ITEMS

    def test_cap_derives_from_measured_per_item_time(self):
        _publish_other_decode()
        gate = ForwardFairnessGate("embed:test")
        gate.settle(0.4, 8, contended=True)  # 50 ms per item
        expected = int((_DECODE_STALL_TARGET_MS / 1000.0) / 0.05)
        assert gate.chunk_cap() == max(1, expected)

    def test_cap_floors_at_one_item(self):
        _publish_other_decode()
        gate = ForwardFairnessGate("embed:test")
        gate.settle(10.0, 1, contended=True)  # pathologically slow items
        assert gate.chunk_cap() == 1


class TestClearCacheDecision:
    def test_clears_when_uncontended(self):
        gate = ForwardFairnessGate("embed:test")
        assert gate.should_clear_cache() is True

    def test_skips_clear_while_contended_below_watermark(self):
        _publish_other_decode()
        gate = ForwardFairnessGate("embed:test")
        gate.set_memory_soft_limit(10 * 1024**3)
        with patch("omlx.engine.forward_fairness.mx") as fake_mx:
            fake_mx.get_active_memory.return_value = 1 * 1024**3
            fake_mx.get_cache_memory.return_value = 1 * 1024**3
            assert gate.should_clear_cache() is False

    def test_clears_under_contention_without_watermark(self):
        # No propagated soft limit means no safe skip: mirror the
        # scheduler's _memory_limit_bytes <= 0 -> clear behavior.
        _publish_other_decode()
        gate = ForwardFairnessGate("embed:test")
        assert gate.should_clear_cache() is True

    def test_clears_under_contention_at_watermark(self):
        # active + cache counts: cached-but-free buffers are exactly what
        # the skip lets accumulate.
        _publish_other_decode()
        gate = ForwardFairnessGate("embed:test")
        gate.set_memory_soft_limit(2 * 1024**3)
        with patch("omlx.engine.forward_fairness.mx") as fake_mx:
            fake_mx.get_active_memory.return_value = 1 * 1024**3
            fake_mx.get_cache_memory.return_value = 1 * 1024**3
            assert gate.should_clear_cache() is True

    def test_watermark_setter_sanitizes_bad_values(self):
        gate = ForwardFairnessGate("embed:test")
        gate.set_memory_soft_limit(-5)
        assert gate._memory_soft_limit_bytes == 0
        gate.set_memory_soft_limit("garbage")
        assert gate._memory_soft_limit_bytes == 0
        gate.set_memory_soft_limit(123)
        assert gate._memory_soft_limit_bytes == 123

    def test_clears_again_when_fairness_disabled(self):
        config = SchedulerConfig(decode_fairness=False)
        gate = ForwardFairnessGate("embed:test", config)
        _publish_other_decode()
        assert gate.should_clear_cache() is True


class EmbeddingEngineCacheClearTests(unittest.TestCase):
    """Engine-level coverage: the per-forward flush under memory pressure.

    Uses a fake model so no weights load; the real global MLX executor
    runs the forward, the mx module is replaced per import site (the
    engine's synchronize/clear_cache and the gate's memory getters).
    """

    def _engine(self):
        from omlx.engine.embedding import EmbeddingEngine
        from omlx.models.embedding import EmbeddingOutput

        engine = EmbeddingEngine("fake-embedding-model", batch_size=4)

        class FakeModel:
            def embed(self, inputs, max_length=None, padding=True, truncation=True):
                return EmbeddingOutput(
                    embeddings=[[0.0] for _ in inputs],
                    total_tokens=len(inputs),
                    dimensions=1,
                )

        engine._model = FakeModel()
        return engine

    def _run_embed(self, engine):
        output = asyncio.run(engine.embed(["alpha", "beta"]))
        assert len(output.embeddings) == 2

    def test_uncontended_forward_clears_cache(self):
        get_decode_activity().clear()
        engine = self._engine()
        with patch("omlx.engine.embedding.mx") as engine_mx:
            self._run_embed(engine)
            engine_mx.synchronize.assert_called()
            engine_mx.clear_cache.assert_called()

    def test_contended_forward_skips_clear_below_watermark(self):
        _publish_other_decode()
        engine = self._engine()
        engine.set_memory_soft_limit(10 * 1024**3)
        with patch("omlx.engine.embedding.mx") as engine_mx, patch(
            "omlx.engine.forward_fairness.mx"
        ) as fair_mx:
            fair_mx.get_active_memory.return_value = 1024
            fair_mx.get_cache_memory.return_value = 1024
            self._run_embed(engine)
            engine_mx.synchronize.assert_called()
            engine_mx.clear_cache.assert_not_called()

    def test_contended_forward_clears_under_memory_pressure(self):
        _publish_other_decode()
        engine = self._engine()
        engine.set_memory_soft_limit(2 * 1024**3)
        with patch("omlx.engine.embedding.mx") as engine_mx, patch(
            "omlx.engine.forward_fairness.mx"
        ) as fair_mx:
            fair_mx.get_active_memory.return_value = 1 * 1024**3
            fair_mx.get_cache_memory.return_value = 2 * 1024**3
            self._run_embed(engine)
            engine_mx.clear_cache.assert_called()

    def test_contended_forward_clears_without_watermark(self):
        _publish_other_decode()
        engine = self._engine()
        with patch("omlx.engine.embedding.mx") as engine_mx:
            self._run_embed(engine)
            engine_mx.clear_cache.assert_called()
