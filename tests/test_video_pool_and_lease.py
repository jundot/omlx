# SPDX-License-Identifier: Apache-2.0
"""Tests for video-model pool rejection and the enforcer video memory lease.

Part A: EnginePool.get_engine must reject model_type == "video" entries with
ModelTypeNotLoadableError BEFORE the memory-admission loop, so a misrouted
chat request can never evict resident LLM engines
(docs/video-generation-engine-spec.md section 3).

Part B: ProcessMemoryEnforcer video lease (spec section 4.4): the lease is
subtracted from the final ceiling at a single choke point, the dynamic
ceiling adds back min(worker_footprint, lease) so the worker is counted
exactly once, and acquire/release move the Metal wired-limit request.
"""

import asyncio
import json
import time
from unittest.mock import MagicMock

import pytest

import omlx.process_memory_enforcer as pme
from omlx.engine_pool import EngineEntry, EnginePool
from omlx.exceptions import EnginePoolError, ModelTypeNotLoadableError

GB = 1024**3

# Deterministic static ceiling patched onto enforcer instances so the
# wired-limit math does not depend on the host machine's RAM.
STATIC_CEILING = 100 * GB
CUSTOM_GB = 20.0


# =========================================================================
# Part A -- pool rejection of video entries
# =========================================================================


class FakeLLMEngine:
    """Loaded-engine stand-in that records eviction attempts."""

    def __init__(self):
        self.stop_called = False

    def has_active_requests(self) -> bool:
        return False

    async def stop(self) -> None:
        self.stop_called = True


def _make_pool_with_video_and_llm():
    pool = EnginePool(scheduler_config=None)
    fake_engine = FakeLLMEngine()
    pool._entries["llm-id"] = EngineEntry(
        model_id="llm-id",
        model_path="/nonexistent/llm-id",
        model_type="llm",
        engine_type="batched",
        estimated_size=4 * GB,
        engine=fake_engine,
        last_access=time.time(),
    )
    pool._entries["video-id"] = EngineEntry(
        model_id="video-id",
        model_path="/nonexistent/video-id",
        model_type="video",
        engine_type="video",
        estimated_size=42 * GB,
        video_pipeline="i2v",
    )
    return pool, fake_engine


class TestVideoPoolRejection:
    def test_model_type_not_loadable_is_engine_pool_error(self):
        assert issubclass(ModelTypeNotLoadableError, EnginePoolError)
        exc = ModelTypeNotLoadableError("video-id", "video")
        assert exc.model_id == "video-id"
        assert exc.model_type == "video"
        assert "/v1/videos" in str(exc)

    def test_model_type_map_has_video_engine(self):
        assert EnginePool._MODEL_TYPE_TO_ENGINE["video"] == "video"

    async def test_get_engine_rejects_video_before_admission(self, monkeypatch):
        """Video rejection fires before admission: no LLM eviction happens.

        Memory is mocked so that, had the 42GB video entry reached the
        admission loop, projected (20 + 42 GB) > ceiling (50 GB) would
        have evicted the idle llm entry. The rejection must fire first.
        """
        pool, fake_engine = _make_pool_with_video_and_llm()
        pool._get_final_ceiling = lambda: 50 * GB
        # Make current usage high enough that admission WOULD evict.
        monkeypatch.setattr(
            "omlx.engine_pool.get_phys_footprint", lambda pid=None: 20 * GB
        )

        with pytest.raises(ModelTypeNotLoadableError) as excinfo:
            await pool.get_engine("video-id")

        assert excinfo.value.model_id == "video-id"
        assert excinfo.value.model_type == "video"
        assert "/v1/videos" in str(excinfo.value)
        # The resident llm engine must be untouched -- not stopped, not
        # unloaded.
        assert pool._entries["llm-id"].engine is fake_engine
        assert fake_engine.stop_called is False


class TestVideoPoolStatus:
    """get_status() emits video_pipeline so /v1/models/status consumers
    (chat UI capability gating) can tell i2v / ti2v / t2v apart."""

    def test_get_status_emits_video_pipeline(self):
        pool, _ = _make_pool_with_video_and_llm()

        models = {m["id"]: m for m in pool.get_status()["models"]}

        assert models["video-id"]["video_pipeline"] == "i2v"
        # Non-video entries carry the empty default
        assert models["llm-id"]["video_pipeline"] == ""

    def test_discover_propagates_video_pipeline_to_status(self, tmp_path):
        """End to end: DiscoveredModel.video_pipeline -> EngineEntry ->
        get_status() wire field."""
        model_dir = tmp_path / "Wan2.2-I2V-A14B"
        (model_dir / "transformer").mkdir(parents=True)
        (model_dir / "model_index.json").write_text(
            json.dumps({"_class_name": "WanImageToVideoPipeline"})
        )
        (model_dir / "transformer" / "model.safetensors").write_bytes(b"\0" * 64)

        pool = EnginePool(scheduler_config=None)
        pool.discover_models(str(tmp_path))

        models = {m["id"]: m for m in pool.get_status()["models"]}
        assert models["Wan2.2-I2V-A14B"]["model_type"] == "video"
        assert models["Wan2.2-I2V-A14B"]["video_pipeline"] == "i2v"


# =========================================================================
# Part B -- enforcer video memory lease
# =========================================================================


@pytest.fixture
def wired_calls(monkeypatch):
    """Replace _apply_metal_wired_limit with a recorder; no mx side effects."""
    calls: list[int] = []

    def _recorder(desired_bytes):
        calls.append(desired_bytes)
        return desired_bytes, None

    monkeypatch.setattr(pme, "_apply_metal_wired_limit", _recorder)
    return calls


def _make_pool_stub():
    pool = MagicMock()
    pool._entries = {}
    pool._lock = asyncio.Lock()
    return pool


def _make_enforcer(monkeypatch, tier="custom", custom_gb=CUSTOM_GB, **kwargs):
    """Enforcer with deterministic ceilings; never started (no loop).

    custom tier -> dynamic ceiling == custom_gb verbatim. The static
    ceiling is pinned to STATIC_CEILING and the Metal cap mocked away so
    get_final_ceiling() == min(STATIC_CEILING, custom) == custom on any
    machine.
    """
    monkeypatch.setattr(pme, "get_effective_metal_cap_bytes", lambda: 0)
    enforcer = pme.ProcessMemoryEnforcer(
        engine_pool=_make_pool_stub(),
        memory_guard_tier=tier,
        memory_guard_custom_ceiling_gb=custom_gb,
        **kwargs,
    )
    enforcer._get_static_ceiling = lambda: STATIC_CEILING
    return enforcer


class TestVideoLeaseCeiling:
    def test_acquire_reduces_final_ceiling_by_lease(self, monkeypatch, wired_calls):
        enforcer = _make_enforcer(monkeypatch)
        base = enforcer.get_final_ceiling()
        assert base == int(CUSTOM_GB * GB)

        enforcer.acquire_video_lease(8 * GB)
        assert enforcer.get_final_ceiling() == base - 8 * GB

    def test_huge_lease_clamps_ceiling_to_one(self, monkeypatch, wired_calls):
        enforcer = _make_enforcer(monkeypatch)
        enforcer.acquire_video_lease(1024 * GB)
        # Never 0: consumers treat ceiling 0 as "guard disabled".
        assert enforcer.get_final_ceiling() == 1

    def test_lease_equal_to_ceiling_clamps_to_one(self, monkeypatch, wired_calls):
        enforcer = _make_enforcer(monkeypatch)
        enforcer.acquire_video_lease(int(CUSTOM_GB * GB))
        assert enforcer.get_final_ceiling() == 1

    def test_double_acquire_raises_runtime_error(self, monkeypatch, wired_calls):
        enforcer = _make_enforcer(monkeypatch)
        enforcer.acquire_video_lease(8 * GB)
        with pytest.raises(RuntimeError):
            enforcer.acquire_video_lease(1 * GB)

    def test_non_positive_lease_raises_value_error(self, monkeypatch, wired_calls):
        enforcer = _make_enforcer(monkeypatch)
        with pytest.raises(ValueError):
            enforcer.acquire_video_lease(0)
        with pytest.raises(ValueError):
            enforcer.acquire_video_lease(-5)
        # Failed acquires must not leave a partial lease behind.
        assert enforcer.video_lease_bytes == 0

    def test_release_restores_ceiling(self, monkeypatch, wired_calls):
        enforcer = _make_enforcer(monkeypatch)
        base = enforcer.get_final_ceiling()
        enforcer.acquire_video_lease(8 * GB)
        assert enforcer.get_final_ceiling() == base - 8 * GB
        enforcer.release_video_lease()
        assert enforcer.get_final_ceiling() == base

    def test_release_when_not_held_is_noop(self, monkeypatch, wired_calls):
        enforcer = _make_enforcer(monkeypatch)
        before = list(wired_calls)
        enforcer.release_video_lease()  # must not raise
        assert enforcer.video_lease_bytes == 0
        # Early return: no Metal wired-limit churn either.
        assert wired_calls == before

    def test_video_lease_bytes_property_tracks(self, monkeypatch, wired_calls):
        enforcer = _make_enforcer(monkeypatch)
        assert enforcer.video_lease_bytes == 0
        enforcer.acquire_video_lease(8 * GB)
        assert enforcer.video_lease_bytes == 8 * GB
        enforcer.release_video_lease()
        assert enforcer.video_lease_bytes == 0

    def test_release_clears_worker_pid(self, monkeypatch, wired_calls):
        enforcer = _make_enforcer(monkeypatch)
        enforcer.acquire_video_lease(8 * GB)
        enforcer.set_video_worker_pid(12345)
        assert enforcer._video_worker_pid == 12345
        enforcer.release_video_lease()
        assert enforcer._video_worker_pid is None

    def test_guard_disabled_ceiling_stays_zero(self, monkeypatch, wired_calls):
        """Guard off: ceiling is 0 (= disabled) and acquire skips Metal calls."""
        enforcer = _make_enforcer(monkeypatch, prefill_memory_guard=False)
        assert enforcer.get_final_ceiling() == 0
        enforcer.acquire_video_lease(8 * GB)
        assert enforcer.get_final_ceiling() == 0
        assert wired_calls == []


class TestVideoLeaseWiredLimit:
    def test_acquire_and_release_move_wired_limit_request(
        self, monkeypatch, wired_calls
    ):
        enforcer = _make_enforcer(monkeypatch)
        assert enforcer._metal_wired_limit_request == 0
        assert wired_calls == []

        enforcer.acquire_video_lease(8 * GB)
        assert wired_calls[-1] == STATIC_CEILING - 8 * GB
        assert enforcer._metal_wired_limit_request == STATIC_CEILING - 8 * GB

        enforcer.release_video_lease()
        assert wired_calls[-1] == STATIC_CEILING
        assert enforcer._metal_wired_limit_request == STATIC_CEILING

    def test_oversized_lease_clamps_wired_target_to_one(
        self, monkeypatch, wired_calls
    ):
        enforcer = _make_enforcer(monkeypatch)
        enforcer.acquire_video_lease(STATIC_CEILING + 5 * GB)
        assert wired_calls[-1] == 1
        assert enforcer._metal_wired_limit_request == 1


class TestDynamicCeilingWorkerAddBack:
    """Non-custom tier: dynamic ceiling adds back min(worker_footprint, lease).

    Inputs are fully mocked: get_macos_vm_stats returns fixed numbers and
    get_phys_footprint is a per-pid fake. balanced tier -> active ratio 0.5,
    so the base dynamic ceiling is own + free + inactive + active * 0.5.
    """

    OWN = 5 * GB
    WORKER_PID = 4242
    VM = {"free": 10 * GB, "inactive": 4 * GB, "active": 8 * GB, "wired": 0}
    # own 5 + free 10 + inactive 4 + active 8 * 0.5 = 23 GB
    BASE = 23 * GB

    def _setup(self, monkeypatch, worker_footprint):
        monkeypatch.setattr(pme, "get_macos_vm_stats", lambda: dict(self.VM))

        def fake_phys(pid=None):
            if pid is None:
                return self.OWN
            if pid == self.WORKER_PID:
                return worker_footprint
            return 0

        monkeypatch.setattr(pme, "get_phys_footprint", fake_phys)
        return _make_enforcer(monkeypatch, tier="balanced")

    def test_no_pid_no_add_back(self, monkeypatch, wired_calls):
        enforcer = self._setup(monkeypatch, worker_footprint=3 * GB)
        enforcer.acquire_video_lease(8 * GB)
        # Lease held but no worker pid bound yet (pre-spawn): add-back 0.
        assert enforcer._get_dynamic_ceiling() == self.BASE

    def test_add_back_equals_worker_footprint_under_lease(
        self, monkeypatch, wired_calls
    ):
        enforcer = self._setup(monkeypatch, worker_footprint=3 * GB)
        enforcer.acquire_video_lease(8 * GB)
        enforcer.set_video_worker_pid(self.WORKER_PID)
        assert enforcer._get_dynamic_ceiling() == self.BASE + 3 * GB

    def test_add_back_clamped_to_lease(self, monkeypatch, wired_calls):
        # Runaway worker (50 GB footprint) must not raise the parent
        # ceiling beyond the 8 GB lease.
        enforcer = self._setup(monkeypatch, worker_footprint=50 * GB)
        enforcer.acquire_video_lease(8 * GB)
        enforcer.set_video_worker_pid(self.WORKER_PID)
        assert enforcer._get_dynamic_ceiling() == self.BASE + 8 * GB

    def test_zero_footprint_read_no_add_back(self, monkeypatch, wired_calls):
        # Footprint read failure (0) degrades to double-counting, which
        # is fail-conservative.
        enforcer = self._setup(monkeypatch, worker_footprint=0)
        enforcer.acquire_video_lease(8 * GB)
        enforcer.set_video_worker_pid(self.WORKER_PID)
        assert enforcer._get_dynamic_ceiling() == self.BASE

    def test_no_lease_no_add_back_even_with_pid(self, monkeypatch, wired_calls):
        enforcer = self._setup(monkeypatch, worker_footprint=3 * GB)
        enforcer.set_video_worker_pid(self.WORKER_PID)
        assert enforcer._get_dynamic_ceiling() == self.BASE

    def test_lease_then_release_round_trip_final_ceiling(
        self, monkeypatch, wired_calls
    ):
        """End to end on a non-custom tier: final ceiling tightens by the
        lease minus the worker add-back, then restores after release."""
        enforcer = self._setup(monkeypatch, worker_footprint=3 * GB)
        base = enforcer.get_final_ceiling()
        assert base == self.BASE  # min(static 100 GB, dynamic 23 GB)

        enforcer.acquire_video_lease(8 * GB)
        enforcer.set_video_worker_pid(self.WORKER_PID)
        # dynamic = BASE + 3 GB add-back; final = dynamic - 8 GB lease.
        assert enforcer.get_final_ceiling() == self.BASE + 3 * GB - 8 * GB

        enforcer.release_video_lease()
        assert enforcer.get_final_ceiling() == base


class TestUpscalerPoolRejection:
    """video_upscaler entries are registry-visible but never loadable."""

    def test_type_engine_map_has_upscaler(self):
        assert EnginePool._MODEL_TYPE_TO_ENGINE["video_upscaler"] == "video_upscaler"

    async def test_get_engine_rejects_upscaler(self):
        pool, _ = _make_pool_with_video_and_llm()
        pool._entries["seedvr2-3b-8bit"] = EngineEntry(
            model_id="seedvr2-3b-8bit",
            model_path="/nonexistent/seedvr2",
            model_type="video_upscaler",
            engine_type="video_upscaler",
            estimated_size=5 * GB,
        )
        with pytest.raises(ModelTypeNotLoadableError) as exc_info:
            await pool.get_engine("seedvr2-3b-8bit")
        # The hint must point at the parameter, not POST /v1/videos alone
        assert "upscale_resolution" in str(exc_info.value)
