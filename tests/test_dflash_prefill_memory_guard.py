# SPDX-License-Identifier: Apache-2.0
"""Tests that DFlashEngine enforces the prefill memory guard.

DFlash bypasses the scheduler (its primary speculative path runs outside the
Scheduler), so it inherited ``BaseEngine``'s no-op ``preflight_chat`` and ran
long prefills completely unguarded — a latent OOM (observed end-to-end against
Qwen3-Coder-Next + DFlash with 56k-token prompts). The fix gives DFlash its own
``_DFlashPrefillGuard`` (a MemoryMonitor + the enforcer's watermarks) and
``preflight_*`` overrides that reuse the shared ``raise_if_prefill_exceeds``.

These tests pin the guard math (mirroring ``test_scheduler_prefill_memory_guard``)
and the engine-level delegation so a refactor can't silently revert it.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import mlx.core as mx
import pytest

from omlx.engine.dflash import DFlashEngine, _DFlashPrefillGuard
from omlx.exceptions import PrefillMemoryExceededError
from omlx.memory_monitor import (
    MemoryMonitor,
    raise_if_prefill_exceeds,
    set_model_info_from_model,
)


class _ModelConfig:
    """Minimal config exposing the fields the estimator reads."""

    def __init__(
        self,
        num_hidden_layers: int = 32,
        num_key_value_heads: int = 8,
        num_attention_heads: int = 32,
        head_dim: int = 192,  # > 128 → SDPA fallback path
    ) -> None:
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim


def _make_target_model() -> MagicMock:
    model = MagicMock()
    model.config = _ModelConfig()
    # Strip make_cache so the KVCache-counting branch doesn't iterate a Mock.
    del model.make_cache
    model.dtype = mx.float16
    return model


def _make_guard(step: int = 2048) -> _DFlashPrefillGuard:
    monitor = MemoryMonitor(max_kv_cache_memory=None, eviction_enabled=False)
    set_model_info_from_model(monitor, _make_target_model())
    return _DFlashPrefillGuard(monitor, step)


def _zero_mem():
    """Patch live-memory probes so the estimate alone drives the check."""
    return patch("omlx.engine.dflash.get_phys_footprint", return_value=0), patch(
        "omlx.memory_monitor.mx.get_active_memory",
        side_effect=AssertionError("preflight must not read MLX directly"),
    )


# --- guard math (mirrors the scheduler guard tests) -----------------------


def test_guard_populates_estimator_dims():
    guard = _make_guard()
    m = guard.memory_monitor
    assert m._num_attention_heads == 32
    assert m._head_dim == 192
    assert m._num_layers == 32
    assert m._num_kv_heads == 8


def test_estimator_produces_nonzero_peak():
    guard = _make_guard()
    assert guard.memory_monitor.estimate_prefill_peak_bytes(65536, 2048) > 0


def test_set_model_info_from_model_handles_dict_nested_text_config():
    model = MagicMock()
    model.config = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "num_hidden_layers": 40,
            "num_key_value_heads": 2,
            "num_attention_heads": 16,
            "head_dim": 256,
        },
    }
    del model.make_cache
    model.dtype = mx.float16

    monitor = MemoryMonitor(max_kv_cache_memory=None, eviction_enabled=False)
    set_model_info_from_model(monitor, model)

    assert monitor._num_layers == 40
    assert monitor._num_kv_heads == 2
    assert monitor._num_attention_heads == 16
    assert monitor._head_dim == 256
    assert monitor.estimate_prefill_peak_bytes(50_000, 2048) > 0


def test_preflight_passes_within_limit():
    """Positive control: a normal prompt under a generous limit must NOT raise."""
    guard = _make_guard()
    guard._prefill_memory_guard = True
    guard._memory_hard_limit_bytes = 10**18
    p1, p2 = _zero_mem()
    with p1, p2:
        guard.preflight_or_raise(num_prompt_tokens=32768)  # no exception


def test_preflight_raises_when_oversized():
    guard = _make_guard()
    guard._prefill_memory_guard = True
    guard._memory_hard_limit_bytes = 1  # any allocation exceeds
    p1, p2 = _zero_mem()
    with p1, p2, pytest.raises(PrefillMemoryExceededError) as exc:
        guard.preflight_or_raise(num_prompt_tokens=65536, request_id="r1")
    err = exc.value
    assert err.estimated_bytes > 0
    assert err.limit_bytes == 1
    assert err.request_id == "r1"
    assert "Prefill would require" in err.message
    assert "KV+SDPA" in err.message


def test_preflight_noop_when_guard_disabled():
    guard = _make_guard()
    guard._prefill_memory_guard = False
    guard._memory_hard_limit_bytes = 1
    guard.preflight_or_raise(num_prompt_tokens=65536)  # no exception


def test_preflight_noop_when_hard_limit_zero():
    guard = _make_guard()
    guard._prefill_memory_guard = True
    guard._memory_hard_limit_bytes = 0
    guard.preflight_or_raise(num_prompt_tokens=65536)  # no exception


def test_shared_helper_noop_when_fully_cached():
    """The fully-cached no-op belongs to ``raise_if_prefill_exceeds`` (for
    engines whose caches keep KV resident); the DFlash guard itself has no
    ``cached_tokens`` parameter."""
    monitor = MemoryMonitor(max_kv_cache_memory=None, eviction_enabled=False)
    set_model_info_from_model(monitor, _make_target_model())
    # new_tokens == 0 → nothing to prefill → no exception.
    raise_if_prefill_exceeds(
        monitor,
        prefill_memory_guard=True,
        hard_limit_bytes=1,
        current_usage_bytes=0,
        prefill_step_size=2048,
        num_prompt_tokens=1000,
        cached_tokens=1000,
    )


def test_shared_helper_uses_caller_supplied_usage_without_mlx_probe():
    monitor = MemoryMonitor(max_kv_cache_memory=None, eviction_enabled=False)
    set_model_info_from_model(monitor, _make_target_model())

    with patch(
        "omlx.memory_monitor.mx.get_active_memory",
        side_effect=AssertionError("preflight must not read MLX directly"),
    ), pytest.raises(PrefillMemoryExceededError):
        raise_if_prefill_exceeds(
            monitor,
            prefill_memory_guard=True,
            hard_limit_bytes=1,
            current_usage_bytes=0,
            prefill_step_size=2048,
            num_prompt_tokens=65536,
        )


def test_guard_uses_cached_active_and_physical_usage_without_mlx_probe():
    guard = _make_guard()
    guard._prefill_memory_guard = True
    cached = 2 * 1024**3
    phys = 3 * 1024**3
    guard.record_mlx_active_memory(cached)
    peak = guard.memory_monitor.estimate_prefill_peak_bytes(65536, 2048)
    guard._memory_hard_limit_bytes = int(phys + peak - 1)

    with (
        patch("omlx.engine.dflash.get_phys_footprint", return_value=phys),
        patch(
            "omlx.memory_monitor.mx.get_active_memory",
            side_effect=AssertionError("preflight must not read MLX directly"),
        ),
        pytest.raises(PrefillMemoryExceededError) as exc,
    ):
        guard.preflight_or_raise(num_prompt_tokens=65536, request_id="r-phys")

    assert exc.value.estimated_bytes >= int(phys + peak)
    assert exc.value.request_id == "r-phys"


def test_guard_uses_cached_active_when_larger_than_physical():
    guard = _make_guard()
    guard._prefill_memory_guard = True
    cached = 3 * 1024**3
    phys = 2 * 1024**3
    guard.record_mlx_active_memory(cached)
    peak = guard.memory_monitor.estimate_prefill_peak_bytes(65536, 2048)
    guard._memory_hard_limit_bytes = int(cached + peak - 1)

    with (
        patch("omlx.engine.dflash.get_phys_footprint", return_value=phys),
        patch(
            "omlx.memory_monitor.mx.get_active_memory",
            side_effect=AssertionError("preflight must not read MLX directly"),
        ),
        pytest.raises(PrefillMemoryExceededError) as exc,
    ):
        guard.preflight_or_raise(num_prompt_tokens=65536, request_id="r-cached")

    assert exc.value.estimated_bytes >= int(cached + peak)
    assert exc.value.request_id == "r-cached"


def test_guard_excludes_hot_cache_bytes_from_physical_usage():
    """Serialized hot-cache CPU bytes must not be charged twice.

    The enforcer already subtracts a hot-cache reservation from the ceiling it
    propagates (``_memory_hot_cache_reserved_bytes`` side); counting the same
    bytes again inside phys_footprint over-rejects by the hot-cache size —
    the same double-count the scheduler guard fixed for issue 1796. The limit
    here is chosen so the prefill fits exactly iff the exclusion is applied.
    """
    guard = _make_guard()
    guard._prefill_memory_guard = True
    phys = 3 * 1024**3
    hot_used = 1 * 1024**3
    guard._memory_hot_cache_used_bytes = hot_used
    peak = guard.memory_monitor.estimate_prefill_peak_bytes(65536, 2048)
    # Fits with the exclusion (phys - hot_used + peak), not without.
    guard._memory_hard_limit_bytes = int(phys - hot_used + peak)

    with (
        patch("omlx.engine.dflash.get_phys_footprint", return_value=phys),
        patch(
            "omlx.memory_monitor.mx.get_active_memory",
            side_effect=AssertionError("preflight must not read MLX directly"),
        ),
    ):
        guard.preflight_or_raise(num_prompt_tokens=65536, request_id="r-hot")

    # Still rejects when genuinely over even after the exclusion.
    guard._memory_hard_limit_bytes = int(phys - hot_used + peak - 1)
    with (
        patch("omlx.engine.dflash.get_phys_footprint", return_value=phys),
        patch(
            "omlx.memory_monitor.mx.get_active_memory",
            side_effect=AssertionError("preflight must not read MLX directly"),
        ),
        pytest.raises(PrefillMemoryExceededError) as exc,
    ):
        guard.preflight_or_raise(num_prompt_tokens=65536, request_id="r-hot2")
    assert exc.value.estimated_bytes >= int(phys - hot_used + peak)


def test_guard_hot_cache_exclusion_clamps_and_keeps_active_floor():
    """A hot-cache figure larger than phys clamps the phys term to 0 instead
    of going negative, and the recorded MLX active sample still floors the
    usage — the exclusion must never eat into *GPU* pressure accounting."""
    guard = _make_guard()
    guard._prefill_memory_guard = True
    active = 1 * 1024**3
    phys = 2 * 1024**3
    guard.record_mlx_active_memory(active)
    guard._memory_hot_cache_used_bytes = 4 * 1024**3  # > phys → clamp to 0
    peak = guard.memory_monitor.estimate_prefill_peak_bytes(65536, 2048)

    # Usage must be the active floor (1 GiB), not raw phys (2 GiB): a limit
    # of active + peak fits with the clamp applied, not without.
    guard._memory_hard_limit_bytes = int(active + peak)
    with (
        patch("omlx.engine.dflash.get_phys_footprint", return_value=phys),
        patch(
            "omlx.memory_monitor.mx.get_active_memory",
            side_effect=AssertionError("preflight must not read MLX directly"),
        ),
    ):
        guard.preflight_or_raise(num_prompt_tokens=65536, request_id="r-clamp")

    # The active floor itself is never reduced by the exclusion.
    guard._memory_hard_limit_bytes = int(active + peak - 1)
    with (
        patch("omlx.engine.dflash.get_phys_footprint", return_value=phys),
        patch(
            "omlx.memory_monitor.mx.get_active_memory",
            side_effect=AssertionError("preflight must not read MLX directly"),
        ),
        pytest.raises(PrefillMemoryExceededError) as exc,
    ):
        guard.preflight_or_raise(num_prompt_tokens=65536, request_id="r-clamp2")
    assert exc.value.estimated_bytes >= int(active + peak)


def test_guard_rejects_cached_tokens():
    """The narrowed signature is deliberate: a DFlash prefix-cache hit
    reconstructs KV into active memory, so accepting a hit count here would
    under-count the prefill peak and defeat the OOM guard."""
    guard = _make_guard()
    guard._prefill_memory_guard = True
    guard._memory_hard_limit_bytes = 1
    with pytest.raises(TypeError):
        guard.preflight_or_raise(num_prompt_tokens=1000, cached_tokens=1000)


def test_preflight_noop_when_no_dims():
    """No model dims → estimator returns 0 → guard must not raise spuriously."""
    monitor = MemoryMonitor(max_kv_cache_memory=None, eviction_enabled=False)
    guard = _DFlashPrefillGuard(monitor, 2048)
    guard._prefill_memory_guard = True
    guard._memory_hard_limit_bytes = 1
    p1, p2 = _zero_mem()
    with p1, p2:
        guard.preflight_or_raise(num_prompt_tokens=65536)  # no exception


# --- engine-level delegation ----------------------------------------------


def _bare_engine() -> DFlashEngine:
    """A DFlashEngine with only the attrs preflight_* touches (no full init)."""
    eng = DFlashEngine.__new__(DFlashEngine)
    eng._loaded = True
    eng._in_fallback_mode = False
    eng._fallback_engine = None
    eng._prefill_guard = None
    return eng


async def test_engine_preflight_chat_delegates_to_guard():
    eng = _bare_engine()
    eng._prefill_guard = MagicMock()
    eng.count_chat_tokens = MagicMock(return_value=12345)

    await eng.preflight_chat([{"role": "user", "content": "hi"}], request_id="r1")

    eng._prefill_guard.preflight_or_raise.assert_called_once_with(
        num_prompt_tokens=12345, request_id="r1"
    )


async def test_engine_preflight_chat_delegates_to_fallback_in_fallback_mode():
    eng = _bare_engine()
    eng._in_fallback_mode = True
    eng._fallback_engine = AsyncMock()
    eng._prefill_guard = MagicMock()  # must NOT be consulted in fallback mode

    await eng.preflight_chat([{"role": "user", "content": "hi"}], request_id="r1")

    eng._fallback_engine.preflight_chat.assert_awaited_once()
    eng._prefill_guard.preflight_or_raise.assert_not_called()


async def test_engine_preflight_chat_noop_without_guard():
    eng = _bare_engine()  # _prefill_guard is None, not in fallback
    with patch("omlx.engine.dflash._warn_scheduler_unreachable_once") as warn:
        await eng.preflight_chat([{"role": "user", "content": "hi"}])
    warn.assert_called_once()


async def test_engine_preflight_completion_delegates_to_guard():
    eng = _bare_engine()
    eng._prefill_guard = MagicMock()
    eng._tokenizer_obj = MagicMock()
    eng._tokenizer_obj.encode.return_value = list(range(777))

    await eng.preflight_completion("hello", request_id="rc")

    eng._prefill_guard.preflight_or_raise.assert_called_once_with(
        num_prompt_tokens=777, request_id="rc"
    )
