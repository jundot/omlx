# SPDX-License-Identifier: Apache-2.0
"""A rank must refuse a prompt it cannot prefill — without hanging its peers."""

from __future__ import annotations

import pytest

from omlx.cluster.prefill_guard import RankPrefillGuard, build_guard, rank_monitor
from omlx.exceptions import PrefillMemoryExceededError

GiB = 1024**3


class _Config:
    """The dims mlx-lm models expose, minimal and real (Qwen3-32B shaped)."""

    num_hidden_layers = 64
    num_key_value_heads = 8
    num_attention_heads = 64
    head_dim = 128
    hidden_size = 5120


class _Model:
    args = _Config()


def _guard(*, layer_count=0, tp=1, ceiling=8 * GiB, rank=0) -> RankPrefillGuard:
    return RankPrefillGuard(
        rank_monitor(_Model(), layer_count=layer_count, tensor_parallel_size=tp),
        rank=rank,
        node_id="studio",
        ceiling_bytes=ceiling,
    )


def test_a_prompt_that_would_not_fit_is_refused():
    guard = _guard(ceiling=4 * GiB)
    with pytest.raises(PrefillMemoryExceededError) as excinfo:
        guard.check(200_000, current_usage_bytes=3 * GiB)
    assert "Prefill would require" in str(excinfo.value)


def test_a_prompt_that_fits_is_allowed():
    _guard(ceiling=64 * GiB).check(2048, current_usage_bytes=1 * GiB)


def test_a_pipeline_rank_is_only_charged_for_the_layers_it_holds():
    """The whole point: 16 of 64 layers must not be charged 64 layers of KV."""

    whole = rank_monitor(_Model())
    stage = rank_monitor(_Model(), layer_count=16)
    assert stage.estimate_prompt_kv_bytes(8192) == pytest.approx(
        whole.estimate_prompt_kv_bytes(8192) / 4, rel=0.01
    )


def test_a_stage_accepts_a_prompt_the_whole_model_would_refuse():
    """Not just smaller arithmetic — a prompt that is served instead of 400ed.

    The threshold is derived from the two estimates rather than guessed, so
    the test states the property and cannot drift with the SDPA model.
    """

    tokens, usage = 120_000, 4 * GiB
    whole = rank_monitor(_Model())
    stage = rank_monitor(_Model(), layer_count=16)
    stage_peak = stage.estimate_prefill_peak_bytes(tokens, 2048)
    whole_peak = whole.estimate_prefill_peak_bytes(tokens, 2048)
    assert stage_peak < whole_peak

    # A ceiling between the two: the uncorrected guard rejects, the corrected
    # one serves.
    ceiling = int(usage + (stage_peak + whole_peak) / 2)
    with pytest.raises(PrefillMemoryExceededError):
        _guard(ceiling=ceiling).check(tokens, current_usage_bytes=usage)
    _guard(layer_count=16, ceiling=ceiling).check(tokens, current_usage_bytes=usage)


def test_a_tensor_parallel_rank_is_charged_for_its_head_shard():
    whole = rank_monitor(_Model())
    half = rank_monitor(_Model(), tensor_parallel_size=2)
    assert half.estimate_prompt_kv_bytes(8192) == pytest.approx(
        whole.estimate_prompt_kv_bytes(8192) / 2, rel=0.01
    )


def test_cached_tokens_are_not_charged_twice():
    """Prefix-cache hits are already resident; charging them over-rejects."""

    guard = _guard(ceiling=6 * GiB)
    with pytest.raises(PrefillMemoryExceededError):
        guard.check(150_000, current_usage_bytes=4 * GiB)
    guard.check(150_000, cached_tokens=149_000, current_usage_bytes=4 * GiB)


def test_request_prefill_step_override_sizes_peak_without_mutating_guard():
    tokens = 120_000
    usage = 2 * GiB
    monitor = rank_monitor(_Model(), layer_count=16)
    configured_peak = monitor.estimate_prefill_peak_bytes(tokens, 2048)
    request_peak = monitor.estimate_prefill_peak_bytes(tokens, 16_384)
    assert configured_peak < request_peak
    guard = RankPrefillGuard(
        monitor,
        ceiling_bytes=int(usage + (configured_peak + request_peak) / 2),
        prefill_step_size=2048,
    )

    guard.check(tokens, current_usage_bytes=usage)
    with pytest.raises(PrefillMemoryExceededError):
        guard.check(
            tokens,
            current_usage_bytes=usage,
            prefill_step_size=16_384,
        )

    assert guard._step == 2048


# --- The desync rule: all ranks vote and leave the request together. ---------


def test_follower_ranks_guard_their_own_slice():
    follower = _guard(ceiling=1 * GiB, rank=1)
    assert follower.active
    with pytest.raises(PrefillMemoryExceededError):
        follower.check(500_000, current_usage_bytes=1 * GiB)


class _CollectiveValue:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


class _CollectiveMX:
    def __init__(self, *, rank, votes):
        self._rank = rank
        self._votes = votes
        self.distributed = self

    def init(self):
        return self

    def rank(self):
        return self._rank

    def size(self):
        return len(self._votes)

    def array(self, value):
        return value

    def all_sum(self, _value):
        return _CollectiveValue(self._votes)


def test_peer_rejection_makes_an_accepting_rank_leave_before_model_execution():
    guard = _guard(ceiling=64 * GiB, rank=0)
    mx = _CollectiveMX(rank=0, votes=[0, 1])

    with pytest.raises(PrefillMemoryExceededError, match="rejected by rank 1"):
        guard.check_collective(
            2048,
            current_usage_bytes=1 * GiB,
            mx_module=mx,
        )


def test_collective_admission_allows_every_rank_to_continue():
    guard = _guard(ceiling=64 * GiB, rank=1)
    mx = _CollectiveMX(rank=1, votes=[0, 0])

    guard.check_collective(
        2048,
        current_usage_bytes=1 * GiB,
        mx_module=mx,
    )


def test_an_unreadable_model_disables_the_guard():
    guard = RankPrefillGuard(rank_monitor(object()), rank=0, ceiling_bytes=8 * GiB)
    assert not guard.active
    guard.check(500_000)


def test_no_ceiling_disables_the_guard():
    assert not _guard(ceiling=0).active


def test_build_guard_uses_this_macs_ceiling(monkeypatch):
    monkeypatch.setattr(
        "omlx.cluster.memory_guard.ceiling_breakdown",
        lambda tier: {"hard_limit": 12 * GiB},
    )
    guard = build_guard(_Model(), rank=0, node_id="mbp", layer_count=32)
    assert guard.active
    assert guard._ceiling == 12 * GiB


def test_build_guard_survives_a_host_with_no_enforcer(monkeypatch):
    def _boom(_tier):
        raise RuntimeError("no enforcer here")

    monkeypatch.setattr("omlx.cluster.memory_guard.ceiling_breakdown", _boom)
    assert not build_guard(_Model(), rank=0).active
