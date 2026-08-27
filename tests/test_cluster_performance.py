# SPDX-License-Identifier: Apache-2.0
"""Performance-aware planner, launch probe, and runtime capability tests."""

import importlib
import json
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.cluster.deployment import ClusterDeployment, ClusterHost
from omlx.cluster.launch import DistributedLaunchError, run_cluster_performance_probe
from omlx.cluster.performance import (
    DeepseekAnePrefillSettings,
    ExecutionSettings,
    NodePerformanceProfile,
    execution_profile,
    performance_profiles_from_records,
    tune_execution_settings,
)
from omlx.cluster.planner import (
    ModelLayout,
    NodeBudget,
    PipelineAssignment,
    plan_unequal_pipeline,
)
from omlx.cluster.runtime_optimizations import (
    _deepseek_v4_fused_decode_capability,
    _deepseek_v4_outer_prefill_step,
    _indexer_row_parallel_capability,
    _MTPVocabCoordinator,
    install_runtime_optimizations,
    pipeline_prefill_schedule,
)

mlx_generate = importlib.import_module("mlx_lm.generate")


def test_deepseek_v4_fused_decode_capability_tracks_mtp_and_rollback(monkeypatch):
    attention = SimpleNamespace(dspark=True, _omlx_decode_consistent=False)
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(attn=attention)])
    )

    monkeypatch.delenv("OMLX_DSV4_EXACT_DECODE", raising=False)
    assert _deepseek_v4_fused_decode_capability(model)[:2] == (True, True)

    attention._omlx_decode_consistent = True
    enabled, active, reason = _deepseek_v4_fused_decode_capability(model)
    assert enabled is True
    assert active is False
    assert "MTP" in reason

    monkeypatch.setenv("OMLX_DSV4_EXACT_DECODE", "1")
    enabled, active, reason = _deepseek_v4_fused_decode_capability(model)
    assert enabled is False
    assert active is False
    assert "EXACT_DECODE" in reason


def test_sparse_indexer_row_parallel_capability_reports_tp_contract(monkeypatch):
    monkeypatch.setenv("OMLX_DSV4_INDEXER_ROW_TP", "1")
    monkeypatch.setenv("OMLX_DSV4_INDEXER_ROW_TP_MIN_POOL", "2048")
    group = object()
    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(
                    attn=SimpleNamespace(
                        indexer=SimpleNamespace(row_sharding_group=group)
                    )
                )
            ]
        )
    )

    enabled, active, reason = _indexer_row_parallel_capability(
        model, world_size=2
    )

    assert enabled is True
    assert active is True
    assert "2048 pooled entries" in reason


def _profile(node_id: str, rank: int, rate: float) -> NodePerformanceProfile:
    return NodePerformanceProfile(
        node_id=node_id,
        rank=rank,
        decode_weight_bytes_per_second=rate,
        prefill_weight_bytes_per_second=rate,
        collective_latency_seconds=0.001,
        collective_bandwidth_bytes_per_second=10_000,
        backend="ring",
        measured_at="2026-07-26T12:00:00+00:00",
        samples=5,
    )


def test_performance_planner_prefers_faster_node_without_exceeding_memory():
    model = ModelLayout(
        source="test",
        fixed_weight_bytes=0,
        layer_weight_bytes=(10,) * 8,
        activation_bytes_per_token=2,
    )
    plan = plan_unequal_pipeline(
        model,
        [
            NodeBudget(
                "slow",
                100,
                rank=0,
                performance=_profile("slow", 0, 10),
            ),
            NodeBudget(
                "fast",
                100,
                rank=1,
                performance=_profile("fast", 1, 40),
            ),
        ],
    )

    slow, fast = plan.assignments
    assert plan.optimization == "performance"
    assert fast.layer_count > slow.layer_count
    assert all(item.headroom_bytes >= 0 for item in plan.assignments)
    assert all(item.predicted_stage_seconds is not None for item in plan.assignments)
    assert plan.to_dict()["strategy"].startswith("performance_aware")


def test_partial_measurements_fall_back_to_original_memory_objective():
    model = ModelLayout(
        source="test",
        fixed_weight_bytes=0,
        layer_weight_bytes=(10,) * 8,
    )
    plan = plan_unequal_pipeline(
        model,
        [
            NodeBudget(
                "first",
                100,
                rank=0,
                performance=_profile("first", 0, 10),
            ),
            NodeBudget("second", 100, rank=1),
        ],
    )

    assert plan.optimization == "memory"
    assert [item.layer_count for item in plan.assignments] == [4, 4]
    assert all(item.predicted_stage_seconds is None for item in plan.assignments)


def test_execution_tuner_reduces_concurrency_and_synchronizes_prompt_cache():
    settings = execution_profile("throughput")
    assert settings.prompt_cache_ssd is False
    assert settings.prompt_cache_ssd_max_bytes == 20 * 1024**3
    assignments = [
        SimpleNamespace(headroom_bytes=3 * 1024**3),
        SimpleNamespace(headroom_bytes=20 * 1024**3),
    ]

    tuned = tune_execution_settings(settings, assignments, backend="jaccl")

    assert tuned.decode_concurrency == 2
    assert tuned.prompt_concurrency == 1
    assert tuned.prefill_step_size == 512
    assert tuned.pipeline_microbatch_size == 1
    assert tuned.prompt_cache_size == 2
    assert tuned.prompt_cache_bytes is None
    assert tuned.ring_connections_per_ip == 1
    assert "critical headroom" in tuned.tuning_reason
    assert "synchronized count-bounded prompt cache" in tuned.tuning_reason


def test_execution_tuner_retains_profile_cache_entries_for_prompt_segments():
    settings = execution_profile("balanced")
    assignments = [
        SimpleNamespace(headroom_bytes=32 * 1024**3),
        SimpleNamespace(headroom_bytes=24 * 1024**3),
    ]

    tuned = tune_execution_settings(settings, assignments, backend="jaccl")

    assert tuned.prompt_concurrency == 4
    assert tuned.prompt_cache_size == 8
    assert tuned.prompt_cache_bytes is None


def test_execution_settings_require_a_positive_ssd_snapshot_budget():
    with pytest.raises(ValueError, match="prompt_cache_ssd_max_bytes"):
        ExecutionSettings(prompt_cache_ssd_max_bytes=None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="prompt_cache_ssd_max_bytes"):
        ExecutionSettings(prompt_cache_ssd_max_bytes=0)


def test_deepseek_ane_execution_contract_round_trips_only_when_enabled():
    disabled = ExecutionSettings()
    assert "deepseek_ane_prefill" not in disabled.to_dict()

    ane = DeepseekAnePrefillSettings(
        enabled=True,
        sequence_length=4096,
        down_fraction=0.5,
        wo_a_enabled=False,
        cpu_enabled=False,
    )
    settings = replace(
        disabled,
        prefill_step_size=4096,
        deepseek_ane_prefill=ane,
    )

    restored = ExecutionSettings.from_dict(settings.to_dict())
    assert restored == settings
    assert restored.deepseek_ane_prefill.to_dict()["wo_a_enabled"] is False


def test_deepseek_ane_execution_contract_rejects_invalid_tile():
    with pytest.raises(ValueError, match="multiple of 64"):
        DeepseekAnePrefillSettings(enabled=True, sequence_length=4095)


def test_prompt_cache_is_synchronized_even_when_auto_tuning_is_disabled():
    settings = replace(
        execution_profile("throughput", auto_tune=False),
        prompt_cache_size=16,
        prompt_cache_bytes=8 * 1024**3,
    )

    tuned = tune_execution_settings(
        settings,
        [
            SimpleNamespace(headroom_bytes=3 * 1024**3),
            SimpleNamespace(headroom_bytes=20 * 1024**3),
        ],
        backend="jaccl",
    )

    assert tuned.decode_concurrency == settings.decode_concurrency
    assert tuned.prompt_cache_size == settings.prompt_cache_size
    assert tuned.prompt_cache_bytes is None
    assert "synchronized count-bounded prompt cache" in tuned.tuning_reason


def test_performance_profiles_reject_nonfinite_measurements():
    payload = _profile("node", 0, 10).to_dict()
    payload["decode_weight_bytes_per_second"] = float("nan")

    with pytest.raises(ValueError, match="finite positive"):
        NodePerformanceProfile.from_dict(payload)


def test_low_power_measurement_round_trips_as_nonpromotable():
    payload = _profile("node", 0, 10).to_dict() | {
        "promotable": False,
        "qualification_reason": "Low Power Mode was enabled during calibration",
    }

    profile = NodePerformanceProfile.from_dict(payload)

    assert profile.promotable is False
    assert "Low Power Mode" in profile.qualification_reason
    assert profile.to_dict()["promotable"] is False


def _deployment() -> ClusterDeployment:
    return ClusterDeployment(
        deployment_id="probe",
        model="org/model",
        backend="ring",
        hosts=(
            ClusterHost("local", "127.0.0.1", ("10.0.0.1",)),
            ClusterHost("peer", "peer.local", ("10.0.0.2",)),
        ),
        assignments=(
            PipelineAssignment("local", 0, 2, 4, 20, 0, 0, 100),
            PipelineAssignment("peer", 1, 0, 2, 20, 0, 0, 100),
        ),
        plan_hash="a" * 64,
        execution=replace(
            execution_profile("balanced"),
            ring_connections_per_ip=3,
        ),
    )


def test_cluster_performance_probe_uses_ring_connections_and_validates_ranks():
    def runner(argv, *, timeout, env):
        assert timeout == 12.0
        assert argv[argv.index("--connections-per-ip") + 1] == "3"
        assert "omlx.cluster.performance_worker" in argv
        assert env["SSH_ASKPASS_REQUIRE"] == "never"
        records = [
            {
                "type": "performance_result",
                "rank": rank,
                "size": 2,
                "decode_weight_bytes_per_second": 100 + rank,
                "prefill_weight_bytes_per_second": 200 + rank,
                "collective_latency_seconds": 0.001,
                "collective_bandwidth_bytes_per_second": 10_000,
                "samples": 5,
                "measured_at": "2026-07-26T12:00:00+00:00",
            }
            for rank in (0, 1)
        ]
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="\n".join(json.dumps(record) for record in records),
            stderr="",
        )

    report = run_cluster_performance_probe(
        _deployment(),
        timeout=12.0,
        python_executable="/opt/omlx/bin/python",
        runner=runner,
    )

    assert report["ok"] is True
    assert report["connections_per_ip"] == 3
    profiles = performance_profiles_from_records(
        [
            {"type": "noise"},
            *[
                {"type": "performance_result"} | profile
                for profile in report["profiles"]
            ],
        ],
        node_ids=("local", "peer"),
        backend="ring",
    )
    assert [profile.rank for profile in profiles] == [0, 1]


def test_cluster_performance_probe_never_passes_ring_connections_to_jaccl():
    deployment = replace(
        _deployment(),
        backend="jaccl",
        hosts=(
            ClusterHost(
                "local",
                "127.0.0.1",
                ("10.0.0.1",),
                (None, "rdma_en5"),
            ),
            ClusterHost(
                "peer",
                "peer.local",
                ("10.0.0.2",),
                ("rdma_en5", None),
            ),
        ),
    )

    def runner(argv, *, timeout, env):
        assert "--connections-per-ip" not in argv
        records = [
            {
                "type": "performance_result",
                "rank": rank,
                "size": 2,
                "decode_weight_bytes_per_second": 100 + rank,
                "prefill_weight_bytes_per_second": 200 + rank,
                "collective_latency_seconds": 0.001,
                "collective_bandwidth_bytes_per_second": 10_000,
                "samples": 5,
                "measured_at": "2026-07-26T12:00:00+00:00",
            }
            for rank in (0, 1)
        ]
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="\n".join(json.dumps(record) for record in records),
            stderr="",
        )

    report = run_cluster_performance_probe(deployment, runner=runner)

    assert report["ok"] is True
    assert report["backend"] == "jaccl"
    assert report["connections_per_ip"] == 1


def test_cluster_performance_probe_marks_low_power_records_unpromotable():
    def runner(argv, *, timeout, env):
        records = [
            {
                "type": "performance_result",
                "rank": rank,
                "size": 2,
                "decode_weight_bytes_per_second": 100 + rank,
                "prefill_weight_bytes_per_second": 200 + rank,
                "collective_latency_seconds": 0.001,
                "collective_bandwidth_bytes_per_second": 10_000,
                "samples": 5,
                "measured_at": "2026-07-26T12:00:00+00:00",
                "promotable": rank == 0,
                "qualification_reason": (
                    "" if rank == 0 else "Low Power Mode was enabled"
                ),
            }
            for rank in (0, 1)
        ]
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="\n".join(json.dumps(record) for record in records),
            stderr="",
        )

    report = run_cluster_performance_probe(_deployment(), runner=runner)

    assert report["ok"] is True
    assert report["promotable"] is False
    assert report["unqualified_ranks"] == [1]
    assert "Low Power Mode" in report["qualification_reason"]
    assert report["profiles"][1]["promotable"] is False


def test_cluster_performance_probe_includes_rank_worker_diagnostic_on_missing_rank():
    def runner(argv, *, timeout, env):
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="oMLX performance probe failed: ValueError: [jaccl] Changing queue pair to RTR failed with errno 96",
            stderr="[WARN] Node with rank 1 exited with code 1",
        )

    with pytest.raises(DistributedLaunchError, match="queue pair.*RTR"):
        run_cluster_performance_probe(_deployment(), runner=runner)


class _ValidatedPipeline:
    pipeline_rank = 0
    pipeline_size = 2

    def __init__(self):
        self.seen = []

    def __call__(self, value, cache=None):
        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size
        self.seen.append(value.tolist())
        if pipeline_rank != 0:
            value = mx.distributed.send(
                value,
                (pipeline_rank - 1) % pipeline_size,
            )
        if pipeline_size > 1:
            value = mx.distributed.all_gather(value)
        return value


class _Group:
    @staticmethod
    def rank():
        return 0

    @staticmethod
    def size():
        return 2


class _WorkerGroup:
    @staticmethod
    def rank():
        return 1

    @staticmethod
    def size():
        return 2


@pytest.mark.parametrize(
    ("group", "proposal", "expected_input"),
    [
        (_Group(), mx.array([129279], dtype=mx.uint32), [129279]),
        (_WorkerGroup(), None, [0]),
    ],
)
def test_mtp_token_decision_uses_point_to_point_transport(
    monkeypatch, group, proposal, expected_input
):
    observed = []

    def send(value, target):
        observed.append(("send", target, value.dtype, value.tolist()))
        return value

    def recv_like(value, source):
        observed.append(("recv", source, value.dtype, value.tolist()))
        return mx.array([129279], dtype=mx.int32)

    monkeypatch.setattr(mx.distributed, "send", send)
    monkeypatch.setattr(mx.distributed, "recv_like", recv_like)
    coordinator = _MTPVocabCoordinator(mx, group, output_size=129280)

    result = coordinator.sync_tokens(proposal, (1,))

    if group.rank() == 0:
        assert observed == [("send", 1, mx.int32, expected_input)]
    else:
        assert observed == [("recv", 0, mx.int32, expected_input)]
    assert result.dtype == mx.uint32
    assert result.tolist() == [129279]


def test_mtp_decision_packet_uses_point_to_point_transport(monkeypatch):
    observed = []

    def send(value, target):
        observed.append((target, value.dtype, value.tolist()))
        return value

    monkeypatch.setattr(mx.distributed, "send", send)
    coordinator = _MTPVocabCoordinator(mx, _Group(), output_size=129280)

    result = coordinator.sync_packet(mx.array([3, 42, 7]), 3)

    assert result == [3, 42, 7]
    assert observed == [(1, mx.int32, [3, 42, 7])]


def test_asymmetric_tensor_capability_reports_the_local_fraction(monkeypatch):
    monkeypatch.setenv("OMLX_TP_SHARD_WEIGHTS", "3,5")
    with install_runtime_optimizations(
        SimpleNamespace(model=SimpleNamespace()),
        _WorkerGroup(),
        execution_profile("balanced"),
        batchable=False,
        pipeline_parallel=False,
    ) as capabilities:
        item = capabilities["asymmetric_tensor_parallel"]
        assert item["active"] is True
        assert "rank 1 holds 5/8" in item["reason"]


def test_sampling_rank_optimization_is_capability_gated_and_restored():
    settings = replace(
        execution_profile("balanced"),
        sampling_rank_only=True,
    )
    model = SimpleNamespace(model=_ValidatedPipeline())
    original_gather = mx.distributed.all_gather
    original_send = mx.distributed.send
    original_call = _ValidatedPipeline.__call__
    original_step = mlx_generate.GenerationBatch._step
    original_prompt = mlx_generate.PromptProcessingBatch.prompt

    with install_runtime_optimizations(
        model,
        _Group(),
        settings,
        batchable=True,
    ) as capabilities:
        assert capabilities["sampling_rank_only"]["active"] is True
        assert capabilities["rank_zero_logits"]["active"] is False
        assert capabilities["pipeline_prefill_overlap"]["active"] is True, (
            capabilities["pipeline_prefill_overlap"]["reason"]
        )
        assert mx.distributed.all_gather is not original_gather
        assert mx.distributed.send is not original_send
        assert _ValidatedPipeline.__call__ is not original_call
        assert mlx_generate.GenerationBatch._step is not original_step
        assert mlx_generate.PromptProcessingBatch.prompt is not original_prompt

    assert mx.distributed.all_gather is original_gather
    assert mx.distributed.send is original_send
    assert _ValidatedPipeline.__call__ is original_call
    assert mlx_generate.GenerationBatch._step is original_step
    assert mlx_generate.PromptProcessingBatch.prompt is original_prompt


def test_worker_rank_skips_vocab_projection_when_adapter_declares_contract(
    monkeypatch,
):
    class Cache:
        state = mx.array([0])

    class RankLocalLogitsModel:
        _omlx_supports_rank_zero_logits = True
        _omlx_output_vocab_size = 32

        def __init__(self):
            self.model = _ValidatedPipeline()
            self.model.pipeline_rank = 1
            self.calls = []

        def __call__(self, value, cache=None, skip_logits=False):
            self.calls.append(skip_logits)
            value = self.model(value, cache=cache)
            if skip_logits:
                return None
            return mx.zeros((*value.shape, self._omlx_output_vocab_size))

    class Batch:
        def __init__(self, model):
            self.model = model
            self.uids = [1]
            self.prompt_cache = [Cache()]
            self.tokens = [[]]
            self.samplers = [None]
            self.fallback_sampler = lambda value: mx.argmax(value, axis=-1)
            self.logits_processors = [[]]
            self.state_machines = []
            self.max_tokens = [2]
            self._current_tokens = None
            self._current_logprobs = []
            self._next_tokens = mx.array([3], dtype=mx.uint32)
            self._next_logprobs = []
            self._token_context = []
            self._num_tokens = [0]
            self._matcher_states = []

    model = RankLocalLogitsModel()
    batch = Batch(model)
    settings = replace(
        execution_profile("balanced"),
        sampling_rank_only=True,
    )
    monkeypatch.setattr(
        mx.distributed,
        "all_sum",
        lambda value, group=None: value,
    )
    monkeypatch.setattr(mx.distributed, "send", lambda value, *_a, **_k: value)
    monkeypatch.setattr(
        mx.distributed,
        "recv_like",
        lambda _value, _source: mx.array([5], dtype=mx.int32),
    )
    monkeypatch.setattr(mx.distributed, "all_gather", lambda value, **_k: value)
    monkeypatch.setattr(mx, "async_eval", lambda *_values: None)

    with install_runtime_optimizations(
        model,
        _WorkerGroup(),
        settings,
        batchable=True,
    ) as capabilities:
        assert capabilities["rank_zero_logits"]["active"] is True
        tokens, _ = mlx_generate.GenerationBatch._step(batch)

    assert model.calls == [True]
    assert tokens == [3]
    assert type(tokens[0]) is int
    assert batch._next_tokens.tolist() == [5]
    assert len(batch._next_logprobs) == 1
    assert batch._next_logprobs[0].shape == (32,)


def test_pipeline_prefill_schedule_has_equal_fill_and_drain_timeline():
    schedules = [
        pipeline_prefill_schedule(10, 4, rank=rank, world_size=3)
        for rank in range(3)
    ]

    assert {len(schedule) for schedule in schedules} == {5}
    # MLX-LM runs the first stage on the highest rank and the final stage on
    # rank zero, so the Exo fill/drain offset is mirrored.
    assert [(slot.start, slot.end) for slot in schedules[0]] == [
        (None, None),
        (None, None),
        (0, 4),
        (4, 8),
        (8, 10),
    ]
    assert [(slot.start, slot.end) for slot in schedules[2]] == [
        (0, 4),
        (4, 8),
        (8, 10),
        (None, None),
        (None, None),
    ]
    assert all(sum(slot.is_real for slot in schedule) == 3 for schedule in schedules)


def test_staggered_prompt_queues_and_flushes_every_real_chunk(monkeypatch):
    sends = []
    gathers = []
    async_values = []
    original_prompt = mlx_generate.PromptProcessingBatch.prompt

    monkeypatch.setattr(
        mx.distributed,
        "send",
        lambda value, destination, **kwargs: sends.append(destination) or value,
    )
    monkeypatch.setattr(
        mx.distributed,
        "all_gather",
        lambda value, **kwargs: gathers.append(value) or value,
    )
    monkeypatch.setattr(mx, "async_eval", lambda *values: async_values.extend(values))

    class Cache:
        state = mx.array([0])

    class Batch:
        uids = ["request"]
        tokens = [[]]
        prompt_cache = [Cache()]
        prefill_step_size = 8

        def __init__(self):
            self.model = _ValidatedPipeline()
            self.model.pipeline_rank = 1

    settings = replace(
        execution_profile("balanced"),
        sampling_rank_only=True,
        async_overlap=True,
        prefill_step_size=8,
    )
    model = SimpleNamespace(model=_ValidatedPipeline())
    batch = Batch()

    with install_runtime_optimizations(
        model,
        _Group(),
        settings,
        batchable=True,
    ) as capabilities:
        assert capabilities["pipeline_prefill_overlap"]["active"] is True, (
            capabilities["pipeline_prefill_overlap"]["reason"]
        )
        mlx_generate.PromptProcessingBatch.prompt(batch, [list(range(9))])

    # The scheduler honours the same eight-token step the memory guard approved,
    # so 9 tokens make two real chunks. Each chunk reaches send; the final
    # hidden-state gather is skipped.
    assert sends == [0, 0]
    assert len(async_values) == 2
    assert gathers == []
    assert batch.tokens == [list(range(9))]
    assert mlx_generate.PromptProcessingBatch.prompt is original_prompt


def test_staggered_prompt_matches_stock_chunking_padding_and_cache_lifecycle(
    monkeypatch,
):
    """The faster scheduler must preserve MLX-LM's prompt/cache contract."""

    original_prompt = mlx_generate.PromptProcessingBatch.prompt
    monkeypatch.setattr(mx.distributed, "send", lambda value, *_a, **_k: value)
    monkeypatch.setattr(mx.distributed, "all_gather", lambda value, **_k: value)
    monkeypatch.setattr(mx, "async_eval", lambda *_values: None)

    class Cache:
        def __init__(self):
            self.state = mx.array([0])
            self.events = []

        def prepare(self, *, lengths, right_padding):
            self.events.append(("prepare", tuple(lengths), tuple(right_padding)))

        def finalize(self):
            self.events.append(("finalize",))

    class Batch:
        uids = ["first", "second"]
        prefill_step_size = 8

        def __init__(self):
            self.tokens = [[], []]
            self.prompt_cache = [Cache()]
            self.model = _ValidatedPipeline()
            self.model.pipeline_rank = 1

    prompts = [list(range(9)), list(range(20, 25))]
    stock = Batch()
    original_prompt(stock, [list(prompt) for prompt in prompts])

    patched = Batch()
    settings = replace(
        execution_profile("balanced"),
        sampling_rank_only=True,
        async_overlap=True,
        prefill_step_size=8,
    )
    with install_runtime_optimizations(
        SimpleNamespace(model=_ValidatedPipeline()),
        _Group(),
        settings,
        batchable=True,
    ):
        mlx_generate.PromptProcessingBatch.prompt(
            patched,
            [list(prompt) for prompt in prompts],
        )

    assert patched.model.seen == stock.model.seen
    assert [len(chunk[0]) for chunk in patched.model.seen] == [8, 1]
    assert patched.tokens == stock.tokens == prompts
    assert patched.prompt_cache[0].events == stock.prompt_cache[0].events


def test_sampling_rank_optimization_keeps_normal_path_for_unvalidated_model():
    settings = replace(
        execution_profile("interactive"),
        sampling_rank_only=True,
    )
    model = SimpleNamespace(model=SimpleNamespace())
    original_gather = mx.distributed.all_gather

    with install_runtime_optimizations(
        model,
        _Group(),
        settings,
        batchable=True,
    ) as capabilities:
        assert capabilities["sampling_rank_only"]["active"] is False
        assert capabilities["pipeline_prefill_overlap"]["active"] is False
        assert mx.distributed.all_gather is original_gather


def test_tensor_prefill_skips_discarded_vocab_projection(monkeypatch):
    class Cache:
        state = mx.array([0])

    class SkipHeadModel:
        def __init__(self):
            self.calls = []

        def __call__(self, value, cache=None, skip_lm_head=False):
            self.calls.append((value.shape[1], skip_lm_head))
            return None if skip_lm_head else mx.zeros((*value.shape, 32))

    class Batch:
        uids = ["request"]
        prefill_step_size = 4

        def __init__(self, model):
            self.model = model
            self.tokens = [[]]
            self.prompt_cache = [Cache()]

    monkeypatch.setattr(mx, "clear_cache", lambda: None)
    model = SkipHeadModel()
    batch = Batch(model)
    original_prompt = mlx_generate.PromptProcessingBatch.prompt

    with install_runtime_optimizations(
        model,
        _Group(),
        execution_profile("balanced"),
        batchable=True,
        pipeline_parallel=False,
    ) as capabilities:
        assert capabilities["prefill_logits_skip"]["active"] is True
        assert capabilities["sampling_rank_only"]["active"] is False
        mlx_generate.PromptProcessingBatch.prompt(batch, [list(range(9))])

    assert model.calls == [(4, True), (4, True), (1, True)]
    assert batch.tokens == [list(range(9))]
    assert mlx_generate.PromptProcessingBatch.prompt is original_prompt


def test_ds4_tensor_prefill_switches_from_2k_to_1k_after_4k(monkeypatch):
    class Cache:
        state = mx.array([0])

    class Attention:
        dspark = True

    class Layer:
        attn = Attention()

    class DS4Model:
        def __init__(self):
            self.model = SimpleNamespace(layers=[Layer()])
            self.calls = []

        def __call__(self, value, cache=None, skip_lm_head=False):
            self.calls.append((value.shape[1], skip_lm_head))
            return None

    class Batch:
        uids = ["request"]
        prefill_step_size = 4096

        def __init__(self, model):
            self.model = model
            self.tokens = [[]]
            self.prompt_cache = [Cache()]

    monkeypatch.setenv("OMLX_DSV4_ADAPTIVE_PREFILL", "1")
    monkeypatch.setenv("OMLX_DSV4_ADAPTIVE_PREFILL_AFTER", "4096")
    monkeypatch.setenv("OMLX_DSV4_ADAPTIVE_PREFILL_STEP", "1024")
    monkeypatch.setenv("OMLX_DSV4_ADAPTIVE_PREFILL_MAX_BASE", "2048")
    monkeypatch.delenv("OMLX_CLUSTER_PREFILL_SHAPE_WARMUP", raising=False)
    clears = []
    monkeypatch.setattr(mx, "clear_cache", lambda: clears.append(True))
    model = DS4Model()

    with install_runtime_optimizations(
        model,
        _Group(),
        execution_profile("throughput"),
        batchable=True,
        pipeline_parallel=False,
    ) as capabilities:
        assert capabilities["deepseek_v4_adaptive_prefill"]["active"] is True
        assert (
            capabilities["deepseek_v4_adaptive_prefill"][
                "shape_warmup_tokens"
            ]
            == 1024
        )
        batch = Batch(model)
        mlx_generate.PromptProcessingBatch.prompt(batch, [list(range(2048))])
        mlx_generate.PromptProcessingBatch.prompt(batch, [list(range(2048))])
        mlx_generate.PromptProcessingBatch.prompt(batch, [list(range(3072))])

    assert model.calls == [
        (2048, True),
        (2048, True),
        (1024, True),
        (1024, True),
        (1024, True),
    ]
    assert clears == [True, True, True]

    monkeypatch.setenv("OMLX_CLUSTER_PREFILL_SHAPE_WARMUP", "0")
    with install_runtime_optimizations(
        model,
        _Group(),
        execution_profile("throughput"),
        batchable=True,
        pipeline_parallel=False,
    ) as capabilities:
        assert (
            capabilities["deepseek_v4_adaptive_prefill"][
                "shape_warmup_tokens"
            ]
            == 0
        )


def test_ds4_long_request_uses_1k_from_its_first_outer_chunk(monkeypatch):
    class Cache:
        state = mx.array([0])

    class Attention:
        dspark = True

    class Model:
        def __init__(self):
            self.model = SimpleNamespace(
                layers=[SimpleNamespace(attn=Attention())]
            )
            self.calls = []

        def __call__(self, value, cache=None, skip_lm_head=False):
            self.calls.append(value.shape[1])

    class Batch:
        uids = [7]
        prefill_step_size = 2048
        _omlx_total_prompt_lengths = {7: 14_000}

        def __init__(self, model):
            self.model = model
            self.tokens = [[]]
            self.prompt_cache = [Cache()]

    monkeypatch.setattr(mx, "clear_cache", lambda: None)
    model = Model()
    with install_runtime_optimizations(
        model,
        _Group(),
        execution_profile("balanced"),
        batchable=True,
        pipeline_parallel=False,
    ):
        mlx_generate.PromptProcessingBatch.prompt(
            Batch(model),
            [list(range(2048))],
        )

    assert model.calls == [1024, 1024]


class _AsyncDS4Cache:
    def __init__(self):
        self.state = mx.array([0], dtype=mx.int32)


class _AsyncDS4Model:
    def __init__(self, dsv4, events, *, fail_on=None):
        self.model = SimpleNamespace(
            layers=[SimpleNamespace(attn=SimpleNamespace(dspark=True))]
        )
        self._dsv4 = dsv4
        self._events = events
        self._fail_on = fail_on
        self._calls = 0
        self.labels = {}

    def __call__(self, value, cache=None, skip_lm_head=False):
        assert skip_lm_head is True
        self._calls += 1
        current = mx.array([self._calls], dtype=mx.int32)
        cache[0].state = current
        self.labels[id(current)] = self._calls
        self._events.append(("model", self._calls, int(value.shape[1])))
        self._dsv4._materialize_cache_arrays(cache)
        if self._calls == self._fail_on:
            raise RuntimeError("synthetic async prefill failure")


class _AsyncDS4Batch:
    uids = [7]
    prefill_step_size = 2048
    _omlx_total_prompt_lengths = {7: 14_000}

    def __init__(self, model):
        self.model = model
        self.tokens = [[]]
        self.prompt_cache = [_AsyncDS4Cache()]


class _SingleRankGroup:
    @staticmethod
    def rank():
        return 0

    @staticmethod
    def size():
        return 1


def _async_ds4_module():
    import sys

    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()
    return sys.modules["mlx_lm.models.deepseek_v4"]


def test_ds4_tp2_depth_two_prefill_queues_current_then_completes_previous(
    monkeypatch,
):
    dsv4 = _async_ds4_module()
    events = []
    model = _AsyncDS4Model(dsv4, events)
    batch = _AsyncDS4Batch(model)
    monkeypatch.setenv("OMLX_DSV4_PREFILL_ASYNC_DEPTH", "2")
    monkeypatch.delenv("OMLX_CLUSTER_PREFILL_CLEAR_CACHE_EVERY", raising=False)

    def record(kind, arrays):
        events.append((kind, tuple(model.labels[id(value)] for value in arrays)))

    monkeypatch.setattr(mx, "async_eval", lambda *arrays: record("async", arrays))
    monkeypatch.setattr(mx, "eval", lambda *arrays: record("eval", arrays))
    monkeypatch.setattr(mx, "clear_cache", lambda: events.append(("clear",)))

    original_prompt = mlx_generate.PromptProcessingBatch.prompt
    with install_runtime_optimizations(
        model,
        _Group(),
        execution_profile("balanced"),
        batchable=True,
        pipeline_parallel=False,
    ) as capabilities:
        item = capabilities["deepseek_v4_prefill_async"]
        assert item["enabled"] is True
        assert item["active"] is True
        mlx_generate.PromptProcessingBatch.prompt(batch, [list(range(2048))])

    assert events == [
        ("model", 1, 1024),
        ("async", (1,)),
        ("model", 2, 1024),
        ("async", (2,)),
        ("eval", (1,)),
        ("eval", (2,)),
        ("clear",),
    ]
    assert model.labels[id(batch.prompt_cache[0].state)] == 2
    assert mlx_generate.PromptProcessingBatch.prompt is original_prompt


def test_ds4_tp2_depth_two_prefill_drains_previous_and_partial_on_error(
    monkeypatch,
):
    dsv4 = _async_ds4_module()
    events = []
    model = _AsyncDS4Model(dsv4, events, fail_on=2)
    batch = _AsyncDS4Batch(model)
    monkeypatch.setenv("OMLX_DSV4_PREFILL_ASYNC_DEPTH", "2")
    monkeypatch.delenv("OMLX_CLUSTER_PREFILL_CLEAR_CACHE_EVERY", raising=False)

    def record(kind, arrays):
        events.append((kind, tuple(model.labels[id(value)] for value in arrays)))

    monkeypatch.setattr(mx, "async_eval", lambda *arrays: record("async", arrays))
    monkeypatch.setattr(mx, "eval", lambda *arrays: record("eval", arrays))
    monkeypatch.setattr(mx, "clear_cache", lambda: events.append(("clear",)))

    original_prompt = mlx_generate.PromptProcessingBatch.prompt
    with install_runtime_optimizations(
        model,
        _Group(),
        execution_profile("balanced"),
        batchable=True,
        pipeline_parallel=False,
    ):
        with pytest.raises(RuntimeError, match="synthetic async prefill failure"):
            mlx_generate.PromptProcessingBatch.prompt(
                batch,
                [list(range(2048))],
            )

    assert events == [
        ("model", 1, 1024),
        ("async", (1,)),
        ("model", 2, 1024),
        ("eval", (1,)),
        ("eval", (2,)),
    ]
    assert mlx_generate.PromptProcessingBatch.prompt is original_prompt


def test_ds4_tp2_depth_two_prefill_keeps_one_chunk_fairness_slice_sync(
    monkeypatch,
):
    dsv4 = _async_ds4_module()
    events = []
    model = _AsyncDS4Model(dsv4, events)
    batch = _AsyncDS4Batch(model)
    monkeypatch.setenv("OMLX_DSV4_PREFILL_ASYNC_DEPTH", "2")
    monkeypatch.delenv("OMLX_CLUSTER_PREFILL_CLEAR_CACHE_EVERY", raising=False)
    monkeypatch.setattr(mx, "async_eval", lambda *_arrays: events.append(("async",)))
    monkeypatch.setattr(mx, "eval", lambda *_arrays: events.append(("eval",)))
    monkeypatch.setattr(mx, "clear_cache", lambda: events.append(("clear",)))

    with install_runtime_optimizations(
        model,
        _Group(),
        execution_profile("balanced"),
        batchable=True,
        pipeline_parallel=False,
    ) as capabilities:
        assert capabilities["deepseek_v4_prefill_async"]["active"] is True
        mlx_generate.PromptProcessingBatch.prompt(batch, [list(range(1024))])

    assert [event for event in events if event[0] == "model"] == [
        ("model", 1, 1024)
    ]
    assert not any(event[0] == "async" for event in events)


def test_ds4_tp2_prefill_async_is_default_off_and_rejects_other_depths(
    monkeypatch,
):
    dsv4 = _async_ds4_module()
    model = _AsyncDS4Model(dsv4, [])
    monkeypatch.delenv("OMLX_DSV4_PREFILL_ASYNC_DEPTH", raising=False)

    with install_runtime_optimizations(
        model,
        _Group(),
        execution_profile("balanced"),
        batchable=True,
        pipeline_parallel=False,
    ) as capabilities:
        item = capabilities["deepseek_v4_prefill_async"]
        assert item["enabled"] is False
        assert item["active"] is False
        assert "disabled by the operator" in item["reason"]

    monkeypatch.setenv("OMLX_DSV4_PREFILL_ASYNC_DEPTH", "1")
    with pytest.raises(ValueError, match="must be 0 or 2"):
        with install_runtime_optimizations(
            model,
            _Group(),
            execution_profile("balanced"),
            batchable=True,
            pipeline_parallel=False,
        ):
            pass


@pytest.mark.parametrize(
    ("settings", "mtp", "clear_every", "reason"),
    [
        (
            replace(execution_profile("balanced"), async_overlap=False),
            False,
            "0",
            "async",
        ),
        (execution_profile("balanced"), True, "0", "MTP"),
        (execution_profile("balanced"), False, "2", "clear-cache cadence 0"),
    ],
)
def test_ds4_tp2_prefill_async_static_safety_gates(
    monkeypatch, settings, mtp, clear_every, reason
):
    dsv4 = _async_ds4_module()
    model = _AsyncDS4Model(dsv4, [])
    model._omlx_mtp_decode_enabled = mtp
    monkeypatch.setenv("OMLX_DSV4_PREFILL_ASYNC_DEPTH", "2")
    monkeypatch.setenv("OMLX_CLUSTER_PREFILL_CLEAR_CACHE_EVERY", clear_every)

    with install_runtime_optimizations(
        model,
        _Group(),
        settings,
        batchable=True,
        pipeline_parallel=False,
    ) as capabilities:
        item = capabilities["deepseek_v4_prefill_async"]
        assert item["enabled"] is True
        assert item["active"] is False
        assert reason in item["reason"]


@pytest.mark.parametrize(
    ("group", "pipeline_parallel"),
    [(_SingleRankGroup(), False), (_Group(), True)],
)
def test_ds4_prefill_async_requires_pure_tp2(
    monkeypatch, group, pipeline_parallel
):
    dsv4 = _async_ds4_module()
    model = _AsyncDS4Model(dsv4, [])
    monkeypatch.setenv("OMLX_DSV4_PREFILL_ASYNC_DEPTH", "2")
    monkeypatch.delenv("OMLX_CLUSTER_PREFILL_CLEAR_CACHE_EVERY", raising=False)

    with install_runtime_optimizations(
        model,
        group,
        execution_profile("balanced"),
        batchable=True,
        pipeline_parallel=pipeline_parallel,
    ) as capabilities:
        item = capabilities["deepseek_v4_prefill_async"]
        assert item["enabled"] is True
        assert item["active"] is False
        assert "requires pure TP2" in item["reason"]


def _outer_prefill_batch(
    *,
    generation_uids=(),
    pending=(),
    current=(),
    prefill_limit=4,
    completion_limit=8,
):
    """Minimal mirrored MLX-LM BatchGenerator state for scheduling tests."""

    prompt_uids = [uid for uid, _segments, _total in current]
    totals = {uid: total for uid, _segments, total in current}
    tokens = {uid: [0] * total for uid, _segments, total in current}
    staged_current = [
        [segments, 0, total] for _uid, segments, total in current
    ]
    staged_pending = []
    for uid, segments, total in pending:
        totals[uid] = total
        tokens[uid] = [0] * total
        staged_pending.append((uid, segments, 1, None, [], None, None, None))
    return SimpleNamespace(
        prefill_step_size=2048,
        prefill_batch_size=prefill_limit,
        completion_batch_size=completion_limit,
        _generation_batch=SimpleNamespace(uids=list(generation_uids)),
        _prompt_batch=SimpleNamespace(
            uids=prompt_uids,
            tokens=[[] for _uid in prompt_uids],
            _omlx_total_prompt_lengths=totals,
        ),
        _currently_processing=staged_current,
        _unprocessed_sequences=staged_pending,
        _omlx_tokens=tokens,
    )


class _PromptRows:
    def __init__(self, uids, totals):
        self.uids = list(uids)
        self.tokens = [[] for _ in self.uids]
        self._omlx_total_prompt_lengths = dict(totals)
        self.prompted = None

    def __len__(self):
        return len(self.uids)

    def _filter(self, indices):
        self.uids = [self.uids[index] for index in indices]
        self.tokens = [self.tokens[index] for index in indices]

    def split(self, indices):
        selected = sorted(indices)
        left = [
            index for index in range(len(self.uids)) if index not in selected
        ]
        selected_batch = _PromptRows(
            [self.uids[index] for index in selected],
            self._omlx_total_prompt_lengths,
        )
        self._filter(left)
        return selected_batch

    def extend(self, other):
        self.uids.extend(other.uids)
        self.tokens.extend(other.tokens)

    def generate(self, _last_inputs):
        return SimpleNamespace(uids=list(self.uids))

    def prompt(self, prompts):
        self.prompted = [list(prompt) for prompt in prompts]


def test_ds4_outer_prefill_yield_is_decode_and_request_aware():
    long_pending = ((7, [[0] * 6000], 6000),)
    short_pending = ((8, [[0] * 4096], 4096),)

    active = _outer_prefill_batch(
        generation_uids=(99,),
        pending=long_pending,
    )
    assert (
        _deepseek_v4_outer_prefill_step(
            active,
            long_after=4096,
            kernel_step=1024,
        )
        == 256
    )

    # Idle long requests retain the wider outer slice (and its lower scheduler
    # / allocator overhead), while the adaptive prompt loop still performs the
    # measured-fast 1024-token model calls internally.
    idle = _outer_prefill_batch(pending=long_pending)
    assert (
        _deepseek_v4_outer_prefill_step(
            idle,
            long_after=4096,
            kernel_step=1024,
        )
        == 2048
    )

    # A short request does not opt into the long-context kernel schedule, so
    # the fairness wrapper must not silently change its prefill shape.
    short = _outer_prefill_batch(
        generation_uids=(99,),
        pending=short_pending,
    )
    assert (
        _deepseek_v4_outer_prefill_step(
            short,
            long_after=4096,
            kernel_step=1024,
        )
        == 2048
    )

    # Only rows that can enter this turn count. A long request behind the one
    # available prompt slot cannot shrink the short request ahead of it.
    queued = _outer_prefill_batch(
        generation_uids=(99,),
        pending=short_pending + long_pending,
        prefill_limit=1,
    )
    assert (
        _deepseek_v4_outer_prefill_step(
            queued,
            long_after=4096,
            kernel_step=1024,
        )
        == 2048
    )


def test_ds4_outer_prefill_yields_when_decode_promotes_this_turn():
    batch = _outer_prefill_batch(
        current=(
            (1, [[1]], 1),  # promoted to generation before prompt processing
            (2, [[0] * 6000], 6000),
        ),
    )

    assert (
        _deepseek_v4_outer_prefill_step(
            batch,
            long_after=4096,
            kernel_step=1024,
        )
        == 256
    )


def test_ds4_outer_prefill_matches_admission_before_boundary_promotion():
    batch = _outer_prefill_batch(
        generation_uids=tuple(range(7)),
        current=((10, [[1]], 1),),
        pending=((11, [[0] * 6000], 6000),),
        completion_limit=8,
    )

    # Pinned MLX-LM admits the pending long row while generation is B7, then
    # promotes the resident boundary row to B8. The long prompt still executes
    # in that turn, so its outer slice must be bounded at B8 pressure.
    assert (
        _deepseek_v4_outer_prefill_step(
            batch,
            long_after=4096,
            kernel_step=1024,
        )
        == 128
    )


def test_ds4_outer_prefill_scans_every_stock_admissible_pending_row():
    short = (8, [[0] * 1000], 1000)
    long = (9, [[0] * 6000], 6000)
    batch = _outer_prefill_batch(
        generation_uids=(99,),
        pending=(short, long),
    )
    assert (
        _deepseek_v4_outer_prefill_step(
            batch,
            long_after=4096,
            kernel_step=1024,
        )
        == 256
    )

    boundary_then_long = _outer_prefill_batch(
        generation_uids=(99,),
        pending=((8, [[1]], 1), long),
    )
    assert (
        _deepseek_v4_outer_prefill_step(
            boundary_then_long,
            long_after=4096,
            kernel_step=1024,
        )
        == 256
    )


@pytest.mark.parametrize(
    ("decode_rows", "expected_quantum"),
    ((1, 256), (2, 256), (4, 128)),
)
def test_ds4_outer_prefill_uses_rank_deterministic_b1_b2_b4_budget(
    decode_rows,
    expected_quantum,
):
    left = _outer_prefill_batch(
        generation_uids=tuple(range(decode_rows)),
        pending=((7, [[0] * 6000], 6000),),
    )
    right = _outer_prefill_batch(
        generation_uids=tuple(range(decode_rows)),
        pending=((7, [[0] * 6000], 6000),),
    )

    decisions = [
        _deepseek_v4_outer_prefill_step(
            batch,
            long_after=4096,
            kernel_step=1024,
        )
        for batch in (left, right)
    ]

    assert decisions == [expected_quantum, expected_quantum]


def test_ds4_runtime_caps_only_outer_turn_and_restores_class(monkeypatch):
    class Attention:
        dspark = True

    class DS4Model:
        def __init__(self):
            self.model = SimpleNamespace(
                layers=[SimpleNamespace(attn=Attention())]
            )

        def __call__(self, value, cache=None, skip_lm_head=False):
            return None

    model = DS4Model()
    batch = _outer_prefill_batch(
        generation_uids=(99,),
        pending=((7, [[0] * 6000], 6000),),
    )
    batch._stream = mx.default_stream(mx.default_device())
    observed = []
    batch._next = lambda: observed.append(
        (batch.prefill_step_size, batch.prefill_batch_size)
    ) or ([], [])
    original_next = mlx_generate.BatchGenerator.next

    with install_runtime_optimizations(
        model,
        _Group(),
        execution_profile("balanced"),
        batchable=True,
        pipeline_parallel=False,
    ) as capabilities:
        assert capabilities["deepseek_v4_prefill_yield"]["active"] is True
        assert mlx_generate.BatchGenerator.next is not original_next
        mlx_generate.BatchGenerator.next(batch)
        assert batch.prefill_step_size == 2048
        assert batch.prefill_batch_size == 4

        def fail_turn():
            assert batch.prefill_step_size == 256
            assert batch.prefill_batch_size == 1
            raise RuntimeError("synthetic scheduler failure")

        batch._next = fail_turn
        with pytest.raises(RuntimeError, match="synthetic scheduler failure"):
            mlx_generate.BatchGenerator.next(batch)
        assert batch.prefill_step_size == 2048
        assert batch.prefill_batch_size == 4

    assert observed == [(256, 1)]
    assert mlx_generate.BatchGenerator.next is original_next

    monkeypatch.setenv("OMLX_DSV4_PREFILL_YIELD", "0")
    with install_runtime_optimizations(
        model,
        _Group(),
        execution_profile("balanced"),
        batchable=True,
        pipeline_parallel=False,
    ) as capabilities:
        item = capabilities["deepseek_v4_prefill_yield"]
        assert item["enabled"] is False
        assert item["active"] is False
        assert "disabled by the operator" in item["reason"]
        assert mlx_generate.BatchGenerator.next is original_next


def test_ds4_runtime_processes_one_existing_prompt_row_and_rotates(monkeypatch):
    class Attention:
        dspark = True

    class DS4Model:
        def __init__(self):
            self.model = SimpleNamespace(
                layers=[SimpleNamespace(attn=Attention())]
            )

        def __call__(self, value, cache=None, skip_lm_head=False):
            return None

    model = DS4Model()
    batch = _outer_prefill_batch(
        generation_uids=(99,),
        current=(
            (1, [[0] * 6000], 6000),
            (2, [[0] * 6000], 6000),
        ),
    )
    batch._prompt_batch = _PromptRows((1, 2), {1: 6000, 2: 6000})
    batch._stream = mx.default_stream(mx.default_device())
    observed = []

    def turn():
        observed.append(
            (
                list(batch._prompt_batch.uids),
                len(batch._currently_processing),
                batch.prefill_step_size,
                batch.prefill_batch_size,
            )
        )
        return ([], [])

    batch._next = turn

    with install_runtime_optimizations(
        model,
        _Group(),
        execution_profile("balanced"),
        batchable=True,
        pipeline_parallel=False,
    ):
        mlx_generate.BatchGenerator.next(batch)
        assert batch._prompt_batch.uids == [2, 1]
        mlx_generate.BatchGenerator.next(batch)

        def fail_turn():
            assert batch._prompt_batch.uids == [1]
            assert batch.prefill_step_size == 256
            assert batch.prefill_batch_size == 1
            raise RuntimeError("synthetic resident-row failure")

        batch._next = fail_turn
        with pytest.raises(RuntimeError, match="synthetic resident-row failure"):
            mlx_generate.BatchGenerator.next(batch)
        assert batch.prefill_step_size == 2048
        assert batch.prefill_batch_size == 4

    assert observed == [
        ([1], 1, 256, 1),
        ([2], 1, 256, 1),
    ]
    assert batch._prompt_batch.uids == [2, 1]
    assert len(batch._currently_processing) == 2
    assert batch._prompt_batch._omlx_total_prompt_lengths == {
        1: 6000,
        2: 6000,
    }


def test_ds4_runtime_promotes_ready_rows_before_bounded_prompt():
    class Attention:
        dspark = True

    class DS4Model:
        def __init__(self):
            self.model = SimpleNamespace(
                layers=[SimpleNamespace(attn=Attention())]
            )

        def __call__(self, value, cache=None, skip_lm_head=False):
            return None

    batch = _outer_prefill_batch(
        current=(
            (1, [[0] * 6000], 6000),
            (2, [[7]], 1),
            (3, [[0] * 6000], 6000),
        ),
    )
    batch._prompt_batch = _PromptRows(
        (1, 2, 3),
        {1: 6000, 2: 1, 3: 6000},
    )
    batch._stream = mx.default_stream(mx.default_device())
    observed = []

    def turn():
        observed.append(
            (
                list(batch._prompt_batch.uids),
                batch.prefill_step_size,
                batch.prefill_batch_size,
            )
        )
        # Mirror MLX-LM's boundary split: UID 2 enters generation before UID 1
        # performs the prompt call.
        batch._prompt_batch._filter([0])
        batch._currently_processing = [batch._currently_processing[0]]
        return ([], [SimpleNamespace(uid=2)])

    batch._next = turn

    with install_runtime_optimizations(
        DS4Model(),
        _Group(),
        execution_profile("balanced"),
        batchable=True,
        pipeline_parallel=False,
    ):
        mlx_generate.BatchGenerator.next(batch)

    assert observed == [([1, 2], 256, 2)]
    assert batch._prompt_batch.uids == [3, 1]
    assert len(batch._currently_processing) == 2


def test_ds4_pinned_next_admits_no_extra_prompt_behind_ready_rows():
    class Attention:
        dspark = True

    class DS4Model:
        def __init__(self):
            self.model = SimpleNamespace(
                layers=[SimpleNamespace(attn=Attention())]
            )

        def __call__(self, value, cache=None, skip_lm_head=False):
            return None

    class GenerationRows:
        def __init__(self):
            self.uids = []

        def __len__(self):
            return len(self.uids)

        def next(self):
            return []

        def extend(self, rows):
            self.uids.extend(rows.uids)

    batch = mlx_generate.BatchGenerator.__new__(mlx_generate.BatchGenerator)
    batch.model = DS4Model()
    batch.prefill_step_size = 2048
    batch.prefill_batch_size = 4
    batch.completion_batch_size = 8
    batch._generation_batch = GenerationRows()
    batch._prompt_batch = _PromptRows(
        (1, 2, 3),
        {1: 1, 2: 1, 3: 6000},
    )
    batch._currently_processing = [
        [[[11]], 0, 1],
        [[[22]], 0, 1],
        [[list(range(6000))], 0, 6000],
    ]
    batch._unprocessed_sequences = [object(), object()]
    batch._gen_tokens_counter = 0
    batch._steps_counter = 0
    batch._prompt_tokens_counter = 0
    batch._prompt_time_counter = 0.0
    batch._stream = mx.default_stream(mx.default_device())
    batch._old_wired_limit = None

    def reject_extra_admission(_count):
        raise AssertionError("ready rows must not admit extra prompt rows")

    batch._make_batch = reject_extra_admission

    with install_runtime_optimizations(
        batch.model,
        _Group(),
        execution_profile("balanced"),
        batchable=True,
        pipeline_parallel=False,
    ):
        prompt_responses, _generation_responses = batch.next()

    assert batch._generation_batch.uids == [1, 2]
    assert batch._prompt_batch.uids == [3]
    assert len(batch._currently_processing) == 1
    assert len(prompt_responses) == 3
    assert len(batch._prompt_batch.prompted) == 1
    assert len(batch._prompt_batch.prompted[0]) == 256


@pytest.mark.parametrize(
    ("queued_total", "expected_width"),
    ((6000, 256), (1000, 2000)),
)
def test_ds4_pinned_next_limits_pending_rows_during_decode(
    queued_total,
    expected_width,
):
    class Attention:
        dspark = True

    class DS4Model:
        def __init__(self):
            self.model = SimpleNamespace(
                layers=[SimpleNamespace(attn=Attention())]
            )

        def __call__(self, value, cache=None, skip_lm_head=False):
            return None

    class GenerationRows:
        def __init__(self):
            self.uids = [99]

        def __len__(self):
            return len(self.uids)

        def next(self):
            return [SimpleNamespace(uid=99)]

        def extend(self, rows):
            self.uids.extend(rows.uids)

    short = (1, [list(range(2000))], 1, None, [], None, None, None)
    queued = (
        2,
        [list(range(queued_total))],
        1,
        None,
        [],
        None,
        None,
        None,
    )
    batch = mlx_generate.BatchGenerator.__new__(mlx_generate.BatchGenerator)
    batch.model = DS4Model()
    batch.prefill_step_size = 2048
    batch.prefill_batch_size = 4
    batch.completion_batch_size = 8
    batch._generation_batch = GenerationRows()
    batch._prompt_batch = _PromptRows((), {})
    batch._currently_processing = []
    batch._unprocessed_sequences = [short, queued]
    batch._gen_tokens_counter = 0
    batch._steps_counter = 0
    batch._prompt_tokens_counter = 0
    batch._prompt_time_counter = 0.0
    batch._stream = mx.default_stream(mx.default_device())
    batch._old_wired_limit = None

    def make_batch(count):
        assert count == 1
        uid, segments, _maximum, _cache, tokens, *_rest = (
            batch._unprocessed_sequences.pop(0)
        )
        total = len(tokens) + sum(len(segment) for segment in segments)
        batch._currently_processing.append([segments, 0, total])
        return _PromptRows((uid,), {uid: total})

    batch._make_batch = make_batch

    with install_runtime_optimizations(
        batch.model,
        _Group(),
        execution_profile("balanced"),
        batchable=True,
        pipeline_parallel=False,
    ):
        prompt_responses, generation_responses = batch.next()

    assert len(generation_responses) == 1
    assert len(prompt_responses) == 1
    assert batch._prompt_batch.uids == [1]
    assert len(batch._prompt_batch.prompted) == 1
    assert len(batch._prompt_batch.prompted[0]) == expected_width
    assert [item[0] for item in batch._unprocessed_sequences] == [2]


def test_ds4_pinned_next_withholds_prompt_when_boundaries_fill_decode_budget():
    class Attention:
        dspark = True

    class DS4Model:
        def __init__(self):
            self.model = SimpleNamespace(
                layers=[SimpleNamespace(attn=Attention())]
            )

        def __call__(self, value, cache=None, skip_lm_head=False):
            return None

    class GenerationRows:
        def __init__(self):
            self.uids = list(range(7))

        def __len__(self):
            return len(self.uids)

        def next(self):
            return [SimpleNamespace(uid=uid) for uid in self.uids]

        def extend(self, rows):
            self.uids.extend(rows.uids)

    long = (11, [list(range(6000))], 1, None, [], None, None, None)
    batch = mlx_generate.BatchGenerator.__new__(mlx_generate.BatchGenerator)
    batch.model = DS4Model()
    batch.prefill_step_size = 2048
    batch.prefill_batch_size = 4
    batch.completion_batch_size = 8
    batch._generation_batch = GenerationRows()
    batch._prompt_batch = _PromptRows((10,), {10: 1})
    batch._currently_processing = [[[[10]], 0, 1]]
    batch._unprocessed_sequences = [long]
    batch._gen_tokens_counter = 0
    batch._steps_counter = 0
    batch._prompt_tokens_counter = 0
    batch._prompt_time_counter = 0.0
    batch._stream = mx.default_stream(mx.default_device())
    batch._old_wired_limit = None

    def reject_prompt_admission(_count):
        raise AssertionError("a full projected decode batch must not admit prompt work")

    batch._make_batch = reject_prompt_admission

    with install_runtime_optimizations(
        batch.model,
        _Group(),
        execution_profile("balanced"),
        batchable=True,
        pipeline_parallel=False,
    ):
        prompt_responses, generation_responses = batch.next()

    assert len(generation_responses) == 7
    assert len(prompt_responses) == 1
    assert batch._generation_batch.uids == [*range(7), 10]
    assert batch._prompt_batch.uids == []
    assert batch._currently_processing == []
    assert [item[0] for item in batch._unprocessed_sequences] == [11]


def test_non_ds4_runtime_does_not_patch_outer_scheduler():
    class OtherModel:
        def __init__(self):
            self.model = SimpleNamespace(layers=[])

        def __call__(self, value, cache=None, skip_lm_head=False):
            return None

    original_next = mlx_generate.BatchGenerator.next
    with install_runtime_optimizations(
        OtherModel(),
        _Group(),
        execution_profile("balanced"),
        batchable=True,
        pipeline_parallel=False,
    ) as capabilities:
        assert capabilities["deepseek_v4_prefill_yield"]["active"] is False
        assert mlx_generate.BatchGenerator.next is original_next

    assert mlx_generate.BatchGenerator.next is original_next


def test_tensor_prefill_reuses_allocator_cache_until_prompt_end(monkeypatch):
    class Cache:
        state = mx.array([0])

    class Model:
        def __call__(self, value, cache=None, skip_lm_head=False):
            assert skip_lm_head is True
            return None

    class Batch:
        uids = ["request"]
        prefill_step_size = 4

        def __init__(self, model):
            self.model = model
            self.tokens = [[]]
            self.prompt_cache = [Cache()]

    clears = []
    monkeypatch.delenv("OMLX_CLUSTER_PREFILL_CLEAR_CACHE_EVERY", raising=False)
    monkeypatch.setattr(mx, "clear_cache", lambda: clears.append(True))
    model = Model()
    batch = Batch(model)

    with install_runtime_optimizations(
        model,
        _Group(),
        execution_profile("balanced"),
        batchable=True,
        pipeline_parallel=False,
    ) as capabilities:
        assert capabilities["prefill_allocator_reuse"]["active"] is True
        mlx_generate.PromptProcessingBatch.prompt(batch, [list(range(9))])

    assert clears == [True]


def test_prefill_allocator_cache_cadence_is_operator_overridable(monkeypatch):
    class Cache:
        state = mx.array([0])

    class Model:
        def __call__(self, value, cache=None, skip_lm_head=False):
            return None

    class Batch:
        uids = ["request"]
        prefill_step_size = 4

        def __init__(self, model):
            self.model = model
            self.tokens = [[]]
            self.prompt_cache = [Cache()]

    clears = []
    monkeypatch.setenv("OMLX_CLUSTER_PREFILL_CLEAR_CACHE_EVERY", "2")
    monkeypatch.setattr(mx, "clear_cache", lambda: clears.append(True))
    model = Model()

    with install_runtime_optimizations(
        model,
        _Group(),
        execution_profile("balanced"),
        batchable=True,
        pipeline_parallel=False,
    ):
        mlx_generate.PromptProcessingBatch.prompt(
            Batch(model),
            [list(range(9))],
        )

    # Three chunks: purge after chunk 2 and unconditionally after the final one.
    assert clears == [True, True]


@pytest.mark.parametrize("batch_size", (1, 4))
def test_tensor_vocab_sampling_reconstructs_logits_only_on_rank_zero(
    monkeypatch, batch_size
):
    class Head:
        _omlx_vocab_parallel = True
        _omlx_output_dims = 6
        _omlx_gather_vocab_logits = True
        weight = mx.zeros((3, 2))

    class Model:
        def __init__(self):
            self.lm_head = Head()

        def __call__(self, value, cache=None):
            assert self.lm_head._omlx_gather_vocab_logits is False
            return mx.broadcast_to(
                mx.array([[[1.0, 2.0, 3.0]]]),
                (batch_size, 1, 3),
            )

    class Batch:
        def __init__(self, model):
            self.model = model
            self.uids = list(range(batch_size))
            self.prompt_cache = []
            self.tokens = [[] for _ in range(batch_size)]
            self.samplers = [None] * batch_size
            self.fallback_sampler = lambda value: mx.argmax(value, axis=-1)
            self.logits_processors = [[] for _ in range(batch_size)]
            self._current_tokens = None
            self._current_logprobs = []
            self._next_tokens = mx.array([3] * batch_size, dtype=mx.uint32)
            self._next_logprobs = []
            self._token_context = [None] * batch_size

    model = Model()
    batch = Batch(model)
    monkeypatch.setattr(
        mx.distributed,
        "recv_like",
        lambda value, source: mx.broadcast_to(
            mx.array([[4.0, 5.0, 9.0]]),
            (batch_size, 3),
        ),
    )
    token_sends = []
    monkeypatch.setattr(
        mx.distributed,
        "send",
        lambda value, destination: token_sends.append(
            (value.tolist(), destination)
        )
        or value,
    )
    monkeypatch.setattr(
        mx.distributed,
        "all_sum",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("rank-zero token decisions must not use all_sum")
        ),
    )
    monkeypatch.setattr(mx, "async_eval", lambda *_values: None)

    with install_runtime_optimizations(
        model,
        _Group(),
        execution_profile("balanced"),
        batchable=True,
        pipeline_parallel=False,
    ) as capabilities:
        assert capabilities["sampling_rank_only"]["active"] is True
        assert capabilities["vocab_parallel_sampling"]["active"] is True
        assert capabilities["rank_zero_logits"]["active"] is False
        mlx_generate.GenerationBatch._step(batch)

    assert batch._next_tokens.tolist() == [5] * batch_size
    assert batch._next_logprobs[0].shape == (6,)
    assert model.lm_head._omlx_gather_vocab_logits is True
    assert token_sends == [([5] * batch_size, 1)]


@pytest.mark.parametrize("batch_size", (1, 4))
def test_tensor_vocab_sampling_worker_sends_local_shard_and_uses_rank_zero_token(
    monkeypatch, batch_size
):
    class Head:
        _omlx_vocab_parallel = True
        _omlx_output_dims = 6
        _omlx_gather_vocab_logits = True
        weight = mx.zeros((3, 2))

    class Model:
        def __init__(self):
            self.lm_head = Head()

        def __call__(self, value, cache=None):
            return mx.broadcast_to(
                mx.array([[[7.0, 8.0, 9.0]]]),
                (batch_size, 1, 3),
            )

    class Batch:
        def __init__(self, model):
            self.model = model
            self.uids = list(range(batch_size))
            self.prompt_cache = []
            self.tokens = [[] for _ in range(batch_size)]
            self.samplers = [None] * batch_size
            self.fallback_sampler = lambda value: mx.argmax(value, axis=-1)
            self.logits_processors = [[] for _ in range(batch_size)]
            self._current_tokens = None
            self._current_logprobs = []
            self._next_tokens = mx.array([3] * batch_size, dtype=mx.uint32)
            self._next_logprobs = []
            self._token_context = [None] * batch_size

    sent = []
    model = Model()
    batch = Batch(model)
    monkeypatch.setattr(
        mx.distributed,
        "send",
        lambda value, destination: sent.append((value.tolist(), destination)) or value,
    )
    monkeypatch.setattr(
        mx.distributed,
        "recv_like",
        lambda value, source: mx.array([5] * batch_size, dtype=mx.int32),
    )
    monkeypatch.setattr(
        mx.distributed,
        "all_sum",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("rank-zero token decisions must not use all_sum")
        ),
    )
    monkeypatch.setattr(mx, "async_eval", lambda *_values: None)

    with install_runtime_optimizations(
        model,
        _WorkerGroup(),
        execution_profile("balanced"),
        batchable=True,
        pipeline_parallel=False,
    ):
        mlx_generate.GenerationBatch._step(batch)

    assert sent == [([[7.0, 8.0, 9.0]] * batch_size, 0)]
    assert batch._next_tokens.tolist() == [5] * batch_size
    assert batch._next_logprobs[0].shape == (6,)


def test_tensor_vocab_sampling_stays_synchronized_for_mtp_model():
    head = SimpleNamespace(
        _omlx_vocab_parallel=True,
        _omlx_output_dims=6,
        weight=mx.zeros((3, 2)),
    )
    model = SimpleNamespace(
        lm_head=head,
        _omlx_mtp_decode_enabled=True,
    )

    with install_runtime_optimizations(
        model,
        _Group(),
        execution_profile("balanced"),
        batchable=True,
        pipeline_parallel=False,
    ) as capabilities:
        assert capabilities["sampling_rank_only"]["active"] is False
        assert capabilities["vocab_parallel_sampling"]["active"] is False
        assert "MTP" in capabilities["vocab_parallel_sampling"]["reason"]


def test_tensor_vocab_sampling_installs_validated_mtp_coordinator():
    head = SimpleNamespace(
        _omlx_vocab_parallel=True,
        _omlx_output_dims=6,
        _omlx_gather_vocab_logits=True,
        weight=mx.zeros((3, 2)),
    )
    auxiliary = SimpleNamespace(
        _omlx_vocab_parallel=True,
        _omlx_output_dims=6,
        _omlx_gather_vocab_logits=True,
        weight=mx.zeros((3, 2)),
    )
    model = SimpleNamespace(
        lm_head=head,
        _omlx_mtp_decode_enabled=True,
        _omlx_mtp_chain=True,
        _omlx_distributed_mtp_vocab_ready=True,
        _omlx_vocab_parallel_aux_heads=(auxiliary,),
    )

    with install_runtime_optimizations(
        model,
        _Group(),
        execution_profile("balanced"),
        batchable=True,
        pipeline_parallel=False,
    ) as capabilities:
        assert capabilities["sampling_rank_only"]["active"] is True
        assert capabilities["vocab_parallel_sampling"]["active"] is True
        assert "MTP" in capabilities["vocab_parallel_sampling"]["reason"]
        coordinator = model._omlx_mtp_vocab_coordinator
        assert coordinator.is_coordinator is True
        assert coordinator.output_size == 6
        assert head._omlx_gather_vocab_logits is False
        assert auxiliary._omlx_gather_vocab_logits is False

    assert not hasattr(model, "_omlx_mtp_vocab_coordinator")
    assert head._omlx_gather_vocab_logits is True
    assert auxiliary._omlx_gather_vocab_logits is True


def test_non_batchable_model_never_reports_continuous_batching_active():
    settings = execution_profile("balanced")
    model = SimpleNamespace(model=SimpleNamespace())

    with install_runtime_optimizations(
        model,
        _Group(),
        settings,
        batchable=False,
    ) as capabilities:
        batching = capabilities["coalesced_batching"]
        assert batching["enabled"] is True
        assert batching["active"] is False
        assert "sequentially" in batching["reason"]
