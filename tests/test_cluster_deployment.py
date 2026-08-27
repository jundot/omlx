# SPDX-License-Identifier: Apache-2.0

import base64
import json
import zlib

import pytest

from omlx.cluster.deployment import (
    ClusterDeployment,
    ClusterHost,
    _assignment_from_dict,
    decode_worker_execution,
    decode_worker_plan,
    decode_worker_serving_mode,
    decode_worker_speculation,
)
from omlx.cluster.performance import NodePerformanceProfile, execution_profile
from omlx.cluster.planner import PipelineAssignment

GIB = 1024**3


def _assignments() -> tuple[PipelineAssignment, ...]:
    return (
        PipelineAssignment(
            node_id="large",
            rank=0,
            start_layer=2,
            end_layer=6,
            layer_weight_bytes=120 * GIB,
            fixed_weight_bytes=2 * GIB,
            reserve_bytes=8 * GIB,
            capacity_bytes=256 * GIB,
        ),
        PipelineAssignment(
            node_id="small",
            rank=1,
            start_layer=0,
            end_layer=2,
            layer_weight_bytes=60 * GIB,
            fixed_weight_bytes=2 * GIB,
            reserve_bytes=8 * GIB,
            capacity_bytes=128 * GIB,
        ),
    )


def _deployment(backend: str = "jaccl") -> ClusterDeployment:
    if backend == "ring":
        hosts = (
            ClusterHost("large", "127.0.0.1", ("192.168.20.1",)),
            ClusterHost("small", "studio.local", ("192.168.20.2",)),
        )
    else:
        hosts = (
            ClusterHost(
                "large",
                "127.0.0.1",
                ("192.168.20.1",),
                (None, "rdma_en5"),
            ),
            ClusterHost(
                "small",
                "studio.local",
                ("192.168.20.2",),
                ("rdma_en5", None),
            ),
        )
    return ClusterDeployment(
        deployment_id="nemotron-ultra",
        model="mlx-community/Nemotron-Ultra-253B-4bit",
        backend=backend,
        hosts=hosts,
        assignments=_assignments(),
        plan_hash="a" * 64,
    )


def test_deployment_round_trip_and_worker_plan_are_json_only():
    deployment = _deployment()

    restored = ClusterDeployment.from_dict(deployment.to_dict())
    plan_hash, assignments = decode_worker_plan(deployment.encode_worker_plan())

    assert restored == deployment
    assert plan_hash == deployment.plan_hash
    assert assignments == deployment.assignments
    assert "MLX_METAL_FAST_SYNCH=1" in deployment.hostfile_dict()["envs"]
    assert deployment.distributed_init_backend == "jaccl"


def test_hostfile_envs_carry_stability_defaults(monkeypatch):
    for name in (
        "MLX_MAX_OPS_PER_BUFFER",
        "MLX_MAX_MB_PER_BUFFER",
        "JACCL_PROGRESS_TIMEOUT_MS",
        "JACCL_TIMEOUT_ACTION",
        "JACCL_TWO_RANK_SMALL_ALLREDUCE",
        "OMLX_CLUSTER_TRACE_COLLECTIVES",
        "OMLX_MTP_DISTRIBUTED_ADAPTIVE_DEPTH",
        "OMLX_MTP_DISTRIBUTED_LOCKSTEP_DEPTH",
        "OMLX_MTP_ROWWISE_BATCH",
        "OMLX_DSV4_MTP_DECODE_CONCURRENCY",
        "OMLX_DSV4_INDEXER_ROW_TP",
        "OMLX_DSV4_INDEXER_GATHER_P2P",
        "OMLX_DSV4_NATIVE_INDEXER",
        "OMLX_DSV4F_MMA_SCORE",
        "OMLX_DSV4_HIERARCHICAL_INDEXER",
        "OMLX_DSV4_HIERARCHICAL_MIN_POOL",
        "OMLX_DSV4_HIERARCHICAL_REFRESH_POOL",
        "OMLX_DSV4_HIERARCHICAL_CANDIDATE_FRACTION",
        "OMLX_DSV4_HIERARCHICAL_NATIVE_UPPER",
        "OMLX_DSV4_INDEXER_ROW_WEIGHTS",
        "OMLX_DSV4_INDEXER_ROW_WEIGHTS_MIN_POOL",
        "OMLX_DSV4_INDEXER_ROW_TP_MIN_POOL",
        "OMLX_DSV4_ADAPTIVE_PREFILL",
        "OMLX_DSV4_ADAPTIVE_PREFILL_AFTER",
        "OMLX_DSV4_ADAPTIVE_PREFILL_STEP",
        "OMLX_DSV4_ADAPTIVE_PREFILL_MAX_BASE",
        "OMLX_CLUSTER_PREFILL_SHAPE_WARMUP",
        "OMLX_DSV4_PREFILL_YIELD",
        "OMLX_CONTENDED_PREFILL_CHUNK",
        "OMLX_DSV4_MIXED_PREFILL_CHUNK",
        "OMLX_MIXED_PREFILL_MIN_QUANTUM",
        "OMLX_DSV4_PREFILL_STEP_TRACE",
        "OMLX_DSV4_PREFILL_ASYNC_DEPTH",
        "OMLX_DSV4_WSDPA",
        "OMLX_DSV4_WSDPA_TP",
        "OMLX_DSV4_WSDPA_TOPK",
        "OMLX_DSV4_B1_SCALAR_OFFSET",
        "OMLX_DSV4_QKV_BUNDLE_PREFILL",
        "OMLX_DSV4_FULL_MOE_DECODE",
        "OMLX_DSV4_FULL_MOE_DECODE_MAX_TOKENS",
        "OMLX_DSV4_ROUTER_TOPK_DECODE",
        "OMLX_DSV4_NAX_OA_PREFILL",
        "OMLX_DSV4_ATTN_FINALIZER_PREFILL",
        "OMLX_DSV4_ATTN_FINALIZER_VERIFY",
        "OMLX_DSV4_OUTPUT_CHAIN_PREFILL",
        "OMLX_DSV4_OUTPUT_CHAIN_EQUAL_TP",
        "OMLX_DSV4_VERIFY_BATCHED_OA_PREPARE",
        "OMLX_DSV4_VERIFY_HC_PRENORM",
        "OMLX_DSV4_HC_RESIDUAL_OVERLAP",
        "OMLX_DSV4_NAX_MOE_BLOCKS",
        "OMLX_DSV4_COMBINED_MOE_PREFILL",
        "OMLX_TP_MOE_SHARD_WEIGHTS",
        "OMLX_TP_NON_MOE_SHARD_WEIGHTS",
        "OMLX_CLUSTER_VOCAB_PARALLEL",
        "OMLX_CLUSTER_VOCAB_PARALLEL_MIN_BYTES",
    ):
        monkeypatch.delenv(name, raising=False)
    deployment = _deployment()

    envs = deployment.hostfile_dict()["envs"]

    # Metal command-buffer caps: an unbounded buffer overruns the GPU
    # driver's ~10 s timeout, which answers with SIGABRT and orphaned wired
    # memory. JACCL knobs pin the wheel's ProgressGuard/teardown-exit posture.
    assert "MLX_MAX_OPS_PER_BUFFER=16" in envs
    assert "MLX_MAX_MB_PER_BUFFER=512" in envs
    assert "JACCL_PROGRESS_TIMEOUT_MS=30000" in envs
    assert "JACCL_TIMEOUT_ACTION=teardown-exit" in envs
    assert "JACCL_TWO_RANK_SMALL_ALLREDUCE=0" in envs
    assert "OMLX_CLUSTER_TRACE_COLLECTIVES=0" in envs
    assert "OMLX_MTP_DISTRIBUTED_ADAPTIVE_DEPTH=0" in envs
    assert "OMLX_MTP_DISTRIBUTED_LOCKSTEP_DEPTH=0" in envs
    assert "OMLX_MTP_ROWWISE_BATCH=0" in envs
    assert "OMLX_DSV4_MTP_DECODE_CONCURRENCY=1" in envs
    assert "OMLX_DSV4_INDEXER_ROW_TP=1" in envs
    assert "OMLX_DSV4_INDEXER_GATHER_P2P=0" in envs
    assert "OMLX_DSV4_NATIVE_INDEXER=1" in envs
    assert "OMLX_DSV4F_MMA_SCORE=1" in envs
    assert "OMLX_DSV4_HIERARCHICAL_INDEXER=0" in envs
    assert "OMLX_DSV4_HIERARCHICAL_MIN_POOL=16000" in envs
    assert "OMLX_DSV4_HIERARCHICAL_REFRESH_POOL=2048" in envs
    assert "OMLX_DSV4_HIERARCHICAL_CANDIDATE_FRACTION=0.30" in envs
    assert "OMLX_DSV4_HIERARCHICAL_NATIVE_UPPER=0" in envs
    assert "OMLX_DSV4_INDEXER_ROW_WEIGHTS=" in envs
    assert "OMLX_DSV4_INDEXER_ROW_WEIGHTS_MIN_POOL=16000" in envs
    assert "OMLX_DSV4_INDEXER_ROW_TP_MIN_POOL=2048" in envs
    assert "OMLX_DSV4_ADAPTIVE_PREFILL=1" in envs
    assert "OMLX_DSV4_ADAPTIVE_PREFILL_AFTER=4096" in envs
    assert "OMLX_DSV4_ADAPTIVE_PREFILL_STEP=1024" in envs
    assert "OMLX_DSV4_ADAPTIVE_PREFILL_MAX_BASE=2048" in envs
    assert "OMLX_CLUSTER_PREFILL_SHAPE_WARMUP=1" in envs
    assert "OMLX_DSV4_PREFILL_YIELD=1" in envs
    assert "OMLX_CONTENDED_PREFILL_CHUNK=512" in envs
    assert "OMLX_DSV4_MIXED_PREFILL_CHUNK=256" in envs
    assert "OMLX_MIXED_PREFILL_MIN_QUANTUM=128" in envs
    assert "OMLX_DSV4_PREFILL_STEP_TRACE=0" in envs
    assert "OMLX_DSV4_PREFILL_ASYNC_DEPTH=0" in envs
    assert "OMLX_DSV4_WSDPA=1" in envs
    assert "OMLX_DSV4_WSDPA_TP=1" in envs
    assert "OMLX_DSV4_WSDPA_TOPK=1" in envs
    assert "OMLX_DSV4_B1_SCALAR_OFFSET=1" in envs
    assert "OMLX_DSV4_QKV_BUNDLE_PREFILL=0" in envs
    assert "OMLX_DSV4_FULL_MOE_DECODE=1" in envs
    assert "OMLX_DSV4_FULL_MOE_DECODE_MAX_TOKENS=1" in envs
    assert "OMLX_DSV4_ROUTER_TOPK_DECODE=1" in envs
    assert "OMLX_DSV4_NAX_OA_PREFILL=0" in envs
    assert "OMLX_DSV4_ATTN_FINALIZER_PREFILL=0" in envs
    assert "OMLX_DSV4_ATTN_FINALIZER_VERIFY=0" in envs
    assert "OMLX_DSV4_OUTPUT_CHAIN_PREFILL=0" in envs
    assert "OMLX_DSV4_OUTPUT_CHAIN_EQUAL_TP=1" in envs
    assert "OMLX_DSV4_VERIFY_BATCHED_OA_PREPARE=0" in envs
    assert "OMLX_DSV4_VERIFY_HC_PRENORM=0" in envs
    assert "OMLX_DSV4_HC_RESIDUAL_OVERLAP=0" in envs
    assert "OMLX_DSV4_NAX_MOE_BLOCKS=0" in envs
    assert "OMLX_DSV4_COMBINED_MOE_PREFILL=0" in envs
    assert "OMLX_TP_MOE_SHARD_WEIGHTS=" in envs
    assert "OMLX_TP_NON_MOE_SHARD_WEIGHTS=" in envs
    assert "OMLX_CLUSTER_VOCAB_PARALLEL=auto" in envs
    assert "OMLX_CLUSTER_VOCAB_PARALLEL_MIN_BYTES=268435456" in envs


def test_hostfile_envs_respect_operator_overrides(monkeypatch):
    monkeypatch.setenv("MLX_MAX_OPS_PER_BUFFER", "32")
    monkeypatch.setenv("JACCL_PROGRESS_TIMEOUT_MS", "60000")
    monkeypatch.setenv("JACCL_TWO_RANK_SMALL_ALLREDUCE", "1")
    monkeypatch.setenv("OMLX_CLUSTER_TRACE_COLLECTIVES", "1")
    monkeypatch.setenv("OMLX_MTP_DISTRIBUTED_ADAPTIVE_DEPTH", "1")
    monkeypatch.setenv("OMLX_MTP_DISTRIBUTED_LOCKSTEP_DEPTH", "1")
    monkeypatch.setenv("OMLX_DSV4_INDEXER_ROW_TP", "0")
    monkeypatch.setenv("OMLX_DSV4_INDEXER_GATHER_P2P", "1")
    monkeypatch.setenv("OMLX_DSV4_NATIVE_INDEXER", "0")
    monkeypatch.setenv("OMLX_DSV4_HIERARCHICAL_NATIVE_UPPER", "1")
    monkeypatch.setenv("OMLX_DSV4_INDEXER_ROW_WEIGHTS", "9,7")
    monkeypatch.setenv("OMLX_DSV4_INDEXER_ROW_WEIGHTS_MIN_POOL", "20000")
    monkeypatch.setenv("OMLX_DSV4_INDEXER_ROW_TP_MIN_POOL", "4096")
    monkeypatch.setenv("OMLX_CLUSTER_PREFILL_SHAPE_WARMUP", "0")
    monkeypatch.setenv("OMLX_DSV4_PREFILL_YIELD", "0")
    monkeypatch.setenv("OMLX_CONTENDED_PREFILL_CHUNK", "640")
    monkeypatch.setenv("OMLX_DSV4_MIXED_PREFILL_CHUNK", "1536")
    monkeypatch.setenv("OMLX_MIXED_PREFILL_MIN_QUANTUM", "64")
    monkeypatch.setenv("OMLX_DSV4_PREFILL_STEP_TRACE", "1")
    monkeypatch.setenv("OMLX_DSV4_PREFILL_ASYNC_DEPTH", "2")
    monkeypatch.setenv("OMLX_DSV4_WSDPA_TP", "0")
    monkeypatch.setenv("OMLX_DSV4_B1_SCALAR_OFFSET", "0")
    monkeypatch.setenv("OMLX_DSV4_QKV_BUNDLE_PREFILL", "1")
    monkeypatch.setenv("OMLX_DSV4_FULL_MOE_DECODE", "1")
    monkeypatch.setenv("OMLX_DSV4_FULL_MOE_DECODE_MAX_TOKENS", "2")
    monkeypatch.setenv("OMLX_DSV4_ROUTER_TOPK_DECODE", "0")
    monkeypatch.setenv("OMLX_DSV4_NAX_OA_PREFILL", "1")
    monkeypatch.setenv("OMLX_DSV4_ATTN_FINALIZER_PREFILL", "1")
    monkeypatch.setenv("OMLX_DSV4_OUTPUT_CHAIN_PREFILL", "1")
    monkeypatch.setenv("OMLX_DSV4_OUTPUT_CHAIN_EQUAL_TP", "0")
    monkeypatch.setenv("OMLX_DSV4_VERIFY_BATCHED_OA_PREPARE", "1")
    monkeypatch.setenv("OMLX_DSV4_VERIFY_HC_PRENORM", "1")
    monkeypatch.setenv("OMLX_DSV4_HC_RESIDUAL_OVERLAP", "1")
    monkeypatch.setenv("OMLX_DSV4_NAX_MOE_BLOCKS", "1")
    monkeypatch.setenv("OMLX_DSV4_COMBINED_MOE_PREFILL", "1")
    monkeypatch.setenv("OMLX_TP_MOE_SHARD_WEIGHTS", "4,4")
    deployment = _deployment()

    envs = deployment.hostfile_dict()["envs"]

    assert "MLX_MAX_OPS_PER_BUFFER=32" in envs
    assert "MLX_MAX_OPS_PER_BUFFER=16" not in envs
    assert "JACCL_PROGRESS_TIMEOUT_MS=60000" in envs
    assert "JACCL_TWO_RANK_SMALL_ALLREDUCE=1" in envs
    assert "JACCL_TWO_RANK_SMALL_ALLREDUCE=0" not in envs
    assert "OMLX_CLUSTER_TRACE_COLLECTIVES=1" in envs
    assert "OMLX_MTP_DISTRIBUTED_ADAPTIVE_DEPTH=1" in envs
    assert "OMLX_MTP_DISTRIBUTED_LOCKSTEP_DEPTH=1" in envs
    assert "OMLX_DSV4_INDEXER_ROW_TP=0" in envs
    assert "OMLX_DSV4_INDEXER_ROW_TP=1" not in envs
    assert "OMLX_DSV4_INDEXER_GATHER_P2P=1" in envs
    assert "OMLX_DSV4_INDEXER_GATHER_P2P=0" not in envs
    assert "OMLX_DSV4_NATIVE_INDEXER=0" in envs
    assert "OMLX_DSV4_HIERARCHICAL_NATIVE_UPPER=1" in envs
    assert "OMLX_DSV4_INDEXER_ROW_WEIGHTS=9,7" in envs
    assert "OMLX_DSV4_INDEXER_ROW_WEIGHTS_MIN_POOL=20000" in envs
    assert "OMLX_DSV4_INDEXER_ROW_TP_MIN_POOL=4096" in envs
    assert "OMLX_DSV4_INDEXER_ROW_TP_MIN_POOL=2048" not in envs
    assert "OMLX_CLUSTER_PREFILL_SHAPE_WARMUP=0" in envs
    assert "OMLX_DSV4_PREFILL_YIELD=0" in envs
    assert "OMLX_CONTENDED_PREFILL_CHUNK=640" in envs
    assert "OMLX_DSV4_MIXED_PREFILL_CHUNK=1536" in envs
    assert "OMLX_MIXED_PREFILL_MIN_QUANTUM=64" in envs
    assert "OMLX_DSV4_PREFILL_STEP_TRACE=1" in envs
    assert "OMLX_DSV4_PREFILL_YIELD=1" not in envs
    assert "OMLX_DSV4_PREFILL_ASYNC_DEPTH=2" in envs
    assert "OMLX_DSV4_PREFILL_ASYNC_DEPTH=0" not in envs
    assert "OMLX_DSV4_WSDPA_TP=0" in envs
    assert "OMLX_DSV4_WSDPA_TP=1" not in envs
    assert "OMLX_DSV4_B1_SCALAR_OFFSET=0" in envs
    assert "OMLX_DSV4_QKV_BUNDLE_PREFILL=1" in envs
    assert "OMLX_DSV4_FULL_MOE_DECODE=1" in envs
    assert "OMLX_DSV4_FULL_MOE_DECODE_MAX_TOKENS=2" in envs
    assert "OMLX_DSV4_ROUTER_TOPK_DECODE=0" in envs
    assert "OMLX_DSV4_NAX_OA_PREFILL=1" in envs
    assert "OMLX_DSV4_ATTN_FINALIZER_PREFILL=1" in envs
    assert "OMLX_DSV4_OUTPUT_CHAIN_PREFILL=1" in envs
    assert "OMLX_DSV4_OUTPUT_CHAIN_EQUAL_TP=0" in envs
    assert "OMLX_DSV4_VERIFY_BATCHED_OA_PREPARE=1" in envs
    assert "OMLX_DSV4_VERIFY_HC_PRENORM=1" in envs
    assert "OMLX_DSV4_HC_RESIDUAL_OVERLAP=1" in envs
    assert "OMLX_DSV4_NAX_MOE_BLOCKS=1" in envs
    assert "OMLX_DSV4_COMBINED_MOE_PREFILL=1" in envs
    assert envs.count("OMLX_TP_MOE_SHARD_WEIGHTS=4,4") == 1
    # Untouched knobs keep their tuned defaults.
    assert "MLX_MAX_MB_PER_BUFFER=512" in envs
    assert "JACCL_TIMEOUT_ACTION=teardown-exit" in envs


def test_deployment_round_trip_preserves_the_selected_context():
    deployment = _deployment()
    deployment = ClusterDeployment(
        deployment_id=deployment.deployment_id,
        model=deployment.model,
        backend=deployment.backend,
        hosts=deployment.hosts,
        assignments=deployment.assignments,
        plan_hash=deployment.plan_hash,
        target_context_tokens=262144,
    )

    restored = ClusterDeployment.from_dict(deployment.to_dict())

    assert restored.target_context_tokens == 262144
    assert restored.to_dict()["target_context_tokens"] == 262144


def test_deployment_carries_mtp_settings_to_every_rank():
    deployment = _deployment()
    deployment = ClusterDeployment(
        deployment_id=deployment.deployment_id,
        model=deployment.model,
        backend=deployment.backend,
        hosts=deployment.hosts,
        assignments=deployment.assignments,
        plan_hash=deployment.plan_hash,
        mtp_enabled=True,
        mtp_num_draft_tokens=5,
    )

    restored = ClusterDeployment.from_dict(deployment.to_dict())

    assert restored.mtp_enabled is True
    assert restored.mtp_num_draft_tokens == 5
    assert decode_worker_speculation(deployment.encode_worker_plan()) == (True, 5)


def _disaggregated_deployment(**overrides):
    base = _deployment()
    assignments = tuple(
        PipelineAssignment(
            node_id=host.node_id,
            rank=rank,
            start_layer=0,
            end_layer=48,
            layer_weight_bytes=14 * GIB,
            fixed_weight_bytes=GIB,
            reserve_bytes=8 * GIB,
            capacity_bytes=(256 if rank == 0 else 128) * GIB,
            tensor_parallel_rank=0,
            tensor_parallel_size=1,
            kv_cache_bytes=2 * GIB,
            kv_bytes_per_token=65536,
            max_context_tokens=1_000_000,
        )
        for rank, host in enumerate(base.hosts)
    )
    fields = dict(
        deployment_id="qwen-phase-split",
        model="mlx-community/Qwen3.8-27B-4bit",
        backend=base.backend,
        hosts=base.hosts,
        assignments=assignments,
        plan_hash="f" * 64,
        tensor_parallel_size=1,
        serving_mode="disaggregated",
        prefill_rank=1,
        decode_rank=0,
    )
    fields.update(overrides)
    return ClusterDeployment(**fields)


def test_disaggregated_deployment_round_trip_and_worker_contract():
    deployment = _disaggregated_deployment()

    restored = ClusterDeployment.from_dict(deployment.to_dict())
    mode = decode_worker_serving_mode(deployment.encode_worker_plan())

    assert restored == deployment
    assert restored.to_dict()["serving_mode"] == "disaggregated"
    assert mode == ("disaggregated", 1, 0)
    assert decode_worker_execution(deployment.encode_worker_plan()) == (
        deployment.execution
    )


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {"prefill_rank": 0, "decode_rank": 0},
            "distinct prefill/decode ranks",
        ),
        ({"tensor_parallel_size": 2}, "tensor-parallel coordinates"),
        ({"mtp_enabled": True}, "does not yet admit speculative"),
    ],
)
def test_disaggregated_deployment_rejects_unsafe_contracts(overrides, message):
    with pytest.raises(ValueError, match=message):
        _disaggregated_deployment(**overrides)


def test_disaggregated_deployment_allows_either_signed_phase_orientation():
    deployment = _disaggregated_deployment(prefill_rank=0, decode_rank=1)

    restored = ClusterDeployment.from_dict(deployment.to_dict())

    assert (restored.prefill_rank, restored.decode_rank) == (0, 1)
    assert decode_worker_serving_mode(deployment.encode_worker_plan()) == (
        "disaggregated",
        0,
        1,
    )


def test_legacy_deployment_decodes_to_sharded_mode():
    payload = _deployment().to_dict()
    payload["schema_version"] = 2
    payload.pop("serving_mode")
    payload.pop("prefill_rank")
    payload.pop("decode_rank")

    restored = ClusterDeployment.from_dict(payload)

    assert restored.serving_mode == "sharded"
    assert decode_worker_serving_mode(restored.encode_worker_plan()) == (
        "sharded",
        None,
        None,
    )


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("mtp_enabled", "false", "mtp_enabled must be a boolean"),
        (
            "mtp_num_draft_tokens",
            "5",
            "mtp_num_draft_tokens must be between 1 and 8",
        ),
    ],
)
def test_deployment_rejects_coerced_mtp_settings(field, value, message):
    payload = _deployment().to_dict()
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        ClusterDeployment.from_dict(payload)


def test_deployment_round_trip_preserves_tensor_parallel_size():
    """Tensor parallel size must survive to_dict/from_dict and worker plan encoding."""
    from omlx.cluster.planner import PipelineAssignment

    assignments = (
        PipelineAssignment(
            node_id="large",
            rank=0,
            start_layer=2,
            end_layer=6,
            layer_weight_bytes=120 * GIB,
            fixed_weight_bytes=2 * GIB,
            reserve_bytes=8 * GIB,
            capacity_bytes=256 * GIB,
            tensor_parallel_rank=0,
            tensor_parallel_size=2,
            tensor_parallel_shard_weight=5,
            sharded_weight_bytes=4 * GIB,
        ),
        PipelineAssignment(
            node_id="small",
            rank=1,
            start_layer=2,
            end_layer=6,
            layer_weight_bytes=120 * GIB,
            fixed_weight_bytes=2 * GIB,
            reserve_bytes=8 * GIB,
            capacity_bytes=256 * GIB,
            tensor_parallel_rank=1,
            tensor_parallel_size=2,
            tensor_parallel_shard_weight=3,
            sharded_weight_bytes=4 * GIB,
        ),
    )
    deployment = ClusterDeployment(
        deployment_id="tp-test",
        model="mlx-community/test",
        backend="jaccl",
        hosts=(
            ClusterHost("large", "127.0.0.1", ("192.168.20.1",), (None, "rdma_en5")),
            ClusterHost("small", "studio.local", ("192.168.20.2",), ("rdma_en5", None)),
        ),
        assignments=assignments,
        plan_hash="a" * 64,
        tensor_parallel_size=2,
    )

    restored = ClusterDeployment.from_dict(deployment.to_dict())
    assert restored == deployment
    assert restored.tensor_parallel_size == 2

    # Worker plan encoding must also carry tensor_parallel_size
    encoded = deployment.encode_worker_plan()
    import base64
    import json
    import zlib

    compressed = base64.b64decode(encoded, altchars=b"-_")
    raw = zlib.decompress(compressed)
    payload = json.loads(raw)
    assert payload["tensor_parallel_size"] == 2
    assert payload["assignments"][0]["tensor_parallel_rank"] == 0
    assert [
        item["tensor_parallel_shard_weight"] for item in payload["assignments"]
    ] == [5, 3]
    assert payload["assignments"][0]["sharded_weight_bytes"] == 4 * GIB


def test_deployment_rejects_non_divisible_tensor_parallel_size():
    """Host count must be divisible by tensor_parallel_size."""
    with pytest.raises(ValueError, match="divisible"):
        ClusterDeployment(
            deployment_id="bad-tp",
            model="model",
            backend="ring",
            hosts=(
                ClusterHost("a", "127.0.0.1", ("10.0.0.1",)),
                ClusterHost("b", "b.local", ("10.0.0.2",)),
                ClusterHost("c", "c.local", ("10.0.0.3",)),
            ),
            assignments=_assignments(),
            plan_hash="c" * 64,
            tensor_parallel_size=2,
        )


def test_deployment_round_trip_preserves_hybrid_rank_map():
    from omlx.cluster.planner import PipelineAssignment

    assignments = tuple(
        PipelineAssignment(
            node_id=f"node-{rank}",
            rank=rank,
            start_layer=20 if rank < 2 else 0,
            end_layer=40 if rank < 2 else 20,
            layer_weight_bytes=20 * GIB,
            fixed_weight_bytes=GIB,
            reserve_bytes=2 * GIB,
            capacity_bytes=64 * GIB,
            tensor_parallel_rank=rank % 2,
            tensor_parallel_size=2,
            sharded_weight_bytes=20 * GIB,
        )
        for rank in range(4)
    )
    deployment = ClusterDeployment(
        deployment_id="hybrid-test",
        model="mlx-community/test",
        backend="ring",
        hosts=tuple(
            ClusterHost(
                f"node-{rank}",
                "127.0.0.1" if rank == 0 else f"node-{rank}.local",
                (f"10.0.0.{rank + 1}",),
            )
            for rank in range(4)
        ),
        assignments=assignments,
        plan_hash="d" * 64,
        tensor_parallel_size=2,
    )

    restored = ClusterDeployment.from_dict(deployment.to_dict())
    assert restored == deployment
    assert restored.world_size == 4
    assert restored.tensor_parallel_size == 2


def test_deployment_round_trip_preserves_execution_and_performance_profiles():
    original = _deployment()
    profiles = tuple(
        NodePerformanceProfile(
            node_id=host.node_id,
            rank=rank,
            decode_weight_bytes_per_second=100 + rank,
            prefill_weight_bytes_per_second=200 + rank,
            collective_latency_seconds=0.001,
            collective_bandwidth_bytes_per_second=10_000,
            backend=original.backend,
            measured_at="2026-07-26T12:00:00+00:00",
            samples=5,
        )
        for rank, host in enumerate(original.hosts)
    )
    deployment = ClusterDeployment(
        deployment_id=original.deployment_id,
        model=original.model,
        backend=original.backend,
        hosts=original.hosts,
        assignments=original.assignments,
        plan_hash=original.plan_hash,
        execution=execution_profile("throughput"),
        performance_profiles=profiles,
    )

    restored = ClusterDeployment.from_dict(deployment.to_dict())

    assert restored == deployment
    assert restored.execution.profile == "throughput"
    assert restored.performance_profiles[1].node_id == original.hosts[1].node_id


@pytest.mark.parametrize(
    "target",
    [
        "-oProxyCommand=bad",
        "studio.local;touch /tmp/pwned",
        "studio.local\nbad",
        "",
    ],
)
def test_ssh_target_rejects_option_and_shell_injection(target):
    with pytest.raises(ValueError, match="invalid SSH target"):
        ClusterHost("node", target, ("192.168.1.2",))


def test_jaccl_requires_complete_matrix_with_null_diagonal():
    with pytest.raises(ValueError, match="full RDMA connectivity matrix"):
        ClusterDeployment(
            deployment_id="test",
            model="model",
            backend="jaccl",
            hosts=(
                ClusterHost("large", "127.0.0.1", ("192.168.1.1",)),
                ClusterHost("small", "small.local", ("192.168.1.2",)),
            ),
            assignments=_assignments(),
            plan_hash="b" * 64,
        )


def test_rank_zero_must_be_local_launcher_process():
    deployment = _deployment("ring")
    with pytest.raises(ValueError, match="rank 0"):
        ClusterDeployment(
            deployment_id=deployment.deployment_id,
            model=deployment.model,
            backend=deployment.backend,
            hosts=(
                ClusterHost("large", "large.local", ("192.168.20.1",)),
                deployment.hosts[1],
            ),
            assignments=deployment.assignments,
            plan_hash=deployment.plan_hash,
        )


def test_decode_worker_plan_rejects_trailing_compressed_payload():
    deployment = _deployment()
    encoded = deployment.encode_worker_plan()
    compressed = base64.urlsafe_b64decode(encoded)
    malformed = base64.urlsafe_b64encode(compressed + zlib.compress(b"{}")).decode()

    with pytest.raises(ValueError, match="malformed"):
        decode_worker_plan(malformed)


def test_decode_worker_plan_rejects_unbounded_decompressed_payload():
    raw = json.dumps(
        {
            "schema_version": 1,
            "plan_hash": "a" * 64,
            "assignments": [],
            "padding": "x" * (300 * 1024),
        }
    ).encode()
    encoded = base64.urlsafe_b64encode(zlib.compress(raw)).decode()

    with pytest.raises(ValueError, match="too large"):
        decode_worker_plan(encoded)


# --- What the rank reads back has to be what the planner wrote --------------
#
# ``_assignment_from_dict`` is the only reader of an assignment on the far side
# of both seams that matter: the registry file the admin server reloads, and
# the ``--plan`` argument the rank decodes. A field ``to_dict`` emits and this
# decoder ignores is a value that silently becomes zero on the machine that
# acts on it, with every round-trip test still green — which is exactly what
# happened to the KV cache below.


def _planned_assignment(**overrides) -> PipelineAssignment:
    """An assignment shaped like one the planner really produces."""

    fields = dict(
        node_id="macbook",
        rank=0,
        start_layer=2,
        end_layer=6,
        layer_weight_bytes=40 * GIB,
        fixed_weight_bytes=2 * GIB,
        reserve_bytes=32 * GIB,
        capacity_bytes=107 * GIB,
        role="workstation",
        kv_cache_bytes=20 * GIB,
        kv_bytes_per_token=2_500_000,
        max_context_tokens=13_000,
    )
    fields.update(overrides)
    return PipelineAssignment(**fields)


def test_every_field_the_planner_writes_survives_the_decoder():
    original = _planned_assignment()

    restored = _assignment_from_dict(original.to_dict())

    assert restored == original
    # The number the rank's memory guard is charged, and the engine pool
    # reserves against. It was arriving 20 GiB light because the KV cache was
    # emitted and never read back.
    assert restored.planned_weight_bytes == original.planned_weight_bytes
    assert restored.kv_cache_bytes == 20 * GIB
    assert restored.max_context_tokens == 13_000


def test_the_role_survives_the_worker_plan_and_the_registry_file():
    assignments = (
        _planned_assignment(role="workstation"),
        _planned_assignment(
            node_id="studio",
            rank=1,
            start_layer=0,
            end_layer=2,
            capacity_bytes=256 * GIB,
            reserve_bytes=25 * GIB,
            role="headless",
        ),
    )
    deployment = ClusterDeployment(
        deployment_id="roles",
        model="org/model",
        backend="ring",
        hosts=(
            ClusterHost("macbook", "127.0.0.1", ("10.0.0.1",)),
            ClusterHost("studio", "studio.local", ("10.0.0.2",)),
        ),
        assignments=assignments,
        plan_hash="d" * 64,
    )

    # The registry writes and reloads this.
    restored = ClusterDeployment.from_dict(json.loads(json.dumps(deployment.to_dict())))
    # The rank decodes this.
    _hash, decoded = decode_worker_plan(deployment.encode_worker_plan())

    assert [item.role for item in restored.assignments] == [
        "workstation",
        "headless",
    ]
    assert [item.role for item in decoded] == ["workstation", "headless"]


def test_the_memory_tier_survives_the_worker_plan_and_legacy_defaults_safely():
    original = _planned_assignment(memory_guard_tier="safe")
    peer = _planned_assignment(
        node_id="studio",
        rank=1,
        start_layer=0,
        end_layer=2,
        capacity_bytes=256 * GIB,
        reserve_bytes=25 * GIB,
        role="headless",
        memory_guard_tier="aggressive",
    )
    deployment = ClusterDeployment(
        deployment_id="memory-tier",
        model="org/model",
        backend="ring",
        hosts=(
            ClusterHost("macbook", "127.0.0.1", ("10.0.0.1",)),
            ClusterHost("studio", "studio.local", ("10.0.0.2",)),
        ),
        assignments=(original, peer),
        plan_hash="e" * 64,
    )

    restored = ClusterDeployment.from_dict(deployment.to_dict())
    _hash, decoded = decode_worker_plan(deployment.encode_worker_plan())
    assert [item.memory_guard_tier for item in restored.assignments] == [
        "safe",
        "aggressive",
    ]
    assert [item.memory_guard_tier for item in decoded] == ["safe", "aggressive"]

    legacy = original.to_dict()
    legacy.pop("memory_guard_tier")
    assert _assignment_from_dict(legacy).memory_guard_tier == "balanced"

    legacy["memory_guard_tier"] = "extreme"
    with pytest.raises(ValueError, match="unknown memory guard tier"):
        _assignment_from_dict(legacy)


def test_a_plan_with_no_role_decodes_unchanged():
    payload = _planned_assignment().to_dict()
    payload.pop("role")

    assert _assignment_from_dict(payload).role == ""


def test_a_plan_carrying_an_unknown_role_refuses_to_launch():
    """Fail the launch, not the person at the keyboard.

    A role nobody recognises means the chain that produced it is broken; the
    lenient reading is "headless", which is the fraction that fills the Mac.
    """

    payload = _planned_assignment().to_dict()
    payload["role"] = "workststion"

    with pytest.raises(ValueError, match="unknown node role"):
        _assignment_from_dict(payload)


def test_link_local_zone_ids_are_stripped_from_communication_ips():
    """macOS announces Thunderbolt addresses as fe80::…%en10; mlx's address
    parser cannot read the zone and the rank died at startup on it."""
    host = ClusterHost(
        "node", "peer.local", ("10.0.0.2", "fe80::23:3ee:ec30:9a92%en10")
    )
    assert host.ips == ("10.0.0.2", "fe80::23:3ee:ec30:9a92")


def test_hostfile_advertises_routable_addresses_before_link_local():
    from omlx.cluster.deployment import _hostfile_ips

    host = ClusterHost(
        "node",
        "peer.local",
        ("fe80::1%en4", "10.0.0.2", "fe80::2%en5"),
    )
    assert _hostfile_ips(host) == ["10.0.0.2", "fe80::1", "fe80::2"]


def test_a_link_local_only_host_keeps_its_zone_free_fallback():
    from omlx.cluster.deployment import _hostfile_ips

    host = ClusterHost("node", "peer.local", ("fe80::1%en4",))
    assert _hostfile_ips(host) == ["fe80::1"]
