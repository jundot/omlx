# SPDX-License-Identifier: Apache-2.0
"""Validated, serializable configuration for one distributed inference job."""

from __future__ import annotations

import base64
import binascii
import ipaddress
import json
import math
import os
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .performance import (
    ExecutionSettings,
    NodePerformanceProfile,
    execution_profile,
)
from .planner import (
    PipelineAssignment,
    normalize_memory_guard_tier,
    normalize_node_role,
)
from .tp_qualifications import TPQualificationProvenance

DistributedBackend = Literal["ring", "jaccl", "jaccl-ring"]
ServingMode = Literal["sharded", "disaggregated"]

# Version 3 adds the signed serving mode and prefill/decode rank ownership.
# Version 2 adds ``path_map``: an optional per-node absolute model path, so
# nodes no longer need the model at the same absolute path on every Mac.
# Version 1 payloads decode with an empty map, which reproduces the legacy
# shared-path behavior exactly. Older schemas decode to ordinary sharded mode.
DEPLOYMENT_SCHEMA_VERSION = 3
_SUPPORTED_DEPLOYMENT_SCHEMAS = (1, 2, DEPLOYMENT_SCHEMA_VERSION)
_MAX_PATH_MAP_ENTRIES = 64
_MAX_MODEL_PATH_BYTES = 4096

_SSH_TARGET = re.compile(
    r"^(?:[A-Za-z0-9._-]+@)?(?:[A-Za-z0-9._-]+|\[[0-9A-Fa-f:]+\])$"
)
_RDMA_DEVICE = re.compile(r"^rdma_[A-Za-z0-9_.-]+$")
_NODE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MAX_MODEL_ID_BYTES = 16 * 1024
_MAX_PLAN_BYTES = 256 * 1024

# Rank-environment defaults carried by the mlx launch hostfile. Each value
# yields to the coordinator's own environment: an operator who pinned a knob
# keeps their value, and everyone else gets the tuned default.
#
# MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER bound how much work lands in
# one Metal command buffer. An unbounded buffer overruns the GPU driver's
# ~10 s execution timeout (kIOGPUCommandBufferCallbackErrorTimeout), which the
# driver answers with SIGABRT and stranded wired memory — the crash class
# ThunderMLX root-caused to missing caps.
#
# JACCL_PROGRESS_TIMEOUT_MS / JACCL_TIMEOUT_ACTION pair with the ProgressGuard
# and emergency-teardown symbols compiled into the serving wheel's libjaccl
# (``progress_timeout_ms()`` and the ``teardown-exit`` action calling
# ``emergency_teardown()``): an RDMA polling loop that sees no completion for
# 30 s unwinds and exits instead of spinning forever with memory wired. The
# wheel ships compiled defaults; pinning them here keeps the posture explicit
# and identical across every rank.
_RANK_ENV_DEFAULTS = (
    ("MLX_METAL_FAST_SYNCH", "1"),
    ("MLX_MAX_OPS_PER_BUFFER", "16"),
    ("MLX_MAX_MB_PER_BUFFER", "512"),
    # One JACCL communicator remains strictly ordered, but batching eight
    # adjacent cache tensors per eval removes 112 Python/Metal submission
    # fences from a typical 128-array phase handoff.
    ("OMLX_CLUSTER_CACHE_TRANSFER_WINDOW", "8"),
    ("JACCL_PROGRESS_TIMEOUT_MS", "30000"),
    ("JACCL_TIMEOUT_ACTION", "teardown-exit"),
    # Live two-Mac RDMA A/B showed the experimental one-buffer two-rank path
    # was neutral below 8 KiB and 3-6% slower at 40-512 KiB. Keep the switch
    # visible for kernel iteration, but ship the measured generic path.
    ("JACCL_TWO_RANK_SMALL_ALLREDUCE", "0"),
    # Rank-local JSONL tracing identifies collective-order divergence without
    # changing JACCL execution. It is intentionally operator-only because one
    # line per tensor operation is too expensive for normal inference.
    ("OMLX_CLUSTER_TRACE_COLLECTIVES", "0"),
    # Prompt-cache structure tracing is operator-only. It must reach every
    # rank because cache reuse is a synchronized prefill decision.
    ("OMLX_CLUSTER_CACHE_TRACE", "0"),
    # Keep distributed prompt snapshots under the ordinary oMLX SSD cache
    # root. The server replaces this default from GlobalSettings before a
    # launch, while each remote rank expands ``~`` for its own account.
    ("OMLX_CLUSTER_PROMPT_CACHE_ROOT", "~/.omlx/cache/cluster-prompt-snapshots"),
    # Qualification gate for rank-zero-coordinated adaptive MTP depth. The
    # coordinator broadcasts the selected verify width and park decision, so
    # every TP rank enters the same collective graph. Production stays on its
    # signed fixed depth until physical parity/throughput qualification.
    ("OMLX_MTP_DISTRIBUTED_ADAPTIVE_DEPTH", "0"),
    # Deterministic accepted-prefix depth controller. Unlike the measured
    # controller it uses no rank-local clock and needs no depth broadcast.
    ("OMLX_MTP_DISTRIBUTED_LOCKSTEP_DEPTH", "0"),
    # Experimental multi-sequence MTP runs independent exact singleton cycles
    # per row. Keep it rank-identical and default-off; physical B2/B4 economics
    # decide whether a serving profile should enable it.
    ("OMLX_MTP_ROWWISE_BATCH", "0"),
    # DS4's current DSpark MTP implementation is singleton-only. Until true
    # N×M verification lands, one decode lane has higher aggregate throughput
    # than forcing concurrent rows onto the slower standard batch path.
    ("OMLX_DSV4_MTP_DECODE_CONCURRENCY", "1"),
    # DS4's sparse prefill indexer is row-independent. TP ranks split prompt
    # rows and exchange only top-k indices instead of redundantly scoring the
    # full chunk on every GPU. Explicit env keeps live rollback one flag away.
    ("OMLX_DSV4_INDEXER_ROW_TP", "1"),
    # Candidate weighted-TP2 transport exchanges exact, unpadded index rows
    # with ordered point-to-point operations. It requires both this flag and
    # OMLX_DSV4_WEIGHTED_INDEXER_ROWS; equal/non-TP2 rows always retain the
    # faster general collective. Keep both off pending a whole-model gate.
    ("OMLX_DSV4_INDEXER_GATHER_P2P", "0"),
    ("OMLX_DSV4_NATIVE_INDEXER", "1"),
    # Bit-exact DS4F ratio-4 MMA score path, physically qualified on M2 Ultra,
    # M3 Ultra, and M5 Max. Carry one rollback value to every TP rank.
    ("OMLX_DSV4F_MMA_SCORE", "1"),
    # Certified low-rank screen + exact candidate rescore. Default-off until
    # its full TP2 100K/250K rate and cache-lifecycle gates clear.
    ("OMLX_DSV4_HIERARCHICAL_INDEXER", "0"),
    ("OMLX_DSV4_HIERARCHICAL_MIN_POOL", "16000"),
    ("OMLX_DSV4_HIERARCHICAL_REFRESH_POOL", "2048"),
    ("OMLX_DSV4_HIERARCHICAL_CANDIDATE_FRACTION", "0.30"),
    # Native exact-bound postprocess for the opt-in hierarchy. It fuses the
    # BF16 approximate sheet's FP32 error bound and 16-row reduction, but stays
    # separately rollbackable until physical real-key/full-model gates pass.
    ("OMLX_DSV4_HIERARCHICAL_NATIVE_UPPER", "0"),
    # Weighted row shards require padding every rank to the largest shard for
    # all_gather. The first 3:5/30K live gate was slower, so retain the exact
    # implementation only as an operator A/B and ship equal row counts.
    ("OMLX_DSV4_WEIGHTED_INDEXER_ROWS", "0"),
    # Optional long-context query-row balance independent of tensor weight
    # ownership. Empty preserves equal rows; the pool threshold prevents a
    # heterogeneous qualification from perturbing short prompts.
    ("OMLX_DSV4_INDEXER_ROW_WEIGHTS", ""),
    ("OMLX_DSV4_INDEXER_ROW_WEIGHTS_MIN_POOL", "16000"),
    # Below 2K pooled entries the fixed top-k exchange can cost more than the
    # saved score work; the threshold turns the split on where context taper
    # begins instead of perturbing short-prompt performance.
    ("OMLX_DSV4_INDEXER_ROW_TP_MIN_POOL", "2048"),
    # DS4's 2K prefill chunk wins at medium context, while 1K avoids the 14K
    # taper. The worker switches only after the first 4K tokens.
    ("OMLX_DSV4_ADAPTIVE_PREFILL", "1"),
    ("OMLX_DSV4_ADAPTIVE_PREFILL_AFTER", "4096"),
    ("OMLX_DSV4_ADAPTIVE_PREFILL_STEP", "1024"),
    ("OMLX_DSV4_ADAPTIVE_PREFILL_MAX_BASE", "2048"),
    # Experimental 2K outer tile with canonical 1K FP32 HC and compressed-
    # attention/cache boundaries. Both ranks must enter the same split graph.
    ("OMLX_DSV4_CANONICAL_WIDE_PREFILL", "0"),
    # Exact FP32 decode HC producer compile. Shape/config gates retain the
    # ordinary path for training, prefill and non-DS4 checkpoints.
    ("OMLX_DSV4_COMPILED_HC_DECODE_PRODUCER", "0"),
    # Shape warmup enters the full distributed graph, so every rank must make
    # the same decision.  Coordinator-only overrides would cross collectives.
    ("OMLX_CLUSTER_PREFILL_SHAPE_WARMUP", "1"),
    # Experimental DS4 replicated-projection ownership. The physical 3:5 gate
    # nominates rank 0/M3, but production remains off until the exact live
    # prefill+decode promotion gate clears.
    ("OMLX_DSV4_PROJECTION_OWNER_RANK", "off"),
    # Exact one-dispatch DS4 B1 ratio-4 Q/KV/compressor bundle. Physical TP2
    # gate: +9.6% short decode with unchanged 14K prefill and exact hashes.
    ("OMLX_DSV4_QKV_BUNDLE_DECODE", "1"),
    # Lossless M=1024 continuation over original packed MXFP8/BF16 storage.
    # M3 uses three dispatches; M5 keeps its reduction-sensitive main BF16
    # banks separate and uses four. Carry one rollback value to every rank;
    # remain default-off until the full cold-prefill TP2 gate is recorded.
    ("OMLX_DSV4_QKV_BUNDLE_PREFILL", "0"),
    # Ratio-0/128 siblings are exact and locally faster on M5, but did not
    # improve whole-model TP2 decode after ratio-4 moved the critical path.
    ("OMLX_DSV4_QKV_BUNDLE_ALL_SCHEDULES", "0"),
    # Exact 256-way B1 router top-6. Default-off until the full TP2 rate/hash
    # gate confirms the isolated 2.6% selection win scales across 40 layers.
    ("OMLX_DSV4_ROUTER_TOPK_DECODE", "1"),
    # Yield long prompt work back to live decode after one bounded DS4 slice.
    # The hostfile value keeps both TP ranks on the same scheduler decision.
    ("OMLX_DSV4_PREFILL_YIELD", "1"),
    # Generic schedulers retain their 512/256/128 B1/B2/B4 policy. DS4's
    # physical TP path has its own live-qualified 1024/1024/512 schedule.
    ("OMLX_CONTENDED_PREFILL_CHUNK", "512"),
    ("OMLX_DSV4_MIXED_PREFILL_CHUNK", "256"),
    ("OMLX_MIXED_PREFILL_MIN_QUANTUM", "128"),
    ("OMLX_DSV4_PREFILL_STEP_TRACE", "0"),
    # Reversible depth-two graph overlap for pure TP2 prefill.  It remains off
    # until a live lossless A/B clears the promotion gate; carrying the value
    # in the hostfile guarantees that both ranks make the same queue decision.
    ("OMLX_DSV4_PREFILL_ASYNC_DEPTH", "0"),
    # The lossless windowed+sparse prefill kernel is head-count agnostic at
    # the Metal level.  Carry its controls in the hostfile so every TP rank
    # makes the same dispatch decision; coordinator-only environment values
    # are otherwise not inherited by remote ranks launched over SSH.
    ("OMLX_DSV4_WSDPA", "1"),
    ("OMLX_DSV4_WSDPA_TP", "1"),
    ("OMLX_DSV4_WSDPA_TOPK", "1"),
    # BatchRotatingKVCache's B=1 host offset unlocks the exact scalar WSDPA
    # ABI. Matched 30K gate: +52.0%, identical completion hash.
    ("OMLX_DSV4_B1_SCALAR_OFFSET", "1"),
    # Decode has one sparse-indexer row, so row TP cannot divide it. The
    # measured fastest rank computes the exact top-k once and broadcasts only
    # 512 int32 indices; ``off`` restores replicated indexer work.
    ("OMLX_DSV4_INDEXER_DECODE_OWNER_RANK", "auto"),
    # Full routed-MoE decode fusion is bit-exact and isolated-kernel faster
    # through B=4 on both 3:5 slices. The live TP2 gate retained only +1.1% at
    # B=2 and regressed aggregate B=4 decode by 3.5%, so production remains B=1.
    # Operators may still use the existing explicit maximum for future A/Bs.
    ("OMLX_DSV4_FULL_MOE_DECODE", "1"),
    ("OMLX_DSV4_FULL_MOE_DECODE_MAX_TOKENS", "1"),
    # Exact M=1024 M3-family MXFP4 route-tail kernels. Keep both ranks on the
    # same explicit A/B value; default remains off pending the TP2 model gate.
    ("OMLX_DSV4_MOE_TAIL8", "0"),
    # Exact equal-TP2 M3 route-tail kernels. M5 keeps NAX; unequal/single
    # topologies retain the explicit legacy gate above.
    ("OMLX_DSV4_MOE_TAIL8_EQUAL_TP", "1"),
    # Combined asymmetric 3:5 routed-MoE prefill: M3 rank 0 tail8 plus
    # separate M5 rank 1 expert-blocked NAX projections. One rollback value
    # must reach every rank; exact local device/shape/rank gates decide use.
    ("OMLX_DSV4_COMBINED_MOE_PREFILL", "0"),
    # Exact M=1024 M5 TensorOps O-A projection for the 40-head 5/8 shard.
    # Default-off until the full TP A/B clears; exporting the value prevents
    # coordinator/worker capability decisions from diverging.
    ("OMLX_DSV4_NAX_OA_PREFILL", "0"),
    # Exact M=1024 BF16 Q/KV RMSNorm+RoPE finalizers for the supported
    # H24/H32/H40 TP shapes. Keep pair selection identical across ranks while
    # the full distributed A/B remains operator-controlled.
    ("OMLX_DSV4_ATTN_FINALIZER_PREFILL", "0"),
    ("OMLX_DSV4_ATTN_FINALIZER_VERIFY", "0"),
    # Exact two-dispatch O-A→BF16→O-B prefill chain for signed 3:5 TP shapes.
    # Both ranks receive one value; exact shape/config guards decide locally.
    ("OMLX_DSV4_OUTPUT_CHAIN_PREFILL", "0"),
    # Exact equal-TP2 M3 O-A->BF16->O-B chain. M5 retains NAX stock and every
    # unequal/single topology retains its separately default-off legacy gate.
    ("OMLX_DSV4_OUTPUT_CHAIN_EQUAL_TP", "1"),
    # Exact DSpark verify graph simplification: prepare all M O-A rows as one
    # grouped view instead of materializing M one-row slices + concatenate.
    # Default-off until the physical equal-TP2 decode gate clears.
    ("OMLX_DSV4_VERIFY_BATCHED_OA_PREPARE", "0"),
    # Exact M=6 HC sinkhorn/collapse continuation into MLX's weighted
    # 4096-wide RMSNorm topology. Default-off pending physical TP2 decode.
    ("OMLX_DSV4_VERIFY_HC_PRENORM", "0"),
    # Exact M=1024 HC sinkhorn/collapse continuation into the following
    # weighted RMSNorm. Default-off pending physical cross-chip prefill A/B.
    ("OMLX_DSV4_PREFILL_HC_PRENORM", "0"),
    # Exact M=1 HC sinkhorn/collapse continuation into weighted RMSNorm.
    # Default-off pending physical non-MTP decode qualification.
    ("OMLX_DSV4_DECODE_HC_PRENORM", "0"),
    # Split the exact FP32 HyperConnection residual branch so Metal can fill
    # communication bubbles without changing collective or arithmetic order.
    ("OMLX_DSV4_HC_RESIDUAL_OVERLAP", "0"),
    # Exact M5 Max rank-1 5/8 expert-blocked NAX routed-MoE path. The flag is
    # carried to both ranks, while the exact hardware/TP gate activates only
    # on the qualified M5 rank. Default-off until full cold-prefill A/B.
    ("OMLX_DSV4_NAX_MOE_BLOCKS", "0"),
    # Structure-first local safetensors slicing. Default-off until a complete
    # 60--100 GiB load clears strict sanitizer/parameter coverage, memory and
    # first-token parity on every rank. Carry the rollback bit symmetrically.
    ("OMLX_DSV4_SHARD_NATIVE_LOAD", "0"),
    # Optional routed-MoE-only TP split layered over a signed unequal outer
    # plan. Empty preserves the outer split. One hostfile value is exported to
    # every rank so a mixed plan cannot diverge locally.
    ("OMLX_TP_MOE_SHARD_WEIGHTS", ""),
    # Conservative mixed-plan form: keep the signed outer plan equal for
    # admission, then opt only non-routed tensors into a qualified unequal
    # split. Empty preserves the signed outer assignment.
    ("OMLX_TP_NON_MOE_SHARD_WEIGHTS", ""),
    # Large standalone output projections are exact row shards. The default
    # threshold avoids adding a collective to small models where it cannot
    # repay its latency, while keeping every rank on the same decision.
    ("OMLX_CLUSTER_VOCAB_PARALLEL", "auto"),
    ("OMLX_CLUSTER_VOCAB_PARALLEL_MIN_BYTES", str(256 * 1024**2)),
)


def _hostfile_envs() -> list[str]:
    return [
        f"{name}={os.environ.get(name) or default}"
        for name, default in _RANK_ENV_DEFAULTS
    ]


RDMAPath = str | tuple[str, ...] | None


def validate_ssh_target(value: str) -> str:
    """Validate an SSH destination without accepting options or shell syntax."""

    value = value.strip()
    if (
        not value
        or len(value) > 255
        or value.startswith("-")
        or _SSH_TARGET.fullmatch(value) is None
    ):
        raise ValueError(f"invalid SSH target: {value!r}")
    return value


def _validate_ip(value: str) -> str:
    value = value.strip()
    # macOS reports Thunderbolt/link-local addresses with a zone id
    # (fe80::…%en10). mlx's address parser cannot read the suffix — a rank
    # died at startup on exactly that — and a zone only means something on
    # the machine that named it, so it never belongs on the wire.
    value = value.split("%", 1)[0]
    try:
        ipaddress.ip_address(value)
    except ValueError as exc:
        raise ValueError(f"invalid communication IP: {value!r}") from exc
    return value


def _hostfile_ips(host: ClusterHost) -> list[str]:
    """Communication IPs in the order a rank should try them.

    Routable addresses first: a link-local IPv6 without its (machine-local)
    zone id is ambiguous, so it can only ever be a fallback. This is what
    keeps a Mac that announces fe80:: Thunderbolt addresses alongside its
    configured link IPs from advertising the unusable one first.
    """

    def _link_local(ip: str) -> bool:
        return ipaddress.ip_address(ip).is_link_local

    routable = [ip for ip in host.ips if not _link_local(ip)]
    return routable + [ip for ip in host.ips if _link_local(ip)]


def _validate_rdma_path(value: Any) -> RDMAPath:
    if value is None:
        return None
    if isinstance(value, str):
        if _RDMA_DEVICE.fullmatch(value) is None:
            raise ValueError(f"invalid RDMA device: {value!r}")
        return value
    if isinstance(value, (list, tuple)):
        paths = tuple(value)
        if not paths:
            raise ValueError("an RDMA path list cannot be empty")
        for path in paths:
            if not isinstance(path, str) or _RDMA_DEVICE.fullmatch(path) is None:
                raise ValueError(f"invalid RDMA device: {path!r}")
        return paths
    raise ValueError("RDMA entries must be a device, a device list, or null")


def validate_model_path_map(
    path_map: Any,
    node_ids: tuple[str, ...] | None = None,
) -> dict[str, str]:
    """Validate an optional per-node model path override map.

    Keys are cluster node IDs; values are absolute paths that exist only on
    the node that resolves them, so no existence check happens here. An empty
    map is the legacy "same absolute path on every node" behavior.
    """

    if path_map is None:
        return {}
    if not isinstance(path_map, dict):
        raise ValueError("path_map must be an object mapping node IDs to paths")
    if len(path_map) > _MAX_PATH_MAP_ENTRIES:
        raise ValueError("path_map cannot hold more than 64 nodes")
    known = set(node_ids) if node_ids is not None else None
    validated: dict[str, str] = {}
    for node_id, raw_path in path_map.items():
        if not isinstance(node_id, str) or _NODE_ID.fullmatch(node_id) is None:
            raise ValueError(f"invalid path_map node ID: {node_id!r}")
        if known is not None and node_id not in known:
            raise ValueError(
                f"path_map names a node outside the deployment: {node_id!r}"
            )
        if not isinstance(raw_path, str):
            raise ValueError(f"path_map path for {node_id!r} must be a string")
        path = raw_path.strip()
        if (
            not path
            or "\x00" in path
            or len(path.encode()) > _MAX_MODEL_PATH_BYTES
            or not Path(path).is_absolute()
        ):
            raise ValueError(f"path_map path for {node_id!r} must be absolute")
        validated[node_id] = path
    return validated


@dataclass(frozen=True)
class ClusterHost:
    """One rank in an MLX hostfile."""

    node_id: str
    ssh: str
    ips: tuple[str, ...]
    rdma: tuple[RDMAPath, ...] = ()
    python_executable: str | None = None

    def __post_init__(self) -> None:
        if _NODE_ID.fullmatch(self.node_id) is None:
            raise ValueError(f"invalid node ID: {self.node_id!r}")
        object.__setattr__(self, "ssh", validate_ssh_target(self.ssh))
        object.__setattr__(self, "ips", tuple(_validate_ip(ip) for ip in self.ips))
        object.__setattr__(
            self,
            "rdma",
            tuple(_validate_rdma_path(path) for path in self.rdma),
        )
        if self.python_executable is not None:
            executable = self.python_executable.strip()
            path = Path(executable)
            if not path.is_absolute() or "\x00" in executable or len(executable) > 4096:
                raise ValueError("cluster host Python executable must be absolute")
            object.__setattr__(self, "python_executable", executable)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "node_id": self.node_id,
            "ssh": self.ssh,
            "ips": list(self.ips),
            "rdma": [
                list(path) if isinstance(path, tuple) else path for path in self.rdma
            ],
        }
        if self.python_executable:
            payload["python_executable"] = self.python_executable
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ClusterHost:
        if not isinstance(payload, dict):
            raise ValueError("cluster host must be an object")
        node_id = payload.get("node_id")
        ssh = payload.get("ssh")
        ips = payload.get("ips", [])
        rdma = payload.get("rdma", [])
        if not isinstance(node_id, str) or not isinstance(ssh, str):
            raise ValueError("cluster host requires string node_id and ssh fields")
        if not isinstance(ips, list) or not isinstance(rdma, list):
            raise ValueError("cluster host ips and rdma fields must be arrays")
        python_executable = payload.get("python_executable")
        if python_executable is not None and not isinstance(python_executable, str):
            raise ValueError("cluster host Python executable must be a string")
        return cls(
            node_id=node_id,
            ssh=ssh,
            ips=tuple(ips),
            rdma=tuple(rdma),
            python_executable=python_executable,
        )


def _assignment_from_dict(payload: dict[str, Any]) -> PipelineAssignment:
    if not isinstance(payload, dict):
        raise ValueError("pipeline assignment must be an object")
    required = (
        "node_id",
        "rank",
        "start_layer",
        "end_layer",
        "layer_weight_bytes",
        "fixed_weight_bytes",
        "reserve_bytes",
        "capacity_bytes",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"pipeline assignment is missing {', '.join(missing)}")
    # Decoded before the constructor so the reason survives: everything inside
    # the try below is reported as "contains an invalid value", and a rank that
    # refuses to launch over a role deserves to say which role.
    role = normalize_node_role(payload.get("role", ""))
    memory_guard_tier = normalize_memory_guard_tier(
        payload.get("memory_guard_tier", "balanced")
    )
    try:
        predicted = {}
        for key in (
            "predicted_compute_seconds",
            "predicted_send_seconds",
            "predicted_stage_seconds",
        ):
            value = payload.get(key)
            if value is not None:
                value = float(value)
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{key} must be finite and non-negative")
            predicted[key] = value
        assignment = PipelineAssignment(
            node_id=str(payload["node_id"]),
            rank=int(payload["rank"]),
            start_layer=int(payload["start_layer"]),
            end_layer=int(payload["end_layer"]),
            layer_weight_bytes=int(payload["layer_weight_bytes"]),
            fixed_weight_bytes=int(payload["fixed_weight_bytes"]),
            reserve_bytes=int(payload["reserve_bytes"]),
            capacity_bytes=int(payload["capacity_bytes"]),
            manual_memory_limit=bool(payload.get("manual_memory_limit", False)),
            role=role,
            memory_guard_tier=memory_guard_tier,
            tensor_parallel_rank=int(payload.get("tensor_parallel_rank", 0)),
            tensor_parallel_size=int(payload.get("tensor_parallel_size", 1)),
            tensor_parallel_shard_weight=int(
                payload.get("tensor_parallel_shard_weight", 1)
            ),
            sharded_weight_bytes=int(payload.get("sharded_weight_bytes", 0)),
            # ``to_dict`` has always emitted these three; nothing read them
            # back, so every decoded assignment claimed a 0-byte KV cache.
            # ``planned_weight_bytes`` includes the cache, and that is the
            # number the rank's memory guard is charged and the engine pool
            # reserves against — dropping it under-charged both by the whole
            # cache. A 40 GiB-weights + 20 GiB-KV stage was admitted as 40.
            kv_cache_bytes=int(payload.get("kv_cache_bytes", 0)),
            kv_bytes_per_token=int(payload.get("kv_bytes_per_token", 0)),
            max_context_tokens=int(payload.get("max_context_tokens", 0)),
            **predicted,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("pipeline assignment contains an invalid value") from exc
    if assignment.rank < 0:
        raise ValueError("pipeline assignment rank must be non-negative")
    if not 0 <= assignment.start_layer < assignment.end_layer:
        raise ValueError("pipeline assignment must contain at least one layer")
    if (
        min(
            assignment.layer_weight_bytes,
            assignment.fixed_weight_bytes,
            assignment.reserve_bytes,
            assignment.capacity_bytes,
            assignment.kv_cache_bytes,
            assignment.kv_bytes_per_token,
            assignment.max_context_tokens,
        )
        < 0
    ):
        raise ValueError("pipeline assignment byte counts must be non-negative")
    if assignment.capacity_bytes <= assignment.reserve_bytes:
        raise ValueError("pipeline assignment reserve must be below capacity")
    if assignment.headroom_bytes < 0:
        raise ValueError("pipeline assignment exceeds node capacity")
    return assignment


@dataclass(frozen=True)
class ClusterDeployment:
    """Immutable input used by the engine and the MLX launcher."""

    deployment_id: str
    model: str
    backend: DistributedBackend
    hosts: tuple[ClusterHost, ...]
    assignments: tuple[PipelineAssignment, ...]
    plan_hash: str
    trust_remote_code: bool = False
    execution: ExecutionSettings = field(
        default_factory=lambda: execution_profile("balanced")
    )
    performance_profiles: tuple[NodePerformanceProfile, ...] = ()
    tensor_parallel_size: int = 1
    target_context_tokens: int = 8192
    mtp_enabled: bool = False
    mtp_num_draft_tokens: int | None = None
    # ``disaggregated`` loads one complete replica per rank and transfers the
    # completed prompt cache from ``prefill_rank`` to ``decode_rank``. The
    # default preserves every pre-v3 tensor/pipeline deployment.
    serving_mode: ServingMode = "sharded"
    prefill_rank: int | None = None
    decode_rank: int | None = None
    # node_id → absolute model path on that node. Empty means every node uses
    # ``model`` — the pre-v2 same-absolute-path requirement. Entries override
    # only the nodes they name; the coordinator path stays the fallback.
    path_map: dict[str, str] = field(default_factory=dict)
    tensor_parallel_qualification: TPQualificationProvenance | None = None

    def __post_init__(self) -> None:
        if _NODE_ID.fullmatch(self.deployment_id) is None:
            raise ValueError(f"invalid deployment ID: {self.deployment_id!r}")
        if (
            not isinstance(self.model, str)
            or not self.model.strip()
            or len(self.model.encode()) > _MAX_MODEL_ID_BYTES
            or "\x00" in self.model
        ):
            raise ValueError("model must be a non-empty path or repository ID")
        if self.backend not in {"ring", "jaccl", "jaccl-ring"}:
            raise ValueError(f"unsupported distributed backend: {self.backend!r}")
        if not 2 <= len(self.hosts) <= 64:
            raise ValueError("distributed inference requires between 2 and 64 hosts")
        if not 1 <= self.tensor_parallel_size <= len(self.hosts):
            raise ValueError(
                "tensor_parallel_size must be between 1 and the host count"
            )
        if len(self.hosts) % self.tensor_parallel_size != 0:
            raise ValueError("host count must be divisible by tensor_parallel_size")
        if (
            not isinstance(self.target_context_tokens, int)
            or isinstance(self.target_context_tokens, bool)
            or not 1 <= self.target_context_tokens <= 1_048_576
        ):
            raise ValueError("target_context_tokens must be between 1 and 1,048,576")
        if not isinstance(self.mtp_enabled, bool):
            raise ValueError("mtp_enabled must be a boolean")
        if self.mtp_num_draft_tokens is not None and (
            not isinstance(self.mtp_num_draft_tokens, int)
            or isinstance(self.mtp_num_draft_tokens, bool)
            or not 1 <= self.mtp_num_draft_tokens <= 8
        ):
            raise ValueError("mtp_num_draft_tokens must be between 1 and 8")
        if self.serving_mode not in {"sharded", "disaggregated"}:
            raise ValueError(f"unsupported serving mode: {self.serving_mode!r}")
        if len(self.assignments) != len(self.hosts):
            raise ValueError("host count must match pipeline assignment count")
        if self.hosts[0].ssh != "127.0.0.1":
            raise ValueError("rank 0 must use SSH target 127.0.0.1")
        if len({host.node_id for host in self.hosts}) != len(self.hosts):
            raise ValueError("cluster node IDs must be unique")
        object.__setattr__(
            self,
            "path_map",
            validate_model_path_map(
                self.path_map,
                tuple(host.node_id for host in self.hosts),
            ),
        )
        if len(self.plan_hash) != 64 or any(
            char not in "0123456789abcdef" for char in self.plan_hash
        ):
            raise ValueError("plan_hash must be a lowercase SHA-256 digest")

        assignments = sorted(self.assignments, key=lambda item: item.rank)
        if [item.rank for item in assignments] != list(range(len(self.hosts))):
            raise ValueError("pipeline ranks must be contiguous from zero")
        for rank, (host, assignment) in enumerate(zip(self.hosts, assignments)):
            if host.node_id != assignment.node_id or assignment.rank != rank:
                raise ValueError("host order must match node IDs and pipeline ranks")
            if (
                assignment.tensor_parallel_size != self.tensor_parallel_size
                or assignment.tensor_parallel_rank != rank % self.tensor_parallel_size
            ):
                raise ValueError(
                    "assignment tensor-parallel coordinates do not match deployment"
                )
        for start in range(0, len(assignments), self.tensor_parallel_size):
            tensor_group = assignments[start : start + self.tensor_parallel_size]
            if len({(item.start_layer, item.end_layer) for item in tensor_group}) != 1:
                raise ValueError(
                    "tensor-parallel group members must hold the same layer range"
                )
        if self.serving_mode == "sharded":
            if self.prefill_rank is not None or self.decode_rank is not None:
                raise ValueError(
                    "sharded serving cannot carry prefill/decode rank ownership"
                )
        else:
            if len(self.hosts) != 2 or self.tensor_parallel_size != 1:
                raise ValueError(
                    "disaggregated serving currently requires two full replicas"
                )
            if {self.prefill_rank, self.decode_rank} != {0, 1}:
                raise ValueError(
                    "disaggregated serving requires distinct prefill/decode ranks"
                )
            if self.mtp_enabled or self.mtp_num_draft_tokens is not None:
                raise ValueError(
                    "disaggregated serving does not yet admit speculative decode"
                )
            replica_ranges = {
                (item.start_layer, item.end_layer) for item in assignments
            }
            replica_weights = {
                (item.layer_weight_bytes, item.fixed_weight_bytes)
                for item in assignments
            }
            replica_kv = {
                (item.kv_cache_bytes, item.kv_bytes_per_token) for item in assignments
            }
            if len(replica_ranges) != 1 or len(replica_weights) != 1:
                raise ValueError(
                    "disaggregated ranks must hold identical complete model replicas"
                )
            if len(replica_kv) != 1:
                raise ValueError(
                    "disaggregated ranks must reserve the same full cache shape"
                )
        qualification = self.tensor_parallel_qualification
        if qualification is not None:
            if self.tensor_parallel_size != len(self.hosts):
                raise ValueError(
                    "tensor layout qualification currently requires pure TP"
                )
            weights = tuple(item.tensor_parallel_shard_weight for item in assignments)
            if weights != qualification.shard_weights:
                raise ValueError(
                    "deployment TP weights do not match qualification provenance"
                )
        if self.performance_profiles:
            if len(self.performance_profiles) != len(self.hosts):
                raise ValueError(
                    "performance profile count must match cluster host count"
                )
            for rank, (host, profile) in enumerate(
                zip(self.hosts, self.performance_profiles)
            ):
                if (
                    profile.rank != rank
                    or profile.node_id != host.node_id
                    or profile.backend != self.backend
                ):
                    raise ValueError(
                        "performance profiles must match host rank, ID, and backend"
                    )

        if self.backend == "ring":
            if any(not host.ips for host in self.hosts):
                raise ValueError("ring hosts require at least one communication IP")
        else:
            size = len(self.hosts)
            for rank, host in enumerate(self.hosts):
                if not host.ips:
                    raise ValueError("JACCL hosts require a communication IP")
                if len(host.rdma) != size:
                    raise ValueError("JACCL requires a full RDMA connectivity matrix")
                if host.rdma[rank] is not None:
                    raise ValueError("JACCL RDMA matrix diagonal must be null")
                if any(
                    path is None
                    for index, path in enumerate(host.rdma)
                    if index != rank
                ):
                    raise ValueError("JACCL RDMA matrix is missing a peer path")

    @property
    def world_size(self) -> int:
        return len(self.hosts)

    @property
    def distributed_init_backend(self) -> str:
        return "jaccl" if self.backend.startswith("jaccl") else "ring"

    def model_path_for(self, node_id: str) -> str:
        """The model directory one rank loads, honouring per-node overrides.

        Nodes absent from ``path_map`` keep the coordinator's shared path,
        which is exactly the legacy same-absolute-path behavior.
        """

        return self.path_map.get(node_id, self.model)

    def hostfile_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "envs": _hostfile_envs(),
            "hosts": [
                {
                    "ssh": host.ssh,
                    "ips": _hostfile_ips(host),
                    "rdma": [
                        list(path) if isinstance(path, tuple) else path
                        for path in host.rdma
                    ],
                }
                for host in self.hosts
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": DEPLOYMENT_SCHEMA_VERSION,
            "deployment_id": self.deployment_id,
            "model": self.model,
            "backend": self.backend,
            "hosts": [host.to_dict() for host in self.hosts],
            "assignments": [assignment.to_dict() for assignment in self.assignments],
            "plan_hash": self.plan_hash,
            "trust_remote_code": self.trust_remote_code,
            "execution": self.execution.to_dict(),
            "performance_profiles": [
                profile.to_dict() for profile in self.performance_profiles
            ],
            "tensor_parallel_size": self.tensor_parallel_size,
            "target_context_tokens": self.target_context_tokens,
            "mtp_enabled": self.mtp_enabled,
            "mtp_num_draft_tokens": self.mtp_num_draft_tokens,
            "serving_mode": self.serving_mode,
            "prefill_rank": self.prefill_rank,
            "decode_rank": self.decode_rank,
            "path_map": dict(sorted(self.path_map.items())),
        }
        if self.tensor_parallel_qualification is not None:
            result["tensor_parallel_qualification"] = (
                self.tensor_parallel_qualification.to_dict()
            )
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ClusterDeployment:
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version", 1) not in _SUPPORTED_DEPLOYMENT_SCHEMAS
        ):
            raise ValueError("unsupported cluster deployment schema")
        hosts = payload.get("hosts")
        assignments = payload.get("assignments")
        performance_profiles = payload.get("performance_profiles", [])
        if (
            not isinstance(hosts, list)
            or not isinstance(assignments, list)
            or not isinstance(performance_profiles, list)
        ):
            raise ValueError("deployment hosts and assignments must be arrays")
        deployment_id = payload.get("deployment_id")
        model = payload.get("model")
        backend = payload.get("backend")
        plan_hash = payload.get("plan_hash")
        if not all(
            isinstance(value, str)
            for value in (deployment_id, model, backend, plan_hash)
        ):
            raise ValueError("deployment identity fields must be strings")
        mtp_enabled = payload.get("mtp_enabled", False)
        if not isinstance(mtp_enabled, bool):
            raise ValueError("deployment mtp_enabled must be a boolean")
        return cls(
            deployment_id=deployment_id,
            model=model,
            backend=backend,
            hosts=tuple(ClusterHost.from_dict(host) for host in hosts),
            assignments=tuple(
                _assignment_from_dict(assignment) for assignment in assignments
            ),
            plan_hash=plan_hash,
            trust_remote_code=bool(payload.get("trust_remote_code", False)),
            execution=ExecutionSettings.from_dict(payload.get("execution")),
            performance_profiles=tuple(
                NodePerformanceProfile.from_dict(profile)
                for profile in performance_profiles
            ),
            tensor_parallel_size=int(payload.get("tensor_parallel_size", 1)),
            target_context_tokens=int(payload.get("target_context_tokens", 8192)),
            mtp_enabled=mtp_enabled,
            mtp_num_draft_tokens=payload.get("mtp_num_draft_tokens"),
            serving_mode=str(payload.get("serving_mode", "sharded")),
            prefill_rank=payload.get("prefill_rank"),
            decode_rank=payload.get("decode_rank"),
            # Schema 1 payloads predate per-node paths; they decode to the
            # empty map, which is the shared-path behavior they ran with.
            path_map=validate_model_path_map(payload.get("path_map")),
            tensor_parallel_qualification=(
                TPQualificationProvenance.from_dict(
                    payload["tensor_parallel_qualification"]
                )
                if payload.get("tensor_parallel_qualification") is not None
                else None
            ),
        )

    def encode_worker_plan(self) -> str:
        """Encode the small trusted plan as a bounded command-line argument."""

        worker_payload = {
            "schema_version": DEPLOYMENT_SCHEMA_VERSION,
            "plan_hash": self.plan_hash,
            "assignments": [assignment.to_dict() for assignment in self.assignments],
            "performance_profiles": [
                profile.to_dict() for profile in self.performance_profiles
            ],
            "tensor_parallel_size": self.tensor_parallel_size,
            "mtp_enabled": self.mtp_enabled,
            "mtp_num_draft_tokens": self.mtp_num_draft_tokens,
            "serving_mode": self.serving_mode,
            "prefill_rank": self.prefill_rank,
            "decode_rank": self.decode_rank,
            "execution": self.execution.to_dict(),
            "path_map": dict(sorted(self.path_map.items())),
        }
        if self.tensor_parallel_qualification is not None:
            worker_payload["tensor_parallel_qualification"] = (
                self.tensor_parallel_qualification.to_dict()
            )
        raw = json.dumps(
            worker_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        if len(raw) > _MAX_PLAN_BYTES:
            raise ValueError("pipeline plan is too large")
        return base64.urlsafe_b64encode(zlib.compress(raw, level=9)).decode()


def _decode_worker_payload(encoded: str) -> dict[str, Any]:
    """Inflate and validate a worker plan payload without accepting code."""

    if not isinstance(encoded, str) or len(encoded) > _MAX_PLAN_BYTES * 2:
        raise ValueError("encoded pipeline plan is too large")
    try:
        compressed = base64.b64decode(encoded, altchars=b"-_", validate=True)
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(compressed, _MAX_PLAN_BYTES + 1)
        if decompressor.unconsumed_tail:
            raise ValueError("decoded pipeline plan is too large")
        raw += decompressor.flush(max(1, _MAX_PLAN_BYTES + 1 - len(raw)))
    except (binascii.Error, zlib.error) as exc:
        raise ValueError("encoded pipeline plan is invalid") from exc
    if (
        len(raw) > _MAX_PLAN_BYTES
        or decompressor.unconsumed_tail
        or decompressor.unused_data
    ):
        raise ValueError("decoded pipeline plan is too large or malformed")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("pipeline plan is not valid JSON") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") not in _SUPPORTED_DEPLOYMENT_SCHEMAS
    ):
        raise ValueError("unsupported pipeline plan schema")
    plan_hash = payload.get("plan_hash")
    assignments = payload.get("assignments")
    performance_profiles = payload.get("performance_profiles", [])
    if (
        not isinstance(plan_hash, str)
        or not isinstance(assignments, list)
        or not isinstance(performance_profiles, list)
    ):
        raise ValueError("pipeline plan is missing required fields")
    if len(plan_hash) != 64 or any(
        char not in "0123456789abcdef" for char in plan_hash
    ):
        raise ValueError("pipeline plan hash is invalid")
    qualification = payload.get("tensor_parallel_qualification")
    if qualification is not None:
        payload["tensor_parallel_qualification"] = TPQualificationProvenance.from_dict(
            qualification
        ).to_dict()
    return payload


def decode_worker_contract(
    encoded: str,
) -> tuple[
    str,
    tuple[PipelineAssignment, ...],
    tuple[NodePerformanceProfile, ...],
    int,
]:
    """Decode and validate the full worker contract without accepting code."""

    payload = _decode_worker_payload(encoded)
    parsed = tuple(_assignment_from_dict(item) for item in payload["assignments"])
    if [item.rank for item in sorted(parsed, key=lambda item: item.rank)] != list(
        range(len(parsed))
    ):
        raise ValueError("pipeline plan ranks must be contiguous from zero")
    profiles = tuple(
        NodePerformanceProfile.from_dict(item)
        for item in payload.get("performance_profiles", [])
    )
    if profiles and (
        len(profiles) != len(parsed)
        or [item.rank for item in profiles] != list(range(len(parsed)))
        or any(profile.node_id != parsed[profile.rank].node_id for profile in profiles)
    ):
        raise ValueError("worker performance profiles do not match the shard plan")
    tensor_parallel_size = int(payload.get("tensor_parallel_size", 1))
    if not 1 <= tensor_parallel_size <= len(parsed):
        raise ValueError(
            "tensor_parallel_size must be between 1 and the assignment count"
        )
    qualification_payload = payload.get("tensor_parallel_qualification")
    if qualification_payload is not None:
        qualification = TPQualificationProvenance.from_dict(qualification_payload)
        ordered = tuple(sorted(parsed, key=lambda item: item.rank))
        if (
            tensor_parallel_size != len(ordered)
            or tuple(item.tensor_parallel_shard_weight for item in ordered)
            != qualification.shard_weights
        ):
            raise ValueError("worker TP weights do not match qualification provenance")
    return payload["plan_hash"], parsed, profiles, tensor_parallel_size


def decode_worker_path_map(encoded: str) -> dict[str, str]:
    """Per-node model path overrides carried inside the worker contract.

    Schema 1 contracts predate per-node paths and decode to an empty map;
    callers then fall back to the shared ``--model`` argument, which is the
    legacy behavior those contracts ran with.
    """

    payload = _decode_worker_payload(encoded)
    return validate_model_path_map(payload.get("path_map"))


def decode_worker_speculation(encoded: str) -> tuple[bool, int | None]:
    """Validated speculative-decode settings carried to every rank."""

    payload = _decode_worker_payload(encoded)
    enabled = payload.get("mtp_enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("worker mtp_enabled must be a boolean")
    raw_depth = payload.get("mtp_num_draft_tokens")
    if raw_depth is None:
        return enabled, None
    if (
        not isinstance(raw_depth, int)
        or isinstance(raw_depth, bool)
        or not 1 <= raw_depth <= 8
    ):
        raise ValueError("worker mtp_num_draft_tokens must be between 1 and 8")
    return enabled, raw_depth


def decode_worker_serving_mode(
    encoded: str,
) -> tuple[ServingMode, int | None, int | None]:
    """Validated phase ownership carried to every persistent rank."""

    payload = _decode_worker_payload(encoded)
    mode = payload.get("serving_mode", "sharded")
    if mode not in {"sharded", "disaggregated"}:
        raise ValueError("worker serving mode is invalid")
    prefill_rank = payload.get("prefill_rank")
    decode_rank = payload.get("decode_rank")
    if mode == "sharded":
        if prefill_rank is not None or decode_rank is not None:
            raise ValueError("sharded worker contract carries phase ownership")
        return mode, None, None
    if {prefill_rank, decode_rank} != {0, 1}:
        raise ValueError("disaggregated worker phase ranks are invalid")
    return mode, int(prefill_rank), int(decode_rank)


def decode_worker_execution(encoded: str) -> ExecutionSettings:
    """Validated execution settings carried to persistent worker ranks.

    Schema 1-3 launch contracts created before this field was added retain the
    balanced profile defaults. New contracts carry the exact signed settings so
    telemetry and serving behavior cannot drift from the dashboard proposal.
    """

    payload = _decode_worker_payload(encoded)
    return ExecutionSettings.from_dict(payload.get("execution"))


def decode_worker_plan(encoded: str) -> tuple[str, tuple[PipelineAssignment, ...]]:
    """Backward-compatible assignment-only worker plan decoder."""

    plan_hash, assignments, _, _ = decode_worker_contract(encoded)
    return plan_hash, assignments
