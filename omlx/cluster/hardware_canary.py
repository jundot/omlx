# SPDX-License-Identifier: Apache-2.0
"""Small, download-free Apple-silicon regression canary."""

from __future__ import annotations

import json
import math
import os
import platform
import select
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from contextlib import suppress
from types import SimpleNamespace
from typing import Any

from .collective import run_local_collective_smoke, run_local_pipeline_smoke
from .deployment import ClusterDeployment, ClusterHost
from .launch import (
    DistributedJobSupervisor,
    DistributedMemoryRecoveryError,
    _process_group_alive,
)
from .planner import PipelineAssignment
from .supervisor import run_worker_smoke


class HardwareCanaryError(RuntimeError):
    """Raised when an Apple-silicon hardware assertion fails."""


class HardwareCanarySkipError(RuntimeError):
    """Raised when this host cannot execute the Metal canary."""


def _require_metal() -> Any:
    if platform.system() != "Darwin" or platform.machine().lower() != "arm64":
        raise HardwareCanarySkipError("requires an Apple-silicon Mac")
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise HardwareCanarySkipError("MLX is not installed") from exc
    if not mx.metal.is_available():
        raise HardwareCanarySkipError("the MLX Metal backend is unavailable")
    return mx


def _vision_prefill_metal_canary() -> dict[str, Any]:
    """Exercise isolated image attention and the tiny ViT on real Metal."""

    mx = _require_metal()
    from omlx.cluster.deepseek_v4_vision_runtime import vision_prefill_chunks
    from omlx.deepseek_v4_vision import (
        IMAGE,
        IMAGE_END,
        IMAGE_NEWLINE,
        IMAGE_START,
    )
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch
    from omlx.patches.deepseek_v4.vision_model import Aligner, ViT

    # The vendored model is registered under its mlx-lm package name so its
    # relative imports resolve exactly as they do during real model loading.
    apply_deepseek_v4_patch()
    from mlx_lm.models.deepseek_v4 import _apply_image_visibility_mask

    vocab_size = 256
    prefix = [7] * 1024
    image = (
        [vocab_size + IMAGE_START]
        + [vocab_size + IMAGE] * 64
        + [vocab_size + IMAGE_NEWLINE, vocab_size + IMAGE_END]
    )
    suffix = [11] * 1024
    prompt = prefix + image + suffix
    chunks = vision_prefill_chunks(
        prompt,
        vocab_size=vocab_size,
        max_chunk_tokens=128,
    )
    image_chunks = [
        (start, end)
        for start, end in chunks
        if any(token >= vocab_size for token in prompt[start:end])
    ]
    expected_image_chunk = (len(prefix), len(prefix) + len(image))
    if image_chunks != [expected_image_chunk]:
        raise HardwareCanaryError(
            f"image span was not isolated from text: {image_chunks}"
        )
    if any(
        end - start > 128 and (start, end) != expected_image_chunk
        for start, end in chunks
    ):
        raise HardwareCanaryError("ordinary text exceeded the prefill chunk bound")

    mx.synchronize()
    mx.clear_cache()
    baseline = int(mx.get_active_memory())
    mx.reset_peak_memory()

    token_array = mx.array([image])
    visibility, applied = _apply_image_visibility_mask(
        token_array,
        None,
        vocab_size,
    )
    if not applied or visibility is None:
        raise HardwareCanaryError("image visibility mask was not applied")

    length = len(image)
    head_dim = 16
    query = mx.ones((1, 2, length, head_dim), dtype=mx.float32)
    key = mx.ones((1, 2, length, head_dim), dtype=mx.float32)
    value = mx.arange(length * head_dim, dtype=mx.float32).reshape(
        1, 1, length, head_dim
    )
    value = mx.broadcast_to(value, (1, 2, length, head_dim))
    attention = mx.fast.scaled_dot_product_attention(
        query,
        key,
        value,
        scale=head_dim**-0.5,
        mask=visibility,
    )

    config = SimpleNamespace(
        vision_patch_size=2,
        vision_dim=32,
        vision_n_heads=4,
        vision_inter_dim=64,
        vision_n_layers=2,
        vision_rope_theta=10_000.0,
        vision_downsample_ratio=2,
        hidden_size=48,
    )
    vision = ViT(config)
    aligner = Aligner(config)
    features = vision(mx.ones((64, 3, 2, 2)), 8, 8)
    embeddings = aligner(features, 8, 8)
    checksum = mx.sum(attention.astype(mx.float32)) + mx.sum(embeddings)
    mx.eval(visibility, attention, features, embeddings, checksum)
    value_sum = float(checksum.item())
    if not math.isfinite(value_sum):
        raise HardwareCanaryError("Metal vision result was not finite")
    if tuple(embeddings.shape) != (16, 48):
        raise HardwareCanaryError(
            f"tiny aligner returned unexpected shape {embeddings.shape}"
        )

    peak = int(mx.get_peak_memory())
    peak_delta = max(0, peak - baseline)
    # This is intentionally generous across MLX releases while still catching
    # an accidental full-context attention allocation in this 2k-token probe.
    peak_limit = 256 * 1024**2
    if peak_delta > peak_limit:
        raise HardwareCanaryError(
            "tiny vision prefill exceeded its 256 MiB Metal transient bound: "
            f"{peak_delta} bytes"
        )

    del attention, embeddings, features, key, query, token_array, value, visibility
    mx.synchronize()
    mx.clear_cache()
    return {
        "ok": True,
        "prompt_tokens": len(prompt),
        "chunk_count": len(chunks),
        "image_chunk": list(expected_image_chunk),
        "image_tokens": length,
        "visibility_cells": length * length,
        "unisolated_attention_cells": len(prompt) * len(prompt),
        "embedding_shape": [16, 48],
        "metal_peak_delta_bytes": peak_delta,
        "metal_peak_limit_bytes": peak_limit,
        "checksum": value_sum,
    }


def _canary_deployment() -> ClusterDeployment:
    capacity = 2 * 1024**3
    assignments = (
        PipelineAssignment("local-0", 0, 1, 2, 8 * 1024**2, 1024, 0, capacity),
        PipelineAssignment("local-1", 1, 0, 1, 8 * 1024**2, 1024, 0, capacity),
    )
    return ClusterDeployment(
        deployment_id="local-hardware-canary",
        model="synthetic/hardware-canary",
        backend="ring",
        hosts=(
            ClusterHost("local-0", "127.0.0.1", ("127.0.0.1",)),
            ClusterHost("local-1", "127.0.0.1", ("127.0.0.1",)),
        ),
        assignments=assignments,
        plan_hash="a" * 64,
    )


def _read_holder_ready(
    process: subprocess.Popen[str], timeout: float
) -> dict[str, Any]:
    assert process.stdout is not None
    readable, _, _ = select.select([process.stdout], [], [], timeout)
    if not readable:
        raise HardwareCanaryError("Metal allocation holder did not become ready")
    line = process.stdout.readline()
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise HardwareCanaryError(
            f"invalid Metal holder output: {line.strip()}"
        ) from exc
    if payload.get("type") != "metal_holder_ready":
        raise HardwareCanaryError(f"unexpected Metal holder output: {payload}")
    return payload


def _crash_recovery_canary(*, allocation_mib: int, timeout: float) -> dict[str, Any]:
    """Force teardown of a stuck Metal owner and check reload quarantine."""

    _require_metal()
    if not 16 <= allocation_mib <= 512:
        raise ValueError("allocation_mib must be between 16 and 512")
    allocation_bytes = allocation_mib * 1024**2
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "omlx.cluster.hardware_canary_worker",
            "--allocation-bytes",
            str(allocation_bytes),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    process_group = process.pid
    started_at = time.monotonic()
    try:
        ready = _read_holder_ready(process, min(timeout, 15.0))
        with tempfile.TemporaryDirectory(prefix="omlx-hardware-canary-") as state_dir:
            deployment = _canary_deployment()
            supervisor = DistributedJobSupervisor(
                deployment,
                state_dir=state_dir,
                stop_timeout=min(1.0, max(0.25, timeout / 10)),
                recovery_timeout=min(5.0, timeout),
                preflight=False,
            )
            supervisor.process = process
            supervisor.stop()
            if _process_group_alive(process_group):
                raise HardwareCanaryError(
                    f"fault-injected process group {process_group} survived teardown"
                )
            if process.returncode != -signal.SIGKILL:
                raise HardwareCanaryError(
                    "SIGTERM-ignoring Metal holder was not escalated to SIGKILL: "
                    f"return code {process.returncode}"
                )

            # The operating system's free-memory estimate legitimately jitters.
            # Use fixed ceilings here to test the persistent barrier itself:
            # one failed reprobe must block reload, and full recovery must clear
            # the marker. The live holder above independently covers process and
            # Metal-allocation cleanup through the production supervisor.
            baseline = 1024**3
            low = baseline - 300 * 1024**2
            supervisor._recovery_baseline_by_rank = {0: baseline, 1: baseline}
            supervisor._write_recovery_quarantine("hardware canary retained memory")
            marker = supervisor._recovery_marker_path()
            recreated = DistributedJobSupervisor(
                deployment,
                state_dir=state_dir,
                recovery_timeout=0,
                preflight=False,
            )
            recreated._probe_recovery_capacity = lambda: [
                {
                    "rank": rank,
                    "node_id": f"local-{rank}",
                    "admission_ceiling_bytes": low,
                }
                for rank in (0, 1)
            ]
            try:
                recreated._check_recovery_quarantine()
            except DistributedMemoryRecoveryError:
                pass
            else:
                raise HardwareCanaryError("retained-memory reload was not quarantined")
            if not marker.is_file():
                raise HardwareCanaryError("reload quarantine marker was not persisted")

            recreated._probe_recovery_capacity = lambda: [
                {
                    "rank": rank,
                    "node_id": f"local-{rank}",
                    "admission_ceiling_bytes": baseline,
                }
                for rank in (0, 1)
            ]
            recreated._check_recovery_quarantine()
            if marker.exists():
                raise HardwareCanaryError("recovered-memory quarantine was not cleared")
    finally:
        if _process_group_alive(process_group):
            with suppress(ProcessLookupError):
                os.killpg(process_group, signal.SIGKILL)
        if process.poll() is None:
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        for stream in (process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()

    return {
        "ok": True,
        "allocation_bytes": allocation_bytes,
        "holder_active_memory_bytes": int(ready["active_memory_bytes"]),
        "forced_signal": "SIGKILL",
        "process_group_reaped": True,
        "reload_blocked_while_retained": True,
        "reload_allowed_after_recovery": True,
        "elapsed_seconds": time.monotonic() - started_at,
    }


def run_local_hardware_canary(
    *,
    timeout: float = 45.0,
    allocation_mib: int = 128,
    worker_runner: Callable[..., dict[str, Any]] = run_worker_smoke,
    collective_runner: Callable[..., dict[str, Any]] = run_local_collective_smoke,
    pipeline_runner: Callable[..., dict[str, Any]] = run_local_pipeline_smoke,
    vision_runner: Callable[..., dict[str, Any]] = _vision_prefill_metal_canary,
    crash_runner: Callable[..., dict[str, Any]] = _crash_recovery_canary,
) -> dict[str, Any]:
    """Run the strongest safe, model-free cluster checks on one Mac."""

    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    if not 16 <= allocation_mib <= 512:
        raise ValueError("allocation_mib must be between 16 and 512")
    _require_metal()
    started_at = time.monotonic()
    checks = {
        "worker": worker_runner(timeout=min(5.0, timeout)),
        "collective": collective_runner(timeout=timeout),
        "pipeline": pipeline_runner(timeout=timeout),
        "vision_prefill": vision_runner(),
        "crash_recovery": crash_runner(
            allocation_mib=allocation_mib,
            timeout=timeout,
        ),
    }
    return {
        "ok": True,
        "skipped": False,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "checks": checks,
        "elapsed_seconds": time.monotonic() - started_at,
        "limitations": [
            "loopback does not validate cross-host Ring, Thunderbolt, or JACCL",
            "the tiny synthetic graph does not validate the full checkpoint",
            "fault injection validates containment, not a real IOGPU driver leak",
        ],
    }
