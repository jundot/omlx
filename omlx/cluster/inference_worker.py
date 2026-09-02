# SPDX-License-Identifier: Apache-2.0
"""Rank process for an oMLX-managed MLX-LM distributed inference job."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from contextlib import contextmanager, nullcontext, suppress
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .deployment import decode_worker_contract
from .liveness import PeerWatchdog
from .memory_guard import (
    admission_budget,
    assignment_memory_safety,
    check_rank_fits,
    guard_rank_load,
    watch_rank_load,
)
from .performance import ExecutionSettings
from .pipeline_compat import (
    install_pipeline_compatibility,
    pipeline_assignment_is_honored,
)
from .planner import PipelineAssignment
from .prefill_guard import build_guard
from .progressive_loading import install_progressive_loader
from .runtime_optimizations import install_runtime_optimizations
from .telemetry import install_server_telemetry

_EVENT_PREFIX = "OMLX_CLUSTER_EVENT:"


def _emit_event(payload: dict[str, Any]) -> None:
    print(
        _EVENT_PREFIX + json.dumps(payload, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


def _cross_thread_generation_stream(mx: Any) -> Any:
    """Create the one MLX stream used by both rank loading and generation.

    ``mlx_lm.server.ResponseGenerator`` performs generation on a background
    thread. A regular MLX stream exists only on the thread that created it, so
    eagerly loading and validating a distributed model on the rank's main
    thread leaves lazy graph nodes referring to a stream that the generation
    thread cannot see. MLX 0.32's thread-unsafe stream is deliberately the
    cross-thread variant; the worker serializes model work on this one stream.
    """

    stream = mx.new_thread_unsafe_stream(mx.default_device())
    mx.set_default_stream(stream)
    return stream


@contextmanager
def _bind_generation_thread_stream(
    response_generator_type: Any,
    mx: Any,
    stream: Any,
):
    """Install the rank's cross-thread stream before MLX-LM starts decoding."""

    original_generate = response_generator_type._generate

    @wraps(original_generate)
    def generate_on_rank_stream(instance: Any) -> Any:
        mx.set_default_stream(stream)
        return original_generate(instance)

    response_generator_type._generate = generate_on_rank_stream
    try:
        yield
    finally:
        response_generator_type._generate = original_generate


def _encode_thinking_marker(tokenizer: Any, text: str) -> list[int] | None:
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        try:
            ids = tokenizer.encode(text)
        except Exception:
            return None
    except Exception:
        return None
    if not ids:
        return None
    encoded = [token_id for token_id in ids if isinstance(token_id, int)]
    return encoded or None


def _think_token_ids(tokenizer: Any) -> tuple[list[int], int | None]:
    """Resolve the close-think token ID(s) and open-think token ID from mlx-lm.

    The single-machine scheduler reads these off the tokenizer and feeds them
    into ``ThinkingBudgetProcessor``. The rank has the same tokenizer, so it
    can resolve them the same way (scheduler.py ``_get_think_token_id`` /
    ``_resolve_think_end_token_ids`` / ``_encode_thinking_marker``).
    """
    think_end_ids: list[int] | None = None

    try:
        end_id = getattr(tokenizer, "think_end_id", None)
        if end_id is not None:
            think_end_ids = [end_id]
    except (ValueError, TypeError):
        pass
    if think_end_ids is None:
        think_end_str = getattr(tokenizer, "think_end", None) or "</think>"
        think_end_ids = _encode_thinking_marker(tokenizer, think_end_str)
    if think_end_ids is None:
        try:
            tid = tokenizer.convert_tokens_to_ids("</think>")
            if isinstance(tid, int) and tid != getattr(tokenizer, "unk_token_id", None):
                think_end_ids = [tid]
        except (AttributeError, KeyError, TypeError):
            pass
    try:
        think_start_id = getattr(tokenizer, "think_start_id", None)
    except (ValueError, TypeError):
        think_start_id = None
    if not isinstance(think_start_id, int):
        try:
            start_text = getattr(tokenizer, "think_start", None) or "<think>"
            candidate = tokenizer.convert_tokens_to_ids(start_text)
            think_start_id = (
                candidate
                if isinstance(candidate, int)
                and candidate != getattr(tokenizer, "unk_token_id", None)
                else None
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            think_start_id = None
    return think_end_ids or [], think_start_id


def _thinking_close_pattern(
    tokenizer: Any, think_end_text: str
) -> tuple[list[int] | None, list[int] | None]:
    """Resolve whitespace tokens surrounding a close-think marker."""
    template = getattr(tokenizer, "_chat_template", None) or getattr(
        tokenizer, "chat_template", None
    )
    if not template:
        return None, None
    template = template if isinstance(template, str) else str(template)
    match = re.search(
        r"(\\n|\\r|[\n\r])*" + re.escape(think_end_text) + r"((?:\\n|\\r|[\n\r])*)",
        template,
    )
    if match is None:
        return None, None
    matched = match.group(0)
    leading, trailing = matched.split(think_end_text, 1)
    leading = leading.replace("\\n", "\n").replace("\\r", "\r")
    trailing = trailing.replace("\\n", "\n").replace("\\r", "\r")
    return (
        _encode_thinking_marker(tokenizer, leading) if leading else None,
        _encode_thinking_marker(tokenizer, trailing) if trailing else None,
    )


def _thinking_budget_token_to_piece(
    tokenizer: Any, token_id: int
) -> str | bytes | None:
    """Best-effort token piece lookup for UTF-8-safe budget forcing."""
    try:
        token = tokenizer.convert_ids_to_tokens(token_id)
        if isinstance(token, bytes):
            return token
        if isinstance(token, str):
            byte_fallback = re.fullmatch(r"(?:<0x[0-9A-Fa-f]{2}>)+", token)
            if byte_fallback is not None:
                return bytes(
                    int(match.group(1), 16)
                    for match in re.finditer(r"<0x([0-9A-Fa-f]{2})>", token)
                )
            byte_decoder = getattr(tokenizer, "byte_decoder", None)
            if isinstance(byte_decoder, dict) and token:
                try:
                    return bytes(byte_decoder[character] for character in token)
                except (KeyError, TypeError, ValueError):
                    pass
            return token
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except TypeError:
        try:
            return tokenizer.decode([token_id])
        except Exception:
            return None
    except Exception:
        return None


@contextmanager
def _install_thinking_budget_support(mlx_server: Any, tokenizer: Any):
    """Make the rank honor a per-request ``thinking_budget``.

    MLX-LM's server builds its logits processors once from CLI args
    (``_make_logits_processors``) and never looks at the request body, so a
    thinking budget forwarded by the coordinator is silently ignored. Wrap that
    factory so a request carrying ``chat_template_kwargs.thinking_budget`` gets
    an ``omlx`` ``ThinkingBudgetProcessor`` appended, enforcing the budget in
    the rank's decode loop exactly like single-machine inference does.
    """
    from omlx.api.thinking import ThinkingBudgetProcessor, prompt_opens_thinking

    think_end_ids, think_start_id = _think_token_ids(tokenizer)
    if not think_end_ids:
        yield
        return

    original = mlx_server._make_logits_processors
    response_generator_type = mlx_server.ResponseGenerator
    original_tokenize = response_generator_type._tokenize
    think_end_text = getattr(tokenizer, "think_end", None) or "</think>"
    leading_ids, trailing_ids = _thinking_close_pattern(tokenizer, think_end_text)

    def token_to_piece(token_id: int) -> str | bytes | None:
        return _thinking_budget_token_to_piece(tokenizer, token_id)

    @wraps(original_tokenize)
    def tokenize_with_thinking_state(
        instance: Any, request_tokenizer: Any, request: Any, args: Any
    ) -> Any:
        result = original_tokenize(instance, request_tokenizer, request, args)
        prompt, initial_state = result[0], result[3]
        active = initial_state == "reasoning"
        if getattr(request, "request_type", None) == "text":
            active, _ = prompt_opens_thinking(
                request_tokenizer,
                getattr(request, "prompt", ""),
                prompt,
            )
        args._omlx_thinking_active = active
        return result

    @wraps(original)
    def with_thinking_budget(args: Any) -> list[Any]:
        processors = original(args)
        kwargs = getattr(args, "chat_template_kwargs", None) or {}
        budget = kwargs.get("thinking_budget")
        if (
            budget is not None
            and kwargs.get("enable_thinking") is not False
            and getattr(args, "_omlx_thinking_active", False)
        ):
            processors = list(processors)
            processors.append(
                ThinkingBudgetProcessor(
                    think_end_token_ids=think_end_ids,
                    budget=budget,
                    think_start_token_id=think_start_id,
                    leading_token_ids=leading_ids,
                    trailing_token_ids=trailing_ids,
                    token_to_piece=token_to_piece,
                )
            )
        return processors

    response_generator_type._tokenize = tokenize_with_thinking_state
    mlx_server._make_logits_processors = with_thinking_budget
    try:
        yield
    finally:
        mlx_server._make_logits_processors = original
        response_generator_type._tokenize = original_tokenize


class RuntimeMarker:
    """Best-effort status shared with the oMLX GUI running on this Mac."""

    def __init__(
        self,
        *,
        state_dir: str,
        deployment_id: str,
        rank: int,
        world_size: int,
        model: str,
        backend: str,
        plan_hash: str,
    ) -> None:
        root = Path(state_dir).expanduser()
        self.path = root / f"{deployment_id}-rank-{rank}.json"
        self.payload = {
            "schema_version": 1,
            "deployment_id": deployment_id,
            "pid": os.getpid(),
            "rank": rank,
            "world_size": world_size,
            "model": model,
            "backend": backend,
            "plan_hash": plan_hash,
        }
        self._extra: dict[str, Any] = {}
        self._phase = "initializing"
        self._lock = threading.Lock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

    def update(self, phase: str, **extra: Any) -> None:
        with self._lock:
            self._phase = phase
            self._extra.update(extra)
            payload = (
                self.payload
                | self._extra
                | {
                    "phase": phase,
                    "updated_at": datetime.now(UTC).isoformat(),
                }
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                dir=self.path.parent,
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    json.dump(
                        payload,
                        stream,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
            finally:
                with suppress(FileNotFoundError):
                    os.unlink(temporary)

    def remove(self) -> None:
        with self._lock, suppress(FileNotFoundError):
            self.path.unlink()

    def start_heartbeat(self, *, interval: float = 10.0) -> None:
        """Keep loading ranks observable before request telemetry exists."""

        if interval <= 0 or self._heartbeat_thread is not None:
            return
        self._heartbeat_stop.clear()

        def run() -> None:
            while not self._heartbeat_stop.wait(interval):
                try:
                    with self._lock:
                        phase = self._phase
                    self.update(phase)
                except Exception:
                    # The owning worker must not die because status publication
                    # failed; peers will detect the missing heartbeat.
                    pass

        self._heartbeat_thread = threading.Thread(
            target=run,
            name="omlx-cluster-rank-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def stop_heartbeat(self, *, timeout: float = 2.0) -> None:
        self._heartbeat_stop.set()
        thread, self._heartbeat_thread = self._heartbeat_thread, None
        if thread is not None:
            thread.join(timeout=timeout)


def _server_arguments(
    args: argparse.Namespace,
    *,
    tensor_parallel_size: int = 1,
) -> SimpleNamespace:
    """Build the pinned MLX-LM server contract without invoking its CLI."""

    return SimpleNamespace(
        model=args.model,
        adapter_path=None,
        draft_model=None,
        # Pure tensor parallelism must not also install pipeline recv/send
        # edges. Hybrid plans remain fail-closed elsewhere.
        pipeline=tensor_parallel_size == 1,
        trust_remote_code=args.trust_remote_code,
        chat_template="",
        chat_template_args={},
        use_default_chat_template=False,
        allowed_origins=[],
        temp=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        max_tokens=512,
        num_draft_tokens=0,
        decode_concurrency=args.decode_concurrency,
        prompt_concurrency=args.prompt_concurrency,
        prefill_step_size=args.prefill_step_size,
        prompt_cache_size=args.prompt_cache_size,
        prompt_cache_bytes=args.prompt_cache_bytes,
        max_kv_size=args.max_kv_size,
    )


def _distributed_model_type(model_path: str | Path) -> str:
    """Read the family discriminator used by cluster-only protocol patches."""

    try:
        config = json.loads((Path(model_path) / "config.json").read_text())
    except (OSError, TypeError, ValueError):
        return ""
    model_type = config.get("model_type") if isinstance(config, dict) else None
    return str(model_type or "").strip().lower()


def _install_distributed_model_protocol(tokenizer: Any, model_path: str | Path) -> str:
    """Install the normal oMLX model protocol on this distributed rank."""

    model_type = _distributed_model_type(model_path)
    from omlx.adapter.output_parser import (
        install_minimax_m3_tokenizer_protocol,
    )

    installed = install_minimax_m3_tokenizer_protocol(
        tokenizer,
        str(model_path),
        {"model_type": model_type} if model_type else None,
    )
    return "minimax_m3" if installed else ""


def _execution_settings(args: argparse.Namespace) -> ExecutionSettings:
    return ExecutionSettings(
        profile=args.execution_profile,
        auto_tune=args.auto_tune,
        decode_concurrency=args.decode_concurrency,
        prompt_concurrency=args.prompt_concurrency,
        prefill_step_size=args.prefill_step_size,
        prompt_cache_size=args.prompt_cache_size,
        prompt_cache_bytes=args.prompt_cache_bytes,
        max_kv_size=args.max_kv_size,
        pipeline_microbatch_size=args.pipeline_microbatch_size,
        cache_affinity=args.cache_affinity,
        prompt_cache_ssd=args.prompt_cache_ssd,
        sampling_rank_only=args.sampling_rank_only,
        async_overlap=args.async_overlap,
        ring_connections_per_ip=args.ring_connections_per_ip,
        tuning_reason=args.tuning_reason,
    )


def _prompt_cache_ssd_dir(args: argparse.Namespace, rank: int) -> str | None:
    """Per-rank SSD directory for prompt-cache snapshots, or None when off.

    Kept beside the runtime markers and scoped by deployment and rank, so two
    deployments never read each other's snapshots and a rank only ever loads its
    own layer slice.
    """

    if not args.prompt_cache_ssd:
        return None
    root = Path(args.state_dir).expanduser() / "prompt-cache-ssd"
    return str(root / f"{args.deployment_id}-rank-{rank}")


def _install_signal_handlers() -> None:
    def interrupt(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)


def _watch_launcher_parent(
    parent_pid: int,
    marker: RuntimeMarker,
    *,
    poll_interval: float = 0.2,
    get_parent_pid: Any = os.getppid,
    wait: Any = time.sleep,
    exit_process: Any = os._exit,
    emit_event: Any = _emit_event,
) -> None:
    """Fail fast if MLX's launcher exits without reaping this rank.

    MLX-LM owns background generation threads that may keep a worker alive
    after its main thread handles SIGTERM. The launcher is the lifetime owner
    of every rank, so a reparented worker cannot make useful collective
    progress and must not remain resident on a node.
    """

    while True:
        wait(poll_interval)
        if get_parent_pid() == parent_pid:
            continue
        current_parent = get_parent_pid()
        reason = (
            f"rank launcher parent changed from {parent_pid} to "
            f"{current_parent}; the rank cannot safely continue"
        )
        # Keep the marker as bounded crash evidence. The next activation
        # overwrites the deterministic path, and liveness already ignores a
        # marker whose owner is dead. Removing it here reduced this exact
        # failure to "heartbeat missing" with no explanation.
        marker.update("launcher_lost", error=reason)
        emit_event({"type": "launcher_lost", "reason": reason})
        exit_process(1)
        return


def _peer_hosts_by_rank(
    assignments: Sequence[PipelineAssignment],
    hosts: Sequence[str],
    *,
    rank: int,
) -> dict[int, tuple[str, str]]:
    """The other ranks, addressed by SSH target. Never this one.

    Including yourself here is not a stricter check, it is a suicide pact.
    ``ClusterDeployment`` pins rank 0's SSH target to ``127.0.0.1`` and
    ``probe_peer`` short-circuits loopback to reachable, so the only signal a
    rank ever got from its own entry was the age of its own marker — and
    nothing refreshed that marker while idle. Sixty seconds after the last
    token, every healthy rank in the cluster independently decided it had lost
    itself and called ``os._exit(1)``, naming the local Mac in the message.
    """

    return {
        assignment.rank: (assignment.node_id, hosts[assignment.rank])
        for assignment in assignments
        if assignment.rank != rank and 0 <= assignment.rank < len(hosts)
    }


def _start_peer_watchdog(
    marker: RuntimeMarker,
    assignments: Sequence[PipelineAssignment],
    hosts: Sequence[str],
    deployment_id: str,
    state_dir: str,
    *,
    rank: int,
) -> PeerWatchdog | None:
    """End the job when a peer stops answering, instead of blocking forever.

    A collective has no timeout: if a peer sleeps or loses its cable, every
    surviving rank waits inside the all-reduce until killed by hand. This turns
    that into a recorded failure naming the missing Mac.

    Only rank zero runs this SSH watchdog. ``ClusterDeployment`` guarantees
    rank zero is the launcher-local process, while every SSH target in the
    worker plan is expressed from that launcher's point of view. On a remote
    rank, the coordinator's target is therefore ``127.0.0.1`` — which names
    the remote Mac itself. Treating that as the coordinator made every healthy
    remote rank look for rank zero's marker on its own disk and self-destruct
    after two polls. Remote ranks already have the launcher-parent watchdog;
    when rank zero or the launcher dies, their SSH parent disappears and that
    watchdog terminates them.

    Armed before the weights are read, not after: a 300 GB model takes twenty
    minutes to load and a peer that goes away during those twenty minutes used
    to leave every other rank blocked in the first collective with nothing
    watching.
    """

    if rank != 0:
        return None
    if len(hosts) != len(assignments) or len(hosts) < 2:
        return None
    hosts_by_rank = _peer_hosts_by_rank(assignments, hosts, rank=rank)
    if not hosts_by_rank:
        return None

    def on_lost(reason: str) -> None:
        marker.update("peer_lost", error=reason)
        _emit_event({"type": "peer_lost", "reason": reason})
        os._exit(1)

    watchdog = PeerWatchdog(
        hosts_by_rank,
        deployment_id=deployment_id,
        state_dir=state_dir,
        on_lost=on_lost,
    )
    thread = threading.Thread(
        target=watchdog.run, name="omlx-cluster-peer-watchdog", daemon=True
    )
    thread.start()
    return watchdog


def _start_launcher_watchdog(marker: RuntimeMarker, parent_pid: int) -> None:
    thread = threading.Thread(
        target=_watch_launcher_parent,
        args=(parent_pid, marker),
        name="omlx-cluster-launcher-watchdog",
        daemon=True,
    )
    thread.start()


def _runtime_assignment(
    assignment: PipelineAssignment,
) -> dict[str, Any]:
    result = {
        "node_id": assignment.node_id,
        "rank": assignment.rank,
        "start_layer": assignment.start_layer,
        "end_layer": assignment.end_layer,
        "layer_count": assignment.layer_count,
        "planned_weight_bytes": assignment.planned_weight_bytes,
        "coordinator_weight_bytes": assignment.coordinator_weight_bytes,
        "reserve_bytes": assignment.reserve_bytes,
        "capacity_bytes": assignment.capacity_bytes,
        "headroom_bytes": assignment.headroom_bytes,
        "tensor_parallel_rank": assignment.tensor_parallel_rank,
        "tensor_parallel_size": assignment.tensor_parallel_size,
        "sharded_weight_bytes": assignment.sharded_weight_bytes,
    }
    if assignment.predicted_stage_seconds is not None:
        result.update(
            {
                "predicted_compute_seconds": (assignment.predicted_compute_seconds),
                "predicted_send_seconds": assignment.predicted_send_seconds,
                "predicted_stage_seconds": assignment.predicted_stage_seconds,
            }
        )
    return result


def _validate_loaded_stage(
    model: Any,
    assignment: PipelineAssignment,
) -> None:
    """Fail closed if a model-specific pipeline hook ignored the shard plan."""

    pipeline_model = getattr(model, "model", None)
    if pipeline_model is None:
        raise RuntimeError("loaded model does not expose an MLX pipeline model")
    start = getattr(pipeline_model, "start_idx", None)
    end = getattr(pipeline_model, "end_idx", None)
    layers = getattr(pipeline_model, "layers", None)
    pure_tensor = assignment.tensor_parallel_size > 1
    expected_complete_tensor_stage = (
        pure_tensor
        and isinstance(layers, list)
        and assignment.start_layer == 0
        and assignment.end_layer == len(layers)
        and start in (None, 0)
        and end in (None, len(layers))
    )
    if (
        not expected_complete_tensor_stage
        and (start, end) != (assignment.start_layer, assignment.end_layer)
    ):
        raise RuntimeError(
            "loaded model pipeline range does not match the approved plan: "
            f"expected [{assignment.start_layer}, {assignment.end_layer}), "
            f"got [{start}, {end})"
        )
    if not isinstance(layers, list) or len(layers) != assignment.end_layer:
        raise RuntimeError("loaded model retained an unexpected pipeline layer view")
    if any(layer is not None for layer in layers[: assignment.start_layer]):
        raise RuntimeError("loaded model retained weights before its pipeline stage")
    if any(layer is None for layer in layers[assignment.start_layer :]):
        raise RuntimeError("loaded model is missing weights inside its pipeline stage")


def _loaded_stage(model: Any) -> dict[str, Any]:
    """The layer range this rank really built, read off the loaded model.

    The plan says which layers a rank should hold; only the model says which it
    does. After a crash the two were indistinguishable, because just the plan
    was ever written down — so a stage pin that silently lost to mlx-lm's even
    split looked exactly like one that worked. Values are None when the model
    cannot be read, which is still the honest answer: not "the plan".
    """

    pipeline_model = getattr(model, "model", None)
    layers = getattr(pipeline_model, "layers", None)
    return {
        "loaded_start_layer": getattr(pipeline_model, "start_idx", None),
        "loaded_end_layer": getattr(pipeline_model, "end_idx", None),
        "loaded_layer_count": (
            sum(1 for layer in layers if layer is not None)
            if isinstance(layers, list)
            else None
        ),
    }


def _measured_weight_bytes(model: Any) -> int | None:
    """Sum the resident parameter bytes, or None if they cannot be read.

    The planner predicts what a rank should hold; this is what it actually holds
    after ``shard()`` ran. Publishing both in the ready marker is the cheapest
    detector for a planner and a sharding implementation that disagree — a
    silent class of bug, since wrong-but-plausible shapes still generate text.
    """

    try:
        from mlx.utils import tree_flatten

        parameters = model.parameters()
    except Exception:
        return None

    total = 0
    for _name, value in tree_flatten(parameters):
        nbytes = getattr(value, "nbytes", None)
        if isinstance(nbytes, int):
            total += nbytes
    return total or None


_MEASURED_WEIGHT_RELATIVE_TOLERANCE_PERCENT = 5
_MEASURED_WEIGHT_ABSOLUTE_TOLERANCE_BYTES = 64 * 1024**2


def _validate_measured_weight_bytes(
    measured_weight_bytes: int | None,
    assignment: PipelineAssignment,
) -> None:
    """Fail closed when the loaded parameter tree exceeds the approved shard.

    ``planned_weight_bytes`` also includes KV cache, while ``model.parameters``
    measures only weights. Compare like with like and allow the larger of 5%
    or 64 MiB for format metadata, tied parameters, and quantizer bookkeeping.
    Models whose parameter tree cannot be inspected remain fail-soft.
    """

    if measured_weight_bytes is None:
        return
    approved_parameter_bytes = (
        assignment.fixed_weight_bytes
        + assignment.coordinator_weight_bytes
        + assignment.layer_weight_bytes
    )
    relative_tolerance = (
        approved_parameter_bytes * _MEASURED_WEIGHT_RELATIVE_TOLERANCE_PERCENT + 99
    ) // 100
    tolerance = max(
        _MEASURED_WEIGHT_ABSOLUTE_TOLERANCE_BYTES,
        relative_tolerance,
    )
    limit = approved_parameter_bytes + tolerance
    if measured_weight_bytes > limit:
        raise RuntimeError(
            "loaded model retains more parameter memory than the approved shard: "
            f"measured {measured_weight_bytes} bytes, approved "
            f"{approved_parameter_bytes} bytes plus {tolerance} bytes tolerance"
        )


def _unpinned_stage(layer_count: int, rank: int, world_size: int) -> tuple[int, int]:
    """The split mlx-lm would take if nothing pinned this rank's stage.

    ``effective_stage`` short-circuits on the pin — that is the whole point of
    it everywhere else — so asking it while the pin is set only ever returns the
    pin. The guard below needs the opposite: what happens if the pin does *not*
    take effect. So the pin is lifted for the length of the question and put
    back, which also makes the answer independent of where in ``run_worker``
    this is called from.
    """

    from omlx.patches.minimax_m3_mlx_lm.pipeline_patch import (
        assigned_stage,
        clear_assigned_stage,
        effective_stage,
        set_assigned_stage,
    )

    pinned = assigned_stage()
    clear_assigned_stage()
    try:
        return effective_stage(layer_count, rank, world_size)
    finally:
        if pinned is not None:
            set_assigned_stage(*pinned)


def _guard_effective_stage(
    assignment: Any,
    assignments: Sequence[Any],
    *,
    rank: int,
    world_size: int,
    role: str = "",
    memory_guard_tier: str = "balanced",
    safety: float | None = None,
    assignment_honored: bool = False,
) -> None:
    """Refuse the stage mlx-lm will really load, not the one that was planned.

    ``PipelineMixin`` derives the split from rank and world size alone, so a
    plan's uneven split — the mechanism that keeps a workstation usable — is
    discarded unless something pins it. A MacBook planned for 14 layers
    (56 GiB) loaded 30 layers (112 GiB) against a 109 GiB limit and died,
    because the guard checked the plan and the plan was not what ran.

    A loader hook carrying the explicit unequal-assignment contract is allowed
    to use the planned range. Otherwise the range is recomputed the way the
    loader will compute it *without* the pin, and the guard runs against that.
    Asking the pinned value or trusting the mere existence of ``pipeline`` is
    asking the mechanism whose failure this exists to survive:
    ``DeepseekV32Model`` defines its own method but ignores the assignment.
    """

    if assignment_honored:
        return

    layer_count = max((int(item.end_layer) for item in assignments), default=0)
    if layer_count <= 0:
        return
    start, end = _unpinned_stage(layer_count, rank, world_size)
    planned_layers = max(1, assignment.end_layer - assignment.start_layer)
    effective_layers = max(0, end - start)
    if effective_layers <= planned_layers:
        return

    # Layers are near-uniform, so scaling the planned weight is a good enough
    # estimate for a guard — and erring high is the safe direction here.
    per_layer = assignment.layer_weight_bytes / planned_layers
    effective_bytes = int(
        assignment.fixed_weight_bytes
        + assignment.coordinator_weight_bytes
        + per_layer * effective_layers
        + assignment.kv_cache_bytes
    )
    # The worker writes to stdout, which the launcher captures per rank.
    print(
        f"rank {rank} was planned {planned_layers} layers but the loader will "
        f"take {effective_layers} ({effective_bytes / 1024**3:.1f} GiB, not "
        f"{assignment.planned_weight_bytes / 1024**3:.1f} GiB) — guarding "
        f"against what will load",
        flush=True,
    )
    # Same role and tier as the admission check above it, or this second gate
    # silently admits at the 0.90 headless default on a Mac someone is using.
    check_rank_fits(
        effective_bytes,
        rank=rank,
        node_id=getattr(assignment, "node_id", ""),
        role=role,
        safety=safety,
        memory_guard_tier=memory_guard_tier,
    )


def _resolve_setting(
    args: argparse.Namespace,
    assignment: Any,
    *,
    flag: str,
    field: str,
    default: str = "",
) -> str:
    """A per-rank setting, from the flag if given and the plan otherwise.

    ``mlx.launch`` runs one identical argv on every host, so a flag cannot say
    "the Studio is headless and the MacBook is not" — the two settings that
    decide how much of a Mac a rank may take were therefore unreachable in
    every real launch, and every rank admitted at the headless default. The
    plan is already per-rank, already signed into the plan hash, and already
    indexed by rank here, so it is the channel that can carry them.

    ``field`` is read with ``getattr`` because the plan may predate it; an
    older plan simply falls through to the default rather than failing to
    launch.

    The flag stays as an override for running a rank by hand.
    """

    from_flag = str(getattr(args, flag, "") or "").strip()
    if from_flag:
        return from_flag
    from_plan = str(getattr(assignment, field, "") or "").strip()
    return from_plan or default


def _apply_rank_wired_limit(budget_bytes: int) -> int:
    """Wire at most what this rank was admitted for, and only if asked to.

    The rank used to raise MLX's wired limit to Apple's
    ``max_recommended_working_set_size`` — 107.5 GiB of a 128 GiB MacBook —
    unconditionally, before any guard had run. Wired memory is not pageable, so
    that is the difference between "the model dies" and "the Mac becomes
    unusable", and it was applied role-blind, tier-blind and with no OS margin.

    Single-node oMLX takes the opposite decision on the same machine:
    ``_apply_metal_wired_limit`` leaves Apple's default cap alone entirely when
    ``iogpu.wired_limit_mb`` is unset, on the grounds that MLX allocator state
    should not change unless the user explicitly raised the kernel cap. A rank
    is not entitled to more of someone's Mac than the engine is, so it reuses
    that function rather than keeping a second, laxer, copy of the rule.

    The target is the rank's own admission budget, not Apple's recommendation:
    wiring memory this rank has already been refused permission to use protects
    nothing. Returns the bytes actually applied — 0 when nothing was changed.
    """

    if budget_bytes <= 0:
        # No measurable ceiling means no admission decision was made, so there
        # is no number here worth wiring against. Leave the allocator alone.
        return 0
    try:
        from omlx.process_memory_enforcer import _apply_metal_wired_limit

        applied, _previous = _apply_metal_wired_limit(int(budget_bytes))
    except Exception as exc:  # pragma: no cover - defensive
        # A worker-only install may not have the enforcer. Not raising the
        # limit is the safe outcome; failing the rank over it is not.
        print(
            f"could not set the Metal wired limit ({type(exc).__name__}: {exc}); "
            f"leaving the default cap in place",
            flush=True,
        )
        return 0
    return int(applied or 0)


def run_worker(args: argparse.Namespace) -> int:
    """Initialize one strict rank, load only its layers, then serve on rank zero."""

    from omlx._torch_stub import install as install_torch_stub

    install_torch_stub()

    import mlx.core as mx
    import mlx_lm.server as mlx_server
    from mlx_lm.server import ModelProvider, ResponseGenerator, run

    from omlx.patches.mlx_lm_pipeline_index import (
        apply_mlx_lm_pipeline_index_patch,
    )
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    # The model is intentionally loaded before the HTTP server is started so
    # readiness can validate resident weights. Keep that eager load and
    # MLX-LM's later generation thread on a single cross-thread stream.
    generation_stream = _cross_thread_generation_stream(mx)

    plan_hash, assignments, performance_profiles, tensor_parallel_size = (
        decode_worker_contract(args.plan)
    )
    execution = _execution_settings(args)
    init_backend = "jaccl" if args.backend.startswith("jaccl") else "ring"
    group = mx.distributed.init(backend=init_backend, strict=True)
    rank = group.rank()
    world_size = group.size()
    try:
        model_config = json.loads(
            (Path(args.model).expanduser() / "config.json").read_text()
        )
    except OSError:
        # ModelProvider remains the authority for ordinary text paths (and
        # test doubles may use a repo ID rather than a local directory). The
        # staged vision path always has its config sidecar locally.
        model_config = {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"distributed model config is unreadable: {exc}") from exc
    from omlx.deepseek_v4_vision import is_deepseek_v4_vision_config
    from omlx.model_discovery import _has_vision_subconfig

    deepseek_v4_vision = is_deepseek_v4_vision_config(model_config)
    if _has_vision_subconfig(model_config) and not deepseek_v4_vision:
        raise RuntimeError(
            "unsupported distributed VLM family; only "
            "DeepSeek-V4-Flash-Vision is enabled"
        )
    if world_size != len(assignments):
        raise RuntimeError(
            f"runtime world size {world_size} does not match plan size "
            f"{len(assignments)}"
        )
    if args.plan_hash != plan_hash:
        raise RuntimeError("worker plan hash does not match launch contract")

    marker = RuntimeMarker(
        state_dir=args.state_dir,
        deployment_id=args.deployment_id,
        rank=rank,
        world_size=world_size,
        model=args.model,
        backend=args.backend,
        plan_hash=plan_hash,
    )
    launcher_parent_pid = os.getppid()
    marker.update(
        "loading",
        load_stage="initializing",
        model_family=("deepseek_v4_vision" if deepseek_v4_vision else "text"),
        vision_owner=(rank == 0 if deepseek_v4_vision else None),
        launcher_parent_pid=launcher_parent_pid,
    )
    marker.start_heartbeat()
    _install_signal_handlers()
    _start_launcher_watchdog(marker, launcher_parent_pid)
    # Before the load, not after it. Loading a 300 GB model takes twenty
    # minutes, and a peer that goes away inside that window left every other
    # rank blocked in its first collective with nothing watching at all.
    _start_peer_watchdog(
        marker,
        assignments,
        [host for host in args.peer_hosts.split(",") if host],
        args.deployment_id,
        args.state_dir,
        rank=rank,
    )

    preserve_failure_marker = False
    try:
        # Register oMLX's model classes before MLX-LM resolves the
        # architecture. Several types oMLX serves — glm_moe_dsa, deepseek_v4,
        # minimax_m3, step3p7, llama4 — do not exist in stock MLX-LM; oMLX
        # injects them here on its single-node path. Without this a rank fails
        # with "Pipeline loading is only supported for MLX converted models" or
        # a wall of missing indexer weights, after the model has been staged
        # and partly loaded. Same call the LLM engine makes.
        # Refuse a stage this Mac cannot hold, using the same admission
        # ceiling the single-node engine pool admits against. Checked *before*
        # any weights are read: without it a rank loads until the OS gives up,
        # taking the machine with it.
        assignment = assignments[rank]
        # Role- and tier-aware: a Mac someone is working on admits at a lower
        # fraction than a headless one, and the guard — not the plan — is what
        # actually stops a load. Both come from this rank's own assignment,
        # because one shared argv cannot say different things to different Macs.
        node_role = _resolve_setting(args, assignment, flag="node_role", field="role")
        guard_tier = _resolve_setting(
            args,
            assignment,
            flag="memory_guard_tier",
            field="memory_guard_tier",
            default="balanced",
        )
        admission_ceiling = guard_rank_load(
            assignment,
            rank=rank,
            role=node_role,
            memory_guard_tier=guard_tier,
        )
        manual_safety = assignment_memory_safety(assignment)
        load_budget = admission_budget(
            admission_ceiling,
            role=node_role,
            safety=manual_safety,
        )
        # Only now, and only this far. Raising the wired limit is what turns a
        # model that dies into a Mac that cannot be typed on, so it happens
        # after something has decided this rank may have the memory at all —
        # and it asks for the budget, not for everything Apple would allow.
        wired_limit = _apply_rank_wired_limit(load_budget)
        marker.update(
            "loading",
            load_stage="loading_weights",
            node_role=node_role,
            memory_guard_tier=guard_tier,
            admission_ceiling_bytes=admission_ceiling,
            admission_budget_bytes=load_budget,
            wired_limit_bytes=wired_limit,
            planned_weight_bytes=assignment.planned_weight_bytes,
            capacity_bytes=assignment.capacity_bytes,
            reserve_bytes=assignment.reserve_bytes,
            headroom_bytes=assignment.headroom_bytes,
            assignments=[_runtime_assignment(item) for item in assignments],
        )

        maybe_apply_pre_load_patches(args.model)
        # MLX-LM's pipeline shard selection rejects any parameter absent
        # from the safetensors index, though it loads with strict=False
        # moments later. Architectures oMLX patches in (glm_moe_dsa's
        # indexer weights) legitimately have such parameters.
        apply_mlx_lm_pipeline_index_patch()

        # Pin this rank's stage so the loader honours the plan instead of an
        # even split. Must precede load: mlx-lm builds the model and calls
        # pipeline() inside load_default().
        from omlx.patches.minimax_m3_mlx_lm.pipeline_patch import (
            set_assigned_stage,
        )

        with install_pipeline_compatibility(assignments):
            # Compatibility hooks are installed before asking about support,
            # so the answer describes the exact methods load_default() will
            # call. Unknown and custom unmarked methods remain fail-closed.
            _guard_effective_stage(
                assignment,
                assignments,
                rank=rank,
                world_size=world_size,
                role=node_role,
                memory_guard_tier=guard_tier,
                safety=manual_safety,
                assignment_honored=(
                    tensor_parallel_size > 1
                    or pipeline_assignment_is_honored(args.model)
                ),
            )
            set_assigned_stage(assignment.start_layer, assignment.end_layer)
            provider = ModelProvider(
                _server_arguments(
                    args,
                    tensor_parallel_size=tensor_parallel_size,
                )
            )
            # Load synchronously on every rank so a "ready" event means the
            # complete distributed graph and weights are resident.
            #
            # Admission was a prediction about a stage nobody had read yet, so
            # it is checked again against what the rank actually holds, all the
            # way through the load. A rank that overruns dies here rather than
            # taking the Mac with it.
            with watch_rank_load(
                load_budget,
                rank=rank,
                node_id=getattr(assignment, "node_id", ""),
                on_sample=lambda observed: marker.update(
                    "loading",
                    load_stage="loading_weights",
                    load_memory_bytes=observed,
                ),
            ), install_progressive_loader(
                mlx_server,
                progress=lambda progress: marker.update(
                    "loading",
                    load_stage=progress.get("phase", "materializing_layers"),
                    **{
                        key: value
                        for key, value in progress.items()
                        if key != "phase"
                    },
                ),
            ):
                provider.load_default()
            if deepseek_v4_vision:
                provider.is_batchable = False
            protocol = _install_distributed_model_protocol(
                provider.tokenizer,
                args.model,
            )
            # Recorded before validation, so a marker read after a stage
            # mismatch still says which layers were built.
            marker.update(
                "loading",
                load_stage="validating",
                output_protocol=protocol or None,
                **_loaded_stage(provider.model),
            )
            # Validate the loaded pipeline stage before any TP mutation
            _validate_loaded_stage(provider.model, assignment)
            # Pure TP was applied layer-by-layer by the progressive loader.
            # Doing it here as well would shard every projection twice.
            measured_weight_bytes = _measured_weight_bytes(provider.model)
            _validate_measured_weight_bytes(measured_weight_bytes, assignment)
            with install_runtime_optimizations(
                provider.model,
                group,
                execution,
                batchable=provider.is_batchable,
                pipeline_parallel=tensor_parallel_size == 1,
            ) as optimizations:
                vision_runtime = nullcontext()
                if deepseek_v4_vision:
                    from .deepseek_v4_vision_runtime import (
                        install_deepseek_v4_vision_runtime,
                    )

                    vision_runtime = install_deepseek_v4_vision_runtime(
                        mlx_server,
                        provider,
                        config=model_config,
                        rank=rank,
                    )
                marker.update(
                    "ready",
                    load_stage="ready",
                    start_layer=assignment.start_layer,
                    end_layer=assignment.end_layer,
                    planned_weight_bytes=assignment.planned_weight_bytes,
                    measured_weight_bytes=measured_weight_bytes,
                    capacity_bytes=assignment.capacity_bytes,
                    reserve_bytes=assignment.reserve_bytes,
                    headroom_bytes=assignment.headroom_bytes,
                    kv_cache_scope="rank_local",
                    assignments=[_runtime_assignment(item) for item in assignments],
                    execution=execution.to_dict(),
                    performance_profiles=[
                        profile.to_dict() for profile in performance_profiles
                    ],
                    optimizations=optimizations,
                    output_protocol=protocol or None,
                )
                _emit_event(
                    {
                        "type": "rank_ready",
                        "protocol_version": 1,
                        "deployment_id": args.deployment_id,
                        "plan_hash": plan_hash,
                        "rank": rank,
                        "node_id": assignment.node_id,
                        "world_size": world_size,
                        "start_layer": assignment.start_layer,
                        "end_layer": assignment.end_layer,
                        "planned_weight_bytes": assignment.planned_weight_bytes,
                        "measured_weight_bytes": measured_weight_bytes,
                        "capacity_bytes": assignment.capacity_bytes,
                        "reserve_bytes": assignment.reserve_bytes,
                        "headroom_bytes": assignment.headroom_bytes,
                    }
                )
                if rank == 0:
                    _emit_event(
                        {
                            "type": "ready",
                            "protocol_version": 1,
                            "deployment_id": args.deployment_id,
                            "plan_hash": plan_hash,
                            "rank": rank,
                            "world_size": world_size,
                            "port": args.port,
                            "optimizations": optimizations,
                        }
                    )
                # Vision must patch the base generator before telemetry creates
                # its subclass. Then all ranks finish sharing the expanded
                # prompt before telemetry enters cache/guard collectives.
                with (
                    vision_runtime,
                    install_server_telemetry(
                        marker,
                        execution=execution,
                        assignment=assignment,
                        ssd_cache_dir=_prompt_cache_ssd_dir(args, rank),
                        prefill_step_size=args.prefill_step_size,
                        prefill_guard=build_guard(
                            provider.model,
                            rank=rank,
                            node_id=getattr(assignment, "node_id", ""),
                            layer_count=(
                                assignment.end_layer - assignment.start_layer
                            ),
                            tensor_parallel_size=assignment.tensor_parallel_size,
                            memory_guard_tier=guard_tier,
                            # The attention peak is set by the prefill chunk, so the
                            # guard has to size it from the step this server will
                            # really use, not from mlx-lm's 2048 default.
                            prefill_step_size=args.prefill_step_size,
                        ),
                    ),
                    _bind_generation_thread_stream(
                        ResponseGenerator,
                        mx,
                        generation_stream,
                    ),
                    _install_thinking_budget_support(
                        mlx_server,
                        provider.tokenizer,
                    ),
                ):
                    # install_server_telemetry also runs the idle heartbeat for
                    # the length of this block, so "stale" means stalled rather
                    # than "nobody has asked anything for 45 seconds".
                    run("127.0.0.1", args.port, provider)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        # An ordinary exception is the most useful evidence a failed remote
        # rank can leave. Removing this marker in ``finally`` turned model,
        # memory and MLX failures into a generic rank-zero socket close.
        preserve_failure_marker = True
        with suppress(Exception):
            marker.update(
                "failed",
                error=f"{type(exc).__name__}: {exc}"[:1000],
            )
        raise
    finally:
        marker.stop_heartbeat()
        if not preserve_failure_marker:
            marker.remove()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Internal oMLX inference rank")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--backend", choices=("ring", "jaccl", "jaccl-ring"), required=True
    )
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--deployment-id", required=True)
    parser.add_argument("--plan-hash", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument(
        "--node-role",
        default="",
        help=(
            '"headless" or "workstation". Decides the fraction of this '
            "Mac's memory a rank may ever be admitted at — a machine someone "
            "is using keeps a third of it back. Overrides the role carried in "
            "the plan; leave unset for a normal launch, because one shared "
            "argv cannot give two Macs different roles."
        ),
    )
    parser.add_argument(
        "--memory-guard-tier",
        default="",
        choices=("", "safe", "balanced", "aggressive", "custom"),
        help=(
            "Which admission ceiling this rank measures against, matching the "
            "single-node memory guard tier. Overrides the tier carried in the "
            "plan. A user who capped this Mac and got a rank admitting 67 GiB "
            "above the cap is the reason this is plumbed at all."
        ),
    )
    parser.add_argument(
        "--peer-hosts",
        default="",
        help=(
            "Comma-separated SSH targets in rank order. Used only to notice a "
            "peer going away; a collective otherwise blocks forever."
        ),
    )
    parser.add_argument(
        "--state-dir",
        default="~/.omlx/cluster/runtime",
        help="Local state directory shown by the oMLX GUI on each node",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--execution-profile",
        choices=("interactive", "balanced", "throughput"),
        default="balanced",
    )
    parser.add_argument("--decode-concurrency", type=int, default=8)
    parser.add_argument("--prompt-concurrency", type=int, default=4)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--prompt-cache-size", type=int, default=4)
    parser.add_argument("--prompt-cache-bytes", type=int)
    parser.add_argument("--max-kv-size", type=int)
    parser.add_argument("--pipeline-microbatch-size", type=int, default=4)
    parser.add_argument("--auto-tune", action="store_true")
    parser.add_argument("--cache-affinity", action="store_true")
    parser.add_argument("--prompt-cache-ssd", action="store_true")
    parser.add_argument("--sampling-rank-only", action="store_true")
    parser.add_argument("--async-overlap", action="store_true")
    parser.add_argument("--ring-connections-per-ip", type=int, default=1)
    parser.add_argument(
        "--tuning-reason",
        default="balanced profile defaults",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # One shared argv launches every rank, so --model must be portable across
    # Macs with different $HOME. Expand ~ in this rank's own home; a coordinator
    # absolute path is left unchanged.
    args.model = str(Path(args.model).expanduser())
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    for name in (
        "decode_concurrency",
        "prompt_concurrency",
        "prefill_step_size",
        "prompt_cache_size",
        "pipeline_microbatch_size",
        "ring_connections_per_ip",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    for name in ("prompt_cache_bytes", "max_kv_size"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.prompt_concurrency > args.decode_concurrency:
        raise SystemExit("--prompt-concurrency cannot exceed decode concurrency")
    if args.pipeline_microbatch_size > args.decode_concurrency:
        raise SystemExit("--pipeline-microbatch-size cannot exceed decode concurrency")
    if args.ring_connections_per_ip > 32:
        raise SystemExit("--ring-connections-per-ip cannot exceed 32")
    if not args.tuning_reason or len(args.tuning_reason) > 1000:
        raise SystemExit("--tuning-reason is invalid")
    try:
        return run_worker(args)
    except Exception as exc:
        print(
            f"oMLX distributed inference worker failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
