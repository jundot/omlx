# SPDX-License-Identifier: Apache-2.0
"""Refuse a prompt that would OOM a rank, before the collective starts.

``memory_guard`` stops a rank *loading* a stage it cannot hold. That is only
half the problem: a stage that loads fine can still be killed by one long
prompt, because prefill allocates KV for every token plus a transient
attention spike. Single-node oMLX already refuses those requests up front —
``raise_if_prefill_exceeds`` is the shared front door, used by engines that
have no ``Scheduler`` — and this gives a cluster rank the same door.

Two things make the cluster case different from single-node.

**A rank holds a slice, not the model.** ``set_model_info_from_model`` reads
``num_hidden_layers`` from the config, which is the whole model. A pipeline
rank holding 22 of 92 layers would be charged 4x its real KV growth and reject
prompts it could serve. So the extracted dims are corrected: KV layers to the
stage's layer count, attention heads to this rank's shard.

**A raise must not desync the collective.** Every rank checks its own slice and
then contributes a rejection vote before the first ``model()`` call. If any
slice would exceed its local ceiling, every rank leaves the request together;
otherwise every rank enters the model collectives together.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# mlx-lm's server prefills in 2048-token chunks; the transient attention peak
# is set by the chunk, not the prompt.
_DEFAULT_PREFILL_STEP = 2048


def rank_monitor(
    model: Any,
    *,
    layer_count: int = 0,
    tensor_parallel_size: int = 1,
) -> Any | None:
    """A ``MemoryMonitor`` calibrated to the slice this rank actually holds.

    Returns None if the model's dimensions could not be read, which makes the
    guard a no-op rather than a source of spurious rejections — the same
    best-effort contract ``set_model_info_from_model`` documents.
    """

    from omlx.memory_monitor import MemoryMonitor, set_model_info_from_model

    monitor = MemoryMonitor(max_kv_cache_memory=None, eviction_enabled=False)
    try:
        set_model_info_from_model(monitor, model)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not read model dims for the prefill guard: %s", exc)
        return None

    num_layers = int(getattr(monitor, "_num_layers", 0) or 0)
    kv_heads = int(getattr(monitor, "_num_kv_heads", 0) or 0)
    head_dim = int(getattr(monitor, "_head_dim", 0) or 0)
    if not kv_heads or not head_dim:
        return None

    heads = int(getattr(monitor, "_num_attention_heads", 0) or kv_heads)
    dtype_size = float(getattr(monitor, "_dtype_size", 2) or 2)
    kv_override = getattr(monitor, "_kv_bytes_per_token_override", None)

    # Pipeline: this rank stores KV for its own layers only.
    stage_layers = int(layer_count) if layer_count else num_layers
    stage_layers = max(1, min(stage_layers, num_layers or stage_layers))

    # Tensor parallel: heads are split across ranks, so both the KV this rank
    # stores and the attention transient it computes shrink with the shard.
    tp = max(1, int(tensor_parallel_size))
    if tp > 1:
        kv_heads = max(1, kv_heads // tp)
        heads = max(1, heads // tp)
        if kv_override:
            kv_override = float(kv_override) / tp

    monitor.set_model_info(
        num_layers=num_layers or stage_layers,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        dtype_size=dtype_size,
        num_attention_heads=heads,
        num_kv_cache_layers=stage_layers,
        kv_bytes_per_token=kv_override,
    )
    return monitor


class RankPrefillGuard:
    """Reject a prompt this rank cannot prefill, with a reason.

    Disabled (``active`` False) when the model's dims are unreadable or the
    host reports no ceiling — matching the single-node rule that an
    unmeasurable machine is never blocked, only an over-large request is.
    """

    def __init__(
        self,
        monitor: Any | None,
        *,
        rank: int = 0,
        node_id: str = "",
        ceiling_bytes: int = 0,
        prefill_step_size: int = _DEFAULT_PREFILL_STEP,
    ) -> None:
        self._monitor = monitor
        self._rank = int(rank)
        self._node_id = node_id
        self._ceiling = max(0, int(ceiling_bytes))
        self._step = max(1, int(prefill_step_size))

    @property
    def active(self) -> bool:
        return self._monitor is not None and self._ceiling > 0

    def check(
        self,
        num_prompt_tokens: int,
        *,
        cached_tokens: int = 0,
        request_id: str | None = None,
        current_usage_bytes: int | None = None,
        prefill_step_size: int | None = None,
    ) -> None:
        """Raise ``PrefillMemoryExceededError`` if this prompt will not fit."""

        if not self.active:
            return

        from omlx.memory_monitor import raise_if_prefill_exceeds

        from .memory_guard import current_usage_bytes as measure_usage

        usage = (
            measure_usage()
            if current_usage_bytes is None
            else max(0, int(current_usage_bytes))
        )
        step = self._step if prefill_step_size is None else max(1, int(prefill_step_size))
        try:
            raise_if_prefill_exceeds(
                self._monitor,
                prefill_memory_guard=True,
                hard_limit_bytes=self._ceiling,
                current_usage_bytes=usage,
                prefill_step_size=step,
                num_prompt_tokens=int(num_prompt_tokens),
                cached_tokens=max(0, int(cached_tokens)),
                request_id=request_id,
            )
        except Exception as exc:
            from omlx.exceptions import PrefillMemoryExceededError

            if not isinstance(exc, PrefillMemoryExceededError):
                raise
            where = f"rank {self._rank}" + (
                f" ({self._node_id})" if self._node_id else ""
            )
            logger.warning("Cluster prefill rejected on %s: %s", where, exc)
            raise

    def check_collective(
        self,
        num_prompt_tokens: int,
        *,
        cached_tokens: int = 0,
        request_id: str | None = None,
        current_usage_bytes: int | None = None,
        mx_module: Any | None = None,
        prefill_step_size: int | None = None,
    ) -> None:
        """Make prefill admission one rank-agreed decision.

        The request has already been broadcast by MLX-LM when its prompt cache
        is consulted. A local raise at that point lets peer ranks continue into
        the model collective and hang. Instead, every rank measures its own
        resident slice, exchanges a one-hot rejection vote, and raises before
        model execution if any rank refused the prompt.
        """

        from omlx.exceptions import PrefillMemoryExceededError

        local_error: PrefillMemoryExceededError | None = None
        try:
            self.check(
                num_prompt_tokens,
                cached_tokens=cached_tokens,
                request_id=request_id,
                current_usage_bytes=current_usage_bytes,
                prefill_step_size=prefill_step_size,
            )
        except PrefillMemoryExceededError as exc:
            local_error = exc

        if mx_module is None:
            import mlx.core as collective_mx
        else:
            collective_mx = mx_module

        group = collective_mx.distributed.init()
        world_size = int(group.size())
        if world_size <= 1:
            if local_error is not None:
                raise local_error
            return

        rank = int(group.rank())
        votes = [0] * world_size
        if local_error is not None:
            votes[rank] = 1
        agreed_votes = collective_mx.distributed.all_sum(
            collective_mx.array(votes)
        ).tolist()
        rejecting_ranks = [
            index for index, rejected in enumerate(agreed_votes) if int(rejected)
        ]
        if not rejecting_ranks:
            return
        if local_error is not None:
            raise local_error

        rejecting = rejecting_ranks[0]
        raise PrefillMemoryExceededError(
            message=(
                f"Cluster prefill rejected by rank {rejecting}: its local model "
                "slice would exceed the host memory limit. Reduce context length "
                "or free memory on that node."
            ),
            request_id=request_id,
        )


def build_guard(
    model: Any,
    *,
    rank: int,
    node_id: str = "",
    layer_count: int = 0,
    tensor_parallel_size: int = 1,
    memory_guard_tier: str = "balanced",
    prefill_step_size: int = _DEFAULT_PREFILL_STEP,
) -> RankPrefillGuard:
    """The guard for a loaded rank, using this Mac's own admission ceiling."""

    from .memory_guard import ceiling_breakdown

    try:
        ceiling = int(ceiling_breakdown(memory_guard_tier).get("hard_limit", 0))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("No admission ceiling available for the prefill guard: %s", exc)
        ceiling = 0

    return RankPrefillGuard(
        rank_monitor(
            model,
            layer_count=layer_count,
            tensor_parallel_size=tensor_parallel_size,
        ),
        rank=rank,
        node_id=node_id,
        ceiling_bytes=ceiling,
        prefill_step_size=prefill_step_size,
    )
