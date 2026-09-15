# SPDX-License-Identifier: Apache-2.0
"""Publish committed Qwen4 generation history through the native prefix cache.

Only text-only scheduler timelines are admitted. The cache owns hashing,
capacity, eviction and live-backbone validation; this module adds no disk format.
"""

from __future__ import annotations

import logging
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from . import prompt_priming as priming

logger = logging.getLogger(__name__)
_PLANS = "_omlx_generated_prefix_plans"
_MAX_PLANS = 4096
_STATE_PLAN = "_omlx_generated_prefix_plan"
_BOUNDARY = "_omlx_generated_prefix_boundary"


@dataclass(frozen=True)
class _Plan:
    cache: Any
    tokens: tuple[int, ...]
    block_size: int


def register(model, uid, request, prefix_cache):
    """Bind a text request to its scheduler UID before generation starts."""
    unregister(model, uid)
    host = priming._eligible_host(model)
    if (
        host is None
        or getattr(host, "model_type", None) not in ("qwen4_exp", "qwen4_exp_text")
        or not priming.priming_enabled()
        or prefix_cache is None
        or any(
            getattr(request, attr, None) is not None
            for attr in (
                "vlm_extra_keys_for_cache",
                "vlm_extra_key_token_start_for_cache",
                "vlm_extra_key_ranges_for_cache",
            )
        )
    ):
        return
    block = int(getattr(prefix_cache, "block_size", 0) or 0)
    if block <= 1 or not callable(
        getattr(prefix_cache, "store_mtp_prefix_snapshot", None)
    ):
        return
    try:
        plan = _Plan(weakref.ref(prefix_cache), tuple(request.prompt_token_ids), block)
    except TypeError:
        return
    plans = getattr(host, _PLANS, None)
    if plans is None:
        plans = OrderedDict()
        setattr(host, _PLANS, plans)
    plans[uid] = plan
    while len(plans) > _MAX_PLANS:
        plans.popitem(last=False)


def unregister(model, uid=None):
    """Release unused plans on completion, cancellation or generator reset."""
    for host in priming._host_candidates(model):
        plans = getattr(host, _PLANS, None)
        if plans is not None:
            if uid is None:
                plans.clear()
            else:
                plans.pop(uid, None)


def record(gen_batch, state, normed_hidden):
    """Remember the pending row only when a committed fold crosses B-1."""
    setattr(state, _BOUNDARY, None)
    try:
        if not hasattr(state, _STATE_PLAN):
            plan = None
            if len(gen_batch.uids) == 1 and state.uid == gen_batch.uids[0]:
                for host in priming._host_candidates(gen_batch.model):
                    plans = getattr(host, _PLANS, {})
                    plan = plans.pop(state.uid, None)
                    if plan is not None:
                        break
            if (
                plan is None
                or len(gen_batch.tokens) != 1
                or tuple(gen_batch.tokens[0]) != plan.tokens
                or state.hist_offset != len(plan.tokens) + 1
            ):
                plan = None
            setattr(state, _STATE_PLAN, plan)
        plan = getattr(state, _STATE_PLAN)
        if plan is None:
            return
        boundary = int(state.hist_offset)
        align = int(getattr(gen_batch.model, "_omlx_mtp_commit_align", 0) or 0)
        if (
            align != plan.block_size
            or boundary <= len(plan.tokens)
            or boundary % align != 0
            or normed_hidden.ndim != 3
            or normed_hidden.shape[0] != 1
            or normed_hidden.shape[1] == 0
        ):
            return
        # The fold pairs hidden[B-1] with token[B]; a snapshot at B keeps
        # only the first B-1 pairs and this one pending hidden row.
        setattr(state, _BOUNDARY, (boundary, normed_hidden[:, -1:] + 0))
    except Exception as exc:
        logger.debug("Generated MTP boundary capture declined: %s", type(exc).__name__)


def candidate(gen_batch, token_id):
    """Detach a snapshot before emission can clear a terminal batch."""
    try:
        state = getattr(gen_batch, "_omlx_mtp_state", None)
        plan = getattr(state, _STATE_PLAN, None)
        tagged = getattr(state, _BOUNDARY, None)
        if plan is None or tagged is None or len(gen_batch.tokens) != 1:
            return None
        if len(gen_batch.uids) != 1 or state.uid != gen_batch.uids[0]:
            return None
        boundary = len(gen_batch.tokens[0]) + 1
        if tagged[0] != boundary:
            return None
        setattr(state, _BOUNDARY, None)
        cache = plan.cache()
        if cache is None:
            return None
        detached = priming._cache_at_offset(state.mtp_cache, boundary - 1)
        if detached is None:
            return None
        snapshot = priming._MtpPrefixSnapshot(boundary, detached, tagged[1])
        # Resolve the detached arrays now so a retained snapshot cannot hold
        # the live model/weight graph. This runs once per full block only.
        import mlx.core as mx

        mx.eval(*priming._snapshot_arrays(snapshot))
        ledger = list(gen_batch.tokens[0]) + [int(token_id)]
        return plan.cache, ledger, snapshot
    except Exception as exc:
        logger.debug("Generated MTP snapshot declined: %s", type(exc).__name__)
        return None


def publish(pending):
    """Publish only after the ordinary emission path completed successfully."""
    if pending is None:
        return
    cache_ref, tokens, snapshot = pending
    cache = cache_ref()
    if cache is None:
        return
    try:
        cache.store_mtp_prefix_snapshot(tokens, snapshot.boundary_tokens, snapshot)
    except Exception as exc:
        logger.debug("Generated MTP snapshot store declined: %s", type(exc).__name__)
