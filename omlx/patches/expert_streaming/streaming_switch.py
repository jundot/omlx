# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F401
"""Streaming MoE switch layers with per-expert LRU cache.

Implements a drop-in replacement for SwitchLinear / QuantizedSwitchLinear and
SwitchGLU that keeps a bounded number of experts resident as mx.arrays and
faults the rest from the SSD-backed ExpertBackingStore (or an in-RAM dict for
tests).  The budget is a total byte budget across all MoE layers; the cache
is global per model.

Compatibility façade: the implementation now lives in the leaf modules
``expert_cache`` (cache stats, the admission worker, the LRU store and
its policy factory), ``bank_io`` (the shared preadv pool, the np -> MLX
promotion rule, the call-shape heuristics and the per-layer load
context) and ``streaming_layers`` (the SwitchLinear variants, the shared
routing plan and StreamingSwitchGLU). Every name this module defined or
imported stays importable here so ``from .streaming_switch import X``
keeps working; new code should import the leaf modules directly.
"""

from __future__ import annotations

import inspect
import logging
import queue as _queue
import threading
import weakref
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, NamedTuple, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .._switch_sort import (
    gather_sort as _gather_sort,
)
from .._switch_sort import (
    scatter_unsort as _scatter_unsort,
)
from ._env import env_bool, env_int, env_str
from .bank_io import (
    _CTX_PREFETCH_AHEAD,
    _CTX_PREFETCH_MAX_BYTES,
    _CTX_UNION_MAX_BYTES,
    _DECODE_UNION_MAX_ROWS,
    _EXPERT_IO_POOL,
    _IO_POOLS,
    _IO_POOLS_LOCK,
    _IO_QD,
    _accepts_kwarg,
    _decode_call_shape,
    _io_pool,
    _layer_ctx_mode,
    _LayerLoadContext,
    io_pool_for,
    promote_np_array,
)
from .expert_cache import (
    _ADMIT_BATCH,
    _ADMIT_Q_MAX,
    _CACHE_POLICY_ENV,
    _LIVE_STREAMING_CACHES,
    CacheStats,
    ExpertLRUCache,
    _AdmissionWorker,
    _layer_index_add,
    _layer_index_drop,
    _layer_index_touch,
    make_expert_cache,
    streaming_gate_state,
)
from .shard_bank import np_to_mx, segment_runs
from .slot_cache import DecodeVisitStats, SlotArena
from .speculation import _STAGED_MAX_IDS, SpeculationState
from .staging import stage_headroom
from .streaming_layers import (
    _ARENA_ENV,
    _ARENA_MAX_BYTES,
    _ARENA_PRODUCE_JOBS,
    _BANK_MAX_BYTES,
    _BANK_RUN_MAX_BYTES,
    _COALESCE_ENV,
    _LAYER_BARRIER_ENV,
    _MAX_ADVISE_ROWS,
    _PREFILL_SHAPE_MIN_ROWS,
    _RA_ENV,
    _RUN_MAX,
    _STAGED_HEADROOM_ENV,
    StreamingQuantizedSwitchLinear,
    StreamingSwitchGLU,
    StreamingSwitchLinear,
    _ArenaBank,
    _build_plan_into,
    _RemapPlan,
    _ResolvedDemand,
    _spec_state_of,
    _StreamingLinearBase,
)

logger = logging.getLogger(__name__)


def __getattr__(name: str):
    # `from .streaming_switch import S3FIFOExpertCache` keeps working —
    # the class lives in cache_policies (which imports the expert_cache
    # leaf, never this façade, so the lazy edge cannot cycle).
    if name == "S3FIFOExpertCache":
        from .cache_policies import S3FIFOExpertCache

        return S3FIFOExpertCache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
