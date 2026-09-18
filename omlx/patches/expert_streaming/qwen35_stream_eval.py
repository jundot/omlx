# SPDX-License-Identifier: Apache-2.0
"""Per-layer eval boundary for the installed qwen3_5_moe decoder (streaming).

The streaming converter flags every converted decoder layer with
``_stream_eval``. The vendored GLM-5.3 decoder honors the flag inline
(``Glm5NextDecoderLayer.__call__``); the installed ``mlx_vlm``
``Qwen3_5MoeDecoderLayer`` does not, and neither does the vendored
``Qwen4ExpDecoderLayer`` — so on qwen4_exp the flag is inert and a long
prefill chunk accumulates one streaming mini-bank per layer in the lazy
graph until the chunk-end eval (~17 MB/token, pool peaks of tens of GiB).
This wraps every such decoder with the DeepSeek/GLM boundary: evaluate the
layer output as soon as the layer returns and trim the allocator cache so
the retained pool cannot grow past the layer's working set and evict the
OS page cache the streaming path depends on.

Without a per-layer boundary the lazy graph accumulates one transient per
layer across all 48 layers; demand-set tiling and the rolling bank load
only bound a *single* layer's transient.

Prefill-shaped calls only (``x.shape[1] > 1``; batch decode is [B, 1, H]):
decode graphs are small and 48 forced syncs/token would erode decode
throughput, and MTP verify passes (``target_verify``) stay lazy for the
same reason.
Output is bit-identical — ``mx.eval`` only materializes what the next layer
reads anyway; ``mx.clear_cache`` frees cached buffers back to Metal.
"""
from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

from ._env import env_bool, env_float

logger = logging.getLogger(__name__)

# Default ON: prefill is where the lazy-graph accumulation hurts and the
# boundary is bit-exact. Disable with OMLX_EXPERT_STREAMING_PER_LAYER_EVAL=0
# or the per-model setting expert_streaming_per_layer_eval=false.
_PER_LAYER_EVAL_DEFAULT = env_bool("OMLX_EXPERT_STREAMING_PER_LAYER_EVAL", True)
_per_layer_eval_enabled = _PER_LAYER_EVAL_DEFAULT

_APPLIED_FLAG = "_omlx_stream_eval_wrapped"
# Class attribute marking decoder layers that honor ``_stream_eval`` inline
# in their own ``__call__`` (``Glm5NextDecoderLayer``) instead of needing
# the wrapper below. The converter reads it for the per-model boundary
# verdict — never hardcode model types here.
_INLINE_STREAM_EVAL_ATTR = "_omlx_honors_inline_stream_eval"


def _cache_threshold_bytes() -> int:
    """Pool size (bytes) above which the per-layer clear is worth running.

    Delegates to ``omlx.utils.metal_sync.cache_clear_threshold_bytes`` so
    the knob has one home: the scheduler's chunk-boundary clears gate on
    the same number as this per-layer clear. Imported lazily — the
    model path must keep working when that module is unavailable, and the
    local parse below is the fallback.
    """
    try:
        from omlx.utils.metal_sync import cache_clear_threshold_bytes

        return cache_clear_threshold_bytes()
    except Exception:  # noqa: BLE001
        pass
    # Fallback for when metal_sync is unavailable: same env (GiB), same
    # 2 GiB default, degrade-to-default on garbage via env_float.
    return max(0, int(env_float("OMLX_EXPERT_STREAMING_CACHE_THRESH", 2.0) * 1024**3))


def configure_from_settings(value: Any) -> bool:
    """Resolve the knob (``None`` = env / built-in default) and store the
    effective flag the wrapper reads per call. Returns the effective value."""
    global _per_layer_eval_enabled
    _per_layer_eval_enabled = (
        _PER_LAYER_EVAL_DEFAULT if value is None else bool(value)
    )
    return _per_layer_eval_enabled


def _clear_cache_synced() -> None:
    """Release the reusable Metal pool, but only after a full sync.

    ``mx.clear_cache`` must never run with command buffers in flight: on M4
    the driver panics with 'completeMemory() prepare count underflow'. The
    caller has already done ``mx.eval(out)``, but ``_sync_and_clear_cache``
    additionally takes the buffer-access lock that keeps the async
    store-cache worker from observing a half-reclaimed pool (#1106), so it is
    preferred whenever it is importable.
    """
    # The bare clear is ONLY the import-failure fallback. A failure INSIDE
    # the synced helper (lock, sync, eval) must not fall through to the
    # unsynchronized clear — that reintroduces the very 'clear with command
    # buffers in flight' race this helper exists to prevent. The
    # scheduler's chunk-boundary clear covers the pool in that case.
    try:
        from omlx.utils.metal_sync import _sync_and_clear_cache
    except Exception:
        try:
            mx.clear_cache()
        except Exception:
            pass
        return
    try:
        _sync_and_clear_cache()
    except Exception:
        logger.warning(
            "_sync_and_clear_cache failed; skipping the per-layer clear",
            exc_info=True,
        )


def _wrap_call(orig_call: Any) -> Any:
    def call(self, x, *args, **kwargs):
        out = orig_call(self, x, *args, **kwargs)
        if (
            _per_layer_eval_enabled
            and getattr(self, "_stream_eval", False)
            and not kwargs.get("target_verify", False)
            and x.ndim >= 2
            and x.shape[1] > 1
        ):
            mx.eval(out)
            get_cache_memory = getattr(mx, "get_cache_memory", None)
            if get_cache_memory is None or get_cache_memory() >= _cache_threshold_bytes():
                _clear_cache_synced()
        return out

    return call


def _candidate_decoder_classes() -> list[tuple[str, Any]]:
    """Decoder layer classes that ignore ``_stream_eval`` and need wrapping.

    The family registry (``model_hooks.all_stream_eval_targets``) owns the
    target list — adding a family is a table row there, not another import
    block here. ``Qwen4ExpDecoderLayer`` derives from ``nn.Module``
    directly, not from ``Qwen3_5MoeDecoderLayer``, so wrapping both cannot
    double-wrap.
    """
    from .model_hooks import all_stream_eval_targets

    return all_stream_eval_targets()


def apply_qwen35_moe_stream_eval() -> bool:
    """Wrap every qwen decoder that ignores ``_stream_eval`` with the
    streaming eval boundary.

    Idempotent (a per-class flag short-circuits re-wrapping); the per-call
    behavior follows the current module flag so a settings change that
    reloads the engine takes effect without a process restart. Returns True
    when at least one target class was found (wrapped now or previously).
    """
    wrapped = False
    for name, cls in _candidate_decoder_classes():
        if getattr(cls, _APPLIED_FLAG, False):
            wrapped = True
            continue
        cls.__call__ = _wrap_call(cls.__call__)
        setattr(cls, _APPLIED_FLAG, True)
        wrapped = True
        logger.info(
            "Qwen3.5/qwen4_exp streaming per-layer eval boundary installed on %s",
            name,
        )
    return wrapped


def wrapped_class_names() -> list[str]:
    """Names of decoder classes currently carrying the boundary (test hook)."""
    return [
        cls.__name__
        for _, cls in _candidate_decoder_classes()
        if getattr(cls, _APPLIED_FLAG, False)
    ]


def boundary_active_for_layers(layers: Any) -> bool:
    """True when a per-layer eval boundary runs on *this* model's layers.

    Per-model verdict for the scheduler's
    ``streaming_guard_info["boundary_active"]`` (definitive streaming
    guard): the module flag must be on AND at least one of the given
    layers must actually run a boundary — either an instance of a wrapped
    decoder class, or a decoder class that honors ``_stream_eval`` inline
    (``Glm5NextDecoderLayer``, via ``_INLINE_STREAM_EVAL_ATTR``).

    A process-global "any wrapped class exists" check would over-credit
    models with no boundary (under-charge risk) and mis-attribute the GLM
    inline boundary — use this per-model verdict for guard metadata.
    """
    if not _per_layer_eval_enabled:
        return False
    try:
        layer_list = list(layers or ())
    except TypeError:
        return False
    if not layer_list:
        return False
    if any(
        getattr(type(layer), _INLINE_STREAM_EVAL_ATTR, False)
        for layer in layer_list
    ):
        return True
    wrapped = {
        cls
        for _, cls in _candidate_decoder_classes()
        if getattr(cls, _APPLIED_FLAG, False)
    }
    if not wrapped:
        return False
    try:
        return any(isinstance(layer, tuple(wrapped)) for layer in layer_list)
    except TypeError:
        return False

