# SPDX-License-Identifier: Apache-2.0
"""Adaptive top-k truncation for MoE routing (opt-in quality/speed knob).

Ports the cumulative-mass idea from macqwen-releases (FlashNext,
MIT licensed): after the router's top-k selection, keep the smallest
score-descending prefix whose cumulative relative mass reaches
``threshold``. Dropped slots are padded with the top expert at score 0
(the duplicate collapses in the streaming plan, so no extra expert I/O)
and the kept scores are renormalized to the ORIGINAL total top-k mass
(blend 1.0), preserving activation magnitude.

Bit-exactness contract: ``threshold`` None or >= 1.0 bypasses everything
— the stock routing body runs untouched. Only 0 < threshold < 1.0
engages the approximation, and it changes outputs by design.

Applies to:
  * qwen4_exp (inherited ``Qwen3_5MoeSparseMoeBlock`` from installed
    mlx_vlm.models.qwen3_5_moe) — monkey-patched, mirroring the
    qwen35_moe_router.py convention;
  * glm5_next — a direct hook in the vendored ``Glm5NextMoE.__call__``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

_THRESHOLD: float | None = None
_MEAN_KEEPS: float | None = None

_MIN_THRESHOLD = 0.05
_MAX_THRESHOLD = 1.0

# Model types with a truncation hook. The qwen hook wraps
# Qwen3_5MoeSparseMoeBlock (shared by the vendored qwen4_exp decoder); the
# glm hook lives in the vendored Glm5NextMoE.__call__. Any other supported
# streaming type silently ignores the threshold without this gate — the
# converter must warn instead of logging it active (see is_topk_applicable).
TOPK_APPLICABLE_TYPES = frozenset(
    {
        "qwen4_exp",
        "qwen4_exp_text",
        "glm5_next",
        "glm5_next_text",
    }
)


def is_topk_applicable(model_type: object) -> bool:
    """True when an adaptive top-k hook exists for this model type."""
    from .residency import normalize_model_type

    return normalize_model_type(model_type) in TOPK_APPLICABLE_TYPES


def configure(threshold: float | None) -> None:
    """Set the active routing threshold (None or >= 1.0 = exact)."""
    global _THRESHOLD
    if threshold is None:
        _THRESHOLD = None
        return
    t = float(threshold)
    if not (_MIN_THRESHOLD <= t <= _MAX_THRESHOLD):
        raise ValueError(
            f"top-k threshold must be in [{_MIN_THRESHOLD}, 1.0], got {t}"
        )
    _THRESHOLD = None if t >= 1.0 else t
    if _THRESHOLD is not None:
        logger.info("Adaptive top-k truncation active: threshold=%.2f", _THRESHOLD)


def configure_from_settings(settings: Any | None, model_type: object = None) -> float | None:
    """Resolve the threshold from ModelSettings with an env fallback.

    When *model_type* is given and has no truncation hook, an approximate
    threshold is a silent no-op downstream — fail soft here instead: warn
    and return None so the caller skips the patch (exact routing stays on).
    """
    t = getattr(settings, "expert_streaming_topk_threshold", None) if settings else None
    if t is None:
        env = os.environ.get("OMLX_MOE_TOPK_THRESHOLD", "")
        if env.strip():
            try:
                t = float(env)
            except ValueError:
                logger.warning("Invalid OMLX_MOE_TOPK_THRESHOLD=%r; ignoring", env)
    configure(t)
    if _THRESHOLD is not None and model_type is not None and not is_topk_applicable(model_type):
        logger.warning(
            "Adaptive top-k threshold %.2f ignored: no truncation hook for model type %r "
            "(exact routing stays on)",
            _THRESHOLD,
            model_type,
        )
        return None
    return _THRESHOLD


def _safe_float_env(name: str, default: float) -> float:
    """Fail-closed env-float parse (audit Fase 1 lesson: never bare-cast
    at import — a malformed value must disable the knob, not brick the
    module)."""
    try:
        raw = os.environ.get(name, "")
        return float(raw) if raw.strip() else default
    except (TypeError, ValueError, AttributeError):
        return default


# Fase 2 P3 (cache-conditional routing, Qualcomm 2412.00099): logit bonus
# for LRU-resident experts before top-k. 0.0 = exact routing (default).
# Approximate by design — opt-in only.
_CACHE_PRIOR = max(0.0, _safe_float_env("OMLX_EXPERT_STREAMING_CACHE_PRIOR", 0.0))


def cache_prior_bonus() -> float:
    """Active cache-prior logit bonus (0.0 = exact routing)."""
    return _CACHE_PRIOR


def configure_cache_prior(value: Any) -> float:
    """Set the active bonus (None = env fallback, <=0 = exact).

    Mirrors configure(): explicit values win, env fills the gap, garbage
    fails closed to exact. Returns the effective bonus."""
    global _CACHE_PRIOR
    if value is None:
        _CACHE_PRIOR = max(0.0, _safe_float_env("OMLX_EXPERT_STREAMING_CACHE_PRIOR", 0.0))
    else:
        try:
            _CACHE_PRIOR = max(0.0, float(value))
        except (TypeError, ValueError):
            _CACHE_PRIOR = 0.0
    return _CACHE_PRIOR


def cache_prior_from_settings(settings: Any | None) -> float:
    """Resolve the bonus from ModelSettings with env fallback."""
    v = getattr(settings, "expert_streaming_cache_prior", None) if settings else None
    return configure_cache_prior(v)


def resident_experts(switch_mlp: Any) -> set[int]:
    """Experts resident in the app-level LRU for this layer.

    Intersection over the GLU's projection linears (an expert needs every
    projection to avoid I/O). Duck-typed and fail-closed: anything
    unexpected yields the empty set (no rerank)."""
    try:
        cache = getattr(switch_mlp, "_cache", None)
        store = getattr(cache, "_store", None)
        if store is None:
            return set()
        n = int(getattr(switch_mlp, "_num_experts", 0) or 0)
        if n <= 0:
            return set()
        lins = [
            getattr(switch_mlp, a, None)
            for a in ("gate_proj", "up_proj", "down_proj", "gate_up_proj")
        ]
        lins = [l for l in lins if l is not None and hasattr(l, "bundle_key")]
        if not lins:
            return set()
        res: set[int] | None = None
        for lin in lins:
            try:
                present = {e for e in range(n) if lin.bundle_key(e) in store}
            except Exception:
                return set()
            res = present if res is None else (res & present)
            if not res:
                return set()
        return res or set()
    except Exception:
        return set()


def rerank_cache_prior(gates: Any, resident: set[int] | None, bonus: float) -> Any:
    """Boost resident experts by *bonus* in logit space before top-k.

    Identity when the bonus is off or the set is empty (exact routing
    untouched). Implemented as two lazy ops (broadcast compare + add),
    no per-expert graph bloat.
    """
    if bonus <= 0 or not resident:
        return gates
    try:
        width = int(gates.shape[-1])
        res = sorted({int(e) for e in resident if 0 <= int(e) < width})
        if not res:
            return gates
        import mlx.core as _mx

        anchors = _mx.array(res, dtype=_mx.int32)
        is_res = (
            _mx.arange(width)[None, :] == anchors[:, None]
        ).any(axis=0)
        LOG = _mx.log(_mx.maximum(gates, 1e-30))
        return _mx.softmax(LOG + is_res.astype(LOG.dtype) * float(bonus), axis=-1)
    except Exception:
        return gates


def apply_cache_prior_to_logits(logits: Any, resident: set[int] | None, bonus: float) -> Any:
    """Boost resident experts by *bonus* on raw (pre-sigmoid) logits.

    The GLM/DeepSeek-style group router consumes raw logits (sigmoid is
    inside group_expert_select), so unlike rerank_cache_prior there is no
    log/softmax roundtrip — a plain masked add. Identity when off/empty;
    fail closed to the input on any error.
    """
    if bonus <= 0 or not resident:
        return logits
    try:
        width = int(logits.shape[-1])
        res = sorted({int(e) for e in resident if 0 <= int(e) < width})
        if not res:
            return logits
        import mlx.core as _mx

        anchors = _mx.array(res, dtype=_mx.int32)
        is_res = (
            _mx.arange(width)[None, :] == anchors[:, None]
        ).any(axis=0)
        return logits.astype(_mx.float32) + is_res.astype(_mx.float32) * float(bonus)
    except Exception:
        return logits


def current_threshold() -> float | None:
    return _THRESHOLD


def mean_keeps() -> float | None:
    """Mean kept experts per routed row of the last truncation pass (None
    when truncation is inactive or has not run)."""
    return _MEAN_KEEPS


def truncate_topk_mass(inds, scores, threshold: float, return_keeps: bool = False):
    """Truncate a top-k routing by cumulative relative mass.

    ``inds``: [..., k] expert ids; ``scores``: [..., k] routing scores
    (any positive scaling — relative mass is used). Returns
    (inds, scores) with the same shapes: kept experts in score-descending
    order, dropped slots holding the top expert at score 0, kept scores
    renormalized to the original total top-k mass.
    """
    global _MEAN_KEEPS
    total = mx.sum(scores, axis=-1, keepdims=True)
    rel = scores / mx.maximum(total, 1e-30)
    order = mx.argsort(-rel, axis=-1)
    s_sorted = mx.take_along_axis(rel, order, axis=-1)
    i_sorted = mx.take_along_axis(inds, order, axis=-1)
    # mass accumulated BEFORE the current expert — keep while it is still
    # below the threshold (the first expert always keeps)
    cum_before = mx.cumsum(s_sorted, axis=-1) - s_sorted
    keep = (cum_before < threshold).astype(scores.dtype)
    first = i_sorted[..., :1]
    i_pad = mx.where(keep > 0, i_sorted, first)
    s_pad = s_sorted * keep
    denom = mx.maximum(mx.sum(s_pad, axis=-1, keepdims=True), 1e-30)
    s_final = (s_pad / denom) * total
    if return_keeps:
        keeps = mx.sum(keep, axis=-1).mean().item()
        _MEAN_KEEPS = keeps
        return i_pad.astype(inds.dtype), s_final.astype(scores.dtype), keeps
    _MEAN_KEEPS = None
    return i_pad.astype(inds.dtype), s_final.astype(scores.dtype)


def apply_qwen35_moe_topk_patch() -> bool:
    """Engage truncation for Qwen3.5/3.6/qwen4_exp sparse MoE blocks.

    Wraps ``Qwen3_5MoeSparseMoeBlock.__call__`` in the installed mlx_vlm
    (qwen3_5_moe.language, shared by the vendored qwen4_exp). When the
    active threshold is exact (None/1.0) the wrapped call — stock or the
    fused-router patch — runs untouched.
    """
    try:
        from mlx_vlm.models.qwen3_5_moe import language as q35
    except ImportError:
        return False
    cls = getattr(q35, "Qwen3_5MoeSparseMoeBlock", None)
    if cls is None or getattr(cls, "_omlx_topk_truncate", False):
        return cls is not None

    orig_call = cls.__call__

    def patched_call(self, x, target_verify: bool = False):
        thr = current_threshold()
        bonus = _CACHE_PRIOR
        if (thr is None or thr >= 1.0) and bonus <= 0:
            return orig_call(self, x, target_verify=target_verify)
        try:
            gates = q35._target_verify_linear(self.gate, x, target_verify)
            gates = mx.softmax(gates, axis=-1, precise=True)
            if bonus > 0:
                gates = rerank_cache_prior(
                    gates, resident_experts(self.switch_mlp), bonus
                )
            k = self.top_k
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            scores = mx.take_along_axis(gates, inds, axis=-1)
            scores = scores / scores.sum(axis=-1, keepdims=True)
            if thr is not None and thr < 1.0:
                inds, scores = truncate_topk_mass(inds, scores, thr)
            y = q35._target_verify_switch_glu(self.switch_mlp, x, inds, target_verify)
            y = (y * scores[..., None]).sum(axis=-2)
            shared_y = self.shared_expert(x, target_verify)
            shared_y = (
                mx.sigmoid(q35._target_verify_linear(self.shared_expert_gate, x, target_verify))
                * shared_y
            )
            return y + shared_y
        except Exception:
            logger.warning("adaptive top-k routing failed; stock fallback", exc_info=True)
            return orig_call(self, x, target_verify=target_verify)

    cls.__call__ = patched_call
    cls._omlx_topk_truncate = True
    logger.info("Qwen3.5/qwen4_exp adaptive top-k patch applied")
    return True
