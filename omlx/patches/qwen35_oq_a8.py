# ruff: noqa: N806
"""Dispatch layer for the oQ mixed-bit QxA8 prefill kernels.

Every ``QuantizedLinear`` is classified exactly once, at the point it is first
seen, into a frozen plan (bits, activation-scale policy, tile). The forward
path then reads that plan off the module instead of re-parsing quantization
metadata on every call.

Importing this module patches nothing. The kernel changes inference numerics
by quantizing activations to INT8, so turning it on is an accuracy decision
and stays behind an explicit opt-in. Installing the model-side routing is an
explicit call to :func:`apply_qwen35_oq_a8_patch`, which the engine makes only
when ``qwen35_oq_a8_enabled`` is set for the model, and which opts that model
in by tagging its modules; ``OMLX_OQ_A8=1`` is the equivalent opt-in for
callers that route through :func:`oq_a8_linear`, :func:`oq_a8_mlp` or
:func:`oq_a8_gdn_projections` directly.

The class wrapper it installs is process-wide and stays installed, but the
tag is per model, so the setting is reversible: a model loaded with it off is
not tagged and the wrapper falls through.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

# Only Q4 and Q5 GS64 affine matter for this checkpoint; everything else is
# fallback.
_SUPPORTED_BITS = frozenset((4, 5))
_GROUP_SIZE = 64

# Variant-rejection messages already logged, so a bad environment variable
# warns once rather than once per projection.
_WARNED_VARIANTS: set[str] = set()

_PLAN_ATTR = "_omlx_oq_a8_plan"
_PREPARED_ATTR = "_omlx_oq_a8_prepared"

# The kernel reads the checkpoint's own packed weight stream, unmodified, and
# gets a lane's codes into one contiguous read by choosing which K each
# fragment slot stands for rather than by moving any bytes. The only prepared
# arrays are the transposed scale/bias tables, four orders of magnitude smaller
# than the weights, so a routed projection costs no extra weight memory and the
# decode path is untouched.

# Rowwise A8 is the throughput path. GS64 activation scales cost
# more preprocessing but align the activation and weight group boundaries
# exactly, which is most attractive for the Q5-protected sensitive layers
#. The policy does not have to be uniform, so Q4 and Q5 read
# separate knobs.
_ACT_MODE_ROW = 0
_ACT_MODE_G64 = 1


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default


# Prefill shorter than this stays on the existing path, where Stage A costs
# more than the faster matmul saves.
_MIN_TOKENS_DEFAULT = 128

# Attached by apply_qwen35_oq_a8_patch() to the modules of one loaded model.
# Its presence is the opt-in.
_CONFIG_ATTR = "_omlx_oq_a8_config"


@dataclass(frozen=True)
class OqA8Config:
    """One model's oQ A8 settings, as the engine passed them in."""

    min_tokens: int = _MIN_TOKENS_DEFAULT


# Opting in without a settings object or a tagged model -- benchmarks, tests,
# and anything driving the dispatcher directly.
_ENV_CONFIG = OqA8Config()


def enabled() -> bool:
    """True when the process-wide environment opt-in is on and the kernels run.

    This is the ``OMLX_OQ_A8`` hook only. Whether a *model* is routed is a
    per-module question -- see :func:`_config_for`.
    """
    if os.environ.get("OMLX_OQ_A8") != "1":
        return False
    return _kernels_available()


def _config_for(module: Any) -> OqA8Config | None:
    """The oQ A8 configuration governing ``module``, or None to stay on MLX.

    Gating is per module rather than process-wide because the patch replaces
    ``Qwen3_5MLP.__call__``, which every resident model shares. A model whose
    settings have the feature off never gets tagged, so the shared wrapper
    falls straight through for it. That is what makes the setting reversible:
    a process-wide flag stays stuck on once any model has turned it on, so
    switching it off and reloading would keep routing. It also keeps two
    resident models with different settings independent, including their
    token floors, rather than last-writer-wins.

    Mirrors ``_omlx_ane_prefill_config`` in the ANE prefill patch, which gates
    the same class the same way.
    """
    if os.environ.get("OMLX_OQ_A8") == "0":
        return None
    config = getattr(module, _CONFIG_ATTR, None)
    if config is None:
        if os.environ.get("OMLX_OQ_A8") != "1":
            return None
        config = _ENV_CONFIG
    return config if _kernels_available() else None


def _tag_modules(model: Any, config: OqA8Config) -> int:
    """Opt every module of one loaded model in. Returns how many were tagged.

    Tags the whole tree rather than just the MLPs: the standalone projections
    the linear backend gets first refusal on (``linear_attn.out_proj``) are
    reached as themselves, not through a parent.
    """
    tagged = 0
    try:
        modules = [module for _, module in model.named_modules()]
    except Exception:
        logger.debug("oq_a8: could not walk the model tree", exc_info=True)
        return 0
    for module in modules:
        try:
            setattr(module, _CONFIG_ATTR, config)
        except Exception:
            continue
        tagged += 1
    return tagged


def _kernels_available() -> bool:
    try:
        from omlx.custom_kernels.qwen35_prefill import fast
    except Exception:
        return False
    return fast.oq_a8_available()


@dataclass(frozen=True)
class OqA8Plan:
    """Frozen per-projection dispatch decision.

    Built once at classification time so the hot forward path never inspects
    quantization configuration.
    """

    bits: int
    group_size: int
    act_mode: int
    variant: int

    @property
    def kernel(self) -> str:
        return f"q{self.bits}a8_g{self.group_size}"


def _act_mode_for_bits(bits: int) -> int:
    if bits == 5:
        return _env_int("OMLX_OQ_A8_Q5_ACT_MODE", _ACT_MODE_ROW)
    return _env_int("OMLX_OQ_A8_Q4_ACT_MODE", _ACT_MODE_ROW)


# 800-806 are the same kernel over different simdgroup grids. The field is
# flat -- within ~3% at M=2048 -- so these are the best of a flat field rather
# than a sharp optimum, and they differ by bit width because Q5 reads a wider
# window per row and prefers a smaller tile.
_DEFAULT_VARIANT_Q4 = 806
_DEFAULT_VARIANT_Q5 = 800

# The only family the kernels carry. A variant outside it names no kernel, so
# it is refused where it enters rather than at the op boundary.
_VARIANT_MIN = 800
_VARIANT_MAX = 806


def _variant_for_bits(bits: int) -> int:
    # Q4 and Q5 are autotuned independently: Q5 reads a second plane per
    # weight row and costs more registers, so the best tile need not match
    #.
    default = _DEFAULT_VARIANT_Q5 if bits == 5 else _DEFAULT_VARIANT_Q4
    shared = _env_int("OMLX_OQ_A8_VARIANT", default)
    if bits == 5:
        return _env_int("OMLX_OQ_A8_Q5_VARIANT", shared)
    return _env_int("OMLX_OQ_A8_Q4_VARIANT", shared)


def check_variant(variant: int) -> int:
    """Return ``variant`` if a kernel exists for it, else raise.

    Settings files and environment variables are both untrusted here: an
    out-of-family number is an error at this boundary rather than a
    missing-kernel failure deep inside the op.
    """
    if _VARIANT_MIN <= variant <= _VARIANT_MAX:
        return variant
    raise ValueError(
        f"oQ A8 variant {variant} is not a shipped kernel; use "
        f"{_VARIANT_MIN}-{_VARIANT_MAX}."
    )


def classify_linear(linear: Any) -> OqA8Plan | None:
    """Return the dispatch plan for ``linear``, or None if it is not eligible.

    The result is memoized on the module, so this runs once per projection
    per process however often the forward path asks for it.
    """
    cached = getattr(linear, _PLAN_ATTR, False)
    if cached is not False:
        return cached

    plan = _classify_uncached(linear)
    try:
        object.__setattr__(linear, _PLAN_ATTR, plan)
    except Exception:
        # nn.Module subclasses that reject attribute writes just pay the
        # classification cost again; correctness is unaffected.
        pass
    return plan


def _classify_uncached(linear: Any) -> OqA8Plan | None:
    if not isinstance(linear, nn.QuantizedLinear):
        return None
    if getattr(linear, "mode", None) != "affine":
        return None
    bits = getattr(linear, "bits", None)
    group_size = getattr(linear, "group_size", None)
    if bits not in _SUPPORTED_BITS or group_size != _GROUP_SIZE:
        return None
    if "bias" in linear:
        return None

    weight = getattr(linear, "weight", None)
    scales = getattr(linear, "scales", None)
    biases = getattr(linear, "biases", None)
    if weight is None or scales is None or biases is None:
        return None
    if weight.dtype != mx.uint32 or weight.ndim != 2:
        return None
    if scales.dtype not in (mx.float16, mx.bfloat16):
        return None
    if biases.dtype != scales.dtype or scales.shape != biases.shape:
        return None
    if scales.ndim != 2 or scales.shape[0] != weight.shape[0]:
        return None

    # The packed layout is the one MLX already stores and oMLX already
    # validates; no checkpoint conversion is required.
    input_dim = scales.shape[1] * _GROUP_SIZE
    if weight.shape[1] * 32 != input_dim * int(bits):
        return None

    try:
        variant = check_variant(_variant_for_bits(int(bits)))
    except ValueError as exc:
        # Once per distinct message: this runs for every eligible projection
        # in the model, and a bad OMLX_OQ_A8_VARIANT is bad for all of them.
        message = str(exc)
        if message not in _WARNED_VARIANTS:
            _WARNED_VARIANTS.add(message)
            logger.warning("oq_a8: %s; leaving these projections alone", message)
        return None
    plan = OqA8Plan(
        bits=int(bits),
        group_size=_GROUP_SIZE,
        act_mode=_act_mode_for_bits(int(bits)),
        variant=variant,
    )

    from omlx.custom_kernels.qwen35_prefill import fast

    # N must tile exactly; the kernel refuses partial column tiles so the
    # weight decoder can stay bounds-check free.
    tile_bn = _variant_bn(plan.variant)
    if weight.shape[0] % tile_bn != 0:
        logger.debug(
            "oq_a8: N=%d is not a multiple of BN=%d; leaving this projection "
            "on the existing path",
            weight.shape[0],
            tile_bn,
        )
        return None
    if not fast.oq_a8_available():
        return None
    return plan


# Mirror of oq_a8_v2_variant() in qwen35_oq_a8.cpp, which both the shipped
# families index into: (BM, BN, WM, WN), keyed by the offset from the family's
# base.
_VARIANT_TILES = {
    0: (64, 64, 2, 2),
    1: (128, 64, 4, 2),
    2: (64, 128, 2, 4),
    3: (128, 128, 4, 4),
    4: (32, 128, 1, 4),
    5: (256, 64, 8, 2),
    6: (32, 64, 1, 2),
}


def _variant_tile(variant: int) -> tuple[int, int, int, int]:
    try:
        return _VARIANT_TILES[variant - _VARIANT_MIN]
    except KeyError as exc:
        raise ValueError(f"Unknown oQ A8 tile variant {variant}") from exc


def _variant_bn(variant: int) -> int:
    return _variant_tile(variant)[1]


def _prepared_weights(linear: Any):
    """Load-time operand transform for ``linear``, computed once and cached.

    The weight stream is handed through untouched -- the kernel reads the
    checkpoint's own bytes -- so this only transposes the scale and bias
    tables to group-major, which is four orders of magnitude less memory than
    a repacked weight copy would be and leaves the module's arrays, and MLX's
    decode path over them, exactly as loaded.
    """
    cached = getattr(linear, _PREPARED_ATTR, None)
    if cached is not None:
        return cached

    prepared = (
        linear.weight,
        mx.contiguous(linear.scales.T),
        mx.contiguous(linear.biases.T),
    )
    mx.eval(*prepared)
    try:
        object.__setattr__(linear, _PREPARED_ATTR, prepared)
    except Exception:
        pass
    return prepared


@dataclass(frozen=True)
class StageA:
    """Quantized activation shared across projections of one input.

    Stage A runs once per activation. The MLP feeds it to gate and up, and a
    linear-attention block feeds it to qkv/z/a/b, whose bit widths may differ
    -- mixed bit widths need no separate activation quantization.
    """

    qa: mx.array
    sa: mx.array
    ra: mx.array
    act_mode: int


def stage_a(x: mx.array, act_mode: int = _ACT_MODE_ROW) -> StageA:
    """Quantize one activation into the layout the kernel reads.

    Qa carries the schedule's within-group K order, and Ra (and Sa in
    per-group mode) come back group-major to match the weight metadata. This
    is the only operand that is reordered, and it is rebuilt on every prefill,
    so the order costs no memory and nothing on disk changes.
    """
    from omlx.custom_kernels.qwen35_prefill import fast

    qa, sa, ra = fast.qwen35_oq_a8_stage_a_v8(x, act_mode)
    return StageA(qa=qa, sa=sa, ra=ra, act_mode=act_mode)


def apply_plan(linear: Any, stage: StageA, plan: OqA8Plan) -> mx.array:
    """Run one projection against an already-quantized activation."""
    if stage.act_mode != plan.act_mode:
        raise ValueError(
            "Stage-A activation mode "
            f"{stage.act_mode} does not match the projection's plan "
            f"({plan.act_mode}); quantize once per activation policy."
        )
    from omlx.custom_kernels.qwen35_prefill import fast

    weight, scales, biases = _prepared_weights(linear)
    return fast.qwen35_oq_a8_qmm_t(
        stage.qa,
        stage.sa,
        stage.ra,
        weight,
        scales,
        biases,
        plan.bits,
        plan.act_mode,
        plan.variant,
    )


def oq_a8_linear(linear: Any, x: mx.array) -> mx.array:
    """Single projection with no shared activation; falls back transparently."""
    config = _config_for(linear)
    plan = classify_linear(linear) if config is not None else None
    if plan is None or not _shape_eligible(x, config):
        return linear(x)
    return apply_plan(linear, stage_a(x, plan.act_mode), plan)


# Below this many rows the Stage-A quantization pass costs more than the
# faster GEMM saves. Measured on a 5120x8704 Q4 MLP block, the crossover sits
# between 4 and 8 rows and the win is already 1.6x by 128. The default is set
# well above the crossover rather than at it, so that decode and speculative
# verify shapes -- which are latency-critical and only a handful of rows --
# stay on the existing path even where a caller does not flag them.


def _min_tokens(config: OqA8Config) -> int:
    # The environment still wins, for benchmarking.
    return _env_int("OMLX_OQ_A8_MIN_TOKENS", config.min_tokens)


def _shape_eligible(x: mx.array, config: OqA8Config) -> bool:
    # Decode is out of scope for version 1; the kernel is built for
    # prefill and the tiles start at 32 rows.
    if x.ndim < 2 or x.dtype not in (mx.float16, mx.bfloat16):
        return False
    if x.shape[-2] < _min_tokens(config):
        return False
    return x.shape[-1] % _GROUP_SIZE == 0


def oq_a8_mlp(mlp: Any, x: mx.array, activation) -> mx.array | None:
    """SwiGLU MLP with Stage A shared between gate and up.

    Both are Q4 everywhere in this checkpoint, so the shared quantization is a
    guaranteed win and no mixed-bit gate/up template is needed.
    Returns None when any of the three projections is not eligible, leaving
    the caller on its existing path.
    """
    config = _config_for(mlp)
    if config is None or not _shape_eligible(x, config):
        return None

    gate, up, down = mlp.gate_proj, mlp.up_proj, mlp.down_proj
    gate_plan = classify_linear(gate)
    up_plan = classify_linear(up)
    down_plan = classify_linear(down)
    if gate_plan is None or up_plan is None or down_plan is None:
        return None
    if gate_plan.act_mode != up_plan.act_mode:
        return None

    shared = stage_a(x, gate_plan.act_mode)
    g = apply_plan(gate, shared, gate_plan)
    u = apply_plan(up, shared, up_plan)
    h = activation(g, u)

    if not _shape_eligible(h, config):
        return down(h)
    return apply_plan(down, stage_a(h, down_plan.act_mode), down_plan)


def oq_a8_gdn_projections(
    gdn: Any,
    x: mx.array,
    names: tuple[str, ...] = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"),
) -> dict[str, mx.array] | None:
    """Linear-attention input projections sharing one Stage A.

    qkv is Q4 while z/a/b may be Q5, and that mix is exactly the case the
    shared activation is designed for. Returns None unless every
    requested projection is eligible under one activation policy.
    """
    config = _config_for(gdn)
    if config is None or not _shape_eligible(x, config):
        return None

    plans: dict[str, OqA8Plan] = {}
    for name in names:
        linear = getattr(gdn, name, None)
        if linear is None:
            return None
        plan = classify_linear(linear)
        if plan is None:
            return None
        plans[name] = plan

    modes = {plan.act_mode for plan in plans.values()}
    if len(modes) != 1:
        # A split policy would need one Stage A per mode, which costs more
        # than the mixed dispatch saves on the shapes measured here.
        return None

    shared = stage_a(x, modes.pop())
    return {
        name: apply_plan(getattr(gdn, name), shared, plan)
        for name, plan in plans.items()
    }


# --------------------------------------------------------------------------
# Model-side routing
# --------------------------------------------------------------------------
#
# Routed only for a model whose ``qwen35_oq_a8_enabled`` setting says so. Two
# entry points carry essentially all of the routable work in this checkpoint:
# the SwiGLU MLP, and the linear-attention input projections.
#
# The MLP is patched directly because its three projections share one Stage A.
# The GDN projections go through the first-refusal hook that the q4 prefill
# patch already exposes, rather than a second wrapper around
# ``GatedDeltaNet.__call__`` -- that keeps the recurrent implementation and
# the Native-MTP call signature in one place.

_MLP_PATCHED = False
_GDN_REGISTERED = False
_MLP_TARGETS = (
    ("mlx_vlm.models.qwen3_5.language", "Qwen3_5MLP"),
    ("mlx_vlm.models.qwen3_5_moe.language", "Qwen3_5MLP"),
    ("mlx_lm.models.qwen3_5", "MLP"),
)


def _swiglu():
    """mlx-lm's SwiGLU, resolved on first routed call.

    Deferred rather than resolved at patch time so the wrapper can be built
    and tested without mlx-lm present; in production the class being patched
    comes from mlx-lm anyway, so the import always succeeds by the time a
    routed call reaches it.
    """
    global _SWIGLU
    if _SWIGLU is None:
        from mlx_lm.models.activations import swiglu

        _SWIGLU = swiglu
    return _SWIGLU


_SWIGLU = None


def _make_patched_mlp(orig_call):
    def patched(self, x, *args, **kwargs):
        # This runs on every MLP call of every layer of every decode step, so
        # the common case has to exit on an attribute read and a shape check
        # alone (#2132) -- no kernel probing, no import.
        config = getattr(self, _CONFIG_ATTR, None) or _ENV_CONFIG
        if x.ndim < 3 or x.shape[-2] < _min_tokens(config):
            return orig_call(self, x, *args, **kwargs)
        target_verify = bool(kwargs.get("target_verify", False))
        if args and isinstance(args[0], bool):
            target_verify = target_verify or bool(args[0])
        if target_verify:
            return orig_call(self, x, *args, **kwargs)
        try:
            out = oq_a8_mlp(self, x, _swiglu())
        except Exception:
            logger.debug("oq_a8: MLP routing failed; falling back", exc_info=True)
            return orig_call(self, x, *args, **kwargs)
        if out is None:
            return orig_call(self, x, *args, **kwargs)
        return out

    return patched


def _gdn_prefill_backend(gdn: Any, inputs: mx.array, target_verify: bool):
    """First-refusal backend for the mlx-lm GDN input projections.

    Returns ``(qkv, z, b, a)`` -- the order the caller unpacks -- or None to
    leave the projections on their existing path.
    """
    if target_verify:
        return None
    try:
        projections = oq_a8_gdn_projections(gdn, inputs)
    except Exception:
        logger.debug("oq_a8: GDN routing failed; falling back", exc_info=True)
        return None
    if projections is None:
        return None
    return (
        projections["in_proj_qkv"],
        projections["in_proj_z"],
        projections["in_proj_b"],
        projections["in_proj_a"],
    )


def _prefill_linear_backend(linear: Any, x: mx.array) -> mx.array | None:
    """First refusal on one standalone prefill projection, or None to decline.

    The MLP and the GDN input projections are intercepted upstream, where a
    single Stage A is shared between several matmuls. What reaches here is the
    projections that stand alone -- ``linear_attn.out_proj`` above all, which
    is Q5 and per layer, and which otherwise runs on the W4A16 path on both
    engines.

    Both gates still apply, and the caller's is the binding one: the q4 linear
    patch only offers this backend a projection it already decided to route,
    behind its own longer prompt floor. The shape check here is what keeps a
    direct caller honest.
    """
    config = _config_for(linear)
    plan = classify_linear(linear) if config is not None else None
    if plan is None or not _shape_eligible(x, config):
        return None
    return apply_plan(linear, stage_a(x, plan.act_mode), plan)


def _patch_mlp_class(module_name: str, class_name: str) -> bool:
    import importlib

    try:
        module = importlib.import_module(module_name)
    except Exception:
        return False
    cls = getattr(module, class_name, None)
    if cls is None:
        return False
    if getattr(cls, "_omlx_oq_a8_patched", False):
        return True
    orig = cls.__call__
    cls.__call__ = _make_patched_mlp(orig)
    cls._omlx_oq_a8_patched = True
    cls._omlx_oq_a8_original_call = orig
    return True


def apply_qwen35_oq_a8_patch(
    model: Any = None,
    min_tokens: int | None = None,
) -> bool:
    """Route ``model``'s eligible Qwen3.5 prefill projections through oQ A8.

    ``model`` is the loaded model whose settings asked for this; its modules
    are tagged, and that tag is the opt-in. Only the tagged model is routed,
    so a second resident model with the feature off is untouched, and turning
    the setting off means the next load simply is not tagged -- the class
    wrapper stays installed and falls through. Called without a model, it
    installs the wrapper and leaves ``OMLX_OQ_A8`` as the only opt-in.

    ``min_tokens`` comes from the model's settings. Passing it here rather than
    through the environment keeps the engine's configuration explicit and
    testable; ``OMLX_OQ_A8_MIN_TOKENS`` still overrides, for callers driving
    the dispatcher directly. There is no kernel to choose -- the tile is picked
    per bit width from a measured default, and ``OMLX_OQ_A8_VARIANT`` overrides
    it for benchmarking.

    Idempotent, and safe to call when the kernels are unavailable -- it simply
    reports False and changes nothing.
    """
    global _MLP_PATCHED, _GDN_REGISTERED

    if not _kernels_available():
        logger.debug("oq_a8: kernels unavailable; patch skipped")
        return False

    floor = _MIN_TOKENS_DEFAULT
    if min_tokens is not None and int(min_tokens) >= 1:
        floor = int(min_tokens)
    config = OqA8Config(min_tokens=floor)

    tagged = _tag_modules(model, config) if model is not None else 0

    if not _MLP_PATCHED:
        patched = False
        for module_name, class_name in _MLP_TARGETS:
            patched |= _patch_mlp_class(module_name, class_name)
        _MLP_PATCHED = patched

    if not _GDN_REGISTERED:
        try:
            from omlx.patches.qwen35_q4_mlp import (
                register_qwen35_lm_gdn_prefill_backend,
                register_qwen35_prefill_linear_backend,
            )

            register_qwen35_lm_gdn_prefill_backend(_gdn_prefill_backend)
            register_qwen35_prefill_linear_backend(_prefill_linear_backend)
            _GDN_REGISTERED = True
        except Exception:
            logger.debug("oq_a8: GDN backend not registered", exc_info=True)

    if _MLP_PATCHED or _GDN_REGISTERED:
        logger.info(
            "oQ A8 prefill kernels enabled (mlp=%s, gdn=%s, modules=%d, "
            "min_tokens=%d)",
            _MLP_PATCHED,
            _GDN_REGISTERED,
            tagged,
            _min_tokens(config),
        )
    return _MLP_PATCHED or _GDN_REGISTERED
