# ruff: noqa: N806
"""Qwen3.5/3.6 quantized MLP prefill matmul patch.

This is an exact path: it replaces eligible affine QuantizedLinear calls
inside the Qwen MLP with the same MLX quantized qmm implementation exposed via
an oMLX native wrapper using a Qwen-friendly tile.  Decode and target-verify
paths fall through to the original implementation.
"""

from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Callable
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu

logger = logging.getLogger(__name__)

_PATCHED = False
_LINEAR_PATCHED = False
_LM_LINEAR_PATCHED = False
_SUPPORTED_QMM_BITS = frozenset((2, 4, 5, 6, 8))
_Q8_MIN_TOKENS = 16384


def _native_qmm_for_bits(bits: int) -> Callable[..., mx.array] | None:
    try:
        from omlx.custom_kernels.qwen35_prefill import fast
    except Exception:
        return None
    name = f"qwen35_q{bits}_affine_qmm_t"
    if bits not in _SUPPORTED_QMM_BITS or not fast.has_symbol(name):
        return None
    return getattr(fast, name)


def _qmm_supports_group_size(group_size: int) -> bool:
    try:
        from omlx.custom_kernels.qwen35_prefill import fast
    except Exception:
        return False
    return fast.qmm_supports_group_size(group_size)


def _has_native_qmm() -> bool:
    return _native_qmm_for_bits(4) is not None


def _is_supported_affine_linear_shape(
    linear: Any,
    dtype: mx.Dtype,
    ndim: int,
    seq_len: int,
    input_dim: int,
) -> bool:
    if not isinstance(linear, nn.QuantizedLinear):
        return False
    if dtype not in (mx.float16, mx.bfloat16):
        return False
    if ndim < 2 or seq_len <= 1:
        return False
    group_size = getattr(linear, "group_size", None)
    if group_size not in (64, 128):
        return False
    if not _qmm_supports_group_size(int(group_size)):
        return False
    bits = getattr(linear, "bits", None)
    if bits not in _SUPPORTED_QMM_BITS or getattr(linear, "mode", None) != "affine":
        return False
    if _native_qmm_for_bits(int(bits)) is None:
        return False
    if "bias" in linear:
        return False
    weight = getattr(linear, "weight", None)
    scales = getattr(linear, "scales", None)
    biases = getattr(linear, "biases", None)
    if weight is None or scales is None or biases is None:
        return False
    if (
        weight.dtype != mx.uint32
        or scales.dtype != dtype
        or biases.dtype != dtype
    ):
        return False
    if weight.ndim != 2 or scales.ndim != 2 or biases.ndim != 2:
        return False
    if weight.shape[1] * 32 != input_dim * int(bits):
        return False
    if input_dim % 64 != 0 or weight.shape[0] % 64 != 0:
        return False
    if scales.shape != biases.shape:
        return False
    return scales.shape[0] == weight.shape[0] and scales.shape[1] == input_dim // group_size


def _is_supported_affine_linear(linear: Any, x: mx.array) -> bool:
    return _is_supported_affine_linear_shape(
        linear,
        x.dtype,
        x.ndim,
        x.shape[-2],
        x.shape[-1],
    )


def _route_min_tokens_for_bits(
    bits: int | None, min_tokens: int, q8_min_tokens: int
) -> int:
    return q8_min_tokens if bits == 8 else min_tokens


def _can_route_affine_linear(
    linear: Any,
    x: mx.array,
    min_tokens: int,
    q8_min_tokens: int,
) -> bool:
    bits = getattr(linear, "bits", None)
    if x.shape[-2] < _route_min_tokens_for_bits(bits, min_tokens, q8_min_tokens):
        return False
    return _is_supported_affine_linear(linear, x)


def _can_route_affine_linear_shape(
    linear: Any,
    dtype: mx.Dtype,
    ndim: int,
    seq_len: int,
    input_dim: int,
    min_tokens: int,
    q8_min_tokens: int,
) -> bool:
    bits = getattr(linear, "bits", None)
    if seq_len < _route_min_tokens_for_bits(bits, min_tokens, q8_min_tokens):
        return False
    return _is_supported_affine_linear_shape(
        linear,
        dtype,
        ndim,
        seq_len,
        input_dim,
    )


def _quantized_linear_output_dim(linear: Any) -> int | None:
    weight = getattr(linear, "weight", None)
    if weight is None or getattr(weight, "ndim", 0) != 2:
        return None
    return int(weight.shape[0])


def _linear_qmm(linear: nn.QuantizedLinear, x: mx.array, variant: int) -> mx.array:
    bits = getattr(linear, "bits", None)
    qmm = _native_qmm_for_bits(int(bits)) if bits is not None else None
    if qmm is None:
        return linear(x)
    if not _is_supported_affine_linear(linear, x):
        return linear(x)
    gs = int(getattr(linear, "group_size", 64))
    return qmm(x, linear.weight, linear.scales, linear.biases, variant, gs)


def _make_patched_mlp(
    orig_call: Callable[..., mx.array],
    variant: int,
    min_tokens: int,
    q8_min_tokens: int,
):
    def patched(self, x, *args, **kwargs):
        # Decode / short-sequence fast path first: this wrapper runs on every
        # MLP call of every layer of every decode step, so the common case
        # must exit on a single shape check (issue #2132 — per-call gate
        # overhead across the qwen35 prefill patches costs ~2% TG).
        if x.ndim < 3 or x.shape[-2] < min_tokens:
            return orig_call(self, x, *args, **kwargs)
        target_verify = bool(kwargs.get("target_verify", False))
        if args and isinstance(args[0], bool):
            target_verify = target_verify or bool(args[0])
        if target_verify or os.environ.get("OMLX_QWEN35_Q4_MLP", "1") == "0":
            return orig_call(self, x, *args, **kwargs)
        gate_proj = getattr(self, "gate_proj", None)
        up_proj = getattr(self, "up_proj", None)
        down_proj = getattr(self, "down_proj", None)
        gate_dim = _quantized_linear_output_dim(gate_proj)
        if not (
            _can_route_affine_linear(gate_proj, x, min_tokens, q8_min_tokens)
            and _can_route_affine_linear(up_proj, x, min_tokens, q8_min_tokens)
            and gate_dim is not None
            and gate_dim == _quantized_linear_output_dim(up_proj)
            and _can_route_affine_linear_shape(
                down_proj,
                x.dtype,
                x.ndim,
                x.shape[-2],
                gate_dim,
                min_tokens,
                q8_min_tokens,
            )
        ):
            return orig_call(self, x, *args, **kwargs)

        gate = _linear_qmm(gate_proj, x, variant)
        up = _linear_qmm(up_proj, x, variant)
        y = swiglu(gate, up)
        return _linear_qmm(down_proj, y, variant)

    return patched


def _patch_class(
    module_name: str,
    class_name: str,
    variant: int,
    min_tokens: int,
    q8_min_tokens: int,
) -> bool:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return False
    cls = getattr(module, class_name, None)
    if cls is None or getattr(cls, "_omlx_q4_mlp_patched", False):
        return cls is not None
    orig = cls.__call__
    cls.__call__ = _make_patched_mlp(orig, variant, min_tokens, q8_min_tokens)
    cls._omlx_q4_mlp_patched = True
    cls._omlx_q4_mlp_original_call = orig
    return True


def apply_qwen35_q4_mlp_patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    if os.environ.get("OMLX_QWEN35_Q4_MLP", "1") == "0":
        return False
    if not _has_native_qmm():
        logger.debug("Qwen MLP native qmm unavailable; patch skipped")
        return False

    variant = int(os.environ.get("OMLX_QWEN35_Q4_MLP_VARIANT", "8"))
    min_tokens = int(os.environ.get("OMLX_QWEN35_Q4_MLP_MIN_TOKENS", "2048"))
    q8_min_tokens = int(
        os.environ.get("OMLX_QWEN35_Q8_MLP_MIN_TOKENS", str(_Q8_MIN_TOKENS))
    )
    patched = False
    patched |= _patch_class(
        "mlx_vlm.models.qwen3_5.language",
        "Qwen3_5MLP",
        variant,
        min_tokens,
        q8_min_tokens,
    )
    patched |= _patch_class(
        "mlx_lm.models.qwen3_5",
        "MLP",
        variant,
        min_tokens,
        q8_min_tokens,
    )
    _PATCHED = patched
    if patched:
        logger.info(
            "Qwen quantized MLP prefill patch applied "
            "(variant=%d, min_tokens=%d, q8_min_tokens=%d)",
            variant,
            min_tokens,
            q8_min_tokens,
        )
    return patched


def apply_qwen35_q4_prefill_linear_patch() -> bool:
    """Patch Qwen3.5 VLM helper linears for exact q4 prefill matmuls."""

    global _LINEAR_PATCHED
    if _LINEAR_PATCHED:
        return True
    if os.environ.get("OMLX_QWEN35_Q4_LINEAR", "1") == "0":
        return False
    if not _has_native_qmm():
        logger.debug("Qwen prefill linear native qmm unavailable; patch skipped")
        return False

    try:
        module = importlib.import_module("mlx_vlm.models.qwen3_5.language")
    except Exception:
        return False

    if getattr(module, "_omlx_q4_prefill_linear_patched", False):
        _LINEAR_PATCHED = True
        return True

    orig_linear = getattr(module, "_target_verify_linear", None)
    orig_linears = getattr(module, "_target_verify_linears", None)
    if orig_linear is None or orig_linears is None:
        return False

    variant = int(os.environ.get("OMLX_QWEN35_Q4_LINEAR_VARIANT", "8"))
    min_tokens = int(os.environ.get("OMLX_QWEN35_Q4_LINEAR_MIN_TOKENS", "2048"))
    q8_min_tokens = int(
        os.environ.get("OMLX_QWEN35_Q8_LINEAR_MIN_TOKENS", str(_Q8_MIN_TOKENS))
    )

    def should_route(linear: Any, x: mx.array, target_verify: bool) -> bool:
        # Shape gates first, env kill-switch last: the runtime toggle only
        # matters on the (rare) routed side, while decode pays this per call.
        return (
            not target_verify
            and x.ndim == 3
            and _can_route_affine_linear(linear, x, min_tokens, q8_min_tokens)
            and os.environ.get("OMLX_QWEN35_Q4_LINEAR", "1") != "0"
        )

    def patched_linear(linear, x: mx.array, target_verify: bool):
        if should_route(linear, x, target_verify):
            return _linear_qmm(linear, x, variant)
        return orig_linear(linear, x, target_verify)

    def patched_linears(linears, x: mx.array, target_verify: bool):
        if (
            x.ndim != 3
            or x.shape[-2] < min_tokens
            or target_verify
            or os.environ.get("OMLX_QWEN35_Q4_LINEAR", "1") == "0"
        ):
            return orig_linears(linears, x, target_verify)

        routes = [should_route(linear, x, target_verify) for linear in linears]
        if not any(routes):
            return orig_linears(linears, x, target_verify)

        outputs = []
        for linear, route in zip(linears, routes, strict=False):
            if route:
                outputs.append(_linear_qmm(linear, x, variant))
            else:
                outputs.append(linear(x))
        return tuple(outputs)

    module._target_verify_linear = patched_linear
    module._target_verify_linears = patched_linears
    module._omlx_q4_prefill_linear_patched = True
    module._omlx_q4_prefill_linear_original = orig_linear
    module._omlx_q4_prefill_linears_original = orig_linears
    _LINEAR_PATCHED = True
    logger.info(
        "Qwen quantized prefill linear patch applied "
        "(variant=%d, min_tokens=%d, q8_min_tokens=%d)",
        variant,
        min_tokens,
        q8_min_tokens,
    )
    return True


def apply_qwen35_q4_lm_prefill_linear_patch() -> bool:
    """Patch mlx-lm Qwen3.5/3.6 attention/GDN linears for q4 prefill."""

    global _LM_LINEAR_PATCHED
    if _LM_LINEAR_PATCHED:
        return True
    if os.environ.get("OMLX_QWEN35_Q4_LM_LINEAR", "1") == "0":
        return False
    if not _has_native_qmm():
        logger.debug("Qwen mlx-lm prefill linear native qmm unavailable")
        return False

    try:
        module = importlib.import_module("mlx_lm.models.qwen3_5")
    except Exception:
        return False

    variant = int(os.environ.get("OMLX_QWEN35_Q4_LINEAR_VARIANT", "8"))
    min_tokens = int(os.environ.get("OMLX_QWEN35_Q4_LINEAR_MIN_TOKENS", "2048"))
    q8_min_tokens = int(
        os.environ.get("OMLX_QWEN35_Q8_LINEAR_MIN_TOKENS", str(_Q8_MIN_TOKENS))
    )

    def should_route(linear: Any, x: mx.array) -> bool:
        # Shape gates first, env kill-switch last (decode pays this per call).
        return (
            x.ndim == 3
            and _can_route_affine_linear(linear, x, min_tokens, q8_min_tokens)
            and os.environ.get("OMLX_QWEN35_Q4_LM_LINEAR", "1") != "0"
        )

    def qmm_or_linear(linear: Any, x: mx.array) -> mx.array:
        if should_route(linear, x):
            return _linear_qmm(linear, x, variant)
        return linear(x)

    patched = False

    attn_cls = getattr(module, "Attention", None)
    if attn_cls is not None and not getattr(
        attn_cls, "_omlx_q4_lm_attention_patched", False
    ):
        orig_attn = attn_cls.__call__
        try:
            attn_module = importlib.import_module(attn_cls.__module__)
            sdpa = attn_module.scaled_dot_product_attention
        except Exception:
            sdpa = None

        def patched_attention(self, x, mask=None, cache=None):
            if (
                sdpa is None
                or x.ndim != 3
                or x.shape[-2] < min_tokens
                or not all(
                    should_route(linear, x)
                    for linear in (self.q_proj, self.k_proj, self.v_proj)
                )
            ):
                return orig_attn(self, x, mask=mask, cache=cache)

            B, L, _ = x.shape
            q_proj_output = qmm_or_linear(self.q_proj, x)
            queries, gate = mx.split(
                q_proj_output.reshape(B, L, self.num_attention_heads, -1),
                2,
                axis=-1,
            )
            gate = gate.reshape(B, L, -1)
            keys = qmm_or_linear(self.k_proj, x)
            values = qmm_or_linear(self.v_proj, x)

            queries = self.q_norm(queries).transpose(0, 2, 1, 3)
            keys = self.k_norm(
                keys.reshape(B, L, self.num_key_value_heads, -1)
            ).transpose(0, 2, 1, 3)
            values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
                0, 2, 1, 3
            )

            if cache is not None:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
                keys, values = cache.update_and_fetch(keys, values)
            else:
                queries = self.rope(queries)
                keys = self.rope(keys)

            output = sdpa(
                queries,
                keys,
                values,
                cache=cache,
                scale=self.scale,
                mask=mask,
            )
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            return qmm_or_linear(self.o_proj, output * mx.sigmoid(gate))

        attn_cls.__call__ = patched_attention
        attn_cls._omlx_q4_lm_attention_patched = True
        attn_cls._omlx_q4_lm_attention_original_call = orig_attn
        patched = True

    gdn_cls = getattr(module, "GatedDeltaNet", None)
    if gdn_cls is not None and not getattr(
        gdn_cls, "_omlx_q4_lm_gdn_patched", False
    ):
        orig_gdn = gdn_cls.__call__
        try:
            gdn_module = importlib.import_module(gdn_cls.__module__)
            gated_delta_update = gdn_module.gated_delta_update
        except Exception:
            gated_delta_update = getattr(module, "gated_delta_update", None)

        def patched_gdn(self, inputs, mask=None, cache=None, n_confirmed: int = 0):
            # n_confirmed is the Native-MTP draft/verify split (patches/mlx_lm_mtp).
            # Verify forwards are always far below min_tokens, so this wrapper
            # never routes them; forward the kwarg to the underlying (MTP-patched)
            # __call__ only when set, so stock GatedDeltaNet stays compatible.
            if (
                gated_delta_update is None
                or inputs.ndim != 3
                or inputs.shape[-2] < min_tokens
                or self.sharding_group is not None
            ):
                if n_confirmed:
                    return orig_gdn(
                        self, inputs, mask=mask, cache=cache, n_confirmed=n_confirmed
                    )
                return orig_gdn(self, inputs, mask=mask, cache=cache)

            input_linears = (
                self.in_proj_qkv,
                self.in_proj_z,
                self.in_proj_b,
                self.in_proj_a,
            )
            if not any(should_route(linear, inputs) for linear in input_linears):
                if n_confirmed:
                    return orig_gdn(
                        self, inputs, mask=mask, cache=cache, n_confirmed=n_confirmed
                    )
                return orig_gdn(self, inputs, mask=mask, cache=cache)

            B, S, _ = inputs.shape
            qkv = qmm_or_linear(self.in_proj_qkv, inputs)
            z = qmm_or_linear(self.in_proj_z, inputs).reshape(
                B, S, self.num_v_heads, self.head_v_dim
            )
            b = qmm_or_linear(self.in_proj_b, inputs)
            a = qmm_or_linear(self.in_proj_a, inputs)

            if cache is not None and cache[0] is not None:
                conv_state = cache[0]
            else:
                conv_state = mx.zeros(
                    (B, self.conv_kernel_size - 1, self.conv_dim),
                    dtype=inputs.dtype,
                )

            if mask is not None:
                qkv = mx.where(mask[..., None], qkv, 0)
            conv_input = mx.concatenate([conv_state, qkv], axis=1)
            if cache is not None:
                n_keep = self.conv_kernel_size - 1
                if cache.lengths is not None:
                    ends = mx.clip(cache.lengths, 0, S)
                    positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                    cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
                else:
                    cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
            conv_out = nn.silu(self.conv1d(conv_input))

            q, k, v = [
                t.reshape(B, S, h, d)
                for t, h, d in zip(
                    mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                    [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                    [self.head_k_dim, self.head_k_dim, self.head_v_dim],
                )
            ]

            state = cache[1] if cache else None
            inv_scale = k.shape[-1] ** -0.5
            q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
            k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

            out, state = gated_delta_update(
                q,
                k,
                v,
                a,
                b,
                self.A_log,
                self.dt_bias,
                state,
                mask,
                use_kernel=not self.training,
            )

            if cache is not None:
                cache[1] = state
                cache.advance(S)

            out = self.norm(out, z)
            return qmm_or_linear(self.out_proj, out.reshape(B, S, -1))

        gdn_cls.__call__ = patched_gdn
        gdn_cls._omlx_q4_lm_gdn_patched = True
        gdn_cls._omlx_q4_lm_gdn_original_call = orig_gdn
        patched = True

    _LM_LINEAR_PATCHED = patched
    if patched:
        logger.info(
            "Qwen mlx-lm quantized prefill linear patch applied "
            "(variant=%d, min_tokens=%d, q8_min_tokens=%d)",
            variant,
            min_tokens,
            q8_min_tokens,
        )
    return patched
