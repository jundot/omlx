# SPDX-License-Identifier: Apache-2.0
"""Lossless DeepSeek-V4 Q/KV/compressor projection bundle.

This is the exact GPU alternative to the ANE attention-input experiment.  It
never dequantizes or requantizes checkpoint weights.  Q-A and raw-KV retain
their original packed MXFP8 codes/scales; both compressor pairs retain BF16.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import mlx.core as mx

from .decode_consistency import is_armed as is_dspark_verify_armed

logger = logging.getLogger(__name__)

ENABLED = os.getenv("OMLX_DSV4_QKV_BUNDLE_PREFILL", "0").strip().lower() in (
    "1",
    "true",
    "on",
    "yes",
)
QUALIFIED_DEVICES = frozenset(("Apple M3 Ultra", "Apple M5 Max"))
_PACK_LOCK = threading.Lock()
_LOGGED = False
_NAX_FAST: Any = None


@dataclass(frozen=True)
class QKVPrefillBanks:
    """Canonical banks whose row views remain the normal module fallback."""

    qkv_weight: mx.array
    qkv_scales: mx.array
    compressor_weight: mx.array
    index_compressor_weight: mx.array
    group_main_compressor: bool
    use_nax_qkv: bool


@lru_cache(maxsize=1)
def device_name() -> str:
    try:
        return str(mx.device_info().get("device_name", ""))
    except Exception:
        return ""


@lru_cache(maxsize=1)
def device_qualified() -> bool:
    global _NAX_FAST
    name = device_name()
    if name == "Apple M3 Ultra":
        return True
    if name != "Apple M5 Max":
        return False
    try:
        from omlx.custom_kernels.glm_moe_dsa import fast

        qualified = bool(
            fast.has_symbol("ds4_projection_mxfp8_qmm")
            and fast.ds4_projection_nax_kernels_built()
            and fast.ds4_projection_nax_device_available()
        )
        if qualified:
            _NAX_FAST = fast
        return qualified
    except Exception:
        return False


def _exact_config(config: Any) -> bool:
    """Fingerprint the one checkpoint whose grouping parity is qualified."""

    return (
        config.model_type == "deepseek_v4"
        and config.vocab_size == 129280
        and config.hidden_size == 4096
        and config.moe_intermediate_size == 2048
        and config.num_hidden_layers == 43
        and config.num_attention_heads == 64
        and config.n_routed_experts == 256
        and config.num_experts_per_tok == 6
        and config.max_position_embeddings == 1048576
        and config.index_n_heads == 64
        and config.index_head_dim == 128
        and config.index_topk == 512
    )


def _value(module: Any, name: str) -> mx.array | None:
    getter = getattr(module, "get", None)
    if callable(getter):
        try:
            value = getter(name)
        except (KeyError, TypeError, ValueError):
            value = None
        if isinstance(value, mx.array):
            return value
    value = getattr(module, name, None)
    return value if isinstance(value, mx.array) else None


def _has_bias(module: Any) -> bool:
    return _value(module, "bias") is not None or _value(module, "biases") is not None


def eligible_modules(attn: Any, x: mx.array) -> tuple[Any, ...] | None:
    """Return the six exact projection modules, otherwise fail closed."""

    if (
        not ENABLED
        or getattr(attn, "training", False)
        or is_dspark_verify_armed()
        or tuple(x.shape) not in ((1, 1024, 4096), (1, 2048, 4096))
        or x.dtype != mx.bfloat16
        or int(getattr(attn, "compress_ratio", -1)) != 4
        or not device_qualified()
    ):
        return None
    if x.shape[1] == 2048 and device_name() != "Apple M3 Ultra":
        return None
    cached = getattr(attn, "_omlx_qkv_prefill_modules", None)
    if isinstance(cached, tuple) and len(cached) == 6:
        return cached
    try:
        if not _exact_config(attn.config):
            return None
    except (AttributeError, TypeError, ValueError):
        return None

    # Identical rank-local code serves single-node and TP2.  It changes no
    # collective; other group widths have no recorded physical qualification.
    group = getattr(attn, "sharding_group", None)
    if group is not None:
        try:
            if int(group.size()) != 2:
                return None
        except (AttributeError, TypeError, ValueError):
            return None

    compressor = getattr(attn, "compressor", None)
    indexer = getattr(attn, "indexer", None)
    index_compressor = getattr(indexer, "compressor", None)
    modules = (
        getattr(attn, "wq_a", None),
        getattr(attn, "wkv", None),
        getattr(compressor, "wkv", None),
        getattr(compressor, "wgate", None),
        getattr(index_compressor, "wkv", None),
        getattr(index_compressor, "wgate", None),
    )
    if any(module is None or _has_bias(module) for module in modules):
        return None

    q_a, raw_kv, *dense = modules
    if any(
        getattr(module, "group_size", None) != 32
        or getattr(module, "bits", None) != 8
        or getattr(module, "mode", None) != "mxfp8"
        for module in (q_a, raw_kv)
    ):
        return None
    quantized_contract = (
        (_value(q_a, "weight"), (1024, 1024), mx.uint32),
        (_value(q_a, "scales"), (1024, 128), mx.uint8),
        (_value(raw_kv, "weight"), (512, 1024), mx.uint32),
        (_value(raw_kv, "scales"), (512, 128), mx.uint8),
    )
    if any(
        value is None or tuple(value.shape) != shape or value.dtype != dtype
        for value, shape, dtype in quantized_contract
    ):
        return None
    dense_shapes = ((1024, 4096), (1024, 4096), (256, 4096), (256, 4096))
    if any(
        (weight := _value(module, "weight")) is None
        or tuple(weight.shape) != shape
        or weight.dtype != mx.bfloat16
        for module, shape in zip(dense, dense_shapes)
    ):
        return None
    return modules


def _cached_banks(attn: Any) -> QKVPrefillBanks | None:
    banks = getattr(attn, "_omlx_qkv_prefill_banks", None)
    if not isinstance(banks, QKVPrefillBanks):
        return None
    contracts = (
        (banks.qkv_weight, (1536, 1024), mx.uint32),
        (banks.qkv_scales, (1536, 128), mx.uint8),
        (banks.compressor_weight, (2048, 4096), mx.bfloat16),
        (banks.index_compressor_weight, (512, 4096), mx.bfloat16),
    )
    if any(
        tuple(value.shape) != shape or value.dtype != dtype
        for value, shape, dtype in contracts
    ):
        return None
    return banks


def canonicalize(attn: Any, modules: tuple[Any, ...]) -> QKVPrefillBanks:
    """Concatenate original rows once, then make fallbacks zero-copy views."""

    banks = _cached_banks(attn)
    if banks is not None:
        return banks
    with _PACK_LOCK:
        banks = _cached_banks(attn)
        if banks is not None:
            return banks
        q_a, raw_kv, compressor_kv, compressor_gate, index_kv, index_gate = modules
        banks = QKVPrefillBanks(
            mx.contiguous(
                mx.concatenate((_value(q_a, "weight"), _value(raw_kv, "weight")))
            ),
            mx.contiguous(
                mx.concatenate((_value(q_a, "scales"), _value(raw_kv, "scales")))
            ),
            mx.contiguous(
                mx.concatenate(
                    (_value(compressor_kv, "weight"), _value(compressor_gate, "weight"))
                )
            ),
            mx.contiguous(
                mx.concatenate(
                    (_value(index_kv, "weight"), _value(index_gate, "weight"))
                )
            ),
            # M5 changes BF16 GEMM reduction geometry when two 1024-row main
            # banks become N=2048. M3 is physically array-equal at N=2048.
            device_name() == "Apple M3 Ultra",
            device_name() == "Apple M5 Max",
        )
        # Materialize all copies before mutating a fallback module.
        mx.eval(
            banks.qkv_weight,
            banks.qkv_scales,
            banks.compressor_weight,
            banks.index_compressor_weight,
        )
        q_a.weight, raw_kv.weight = banks.qkv_weight[:1024], banks.qkv_weight[1024:]
        q_a.scales, raw_kv.scales = banks.qkv_scales[:1024], banks.qkv_scales[1024:]
        compressor_kv.weight = banks.compressor_weight[:1024]
        compressor_gate.weight = banks.compressor_weight[1024:]
        index_kv.weight = banks.index_compressor_weight[:256]
        index_gate.weight = banks.index_compressor_weight[256:]
        object.__setattr__(attn, "_omlx_qkv_prefill_banks", banks)
        object.__setattr__(attn, "_omlx_qkv_prefill_modules", modules)
        return banks


def project_banks(x: mx.array, banks: QKVPrefillBanks) -> tuple[mx.array, ...]:
    """Run the hardware-exact banks and restore the original six views."""

    if banks.use_nax_qkv:
        if _NAX_FAST is None:
            raise RuntimeError("qualified M5 NAX projection backend is unavailable")
        qkv = _NAX_FAST.ds4_projection_mxfp8_qmm(
            x,
            banks.qkv_weight,
            banks.qkv_scales,
            variant=0,
            use_nax=True,
            nax_variant=5,
        )
    else:
        qkv = mx.quantized_matmul(
            x,
            banks.qkv_weight,
            scales=banks.qkv_scales,
            biases=None,
            transpose=True,
            group_size=32,
            bits=8,
            mode="mxfp8",
        )
    if banks.group_main_compressor:
        compressor = x @ banks.compressor_weight.T
        compressor_kv, compressor_gate = mx.split(compressor, (1024,), axis=-1)
    else:
        compressor_kv = x @ banks.compressor_weight[:1024].T
        compressor_gate = x @ banks.compressor_weight[1024:].T
    if x.shape[1] == 1024:
        index_compressor = x @ banks.index_compressor_weight.T
        index_kv, index_gate = mx.split(index_compressor, (256,), axis=-1)
    else:
        # M=2048 changes MLX's reduction geometry when the two 256-row banks
        # become N=512. Keep those two exact stock GEMMs while still grouping
        # Q-A/raw-KV and the main M3 compressor pair (4 vs 6 dispatches).
        index_kv = x @ banks.index_compressor_weight[:256].T
        index_gate = x @ banks.index_compressor_weight[256:].T
    q_a, raw_kv = mx.split(qkv, (1024,), axis=-1)
    return q_a, raw_kv, compressor_kv, compressor_gate, index_kv, index_gate


def prefill_qkv_projection_bundle(
    attn: Any, x: mx.array
) -> tuple[mx.array, ...] | None:
    """Return an exact qualified bundle or leave the path untouched."""

    modules = eligible_modules(attn, x)
    if modules is None or getattr(attn, "_omlx_qkv_prefill_failed", False):
        return None
    try:
        outputs = project_banks(x, canonicalize(attn, modules))
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        object.__setattr__(attn, "_omlx_qkv_prefill_failed", True)
        logger.warning(
            "DS4 M=1024 Q/KV/compressor bundle disabled for layer %s",
            getattr(attn, "layer_idx", "?"),
            exc_info=True,
        )
        return None
    global _LOGGED
    if not _LOGGED:
        _LOGGED = True
        tokens = int(x.shape[1]) if hasattr(x, "shape") else 1024
        dispatches = (
            4
            if tokens == 2048
            else (3 if device_name() == "Apple M3 Ultra" else 4)
        )
        logger.info(
            "DS4 exact M=%d Q/KV/compressor bundle active "
            "(original MXFP8/BF16 storage; %d dispatches)",
            tokens,
            dispatches,
        )
    return outputs
