# SPDX-License-Identifier: Apache-2.0
"""Storage codec for GDN (Gated DeltaNet) recurrent state.

The recurrent member of an ``ArraysCache``-family layer is fp32 and, unlike
attention KV, is not sliceable: every snapshot carries the full state. On a
hybrid model such as Qwen3-Next or GLM-5.3-Flash that state dominates what a
boundary snapshot or a cache block has to persist.

This module is layout-independent. It turns one fp32 recurrent tensor into a
stored payload and back without knowing whether the caller is the SSD sidecar
(``boundary_snapshot_store``) or the embedded block layout
(``paged_ssd_cache``); each caller owns its metadata key names and counters.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:  # pragma: no cover - exercised only without MLX
    HAS_MLX = False
    mx = None


BF16_CODEC = "bf16_v1"
INT8_CODEC = "int8_rowwise_last_axis_v1"
RHT_INT8_CODEC = "rht_int8_rowwise_last_axis_v1"
RHT_INT16_CODEC = "rht_int16_rowwise_last_axis_v1"

INT8_CODECS = frozenset({INT8_CODEC, RHT_INT8_CODEC})
INT16_CODECS = frozenset({RHT_INT16_CODEC})
INTEGER_CODECS = frozenset(INT8_CODECS | INT16_CODECS)
RHT_CODECS = frozenset({RHT_INT8_CODEC, RHT_INT16_CODEC})
REDUCED_CODECS = frozenset(INTEGER_CODECS | {BF16_CODEC})

STATE_DTYPES = frozenset({"fp32", "bf16", "int8", "rht_int8", "rht_int16"})

CODEC_BY_DTYPE = {
    "fp32": "fp32",
    "bf16": BF16_CODEC,
    "int8": INT8_CODEC,
    "rht_int8": RHT_INT8_CODEC,
    "rht_int16": RHT_INT16_CODEC,
}

RHT_SEED = 0

# Metadata suffixes shared by both layouts; sidecars prefix with
# ``state_{k}_`` and embedded blocks with ``{elem_key}_``.
METADATA_CODEC = "storage_codec"
METADATA_ORIGINAL_DTYPE = "original_dtype"
METADATA_RHT_SEED = "rht_seed"
METADATA_RHT_DIM = "rht_dim"

# Reduced codecs only encode fp32 recurrent state, so any other declared
# source dtype is rejected rather than reinterpreted as fp32.
REQUIRED_ORIGINAL_DTYPE = "float32"

_ARRAYS_CACHE_SUFFIX = "ArraysCache"
_RECURRENT_STATE_INDEX = 1


def rht_dimension_supported(dim: int) -> bool:
    """Report whether the RHT codec can rotate this last-axis width."""
    return isinstance(dim, int) and dim > 0 and not (dim & (dim - 1))


@lru_cache(maxsize=32)
def rht_sign_values(dim: int, seed: int) -> tuple[float, ...]:
    """Return a version-stable random-sign diagonal without touching MLX RNG."""
    if not rht_dimension_supported(dim):
        raise ValueError(
            f"GDN RHT requires a positive power-of-two last dimension, got {dim}"
        )
    prefix = f"omlx-gdn-rht-v1:{seed}:{dim}:".encode()
    return tuple(
        (
            1.0
            if hashlib.sha256(prefix + index.to_bytes(4, "little")).digest()[0] & 1
            else -1.0
        )
        for index in range(dim)
    )


# Cache the sign array across decode graphs so a chain restore does not
# re-materialize the diagonal per layer per block.
@lru_cache(maxsize=32)
def rht_sign_array(dim: int, seed: int) -> Any:
    """Return the cached MLX fp32 sign diagonal for ``(dim, seed)``."""
    return mx.array(rht_sign_values(dim, seed), dtype=mx.float32)


def rht_forward(tensor: Any, seed: int) -> Any:
    """Apply the normalized randomized Hadamard transform on the last axis."""
    dim = int(tensor.shape[-1])
    return mx.hadamard_transform(
        tensor.astype(mx.float32) * rht_sign_array(dim, seed),
        scale=1.0 / math.sqrt(dim),
    )


def rht_inverse(tensor: Any, seed: int) -> Any:
    """Invert ``rht_forward`` on the last axis."""
    dim = int(tensor.shape[-1])
    return mx.hadamard_transform(
        tensor.astype(mx.float32),
        scale=1.0 / math.sqrt(dim),
    ) * rht_sign_array(dim, seed)


def is_arrays_cache_family(class_name: Any, cache_type: Any = None) -> bool:
    """Report whether a layer belongs to the ArraysCache family."""
    return str(class_name or "").endswith(_ARRAYS_CACHE_SUFFIX) or str(
        cache_type or ""
    ).endswith(_ARRAYS_CACHE_SUFFIX)


def is_recurrent_state_payload(
    state_index: int,
    tensor: Any,
    *,
    min_rank: int = 1,
) -> bool:
    """Select the fp32 recurrent member from an Arrays-family state tuple.

    ``min_rank=2`` skips the rank-1 ``(mx.zeros((1,)), mx.zeros((1,)))``
    placeholder stored in non-tip embedded blocks and split-GDN main blocks.
    """
    if state_index != _RECURRENT_STATE_INDEX:
        return False
    if getattr(tensor, "dtype", None) != mx.float32:
        return False
    return len(getattr(tensor, "shape", ())) >= min_rank


def is_recurrent_state_element(
    class_name: Any,
    cache_type: Any,
    state_index: int,
    tensor: Any,
    *,
    min_rank: int = 1,
) -> bool:
    """Select the fp32 recurrent member of an Arrays-family cache."""
    return is_arrays_cache_family(class_name, cache_type) and (
        is_recurrent_state_payload(state_index, tensor, min_rank=min_rank)
    )


def codec_supports_tensor(codec_or_dtype: str, tensor: Any) -> bool:
    """Report whether the configured codec can encode this tensor's width."""
    if codec_or_dtype not in {"rht_int8", "rht_int16", RHT_INT8_CODEC, RHT_INT16_CODEC}:
        return True
    shape = getattr(tensor, "shape", ())
    dim = int(shape[-1]) if shape else 0
    return rht_dimension_supported(dim) and hasattr(mx, "hadamard_transform")


@dataclass(frozen=True)
class GDNEncoding:
    """Self-describing metadata for one encoded recurrent-state payload."""

    codec: str
    original_dtype: str = REQUIRED_ORIGINAL_DTYPE
    rht_seed: int | None = None
    rht_dim: int | None = None

    @property
    def needs_scale(self) -> bool:
        """Whether this codec stores a per-row scale tensor."""
        return self.codec in INTEGER_CODECS

    @property
    def payload_dtype(self) -> Any:
        """MLX dtype expected for the stored payload tensor."""
        if self.codec in INT16_CODECS:
            return mx.int16
        if self.codec in INT8_CODECS:
            return mx.int8
        return mx.bfloat16

    def metadata_suffixes(self) -> dict[str, str]:
        """Return metadata key suffixes and values needed by a reader."""
        described = {
            METADATA_CODEC: self.codec,
            METADATA_ORIGINAL_DTYPE: self.original_dtype,
        }
        if self.rht_seed is not None:
            described[METADATA_RHT_SEED] = str(self.rht_seed)
            described[METADATA_RHT_DIM] = str(self.rht_dim)
        return described


def encode_state(
    tensor: Any,
    state_dtype: str,
    checks: list[tuple[str, tuple[int, ...], Any]],
) -> tuple[Any, Any | None, GDNEncoding]:
    """Encode one fp32 recurrent tensor for storage only.

    Validity flags are appended unevaluated to ``checks`` so callers evaluate
    a whole snapshot or block with one GPU->CPU sync via
    ``verify_encode_checks``.
    """
    checks.append(
        (
            "non-finite GDN recurrent state",
            tuple(getattr(tensor, "shape", ())),
            mx.all(mx.isfinite(tensor)),
        )
    )
    if state_dtype == "bf16":
        return tensor.astype(mx.bfloat16), None, GDNEncoding(codec=BF16_CODEC)

    encoded = tensor
    codec = INT8_CODEC
    qmax = 127
    payload_dtype = mx.int8
    rht_seed: int | None = None
    rht_dim: int | None = None
    if state_dtype == "rht_int8":
        encoded = rht_forward(tensor, RHT_SEED)
        codec = RHT_INT8_CODEC
        rht_seed, rht_dim = RHT_SEED, int(tensor.shape[-1])
    elif state_dtype == "rht_int16":
        encoded = rht_forward(tensor, RHT_SEED)
        codec = RHT_INT16_CODEC
        qmax = 32767
        payload_dtype = mx.int16
        rht_seed, rht_dim = RHT_SEED, int(tensor.shape[-1])

    scale = mx.max(mx.abs(encoded), axis=-1, keepdims=True) / qmax
    scale = mx.maximum(scale, mx.array(1e-12, dtype=mx.float32))
    # Hadamard sums can overflow fp32 on extreme finite inputs; check the scale.
    checks.append(
        (
            "non-finite GDN row scale (encoded state overflowed fp32)",
            tuple(getattr(encoded, "shape", ())),
            mx.all(mx.isfinite(scale) & (scale > 0)),
        )
    )
    quantized = mx.clip(mx.round(encoded / scale), -qmax, qmax).astype(payload_dtype)
    return (
        quantized,
        scale.astype(mx.float32),
        GDNEncoding(codec=codec, rht_seed=rht_seed, rht_dim=rht_dim),
    )


def verify_encode_checks(
    checks: list[tuple[str, tuple[int, ...], Any]],
    state_dtype: str,
) -> None:
    """Evaluate every deferred encode check with a single sync."""
    if not checks:
        return
    flags = mx.stack([flag for _reason, _shape, flag in checks])
    mx.eval(flags)
    if bool(mx.all(flags)):
        return
    reason, shape, _flag = next(
        check for check, ok in zip(checks, flags.tolist()) if not ok
    )
    raise ValueError(
        f"refusing to encode {reason} (shape={shape}, codec={state_dtype})"
    )


def parse_encoding(
    codec: str,
    original_dtype: str | None,
    rht_seed_raw: str | None,
    rht_dim_raw: str | None,
    tensor: Any,
) -> GDNEncoding:
    """Validate stored codec metadata and return a ``GDNEncoding``."""
    if codec not in REDUCED_CODECS:
        raise ValueError(f"unsupported GDN storage codec: {codec}")
    if original_dtype != REQUIRED_ORIGINAL_DTYPE:
        raise ValueError(
            f"invalid GDN source dtype for codec {codec}: "
            f"expected {REQUIRED_ORIGINAL_DTYPE}, got {original_dtype!r}"
        )
    if codec not in RHT_CODECS:
        stray = [
            name
            for name, value in (
                (METADATA_RHT_SEED, rht_seed_raw),
                (METADATA_RHT_DIM, rht_dim_raw),
            )
            if value is not None
        ]
        if stray:
            raise ValueError(
                f"GDN RHT metadata present under non-RHT codec {codec}: {stray}"
            )
        return GDNEncoding(codec=codec, original_dtype=original_dtype)

    if rht_seed_raw is None or rht_dim_raw is None:
        raise ValueError("missing GDN RHT metadata")
    # Reject whitespace, signs, and underscores that Python int() accepts.
    if not (
        isinstance(rht_seed_raw, str)
        and isinstance(rht_dim_raw, str)
        and rht_seed_raw.isdigit()
        and rht_dim_raw.isdigit()
    ):
        raise ValueError(
            f"invalid GDN RHT metadata literals: "
            f"seed={rht_seed_raw!r}, dim={rht_dim_raw!r}"
        )
    seed = int(rht_seed_raw)
    dim = int(rht_dim_raw)
    if seed != RHT_SEED:
        raise ValueError(f"unsupported GDN RHT seed: {seed}")
    if not rht_dimension_supported(dim):
        raise ValueError(f"invalid GDN RHT dimension: {dim} is not a power of two")
    if dim != int(tensor.shape[-1]):
        raise ValueError(
            f"invalid GDN RHT dimension: expected {tensor.shape[-1]}, got {dim}"
        )
    rht_sign_values(dim, seed)
    return GDNEncoding(
        codec=codec, original_dtype=original_dtype, rht_seed=seed, rht_dim=dim
    )


def validate_scale(
    tensor: Any,
    scale: Any,
    scale_key: str,
    *,
    payload_dtype: Any = None,
    checks: list[tuple[str, tuple[int, ...], Any]] | None = None,
) -> None:
    """Validate a scale tensor's shape, dtype, and finite positive values.

    Pass ``checks`` to defer the value check into a single batch sync via
    ``verify_decode_checks`` rather than syncing once per decoded layer.
    """
    if payload_dtype is None:
        payload_dtype = mx.int8
    kind = "int16" if payload_dtype == mx.int16 else "int8"
    expected_shape = tuple(tensor.shape[:-1]) + (1,)
    if tuple(scale.shape) != expected_shape:
        raise ValueError(
            f"invalid GDN {kind} scale shape for {scale_key}: "
            f"expected {expected_shape}, got {tuple(scale.shape)}"
        )
    if getattr(tensor, "dtype", None) != payload_dtype:
        raise ValueError(f"invalid GDN {kind} payload dtype for {scale_key}")
    if getattr(scale, "dtype", None) != mx.float32:
        raise ValueError(
            f"invalid GDN {kind} scale dtype for {scale_key}: "
            f"expected float32, got {getattr(scale, 'dtype', None)}"
        )
    valid = mx.all(mx.isfinite(scale) & (scale > 0))
    if checks is not None:
        checks.append(
            (
                f"invalid GDN {kind} scale values for {scale_key}",
                tuple(getattr(scale, "shape", ())),
                valid,
            )
        )
        return
    if not bool(valid):
        raise ValueError(f"invalid GDN {kind} scale values for {scale_key}")


def verify_decode_checks(
    checks: list[tuple[str, tuple[int, ...], Any]],
) -> None:
    """Evaluate every deferred decode check with a single sync."""
    if not checks:
        return
    flags = mx.stack([flag for _reason, _shape, flag in checks])
    mx.eval(flags)
    if bool(mx.all(flags)):
        return
    reason, _shape, _flag = next(
        check for check, ok in zip(checks, flags.tolist()) if not ok
    )
    raise ValueError(reason)


def decode_state(
    payload: Any,
    scale: Any | None,
    encoding: GDNEncoding,
    scale_key: str,
    checks: list[tuple[str, tuple[int, ...], Any]] | None = None,
) -> Any:
    """Restore an fp32 recurrent tensor as an unevaluated MLX graph."""
    if encoding.codec == BF16_CODEC:
        if getattr(payload, "dtype", None) != mx.bfloat16:
            raise ValueError(
                f"invalid GDN bf16 payload dtype: "
                f"expected bfloat16, got {getattr(payload, 'dtype', None)}"
            )
        return payload.astype(mx.float32)
    if encoding.codec in INTEGER_CODECS:
        if scale is None:
            kind = "int16" if encoding.codec in INT16_CODECS else "int8"
            raise ValueError(f"missing GDN {kind} scale tensor: {scale_key}")
        validate_scale(
            payload,
            scale,
            scale_key,
            payload_dtype=encoding.payload_dtype,
            checks=checks,
        )
        restored = payload.astype(mx.float32) * scale
        if encoding.codec in RHT_CODECS:
            restored = rht_inverse(restored, encoding.rht_seed)
        return restored
    raise ValueError(f"unsupported GDN storage codec: {encoding.codec}")
