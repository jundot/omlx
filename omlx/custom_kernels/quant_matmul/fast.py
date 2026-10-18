"""Quantized matmul kernels, the MoE weighted sum and the NAX probes."""

from __future__ import annotations

import logging
import platform
import re
from pathlib import Path
from typing import Any

import mlx.core as mx

from .._loader import load_extension
from .._loader import native_stream_kwargs as _native_stream_kwargs

logger = logging.getLogger(__name__)
_ext, _IMPORT_ERROR = load_extension(__package__, Path(__file__).parent)


def is_native_available() -> bool:
    return _ext is not None


def import_error() -> Exception | None:
    return _IMPORT_ERROR


def has_symbol(name: str) -> bool:
    return hasattr(_ext, name)


def set_command_buffer_caps(ops: int, mb: int) -> tuple[int, int] | None:
    """Set MLX's per-command-buffer caps and return the previous pair.

    Returns None when the native extension is unavailable; MLX keeps its caps.
    """
    setter = getattr(_ext, "set_command_buffer_caps", None)
    if setter is None:
        return None
    return tuple(setter(int(ops), int(mb)))


def share_command_queue(stream) -> int | None:
    """Move the calling thread's encoder for ``stream`` onto the shared queue.

    Returns the encoder's queue id afterward (0 for a CPU stream), or None
    when the native extension lacks the call and the stream keeps its queue.
    """
    share = getattr(_ext, "share_command_queue", None)
    if share is None:
        return None
    return int(share(stream))


# Extensions built before the NAX split reject the use_nax/nax_variant kwargs,
# so only pass them when the rebuilt binding is present.
_EXT_HAS_NAX = _ext is not None and hasattr(_ext, "is_nax_available")


# Extensions built before the Bonsai group_size=128 support reject the
# group_size kwarg on the qmm bindings, so only forward it when the rebuilt
# binding is present.  The q2 binding ships in the same rebuild, so its
# presence marks the new ABI.  Older builds only carry gs=64 kernels; callers
# must keep gs!=64 layers on stock mlx (see qmm_supports_group_size).
_EXT_HAS_QMM_GROUP_SIZE = _ext is not None and hasattr(_ext, "affine_q2_qmm_t")


def _qmm_group_size_kwargs(group_size: int) -> dict:
    if _EXT_HAS_QMM_GROUP_SIZE:
        return {"group_size": group_size}
    return {}


def qmm_supports_group_size(group_size: int) -> bool:
    """True if the compiled extension can run qmm at this group size.

    gs=64 always works (the pre-group_size builds are gs=64-only).  Other
    group sizes require the rebuilt binding; routing a gs=128 layer through
    an old build would silently run the gs=64 kernel on gs=128 data.
    """
    return group_size == 64 or _EXT_HAS_QMM_GROUP_SIZE


_NAX_ARCH_RE = re.compile(r"applegpu_g(\d+)([a-z])")


_NAX_KERNEL_NEEDLE = b"affine_qmm_t_nax"


_nax_available_cache: bool | None = None


_stock_nax_cache: bool | None = None


_qmm_nax_cache: bool | None = None


# Bundled NAX tiles (must match affine_qmm_nax_variant in affine_qmm.cpp):
#   0: 64x64x64 wm2 wn2 (stock MLX tile, default)   1: bm 32   2: bm 128
#   3: bn 128   4: bk 32   5: wm4 wn1
NAX_QMM_VARIANTS = range(6)


QMM_NAX_VARIANT = 0


def _nax_available_fallback(
    version: str | None = None, arch: str | None = None
) -> bool:
    """Python mirror of mlx metal::is_nax_available() for pre-NAX extensions."""
    if version is None or arch is None:
        if not mx.metal.is_available():
            return False
        if version is None:
            version = platform.mac_ver()[0]
        if arch is None:
            arch = str(mx.device_info().get("architecture", ""))
    try:
        release = tuple(int(part) for part in version.split(".")[:2])
    except ValueError:
        return False
    if len(release) < 2 or release < (26, 2):
        return False
    match = _NAX_ARCH_RE.fullmatch(arch)
    if match is None:
        return False
    gen = int(match.group(1))
    suffix = match.group(2)
    return gen >= (18 if suffix == "p" else 17)


def _stock_mlx_has_nax(lib_path: Path | None = None) -> bool:
    """True when the installed mlx wheel ships the NAX kernels.

    Wheels built for macOS < 26.2 are compiled with MLX_METAL_NO_NAX (e.g.
    the macosx_15 wheel in the sequoia app bundle): on those installs stock
    stays on the classic kernels even on M5 hardware, so route-to-stock
    decisions must not fire. Unreadable/absent metallibs (JIT builds) fall
    back to True, which reduces to the hardware-only gate.
    """
    global _stock_nax_cache
    if lib_path is None and _stock_nax_cache is not None:
        return _stock_nax_cache
    path = lib_path
    if path is None:
        core_file = getattr(mx, "__file__", None)
        if core_file is None:
            return True
        path = Path(core_file).parent / "lib" / "mlx.metallib"
    found = True
    try:
        if path.is_file():
            overlap = len(_NAX_KERNEL_NEEDLE) - 1
            tail = b""
            found = False
            with open(path, "rb") as f:
                while chunk := f.read(1 << 23):
                    if _NAX_KERNEL_NEEDLE in tail + chunk:
                        found = True
                        break
                    tail = chunk[-overlap:]
    except OSError:
        found = True
    if lib_path is None:
        _stock_nax_cache = found
    return found


def is_nax_available() -> bool:
    """True when stock MLX will dispatch to the M5 tensor-unit (NAX) kernels.

    Requires both NAX hardware (mirroring mlx metal::is_nax_available) and an
    mlx install whose metallib actually ships the NAX kernels.
    """
    global _nax_available_cache
    if _nax_available_cache is None:
        if _EXT_HAS_NAX:
            hardware = bool(_ext.is_nax_available())
        else:
            hardware = _nax_available_fallback()
        _nax_available_cache = hardware and _stock_mlx_has_nax()
    return _nax_available_cache


def nax_qmm_kernels_built() -> bool:
    if not _EXT_HAS_NAX:
        return False
    return bool(_ext.nax_qmm_kernels_built())


def _qmm_use_nax() -> bool:
    global _qmm_nax_cache
    if _qmm_nax_cache is None:
        _qmm_nax_cache = (
            _EXT_HAS_NAX
            and bool(_ext.is_nax_available())
            and bool(_ext.nax_qmm_kernels_built())
        )
        if _qmm_nax_cache:
            logger.info(
                "Affine qmm NAX dispatch enabled (nax_variant=%d)",
                QMM_NAX_VARIANT,
            )
    return _qmm_nax_cache


def _qmm_nax_kwargs(bits: int) -> dict[str, object]:
    # The NAX metal kernel only defines bits 4/5/6/8 (affine_qmm_nax.metal).
    # Routing a q2 call through NAX anyway hits a kernel-lookup failure that
    # latches `nax_qmm_runtime_ok=false` process-wide, permanently demoting
    # every q4/q5/q6/q8 layer to the classic kernel for the rest of the
    # process. Never request NAX for bits==2.
    if bits == 2 or not _EXT_HAS_NAX:
        return {}
    return {"use_nax": _qmm_use_nax(), "nax_variant": QMM_NAX_VARIANT}


def affine_q2_qmm_t(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    variant: int = 8,
    group_size: int = 64,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(_ext, "affine_q2_qmm_t"):
        return _ext.affine_q2_qmm_t(
            x,
            weight,
            scales,
            biases,
            variant,
            **_qmm_nax_kwargs(2),
            **_qmm_group_size_kwargs(group_size),
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError("affine_q2_qmm_t native kernel is unavailable")


def affine_q4_qmm_t(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    variant: int = 8,
    group_size: int = 64,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(_ext, "affine_q4_qmm_t"):
        return _ext.affine_q4_qmm_t(
            x,
            weight,
            scales,
            biases,
            variant,
            **_qmm_nax_kwargs(4),
            **_qmm_group_size_kwargs(group_size),
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError("affine_q4_qmm_t native kernel is unavailable")


def affine_q5_qmm_t(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    variant: int = 8,
    group_size: int = 64,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(_ext, "affine_q5_qmm_t"):
        return _ext.affine_q5_qmm_t(
            x,
            weight,
            scales,
            biases,
            variant,
            **_qmm_nax_kwargs(5),
            **_qmm_group_size_kwargs(group_size),
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError("affine_q5_qmm_t native kernel is unavailable")


def affine_q6_qmm_t(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    variant: int = 8,
    group_size: int = 64,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(_ext, "affine_q6_qmm_t"):
        return _ext.affine_q6_qmm_t(
            x,
            weight,
            scales,
            biases,
            variant,
            **_qmm_nax_kwargs(6),
            **_qmm_group_size_kwargs(group_size),
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError("affine_q6_qmm_t native kernel is unavailable")


def affine_q8_qmm_t(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    variant: int = 8,
    group_size: int = 64,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(_ext, "affine_q8_qmm_t"):
        return _ext.affine_q8_qmm_t(
            x,
            weight,
            scales,
            biases,
            variant,
            **_qmm_nax_kwargs(8),
            **_qmm_group_size_kwargs(group_size),
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError("affine_q8_qmm_t native kernel is unavailable")


def moe_weighted_sum(
    x_sorted: mx.array,
    inv_order: mx.array,
    scores: mx.array,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(_ext, "moe_weighted_sum"):
        return _ext.moe_weighted_sum(
            x_sorted,
            inv_order,
            scores,
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError("moe_weighted_sum native kernel is unavailable")


def oq_a8_available() -> bool:
    """True when the INT8 NAX GEMM can actually run on this machine.

    Distinct from ``has_symbol``: the binding can exist in a build whose NAX
    metallib was skipped (SDK < 26.2) or on hardware without tensor units.
    """
    if _ext is None or not hasattr(_ext, "oq_a8_kernels_available"):
        return False
    try:
        return bool(_ext.oq_a8_kernels_available())
    except Exception:
        return False


def oq_a8_quantize(
    x: mx.array,
    act_mode: int = 0,
    *,
    stream=None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Stage A: BF16/FP16 activations -> (Qa int8, Sa float32, Ra int16).

    Run this once per shared activation and feed the result to every
    projection that consumes it, whatever their bit widths.
    """
    if _ext is None or not hasattr(_ext, "oq_a8_quantize"):
        raise RuntimeError("oq_a8_quantize native kernel is unavailable")
    qa, sa, ra = _ext.oq_a8_quantize(
        x,
        act_mode,
        **_native_stream_kwargs(stream),
    )
    return qa, sa, ra


def oq_a8_qmm_t(
    qa: mx.array,
    sa: mx.array,
    ra: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    bits: int,
    act_mode: int = 0,
    variant: int = 800,
    *,
    packed: bool = False,
    stream=None,
) -> mx.array:
    if _ext is None or not hasattr(_ext, "oq_a8_qmm_t"):
        raise RuntimeError("oq_a8_qmm_t native kernel is unavailable")
    return _ext.oq_a8_qmm_t(
        qa,
        sa,
        ra,
        weight,
        scales,
        biases,
        bits,
        act_mode,
        variant,
        packed=packed,
        **_native_stream_kwargs(stream),
    )


def oq_a8_linear(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    bits: int,
    act_mode: int = 0,
    variant: int = 800,
    *,
    stream=None,
) -> mx.array:
    """Convenience Stage-A + GEMM for a projection with no shared activation."""
    if bits == 8:
        qa, sa, ra = oq_a8_stage_a_natural(x, act_mode, stream=stream)
    else:
        qa, sa, ra = oq_a8_stage_a_v8(x, act_mode, stream=stream)
        scales, biases = mx.contiguous(scales.T), mx.contiguous(biases.T)
    return oq_a8_qmm_t(
        qa,
        sa,
        ra,
        weight,
        scales,
        biases,
        bits,
        act_mode,
        variant,
        stream=stream,
    )


def oq_a8_stage_a_v8(
    x: mx.array,
    act_mode: int = 0,
    *,
    stream=None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Reorder activations for the native GEMM and transpose group metadata.

    Within each GS64 group, slot ``16c + 4t + j`` holds
    ``k = 16c + 8*(t>>1) + 2j + (t&1)``. This permutation preserves the
    group sum and requires a temporary contiguous INT8 activation copy.
    """
    qa, sa, ra = oq_a8_quantize(x, act_mode, stream=stream)
    shape = qa.shape
    k = shape[-1]
    m = qa.size // k
    # Reshaped back to the input's own rank, not to [M, K]: the op derives the
    # output shape from Qa, so flattening a [B, S, K] activation here would
    # hand the caller a [B*S, N] result. With B == 1 that broadcasts against
    # the residual and hides; with B > 1 it is silently wrong.
    qa = mx.contiguous(
        qa.reshape(m, k // 64, 4, 2, 4, 2).transpose(0, 1, 2, 3, 5, 4).reshape(shape)
    )
    # Flattened to [M, groups] before transposing, not transposed in place:
    # mx.transpose reverses *every* axis, so a [B, S, groups] Ra would come
    # back as [groups, S, B] -- which is the layout the kernel wants only when
    # B == 1, and silently interleaves the sequences when it is not.
    ra = mx.contiguous(ra.reshape(m, -1).T)
    if act_mode != 0:
        sa = mx.contiguous(sa.reshape(m, -1).T)
    return qa, sa, ra


def oq_a8_stage_a_natural(
    x: mx.array,
    act_mode: int = 0,
    *,
    stream=None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Stage A for the Q8 kernel: activations stay in checkpoint K order.

    Q4/Q5 use :func:`oq_a8_stage_a_v8`. Only the group metadata is
    transposed here, so there is no INT8 activation copy.
    """
    qa, sa, ra = oq_a8_quantize(x, act_mode, stream=stream)
    ra = mx.contiguous(ra.reshape(qa.size // qa.shape[-1], -1).T)
    if act_mode != 0:
        sa = mx.contiguous(sa.reshape(qa.size // qa.shape[-1], -1).T)
    return qa, sa, ra


def oq_a8_decode_weights(
    weight: mx.array,
    bits: int,
    group_count: int,
    *,
    stream=None,
) -> mx.array:
    """Unpack Q4/Q5/Q8 codes to INT8 (Q8 centered to q - 128).

    Test helper only: the production path never materializes unpacked weights
    in device memory.
    """
    if _ext is None or not hasattr(_ext, "oq_a8_decode_weights"):
        raise RuntimeError(
            "oq_a8_decode_weights native kernel is unavailable"
        )
    return _ext.oq_a8_decode_weights(
        weight,
        bits,
        group_count,
        **_native_stream_kwargs(stream),
    )


def q8_head_flat_qmm(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(_ext, "q8_head_flat_qmm"):
        return _ext.q8_head_flat_qmm(
            x,
            weight,
            scales,
            biases,
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError("q8_head_flat_qmm native kernel is unavailable")


def mxfp4_gather_qmm_blocks(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    block_meta: mx.array,
    block_count: mx.array,
    variant: int = 0,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(_ext, "mxfp4_gather_qmm_blocks"):
        return _ext.mxfp4_gather_qmm_blocks(
            x,
            weight,
            scales,
            block_meta,
            block_count,
            variant,
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError("mxfp4_gather_qmm_blocks native kernel is unavailable")


def mxfp4_gather_qmm_pair_blocks(
    x: mx.array,
    weight0: mx.array,
    scales0: mx.array,
    weight1: mx.array,
    scales1: mx.array,
    block_meta: mx.array,
    block_count: mx.array,
    variant: int = 0,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(_ext, "mxfp4_gather_qmm_pair_blocks"):
        return _ext.mxfp4_gather_qmm_pair_blocks(
            x,
            weight0,
            scales0,
            weight1,
            scales1,
            block_meta,
            block_count,
            variant,
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError(
        "mxfp4_gather_qmm_pair_blocks native kernel is unavailable"
    )


def mxfp4_gather_qmm_pair_concat_blocks(
    x: mx.array,
    weight0: mx.array,
    scales0: mx.array,
    weight1: mx.array,
    scales1: mx.array,
    block_meta: mx.array,
    block_count: mx.array,
    variant: int = 0,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(
        _ext, "mxfp4_gather_qmm_pair_concat_blocks"
    ):
        return _ext.mxfp4_gather_qmm_pair_concat_blocks(
            x,
            weight0,
            scales0,
            weight1,
            scales1,
            block_meta,
            block_count,
            variant,
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError(
        "mxfp4_gather_qmm_pair_concat_blocks native kernel is unavailable"
    )


def affine_gather_qmm_blocks(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    block_meta: mx.array,
    block_count: mx.array,
    group_size: int,
    bits: int,
    variant: int = 0,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(_ext, "affine_gather_qmm_blocks"):
        return _ext.affine_gather_qmm_blocks(
            x,
            weight,
            scales,
            biases,
            block_meta,
            block_count,
            group_size,
            bits,
            variant,
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError("affine_gather_qmm_blocks native kernel is unavailable")


def affine_gather_qmm_pair_concat_blocks(
    x: mx.array,
    weight0: mx.array,
    scales0: mx.array,
    biases0: mx.array,
    weight1: mx.array,
    scales1: mx.array,
    biases1: mx.array,
    block_meta: mx.array,
    block_count: mx.array,
    group_size: int,
    bits: int,
    variant: int = 0,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(
        _ext, "affine_gather_qmm_pair_concat_blocks"
    ):
        return _ext.affine_gather_qmm_pair_concat_blocks(
            x,
            weight0,
            scales0,
            biases0,
            weight1,
            scales1,
            biases1,
            block_meta,
            block_count,
            group_size,
            bits,
            variant,
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError(
        "affine_gather_qmm_pair_concat_blocks native kernel is unavailable"
    )


def mxfp8_qmv_pair(
    input: mx.array,
    weight_a: mx.array,
    scales_a: mx.array,
    weight_b: mx.array,
    scales_b: mx.array,
    *,
    stream=None,
) -> mx.array:
    if _ext is None or not hasattr(_ext, "mxfp8_qmv_pair"):
        raise RuntimeError("mxfp8_qmv_pair native kernel is unavailable")
    return _ext.mxfp8_qmv_pair(
        input,
        weight_a,
        scales_a,
        weight_b,
        scales_b,
        **_native_stream_kwargs(stream),
    )


def grouped_expert(gate, up, activation, down):
    """Submit one routed expert pipeline using its existing MLX primitives."""
    if _ext is not None and hasattr(_ext, "grouped_expert"):
        return _ext.grouped_expert(gate, up, activation, down)
    raise RuntimeError("grouped_expert native kernel is unavailable")


def __getattr__(name: str) -> Any:
    if _ext is not None and hasattr(_ext, name):
        return getattr(_ext, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    names = set(globals())
    if _ext is not None:
        names.update(dir(_ext))
    return sorted(names)
