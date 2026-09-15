# SPDX-License-Identifier: Apache-2.0
"""Mmap-backed per-expert slice reader for MoE banks.

Follows the _SafeTensorMMap pattern from qwen4_exp (mmap + MADV_RANDOM)
but slices a stacked bank (E, O, I) per expert id.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import bisect
import concurrent.futures
import ctypes
import fcntl
import json
import math
import logging
import mmap
import os
import re
import struct
import sys
import threading
from pathlib import Path
from typing import Any, Dict, NamedTuple, Tuple

import mlx.core as mx
import numpy as np

from ._env import env_int

_PAGE_SIZE = os.sysconf("SC_PAGESIZE") if hasattr(os, "sysconf") else 4096

_libc = ctypes.CDLL(None, use_errno=True)
_libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.mlock.restype = ctypes.c_int
_libc.munlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.munlock.restype = ctypes.c_int


class _PyBuffer(ctypes.Structure):
    _fields_ = [
        ("buf", ctypes.c_void_p),
        ("obj", ctypes.py_object),
        ("len", ctypes.c_ssize_t),
        ("itemsize", ctypes.c_ssize_t),
        ("readonly", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("format", ctypes.c_void_p),
        ("shape", ctypes.POINTER(ctypes.c_ssize_t)),
        ("strides", ctypes.POINTER(ctypes.c_ssize_t)),
        ("suboffsets", ctypes.POINTER(ctypes.c_ssize_t)),
        ("internal", ctypes.c_void_p),
    ]


_pyapi = ctypes.pythonapi
_pyapi.PyObject_GetBuffer.argtypes = [ctypes.py_object, ctypes.POINTER(_PyBuffer), ctypes.c_int]
_pyapi.PyObject_GetBuffer.restype = ctypes.c_int
_pyapi.PyBuffer_Release.argtypes = [ctypes.POINTER(_PyBuffer)]

logger = logging.getLogger(__name__)

# macOS kernel readahead (fcntl F_RDADVISE / struct radvisory): an async hint
# that pulls a file range into the page cache without copying into userspace
# — the zero-copy alternative to the warmer's discarded preads. Best-effort:
# any failure is silent.
# F_RDADVISE is 44 on Darwin (not exported by Python's fcntl module).
_F_RDADVISE = getattr(fcntl, "F_RDADVISE", 44 if sys.platform == "darwin" else None)
_RADVISORY = struct.Struct("=qi4x")  # off_t ra_offset; int ra_count; + tail pad

_MLOCK_FAILED_LOGGED = False


def _mm_page_range(mm: mmap.mmap, offset: int, length: int):
    """Page-aligned ``(base, nbytes)`` for ``mm[offset:offset+length]``.

    The mapping's base address is obtained via PyObject_GetBuffer (the
    mmap is read-only, so from_buffer's writable requirement does not
    apply). The address stays valid while ``mm`` is open — the released
    buffer export only drops a refcount, not the mapping.
    """
    view = _PyBuffer()
    try:
        if _pyapi.PyObject_GetBuffer(mm, ctypes.byref(view), 0) != 0 or not view.buf:
            return None
        try:
            start = (offset // _PAGE_SIZE) * _PAGE_SIZE
            end = min(view.len, ((offset + length + _PAGE_SIZE - 1) // _PAGE_SIZE) * _PAGE_SIZE)
            if end <= start:
                return None
            return view.buf + start, end - start
        finally:
            _pyapi.PyBuffer_Release(ctypes.byref(view))
    except Exception:
        return None


def _mlock_range(mm: mmap.mmap, offset: int, length: int) -> bool:
    """mlock a page-aligned range of an existing file mapping. Zero-copy:
    the locked pages are the file cache pages themselves."""
    global _MLOCK_FAILED_LOGGED
    try:
        r = _mm_page_range(mm, offset, length)
        if r is None:
            return False
        base, size = r
        rc = _libc.mlock(ctypes.c_void_p(base), ctypes.c_size_t(size))
        if rc != 0 and not _MLOCK_FAILED_LOGGED:
            import errno

            _MLOCK_FAILED_LOGGED = True
            logger.warning(
                "mlock failed (errno=%s) — pinned-expert mode disabled for new pins",
                errno.errorcode.get(ctypes.get_errno(), ctypes.get_errno()),
            )
        return rc == 0
    except Exception as e:
        if not _MLOCK_FAILED_LOGGED:
            _MLOCK_FAILED_LOGGED = True
            logger.warning("mlock unavailable: %s", e)
        return False


def _munlock_range(mm: mmap.mmap, offset: int, length: int) -> bool:
    """Release an mlock'd page-aligned range.

    Exists so wired pages come back without waiting for the implicit
    unlock in ``munmap`` at reader close; paired with pin bookkeeping so
    ``pinned_bytes`` cannot stay stale across an unload/reload.
    """
    try:
        r = _mm_page_range(mm, offset, length)
        if r is None:
            return False
        base, size = r
        return _libc.munlock(ctypes.c_void_p(base), ctypes.c_size_t(size)) == 0
    except Exception:
        logger.debug("munlock failed", exc_info=True)
        return False


_DTYPE_MAP: dict[str, tuple[np.dtype, int]] = {
    "BF16": (np.dtype("<u2"), 2),
    "F16": (np.dtype("<f2"), 2),
    "F32": (np.dtype("<f4"), 4),
    "U32": (np.dtype("<u4"), 4),
    "U8": (np.dtype("u1"), 1),
    "I32": (np.dtype("<i4"), 4),
    "I64": (np.dtype("<i8"), 8),
    "F8_E4M3": (np.dtype("u1"), 1),
}


def _np_to_mx(key: str, np_view: np.ndarray, dtype_str: str) -> mx.array:
    """Promote an expert's np.ndarray slice to the MLX representation."""
    if dtype_str == "BF16":
        # bf16 is stored as raw uint16 bits — reinterpret directly. This
        # matches mx.load's native handling exactly (a
        # shift->f32->astype roundtrip would flush bf16 subnormals to zero
        # via Metal FTZ) and is ~9x faster on 4 MB slices: no numpy shift,
        # half the copy bytes, no GPU conversion kernel.
        return mx.array(np_view).view(mx.bfloat16)
    if dtype_str == "F8_E4M3":
        return mx.from_fp8(mx.array(np_view), dtype=mx.bfloat16)
    return mx.array(np_view)


# Queue depth for the per-run preadvs issued inside one read_expert_into
# call. Decode demand is sparse (a handful of scattered experts per
# layer), so it has no readahead to fall back on and queue depth is all
# it has — prefill, which asks for hundreds of contiguous experts,
# already reaches the device's throughput ceiling even serially.
#
# Default 16 sits at the device's useful queue depth; deeper
# oversubscribes it and regresses. OMLX_EXPERT_STREAMING_RUN_QD overrides.
_RUN_IO_QD = env_int("OMLX_EXPERT_STREAMING_RUN_QD", 16, lo=1)

# NOTE: the singleton and the accessor must not share a name — a module-level
# def _run_io_pool would rebind the very global the accessor is meant to
# populate, and it would then return itself instead of an executor.
_RUN_IO_POOL_SINGLETON: ThreadPoolExecutor | None = None
_run_io_pool_lock = threading.Lock()


def _run_io_pool() -> ThreadPoolExecutor:
    """Threads for the per-run preadvs of read_expert_into calls.

    Deliberately NOT the caller's pool. read_expert_into is dispatched on
    _EXPERT_IO_POOL (16 workers) via the layer-context prefetch and the
    union path's pool.map, so submitting to that same bounded pool and then
    waiting would deadlock as soon as every worker was a parent blocked on a
    queued child. This pool is separate and its tasks never submit anywhere,
    so it always drains.

    The SINGLETON is bounded at _RUN_IO_QD workers PROCESS-WIDE: every
    concurrent parent shares the same run-read workers — the executor caps
    the process, so extra workers oversubscribe the device's useful queue
    depth rather than multiply it. A single call keeps at most _RUN_IO_QD
    reads in flight because its planning window is sized to the pool; the
    bound also keeps the transient buffer memory bounded.
    """
    global _RUN_IO_POOL_SINGLETON
    if _RUN_IO_POOL_SINGLETON is None:
        with _run_io_pool_lock:
            if _RUN_IO_POOL_SINGLETON is None:
                _RUN_IO_POOL_SINGLETON = ThreadPoolExecutor(
                    max_workers=_RUN_IO_QD,
                    thread_name_prefix="omlx-expert-run",
                )
    return _RUN_IO_POOL_SINGLETON


# Every blocking wait on an IO future is bounded. A stalled, wedged or
# removed NVMe would otherwise park the MLX inference thread in
# fut.result() forever with no recovery path and no way to cancel on
# request abort. Timing out turns that hang into a failed read, which the
# existing fallback path already handles. The default is deliberately
# generous (a cold 256 MiB bank run at even 50 MB/s finishes in ~5 s);
# set 0 to disable.
def _io_timeout_s(default: float = 120.0) -> float:
    try:
        raw = os.environ.get("OMLX_EXPERT_STREAMING_IO_TIMEOUT_S", "")
        return float(raw) if raw.strip() else default
    except (TypeError, ValueError):
        return default


_IO_TIMEOUT_S = max(0.0, _io_timeout_s())
_IO_TIMEOUT_LOGGED = False


def _log_io_timeout() -> None:
    global _IO_TIMEOUT_LOGGED
    if _IO_TIMEOUT_LOGGED:
        return
    _IO_TIMEOUT_LOGGED = True
    logger.error(
        "expert streaming: IO read exceeded %.0fs — the device may be stalled "
        "or gone; failing this read (fallback path) instead of hanging",
        _IO_TIMEOUT_S,
    )


def _await_io_future(fut) -> bool:
    """Block on one IO future with a bounded timeout; True when it succeeded.

    Returns False (never raises) on timeout, cancellation or read error, so
    callers keep their existing "ok = False -> fallback" shape.
    """
    global _IO_TIMEOUT_LOGGED
    try:
        if _IO_TIMEOUT_S > 0:
            fut.result(timeout=_IO_TIMEOUT_S)
        else:
            fut.result()
        return True
    except concurrent.futures.TimeoutError:
        _log_io_timeout()
        return False
    except Exception:
        return False


def segment_runs(
    eids_sorted: list[int],
    *,
    same: Any | None = None,
    merge_gap: int = 0,
    max_run: int | None = None,
) -> list[tuple[int, int]]:
    """Split ascending expert ids into (first, count) runs.

    ONE shared segmentation for the demand path (_group_runs), the
    advisor and read_expert_into, so the three callers can never
    diverge: a run groups CONSECUTIVE ids while ``same(first, nxt)``
    holds (reader identity for tier-aware paths; tier match for the demand
    fallback).

    With ``merge_gap > 0`` a run may BRIDGE a hole of up to merge_gap
    missing ids when the next demanded id still satisfies ``same`` — the
    hole rows are read with the run but the caller scatters only the
    demanded ids (gap rows never enter the output, the LRU or any promote).
    ``max_run`` bounds the run length (a bridge clamps to the cap).
    """
    if not eids_sorted:
        return []
    same = same if same is not None else (lambda a, b: True)
    limit = max_run if max_run is not None and max_run > 0 else None
    runs: list[tuple[int, int]] = []
    i = 0
    n = len(eids_sorted)
    while i < n:
        first = eids_sorted[i]
        count = 1
        j = i + 1
        while j < n:
            if limit is not None and count >= limit:
                break
            nxt = eids_sorted[j]
            gap = nxt - (first + count)
            if gap == 0 and same(first, nxt):
                count += 1
                j += 1
                continue
            if merge_gap > 0 and 1 <= gap <= merge_gap and same(first, nxt):
                add = gap + 1
                if limit is not None and count + add > limit:
                    add = max(1, limit - count)
                count += add
                # Consume nxt only when the bridge fully covers it; a
                # clamped bridge leaves nxt for the NEXT run (the
                # max_run clamp must not swallow demanded ids).
                if add == gap + 1:
                    j += 1
                continue
            break
        runs.append((first, count))
        i = j
    return runs

class _ReadParams(NamedTuple):
    """Immutable, precomputed per-key read parameters.

    The safetensors header is fixed after __init__, so once derived a key's
    params never need re-validation on the hot path (no per-call shape/dtype
    re-derivation, no repeated size-mismatch arithmetic).
    """

    shape: tuple[int, ...]
    np_dtype: np.dtype
    item: int
    num_experts: int
    expert_bytes: int
    tensor_abs_off: int

    @property
    def per_shape(self) -> tuple[int, ...]:
        return self.shape[1:] if len(self.shape) > 1 else self.shape


class _ShardReader:
    def __init__(self, path: Path):
        self.path = path
        self._file = path.open("rb")
        self._rp: dict[str, _ReadParams] = {}
        hsize = struct.unpack("<Q", self._file.read(8))[0]
        self.header: dict = json.loads(self._file.read(hsize))
        self.data_start = 8 + hsize
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            self._mmap.madvise(mmap.MADV_RANDOM)  # type: ignore[attr-defined]
        except Exception:
            pass

    def close(self) -> None:
        try:
            if self._mmap is not None:
                self._mmap.close()
        except Exception:
            pass
        try:
            if self._file is not None:
                self._file.close()
        except Exception:
            pass
        self._mmap = None  # type: ignore[assignment]
        self._file = None  # type: ignore[assignment]

    def _rp_for(self, key: str) -> _ReadParams:
        """Precompute (and cache) the immutable per-key read parameters.

        The safetensors header is fixed after __init__, so once derived a key's
        params never need re-validation on the hot path (no per-call shape/dtype
        re-derivation, no repeated size-mismatch arithmetic). Raises ValueError
        on a header size mismatch — computed once, then cached so it stays stable.
        """
        rp = self._rp.get(key)
        if rp is not None:
            return rp
        entry = self.header[key]
        shape = tuple(entry["shape"])
        dtype_str = str(entry["dtype"])
        np_dtype, item = _DTYPE_MAP.get(dtype_str, (None, None))  # type: ignore[assignment]
        if np_dtype is None:
            raise TypeError(f"Unsupported dtype {dtype_str} for {key}")
        start, end = entry["data_offsets"]
        n_elements = 1
        for d in shape:
            n_elements *= int(d)
        expected_bytes = n_elements * int(item)
        if (end - start) != expected_bytes:
            raise ValueError(f"Header size mismatch for {key}: {end-start} vs {expected_bytes}")
        num_experts = shape[0] if shape else 1
        expert_bytes = (end - start) // num_experts
        if expert_bytes % item != 0:
            # Slices would misalign against the dtype; surface it rather than
            # produce a silently-corrupt view.
            raise ValueError(
                f"Unaligned expert slice for {key}: expert_bytes={expert_bytes} "
                f"not divisible by itemsize {item}"
            )
        rp = _ReadParams(
            shape=shape,
            np_dtype=np_dtype,
            item=item,
            num_experts=num_experts,
            expert_bytes=expert_bytes,
            tensor_abs_off=self.data_start + int(start),
        )
        self._rp[key] = rp
        return rp

    def _read_into(self, abs_off: int, out: np.ndarray) -> None:
        """Zero-copy read of out.nbytes bytes at abs_off into the writable
        uint8 buffer out.

        os.preadv writes straight into the buffer — no intermediate heap copy.
        Raises OSError on a short read or IO error. Falls back to os.pread +
        copy when preadv is unavailable; short reads still surface as OSError
        so the caller can fail-high.
        """
        n = out.nbytes
        fd = self._file.fileno()
        try:
            got = os.preadv(fd, [memoryview(out)], abs_off)
        except (AttributeError, OSError):
            data = os.pread(fd, n, abs_off)
            if len(data) != n:
                raise OSError(f"Short read at {abs_off}: {len(data)} of {n}") from None
            out[:] = np.frombuffer(data, dtype=np.uint8)
            return
        if got != n:
            raise OSError(f"Short read (preadv) at {abs_off}: {got} of {n}") from None

    def _read_iovecs(self, abs_off: int, iovecs: list) -> None:
        """Scatter-gather preadv: each iovec receives its row in order.

        Used when a run's rows map 1:1 onto the output rows (no bridged
        holes), so the kernel lands every expert straight at its demand
        position — no temp buffer, no userspace copy.
        """
        n = sum(iov.nbytes for iov in iovecs)
        fd = self._file.fileno()
        try:
            got = os.preadv(fd, iovecs, abs_off)
        except (AttributeError, OSError):
            data = os.pread(fd, n, abs_off)
            if len(data) != n:
                raise OSError(
                    f"Short read at {abs_off}: {len(data)} of {n}"
                ) from None
            pos = 0
            for iov in iovecs:
                iov[:] = data[pos : pos + iov.nbytes]
                pos += iov.nbytes
            return
        if got != n:
            raise OSError(
                f"Short read (preadv) at {abs_off}: {got} of {n}"
            ) from None

    def expert_slice(self, key: str, expert_id: int) -> np.ndarray:
        rp = self._rp_for(key)
        if expert_id < 0 or expert_id >= rp.num_experts:
            raise ValueError(
                f"expert_slice: id {expert_id} out of range "
                f"(num_experts={rp.num_experts}) for {key}"
            )
        off = rp.tensor_abs_off + expert_id * rp.expert_bytes
        # Single zero-copy preadv into a writable buffer; the typed view is a
        # zero-copy reinterpret (no bytearray double-copy).
        out = np.empty(rp.expert_bytes, dtype=np.uint8)
        self._read_into(off, out)
        return np.frombuffer(out, dtype=rp.np_dtype).reshape(rp.per_shape)

    def expert_byte_range(self, key: str, expert_id: int) -> Tuple[int, int]:
        """Absolute file offsets (start, end) of one expert's slice."""
        rp = self._rp_for(key)
        if expert_id < 0 or expert_id >= rp.num_experts:
            raise ValueError(
                f"expert_byte_range: id {expert_id} out of range "
                f"(num_experts={rp.num_experts}) for {key}"
            )
        off = rp.tensor_abs_off + expert_id * rp.expert_bytes
        return off, off + rp.expert_bytes

    def advise_range(self, offset: int, length: int) -> bool:
        """Kernel readahead hint (F_RDADVISE) for one file range.

        Tells macOS to start pulling [offset, offset+length) into the page
        cache asynchronously — no userspace copy, no buffer, nothing to
        free. Used to overlap the NVMe fetch of a predicted demand set with
        GPU compute. Best-effort: returns False on any failure.
        """
        if _F_RDADVISE is None or length <= 0:
            return False
        try:
            fd = self._file.fileno()
            end = offset + length
            pos = offset
            while pos < end:
                chunk = min(end - pos, 0x7FFFFFFF)
                fcntl.fcntl(fd, _F_RDADVISE, _RADVISORY.pack(pos, chunk))
                pos += chunk
            return True
        except Exception:
            return False

    def expert_run(self, key: str, first_id: int, count: int) -> list:
        """One preadv covering experts [first_id, first_id+count), sliced per expert.

        Row-major stacked banks give consecutive ids contiguous offsets, so a
        run reads back as one sequential transfer — fewer syscalls and larger
        requests than per-expert preads (matters most when the demand set is
        dense, i.e. long-prompt prefill). The single buffer stays alive via the
        returned per-expert views (zero-copy, no bytearray double-copy).
        """
        rp = self._rp_for(key)
        first_id = int(first_id)
        count = int(count)
        # Reject instead of silently clamping: a `first_id=-1` or an
        # overflowing run would read the WRONG experts while reporting
        # them as the requested ids.
        if first_id < 0 or first_id >= rp.num_experts:
            raise ValueError(
                f"expert_run: first_id {first_id} out of range "
                f"(num_experts={rp.num_experts}) for {key}"
            )
        if count < 1 or first_id + count > rp.num_experts:
            raise ValueError(
                f"expert_run: range [{first_id}, {first_id + count}) exceeds "
                f"num_experts={rp.num_experts} for {key}"
            )
        off = rp.tensor_abs_off + first_id * rp.expert_bytes
        buf = np.empty(rp.expert_bytes * count, dtype=np.uint8)
        self._read_into(off, buf)
        per = rp.per_shape
        per_elements = rp.expert_bytes // rp.item
        return [
            np.frombuffer(buf, dtype=rp.np_dtype, count=per_elements, offset=i * rp.expert_bytes).reshape(per)
            for i in range(count)
        ]
    def pin_expert(self, key: str, expert_id: int) -> int:
        """mlock the page-aligned file range of one expert slice.

        Returns the locked byte count (page-rounded) or 0 on failure. The
        locked pages are the file-cache pages themselves — no copy, no
        committed anonymous memory (wired, though: it cannot be evicted).
        """
        off, end = self.expert_byte_range(key, expert_id)
        length = end - off
        ok = _mlock_range(self._mmap, off, length)
        if not ok:
            return 0
        start = (off // _PAGE_SIZE) * _PAGE_SIZE
        end_pg = min(len(self._mmap), ((end + _PAGE_SIZE - 1) // _PAGE_SIZE) * _PAGE_SIZE)
        return end_pg - start

    def unpin_expert(self, key: str, expert_id: int) -> int:
        """munlock the range pinned by ``pin_expert``; 0 when nothing moved."""
        try:
            off, end = self.expert_byte_range(key, expert_id)
        except Exception:
            return 0
        length = end - off
        if length <= 0:
            return 0
        if not _munlock_range(self._mmap, off, length):
            return 0
        start = (off // _PAGE_SIZE) * _PAGE_SIZE
        end_pg = min(len(self._mmap), ((end + _PAGE_SIZE - 1) // _PAGE_SIZE) * _PAGE_SIZE)
        return end_pg - start


_COLD_BANK_MARKERS = (
    ".switch_mlp.gate_proj.",
    ".switch_mlp.up_proj.",
    ".switch_mlp.down_proj.",
    ".switch_mlp.gate_up_proj.",
)


# The env override is the bench/developer opt-in — an EMPTY env
# means "no opinion" (None) so the runtime contract stays "unset = uniform
# tier"; the settings key (expert_streaming_hot_fraction) is the per-model UI.
HOT_FRACTION_ENV: str | None = os.environ.get("OMLX_EXPERT_STREAMING_HOT_FRACTION", "") or None


def load_hot_set_from_profile(
    profile_path: str | Path,
    hot_fraction: float,
    num_experts: int | None = None,
) -> dict[str, set]:
    """HOBBIT hot set from a learned pin profile.

    The profile's `freq` maps layer -> [[expert, count], ...]; the top
    ceil(fraction * experts) by count per layer keep the ORIGINAL packing
    while the rest read the cold tier. Keys are the backing's bare layer
    keys ("layer_<i>"); MTP stages are absent from profiles (uniform cold
    there). Missing profile → empty dict (split stays off, uniform cold).

    ``num_experts`` is the REAL per-layer expert width from the
    model estimate; the fraction's denominator must be it, not the number
    of recorded profile entries — the profile keep cap (and old profiles)
    truncate the record list, so len(counts) would elect an arbitrary
    id-prefix subset on a wide model. The hot count is
    still clamped to the available records (a sparse profile cannot elect
    experts it never observed)."""
    try:
        data = json.loads(Path(profile_path).read_text())
        # Profile v2: the HOBBIT hot set is the DECODE regime (the
        # split targets decode-hot experts); v1 profiles keep the
        # top-level freq.
        regimes = data.get("regimes")
        freq = None
        if isinstance(regimes, dict):
            decode_regime = regimes.get("decode")
            if isinstance(decode_regime, dict):
                freq = decode_regime.get("freq") or {}
        if not freq:
            freq = data.get("freq") or {}
        if not freq or hot_fraction <= 0.0:
            return {}
        hot: dict[str, set] = {}
        for layer_key, pairs in freq.items():
            counts = [(int(e), int(c)) for e, c in pairs]
            if not counts:
                continue
            width = num_experts if num_experts and num_experts > 0 else len(counts)
            n_hot = max(1, math.ceil(hot_fraction * width))
            n_hot = min(n_hot, len(counts))  # cannot elect unseen experts
            top = sorted(counts, key=lambda kv: (-kv[1], kv[0]))[:n_hot]
            hot[f"layer_{int(layer_key)}"] = {e for e, _ in top}
        return hot
    except Exception:
        return {}

def _cold_tier_status_dir(cold_dir: Path, model_path: Path) -> tuple[bool, str]:
    """Is the tier rooted at *cold_dir* complete for *model_path*?

    Complete = every switch_mlp bank weight key of the checkpoint exists in
    some expert_cold/ shard header (partial tiers are rejected: the runtime
    uniform-packing assumption would silently break). The dir may be the
    default <model>/expert_cold or an OMLX_EXPERT_STREAMING_COLD_ROOT
    override — same completeness rule either way."""
    if not cold_dir.is_dir():
        return False, f"cold tier dir missing: {cold_dir}"
    index = model_path / "model.safetensors.index.json"
    if not index.is_file():
        return False, "no model.safetensors.index.json"
    try:
        weight_map = json.loads(index.read_text()).get("weight_map") or {}
    except Exception as e:
        return False, f"unreadable index: {e}"
    needed = {
        key
        for key in weight_map
        if key.endswith(".weight") and any(m in key for m in _COLD_BANK_MARKERS)
    }
    if not needed:
        return False, "no expert banks in the checkpoint"
    have: set[str] = set()
    for shard in cold_dir.glob("*.safetensors"):
        try:
            have.update(_read_header_keys(shard))
        except Exception:
            continue
    missing = needed - have
    if missing:
        return False, f"{len(missing)} bank key(s) missing from expert_cold/"
    return True, f"complete ({len(needed)} banks)"


def _read_header_keys(path: Path) -> set[str]:
    with path.open("rb") as f:
        hsize = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hsize))
    return {k for k in hdr if k != "__metadata__"}


class ExpertBackingStore:
    """Open shard readers and serve per-expert mx.arrays on demand."""

    def __init__(
        self,
        model_path: str | Path,
        extra_roots: list[str | Path] | None = None,
        cold_root: str | Path | None = None,
    ):
        self.model_path = Path(model_path).expanduser().resolve()
        # Optional stripe roots: per-shard files are resolved primary-first;
        # roots listed here win for shards mirrored onto them (see
        # _resolve_file) — used to stripe a big MoE across two SSDs.
        self._extra_roots = [Path(r).expanduser().resolve() for r in (extra_roots or [])]
        # Cold precision tier: when set, expert-bank keys that
        # exist under <model>/expert_cold/ resolve there FIRST — the whole
        # runtime (slices, runs, pins, readahead, dtypes) reads the cold
        # packing uniformly. Same filenames/key names, lower bit width.
        self.cold_root = Path(cold_root).expanduser().resolve() if cold_root else None
        self._cold_readers: Dict[str, _ShardReader] = {}
        self._cold_key_to_reader: Dict[str, _ShardReader] = {}
        # HOBBIT hot/cold split: {stacked_key_prefix -> set(expert_id)}
        # of experts served from the ORIGINAL (higher-precision) shards while
        # the rest read expert_cold/. Empty/absent = uniform cold tier.
        # Keyed by the bank prefix (…switch_mlp.<proj>) with a
        # per-layer fallback key ("layer_<i>") for profile sources that only
        # know the layer.
        self._hot_experts: Dict[str, set] = {}
        self._readers: Dict[str, _ShardReader] = {}
        self._key_to_reader: Dict[str, _ShardReader] = {}
        # Lazy reader resolution runs on the inference thread and on
        # warm-pool workers at the same time. The resolve must be atomic so
        # exactly one _ShardReader per path is ever published (and the
        # duplicate gets closed instead of leaking its fd + mmap).
        self._readers_lock = threading.Lock()
        # Store-level closed latch: reader CREATION re-checks it under
        # _readers_lock so a close() racing a lazy resolve can never
        # publish a fresh fd+mmap after the dicts were swapped out (the
        # plan-level _closed guards one frame up; this one is the store's
        # own no-resurrection guarantee).
        self._closed = False
        # Header cache must exist before _load_weight_map: the no-index
        # fallback path calls _header_for_file, which reads it.
        self._header_cache: Dict[str, dict] = {}
        weight_map = self._load_weight_map()
        # We lazily open per key, but also keep weight_map for lookup
        self._weight_map = weight_map
        # mlock pin tracking (expert_streaming pin mode). Pins are taken
        # from warm-pool workers (warmer._WARM_POOL) as well as the
        # inference thread, and close()/unpin_all() can run concurrently
        # with them, so every touch of these three fields is serialized.
        self._pin_lock = threading.Lock()
        self._pinned: set = set()
        self.pinned_bytes = 0
        # Unique locked page accounting per reader file, so
        # pinned_bytes reports the true wired bytes after page dedupe.
        self._pinned_pages: dict[str, set[int]] = {}
        # A dir carrying expert_order.json is a repacked co-activation
        # checkpoint — expert ordering is unsupported. Serving it as
        # identity would silently return permuted experts — refuse loudly.
        if (self.model_path / "expert_order.json").is_file():
            raise ValueError(
                "repacked co-activation checkpoint (expert_order.json): "
                "ordering support was removed — use the original checkpoint"
            )
    def _roots(self) -> list[Path]:
        return [self.model_path, *self._extra_roots]

    def absorb_extra_map(self, root: str | Path, mapping: dict[str, str]) -> int:
        """Serve spill/derived shards under *root* for *mapping* keys.

        Registers *root* for file resolution (extra roots win) and maps
each key to its shard filename, so stacked banks spilled outside the
        checkpoint dir (dsv4 spill-stacking) resolve without header
        scans. Returns the number of keys absorbed. Idempotent.
        """
        rp = Path(root).expanduser().resolve()
        if rp not in self._extra_roots:
            self._extra_roots.append(rp)
        n = 0
        for key, fname in mapping.items():
            if self._weight_map.get(key) != fname:
                self._weight_map[key] = fname
                n += 1
        return n

    def _resolve_file(self, fname: str) -> Path | None:
        """Resolve a shard filename across roots (extra roots win: mirrored
        shards on the stripe SSD are the ones we want served from there)."""
        for root in reversed(self._roots()):
            p = root / fname
            if p.is_file():
                return p
        return None

    def _load_weight_map(self) -> Dict[str, str]:
        idx = self.model_path / "model.safetensors.index.json"
        if idx.is_file():
            try:
                return json.loads(idx.read_text()).get("weight_map") or {}
            except Exception:
                return {}
        # fallback: single shard case — enumerate headers
        wm: Dict[str, str] = {}
        for shard in self.model_path.glob("*.safetensors"):
            hdr = self._header_for_file(shard)
            for k in hdr.keys():
                if k == "__metadata__":
                    continue
                wm[k] = shard.name
        return wm

    def _header_for_file(self, path: Path) -> dict:
        key = str(path)
        if key in self._header_cache:
            return self._header_cache[key]
        try:
            with path.open("rb") as f:
                hsize = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(hsize))
                self._header_cache[key] = hdr
                return hdr
        except Exception:
            return {}

    def set_hot_experts(self, hot: Dict[str, set] | None) -> None:
        """Declare the HOBBIT hot set: experts that keep the ORIGINAL packing
        while the cold tier is active. Keys are stacked-bank prefixes (as in
        the weight map, e.g. "…switch_mlp.gate_proj") and/or bare layer keys
        ("layer_12"); values are expert-id sets. None/empty = uniform cold."""
        self._hot_experts = {
            str(k): {int(e) for e in v} for k, v in (hot or {}).items() if v
        }

    def _hot_key_for(self, key: str, expert_id: int | None) -> str | None:
        """Hot-set key matching *key*/*expert_id*, or None when not hot."""
        if not self._hot_experts or expert_id is None:
            return None
        # bank prefix: strip the trailing ".weight"/".scales"/".biases"
        prefix = key
        for suffix in (".biases", ".scales", ".weight"):
            if prefix.endswith(suffix):
                prefix = prefix[: -len(suffix)]
                break
        if prefix in self._hot_experts and int(expert_id) in self._hot_experts[prefix]:
            return prefix
        # layer_<i> fallback: any prefix containing ".layers.<i>."
        m = re.search(r"\.layers\.(\d+)\.", key)
        if m:
            lk = f"layer_{m.group(1)}"
            if lk in self._hot_experts and int(expert_id) in self._hot_experts[lk]:
                return lk
        return None

    def _cold_reader_for_key(
        self, key: str, expert_id: int | None = None
    ) -> _ShardReader | None:
        """Cold-tier reader for *key*, or None (not in the cold root).
        HOBBIT hot experts never resolve cold — they stay on the
        original packing, so the demand path must fetch them from the
        source shards and compute at the source bits."""
        if self.cold_root is None:
            return None
        if self._hot_key_for(key, expert_id) is not None:
            return None
        if key in self._cold_key_to_reader:
            return self._cold_key_to_reader[key]
        fname = self._weight_map.get(key)
        if fname is None:
            return None
        cold_path = self.cold_root / fname
        if not cold_path.is_file():
            return None
        ckey = str(cold_path)
        with self._readers_lock:
            if self._closed:
                # Closed store answers "no cold tier"; the source-reader
                # path the caller falls back to raises the real error.
                return None
            reader = self._cold_readers.get(ckey)
            if reader is None:
                try:
                    mine = _ShardReader(cold_path)
                except Exception:
                    return None
                # Same canonical-instance rule as _reader_for_key_source.
                winner = self._cold_readers.setdefault(ckey, mine)
                if winner is not mine:
                    try:
                        mine.close()
                    except Exception:
                        logger.debug("closing duplicate cold reader failed", exc_info=True)
                reader = winner
            if key not in reader.header:
                return None
            self._cold_key_to_reader.setdefault(key, reader)
            return self._cold_key_to_reader[key]

    def _reader_for_key(self, key: str, expert_id: int | None = None) -> _ShardReader:
        # expert_id routes the HOBBIT split: a hot expert resolves the
        # ORIGINAL shard even when a cold copy exists; everyone else cold.
        # The no-id lookup stays tier-blind (cold-first) for dtype/metadata
        # probes — never fall into it from the id-aware path when the id is
        # hot, or a hot expert would land on the cold packing.
        if expert_id is not None:
            k2 = (key, int(expert_id))
            cached = self._key_to_reader.get(k2)
            if cached is not None:
                return cached
            cold = self._cold_reader_for_key(key, int(expert_id))
            reader = cold if cold is not None else self._reader_for_key_source(key)
            # Concurrent lazy opens must return ONE canonical
            # instance — two threads resolving the same key simultaneously
            # could otherwise hand out different reader objects and the
            # tier contract (all ids -> the same reader) would reject the
            # component with a False.
            return self._key_to_reader.setdefault(k2, reader)
        if key in self._key_to_reader:
            return self._key_to_reader[key]
        cold = self._cold_reader_for_key(key)
        if cold is not None:
            return cold
        return self._reader_for_key_source(key)

    def _reader_for_key_source(self, key: str) -> _ShardReader:
        """Resolve the ORIGINAL shard for *key* — never the cold tier."""
        if ("src", key) in self._key_to_reader:
            return self._key_to_reader[("src", key)]
        fname = self._weight_map.get(key)
        if fname is None:
            # key may be a stacked name not in weight_map for sharded raw experts case;
            # try to find file containing key by scanning headers
            for root in self._roots():
                for shard in root.glob("*.safetensors"):
                    hdr = self._header_for_file(shard)
                    if key in hdr:
                        fname = shard.name
                        break
                if fname is not None:
                    break
        if fname is None:
            raise KeyError(f"Expert key {key!r} not in weight_map and not found in any shard")
        fpath = self._resolve_file(fname)
        if fpath is None:
            raise FileNotFoundError(f"Shard {fname!r} not found in any root of {self.model_path}")
        fkey = str(fpath)
        with self._readers_lock:
            if self._closed:
                raise RuntimeError(
                    "ExpertBackingStore is closed — cannot open reader for "
                    f"{key!r}"
                )
            reader = self._readers.get(fkey)
            if reader is None:
                reader = _ShardReader(fpath)
                # Canonical instance under concurrency (see
                # _reader_for_key): publish first, then close the loser
                # so its file descriptor and mmap do not leak for the
                # engine lifetime.
                winner = self._readers.setdefault(fkey, reader)
                if winner is not reader:
                    try:
                        reader.close()
                    except Exception:
                        logger.debug("closing duplicate shard reader failed", exc_info=True)
                    reader = winner
            # Canonical instance, so the plain store cannot publish two
            # different readers for one key (the tier contract in
            # read_expert_into checks "if any(r is not reader)").
            self._key_to_reader[("src", key)] = reader
        return reader

    def cold_quant_params(self, key: str) -> Tuple[int, int] | None:
        """(bits, group_size) of the cold packing for *key*, from the cold
        shard's __metadata__ — None when the cold tier is not active for it."""
        cold = self._cold_reader_for_key(key)
        if cold is None:
            return None
        meta = cold.header.get("__metadata__") or {}
        try:
            return int(meta["omlx_cold_bits"]), int(meta["omlx_cold_group_size"])
        except (KeyError, TypeError, ValueError):
            return None
    def tensor_dtype(self, key: str) -> str | None:
        """Safetensors dtype string for *key* (e.g. "U32", "BF16"), or None."""
        try:
            reader = self._reader_for_key(key)
            return str(reader.header[key]["dtype"])
        except Exception:
            return None

    def expert_bytes(self, key: str) -> int:
        """Bytes of one expert slice for a stacked (E, ...) key, or 0."""
        try:
            entry = self._reader_for_key(key).header[key]
            shape = entry["shape"]
            num_experts = shape[0] if shape else 1
            start, end = entry["data_offsets"]
            return (end - start) // num_experts
        except Exception:
            return 0

    def load_expert(self, key: str, expert_id: int) -> mx.array:
        np_view = self.load_expert_slice(key, expert_id)
        return _np_to_mx(key, np_view, self._reader_for_key(key, expert_id).header[key]["dtype"])

    def load_expert_slice(self, key: str, expert_id: int) -> np.ndarray:
        """Return a fresh np.ndarray copy of one expert's slice (mmap-backed).

        Use this when the caller (e.g. the async prefetcher) does not want
        MLX ops allocated off the inference thread — they stay on the caller
        thread as plain numpy buffers and the inference thread promotes them
        to mx.array at use time, avoiding cross-thread stream errors.
        """
        reader = self._reader_for_key(key, expert_id)
        return reader.expert_slice(key, expert_id)

    def load_expert_run(self, key: str, first_id: int, count: int) -> list[np.ndarray]:
        """Read *count* consecutive experts starting at *first_id* in one pread.

        Adjacent expert ids occupy contiguous byte ranges in a row-major
        stacked bank, so a run of ids collapses into a single sequential
        read instead of *count* separate ones. Returns per-expert numpy
        views into the run buffer (views are safe: promotion copies).
        """
        reader = self._reader_for_key(key, first_id)
        return reader.expert_run(key, first_id, count)

    def read_expert_into(
        self,
        components: list[tuple[str, list[int]]],
        outs: list[np.ndarray],
        *,
        merge_gap: int = 0,
        max_run_bytes: int = 0,
    ) -> bool:
        """Coalesced zero-copy read of several (key, expert-ids) components.

        For each (key, eids) in components this resolves the backing reader
        per expert (tier-aware) and issues batched preadv calls covering every
        requested expert id (row-major banks make consecutive ids one
        contiguous range), writing the raw bytes into outs[i] — a caller-owned
        writable uint8 buffer of shape (len(eids), per_expert_bytes). This
        replaces the per-expert load_expert_slice loop used by the
        demand-assembly miss path: one reader resolution per component and one
        syscall per run instead of one resolution + one read per expert.

        Tier contract (HOBBIT split): a component's expert ids must all
        resolve to the SAME reader (same tier packing) — hot and cold copies
        of one key can have different per-expert byte sizes, so a mixed
        component cannot share an output buffer layout. The caller splits
        demand by tier before calling; if a mixed component slips through
        this returns False and the caller falls back to per-expert loads.

        The per-run preadv calls of a single component are issued
        concurrently on _run_io_pool (see _RUN_IO_QD), in batches, so a
        fragmented demand set keeps the device's queue depth above 1.

        Returns True on success, False if any component could not be served
        (caller must fall back to load_expert_slice). Bytes are written in
        expert-id order so outs[i][j] is expert eids[j].
        """
        if len(components) != len(outs):
            return False
        for (key, eids), out in zip(components, outs):
            if out.dtype != np.uint8 or out.ndim != 2:
                return False
            if not self._read_component_into(
                key, eids, out, merge_gap, max_run_bytes
            ):
                return False
        return True

    def _resolve_component_reader(self, key, eids, out, n):
        """Tier-aware single-reader resolution + output validation.

        Resolve the reader per expert (tier-aware) and require
        ONE reader for the component — mixed tiers cannot share the
        uniform output layout. Returns ``(reader, rp)`` to proceed, or
        None — the caller fails the component.
        """
        try:
            readers = [self._reader_for_key(key, int(e)) for e in eids]
        except Exception:
            return None
        reader = readers[0]
        if any(r is not reader for r in readers[1:]):
            return None
        try:
            rp = reader._rp_for(key)
        except Exception:
            return None
        if out.shape[0] != n or out.shape[1] != rp.expert_bytes:
            return None
        if any(eid < 0 or eid >= rp.num_experts for eid in eids):
            return None
        return reader, rp

    def _plan_component_runs(
        self, eids, n, rp, merge_gap, max_run_bytes
    ):
        """Sorted-order run plan for one component's demanded ids.

        Returns ``runs``: a list of ``(abs_off, first_id, count,
        order-slice)`` — one contiguous (or gap-bridged) preadv per entry.
        """
        # Read contiguous runs separately. This keeps sparse demand from
        # over-reading the gap between the first and last expert. With
        # merge_gap > 0 a run may BRIDGE holes of up to
        # merge_gap missing ids — the hole rows are read with the run
        # but the scatter below writes ONLY the demanded ids, so gap
        # bytes can never enter the output (or the LRU). Same shared
        # segmentation as the demand planner and the advisor.
        order = sorted(range(n), key=lambda j: eids[j])
        sorted_ids = [eids[j] for j in order]
        # An uncapped contiguous run is ONE preadv on
        # ONE worker — dense prefill demand (~500 experts, ~450 MB) would
        # serialize at single-stream speed. max_run_bytes bounds each
        # run so the sliding window below keeps _RUN_IO_QD reads in
        # flight. 0 keeps the unbounded segmentation.
        run_cap = (
            max(1, int(max_run_bytes) // rp.expert_bytes)
            if max_run_bytes > 0
            else None
        )
        segments = segment_runs(
            sorted_ids, merge_gap=merge_gap, max_run=run_cap
        )
        # Map each run back to its order slice: every demanded id inside
        # [first, first+count) belongs to this run in id order. The
        # scatter base for a demanded id is (eid - first) rows into the
        # buffer — bridge rows shift the demanded ids away from the
        # leading positions, so the base can never be the index of the
        # order slice.
        runs: list[tuple[int, int, int, list[int]]] = []
        # (abs_off, first_id, count, order slice)
        for first, count in segments:
            lo = bisect.bisect_left(sorted_ids, first)
            hi = bisect.bisect_left(sorted_ids, first + count)
            off = rp.tensor_abs_off + first * rp.expert_bytes
            runs.append((off, first, count, order[lo:hi]))
        return runs

    @staticmethod
    def _scatter_run_rows(eids, out, rp, first, js, buf) -> None:
        # The row base for a demanded id is (eid - first) — never
        # the slice index, which is what would corrupt every id
        # behind a bridge. Rows are disjoint per descriptor, so the
        # byte content of out never depends on completion order.
        rows = buf.reshape(-1, rp.expert_bytes)
        out[np.asarray(js)] = rows[[eids[j] - first for j in js]]

    def _read_component_into(
        self, key, eids, out, merge_gap, max_run_bytes
    ) -> bool:
        """One (key, eids) component of ``read_expert_into``.

        Reader resolution, run planning, and the windowed reads.
        False → the caller fails the whole call (fallback to
        per-expert loads).
        """
        n = len(eids)
        if n == 0:
            return True
        resolved = self._resolve_component_reader(key, eids, out, n)
        if resolved is None:
            return False
        reader, rp = resolved
        runs = self._plan_component_runs(
            eids, n, rp, merge_gap, max_run_bytes
        )

        # Issue each batch's preadvs concurrently. One at a time they are
        # a chain of blocking syscalls, which pins the device's I/O queue
        # depth at 1: the device idles for a full round trip between runs.
        # Prefill largely hides this — its runs are long and contiguous, so
        # kernel readahead covers it. Decode has no such luck:
        # its demand is a handful of scattered experts per projection and
        # queue depth is all it has.
        #
        # Batched rather than all-at-once so peak transient memory stays
        # at _RUN_IO_QD run buffers instead of one per run — a fully
        # fragmented component is one run per expert, which would
        # otherwise double the component's footprint.
        if len(runs) == 1:
            off, first, count, js = runs[0]
            # The single-run path reads inline on the CALLER thread.
            try:
                if count == len(js) and out.flags["C_CONTIGUOUS"]:
                    reader._read_iovecs(
                        off, [memoryview(out[j]) for j in js]
                    )
                else:
                    buf = np.empty(count * rp.expert_bytes, dtype=np.uint8)
                    reader._read_into(off, buf)
                    self._scatter_run_rows(eids, out, rp, first, js, buf)
            except Exception:
                return False
        else:
            # Sliding window: keep up to _RUN_IO_QD
            # reads in flight continuously — draining the queue at every
            # batch boundary would let the device idle even though demand
            # was waiting. The oldest submission is reaped once the
            # window fills; out bytes stay deterministic since each
            # descriptor scatters its own disjoint rows.
            io_exec = _run_io_pool()
            if not self._read_runs_ordered_window(
                runs, reader, rp, eids, out, io_exec
            ):
                return False
        return True

    def _read_runs_ordered_window(
        self, runs, reader, rp, eids, out, io_exec,
    ) -> bool:
        """Submission-order sliding window (default): the OLDEST submitted
        read is reaped once the window fills. False → caller fails."""
        # Window entries: (first, js, buf_or_iovecs, future). buf is a
        # temp run buffer to scatter from; iovecs are out-row memoryviews
        # the kernel fills in place (set when the run has no bridged
        # rows, so row order IS the js order).
        window: list = []
        ok = True
        direct = out.flags["C_CONTIGUOUS"]

        for idx, (off, first, count, js) in enumerate(runs):
            if count == len(js) and direct:
                iovecs = [memoryview(out[j]) for j in js]
                window.append(
                    (first, js, None, io_exec.submit(reader._read_iovecs, off, iovecs))
                )
            else:
                buf = np.empty(count * rp.expert_bytes, dtype=np.uint8)
                window.append(
                    (first, js, buf, io_exec.submit(reader._read_into, off, buf))
                )
            if len(window) >= _RUN_IO_QD or idx == len(runs) - 1:
                wfirst, wjs, wbuf, wfut = window.pop(0)
                if not _await_io_future(wfut):
                    ok = False
                if ok and wbuf is not None:
                    self._scatter_run_rows(eids, out, rp, wfirst, wjs, wbuf)
            if not ok:
                for _f, _j, _b, fut in window:
                    # Drain instead of dropping — a read
                    # exception in an abandoned prefetch future
                    # would vanish with it.
                    _await_io_future(fut)
                return False
        for wfirst, wjs, wbuf, wfut in window:
            if not _await_io_future(wfut):
                ok = False
            if ok and wbuf is not None:
                self._scatter_run_rows(eids, out, rp, wfirst, wjs, wbuf)
        return ok


    def advise_expert_run(
        self, key: str, first_id: int, count: int
    ) -> tuple[bool, int, int]:
        """Kernel readahead of experts [first_id, first_id+count) of one bank.

        Row-major stacked banks make a run of ids one contiguous byte range,
        so the whole run collapses into a single F_RDADVISE — the zero-copy
        readahead counterpart of load_expert_run. Under the HOBBIT split
        the run may straddle the hot/cold boundary; a run reads ONE
        reader (the one its first id resolves), so advise breaks at tier
        boundaries exactly like the demand path's _group_runs.

        Returns (ok, bytes_advised, tier_segments): bytes_advised is the
        total file range covered by the accepted advisories,
        tier_segments counts the
        reader groups the run needed. ok is False when the platform lacks
        F_RDADVISE or nothing resolved.
        """
        try:
            if count <= 0:
                return (False, 0, 0)
            ids = [first_id + i for i in range(count)]
            readers = [self._reader_for_key(key, eid) for eid in ids]
            ok = False
            total_bytes = 0
            segments = 0
            # group consecutive ids sharing the same reader (tier boundary)
            i = 0
            while i < len(ids):
                j = i
                reader = readers[i]
                while j + 1 < len(ids) and readers[j + 1] is reader:
                    j += 1
                start, _ = reader.expert_byte_range(key, ids[i])
                _, end = reader.expert_byte_range(key, ids[j])
                if end > start:
                    ok = reader.advise_range(start, end - start) or ok
                    total_bytes += end - start
                    segments += 1
                i = j + 1
            return (ok, total_bytes, segments)
        except Exception:
            return (False, 0, 0)

    def pin_expert(self, key: str, expert_id: int) -> int:
        """mlock one expert's file range across the resolved shard.

        Returns locked bytes (0 on failure). Duplicate pins of the same
        (key, expert) are tracked and skipped. Safe to call from warm-pool
        workers: the pin set, the per-file page set and ``pinned_bytes``
        are updated under ``_pin_lock``.
        """
        reader = self._reader_for_key(key, expert_id)
        # The pin key carries the expert row so unpin_all can pass it
        # straight to reader.unpin_expert (a byte-range operation).
        pkey = (str(reader.path), key, expert_id)
        with self._pin_lock:
            if pkey in self._pinned:
                return 0
            locked = reader.pin_expert(key, expert_id)
            if locked > 0:
                self._pinned.add(pkey)
                # Count only NEWLY locked pages — adjacent experts
                # share the boundary page and must not double-charge the
                # budget.
                try:
                    off, end = reader.expert_byte_range(key, expert_id)
                    sp = off // _PAGE_SIZE
                    ep = (end + _PAGE_SIZE - 1) // _PAGE_SIZE
                    pages = self._pinned_pages.setdefault(str(reader.path), set())
                    new_pages = set(range(sp, ep)) - pages
                    pages |= new_pages
                    self.pinned_bytes += len(new_pages) * _PAGE_SIZE
                except Exception:
                    self.pinned_bytes += locked
        return locked

    def unpin_all(self) -> int:
        """munlock every pinned range and reset the pin bookkeeping.

        Returns the number of pins released. Reclaims wired memory without
        waiting for the implicit unlock inside ``munmap`` at reader close,
        and keeps ``pinned_bytes`` from reporting wired bytes that no
        longer exist after an unload/reload.
        """
        with self._pin_lock:
            pins = list(self._pinned)
            if not pins:
                return 0
            released = 0
            for pkey in pins:
                path, key, expert_id = pkey
                reader = self._readers.get(path) or self._cold_readers.get(path)
                if reader is None:
                    continue
                try:
                    if reader.unpin_expert(key, expert_id) > 0:
                        released += 1
                except Exception:
                    logger.debug("unpin failed for %s", pkey, exc_info=True)
            self._pinned.clear()
            self._pinned_pages.clear()
            self.pinned_bytes = 0
            return released

    @property
    def pinned_count(self) -> int:
        with self._pin_lock:
            return len(self._pinned)

    def close(self) -> None:
        # Release wired pages BEFORE the mappings go away. munmap
        # would unlock them implicitly, but the pin bookkeeping (_pinned,
        # _pinned_pages, pinned_bytes) must be cleared too or it reports
        # wired memory that no longer exists on the next load.
        try:
            self.unpin_all()
        except Exception:
            logger.debug("unpin_all failed during close", exc_info=True)
        # Stop speculation before the readers die — a live
        # advisor would otherwise reference closed files past its owning
        # engine's lifetime.
        spec = getattr(self, "spec_state", None)
        if spec is not None:
            try:
                spec.close()
            except Exception:
                pass
            try:
                self.spec_state = None  # type: ignore[attr-defined]
            except Exception:
                pass
        # Stop the detached admission worker with the engine — it is a
        # daemon thread, but an explicit close keeps repeated conversions
        # from leaking idle workers.
        cache = getattr(self, "_streaming_cache", None)
        if cache is not None:
            try:
                close = getattr(cache, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
        # Latch + swap under _readers_lock: a lazy resolve that already
        # passed the closed check finishes with a live dict; one that
        # arrives after raises/returns before creating a reader, so no
        # fd+mmap can be published once the dicts are emptied here.
        with self._readers_lock:
            self._closed = True
            readers_to_close = list(self._readers.values()) + list(
                self._cold_readers.values()
            )
            self._readers.clear()
            self._cold_readers.clear()
            self._key_to_reader.clear()
            # Key->reader memo for the cold tier too: without this a
            # post-close lookup hands back a CLOSED reader instead of
            # rebuilding.
            self._cold_key_to_reader.clear()
        for r in readers_to_close:
            try:
                r.close()
            except Exception:
                pass
