"""qwen4_exp MoE expert weight streaming -- load-time interception.

Streams the ~46.87 GB of routed-expert (``switch_mlp``) weights off the wired /
phys memory budget by binding them as mmap-backed mx.arrays over a page-aligned
artifact (built by ``moe_repack``), instead of loading them resident.

Interception seam (critical): we replace the ``switch_mlp.*`` entries in the
weights list **before** ``model.load_weights`` -> ``mx.eval(model.parameters())``.
Because safetensors weights are loaded lazily, swapping the value before that
eval means the original resident arrays are *never materialized* -- that is the
entire point (a post-load module walk would first spike ~46.87 GB resident, then
free it, defeating streaming). The wrapped arrays are already "available"
(external buffer), so the eval over them is a no-op copy.

Key matching is canonical -- (is_mtp, layer_idx, projection, suffix) -- so it is
robust to the MTP prefix differing between the checkpoint key
(``language_model.mtp.layers.N...``) and the bound module path (``mtp.layers.N...``).

Quantization params (bits/group_size/mode) are NOT touched here: they live on the
``QuantizedSwitchLinear`` module and drive gather_qmm; we only swap the three
arrays (weight/scales/biases), whose shapes/dtypes match the module slots.
"""

from __future__ import annotations

import json
import logging
import re
import struct
from pathlib import Path
from typing import Optional

from omlx.utils import proc_memory

from . import fast

logger = logging.getLogger(__name__)

# Registry name for the external-wired provider (see omlx/utils/proc_memory.py).
# Name-keyed + replace-on-register => registering on every load is idempotent.
_PROVIDER_NAME = "qwen4_moe_stream"

# ...layers.{i}.mlp.switch_mlp.{proj}.{suffix}  (proj in gate/up/down_proj,
# suffix in weight/scales/biases). Matches both the regular and MTP forms.
_CANON_RE = re.compile(
    r"layers\.(\d+)\.mlp\.switch_mlp\."
    r"(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$"
)


def canonical_key(key: str) -> Optional[tuple[bool, int, str, str]]:
    """Parse a state-dict / manifest key into (is_mtp, layer_idx, proj, suffix),
    or None if it is not a switch_mlp expert tensor."""
    m = _CANON_RE.search(key)
    if m is None:
        return None
    is_mtp = ".mtp." in key or key.startswith("mtp.")
    return (is_mtp, int(m.group(1)), m.group(2), m.group(3))


class StreamingArtifact:
    """Owns the mmap of a streaming artifact and hands out wrapped mx.arrays,
    indexed by canonical key so lookups survive prefix differences."""

    def __init__(self, path: str):
        self.path = str(path)
        with open(self.path, "rb") as f:
            mlen = struct.unpack("<Q", f.read(8))[0]
            manifest = json.loads(f.read(mlen))
        self.page_size = int(manifest.get("page_size", 16384))
        # canonical tuple -> manifest tensor entry
        self._by_canon: dict[tuple, dict] = {}
        for k, entry in manifest["tensors"].items():
            canon = canonical_key(k)
            if canon is None:
                continue
            self._by_canon[canon] = entry
        self._id: Optional[int] = None

    def __len__(self) -> int:
        return len(self._by_canon)

    def open(self) -> None:
        if self._id is None:
            self._id = fast.mmap_artifact(self.path)
            # Register the enforcer external-wired provider exactly once (by
            # name; replace-on-register makes repeated opens harmless). The
            # provider reports the GLOBAL mmap'd total, so it must never be
            # registered per-open under an append-only API -- proc_memory's
            # name-keyed registry guarantees that. Never unregistered: it
            # returns 0 once nothing is mapped.
            proc_memory.register_external_wired_provider(
                _PROVIDER_NAME, fast.mapped_bytes
            )

    def close(self) -> None:
        if self._id is not None:
            fast.close_artifact(self._id)
            self._id = None

    def has(self, canon: tuple) -> bool:
        return canon in self._by_canon

    def wrap(self, canon: tuple):
        """Return the mmap-backed mx.array for a canonical key."""
        if self._id is None:
            self.open()
        e = self._by_canon[canon]
        return fast.wrap_tensor(self._id, e["offset"], e["length"],
                                e["shape"], e["dtype"])

    def provider(self) -> Callable[[], int]:
        """Enforcer external-wired provider: bytes currently mmap'd (global sum
        across all live artifacts). Cheap, non-blocking, never touches MLX."""
        return fast.mapped_bytes


def stream_weight_items(items, artifact: StreamingArtifact, *, expect_all: bool = False):
    """Transform a ``[(key, array), ...]`` weights list, replacing every
    switch_mlp expert tensor the artifact provides with its mmap-backed array.

    MUST run POST-sanitize: the vendored ``sanitize`` renames keys (e.g.
    ``language_model.mtp.*`` -> ``mtp.*``) and can restack expert layouts. Run
    pre-sanitize and canonical matching silently finds nothing -- every tensor
    passes through and the model loads fully resident with no error (the
    pass-through design hides it). ``expect_all`` closes that trap: when the
    caller knows the full artifact must be consumed (the live wire-in), it hard
    -fails on any shortfall so silent-resident is impossible.

    Returns ``(new_items, n_swapped, n_missing)``. Non-expert entries and any
    expert the artifact lacks or that shape/dtype-mismatches pass through
    untouched (partial artifact / stray tensor degrades to resident, not crash).
    """
    new_items = []
    n_swapped = 0
    n_missing = 0
    for key, value in items:
        canon = canonical_key(key)
        if canon is None:
            new_items.append((key, value))
            continue
        if not artifact.has(canon):
            n_missing += 1
            new_items.append((key, value))
            continue
        wrapped = artifact.wrap(canon)
        # Shape/dtype must match the slot the checkpoint value would fill.
        if list(wrapped.shape) != list(value.shape) or wrapped.dtype != value.dtype:
            logger.warning(
                "qwen4_moe_stream: shape/dtype mismatch for %s "
                "(wrapped %s/%s vs checkpoint %s/%s); keeping resident",
                key, wrapped.shape, wrapped.dtype, value.shape, value.dtype,
            )
            new_items.append((key, value))
            continue
        new_items.append((key, wrapped))
        n_swapped += 1

    # Mandatory telemetry: silent-resident must be impossible to miss.
    expected = len(artifact)
    if n_swapped == expected:
        logger.info(
            "qwen4_moe_stream: streamed %d/%d expert tensors (all resident "
            "expert weight materialization avoided)", n_swapped, expected,
        )
    else:
        logger.warning(
            "qwen4_moe_stream: streamed only %d/%d expert tensors "
            "(%d missing from artifact) -- the rest load RESIDENT. If this is "
            "the full wire-in, the swap likely ran in the wrong place "
            "(must be POST-sanitize).", n_swapped, expected, n_missing,
        )
        if expect_all:
            raise RuntimeError(
                f"qwen4_moe_stream: expected to stream all {expected} expert "
                f"tensors but only swapped {n_swapped} (missing {n_missing}). "
                f"Refusing silent-resident load; check swap runs post-sanitize."
            )
    return new_items, n_swapped, n_missing


# Provider registration is now handled by StreamingArtifact.open() via the
# name-keyed proc_memory registry (idempotent by construction). The former
# id(enforcer)-guarded ensure_provider_registered() was removed: object-id reuse
# after enforcer teardown could silently skip registration for a new enforcer at
# a recycled address (Fable review 2, issue 5).


def default_artifact_path(model_dir: str) -> Optional[str]:
    """The PLE-style artifact sits next to the checkpoint."""
    p = Path(model_dir) / "moe_experts_streaming.artifact"
    return str(p) if p.exists() else None


def streaming_offload_bytes(model_dir: str) -> int:
    """Bytes that MoE expert streaming would take off the resident weight budget
    for this model, or 0 if streaming would not engage. Mirrors the wire-in
    gating (``_stream_qwen4_exp_experts_on_load``) EXACTLY so admission's
    resident estimate matches whether streaming actually happens: native ext
    available, ``OMLX_QWEN4_MOE_STREAM`` not disabled, artifact present. Returns
    the artifact file size (== worst-case ``mapped_bytes`` once loaded). Cheap:
    a single ``stat``, no mmap.
    """
    import os

    if os.environ.get("OMLX_QWEN4_MOE_STREAM", "1").strip().lower() in (
        "0", "false", "no", "off",
    ):
        return 0
    if not fast.is_native_available():
        return 0
    path = default_artifact_path(str(model_dir))
    if path is None:
        return 0
    try:
        return int(os.path.getsize(path))
    except OSError:
        return 0


def run_load_canary(artifact: StreamingArtifact, *, window: int = 4096) -> None:
    """Load-time bit-exactness canary: force MLX to read a window of one wrapped
    expert tensor through its external Metal buffer and compare, byte-for-byte,
    against an independent CPU (numpy) read of the same artifact bytes. Raises on
    any mismatch -- a silent wrong-bytes read (the page-alignment trap that
    started this whole effort) becomes a hard, immediate load failure.

    Picks the first available tensor deterministically; cheap (one small slice).
    """
    import mlx.core as mx
    import numpy as np

    if len(artifact) == 0:
        raise RuntimeError("qwen4_moe_stream canary: artifact has no tensors")
    canon = sorted(artifact._by_canon.keys())[0]
    entry = artifact._by_canon[canon]
    # Reinterpret through the integer storage dtype so numpy compares raw bytes
    # exactly (BF16 has no numpy dtype; view it as uint16).
    store_mx, store_np = (
        (mx.uint16, np.uint16) if entry["dtype"] == "BF16" else (mx.uint32, np.uint32)
    )
    itemsize = np.dtype(store_np).itemsize
    n = min(window, entry["length"] // itemsize)

    # GPU/external path: force MLX to read a flat window through the ext buffer.
    wrapped = artifact.wrap(canon)
    raw = wrapped.reshape(-1)[:n].view(store_mx)
    gpu_bytes = np.array(raw).tobytes()

    # Independent CPU reference: raw bytes straight from the artifact file.
    with open(artifact.path, "rb") as f:
        f.seek(entry["offset"])
        ref = f.read(n * itemsize)

    if gpu_bytes != ref:
        raise RuntimeError(
            f"qwen4_moe_stream canary FAILED for {canon}: external buffer read "
            f"does not match the artifact bytes (page-alignment / wrap bug). "
            f"Refusing to serve wrong weights."
        )
    logger.info(
        "qwen4_moe_stream: load-time bit-exactness canary passed (%s, %d %s vals)",
        canon, n, entry["dtype"],
    )
