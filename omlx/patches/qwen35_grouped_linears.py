# SPDX-License-Identifier: Apache-2.0
"""One quantized matmul per quantisation signature for small-row projections.

``_target_verify_linears`` in mlx-vlm's Qwen3.5 family issues one qmm per
projection for B=1 rows (its fused helper needs exactly four identically
quantised linears and T == 1). Concatenating weights along the output axis is
bit-exact per output row, so grouping by (bits, group_size, mode) removes 1-3
dispatches per call site on MTP verify rows (2..8; single-row decode measured
slower grouped). Each group is a cached ``nn.QuantizedLinear`` so the
verify-qmm routing patch sees the same call shape it sees for single linears;
a group whose routing would differ from its members' falls back to separate
calls. Disable with OMLX_QWEN35_GROUPED_LINEARS=0.
"""

from __future__ import annotations

import logging
import os
import sys

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

MIN_ROWS = 2  # T=1 measured -2% serial (grouping costs more than the saved dispatches)
MAX_ROWS = 8  # verify rows; MLX switches qmm kernels above this and rows stop matching per output row
_PATCHED = False
_CACHE_ATTR = "_omlx_grouped_linears"


def enabled() -> bool:
    return os.environ.get("OMLX_QWEN35_GROUPED_LINEARS", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _signature(linear):
    return (int(linear.bits), int(linear.group_size), str(linear.mode))


def _eligible(linears, x) -> bool:
    if not (
        isinstance(x, mx.array)
        and x.ndim == 3
        and x.shape[0] == 1
        and MIN_ROWS <= x.shape[1] <= MAX_ROWS
        and len(linears) >= 2
    ):
        return False
    for lin in linears:
        if (
            not isinstance(lin, nn.QuantizedLinear)
            or "bias" in lin
            or getattr(lin, "mode", "affine") != "affine"
            or lin.biases is None
            or lin.scales.dtype != x.dtype
            or lin.biases.dtype != x.dtype
        ):
            return False
    return True


class _GroupLinear(nn.QuantizedLinear):
    """A QuantizedLinear over concatenated member weights, built without the random init."""

    def __init__(self, weight, scales, biases, *, bits, group_size, mode):
        nn.Module.__init__(self)
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self.weight, self.scales, self.biases = weight, scales, biases
        self.freeze()


def _make_group_linear(members):
    bits, group_size, mode = _signature(members[0])
    weight = mx.concatenate([m.weight for m in members], axis=0)
    scales = mx.concatenate([m.scales for m in members], axis=0)
    biases = mx.concatenate([m.biases for m in members], axis=0)
    mx.eval(weight, scales, biases)
    lin = _GroupLinear(weight, scales, biases, bits=bits, group_size=group_size, mode=mode)
    splits, off = [], 0
    for m in members[:-1]:
        off += int(m.weight.shape[0])
        splits.append(off)
    return lin, splits


def _groups(linears):
    """Cache the group linears on the first member (weights change identity on reload)."""
    first = linears[0]
    key = tuple((id(l.weight), id(l.scales), id(l.biases)) for l in linears)
    cached = getattr(first, _CACHE_ATTR, None)
    if cached is not None and cached[0] == key:
        return cached[1]
    by_sig: dict = {}
    for idx, lin in enumerate(linears):
        by_sig.setdefault(_signature(lin), []).append(idx)
    groups = []
    for sig, idxs in by_sig.items():
        members = [linears[i] for i in idxs]
        if len(members) == 1:
            groups.append((idxs, None, None, members))
            continue
        lin, splits = _make_group_linear(members)
        groups.append((idxs, lin, splits, members))
    # Plain attribute assignment would register the group as a parameter.
    object.__setattr__(first, _CACHE_ATTR, (key, groups))
    return groups


def _routing_matches(group_linear, members, x) -> bool:
    """True unless the verify-qmm patch would route the group differently from a member."""
    try:
        from omlx.patches import qwen35_verify_qmm as vk
    except ImportError:
        return True
    if not (vk._is_armed() and 2 <= x.shape[1] <= 6):
        return True
    args = (x.shape[1], x.shape[-1])
    grouped = vk.vk_eligible(*args, group_linear.scales.shape[0], group_linear.bits,
                             group_linear.group_size, x.dtype)
    return all(
        vk.vk_eligible(*args, m.scales.shape[0], m.bits, m.group_size, x.dtype) == grouped
        for m in members
    )


def grouped_quantized_linears(linears, x):
    """Outputs in the order of ``linears``; None when the call is not eligible."""
    if not _eligible(linears, x):
        return None
    outs = [None] * len(linears)
    for idxs, group_linear, splits, members in _groups(linears):
        if group_linear is None or not _routing_matches(group_linear, members, x):
            for idx, m in zip(idxs, members):
                outs[idx] = m(x)
            continue
        y = group_linear(x)
        for idx, part in zip(idxs, mx.split(y, splits, axis=-1)):
            outs[idx] = part
    return tuple(outs)


def _rebind_everywhere(orig, patched) -> None:
    """Modules that imported the helper by name hold their own binding (vendored Qwen4)."""
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith(("mlx_vlm.", "omlx.")):
            continue
        # __dict__ lookup: getattr on lazy modules would import their backends.
        if vars(module).get("_target_verify_linears") is orig:
            module._target_verify_linears = patched


def apply_qwen35_grouped_linears_patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    if not enabled() or not mx.metal.is_available():
        return False
    try:
        from mlx_vlm.models.qwen3_5 import language as q35
    except ImportError:
        return False
    orig = q35._target_verify_linears
    state = {"failed": False}

    def patched(linears, x, target_verify):
        if not state["failed"]:
            try:
                out = grouped_quantized_linears(linears, x)
            except Exception:  # noqa: BLE001 - optional path
                state["failed"] = True
                logger.warning("grouped linears failed closed", exc_info=True)
                out = None
            if out is not None:
                return out
        return orig(linears, x, target_verify)

    patched.__name__ = "patched"
    _rebind_everywhere(orig, patched)
    _PATCHED = True
    logger.info("Qwen3.5-family grouped small-row linears engaged")
    return True
