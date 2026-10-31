# SPDX-License-Identifier: Apache-2.0
"""Blockwise Hadamard rotation for Qwen4-Exp checkpoints.

The v4 exporter stores the residual-path weights in a rotated basis and the
runtime keeps the residual stream rotated, which is free for every linear.
Three places do **not** commute with that rotation and therefore need this
patch:

* the PLE short conv is a *depthwise* conv (``groups = hc*hidden``), so each
  channel has its own kernel and ``R D R^T`` is not diagonal;
* the per-channel ``γ`` in ``norm_conv`` acts on the rotated input;
* the hyper-connection block reads the per-channel-normalised stream directly,
  with no linear to absorb ``hc_norm``'s ``γ`` into.

Both PLE and hc therefore run in the unrotated basis: unrotate at entry, run,
rotate the outputs back.

Exporter contract
-----------------
``R`` for one block of size ``B`` is ``diag(signs) @ H_B / sqrt(B)`` with
``signs`` drawn from ``numpy.random.default_rng(seed)`` in block order, exactly
as ``calib/scripts/v4_quant.py::hadamard_blocks`` builds it. The residual
stream is ``x @ R``; a linear that reads it stores ``W @ R`` and one that
writes it stores ``R^T @ W``. Tensors inside PLE and the hc block are stored
**exact** (unrotated), including ``key_proj``, ``value_proj`` and the ngram
table, because the patch unrotates their input and rotates their result.

Usage
-----
Call ``configure_qwen4_hadamard(model_dir)`` and ``apply_qwen4_hadamard()``
before the model is built. ``apply`` wraps the hc forward and
``Qwen4ExpModel.__init__``, which tags the built PLE/hc modules with their
rotations. A checkpoint declares the transform in ``config.json`` as::

    "hadamard": {"block": 512, "signs_seed": 1234}
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

_CONFIG_KEY = "hadamard"
_ACTIVE: "HadamardSpec | None" = None
_CACHE: dict[tuple[int, int, int, int | None], tuple[mx.array, ...]] = {}


@dataclass(frozen=True)
class HadamardSpec:
    block: int = 512
    signs_seed: int = 1234


def _hadamard(n: int) -> np.ndarray:
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    if h.shape[0] != n:
        raise ValueError(f"block {n} is not a power of two")
    return h / np.sqrt(n)


def build_rotations(dim: int, spec: HadamardSpec, period: int | None = None) -> tuple[mx.array, ...]:
    """Per-block ``R`` matrices, in block order. Cached."""
    key = (dim, spec.block, spec.signs_seed, period)
    if key in _CACHE:
        return _CACHE[key]
    if dim % spec.block:
        raise ValueError(f"dim {dim} is not divisible by block {spec.block}")
    if period is not None and (
        period <= 0 or dim % period or period % spec.block
    ):
        raise ValueError(
            f"period {period} must be a positive multiple of block {spec.block} "
            f"that divides dim {dim}"
        )
    n_blocks = dim // spec.block
    period_blocks = n_blocks if period is None else period // spec.block
    rng = np.random.default_rng(spec.signs_seed)
    base = _hadamard(spec.block)
    unique = []
    for _ in range(period_blocks):
        signs = rng.choice([-1.0, 1.0], spec.block)
        unique.append(mx.array((signs[:, None] * base).astype(np.float32)))
    _CACHE[key] = tuple(unique[i % period_blocks] for i in range(n_blocks))
    return _CACHE[key]


_STACKS: dict[tuple, mx.array] = {}


def _batched_stack(rots: tuple[mx.array, ...], inverse: bool, dim: int, block: int) -> mx.array:
    """All block matrices stacked into one [n_blocks, B, B] tensor, rebuilt only when specs change.

    The per-block loop it replaces formed twenty matmul nodes plus a stack for every 10240-wide
    conversion (290 conversions per decode token), and the measured conversion cost is ~32 ms/token
    against ~14 ms of actual matmul work - the rest was CPU graph construction.
    """
    key = (dim, block, inverse, id(rots))
    cached = _STACKS.get(key)
    if cached is None:
        cached = mx.array(np.stack([np.array(r.T if inverse else r) for r in rots]).astype(np.float32))
        _STACKS[key] = cached
    return cached


def apply_rotation(x: mx.array, rots: tuple[mx.array, ...], inverse: bool = False) -> mx.array:
    """Rotate the last axis blockwise. ``inverse=True`` applies ``R^T``.

    The rotations are fp32; the result is cast back to ``x.dtype`` so a bf16
    residual stream is not promoted to fp32 for the rest of the model.
    """
    block = rots[0].shape[0]
    lead = x.shape[:-1]
    dim = x.shape[-1]
    v = x.reshape(*lead, dim // block, block)
    out = mx.einsum("...nb,nbj->...nj", v, _batched_stack(rots, inverse, dim, block))
    out = out.reshape(*lead, dim)
    return out.astype(x.dtype) if out.dtype != x.dtype else out


def configure_qwen4_hadamard(model_dir: str | os.PathLike | None) -> "HadamardSpec | None":
    """Read ``hadamard`` from a checkpoint's config.json and activate it."""
    global _ACTIVE
    _ACTIVE = None
    if model_dir is None:
        return None
    path = os.path.join(str(model_dir), "config.json")
    try:
        with open(path, "r") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return None
    spec = cfg.get(_CONFIG_KEY)
    if not isinstance(spec, dict):
        # Some exporters nest the transform next to the text config.
        text = cfg.get("text_config")
        if isinstance(text, dict):
            spec = text.get(_CONFIG_KEY)
    if not isinstance(spec, dict):
        return None
    block = int(spec.get("block", 512))
    seed = int(spec.get("signs_seed", 1234))
    if block <= 0 or block & (block - 1):
        raise ValueError(f"Hadamard block must be a positive power of two, got {block}")
    _ACTIVE = HadamardSpec(block=block, signs_seed=seed)
    logger.info("Qwen4-Exp Hadamard rotation active: block=%d signs_seed=%d", block, seed)
    return _ACTIVE


def active_spec() -> "HadamardSpec | None":
    return _ACTIVE


def rotations_for(dim: int, period: int | None = None) -> "tuple[mx.array, ...] | None":
    """Rotations for an activation of width ``dim``, or None when inactive."""
    if _ACTIVE is None:
        return None
    return build_rotations(dim, _ACTIVE, period)


def apply_qwen4_hadamard() -> int:
    """Install the runtime half of the Hadamard patch.

    The hc block reads the per-channel-normalised stream directly, so a channel
    rotation cannot be folded into any of its weights: unrotate at entry, run,
    rotate the outputs back. The GDN, attention and MoE branches stay folded,
    i.e. they expect rotated activations. ``Qwen4ExpModel.__init__`` tags every
    built hc/PLE module with its rotations, so no post-load hook is needed.
    """
    from mlx_vlm.models.qwen4_exp.language import (
        Qwen4ExpGatedResidual,
        Qwen4ExpModel,
    )

    wrapped = 0
    original = Qwen4ExpGatedResidual.__call__
    if not getattr(original, "_omlx_hadamard_wrapped", False):

        def converted(self, hyper_input, *args, **kwargs):
            rots = getattr(self, "_qh_rot", None)
            if rots is None:
                # opt-in per module: only a model tagged by ``tag_hadamard_modules`` converts
                return original(self, hyper_input, *args, **kwargs)
            rots_mixed = getattr(self, "_qh_rot_mixed", None)
            if rots_mixed is None:
                spec = active_spec()
                if spec is None:
                    return original(self, hyper_input, *args, **kwargs)
                rots_mixed = build_rotations(
                    self.hidden_size, spec, self.hidden_size
                )
            write = kwargs.get("write")
            if write is not None:
                # the pending (branch, gate) residual arrives in the rotated basis
                branch, gate = write
                kwargs["write"] = (
                    apply_rotation(branch, rots_mixed, inverse=True),
                    gate,
                )
            if getattr(self, "_qh_busy", False):
                # the deferred-write path recurses through self(...); the inner call already has a
                # plain-basis input, so it must not convert again
                return original(self, hyper_input, *args, **kwargs)
            self._qh_busy = True
            try:
                out = original(
                    self,
                    apply_rotation(hyper_input, rots, inverse=True),
                    *args,
                    **kwargs,
                )
            finally:
                self._qh_busy = False
            if isinstance(out, tuple) and len(out) == 2:
                # (mixed, written) from a module without inject weights: the
                # first is hidden-wide, the passthrough is hc-wide.
                mixed, hyper = out
                return (
                    apply_rotation(mixed, rots_mixed),
                    apply_rotation(hyper, rots),
                )
            if isinstance(out, tuple):
                mixed, hyper, *rest = out
                return (
                    apply_rotation(mixed, rots_mixed),
                    apply_rotation(hyper, rots),
                    *rest,
                )
            return apply_rotation(out, rots_mixed)

        if hasattr(Qwen4ExpGatedResidual, "_qh_impl"):
            # the vendored class already converts inside its own __call__ (v4); installing a
            # wrapper on top would convert twice and re-add the dispatch this inlining removed
            pass
        else:
            converted._omlx_hadamard_wrapped = True
            Qwen4ExpGatedResidual.__call__ = converted
            wrapped += 1

    original_init = Qwen4ExpModel.__init__
    if not getattr(original_init, "_omlx_hadamard_model_wrapped", False):

        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            tag_hadamard_modules(self)

        init._omlx_hadamard_model_wrapped = True
        Qwen4ExpModel.__init__ = init
        wrapped += 1
    return wrapped


def _is_descendant_of_mtp(module, head, modules) -> bool:
    """True when ``module`` is one of ``head``'s own submodules (identity walk)."""
    stack = [head]
    seen = set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if node is module:
            return True
        for _name, child in getattr(node, "children", lambda: {})().items() if hasattr(node, "children") else ():
            stack.append(child)
        for attr in ("layers", "hyper_connection_mixer"):
            value = getattr(node, attr, None)
            if isinstance(value, (list, tuple)):
                stack.extend(v for v in value if v is not None)
            elif value is not None:
                stack.append(value)
    return False


def tag_hadamard_modules(model) -> int:
    """Opt every PLE and hyper-connection module of one loaded model into the conversion."""
    spec = active_spec()
    if spec is None:
        return 0
    tagged = 0
    modules = list(getattr(model, "modules", lambda: [])())
    # The MTP draft head runs in the plain basis: tag it once for the input conversion and leave
    # its internal hyper-connections untagged so the stream is not converted twice.
    mtp_ids = set()
    mtp_heads = []
    for module in modules:
        if type(module).__name__.startswith("Qwen4ExpMTP"):
            mtp_ids.add(id(module))
            mtp_heads.append(module)
    for module in modules:
        hc = getattr(module, "hc_count", None)
        hidden = getattr(module, "hidden_size", None)
        if hc is None or hidden is None:
            continue
        if id(module) in mtp_ids or type(module).__name__.startswith("Qwen4ExpMTP"):
            continue
        if any(_is_descendant_of_mtp(module, head, modules) for head in mtp_heads):
            continue
        module._qh_rot = build_rotations(hc * hidden, spec, hidden)
        # The mixed output is hidden-wide, so it uses the per-stream rotation.
        module._qh_rot_mixed = build_rotations(hidden, spec, hidden)
        tagged += 1
    args = getattr(model, "args", None)
    if args is None or getattr(args, "hidden_size", None) is None:
        # the VLM wrapper does not carry ``args``; the language model does
        language = getattr(model, "language_model", None)
        args = getattr(language, "args", None) or args
    hidden = getattr(args, "hidden_size", None)
    if isinstance(hidden, int) and hidden % spec.block == 0:
        # DFlash2 captures the hc mean and the MTP head fuses the hc stream; both need the
        # hc-wide inverse rotation.
        hc_count = getattr(args, "hc_count", 1) or 1
        rots_hc = build_rotations(hidden * hc_count, spec, hidden)
        model._omlx_hadamard_capture_rots = build_rotations(hidden, spec, hidden)
        for head in mtp_heads:
            head._omlx_hadamard_capture_rots = rots_hc          # trunk hc stream
            head._omlx_hadamard_token_rots = model._omlx_hadamard_capture_rots  # embeddings
            head._omlx_hadamard_out_rots = model._omlx_hadamard_capture_rots    # folded lm_head
    return tagged


def _self_check() -> None:
    spec = HadamardSpec(block=512, signs_seed=1234)
    dims = 10240
    rots = build_rotations(dims, spec)
    x = (np.random.default_rng(7).normal(0, 0.1, (1, 3, dims))).astype(np.float32)
    xm = mx.array(x)
    back = apply_rotation(apply_rotation(xm, rots), rots, inverse=True)
    err = float(mx.max(mx.abs(back - xm)))
    dense = np.zeros((dims, dims), dtype=np.float32)
    b = spec.block
    for i, r in enumerate(rots):
        dense[i * b:(i + 1) * b, i * b:(i + 1) * b] = np.array(r)
    orth = float(np.abs(dense.T @ dense - np.eye(dims)).max())
    matmul = float(mx.max(mx.abs(mx.array(x) @ mx.array(dense) @ mx.array(dense.T) - xm)))
    scale = float(np.abs(x).max())
    print(f"self-check: round-trip max|err| = {err:.2e} ({err/scale:.1e} relative, fp32 on this GPU)")
    print(f"self-check: blockwise == dense matmul max|err| = {matmul:.2e}")
    print(f"self-check: orthogonality max|R^T R - I| = {orth:.2e}")
    # Metal fp32 matmul is not exact: ~1e-3 relative on a 512-wide butterfly is the
    # measured floor here, and it is ~1 % of the 4/64 quantisation error.
    assert err / scale < 5e-3 and matmul / scale < 5e-3 and orth < 1e-5


if __name__ == "__main__":
    _self_check()
