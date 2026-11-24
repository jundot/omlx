"""oQh linear modules.

``__call__`` takes the activation in the plain basis, so any caller gets
correct results. It first looks for a rotated copy of that exact array left by
a fused producer or a sibling linear with the same rotation (a small per-thread
memo keyed by array identity), and rotates it itself otherwise.
``matmul_rotated`` takes an activation that is already rotated.
"""

from __future__ import annotations

import contextlib
import threading
import weakref

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.speculative.ops.linear import _target_verify_linear

from ..custom_kernels.oqh import fast
from . import format as fmt

# In row-exact forwards (Lightning MTP decode and verify), up to this many rows
# run the native GEMV, whose result per row is the one-row result. Elsewhere
# the GEMV takes up to _GEMV_ROWS rows; from three rows the stock quantized
# matmul is as fast as for affine weights.
QMV_MAX_ROWS = 16
_GEMV_ROWS = 2
_MEMO_SIZE = 8


class _Exact(threading.local):
    depth = 0


_EXACT = _Exact()


@contextlib.contextmanager
def exact_rows():
    """Give every row of decode-sized inputs the one-row GEMV arithmetic."""
    _EXACT.depth += 1
    try:
        yield
    finally:
        _EXACT.depth -= 1


class _Memo(threading.local):
    def __init__(self):
        self.entries: list[tuple[weakref.ref, mx.array, int]] = []


_MEMO = _Memo()


def remember(plain: mx.array, rotated: mx.array, key: int) -> None:
    """Keep rotated for the readers of plain while plain is alive."""
    entries = _MEMO.entries

    def drop(ref):
        entries[:] = [e for e in entries if e[0] is not ref]

    entries.append((weakref.ref(plain, drop), rotated, key))
    if len(entries) > _MEMO_SIZE:
        del entries[0]


def _recall(x: mx.array, key: int) -> mx.array | None:
    for ref, rotated, k in reversed(_MEMO.entries):
        if k == key and ref() is x:
            return rotated
    return None


class _Handoff(threading.local):
    def __init__(self):
        self.entry: tuple[int, mx.array] | None = None


_HANDOFF = _Handoff()


def hand_off(key: int, rotated: mx.array) -> None:
    """Leave the rotated input of the linear with this key.

    For producers whose output the caller reshapes before the linear, so the
    memo cannot match the array. The linear takes it once, if the size fits.
    """
    _HANDOFF.entry = (key, rotated)


def _take(key: int, size: int) -> mx.array | None:
    entry = _HANDOFF.entry
    if entry is None or entry[0] != key or entry[1].size != size:
        return None
    _HANDOFF.entry = None
    return entry[1]


class _Rotating(nn.Module):
    """Input rotation shared by OqhLinear and OqhSwitchLinear.

    Dimensions come from the tensors, so they follow tensor-parallel splits.
    """

    # Set per instance by the runtime. The class defaults keep the per-call
    # lookups off nn.Module.__getattr__, which raises for missing names.
    _oqh_key = None
    _oqh_handoff = None
    _oqh_affine = None

    @property
    def input_dims(self) -> int:
        return self["oqh_signs"].size

    @property
    def output_dims(self) -> int:
        return self["weight"].shape[-2]

    @property
    def block(self) -> int:
        return self["oqh_signs"].shape[-1]

    @property
    def signs(self) -> mx.array:
        """The signs as stored, [inputs / block, block]; the kernels read them flat."""
        return self["oqh_signs"]

    def rotate(self, x: mx.array) -> mx.array:
        return fast.rotate(x, self.signs, self.block)

    def rotated_input(self, x: mx.array) -> mx.array:
        """x rotated for this module; reuses the copy of a producer or sibling."""
        handoff = self._oqh_handoff
        if handoff is not None:
            rot = _take(handoff, x.size)
            if rot is not None:
                return rot.reshape(x.shape)
        key = self._oqh_key
        if key is None:
            return self.rotate(x)
        rot = _recall(x, key)
        if rot is None:
            rot = self.rotate(x)
            remember(x, rot, key)
        return rot


class OqhLinear(_Rotating):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bits: int,
        block: int,
        bias: bool = False,
    ):
        super().__init__()
        self.bits = bits
        self.weight = mx.zeros((output_dims, input_dims * bits // 32), dtype=mx.uint32)
        self.rscale = mx.zeros((output_dims,))
        self.oqh_signs = mx.zeros((input_dims // block, block))
        if bias:
            self.bias = mx.zeros((output_dims,))
        self.freeze()

    def gemv_rows(self, x: mx.array) -> bool:
        """Whether x runs the native GEMV."""
        limit = QMV_MAX_ROWS if _EXACT.depth else _GEMV_ROWS
        return x.size // self.input_dims <= limit

    def _affine(self) -> nn.QuantizedLinear:
        """The codes as an affine linear with unit scales and -center biases.

        It shares the weight and the per-shape unit views, so it holds no new
        memory, and its decode-batch calls take oMLX's batched affine kernels.
        """
        ql = self._oqh_affine
        w = self["weight"]
        if ql is None or ql["weight"] is not w:
            s1, b1, gs = fast.unit_view(w, self.bits, self["rscale"].dtype)
            ql = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
            nn.Module.__init__(ql)
            ql.weight, ql.scales, ql.biases = w, s1, b1
            ql.group_size, ql.bits, ql.mode = gs, self.bits, "affine"
            object.__setattr__(self, "_oqh_affine", ql)
        return ql

    def matmul_rotated(self, x: mx.array) -> mx.array:
        if self.gemv_rows(x):
            y = fast.qmv(x, self["weight"], self["rscale"], self.bits)
        elif x.size // self.input_dims <= QMV_MAX_ROWS:
            y = self._affine()(x) * self["rscale"]
        else:
            y = fast.qmm(x, self["weight"], self["rscale"], self.bits)
        if "bias" in self:
            y = y + self["bias"]
        return y

    def pair_rotated(self, x: mx.array, up: OqhLinear) -> mx.array:
        """[x @ W.T | x @ up.T] in one GEMV, for gemv_rows inputs (pairs have no bias)."""
        return fast.qmv(
            x, self["weight"], self["rscale"], self.bits, up["weight"], up["rscale"]
        )

    def __call__(self, x: mx.array) -> mx.array:
        if x.dtype == mx.float32:
            return self(x.astype(self["rscale"].dtype)).astype(mx.float32)
        return self.matmul_rotated(self.rotated_input(x))


class OqhSwitchLinear(_Rotating):
    """Routed experts: weight [E, N, K * bits / 32], rscale [E, N]."""

    def __init__(
        self, input_dims: int, output_dims: int, num_experts: int, bits: int, block: int
    ):
        super().__init__()
        self.num_experts = num_experts
        self.bits = bits
        self.weight = mx.zeros(
            (num_experts, output_dims, input_dims * bits // 32), dtype=mx.uint32
        )
        self.rscale = mx.zeros((num_experts, output_dims))
        self.oqh_signs = mx.zeros((input_dims // block, block))
        self.freeze()

    def gemv_rows(self, x: mx.array, sorted_indices: bool = False) -> bool:
        """Whether x runs the native gather GEMV; callers sort large route sets."""
        return not sorted_indices and x.shape[-2] == 1

    def _gemv(self, x: mx.array, indices: mx.array, *pair) -> mx.array:
        rows = x.size // self.input_dims
        pairs = indices.size
        if rows not in (1, pairs):
            # Batched decode and verify: one row per pair keeps the GEMV.
            x = mx.broadcast_to(x, (*indices.shape, 1, self.input_dims))
            rows = pairs
        y = fast.gather_qmv(
            x.reshape(rows, self.input_dims),
            self["weight"],
            self["rscale"],
            indices.reshape(-1),
            self.bits,
            *pair,
        )
        return y.reshape(*indices.shape, 1, -1)

    def matmul_rotated(
        self, x: mx.array, indices: mx.array, sorted_indices: bool = False
    ):
        if self.gemv_rows(x, sorted_indices):
            return self._gemv(x, indices)
        return fast.gather_qmm(
            x, self["weight"], self["rscale"], indices, self.bits, sorted_indices
        )

    def pair_rotated(
        self, x: mx.array, indices: mx.array, up: OqhSwitchLinear
    ) -> mx.array:
        """[gate | up] routed products in one gather GEMV, for gemv_rows inputs."""
        return self._gemv(x, indices, up["weight"], up["rscale"])

    def __call__(
        self,
        x: mx.array,
        indices: mx.array,
        sorted_indices: bool = False,
        block_plan=None,
    ):
        # block_plan belongs to the affine block kernels of DeepSeek-V4 layouts.
        if x.dtype == mx.float32:
            # The kernels read half-precision activations (DeepSeek V4.1 passes fp32).
            y = self(x.astype(self["rscale"].dtype), indices, sorted_indices)
            return y.astype(mx.float32)
        return self.matmul_rotated(self.rotated_input(x), indices, sorted_indices)


class OqhAffineLinear(_Rotating):
    """Rotated codes with group scales and biases, served by the stock affine kernels."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bits: int,
        group_size: int,
        block: int,
        bias: bool = False,
    ):
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        groups = input_dims // group_size
        self.weight = mx.zeros((output_dims, input_dims * bits // 32), dtype=mx.uint32)
        self.oqh_scales = mx.zeros((output_dims, groups))
        self.oqh_biases = mx.zeros((output_dims, groups))
        self.oqh_signs = mx.zeros((input_dims // block, block))
        if bias:
            self.bias = mx.zeros((output_dims,))
        self.freeze()

    def gemv_rows(self, x: mx.array) -> bool:
        return False

    def _affine(self) -> nn.QuantizedLinear:
        """A stock affine linear sharing the codes, scales and biases.

        Its calls take oMLX's batched affine kernels and verify paths as oQe does.
        """
        ql = self._oqh_affine
        w, s, b = self["weight"], self["oqh_scales"], self["oqh_biases"]
        if ql is None or ql["weight"] is not w or ql["scales"] is not s:
            ql = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
            nn.Module.__init__(ql)
            ql.weight, ql.scales, ql.biases = w, s, b
            ql.group_size, ql.bits, ql.mode = self.group_size, self.bits, "affine"
            object.__setattr__(self, "_oqh_affine", ql)
        return ql

    def matmul_rotated(self, x: mx.array) -> mx.array:
        ql = self._affine()
        # Row-exact forwards give verify rows the one-row affine arithmetic.
        y = _target_verify_linear(ql, x) if _EXACT.depth else ql(x)
        if "bias" in self:
            y = y + self["bias"]
        return y

    def __call__(self, x: mx.array) -> mx.array:
        if x.dtype == mx.float32:
            return self(x.astype(self["oqh_scales"].dtype)).astype(mx.float32)
        return self.matmul_rotated(self.rotated_input(x))


class OqhAffineSwitchLinear(_Rotating):
    """Routed experts with group scales: weight [E, N, K * bits / 32]."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bits: int,
        group_size: int,
        block: int,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.bits = bits
        self.group_size = group_size
        groups = input_dims // group_size
        self.weight = mx.zeros(
            (num_experts, output_dims, input_dims * bits // 32), dtype=mx.uint32
        )
        self.oqh_scales = mx.zeros((num_experts, output_dims, groups))
        self.oqh_biases = mx.zeros((num_experts, output_dims, groups))
        self.oqh_signs = mx.zeros((input_dims // block, block))
        self.freeze()

    def gemv_rows(self, x: mx.array, sorted_indices: bool = False) -> bool:
        return False

    def matmul_rotated(
        self, x: mx.array, indices: mx.array, sorted_indices: bool = False
    ):
        return mx.gather_qmm(
            x,
            self["weight"],
            self["oqh_scales"],
            self["oqh_biases"],
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode="affine",
            sorted_indices=sorted_indices,
        )

    def __call__(
        self,
        x: mx.array,
        indices: mx.array,
        sorted_indices: bool = False,
        block_plan=None,
    ):
        if x.dtype == mx.float32:
            y = self(x.astype(self["oqh_scales"].dtype), indices, sorted_indices)
            return y.astype(mx.float32)
        return self.matmul_rotated(self.rotated_input(x), indices, sorted_indices)


OQH_DENSE = (OqhLinear, OqhAffineLinear)
OQH_SWITCH = (OqhSwitchLinear, OqhAffineSwitchLinear)
OQH_MODULES = OQH_DENSE + OQH_SWITCH


def from_tensors(prefix: str, weights: dict[str, mx.array]) -> nn.Module:
    """Build the module whose tensors sit under ``prefix`` in a checkpoint."""
    w = weights[f"{prefix}.weight"]
    signs = weights[f"{prefix}.oqh_signs"]
    block = int(signs.shape[-1])
    if block > fmt.MAX_BLOCK:
        raise ValueError(f"oQh {prefix}: rotation block {block} > {fmt.MAX_BLOCK}")
    in_dims = int(signs.size)
    bits = int(w.shape[-1]) * 32 // in_dims
    scales = weights.get(f"{prefix}.oqh_scales")
    bias = f"{prefix}.bias" in weights
    if scales is not None:
        group_size = in_dims // int(scales.shape[-1])
        if w.ndim == 3:
            return OqhAffineSwitchLinear(
                in_dims, int(w.shape[1]), int(w.shape[0]), bits, group_size, block
            )
        return OqhAffineLinear(in_dims, int(w.shape[0]), bits, group_size, block, bias)
    if w.ndim == 3:
        return OqhSwitchLinear(in_dims, int(w.shape[1]), int(w.shape[0]), bits, block)
    return OqhLinear(in_dims, int(w.shape[0]), bits, block, bias=bias)
