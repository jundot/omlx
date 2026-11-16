"""oQh linear modules.

``__call__`` takes the activation in the plain basis, so any caller gets
correct results. It first looks for a rotated copy of that exact array left by
a fused producer or a sibling linear with the same rotation (a small per-thread
memo keyed by array identity), and rotates it itself otherwise.
``matmul_rotated`` takes an activation that is already rotated.
"""

from __future__ import annotations

import threading

import mlx.core as mx
import mlx.nn as nn

from ..custom_kernels.oqh import fast

# Up to this many rows run the native GEMV once per row: MTP verify batches
# then use the same arithmetic as one-token decode. Larger batches go through
# the stock quantized matmul.
QMV_MAX_ROWS = 16
_MEMO_SIZE = 8


class _Memo(threading.local):
    def __init__(self):
        self.entries: list[tuple[mx.array, mx.array, int]] = []


_MEMO = _Memo()


def remember(plain: mx.array, rotated: mx.array, key: int) -> None:
    entries = _MEMO.entries
    entries.append((plain, rotated, key))
    if len(entries) > _MEMO_SIZE:
        del entries[0]


def _recall(x: mx.array, key: int) -> mx.array | None:
    for plain, rotated, k in reversed(_MEMO.entries):
        if plain is x and k == key:
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
        return self["oqh_signs"].reshape(-1)

    def update(self, parameters: dict, strict: bool = True) -> nn.Module:
        shape = self["weight"].shape
        out = super().update(parameters, strict)
        # A tensor-parallel split of gate and up fuses the halves again.
        on_split = getattr(self, "_oqh_on_split", None)
        if on_split is not None and self["weight"].shape != shape:
            on_split()
        return out

    def rotate(self, x: mx.array) -> mx.array:
        return fast.rotate(x, self.signs, self.block)

    def rotated_input(self, x: mx.array) -> mx.array:
        """x rotated for this module; reuses the copy of a producer or sibling."""
        handoff = getattr(self, "_oqh_handoff", None)
        if handoff is not None:
            rot = _take(handoff, x.size)
            if rot is not None:
                return rot.reshape(x.shape)
        key = getattr(self, "_oqh_key", None)
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

    def matmul_rotated(self, x: mx.array) -> mx.array:
        if x.size // self.input_dims <= QMV_MAX_ROWS:
            y = fast.qmv(x, self["weight"], self["rscale"], self.bits)
        else:
            y = fast.qmm(x, self["weight"], self["rscale"], self.bits)
        if "bias" in self:
            y = y + self["bias"]
        return y

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

    def matmul_rotated(
        self, x: mx.array, indices: mx.array, sorted_indices: bool = False
    ):
        rows = x.size // self.input_dims
        pairs = indices.size
        # Callers sort large route sets; unsorted ones are decode-sized.
        if not sorted_indices and x.shape[-2] == 1:
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
            )
            return y.reshape(*indices.shape, 1, self.output_dims)
        return fast.gather_qmm(
            x, self["weight"], self["rscale"], indices, self.bits, sorted_indices
        )

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


OQH_MODULES = (OqhLinear, OqhSwitchLinear)


def from_tensors(prefix: str, weights: dict[str, mx.array]) -> nn.Module:
    """Build the module whose tensors sit under ``prefix`` in a checkpoint."""
    w = weights[f"{prefix}.weight"]
    signs = weights[f"{prefix}.oqh_signs"]
    block = int(signs.shape[-1])
    in_dims = int(signs.size)
    bits = int(w.shape[-1]) * 32 // in_dims
    if w.ndim == 3:
        return OqhSwitchLinear(in_dims, int(w.shape[1]), int(w.shape[0]), bits, block)
    return OqhLinear(
        in_dims, int(w.shape[0]), bits, block, bias=f"{prefix}.bias" in weights
    )
