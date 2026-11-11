# SPDX-License-Identifier: Apache-2.0
"""MLX-side helpers shared by the family stages (workers only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from . import checkpoint
from .wire import HostArray


def memory_limit() -> int:
    """GPU bytes a stage may hold before it gives a prompt back, or 0."""
    info = mx.device_info()
    total = info.get("total_memory") or info.get("memory_size") or 0
    return int(total * 0.95)


def check_backend() -> None:
    """Refuse an MLX CUDA build without the patches in mlx-build/.

    Stock MLX rounds twice when it rebuilds an affine weight in a 16-bit type
    and lets cuBLAS round the partial sums of some 16-bit products.
    """
    if not mx.cuda.is_available():
        return
    mx.random.seed(0)
    x = mx.random.normal((1, 8192, 2560)).astype(mx.bfloat16)
    w = (mx.random.normal((2560, 512)) * 0.02).astype(mx.bfloat16)
    once = (x.astype(mx.float32) @ w.astype(mx.float32)).astype(mx.bfloat16)
    equal = (x @ w == once).astype(mx.float32).mean().item()
    packed, scales, biases = mx.quantize(
        mx.random.normal((64, 256)) * 0.02, group_size=64, bits=4
    )
    low = [scales.astype(mx.bfloat16), biases.astype(mx.bfloat16)]
    wide = [a.astype(mx.float32) for a in low]
    rebuilt = mx.dequantize(packed, *low, group_size=64, bits=4)
    exact = mx.dequantize(packed, *wide, group_size=64, bits=4).astype(mx.bfloat16)
    if equal < 0.99 or not mx.array_equal(rebuilt, exact).item():
        raise RuntimeError(
            "this MLX CUDA build rounds 16-bit products or affine weights "
            "more than once; build MLX with the patches in mlx-build/"
        )


def to_host(array: mx.array) -> HostArray:
    """An evaluated array as contiguous numpy data and its MLX dtype name."""
    dtype = str(array.dtype).rsplit(".", 1)[-1]
    if array.dtype == mx.bfloat16:
        # numpy has no bfloat16; the bits travel as uint16.
        array = array.view(mx.uint16)
    array = mx.contiguous(array)
    mx.eval(array)
    return np.ascontiguousarray(np.asarray(array)), dtype


def from_host(value: HostArray) -> mx.array:
    data, dtype = value
    array = mx.array(data)
    if dtype == "bfloat16":
        array = array.view(mx.bfloat16)
    return array


# Set by the worker where kernels read CUDA managed memory slowly (an integrated
# GPU, WSL): weights then load on the GPU stream, into the GPU's memory pool.
weights_on_gpu = False


def load(model_dir: str | Path, names: list[str]) -> dict[str, mx.array]:
    """The named tensors, lazy until evaluated. Each shard is opened once."""
    root = Path(model_dir)
    index = checkpoint.tensors(root)
    shards: dict[str, dict[str, mx.array]] = {}
    out = {}
    for name in names:
        file = index[name].file
        if file not in shards:
            stream = mx.gpu if weights_on_gpu else None
            shards[file] = mx.load(str(root / file), stream=stream)
        out[name] = shards[file][name]
    return out


def unpack(
    weight: mx.array,
    scales: mx.array,
    biases: mx.array | None,
    quant: dict[str, Any],
    dtype: mx.Dtype | None = None,
) -> mx.array:
    """The dense weight of a quantized tensor, rounded once.

    An affine weight is rebuilt in float32 and rounded to the target type
    afterwards, which gives the values Metal's quantized matmul multiplies with.
    """
    mode = quant.get("mode", "affine")
    sizes = {"group_size": quant["group_size"], "bits": quant["bits"]}
    if mode == "affine":
        target = dtype or scales.dtype
        if biases is not None:
            biases = biases.astype(mx.float32)
        dense = mx.dequantize(weight, scales.astype(mx.float32), biases, **sizes)
        return dense.astype(target)
    # The floating point modes store exponents, not scales of a float type.
    return mx.dequantize(weight, scales, mode=mode, dtype=dtype or mx.bfloat16, **sizes)


class Linear:
    """A bias-free linear layer from a dense or a quantized weight.

    ``dense=True`` unpacks a quantized weight at load: the dense product is
    about twice as fast on CUDA and costs the unpacked bytes of GPU memory.
    """

    def __init__(
        self,
        weights: dict[str, mx.array],
        prefix: str,
        quant: dict[str, Any] | None,
        *,
        dense: bool = True,
        dtype: mx.Dtype | None = None,
    ):
        weight = weights[f"{prefix}.weight"]
        self.quant = None
        if f"{prefix}.scales" in weights:
            scales, biases = weights[f"{prefix}.scales"], weights.get(
                f"{prefix}.biases"
            )
            if dense:
                weight = unpack(weight, scales, biases, quant, dtype)
            else:
                self.quant = (scales, biases, quant)
        elif dtype is not None:
            weight = weight.astype(dtype)
        self.weight = weight

    def parameters(self) -> list[mx.array]:
        extra = [a for a in self.quant[:2] if a is not None] if self.quant else []
        return [self.weight, *extra]

    def __call__(self, x: mx.array) -> mx.array:
        if self.quant is None:
            return x @ self.weight.T
        scales, biases, quant = self.quant
        return mx.quantized_matmul(
            x,
            self.weight,
            scales,
            biases,
            transpose=True,
            group_size=quant["group_size"],
            bits=quant["bits"],
            mode=quant.get("mode", "affine"),
        )
