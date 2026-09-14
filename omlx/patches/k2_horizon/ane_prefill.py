# SPDX-License-Identifier: Apache-2.0
"""Opt-in ANE MLP prefill for dense and MoVA K2 models."""

import math
import struct
import weakref

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.activations import swiglu

from .compiled import CompiledBody
from .uno_adapter import ConditionalLoRALinear

TILE = 2048
FRACTION = 1 / 3


class _Blob:
    def __init__(self):
        self.data = bytearray(64)
        self.count = 0

    def add(self, tensor):
        values = np.asarray(tensor.astype(mx.float32))
        if not np.isfinite(values).all() or (np.abs(values) > 65504).any():
            raise ValueError("ANE weights must be finite and representable in FP16")
        raw = values.astype(np.float16).tobytes()
        self.data.extend(bytes((-len(self.data)) % 64))
        offset = len(self.data)
        header = bytearray(64)
        struct.pack_into("<IIQQ", header, 0, 0xDEADBEEF, 1, len(raw), offset + 64)
        self.data.extend(header)
        self.data.extend(raw)
        self.count += 1
        return offset

    def array(self):
        struct.pack_into("<II", self.data, 0, self.count, 2)
        result = mx.array(np.frombuffer(self.data, dtype=np.uint8))
        mx.eval(result)
        return result


def _projection_statements(weight, width, blob, prefix, source="x"):
    out_dim, in_dim = weight.shape
    shape = f"[{out_dim}, {in_dim}, 1, 1]"
    offset = blob.add(weight)
    return "\n".join(
        [
            f'    tensor<fp16, {shape}> {prefix}_w = const()[name=string("{prefix}_w"), val=tensor<fp16, {shape}>(BLOBFILE(path=string("@model_path/weights/weight.bin"), offset=uint64({offset})))];',
            f'    tensor<fp16, [1, {out_dim}, 1, {width}]> {prefix}_y = conv(x={source}, weight={prefix}_w, strides=tensor<int32, [2]>([1,1]), pad_type=string("valid"), pad=tensor<int32, [4]>([0,0,0,0]), dilations=tensor<int32, [2]>([1,1]), groups=int32(1))[name=string("{prefix}_y")];',
        ]
    )


def _full_mlp_procedure(projections, width, blob):
    """Undo input scaling before SiLU. Use separate down-input scaling."""
    dim = projections[0].shape[1]
    hidden = projections[0].shape[0]
    lines = [
        f"  func procedure000<ios18>(tensor<fp16, [1, {dim + 1}, 1, {width}]> input) {{",
        f'    tensor<fp16, [1, {dim}, 1, {width}]> x = slice_by_size(x=input, begin=tensor<int32, [4]>([0,0,0,0]), size=tensor<int32, [4]>([1,{dim},1,{width}]))[name=string("slice_x")];',
        f'    tensor<fp16, [1, 1, 1, {width}]> input_scale = slice_by_size(x=input, begin=tensor<int32, [4]>([0,{dim},0,0]), size=tensor<int32, [4]>([1,1,1,{width}]))[name=string("input_scale")];',
    ]
    lines.append(_projection_statements(projections[0], width, blob, "g"))
    lines.append(_projection_statements(projections[1], width, blob, "u"))
    shape = f"[1, {hidden}, 1, {width}]"
    lines += [
        f'    tensor<fp16, {shape}> gate = real_div(x=g_y, y=input_scale)[name=string("gate")];',
        f'    tensor<fp16, {shape}> up = real_div(x=u_y, y=input_scale)[name=string("up")];',
    ]
    lines += [
        f'    tensor<fp16, {shape}> half_gate = mul(x=gate, y=fp16(0.5))[name=string("half_gate")];',
        f'    tensor<fp16, {shape}> gate_tanh = tanh(x=half_gate)[name=string("gate_tanh")];',
        f'    tensor<fp16, {shape}> gate_factor = add(x=gate_tanh, y=fp16(1.0))[name=string("gate_factor")];',
        f'    tensor<fp16, {shape}> activated = mul(x=half_gate, y=gate_factor)[name=string("activated")];',
    ]
    lines += [
        f'    tensor<fp16, {shape}> act = mul(x=activated, y=up)[name=string("act")];',
        f'    tensor<fp16, {shape}> act_abs = abs(x=act)[name=string("act_abs")];',
        f'    tensor<fp16, [1, 1, 1, {width}]> act_max = reduce_max(x=act_abs, axes=tensor<int32, [1]>([1]), keep_dims=bool(true))[name=string("act_max")];',
        f'    tensor<fp16, [1, 1, 1, {width}]> bounded_max = maximum(x=act_max, y=fp16(64.0))[name=string("bounded_max")];',
        f'    tensor<fp16, [1, 1, 1, {width}]> down_scale = real_div(x=fp16(1024.0), y=bounded_max)[name=string("down_scale")];',
        f'    tensor<fp16, {shape}> down_input = mul(x=act, y=down_scale)[name=string("down_input")];',
    ]
    lines.append(_projection_statements(projections[2], width, blob, "d", "down_input"))
    out = projections[2].shape[0]
    lines += [
        f'    tensor<fp16, [1, {out}, 1, {width}]> unscaled = real_div(x=d_y, y=down_scale)[name=string("unscaled")];',
        f'    tensor<fp16, [1, {out}, 1, {width}]> result = mul(x=unscaled, y=fp16(1.0))[name=string("result")];',
        "  } -> (result);",
    ]
    return "\n".join(lines)


def _linear(ref):
    return ref.linear if isinstance(ref, ConditionalLoRALinear) else ref


def _projection_part(linear, cut, *, down=False, suffix=False):
    if isinstance(linear, nn.QuantizedLinear):
        if getattr(linear, "mode", "affine") != "affine":
            raise ValueError("K2 ANE prefill requires affine quantization")
        bits, group = linear.bits, linear.group_size
        if cut % group:
            raise ValueError("ANE partition must align to quantization groups")
        if down:
            cols = (
                slice(cut * bits // 32, None)
                if suffix
                else slice(None, cut * bits // 32)
            )
            groups = slice(cut // group, None) if suffix else slice(None, cut // group)
            values = (
                linear.weight[:, cols],
                linear.scales[:, groups],
                linear.biases[:, groups],
            )
        else:
            rows = slice(cut, None) if suffix else slice(None, cut)
            values = (linear.weight[rows], linear.scales[rows], linear.biases[rows])
        values = tuple(mx.contiguous(v) for v in values)
        mx.eval(values)
        return values, dict(bits=bits, group_size=group)
    rows = slice(cut, None) if suffix else slice(None, cut)
    return (linear.weight[:, rows] if down else linear.weight[rows],), None


def _dense_part(part):
    values, quant = part
    return mx.dequantize(*values, **quant) if quant else values[0]


def _matmul(x, part):
    values, quant = part
    return (
        mx.quantized_matmul(x, *values, transpose=True, **quant)
        if quant
        else x @ values[0].T
    )


class PrefillMLP:
    def __init__(self, reference, *, cut, width=TILE):
        from omlx.custom_kernels.qwen35_prefill import fast

        if fast._ext is None or not hasattr(fast._ext, "ane_compile_program"):
            raise RuntimeError("K2 ANE prefill requires the native ANE extension")
        projections = tuple(
            _linear(getattr(reference, n))
            for n in ("gate_proj", "up_proj", "down_proj")
        )
        hidden = projections[0].weight.shape[0]
        dim = projections[2].weight.shape[0]
        if not 0 < cut <= hidden or width < 32 or width % 32:
            raise ValueError("Invalid K2 ANE prefill partition")
        blob = _Blob()
        ane_weights = tuple(
            _dense_part(_projection_part(ref, cut, down=i == 2))
            for i, ref in enumerate(projections)
        )
        proc = _full_mlp_procedure(ane_weights, width, blob)
        del ane_weights
        source = (
            'program(1.3)\n[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3520.4.1"}, {"coremlc-version", "3520.5.1"}})]\n{\n'
            + proc
            + "\n}\n"
        )
        self.program = fast._ext.ane_compile_program(
            source, blob.array(), dim + 1, dim, width
        )
        self.width = width
        base = projections[0]
        self.dtype = (
            base.scales.dtype
            if isinstance(base, nn.QuantizedLinear)
            else base.weight.dtype
        )
        parts = (
            tuple(
                _projection_part(ref, cut, down=i == 2, suffix=True)
                for i, ref in enumerate(projections)
            )
            if cut < hidden
            else ()
        )
        mx.eval([v for values, _ in parts for v in values])

        # Keep original float views. Materialize packed GPU suffixes once.
        def gpu(x):
            if not parts:
                return mx.zeros_like(x)
            gate, up, down = parts
            return _matmul(swiglu(_matmul(x, gate), _matmul(x, up)), down)

        self._gpu = mx.compile(gpu)

    def prepare(self, x):
        if x.shape[0] != 1 or not 0 < x.shape[1] <= self.width or x.dtype != self.dtype:
            raise ValueError(
                "K2 ANE prefill requires one prompt tile in the base activation dtype"
            )
        count = x.shape[1]
        if count < self.width:
            x = mx.pad(x, [(0, 0), (0, self.width - count), (0, 0)])
        rows = x.reshape(-1, x.shape[-1])
        maximum = mx.maximum(
            mx.max(mx.abs(rows).astype(mx.float32), axis=-1, keepdims=True), 1e-20
        )
        scales = mx.power(2, mx.minimum(4, mx.floor(mx.log2(1024 / maximum)))).astype(
            x.dtype
        )
        packed = mx.concatenate([rows * scales, scales], axis=-1)
        planar = mx.contiguous(packed.astype(mx.float16).T)
        # Complete packing before submitting the independent GPU and ANE work.
        mx.eval(planar, scales)
        return x, planar, count

    def finish(self, prepared, *, alongside=None):
        from omlx.custom_kernels.qwen35_prefill import fast

        x, planar, count = prepared
        gpu = self._gpu(x)
        mx.async_eval(gpu if alongside is None else (gpu, alongside))
        ane = fast._ext.ane_planar(planar, self.program)
        mx.eval(ane)
        return (gpu + ane.T.astype(x.dtype).reshape(x.shape))[:, :count]

    def __call__(self, x):
        return self.finish(self.prepare(x))


def partition_channels(hidden, fraction, alignment=64):
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("K2 ANE prefill fraction must be in (0, 1].")
    cut = hidden if fraction == 1 else int(hidden * fraction) // alignment * alignment
    if not 0 < cut <= hidden:
        raise ValueError(
            "K2 ANE allocation must contain at least one aligned channel group."
        )
    return cut


def prefill_memory_reservation(
    config, fraction=FRACTION, shared_fraction=1.0, width=TILE
):
    dims = config["hidden_size"]
    layers = config["num_hidden_layers"]
    dense = set(config.get("mlp_only_layers", []))
    step = config.get("decoder_sparse_step", 1)
    weights = surfaces = largest = gpu_copies = 0
    quantization = config.get("quantization") or {}
    for i in range(layers - 1):
        sparse = config.get("num_experts", 0) and i not in dense and (i + 1) % step == 0
        hidden = (
            config["moe_intermediate_size"] if sparse else config["intermediate_size"]
        )
        share = shared_fraction if sparse else fraction
        if share == 0:
            continue
        cut = partition_channels(hidden, share)
        size = 3 * dims * cut * 2
        prefix = f"model.layers.{i}.mlp" + (".shared_experts" if sparse else "")
        for name in ("gate_proj", "up_proj", "down_proj"):
            quant = quantization.get(f"{prefix}.{name}", quantization)
            if quant and hidden > cut:
                rows, cols = (
                    (dims, hidden - cut)
                    if name == "down_proj"
                    else (hidden - cut, dims)
                )
                gpu_copies += math.ceil(rows * cols * quant["bits"] / 8)
                gpu_copies += rows * math.ceil(cols / quant["group_size"]) * 4
        weights += size
        largest = max(largest, size)
        surfaces += (2 * dims + 1) * width * 2
    # Allow compiled weights, retained blobs, surfaces, and one staging program.
    return 3 * weights + 2 * surfaces + largest + gpu_copies


def enable_ane_prefill(model, *, fraction=FRACTION, shared_fraction=1.0, width=TILE):
    if getattr(model, "model_type", None) != "k2_horizon":
        raise ValueError("ANE prefill requires a K2 model")
    if type(width) is not int or width < 32 or width % 32:
        raise ValueError("K2 ANE prefill tile must be a multiple of 32")
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Invalid K2 ANE dense MLP fraction")
    references = []
    if not math.isfinite(shared_fraction) or not 0 <= shared_fraction <= 1:
        raise ValueError("Invalid K2 ANE shared MLP fraction")
    for layer in model.layers[:-1]:
        sparse = hasattr(layer.mlp, "shared_experts")
        if sparse and shared_fraction == 0:
            continue
        mlp = getattr(layer.mlp, "shared_experts", layer.mlp)
        for name in ("gate_proj", "up_proj", "down_proj"):
            ref = getattr(mlp, name)
            linear = _linear(ref)
            if not isinstance(linear, (nn.Linear, nn.QuantizedLinear)):
                raise ValueError("ANE prefill requires linear K2 MLP projections")
            dtype = (
                linear.scales.dtype
                if isinstance(linear, nn.QuantizedLinear)
                else linear.weight.dtype
            )
            if dtype not in (mx.bfloat16, mx.float16) or "bias" in linear:
                raise ValueError(
                    "ANE prefill requires FP16/BF16 activations and bias-free projections"
                )
        gate = _linear(mlp.gate_proj).weight
        alignment = max(
            64,
            *(
                getattr(_linear(getattr(mlp, n)), "group_size", 64)
                for n in ("gate_proj", "up_proj", "down_proj")
            ),
        )
        references.append(
            (
                mlp,
                partition_channels(
                    gate.shape[0], shared_fraction if sparse else fraction, alignment
                ),
            )
        )
    if not references:
        raise ValueError("K2 model has no prefill MLPs to partition")
    programs = tuple(PrefillMLP(ref, cut=cut, width=width) for ref, cut in references)
    for (ref, _), program in zip(references, programs):
        program.active = False
        ref._omlx_ane_prefill = program
    if isinstance(model.model, CompiledBody):
        model.model._prefill_mlps = programs

    model_ref = weakref.ref(model)
    # Only pad near-full tails: at most 1/32 of a tile (64 rows at 2048).
    min_ane_rows = width - width // 32

    def prefill(inputs, *, cache):
        target = model_ref()
        if target is None:
            raise RuntimeError("K2 ANE model has been unloaded")
        if inputs.shape[0] != 1:
            raise ValueError("K2 ANE prefill requires one prompt per forward")
        try:
            for start in range(0, inputs.shape[1], width):
                chunk = inputs[:, start : start + width]
                use_ane = chunk.shape[1] >= min_ane_rows
                for program in programs:
                    program.active = use_ane
                if isinstance(target.model, CompiledBody):
                    target.model(chunk, cache=cache, prefill=use_ane)
                else:
                    target(chunk, cache=cache)
                mx.eval([c.state for c in cache])
        finally:
            for program in programs:
                program.active = False

    model._omlx_prefill = prefill
    model._omlx_k2_ane_signature = (
        f"k2-ane-v3-tail31of32-{fraction:.17g}-{shared_fraction:.17g}-{width}"
        + ("-compiled" if isinstance(model.model, CompiledBody) else "")
    )
    model._omlx_k2_ane_prefill_count = len(programs)
    return prefill
