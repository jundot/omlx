# SPDX-License-Identifier: MIT
"""Measure layer-output distortion on real V4.1 full-forward inputs."""

from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .loading import set_module
from .oq import requantize_projection
from .quantization import QuantizedProjection


class _CaptureBlock:
    def __init__(self, block, samples):
        self.block = block
        self.samples = samples

    def __contains__(self, name):
        return name in self.block

    def __getattr__(self, name):
        return getattr(self.block, name)

    def __call__(self, h, pre, cache, shared, start, image_mask):
        # Cache writes replace arrays; separate containers preserve the input
        # snapshot while the ordinary forward updates its own cache and KV map.
        self.samples.append((h, pre, cache.extract(0), dict(shared), start, image_mask))
        return self.block(h, pre, cache, shared, start, image_mask)


@contextmanager
def _quantized_block(block, bits, eligible=None, progress=None):
    saved = {}
    try:
        for path, module in tree_flatten(
            block.leaf_modules(), is_leaf=nn.Module.is_module
        ):
            if eligible is not None and path not in eligible:
                continue
            if path.startswith("engram.") or path.endswith("wo_a"):
                # Engram runs outside Block.__call__; wo_a is consumed directly
                # as a dense weight by the attention output contraction.
                continue
            source_spec = None
            if isinstance(module, QuantizedProjection):
                source_spec = dict(
                    bits=module.bits,
                    mode=module.mode,
                    group_size=module.group_size,
                    quantize_input=module.quantize_input,
                )
            elif not isinstance(module, nn.Linear) or "bias" in module:
                continue
            width = module.input_dims if source_spec else module.weight.shape[-1]
            if width % 64:
                continue
            values = {path + ".weight": module.weight}
            if source_spec:
                values[path + ".scales"] = module.scales
                if "biases" in module:
                    values[path + ".biases"] = module.biases
            quantized, spec = requantize_projection(
                values, path, source_spec, bits=bits, progress=progress
            )
            replacement = QuantizedProjection(
                quantized[path + ".weight"],
                quantized[path + ".scales"],
                biases=quantized[path + ".biases"],
                **spec,
            )
            saved[path] = module
            set_module(block, path, replacement)
        yield len(saved)
    finally:
        for path, module in saved.items():
            set_module(block, path, module)


def measure_sensitivity(
    model,
    tokenizer,
    *,
    bits=3,
    num_samples=32,
    seq_length=256,
    calib_dataset="code_multilingual",
    progress=None,
):
    from ...oq import _load_calibration_data

    tokens = _load_calibration_data(tokenizer, calib_dataset, num_samples, seq_length)
    if tokens is None or len(tokens) == 0:
        raise ValueError("V4.1 sensitivity requires nonempty calibration data")
    core = model.language_model
    layers = list(core.layers)
    inputs = [[] for _ in layers]
    try:
        core.layers = [
            _CaptureBlock(layer, rows) for layer, rows in zip(layers, inputs)
        ]
        for index, row in enumerate(tokens):
            cache = core.make_cache()
            logits = core(row[None], cache=cache)
            mx.eval(logits, [item.state for item in cache])
            if progress:
                progress("capture", index + 1, len(tokens))
            del logits, cache
    finally:
        core.layers = layers

    scores = {}
    for index, (block, rows) in enumerate(zip(layers, inputs)):
        if len(rows) != len(tokens):
            raise RuntimeError(f"Missing full-forward samples for V4.1 layer {index}")

        def forward(row, block=block):
            h, pre, cache, shared, start, mask = row
            return block(h, pre, cache.extract(0), dict(shared), start, mask)

        baseline = [forward(row) for row in rows]
        mx.eval(baseline)

        def quantize_progress(done, total, index=index):
            if progress:
                progress("measure", index, len(layers))

        with _quantized_block(block, bits, progress=quantize_progress) as changed:
            if not changed:
                raise ValueError(f"No quantizable projections in V4.1 layer {index}")
            error, power = 0.0, 0.0
            for row, reference in zip(rows, baseline):
                actual = forward(row)
                for original, converted in zip(reference, actual):
                    original = original.astype(mx.float32)
                    converted = converted.astype(mx.float32)
                    error += mx.sum(mx.square(original - converted)).item()
                    power += mx.sum(mx.square(original)).item()
            scores[index] = error / max(power, 1e-10)
        # Release this layer's retained inputs as soon as its measurement ends.
        rows.clear()
        del baseline
        if progress:
            progress("measure", index + 1, len(layers))
    return scores
