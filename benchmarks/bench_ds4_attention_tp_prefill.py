#!/usr/bin/env python3
"""Profile DS4 TP4/4 attention prefill with all 43 real projection weights."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from contextlib import ExitStack
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open


def qmm(x, weight, scales):
    return mx.quantized_matmul(
        x,
        weight,
        scales=scales,
        biases=None,
        transpose=True,
        group_size=32,
        bits=8,
        mode="mxfp8",
    )


class Weights:
    def __init__(self, model: Path, *, rank: int):
        self.model = model
        self.rank = rank
        self.index = json.loads(
            (model / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self.stack = ExitStack()
        self.files = {}
        self.layers = []
        for layer in range(43):
            wqa = self.quant(layer, "wq_a")
            wkv = self.quant(layer, "wkv")
            wqb = self.segmented_output_shard(*self.quant_np(layer, "wq_b"))
            woa = self.grouped_input_shard(*self.quant_np(layer, "wo_a"))
            wob = self.quant(layer, "wo_b")
            # safetensors' NumPy adapter cannot materialize BF16 on this build.
            # Norm values do not affect dispatch/timing, so use shape-exact BF16
            # ones while every matrix byte remains the real checkpoint tensor.
            qnorm = mx.ones((1024,), dtype=mx.bfloat16)
            kvnorm = mx.ones((512,), dtype=mx.bfloat16)
            pair = (
                mx.concatenate([wqa[0], wkv[0]], axis=0),
                mx.concatenate([wqa[1], wkv[1]], axis=0),
            )
            self.layers.append(
                {
                    "wqa": wqa,
                    "wkv": wkv,
                    "wqb": wqb,
                    "woa": woa,
                    "wob": wob,
                    "qnorm": qnorm,
                    "kvnorm": kvnorm,
                    "wqa_wkv": pair,
                }
            )
        mx.eval(
            [
                value
                for layer in self.layers
                for item in layer.values()
                for value in (item if isinstance(item, tuple) else (item,))
            ]
        )

    def close(self):
        self.stack.close()

    def source(self, key: str):
        filename = self.index[key]
        if filename not in self.files:
            self.files[filename] = self.stack.enter_context(
                safe_open(self.model / filename, framework="numpy")
            )
        return self.files[filename]

    def tensor_np(self, key: str):
        return self.source(key).get_tensor(key)

    def tensor(self, key: str):
        return mx.array(self.tensor_np(key))

    def quant_np(self, layer: int, name: str):
        prefix = f"layers.{layer}.attn.{name}"
        return self.tensor_np(prefix + ".weight"), self.tensor_np(prefix + ".scales")

    def quant(self, layer: int, name: str):
        weight, scales = self.quant_np(layer, name)
        return mx.array(weight), mx.array(scales)

    def segmented_output_shard(self, weight: np.ndarray, scales: np.ndarray):
        # wq_b output rows are eight o_group segments. TP4/4 retains half of
        # every segment, not one contiguous half of the complete matrix.
        weight_segments = np.split(weight, 8, axis=0)
        scale_segments = np.split(scales, 8, axis=0)
        start = self.rank * (weight_segments[0].shape[0] // 2)
        stop = start + weight_segments[0].shape[0] // 2
        return (
            mx.array(np.concatenate([part[start:stop] for part in weight_segments])),
            mx.array(np.concatenate([part[start:stop] for part in scale_segments])),
        )

    def grouped_input_shard(self, weight: np.ndarray, scales: np.ndarray):
        # sanitize reshapes wo_a output rows into eight independent groups;
        # sharded-to-all slices the packed contraction/group axis.
        weight = weight.reshape(8, 1024, -1)
        scales = scales.reshape(8, 1024, -1)
        width = weight.shape[-1] // 2
        scale_width = scales.shape[-1] // 2
        start = self.rank * width
        scale_start = self.rank * scale_width
        return (
            mx.array(weight[..., start : start + width]),
            mx.array(scales[..., scale_start : scale_start + scale_width]),
        )


def run_timed(fn, *, cycles: int):
    fn()
    mx.synchronize()
    samples = []
    for _ in range(cycles):
        started = time.perf_counter()
        fn()
        mx.synchronize()
        samples.append(time.perf_counter() - started)
    return {
        "median_ms": statistics.median(samples) * 1000,
        "samples_ms": [sample * 1000 for sample in samples],
    }


def profile_width(weights: Weights, rows: int, *, cycles: int):
    x = mx.random.normal((rows, 4096)).astype(mx.bfloat16)
    qres = mx.random.normal((rows, 1024)).astype(mx.bfloat16)
    woa_input = mx.random.normal((8, rows, 2048)).astype(mx.bfloat16)
    wob_input = mx.random.normal((rows, 8192)).astype(mx.bfloat16)
    attention_out = mx.random.normal((1, 32, rows, 512)).astype(mx.bfloat16)
    mx.eval(x, qres, woa_input, wob_input, attention_out)

    def every(call):
        def run():
            for layer in weights.layers:
                mx.eval(call(layer))

        return run

    components = {}
    components["wq_a"] = run_timed(
        every(lambda layer: qmm(x, *layer["wqa"])), cycles=cycles
    )
    components["q_norm"] = run_timed(
        every(
            lambda layer: mx.fast.rms_norm(
                qres, layer["qnorm"], 1e-6
            )
        ),
        cycles=cycles,
    )
    components["wq_b_local"] = run_timed(
        every(lambda layer: qmm(qres, *layer["wqb"])), cycles=cycles
    )
    components["wq_a_norm_wq_b"] = run_timed(
        every(
            lambda layer: qmm(
                mx.fast.rms_norm(qmm(x, *layer["wqa"]), layer["qnorm"], 1e-6),
                *layer["wqb"],
            )
        ),
        cycles=cycles,
    )
    components["wkv_norm"] = run_timed(
        every(
            lambda layer: mx.fast.rms_norm(
                qmm(x, *layer["wkv"]), layer["kvnorm"], 1e-6
            )
        ),
        cycles=cycles,
    )

    def input_current(layer):
        qa = qmm(x, *layer["wqa"])
        q = qmm(mx.fast.rms_norm(qa, layer["qnorm"], 1e-6), *layer["wqb"])
        kv = mx.fast.rms_norm(qmm(x, *layer["wkv"]), layer["kvnorm"], 1e-6)
        return q, kv

    def input_paired(layer):
        qakv = qmm(x, *layer["wqa_wkv"])
        qa, kv = qakv[..., :1024], qakv[..., 1024:]
        q = qmm(mx.fast.rms_norm(qa, layer["qnorm"], 1e-6), *layer["wqb"])
        kv = mx.fast.rms_norm(kv, layer["kvnorm"], 1e-6)
        return q, kv

    components["input_chain_current"] = run_timed(
        every(input_current), cycles=cycles
    )
    components["input_chain_pair_wqa_wkv"] = run_timed(
        every(input_paired), cycles=cycles
    )
    current = components["input_chain_current"]["median_ms"]
    paired = components["input_chain_pair_wqa_wkv"]["median_ms"]
    components["input_chain_pair_gain"] = current / paired

    components["wo_a_local"] = run_timed(
        every(lambda layer: qmm(woa_input, *layer["woa"])), cycles=cycles
    )
    components["wo_b_replicated"] = run_timed(
        every(lambda layer: qmm(wob_input, *layer["wob"])), cycles=cycles
    )

    def output_chain(layer):
        latent = qmm(woa_input, *layer["woa"])
        latent = latent.transpose(1, 0, 2).reshape(rows, 8192)
        return qmm(latent, *layer["wob"])

    components["wo_a_wo_b"] = run_timed(every(output_chain), cycles=cycles)

    # The current TP kernel rejects H=32 and falls through to stock SDPA. The
    # existing Metal source itself is head-count agnostic; dispatching 8 rather
    # than 16 four-head groups is the candidate measured here.
    from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
    from omlx.patches.deepseek_v4 import wsdpa_attention as ws

    kv = mx.random.normal((1, 1, rows, 512)).astype(mx.bfloat16)
    sinks = mx.zeros((32,), dtype=mx.bfloat16)
    mask = create_attention_mask(
        mx.zeros((1, rows, 1), dtype=mx.bfloat16),
        window_size=128,
        return_array=True,
    )
    kernel = ws._get_kernel()
    params = mx.array([0, 128, 1, 0, rows, rows], dtype=mx.int32)
    scale = mx.array([512**-0.5], dtype=mx.float32)
    dummy_pool = mx.zeros((1, 512), dtype=mx.bfloat16)

    def stock_attention():
        for _ in range(43):
            out = scaled_dot_product_attention(
                attention_out,
                kv,
                kv,
                cache=None,
                scale=512**-0.5,
                mask=mask,
                sinks=sinks,
            )
            mx.eval(out)

    def tp_wsdpa():
        for _ in range(43):
            out = kernel(
                inputs=[
                    mx.contiguous(attention_out[0]),
                    mx.contiguous(kv[0, 0]),
                    dummy_pool,
                    sinks,
                    params,
                    scale,
                ],
                grid=(8 * 128, rows, 1),
                threadgroup=(128, 1, 1),
                output_shapes=[(32, rows, 512)],
                output_dtypes=[mx.bfloat16],
            )[0]
            mx.eval(out)

    components["attention_stock_h32"] = run_timed(stock_attention, cycles=cycles)
    components["attention_wsdpa_h32_candidate"] = run_timed(tp_wsdpa, cycles=cycles)
    components["attention_wsdpa_gain"] = (
        components["attention_stock_h32"]["median_ms"]
        / components["attention_wsdpa_h32_candidate"]["median_ms"]
    )
    stock = scaled_dot_product_attention(
        attention_out,
        kv,
        kv,
        cache=None,
        scale=512**-0.5,
        mask=mask,
        sinks=sinks,
    )
    candidate = kernel(
        inputs=[
            mx.contiguous(attention_out[0]),
            mx.contiguous(kv[0, 0]),
            dummy_pool,
            sinks,
            params,
            scale,
        ],
        grid=(8 * 128, rows, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(32, rows, 512)],
        output_dtypes=[mx.bfloat16],
    )[0][None]
    mx.eval(stock, candidate)
    delta = mx.abs(stock.astype(mx.float32) - candidate.astype(mx.float32))
    components["attention_wsdpa_parity"] = {
        "exact_fraction": float(mx.mean(stock == candidate).item()),
        "max_abs": float(mx.max(delta).item()),
        "mean_abs": float(mx.mean(delta).item()),
    }
    return components


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--rank", type=int, default=0, choices=(0, 1))
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--widths", default="512,1024,2048")
    parser.add_argument("--collective-latency-us", type=float, default=28.5)
    parser.add_argument("--collective-bandwidth-gbps", type=float, default=6.2)
    args = parser.parse_args()
    mx.random.seed(17)
    weights = Weights(args.model.expanduser(), rank=args.rank)
    try:
        output = {
            "device": mx.device_info(),
            "rank": args.rank,
            "layers": len(weights.layers),
            "widths": {},
        }
        for rows in (int(value) for value in args.widths.split(",")):
            components = profile_width(weights, rows, cycles=args.cycles)
            payload = rows * 4096 * 2
            per_layer = args.collective_latency_us * 1e-6 + payload / (
                args.collective_bandwidth_gbps * 1e9
            )
            components["attention_output_all_sum_priced"] = {
                "payload_bytes_per_layer": payload,
                "milliseconds_43_layers": per_layer * 43 * 1000,
            }
            output["widths"][str(rows)] = components
        print(json.dumps(output, indent=2, sort_keys=True))
    finally:
        weights.close()


if __name__ == "__main__":
    main()
