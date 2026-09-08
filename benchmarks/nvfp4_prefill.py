# SPDX-License-Identifier: Apache-2.0
"""Warm ABBA prefill comparison; run with --model /path/to/model."""

import argparse
import gc
import json
import os
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from omlx.patches.nvfp4_prefill import NVFP4PrefillLinear, apply_nvfp4_prefill


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", type=int, default=2049)
    args = parser.parse_args()
    if args.tokens < 2:
        parser.error("--tokens must be at least 2")
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())
    os.environ["OMLX_NVFP4_PREFILL"] = "1"
    if apply_nvfp4_prefill(model) != 189:
        parser.error("model does not match the qualified dense Qwen NVFP4 layout")
    layers = [
        p
        for l in model.layers[:-1]
        for p in (l.mlp.gate_proj, l.mlp.up_proj, l.mlp.down_proj)
    ]
    ids = tokenizer.encode(
        "Explain database consistency under concurrent updates. " * args.tokens,
        add_special_tokens=False,
    )[: args.tokens]
    x = mx.array([ids])
    mx.eval(x)
    reference = None
    for index, variant in enumerate("ABABBA"):
        for layer in layers:
            layer.__class__ = (
                nn.QuantizedLinear if variant == "A" else NVFP4PrefillLinear
            )
        cache = make_prompt_cache(model)
        mx.reset_peak_memory()
        start = time.perf_counter()
        model(x[:, :-1], cache=cache)
        mx.eval([c.state for c in cache])
        prefix_end = time.perf_counter()
        logits = model(x[:, -1:], cache=cache)[:, -1, :]
        pick = mx.argmax(logits, axis=-1)
        mx.eval(logits, pick)
        end = time.perf_counter()
        if reference is None:
            reference = logits
        print(
            json.dumps(
                dict(
                    variant=variant,
                    warmup=index < 2,
                    tokens=len(ids),
                    prefix_s=prefix_end - start,
                    total_s=end - start,
                    peak_bytes=mx.get_peak_memory(),
                    max_logit_error=mx.max(
                        mx.abs(logits.astype(mx.float32) - reference.astype(mx.float32))
                    ).item(),
                    first_token=pick.tolist(),
                )
            ),
            flush=True,
        )
        del cache, logits, pick
        gc.collect()


if __name__ == "__main__":
    main()
