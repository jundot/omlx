# SPDX-License-Identifier: Apache-2.0
"""Compile dense K2 GPU regions without tracing mutable KV state."""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention


def make_regions(layer):
    attention = layer.self_attn

    def pre(x, offset, mask):
        return attention.project(layer.input_layernorm(x), offset, mask)

    def prefix(x, out, mask):
        return layer.residual(x, attention.project_output(out, x, mask))

    def post(x, out, mask):
        return layer.mlp_output(*prefix(x, out, mask), lora_mask=mask)

    # Separate traces keep clean and conditional projection paths distinct.
    return {
        (kind, conditional): mx.compile(
            (lambda *args, fn=fn: fn(*args))
            if conditional
            else (lambda *args, fn=fn: fn(*args, None))
        )
        for kind, fn in (("pre", pre), ("post", post), ("prefix", prefix))
        for conditional in (False, True)
        if kind != "prefix" or not conditional
    }


class CompiledBody(nn.Module):
    def __init__(self, body):
        super().__init__()
        self.layers, self.embed_tokens, self.norm = (
            body.layers,
            body.embed_tokens,
            body.norm,
        )
        self._regions = [make_regions(layer) for layer in self.layers]
        self._prefill_mlps = ()

    def __call__(self, inputs, cache=None, lora_mask=None, *, prefill=False):
        if prefill and lora_mask is not None:
            raise ValueError("ANE prefill cannot apply conditional LoRA")
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        offsets = [c.offset if c is not None else 0 for c in cache]
        positions = [getattr(c, "_idx", offset) for c, offset in zip(cache, offsets)]
        if len(cache) != len(self.layers) or any(v != positions[0] for v in positions):
            raise ValueError("Compiled K2 requires aligned layer cache offsets")
        mask = create_attention_mask(h, cache[0])
        conditional = lora_mask is not None
        extra = (lora_mask,) if conditional else ()
        for index, (layer, c, regions) in enumerate(
            zip(self.layers, cache, self._regions)
        ):
            offset = mx.array(offsets[index], dtype=mx.int32)
            q, k, v = regions["pre", conditional](h, offset, *extra)
            if c is not None:
                k, v = c.update_and_fetch(k, v)
            out = scaled_dot_product_attention(
                q, k, v, cache=c, scale=layer.self_attn.scale, mask=mask
            )
            if prefill and index < len(self._prefill_mlps):
                residual, norm = regions["prefix", False](h, out)
                h = residual + self._prefill_mlps[index](norm)
            else:
                h = regions["post", conditional](h, out, *extra)
        return self.norm(h)


def can_compile_blocks(model):
    args = model.args
    return (
        not args.num_experts
        and args.attention_gate_func is None
        and args.rope_parameters.get("rope_type", "default") == "default"
        and args.rope_head_dim == args.head_dim
    )


def install_compiled_blocks(model):
    args = model.args
    if (
        args.rope_parameters.get("rope_type", "default") != "default"
        or args.rope_head_dim != args.head_dim
    ):
        raise ValueError("Compiled K2 requires full-head default RoPE")
    for layer in model.layers:
        attn = layer.self_attn
        if (
            attn.mova
            or "gate_proj" in attn
            or not isinstance(attn.rope, nn.RoPE)
            or not hasattr(layer.mlp, "gate_proj")
        ):
            raise ValueError("Compiled K2 requires dense ungated layers")
    model.model = CompiledBody(model.model)
