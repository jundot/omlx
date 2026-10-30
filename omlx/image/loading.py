# SPDX-License-Identifier: Apache-2.0
"""Strict, local-only denoiser loading for the one supported native layout."""

from pathlib import Path

import mlx.core as mx

from omlx.image_bundle import read_json
from omlx.image_inspect import contained, tensor_header

from .conditioning import check_cancel, native_bundle
from .transformer import Transformer


def transformer_shapes():
    result = {
        "img_in.weight": [4096, 64],
        "modulation.weight": [16384, 4096],
        "norm_out.linear.weight": [4096, 4096],
        "proj_out.weight": [64, 4096],
        "time_text_embed.timestep_embedder.linear_1.weight": [4096, 256],
        "time_text_embed.timestep_embedder.linear_2.weight": [4096, 4096],
        "txt_in.in_layer.weight": [4096, 4096],
        "txt_in.out_layer.weight": [4096, 4096],
        "txt_in.text_norm.weight": [4096],
    }
    block = {
        "attn.norm_k.weight": [128],
        "attn.norm_q.weight": [128],
        "attn.to_k.weight": [4096, 4096],
        "attn.to_q.weight": [4096, 4096],
        "attn.to_v.weight": [4096, 4096],
        "attn.to_out.0.weight": [4096, 4096],
        "img_mlp.gate_up.weight": [24576, 4096],
        "img_mlp.out.weight": [4096, 12288],
    }
    for i in range(32):
        result.update({f"transformer_blocks.{i}.{k}": v for k, v in block.items()})
    return result


def load_transformer(bundle, *, cancel=None):
    root = native_bundle(bundle)
    check_cancel(cancel)
    config = read_json(contained(root, "transformer/config.json"))
    expected = {
        "_class_name": "QwenImage21Transformer2DModel",
        "num_layers": 32,
        "num_attention_heads": 32,
        "attention_head_dim": 128,
        "context_in_dim": 4096,
        "in_channels": 64,
        "out_channels": 64,
        "patch_size": 1,
        "mlp_ratio": 3,
        "axes_dims_rope": [16, 56, 56],
        "causal_condition": True,
        "eps": 1e-6,
    }
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("unsupported denoiser architecture")
    index = read_json(contained(root, "transformer/model.safetensors.index.json"))[
        "weight_map"
    ]
    shapes = transformer_shapes()
    if not isinstance(index, dict) or set(index) != set(shapes):
        raise ValueError("denoiser tensor names do not match native layout v1")
    paths = []
    seen = set()
    for shard in sorted(set(index.values())):
        path = contained(root, str(Path("transformer") / shard))
        if not path.is_relative_to(root / "transformer"):
            raise ValueError("denoiser shard escapes component")
        header, _, _, _ = tensor_header(path, False)
        for name, item in header.items():
            if (
                index.get(name) != shard
                or item["shape"] != shapes.get(name)
                or item["dtype"] != "BF16"
                or name in seen
            ):
                raise ValueError(f"incompatible denoiser tensor: {name}")
            seen.add(name)
        paths.append(path)
    if seen != set(shapes):
        raise ValueError("missing denoiser tensors")
    model = Transformer()
    weights = {}
    for path in paths:
        check_cancel(cancel)
        weights.update(mx.load(str(path)))
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    check_cancel(cancel)
    return model
