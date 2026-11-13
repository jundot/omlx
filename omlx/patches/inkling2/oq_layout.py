# SPDX-License-Identifier: Apache-2.0
"""oQ's Inkling 2 layout: the source checkpoint mapped to the runtime layout.

The source keeps the ``model.llm.*`` names. oQ writes the layout that the
runtime model loads, so ``sanitizer`` stacks the attention projections that
share an input, moves the attention-residual queries onto their layers and
renames the rest. It only uses the ops oQ's streaming plan can replay. The
routed experts need ops it cannot replay (``view`` and a tensor-by-tensor
scale), so ``oq_virtual`` exposes them as ``.weight`` and ``.global_scale``
tensors before this mapping runs.

Calibration runs on the packed source model, resident, through the runtime
model (the source is about 390 GB of 4/8-bit experts, and no proxy is smaller).
The helpers below give oQ's layer walk what that model needs: Inkling 2 layers
exchange the completed attention-residual blocks, so the walk carries them, and
the head input is the last layer's mixture over all blocks.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import mlx.core as mx

MODEL_TYPE = "inkling2"
SOURCE_PREFIX = "model.llm."
# ModelOpt stores the NVFP4 tensor scale as amax / (E4M3 max x E2M1 max). MLX
# keeps the amax, and the runtime applies global_scale / NVFP4_GLOBAL_SCALE.
NVFP4_GLOBAL_SCALE = 448.0 * 6.0
# The only embedding table: a row lookup, so oQ never measures its inputs.
EMBED_PATH = "model.embed"
# The DSpark draft ships next to the weights and is never quantized.
DRAFT_WEIGHTS = "mtp.safetensors"
_BUNDLED_FILES = (DRAFT_WEIGHTS, "tiktoken")

_ATTENTION_RE = re.compile(r"(?:^|\.)attn\.(?:in_proj|q_proj|k_proj|v_proj|o_proj)$")
_RENAMES = {
    "embed.weight": "model.embed.weight",
    "embed_norm.weight": "model.embed_norm.weight",
    "norm.weight": "model.norm.weight",
    "unembed.weight": "lm_head.weight",
}


def is_inkling2(config: dict) -> bool:
    return str(config.get("model_type", "")).lower() == MODEL_TYPE


def text_config(config: dict) -> dict:
    nested = config.get("text_config")
    return nested if isinstance(nested, dict) else config


def n_routed_experts(config: dict) -> int:
    """Routed expert count, or 0 for any other model type."""
    if not is_inkling2(config):
        return 0
    return int(text_config(config).get("n_routed_experts") or 0)


def is_attention_projection(config: dict, path: str) -> bool:
    """A projection of the attention layers: under 1% of the weights."""
    return is_inkling2(config) and _ATTENTION_RE.search(path) is not None


def is_expert_carrier(module) -> bool:
    """The runtime's stacked expert projection, which owns its quantization."""
    return type(module).__name__ == "ExpertProj"


def cast_predicate(key: str) -> bool:
    """Keep the attention-residual queries, router and NVFP4 scales in FP32."""
    return not key.endswith((".query", "router_weight", "router_bias", ".global_scale"))


def sanitizer(config: dict):
    """oQ's sanitize for a source checkpoint, replayable by oQ's streaming plan."""
    text = text_config(config)
    hidden = int(text["hidden_size"])
    kernel = int(text.get("sconv_kernel_size") or text.get("conv_kernel_size") or 4)

    def sanitize(weights: dict) -> dict:
        out: dict = {}
        per_layer: dict[int, dict] = {}
        for key, value in weights.items():
            if not key.startswith(SOURCE_PREFIX):
                # The vision tower stays out of the text-only output.
                continue
            name = key[len(SOURCE_PREFIX) :]
            if name in _RENAMES:
                out[_RENAMES[name]] = value
            elif name == "attn_res.output_query":
                out["model.final_input.query"] = value.reshape(-1, hidden).astype(
                    mx.float32
                )
            elif name.startswith("attn_res.attn_queries."):
                # attn_queries[i] belongs to layer i + 1 (layer 0 has none).
                idx = int(name.rsplit(".", 1)[1]) + 1
                out[f"model.layers.{idx}.attn_input.query"] = value.reshape(
                    -1, hidden
                ).astype(mx.float32)
            elif name.startswith("attn_res.mlp_queries."):
                idx = int(name.rsplit(".", 1)[1])
                out[f"model.layers.{idx}.mlp_input.query"] = value.reshape(
                    -1, hidden
                ).astype(mx.float32)
            elif name.startswith("layers."):
                _, idx, rest = name.split(".", 2)
                per_layer.setdefault(int(idx), {})[rest] = value
            else:
                raise ValueError(f"Unmapped Inkling 2 tensor: {key}")

        for idx, d in per_layer.items():
            _map_layer(out, d, idx, hidden, kernel)
        return out

    sanitize._omlx_cast_predicate = cast_predicate
    return sanitize


def _map_layer(out: dict, d: dict, idx: int, hidden: int, kernel: int) -> None:
    pre = f"model.layers.{idx}."
    if "attn.k_sconv.static_w" not in d:
        out[pre + "attn.in_proj.weight"] = mx.concatenate(
            [
                d.pop("attn.wq_du.weight"),
                d.pop("attn.wk_dv.weight"),
                d.pop("attn.wv_dv.weight"),
                d.pop("attn.wr_du.weight"),
                d.pop("attn.gate_attn.weight"),
                d.pop("attn.attn_sink_proj.weight"),
            ],
            axis=0,
        )
        out[pre + "attn.k_conv"] = d.pop("attn.k_sconv.weight").reshape(-1, kernel)
        out[pre + "attn.v_conv"] = d.pop("attn.v_sconv.weight").reshape(-1, kernel)
        out[pre + "attn_norm.weight"] = d.pop("attn_norm.weight")
    else:
        out[pre + "attn.q_proj.weight"] = mx.concatenate(
            [
                d.pop("attn.wq_du.weight"),
                d.pop("attn.wr_du.weight"),
                d.pop("attn.gate_attn.weight"),
                d.pop("attn.attn_sink_proj.weight"),
            ],
            axis=0,
        )
        out[pre + "attn.k_proj.weight"] = mx.concatenate(
            [d.pop("attn.wk_dv.weight"), d.pop("attn.k_sconv.weight_proj")], axis=0
        )
        out[pre + "attn.v_proj.weight"] = mx.concatenate(
            [d.pop("attn.wv_dv.weight"), d.pop("attn.v_sconv.weight_proj")], axis=0
        )
        out[pre + "attn.k_static"] = d.pop("attn.k_sconv.static_w")
        out[pre + "attn.v_static"] = d.pop("attn.v_sconv.static_w")
        for n in ("q", "k", "v"):
            out[pre + f"{n}_input_norm.weight"] = d.pop(f"{n}_input_norm.weight")
    out[pre + "attn.o_proj.weight"] = d.pop("attn.wo_ud.weight")
    out[pre + "attn.q_scale"] = d.pop("attn.q_l2_clip.scale")
    out[pre + "attn.r_scale"] = d.pop("attn.rel_l2_clip.scale")
    out[pre + "attn.rel_proj"] = d.pop("attn.rel_logits_proj.proj")
    out[pre + "attn_conv"] = d.pop("attn_sconv.weight").reshape(hidden, kernel)
    out[pre + "mlp_conv"] = d.pop("mlp_sconv.weight").reshape(hidden, kernel)
    out[pre + "mlp_norm.weight"] = d.pop("mlp_norm.weight")
    out[pre + "mlp.router_weight"] = d.pop("mlp.gate.weight").astype(mx.float32)
    out[pre + "mlp.router_bias"] = d.pop("mlp.gate.bias").astype(mx.float32)
    for which in ("w13", "w2"):
        weight = d.pop(f"mlp.experts.{which}.weight", None)
        if weight is None:
            raise ValueError(f"Inkling 2 layer {idx} has no {which} expert weights")
        out[f"{pre}mlp.experts.{which}.weight"] = weight
        scale = d.pop(f"mlp.experts.{which}.global_scale", None)
        if scale is not None:
            out[f"{pre}mlp.experts.{which}.global_scale"] = scale
    if d:
        raise ValueError(f"Unmapped Inkling 2 tensors in layer {idx}: {sorted(d)}")


def is_draft_file(path) -> bool:
    return Path(path).name == DRAFT_WEIGHTS


def has_bundled_draft(source) -> bool:
    return (Path(source) / DRAFT_WEIGHTS).is_file()


def copy_bundled_files(source, output) -> None:
    """Copy the DSpark draft and the tiktoken table next to the output weights."""
    source, output = Path(source), Path(output)
    for name in _BUNDLED_FILES:
        path = source / name
        if path.is_dir():
            shutil.copytree(path, output / name, dirs_exist_ok=True)
        elif path.is_file():
            shutil.copy2(path, output / name)


def embed_fn(model):
    """Token ids to the first layer's input, as ``Inkling2TextModel`` computes it."""
    core = model.model
    return lambda tokens: core.embed_norm(core.embed(tokens))


def head_input(model, hidden, blocks: list):
    """The ``lm_head`` input for the last layer's output and all its blocks."""
    core = model.model
    normed = mx.fast.rms_norm(hidden.astype(mx.float32), None, core.args.rms_norm_eps)
    return core.norm(core.final_input(blocks, normed)[:, :, 0]).astype(hidden.dtype)
