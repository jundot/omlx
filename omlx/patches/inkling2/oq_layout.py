# SPDX-License-Identifier: Apache-2.0
"""oQ's Inkling 2 layout: the source checkpoint mapped to the runtime layout.

``sanitizer`` uses only ops oQ's streaming plan can replay; ``oq_virtual``
exposes the routed experts before it runs. Calibration runs on the packed
source model (no proxy would be smaller), and the walk helpers carry the
completed attention-residual blocks between layers. A looped checkpoint walks
its layer visits in run order, as the runtime does.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import mlx.core as mx

from . import INKLING2_MODEL_TYPES
from .loader import _LOOP_QUERIES, _loop_controllers, _loop_message_queries
from .model import _unit_rms

SOURCE_PREFIX = "model.llm."
VISUAL_PREFIX = "model.visual."
# ModelOpt stores the NVFP4 tensor scale as amax / (E4M3 max x E2M1 max). MLX
# keeps the amax, and the runtime applies global_scale / NVFP4_GLOBAL_SCALE.
NVFP4_GLOBAL_SCALE = 448.0 * 6.0
# The only embedding table: a row lookup, so oQ never measures its inputs.
EMBED_PATH = "model.embed"
# The DSpark draft ships next to the weights and is never quantized.
DRAFT_WEIGHTS = "mtp.safetensors"

_ATTENTION_RE = re.compile(r"(?:^|\.)attn\.(?:in_proj|q_proj|k_proj|v_proj|o_proj)$")
_RENAMES = {
    "embed.weight": "model.embed.weight",
    "embed_norm.weight": "model.embed_norm.weight",
    "norm.weight": "model.norm.weight",
    "unembed.weight": "lm_head.weight",
}


def is_inkling2(config: dict) -> bool:
    return str(config.get("model_type", "")).lower() in INKLING2_MODEL_TYPES


def is_inkling2_model(model) -> bool:
    return str(getattr(model, "model_type", "")) in INKLING2_MODEL_TYPES


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
    """Keep the attention-residual and loop queries, loop controllers, router and
    NVFP4 scales in FP32."""
    return not key.endswith(
        (
            ".query",
            "loop_down",
            "loop_gain_forget",
            "router_weight",
            "router_bias",
            ".global_scale",
        )
    )


def sanitizer(config: dict, *, text_only: bool = False):
    """oQ's sanitize for a source checkpoint, replayable by oQ's streaming plan.

    The vision tower is renamed to the runtime's ``vision_tower.*`` and kept as
    is (oQ leaves vision tensors unquantized), unless the output is text only.
    """
    text = text_config(config)
    keep_vision = bool(config.get("vision_config")) and not text_only
    hidden = int(text["hidden_size"])
    kernel = int(text.get("sconv_kernel_size") or text.get("conv_kernel_size") or 4)
    loop = text.get("loop_config")
    loop_core = range(loop["core_start"], loop["core_end"]) if loop else range(0)

    def sanitize(weights: dict) -> dict:
        out: dict = {}
        per_layer: dict[int, dict] = {}
        loop_queries: dict = {}
        for key, value in weights.items():
            if key.startswith(VISUAL_PREFIX):
                if keep_vision:
                    out["vision_tower." + key[len(VISUAL_PREFIX) :]] = value
                continue
            if not key.startswith(SOURCE_PREFIX):
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
            elif name in _LOOP_QUERIES:
                loop_queries[name] = value
            elif name.startswith("layers."):
                _, idx, rest = name.split(".", 2)
                per_layer.setdefault(int(idx), {})[rest] = value
            else:
                raise ValueError(f"Unmapped Inkling 2 tensor: {key}")

        for idx, d in per_layer.items():
            _map_layer(out, d, idx, hidden, kernel)
        if loop_queries:
            _loop_message_queries(loop_queries, loop_core, out)
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
    if "controllers.attn.down_weight" in d:
        _loop_controllers(d, out, pre)
    if d:
        raise ValueError(f"Unmapped Inkling 2 tensors in layer {idx}: {sorted(d)}")


def is_draft_file(path) -> bool:
    return Path(path).name == DRAFT_WEIGHTS


def has_bundled_draft(source) -> bool:
    return (Path(source) / DRAFT_WEIGHTS).is_file()


def copy_bundled_draft(source, output) -> None:
    """Copy the DSpark draft next to the output weights."""
    if has_bundled_draft(source):
        shutil.copy2(Path(source) / DRAFT_WEIGHTS, Path(output) / DRAFT_WEIGHTS)


def embed_fn(model):
    """Token ids to the first layer's input, as ``Inkling2TextModel`` computes it."""
    core = model.model
    return lambda tokens: core.embed_norm(core.embed(tokens))


def walk_layers(model):
    """The decoder layers in run order: a looped layer once per pass."""
    args = model.model.args
    if not args.loop_config:
        return model.layers
    return [model.layers[idx] for idx, _ in args.layer_schedule()]


def walk_state(model) -> dict:
    """The completed blocks, plus the loop bookkeeping of a looped model."""
    args = model.model.args
    if not args.loop_config:
        return {"blocks": []}
    return {
        "blocks": [],
        "schedule": tuple(args.layer_schedule()),
        "core_start": args.loop_core.start,
        "prelude": None,
        "bank": None,
    }


def walk_step(layer, state: dict, step: int, h):
    """Run walk step ``step`` as ``Inkling2TextModel`` runs that layer visit.

    Returns the output and the state after the step. ``state`` stays as is,
    because the sensitivity walk runs each step twice on the same inputs.
    """
    blocks = list(state["blocks"])
    schedule = state.get("schedule")
    if schedule is None:
        return layer(h, blocks, cache=None), {**state, "blocks": blocks}
    idx, loop_pass = schedule[step]
    prelude, bank = state["prelude"], state["bank"]
    if idx == state["core_start"] and loop_pass == 0:
        prelude = (h, list(blocks))
    elif idx == state["core_start"]:
        # The next pass restarts from the prelude output; the blocks the last
        # pass wrote become its message bank.
        bank = (blocks[len(prelude[1]) + 1 :], _unit_rms(h.astype(mx.float32)))
        h, blocks = prelude[0], list(prelude[1])
    out = layer(h, blocks, cache=None, bank=bank if loop_pass else None)
    return out, {**state, "blocks": blocks, "prelude": prelude, "bank": bank}


def walk_layer_index(state: dict, step: int) -> int:
    """The decoder layer that walk step ``step`` runs."""
    schedule = state.get("schedule")
    return step if schedule is None else schedule[step][0]


def head_input(model, hidden, blocks: list):
    """The ``lm_head`` input for the last layer's output and all its blocks."""
    core = model.model
    normed = mx.fast.rms_norm(hidden.astype(mx.float32), None, core.args.rms_norm_eps)
    return core.norm(core.final_input(blocks, normed)[:, :, 0]).astype(hidden.dtype)
