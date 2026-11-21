# SPDX-License-Identifier: Apache-2.0
"""Map the published Inkling 2 checkpoint to the runtime layout.

Attention projections that share an input are stacked, the attention-residual
and loop message queries move onto their layers, and the expert codes are
viewed as MLX ``uint32`` carriers without re-rounding.
"""

from __future__ import annotations

import mlx.core as mx

_PREFIX = "model.llm."
_VISUAL_PREFIX = "model.visual."
_NVFP4_DENOMINATOR = 448.0 * 6.0
_LOOP_QUERIES = (
    "middle_cycle_cross_pass_queries",
    "middle_cycle_cross_pass_kv_queries",
)
# Runtime order of the loop message inputs (see ``Inkling2Layer``).
_LOOP_INPUTS = ("attn", "k", "v", "mlp")


def _vision_name(key: str) -> str:
    """``model.visual.{backbone,proj_in,proj_out}.*`` -> ``vision_tower.*``."""
    return "vision_tower." + key[len(_VISUAL_PREFIX) :]


def is_original_layout(weights: dict) -> bool:
    return any(k.startswith(_PREFIX) for k in weights)


def _expert_tensors(d: dict, which: str, out: dict, pre: str) -> str:
    """Map one routed-expert projection; returns its source format."""
    w = d.pop(f"mlp.experts.{which}_weight")
    s = d.pop(f"mlp.experts.{which}_weight_scale")
    base = f"{pre}mlp.experts.{which}"
    out[f"{base}.weight"] = w.view(mx.uint32)
    out[f"{base}.scales"] = s
    amax = d.pop(f"mlp.experts.{which}_weight_amax", None)
    if amax is None:
        return "mxfp8"
    denom = d.pop(f"mlp.experts.{which}_weight_outer_scale_denominator")
    # MLX's NVFP4 global scale is the amax over (6 * 448); rescale when the
    # checkpoint uses another denominator.
    out[f"{base}.global_scale"] = amax.astype(mx.float32) * (
        _NVFP4_DENOMINATOR / denom.astype(mx.float32)
    )
    return "nvfp4"


def _loop_controllers(d: dict, out: dict, pre: str) -> None:
    """``down`` [r, 2H] -> [2, H, r] halves; ``gain`` and ``forget`` [H, r]
    -> one [r, 2H] projection. Both are widened to FP32.

    Each part is cast and transposed before the final stack, because oQ's
    streaming plan replays only a stack or concatenate as the last step."""
    downs, gates = [], []
    for m in _LOOP_INPUTS:
        down = d.pop(f"controllers.{m}.down_weight").astype(mx.float32)
        rank = down.shape[0]
        downs.append(down.reshape(rank, 2, -1).transpose(1, 2, 0))
        gain = d.pop(f"controllers.{m}.gain_weight").astype(mx.float32)
        forget = d.pop(f"controllers.{m}.forget_weight").astype(mx.float32)
        gates.append(mx.concatenate([gain.T, forget.T], axis=1))
    out[pre + "loop_down"] = mx.stack(downs)
    out[pre + "loop_gain_forget"] = mx.stack(gates)


def _loop_message_queries(queries: dict, core: range, out: dict) -> None:
    """Row ``2 * i`` of each table serves core layer ``i``'s q (or k) input
    and row ``2 * i + 1`` its MLP (or v) input."""
    q = queries["middle_cycle_cross_pass_queries"]
    kv = queries["middle_cycle_cross_pass_kv_queries"]
    for i, idx in enumerate(core):
        rows = [q[2 * i], kv[2 * i], kv[2 * i + 1], q[2 * i + 1]]
        out[f"model.layers.{idx}.loop_message.query"] = mx.stack(
            [row.astype(mx.float32) for row in rows]
        )


def sanitize_weights(model, weights: dict) -> tuple[dict, dict]:
    """Returns the mapped weights and the routed-expert format per layer."""
    if not is_original_layout(weights):
        return weights, {}
    args = model.args.text
    has_vision = getattr(model, "vision_tower", None) is not None
    hidden = args.hidden_size
    k = args.sconv_kernel_size
    out: dict = {}
    formats: dict[int, str] = {}
    per_layer: dict[int, dict] = {}
    loop_queries: dict = {}
    for key, value in weights.items():
        if key.startswith(_VISUAL_PREFIX):
            if has_vision:
                out[_vision_name(key)] = value
            continue
        if not key.startswith(_PREFIX):
            continue
        name = key[len(_PREFIX) :]
        if name == "embed.weight":
            out["model.embed.weight"] = value
        elif name == "embed_norm.weight":
            out["model.embed_norm.weight"] = value
        elif name == "norm.weight":
            out["model.norm.weight"] = value
        elif name == "unembed.weight":
            out["lm_head.weight"] = value
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
        elif name == "attn_res.output_query":
            out["model.final_input.query"] = value.reshape(-1, hidden).astype(
                mx.float32
            )
        elif name in _LOOP_QUERIES:
            loop_queries[name] = value
        elif name.startswith("layers."):
            _, idx, rest = name.split(".", 2)
            per_layer.setdefault(int(idx), {})[rest] = value

    for idx, d in per_layer.items():
        pre = f"model.layers.{idx}."
        if args.is_sliding(idx):
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
            out[pre + "attn.k_conv"] = d.pop("attn.k_sconv.weight").reshape(-1, k)
            out[pre + "attn.v_conv"] = d.pop("attn.v_sconv.weight").reshape(-1, k)
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
                [d.pop("attn.wk_dv.weight"), d.pop("attn.k_sconv.weight_proj")],
                axis=0,
            )
            out[pre + "attn.v_proj.weight"] = mx.concatenate(
                [d.pop("attn.wv_dv.weight"), d.pop("attn.v_sconv.weight_proj")],
                axis=0,
            )
            out[pre + "attn.k_static"] = d.pop("attn.k_sconv.static_w")
            out[pre + "attn.v_static"] = d.pop("attn.v_sconv.static_w")
            for n in ("q", "k", "v"):
                out[pre + f"{n}_input_norm.weight"] = d.pop(f"{n}_input_norm.weight")
        out[pre + "attn.o_proj.weight"] = d.pop("attn.wo_ud.weight")
        out[pre + "attn.q_scale"] = d.pop("attn.q_l2_clip.scale")
        out[pre + "attn.r_scale"] = d.pop("attn.rel_l2_clip.scale")
        out[pre + "attn.rel_proj"] = d.pop("attn.rel_logits_proj.proj")
        out[pre + "attn_conv"] = d.pop("attn_sconv.weight").reshape(hidden, k)
        out[pre + "mlp_conv"] = d.pop("mlp_sconv.weight").reshape(hidden, k)
        out[pre + "mlp_norm.weight"] = d.pop("mlp_norm.weight")
        out[pre + "mlp.router_weight"] = d.pop("mlp.gate.weight").astype(mx.float32)
        out[pre + "mlp.router_bias"] = d.pop("mlp.gate.bias").astype(mx.float32)
        formats[idx] = _expert_tensors(d, "w13", out, pre)
        if _expert_tensors(d, "w2", out, pre) != formats[idx]:
            raise ValueError(f"Inkling 2 layer {idx} mixes expert formats")
        if "controllers.attn.down_weight" in d:
            _loop_controllers(d, out, pre)
        if d:
            raise ValueError(f"Unmapped Inkling 2 tensors in layer {idx}: {sorted(d)}")
    if loop_queries:
        _loop_message_queries(loop_queries, args.loop_core, out)
    return out, formats
