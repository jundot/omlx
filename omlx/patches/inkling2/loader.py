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
    if is_mlx_vlm_layout(weights):
        return sanitize_mlx_vlm_weights(model, weights), {}
    if is_mlx_lm_layout(weights):
        return sanitize_mlx_lm_weights(model, weights)
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


_VLM_TEXT = "language_model."
_VLM_LAYER = "language_model.model.layers."
_VLM_VISION = "vision_tower."
_VLM_VISION_RENAMES = (
    ("pre_layernorm.", "ln_pre."),
    (".layernorm_before.", ".ln_1."),
    (".layernorm_after.", ".ln_2."),
    (".lambda_1.", ".ls_1.gamma."),
    (".lambda_2.", ".ls_2.gamma."),
    (".mlp.fc1.", ".mlp.c_fc."),
    (".mlp.fc2.", ".mlp.c_proj."),
)
_VLM_SLIDING_IN_PROJ = (
    "q_proj",
    "k_proj",
    "v_proj",
    "r_proj",
    "gate_proj",
    "sink_proj",
)
_VLM_GLOBAL_Q_PROJ = ("q_proj", "r_proj", "gate_proj", "sink_proj")
_QUANT_SUFFIXES = ("weight", "scales", "biases")


def is_mlx_vlm_layout(weights: dict) -> bool:
    return any(k.startswith(_VLM_LAYER) for k in weights)


def is_mlx_vlm_quantization(quantization: dict) -> bool:
    """Whether ``quantization`` still names mlx-vlm's modules.

    Only mlx-vlm names layer modules ``self_attn`` and ``switch_mlp``; the
    ``language_model.``-prefixed variants of runtime paths that
    ``expand_per_layer_quant_keys`` adds do not, so a rewritten config is
    left alone.
    """
    return any(
        k.startswith(_VLM_LAYER) and (".self_attn." in k or ".mlp.switch_mlp." in k)
        for k, v in quantization.items()
        if isinstance(v, dict)
    )


def _vlm_vision_name(name: str) -> str:
    sub = name[len(_VLM_VISION) :] + "."
    for old, new in _VLM_VISION_RENAMES:
        sub = sub.replace(old, new)
    sub = sub[:-1]
    if not sub.startswith(("proj_in", "proj_out")):
        sub = "backbone." + sub
    return _VLM_VISION + sub


def _vlm_module_path(path: str, args) -> str | None:
    if path.startswith(_VLM_VISION):
        return _vlm_vision_name(path)
    if path == "language_model.lm_head":
        return "lm_head"
    if path == "language_model.model.embed_tokens":
        return "model.embed"
    if not path.startswith(_VLM_LAYER):
        return None
    idx, rest = path[len(_VLM_LAYER) :].split(".", 1)
    pre = f"model.layers.{idx}."
    if rest == "mlp.switch_mlp.down_proj":
        return pre + "mlp.experts.w2"
    if rest in ("mlp.switch_mlp.gate_proj", "mlp.switch_mlp.up_proj"):
        return pre + "mlp.experts.w13"
    if not rest.startswith("self_attn."):
        return None
    name = rest[len("self_attn.") :]
    if name == "o_proj":
        return pre + "attn.o_proj"
    if args.is_sliding(int(idx)):
        return pre + "attn.in_proj" if name in _VLM_SLIDING_IN_PROJ else None
    if name in _VLM_GLOBAL_Q_PROJ:
        return pre + "attn.q_proj"
    for n in ("k", "v"):
        if name in (f"{n}_proj", f"{n}_sconv.weight_proj"):
            return pre + f"attn.{n}_proj"
    return None


def mlx_vlm_quantization(quantization: dict, args) -> dict:
    """Rewrite mlx-vlm per-module quantization onto runtime module paths.

    A fused module takes the widest of its modules' affine specs when they
    share a group size; ``sanitize_mlx_vlm_weights`` re-encodes the narrower
    rows losslessly and quantizes the unquantized rows (the attention sink, the
    dynamic conv projections) to that spec. Other mixes raise.
    """
    out = {k: v for k, v in quantization.items() if not isinstance(v, dict)}
    for path, spec in quantization.items():
        if not isinstance(spec, dict):
            continue
        target = _vlm_module_path(path, args)
        if target is None:
            raise ValueError(f"Unmapped quantized Inkling 2 module: {path}")
        out[target] = _merge_spec(target, out.get(target), dict(spec))
    return out


def _merge_spec(target: str, old: dict | None, new: dict) -> dict:
    if old is None or old == new:
        return new
    if (
        old.get("mode", "affine") == new.get("mode", "affine") == "affine"
        and old["group_size"] == new["group_size"]
    ):
        return max(old, new, key=lambda spec: spec["bits"])
    raise ValueError(
        f"Inkling 2 {target} fuses modules with different quantization: "
        f"{old} and {new}"
    )


def _pack_codes(codes: mx.array, bits: int) -> mx.array:
    *lead, n = codes.shape
    q = codes.reshape(*lead, n // 32, 32)
    words = [mx.zeros(q.shape[:-1], dtype=mx.uint32) for _ in range(bits)]
    for j in range(32):
        word, shift = divmod(j * bits, 32)
        code = q[..., j].astype(mx.uint32)
        words[word] = words[word] | (code << shift)
        if shift + bits > 32:
            words[word + 1] = words[word + 1] | (code >> (32 - shift))
    return mx.stack(words, axis=-1).reshape(*lead, n * bits // 32)


def _widen(part: dict, spec: dict) -> dict:
    if "biases" not in part or spec.get("mode", "affine") != "affine":
        return part
    group_size, bits = spec["group_size"], spec["bits"]
    in_dims = part["scales"].shape[-1] * group_size
    have = part["weight"].shape[-1] * 32 // in_dims
    if have == bits:
        return part
    if have > bits:
        raise ValueError(f"Cannot narrow Inkling 2 {have}-bit rows to {bits} bits")
    scales = part["scales"]
    codes = mx.dequantize(
        part["weight"],
        mx.ones_like(scales),
        mx.zeros_like(scales),
        group_size=group_size,
        bits=have,
    ).astype(mx.uint8)
    weight = _pack_codes(codes, bits)
    mx.eval(weight)
    return {**part, "weight": weight}


def _pop_linear(d: dict, name: str) -> dict:
    return {s: d.pop(f"{name}.{s}") for s in _QUANT_SUFFIXES if f"{name}.{s}" in d}


def _fuse_rows(parts: list, spec, path: str, out: dict) -> None:
    if not any("scales" in p for p in parts):
        out[path + ".weight"] = mx.concatenate([p["weight"] for p in parts], axis=0)
        return
    if not isinstance(spec, dict):
        raise ValueError(f"Inkling 2 {path} is quantized but has no quantization spec")
    columns: dict = {}
    for part in parts:
        if "scales" not in part:
            q = mx.quantize(
                part["weight"],
                group_size=spec["group_size"],
                bits=spec["bits"],
                mode=spec.get("mode", "affine"),
            )
            part = dict(zip(_QUANT_SUFFIXES, q))
        else:
            part = _widen(part, spec)
        for suffix, value in part.items():
            columns.setdefault(suffix, []).append(value)
    for suffix, values in columns.items():
        out[f"{path}.{suffix}"] = mx.concatenate(values, axis=0)


def _vlm_experts(d: dict, out: dict, pre: str, spec) -> None:
    names = ("gate_proj", "up_proj", "down_proj")
    gate, up, down = (_pop_linear(d, "mlp.switch_mlp." + n) for n in names)
    if isinstance(spec, dict):
        gate, up = _widen(gate, spec), _widen(up, spec)
    scale2 = {n: d.pop(f"mlp.switch_mlp.{n}.weight_scale_2", None) for n in names}
    w13 = pre + "mlp.experts.w13."
    for suffix in gate:
        g, u = gate[suffix], up[suffix]
        out[w13 + suffix] = mx.stack([g, u], axis=-2).reshape(
            *g.shape[:-2], 2 * g.shape[-2], g.shape[-1]
        )
    if scale2["gate_proj"] is not None:
        if not mx.array_equal(scale2["gate_proj"], scale2["up_proj"]):
            raise ValueError(f"Inkling 2 {pre}gate and up outer scales differ")
        out[w13 + "global_scale"] = (
            scale2["gate_proj"].astype(mx.float32) * _NVFP4_DENOMINATOR
        )
    w2 = pre + "mlp.experts.w2."
    for suffix, value in down.items():
        out[w2 + suffix] = value
    if scale2["down_proj"] is not None:
        out[w2 + "global_scale"] = (
            scale2["down_proj"].astype(mx.float32) * _NVFP4_DENOMINATOR
        )


def _vlm_layer(d: dict, idx: int, args, quant: dict, out: dict) -> None:
    pre = f"model.layers.{idx}."
    a = "self_attn."
    hidden, k = args.hidden_size, args.sconv_kernel_size
    if args.is_sliding(idx):
        parts = [_pop_linear(d, a + n) for n in _VLM_SLIDING_IN_PROJ]
        _fuse_rows(parts, quant.get(pre + "attn.in_proj"), pre + "attn.in_proj", out)
        out[pre + "attn.k_conv"] = d.pop(a + "k_sconv.conv.weight").reshape(-1, k)
        out[pre + "attn.v_conv"] = d.pop(a + "v_sconv.conv.weight").reshape(-1, k)
        out[pre + "attn_norm.weight"] = d.pop("input_layernorm.weight")
    else:
        parts = [_pop_linear(d, a + n) for n in _VLM_GLOBAL_Q_PROJ]
        _fuse_rows(parts, quant.get(pre + "attn.q_proj"), pre + "attn.q_proj", out)
        for n in ("k", "v"):
            parts = [
                _pop_linear(d, a + f"{n}_proj"),
                _pop_linear(d, a + f"{n}_sconv.weight_proj"),
            ]
            path = pre + f"attn.{n}_proj"
            _fuse_rows(parts, quant.get(path), path, out)
            out[pre + f"attn.{n}_static"] = d.pop(a + f"{n}_sconv.static_weight")
        for n in ("q", "k", "v"):
            out[pre + f"{n}_input_norm.weight"] = d.pop(f"{n}_input_layernorm.weight")
    for suffix, value in _pop_linear(d, a + "o_proj").items():
        out[pre + "attn.o_proj." + suffix] = value
    out[pre + "attn.q_scale"] = d.pop(a + "q_scale")
    out[pre + "attn.r_scale"] = d.pop(a + "rel_scale")
    out[pre + "attn.rel_proj"] = d.pop(a + "rel_proj")
    out[pre + "attn_conv"] = d.pop("attn_sconv.conv.weight").reshape(hidden, k)
    out[pre + "mlp_conv"] = d.pop("mlp_sconv.conv.weight").reshape(hidden, k)
    out[pre + "mlp_norm.weight"] = d.pop("post_attention_layernorm.weight")
    out[pre + "mlp.router_weight"] = d.pop("mlp.gate_weight").astype(mx.float32)
    out[pre + "mlp.router_bias"] = d.pop("mlp.e_score_correction_bias").astype(
        mx.float32
    )
    for src, dst in (("attention_input", "attn_input"), ("mlp_input", "mlp_input")):
        if f"{src}.query" in d:
            out[pre + f"{dst}.query"] = (
                d.pop(f"{src}.query").reshape(-1, hidden).astype(mx.float32)
            )
    _vlm_experts(d, out, pre, quant.get(pre + "mlp.experts.w13"))
    if "controllers.attn.down_weight" in d:
        _loop_controllers(d, out, pre)
    if d:
        raise ValueError(f"Unmapped Inkling 2 tensors in layer {idx}: {sorted(d)}")


def sanitize_mlx_vlm_weights(model, weights: dict) -> dict:
    """Map an mlx-vlm-layout Inkling 2 checkpoint to the runtime layout."""
    args = model.args.text
    quant = model.args.quantization or {}
    has_vision = getattr(model, "vision_tower", None) is not None
    hidden = args.hidden_size
    out: dict = {}
    per_layer: dict[int, dict] = {}
    for key, value in weights.items():
        if key.startswith(_VLM_VISION):
            if has_vision:
                out[_vlm_vision_name(key)] = value
            continue
        if key.startswith(_VLM_LAYER):
            idx, rest = key[len(_VLM_LAYER) :].split(".", 1)
            per_layer.setdefault(int(idx), {})[rest] = value
            continue
        name = key.removeprefix(_VLM_TEXT)
        if name.startswith("lm_head.") or name in (
            "model.embed_norm.weight",
            "model.norm.weight",
        ):
            out[name] = value
        elif name.startswith("model.embed_tokens."):
            out["model.embed." + name.rsplit(".", 1)[1]] = value
        elif name == "model.final_input.query":
            out[name] = value.reshape(-1, hidden).astype(mx.float32)
        elif name == "model.message_queries":
            core = args.loop_core
            if value.shape[0] != len(core):
                raise ValueError(
                    f"Inkling 2 message_queries has {value.shape[0]} rows "
                    f"for {len(core)} loop layers"
                )
            for i, idx in enumerate(core):
                out[f"model.layers.{idx}.loop_message.query"] = value[i].astype(
                    mx.float32
                )
        else:
            raise ValueError(f"Unmapped Inkling 2 tensor: {key}")
    for idx, d in per_layer.items():
        _vlm_layer(d, idx, args, quant, out)
    return out


_LM_TEXT = "model.language_model."
_LM_LAYER = _LM_TEXT + "layers."
_LM_VISION = "model.vision_tower."
_LM_VISION_RENAMES = (
    ("merger.linear_1.", "proj_in."),
    ("merger.linear_2.", "proj_out."),
    ("embeddings.position_embedding.weight", "embeddings.position_embedding"),
)
_LM_LOOP_GATES = (("q", "attn"), ("k", "k"), ("v", "v"), ("mlp", "mlp"))
_LM_LOOP_PARTS = (
    ("down_proj", "down"),
    ("message_gate_proj", "gain"),
    ("forget_gate_proj", "forget"),
)


def is_mlx_lm_layout(weights: dict) -> bool:
    return any(k.startswith(_LM_LAYER) for k in weights)


def _lm_layer(d: dict, idx: int, args, spec: dict, out: dict) -> str:
    pre = f"model.layers.{idx}."
    a = "self_attn."
    hidden, k = args.hidden_size, args.sconv_kernel_size
    if args.is_sliding(idx):
        parts = [_pop_linear(d, a + n) for n in _VLM_SLIDING_IN_PROJ]
        _fuse_rows(parts, spec, pre + "attn.in_proj", out)
        for n in ("k", "v"):
            conv = d.pop(a + f"{n}_sconv.conv1d.weight")
            out[pre + f"attn.{n}_conv"] = conv.reshape(-1, k)
        out[pre + "attn_norm.weight"] = d.pop("input_layernorm.weight")
    else:
        parts = [_pop_linear(d, a + n) for n in _VLM_GLOBAL_Q_PROJ]
        _fuse_rows(parts, spec, pre + "attn.q_proj", out)
        for n in ("k", "v"):
            parts = [
                _pop_linear(d, a + f"{n}_proj"),
                _pop_linear(d, a + f"{n}_sconv.weight_proj"),
            ]
            _fuse_rows(parts, spec, pre + f"attn.{n}_proj", out)
            out[pre + f"attn.{n}_static"] = d.pop(a + f"{n}_sconv.static_weight")
        for n in ("q", "k", "v"):
            out[pre + f"{n}_input_norm.weight"] = d.pop(f"{n}_input_layernorm.weight")
    for suffix, value in _pop_linear(d, a + "o_proj").items():
        out[pre + "attn.o_proj." + suffix] = value
    out[pre + "attn.q_scale"] = d.pop(a + "q_l2_clip.scale")
    out[pre + "attn.r_scale"] = d.pop(a + "rel_l2_clip.scale")
    out[pre + "attn.rel_proj"] = d.pop(a + "rel_logits_proj.proj")
    attn_conv = d.pop("post_attention_convolution.conv1d.weight")
    out[pre + "attn_conv"] = attn_conv.reshape(hidden, k)
    mlp_conv = d.pop("post_mlp_convolution.conv1d.weight")
    out[pre + "mlp_conv"] = mlp_conv.reshape(hidden, k)
    out[pre + "mlp_norm.weight"] = d.pop("post_attention_layernorm.weight")
    out[pre + "mlp.router_weight"] = d.pop("mlp.gate.weight").astype(mx.float32)
    out[pre + "mlp.router_bias"] = d.pop("mlp.gate.e_score_correction_bias").astype(
        mx.float32
    )
    for src, dst in (("attention_input", "attn_input"), ("mlp_input", "mlp_input")):
        if f"{src}.query" in d:
            out[pre + f"{dst}.query"] = (
                d.pop(f"{src}.query").reshape(-1, hidden).astype(mx.float32)
            )
    for src, dst in (("gate_up_proj", "w13"), ("down_proj", "w2")):
        for name in [n for n in d if n.startswith(f"mlp.experts.{src}.")]:
            out[f"{pre}mlp.experts.{dst}.{name.rsplit('.', 1)[1]}"] = d.pop(name)
    controllers = {
        f"controllers.{dst}.{part}_weight": d.pop(f"{src}_message_gate.{name}.weight")
        for src, dst in _LM_LOOP_GATES
        for name, part in _LM_LOOP_PARTS
        if f"{src}_message_gate.{name}.weight" in d
    }
    if controllers:
        _loop_controllers(controllers, out, pre)
    if d:
        raise ValueError(f"Unmapped Inkling 2 tensors in layer {idx}: {sorted(d)}")
    return "nvfp4" if f"{pre}mlp.experts.w13.global_scale" in out else "mxfp8"


def sanitize_mlx_lm_weights(model, weights: dict) -> tuple[dict, dict]:
    """Map an mlx-lm-layout Inkling 2 checkpoint to the runtime layout."""
    args = model.args.text
    quant = model.args.quantization or {}
    spec = {n: quant[n] for n in ("group_size", "bits", "mode") if n in quant}
    has_vision = getattr(model, "vision_tower", None) is not None
    hidden = args.hidden_size
    out: dict = {}
    per_layer: dict[int, dict] = {}
    for key, value in weights.items():
        if key.startswith(_LM_VISION):
            if has_vision:
                name = _VLM_VISION + key[len(_LM_VISION) :]
                for old, new in _LM_VISION_RENAMES:
                    name = name.replace(old, new)
                out[_vlm_vision_name(name)] = value
            continue
        if key.startswith(_LM_LAYER):
            idx, rest = key[len(_LM_LAYER) :].split(".", 1)
            per_layer.setdefault(int(idx), {})[rest] = value
            continue
        name = key.removeprefix(_LM_TEXT)
        if key.startswith("lm_head."):
            out[key] = value
        elif name.startswith("embed_tokens."):
            out["model.embed." + name.rsplit(".", 1)[1]] = value
        elif name in ("embed_norm.weight", "norm.weight"):
            out["model." + name] = value
        elif name == "final_input.query":
            out["model.final_input.query"] = value.reshape(-1, hidden).astype(
                mx.float32
            )
        elif name == "message_input.query":
            for i, idx in enumerate(args.loop_core):
                out[f"model.layers.{idx}.loop_message.query"] = value[i].astype(
                    mx.float32
                )
        else:
            raise ValueError(f"Unmapped Inkling 2 tensor: {key}")
    formats = {idx: _lm_layer(d, idx, args, spec, out) for idx, d in per_layer.items()}
    return out, formats
