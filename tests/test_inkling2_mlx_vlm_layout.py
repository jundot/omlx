# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 checkpoints in mlx-vlm's layout load like the published layout."""

from __future__ import annotations

import copy
import json
import shutil

import mlx.core as mx
import mlx.nn as nn
import mlx_lm.utils as mlx_lm_utils
import pytest
from mlx.utils import tree_flatten

from omlx.patches.inkling2.model import TextArgs
from omlx.utils import model_loading
from tests.test_oq import _INK2_LAYERS, _inkling2_model, _inkling2_source

_NVFP4 = {"group_size": 16, "bits": 4, "mode": "nvfp4"}
_MXFP8 = {"group_size": 32, "bits": 8, "mode": "mxfp8"}
_Q8 = {"group_size": 64, "bits": 8, "mode": "affine"}
_ATTN_PROJ = {
    "wq_du": "q_proj",
    "wk_dv": "k_proj",
    "wv_dv": "v_proj",
    "wr_du": "r_proj",
    "wo_ud": "o_proj",
    "gate_attn": "gate_proj",
    "attn_sink_proj": "sink_proj",
}
_NORMS = {
    "attn_norm": "input_layernorm",
    "mlp_norm": "post_attention_layernorm",
    "q_input_norm": "q_input_layernorm",
    "k_input_norm": "k_input_layernorm",
    "v_input_norm": "v_input_layernorm",
}
_VISION = (
    ("ln_pre.", "pre_layernorm."),
    (".ln_1.", ".layernorm_before."),
    (".ln_2.", ".layernorm_after."),
    (".ls_1.gamma", ".lambda_1"),
    (".ls_2.gamma", ".lambda_2"),
    (".mlp.c_fc.", ".mlp.fc1."),
    (".mlp.c_proj.", ".mlp.fc2."),
)


def _split_gate_up(v):
    *lead, rows, cols = v.shape
    w = v.reshape(*lead, rows // 2, 2, cols)
    return mx.contiguous(w[..., 0, :]), mx.contiguous(w[..., 1, :])


def _to_mlx_conv(v):
    return v if v.ndim == 3 and v.shape[-1] == 1 else v.transpose(0, 2, 1)


def _mlx_vlm_tensors(source: dict, hidden: int) -> tuple[dict, dict]:
    """The tensors and expert specs mlx-vlm's ``Model.sanitize`` produces.

    Mirrors mlx_vlm/models/inkling2/inkling2.py at Blaizzy/mlx-vlm d998a62,
    the commit community MLX builds (orcarouter) were made with.
    """
    out, experts, specs = {}, {}, {}
    for key, value in source.items():
        if key.startswith("model.llm.layers."):
            layer, sub = key[len("model.llm.layers.") :].split(".", 1)
            base = f"language_model.model.layers.{layer}."
            if sub.startswith("mlp.experts."):
                experts.setdefault(layer, {})[sub[len("mlp.experts.") :]] = value
                continue
            if sub.startswith("attn."):
                name, _, leaf = sub[len("attn.") :].partition(".")
                sub = _ATTN_PROJ.get(name, name) + ("." + leaf if leaf else "")
                sub = {
                    "q_l2_clip.scale": "q_scale",
                    "rel_l2_clip.scale": "rel_scale",
                    "rel_logits_proj.proj": "rel_proj",
                }.get(sub, sub)
                for conv in ("k_sconv", "v_sconv"):
                    if sub == conv + ".weight":
                        sub, value = conv + ".conv.weight", _to_mlx_conv(value)
                    elif sub == conv + ".static_w":
                        sub = conv + ".static_weight"
                    elif sub == conv + ".weight_proj":
                        sub += ".weight"
                sub = "self_attn." + sub
            elif sub in ("attn_sconv.weight", "mlp_sconv.weight"):
                sub, value = sub.replace(".weight", ".conv.weight"), _to_mlx_conv(value)
            elif sub in ("mlp.gate.weight", "mlp.gate.bias"):
                sub = (
                    "mlp.gate_weight"
                    if sub.endswith("weight")
                    else "mlp.e_score_correction_bias"
                )
            else:
                name, _, leaf = sub.partition(".")
                sub = _NORMS.get(name, name) + "." + leaf
            out[base + sub] = value
        elif key.startswith("model.llm.attn_res."):
            sub = key[len("model.llm.attn_res.") :]
            if sub == "output_query":
                target = "final_input.query"
            else:
                kind, index = sub.split(".")
                layer = int(index) + (kind == "attn_queries")
                name = "attention_input" if kind == "attn_queries" else "mlp_input"
                target = f"layers.{layer}.{name}.query"
            out["language_model.model." + target] = value
        elif key.startswith("model.llm."):
            sub = key[len("model.llm.") :]
            out[
                (
                    "language_model.lm_head.weight"
                    if sub == "unembed.weight"
                    else "language_model.model."
                    + sub.replace("embed.weight", "embed_tokens.weight")
                )
            ] = value
        elif key.startswith("model.visual."):
            sub = key[len("model.visual.") :].removeprefix("backbone.")
            for old, new in _VISION:
                sub = sub.replace(old, new)
            out["vision_tower." + sub] = value
    for layer, tensors in experts.items():
        base = f"language_model.model.layers.{layer}.mlp.switch_mlp."
        for name, projections in (
            ("w13_weight", ("gate_proj", "up_proj")),
            ("w2_weight", ("down_proj",)),
        ):
            values = {
                "weight": tensors[name].view(mx.uint32),
                "scales": tensors[name + "_scale"],
            }
            if name + "_amax" in tensors:
                values["weight_scale_2"] = (
                    tensors[name + "_amax"] / tensors[name + "_outer_scale_denominator"]
                )
            for suffix, v in values.items():
                parts = (
                    _split_gate_up(v)
                    if len(projections) == 2 and suffix != "weight_scale_2"
                    else (v,) * len(projections)
                )
                for projection, part in zip(projections, parts):
                    out[base + projection + "." + suffix] = part
            for projection in projections:
                specs[base + projection] = (
                    _NVFP4 if "weight_scale_2" in values else _MXFP8
                )
    base = "language_model.model."
    if base + "middle_cycle_cross_pass_queries" in out:
        attn_mlp = out.pop(base + "middle_cycle_cross_pass_queries").reshape(
            -1, 2, hidden
        )
        kv = out.pop(base + "middle_cycle_cross_pass_kv_queries").reshape(-1, 2, hidden)
        out[base + "message_queries"] = mx.concatenate(
            [attn_mlp[:, :1], kv, attn_mlp[:, 1:]], axis=1
        )
    return out, specs


def _mlx_vlm_checkpoint(
    source_dir, dest, *, quantize_attention=False, expert_bits=None
):
    """Write ``source_dir`` the way an mlx-vlm quantized build ships it.

    ``expert_bits`` maps gate/up/down to affine bits, replacing the source's
    NVFP4/MXFP8 experts the way mixed-precision builds quantize them.
    """
    shutil.copytree(
        source_dir,
        dest,
        ignore=shutil.ignore_patterns("*.safetensors", "hf_quant_config.json"),
    )
    shutil.copy(source_dir / "mtp.safetensors", dest / "mtp.safetensors")
    config = json.loads((source_dir / "config.json").read_text())
    tensors, specs = _mlx_vlm_tensors(
        mx.load(str(source_dir / "model.safetensors")),
        config["text_config"]["hidden_size"],
    )
    quant = dict(_NVFP4, **specs)
    if quantize_attention:
        for layer in range(_INK2_LAYERS):
            for proj in ("q_proj", "k_proj", "v_proj", "r_proj", "gate_proj", "o_proj"):
                path = f"language_model.model.layers.{layer}.self_attn.{proj}"
                w, s, b = mx.quantize(
                    tensors.pop(path + ".weight"), group_size=64, bits=8
                )
                tensors.update(
                    {path + ".weight": w, path + ".scales": s, path + ".biases": b}
                )
                quant[path] = dict(_Q8)
    if expert_bits:
        rng = mx.random.key(0)
        for layer in range(_INK2_LAYERS):
            base = f"language_model.model.layers.{layer}.mlp.switch_mlp."
            for proj, bits in expert_bits.items():
                path = base + f"{proj}_proj"
                text = config["text_config"]
                in_dims = text["intermediate_size" if proj == "down" else "hidden_size"]
                shape = tensors[path + ".weight"].shape[:-1] + (in_dims,)
                rng, key = mx.random.split(rng)
                w = (mx.random.normal(shape, key=key) * 0.05).astype(mx.bfloat16)
                q, sc, b = mx.quantize(w, group_size=32, bits=bits)
                for suffix in ("weight", "scales", "weight_scale_2"):
                    tensors.pop(f"{path}.{suffix}", None)
                tensors.update(
                    {path + ".weight": q, path + ".scales": sc, path + ".biases": b}
                )
                quant[path] = {"group_size": 32, "bits": bits, "mode": "affine"}
    config["quantization"] = config["quantization_config"] = quant
    (dest / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(dest / "model.safetensors"), tensors, {"format": "mlx"})
    return dest


@pytest.fixture
def patched_load_config(monkeypatch):
    monkeypatch.setattr(model_loading, "_MLX_LM_LOAD_CONFIG_PATCHED", False)
    monkeypatch.setattr(mlx_lm_utils, "load_config", mlx_lm_utils.load_config)
    model_loading._patch_mlx_lm_load_config()


@pytest.mark.parametrize("looped", [False, True], ids=["plain", "looped"])
def test_mlx_vlm_layout_loads_like_published_layout(
    tmp_path, looped, patched_load_config
):
    source = tmp_path / "source"
    _inkling2_source(source, looped=looped)
    vlm = _mlx_vlm_checkpoint(source, tmp_path / "mlx_vlm")

    expected = dict(tree_flatten(_inkling2_model(source).parameters()))
    loaded = dict(tree_flatten(_inkling2_model(vlm).parameters()))

    assert loaded.keys() == expected.keys()
    for key, value in expected.items():
        assert loaded[key].dtype == value.dtype, key
        if key.endswith(".global_scale"):
            # mlx-vlm stores amax / denominator; the runtime keeps amax.
            assert mx.allclose(loaded[key], value, rtol=1e-6).item(), key
        else:
            assert mx.array_equal(loaded[key], value, equal_nan=True).item(), key


def test_quantized_attention_keeps_its_codes(tmp_path, patched_load_config):
    source = tmp_path / "source"
    _inkling2_source(source, looped=True)
    vlm = _mlx_vlm_checkpoint(source, tmp_path / "mlx_vlm", quantize_attention=True)

    expected = _inkling2_model(source)
    loaded = _inkling2_model(vlm)

    # Affine codes are local to a row, so the fused projections, including the
    # unquantized sink and dynamic-conv rows, match quantizing the fused BF16
    # weight in one go.
    for layer, (ref, got) in enumerate(zip(expected.model.layers, loaded.model.layers)):
        names = ["o_proj"] + (
            ["in_proj"] if ref.attn.sliding else ["q_proj", "k_proj", "v_proj"]
        )
        for name in names:
            module = getattr(got.attn, name)
            assert isinstance(module, nn.QuantizedLinear), f"layer {layer} {name}"
            w, s, b = mx.quantize(getattr(ref.attn, name).weight, group_size=64, bits=8)
            for want, have in (
                (w, module.weight),
                (s, module.scales),
                (b, module.biases),
            ):
                assert mx.array_equal(want, have).item(), f"layer {layer} {name}"


@pytest.mark.parametrize("bits", [(2, 3, 4), (4, 2, 8), (3, 6, 3)], ids=str)
def test_mixed_bit_experts_widen_losslessly(tmp_path, bits, patched_load_config):
    # Mixed-precision builds can quantize gate and up at different bits; the
    # runtime fuses them into w13, so the narrower rows are widened.
    source = tmp_path / "source"
    _inkling2_source(source, looped=True)
    gate, up, down = bits
    vlm = _mlx_vlm_checkpoint(
        source,
        tmp_path / "mlx_vlm",
        expert_bits={"gate": gate, "up": up, "down": down},
    )
    tensors = mx.load(str(vlm / "model.safetensors"))
    loaded = _inkling2_model(vlm)

    def dequant(path, b):
        return mx.dequantize(
            tensors[path + ".weight"],
            tensors[path + ".scales"],
            tensors[path + ".biases"],
            group_size=32,
            bits=b,
        )

    for layer, block in enumerate(loaded.model.layers):
        experts = block.mlp.experts
        base = f"language_model.model.layers.{layer}.mlp.switch_mlp."
        g, u = dequant(base + "gate_proj", gate), dequant(base + "up_proj", up)
        w13 = mx.stack([g, u], axis=-2).reshape(*g.shape[:-2], -1, g.shape[-1])
        assert experts.w13.bits == max(gate, up)
        for proj, want in (
            (experts.w13, w13),
            (experts.w2, dequant(base + "down_proj", down)),
        ):
            got = mx.dequantize(
                proj.weight, proj.scales, proj.biases, group_size=32, bits=proj.bits
            )
            assert mx.array_equal(got, want).item(), f"layer {layer}"


def test_fused_specs_take_the_widest_affine_bits():
    from omlx.patches.inkling2.loader import mlx_vlm_quantization

    args = TextArgs.from_dict(
        {"num_hidden_layers": 2, "local_layer_ids": [0], "hidden_size": 64}
    )
    base = "language_model.model.layers.0.self_attn."
    quant = mlx_vlm_quantization(
        {base + "q_proj": dict(_Q8, bits=4), base + "k_proj": _Q8}, args
    )
    assert quant["model.layers.0.attn.in_proj"] == _Q8
    # Rows with another group size or mode cannot be re-encoded losslessly.
    for other in (dict(_Q8, group_size=32), _NVFP4):
        with pytest.raises(ValueError, match="different quantization"):
            mlx_vlm_quantization({base + "q_proj": _Q8, base + "k_proj": other}, args)


def test_rewritten_quantization_is_left_alone(tmp_path):
    # load_config may run the normalizers again on a rewritten config, which by
    # then also carries the language_model.-prefixed variants of its keys.
    source = tmp_path / "source"
    _inkling2_source(source, looped=True)
    vlm = _mlx_vlm_checkpoint(source, tmp_path / "mlx_vlm", quantize_attention=True)
    cfg = json.loads((vlm / "config.json").read_text())

    model_loading.normalize_inkling2_mlx_vlm_quant(cfg)
    model_loading.expand_per_layer_quant_keys(cfg)
    once = copy.deepcopy(cfg["quantization"])
    assert "model.layers.0.attn.in_proj" in once
    model_loading.normalize_inkling2_mlx_vlm_quant(cfg)

    assert cfg["quantization"] == once
