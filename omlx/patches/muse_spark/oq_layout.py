# SPDX-License-Identifier: Apache-2.0
"""oQ's Muse Spark layout and its mapping to mlx-lm's Muse Spark model.

oQ quantizes Muse Spark tensors under HF-style names: q, k, v and the
attention output gate separate, the experts at ``mlp.switch_mlp``, text weights
under ``language_model.model.``. The residual gates and norms keep the values of
the source: the gates and zero-centered norms of an HF checkpoint, or the gate
coefficients and gains of an mlx-lm one. mlx-lm's model, which serves the
output, arranges the same weights for its fused kernels:

* ``self_attn.qkvg_proj``: q, k, v and the attention output gate as one
  projection. The q and k rows of each head are interleaved for traditional
  RoPE: pairs (2i, 2i + 1) instead of (i, i + head_dim / 2).
* ``mlp.experts``: the routed experts and ``post_expert_norm``.
* ``add_attn_alpha`` and the like: the fixed residual gate coefficients, in
  place of the gates.
* The zero-centered norms as their gains, ``T(1 + weight)``.
* ``attn.qkv_proj`` in the vision tower, with interleaved q and k rows.
* Text weights under ``model.language_model.``, vision weights under ``model.``.

``sanitize`` maps a source checkpoint to oQ's layout, ``UpstreamWriter`` writes
oQ output in mlx-lm's layout, and the calibration helpers run oQ's layers and
name their statistics through mlx-lm's decoder block.
"""

from __future__ import annotations

import re

import mlx.core as mx
from mlx_lm.models import muse_spark

RESIDUAL_GATE_EPS = 1e-3
# ModelOpt stores the NVFP4 tensor scale as a multiplier. mlx-lm keeps the
# absolute maximum it came from: amax = scale * 6 * 448 (E2M1 max x E4M3 max).
NVFP4_GLOBAL_SCALE = 6.0 * 448.0
# oQ's text layers. Calibration statistics are named under this prefix too.
LAYER_PREFIX = "language_model.model.layers."
_FP32_PARAMS = ("mlp.gate.", "residual_gate.", "global_scale", "_alpha", "_beta")
_ATTENTION_PARTS = ("q_proj", "k_proj", "v_proj", "gate_proj")

# The checkpoint stores these norm weights centered at zero.
CENTERED_NORMS = (
    "input_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
    "mlp.pre_expert_norm",
    "mlp.post_expert_norm",
)
# This package's residual gates and the names of their upstream coefficients.
RESIDUAL_GATES = {
    "post_attention_residual_gate": "add_attn",
    "post_feedforward_residual_gate": "add_ffn",
}

_TEXT_LAYER = r"^model\.language_model\.layers\.\d+\."
_VISION_LAYER = r"^model\.vision_tower\.layers\.\d+\."
# Projections that upstream fuses, as (prefix, part, param) of a tensor key.
_TEXT_PART_RE = re.compile(rf"({_TEXT_LAYER}self_attn\.)(q|k|v|gate)_proj\.(\w+)$")
_VISION_PART_RE = re.compile(rf"({_VISION_LAYER}attn\.)(q|k|v)_proj\.(\w+)$")
_TEXT_PARTS = ("q", "k", "v", "gate")
_VISION_PARTS = ("q", "k", "v")
# Module renames, applied to rebased tensor keys and module paths.
_RENAMES = (
    (re.compile(rf"({_TEXT_LAYER}self_attn\.)(?:q|k|v|gate)_proj\b"), r"\1qkvg_proj"),
    (re.compile(rf"({_TEXT_LAYER}mlp\.)switch_mlp\b"), r"\1experts"),
    (re.compile(rf"({_TEXT_LAYER}mlp\.)(post_expert_norm)\b"), r"\1experts.\2"),
    (re.compile(rf"({_VISION_LAYER}attn\.)[qkv]_proj\b"), r"\1qkv_proj"),
)
_CENTERED_NORM_RE = re.compile(
    rf"{_TEXT_LAYER}(?:{'|'.join(re.escape(n) for n in CENTERED_NORMS)})\.weight$"
)
_GATE_RE = re.compile(rf"({_TEXT_LAYER})({'|'.join(RESIDUAL_GATES)})\.gate$")
_VISION_PREFIXES = ("vision_tower.", "vision_adapter.", "vision_projection.")


@mx.compile
def gate_coefficients(gate, temperature, eps):
    """FP32 ``(alpha, beta)`` of a residual gate: ``alpha * x + beta * branch``."""
    w = gate.astype(mx.float32) / temperature
    beta = mx.sigmoid(w)
    alpha = mx.sqrt(mx.maximum(mx.sigmoid(-w) * (1.0 + beta), eps))
    return alpha, beta


def is_upstream_layout(weights) -> bool:
    """Only the upstream layout stores the residual gate coefficients."""
    return any(key.endswith("_alpha") for key in weights)


def interleave_rope_rows(w: mx.array, head_dim: int) -> mx.array:
    """Move rows (i, i + head_dim / 2) of each head to rows (2i, 2i + 1)."""
    shape = w.shape
    w = w.reshape(-1, 2, head_dim // 2, *shape[1:])
    return mx.swapaxes(w, 1, 2).reshape(shape)


def deinterleave_rope_rows(w: mx.array, head_dim: int) -> mx.array:
    """Undo ``interleave_rope_rows``."""
    shape = w.shape
    w = w.reshape(-1, head_dim // 2, 2, *shape[1:])
    return mx.swapaxes(w, 1, 2).reshape(shape)


def read_upstream_text(
    weights: dict,
    layers_prefix: str,
    *,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> dict:
    """Map the upstream decoder layers under ``layers_prefix`` to oQ's names.

    The residual gate coefficients and the norm gains keep their upstream names
    and values.
    """
    q_rows = n_heads * head_dim
    kv_rows = n_kv_heads * head_dim
    out = {}
    for key, value in weights.items():
        if not key.startswith(layers_prefix):
            out[key] = value
            continue
        index, _, name = key[len(layers_prefix) :].partition(".")
        layer = f"{layers_prefix}{index}."
        if name.startswith("self_attn.qkvg_proj."):
            param = name[len("self_attn.qkvg_proj.") :]
            # oQ's sanitize plan replays slices, but not later ops on split parts.
            q = value[:q_rows]
            k = value[q_rows : q_rows + kv_rows]
            v = value[q_rows + kv_rows : q_rows + 2 * kv_rows]
            g = value[q_rows + 2 * kv_rows :]
            out[f"{layer}self_attn.q_proj.{param}"] = deinterleave_rope_rows(
                q, head_dim
            )
            out[f"{layer}self_attn.k_proj.{param}"] = deinterleave_rope_rows(
                k, head_dim
            )
            out[f"{layer}self_attn.v_proj.{param}"] = v
            out[f"{layer}self_attn.gate_proj.{param}"] = g
        elif name.startswith("mlp.experts.post_expert_norm."):
            out[layer + "mlp." + name[len("mlp.experts.") :]] = value
        elif name.startswith("mlp.experts."):
            out[layer + "mlp.switch_mlp." + name[len("mlp.experts.") :]] = value
        else:
            out[key] = value
    return out


def read_upstream_vision(weights: dict, layers_prefix: str, head_dim: int) -> dict:
    """Split the fused vision ``attn.qkv_proj`` under ``layers_prefix``."""
    out = {}
    for key, value in weights.items():
        if not (key.startswith(layers_prefix) and ".attn.qkv_proj." in key):
            out[key] = value
            continue
        attn, param = key.split(".attn.qkv_proj.")
        rows = value.shape[0] // 3
        q, k, v = value[:rows], value[rows : 2 * rows], value[2 * rows :]
        out[f"{attn}.attn.q_proj.{param}"] = deinterleave_rope_rows(q, head_dim)
        out[f"{attn}.attn.k_proj.{param}"] = deinterleave_rope_rows(k, head_dim)
        out[f"{attn}.attn.v_proj.{param}"] = v
    return out


def _rebase(key: str) -> str:
    """Move an oQ key (mlx-vlm or mlx-lm tree) under the upstream prefixes."""
    if key.startswith("language_model.model."):
        return "model.language_model." + key[len("language_model.model.") :]
    if key.startswith("language_model."):
        return key[len("language_model.") :]
    if key.startswith("model."):
        return "model.language_model." + key[len("model.") :]
    if key.startswith(_VISION_PREFIXES):
        return "model." + key
    return key


def _rename(key: str) -> str:
    for pattern, replacement in _RENAMES:
        key = pattern.sub(replacement, key)
    return key


def upstream_key(key: str) -> str:
    """The upstream name of an oQ tensor key or module path."""
    return _rename(_rebase(key))


class UpstreamWriter:
    """Writes oQ output in the upstream layout.

    Call it on each batch of finished tensors before saving them. It returns
    the tensors to save and keeps the parts of a fused projection until all of
    them have arrived. ``finish`` checks that none is left. Without
    ``centered_norms`` the norms are gains already, as in an mlx-lm source.
    """

    def __init__(self, config: dict, *, centered_norms: bool = True):
        text = config.get("text_config") or config
        self.head_dim = int(text["head_dim"])
        self.temperature = float(text.get("residual_gate_temperature", 0.3))
        vision = config.get("vision_config") or {}
        self.vision_head_dim = (
            int(vision["hidden_size"]) // int(vision["num_attention_heads"])
            if vision
            else 0
        )
        self.centered_norms = centered_norms
        self._parts: dict[str, dict[str, mx.array]] = {}
        self._global_scales: set[str] = set()

    def __call__(self, tensors: dict) -> dict:
        out = {}
        for key, value in tensors.items():
            key = _rebase(key)
            if match := _GATE_RE.match(key):
                layer, gate = match.groups()
                name = layer + RESIDUAL_GATES[gate]
                alpha, beta = gate_coefficients(
                    value, self.temperature, RESIDUAL_GATE_EPS
                )
                out[f"{name}_alpha"], out[f"{name}_beta"] = alpha, beta
                continue
            if self.centered_norms and _CENTERED_NORM_RE.match(key):
                # The runtime norms compute the same T(1 + weight).
                value = (1.0 + value.astype(mx.float32)).astype(value.dtype)
            if match := _TEXT_PART_RE.match(key):
                self._add_part(out, match, value, "qkvg_proj", _TEXT_PARTS)
                continue
            if match := _VISION_PART_RE.match(key):
                self._add_part(out, match, value, "qkv_proj", _VISION_PARTS)
                continue
            name = _rename(key)
            if name.endswith(".global_scale"):
                self._global_scales.add(name[: -len(".global_scale")])
            out[name] = value
        return out

    def _add_part(self, out, match, value, fused, parts):
        prefix, part, param = match.groups()
        name = f"{prefix}{fused}.{param}"
        group = self._parts.setdefault(name, {})
        group[part] = value
        if len(group) < len(parts):
            return
        del self._parts[name]
        head_dim = self.head_dim if fused == "qkvg_proj" else self.vision_head_dim
        rows = [
            interleave_rope_rows(group[p], head_dim) if p in ("q", "k") else group[p]
            for p in parts
        ]
        out[name] = mx.concatenate(rows)

    def finish(self):
        if self._parts:
            missing = ", ".join(sorted(self._parts)[:4])
            raise ValueError(f"Fused projections without all their parts: {missing}")

    def quantization(self, per_layer: dict) -> dict:
        """Per-module quantization keyed by the upstream module paths."""
        out = {}
        for path, spec in per_layer.items():
            name = upstream_key(path)
            if name in self._global_scales and isinstance(spec, dict):
                # mlx-lm adds no tensor scale to a module it finds in the map.
                spec = {**spec, "global_scale": True}
            if out.setdefault(name, spec) != spec:
                raise ValueError(
                    f"The parts of {name} need one quantization: "
                    f"{out[name]} and {spec}"
                )
        return out


def is_modelopt_nvfp4(config: dict) -> bool:
    """Whether ``config`` is a ModelOpt NVFP4 export."""
    quant = config.get("quantization_config") or {}
    return (
        quant.get("quant_method") == "modelopt" and quant.get("quant_algo") == "NVFP4"
    )


def modelopt_nvfp4_expert_layers(config: dict) -> list[int]:
    """The layers whose routed experts a ModelOpt export packed as NVFP4."""
    if not is_modelopt_nvfp4(config):
        return []
    text = config.get("text_config") or config
    layers = int(text["num_hidden_layers"])
    kinds = text.get("mlp_layer_types") or ["sparse"] * layers
    skip = set(
        (config.get("quantization_config") or {}).get("modules_to_not_convert") or ()
    )
    return [
        i
        for i, kind in enumerate(kinds)
        if kind == "sparse"
        and f"model.language_model.layers.{i}.mlp.experts" not in skip
    ]


def cast_predicate(key: str) -> bool:
    """Keep the router, the residual gates and the expert scales in FP32."""
    return not any(part in key for part in _FP32_PARAMS)


def sanitize_experts(weights: dict) -> dict:
    """Map the HF routed-expert tensors to the ``switch_mlp`` layout."""
    out = {}
    for key, value in weights.items():
        if ".mlp.experts." not in key:
            out[key] = value
            continue

        prefix, name = key.split(".mlp.experts.")
        if name.startswith("post_expert_norm."):
            out[f"{prefix}.mlp.{name}"] = value
            continue
        if name.endswith("_input_scale"):
            # Activation scales of the W4A4 export; activations stay in BF16.
            continue
        proj = "gate_up_proj" if name.startswith("gate_up_proj") else "down_proj"
        field = {
            "": "weight",
            "_weight_scale": "scales",
            "_weight_scale_2": "global_scale",
        }[name[len(proj) :]]
        if field == "weight" and value.dtype == mx.uint8:
            # Packed NVFP4 codes, two per byte.
            value = value.view(mx.uint32)
        elif field == "global_scale":
            value = value.astype(mx.float32) * NVFP4_GLOBAL_SCALE

        switch = f"{prefix}.mlp.switch_mlp."
        if proj == "down_proj":
            out[f"{switch}down_proj.{field}"] = value
        elif field == "global_scale":
            # The fused projection has one tensor scale per half.
            out[f"{switch}gate_proj.global_scale"] = value[:, 0]
            out[f"{switch}up_proj.global_scale"] = value[:, 1]
        else:
            # HF stores the experts as (experts, 2 * hidden, dim), gate first.
            gate, up = mx.split(value, 2, axis=1)
            out[f"{switch}gate_proj.{field}"] = gate
            out[f"{switch}up_proj.{field}"] = up
    return out


def sanitizer(config: dict, *, text_only: bool = False):
    """oQ's sanitize for a Muse Spark source checkpoint (HF, ModelOpt or mlx-lm)."""
    text = config.get("text_config") or config
    vision = config.get("vision_config") or {}

    def sanitize(weights: dict) -> dict:
        upstream = is_upstream_layout(weights)
        out = {}
        for key, value in weights.items():
            if key.startswith("model.audio_tower."):
                continue
            if key.startswith("model.language_model."):
                key = "language_model.model." + key[len("model.language_model.") :]
            elif key.startswith("lm_head."):
                key = "language_model." + key
            elif key.startswith("model."):
                key = key[len("model.") :]
            if text_only and key.startswith("vision_"):
                continue
            out[key] = value
        if not upstream:
            return sanitize_experts(out)
        out = read_upstream_text(
            out,
            LAYER_PREFIX,
            n_heads=int(text["num_attention_heads"]),
            n_kv_heads=int(text["num_key_value_heads"]),
            head_dim=int(text["head_dim"]),
        )
        if vision:
            head_dim = int(vision["hidden_size"]) // int(vision["num_attention_heads"])
            out = read_upstream_vision(out, "vision_tower.layers.", head_dim)
        return out

    sanitize._omlx_cast_predicate = cast_predicate
    return sanitize


class CalibrationBlock(muse_spark.TransformerBlock):
    """mlx-lm's decoder block with the call signature of oQ's calibration."""

    is_moe_layer = True

    def __call__(self, x, mask=None, cache=None):
        h, _ = super().__call__(x, mask=mask, cache=cache)
        return h


def calibration_args(config: dict):
    """The text arguments of mlx-lm's model for ``config``."""
    return muse_spark.ModelArgs.from_dict({**config, "vision_config": None}).text


def calibration_block(args, layer_idx: int) -> CalibrationBlock:
    return CalibrationBlock(
        args,
        use_rope=bool(args.layer_rope_theta[layer_idx]),
        use_sliding=args.layer_types[layer_idx] == "sliding_attention",
    )


def calibration_items(items, layer_idx: int, config: dict) -> list:
    """Map one layer's tensors in oQ's layout to ``CalibrationBlock`` names.

    NVFP4 experts arrive dequantized without their tensor scale, so the scale
    is folded into the weights here.
    """
    prefix = f"{LAYER_PREFIX}{layer_idx}."
    tensors = {prefix + key: value for key, value in items}
    for key in [k for k in tensors if k.endswith(".global_scale")]:
        weight = key[: -len("global_scale")] + "weight"
        scale = tensors.pop(key).astype(mx.float32) * (1.0 / NVFP4_GLOBAL_SCALE)
        dtype = tensors[weight].dtype
        tensors[weight] = (tensors[weight] * scale[:, None, None]).astype(dtype)
    writer = UpstreamWriter(config, centered_norms=not is_upstream_layout(tensors))
    out = writer(tensors)
    writer.finish()
    upstream_prefix = f"model.language_model.layers.{layer_idx}."
    return [(key[len(upstream_prefix) :], value) for key, value in out.items()]


def oq_names(name: str) -> list[str]:
    """oQ's tensor names for a calibration statistic named by mlx-lm's model."""
    if name.startswith("model.language_model."):
        name = "language_model.model." + name[len("model.language_model.") :]
    if re.search(r"(^|\.)self_attn\.qkvg_proj$", name):
        # q, k, v and the gate read the same input.
        base = name[: -len("qkvg_proj")]
        return [base + part for part in _ATTENTION_PARTS]
    return [re.sub(r"(^|\.)mlp\.experts\.", r"\1mlp.switch_mlp.", name)]
