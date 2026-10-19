# SPDX-License-Identifier: Apache-2.0
"""The checkpoint layout of mlx-lm's own Muse Spark model.

mlx-lm 0.32.2 added ``mlx_lm.models.muse_spark``. Its checkpoints hold the
same weights as this package, arranged for its fused kernels:

* ``self_attn.qkvg_proj``: q, k, v and the attention output gate as one
  projection. The q and k rows of each head are interleaved for traditional
  RoPE: pairs (2i, 2i + 1) instead of (i, i + head_dim / 2).
* ``mlp.experts``: the routed experts and ``post_expert_norm``, which this
  package keeps at ``mlp.switch_mlp`` and ``mlp.post_expert_norm``.
* ``add_attn_alpha`` and the like: the fixed residual gate coefficients, in
  place of the gates.
* The zero-centered norms as their gains, ``T(1 + weight)``.
* ``attn.qkv_proj`` in the vision tower, with interleaved q and k rows.
* Text weights under ``model.language_model.``, vision weights under ``model.``.

``read_upstream_*`` map such a checkpoint to this package's names, and
``UpstreamWriter`` writes oQ output in the upstream layout.
"""

from __future__ import annotations

import re

import mlx.core as mx

RESIDUAL_GATE_EPS = 1e-3

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
    """Map the upstream decoder layers under ``layers_prefix`` to this package."""
    q_rows = n_heads * head_dim
    kv_rows = n_kv_heads * head_dim
    splits = [q_rows, q_rows + kv_rows, q_rows + 2 * kv_rows]
    coefficients = {
        f"{name}_{part}": f"{gate}.{part}"
        for gate, name in RESIDUAL_GATES.items()
        for part in ("alpha", "beta")
    }
    out = {}
    for key, value in weights.items():
        if not key.startswith(layers_prefix):
            out[key] = value
            continue
        index, _, name = key[len(layers_prefix) :].partition(".")
        layer = f"{layers_prefix}{index}."
        if name.startswith("self_attn.qkvg_proj."):
            param = name[len("self_attn.qkvg_proj.") :]
            q, k, v, g = mx.split(value, splits)
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
            out[layer + coefficients.get(name, name)] = value
    return out


def read_upstream_vision(weights: dict, layers_prefix: str, head_dim: int) -> dict:
    """Split the fused vision ``attn.qkv_proj`` under ``layers_prefix``."""
    out = {}
    for key, value in weights.items():
        if not (key.startswith(layers_prefix) and ".attn.qkv_proj." in key):
            out[key] = value
            continue
        attn, param = key.split(".attn.qkv_proj.")
        q, k, v = mx.split(value, 3)
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
    them have arrived. ``finish`` checks that none is left.
    """

    def __init__(self, config: dict):
        text = config.get("text_config") or config
        self.head_dim = int(text["head_dim"])
        self.temperature = float(text.get("residual_gate_temperature", 0.3))
        vision = config.get("vision_config") or {}
        self.vision_head_dim = (
            int(vision["hidden_size"]) // int(vision["num_attention_heads"])
            if vision
            else 0
        )
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
            if _CENTERED_NORM_RE.match(key):
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
