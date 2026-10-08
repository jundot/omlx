# SPDX-License-Identifier: Apache-2.0
"""The KV cache geometry the local model expects, so a handoff made for another model is refused unapplied."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import wire
from .receiver import HandoffError


@dataclass(frozen=True)
class KVGeometry:
    """Every cache layer of the local model: its kind, KV heads and head size, or MLA latent and rope sizes."""

    kind: str
    layers: int
    heads: int = 0
    head_size: int = 0
    latent_size: int = 0
    rope_size: int = 0


def _config(model: Any) -> Any:
    args = getattr(model, "args", None)
    text = getattr(args, "text_config", None)
    return text if text is not None else args


def _read(config: Any, name: str) -> int | None:
    value = (
        config.get(name) if isinstance(config, dict) else getattr(config, name, None)
    )
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def model_geometry(model: Any, layers: int) -> tuple[KVGeometry | None, str]:
    """The geometry of `model`'s `layers` cache layers, or None and why it cannot be read."""
    config = _config(model)
    if config is None:
        return None, "its KV geometry is not in its config"
    latent, rope = _read(config, "kv_lora_rank"), _read(config, "qk_rope_head_dim")
    if latent and rope:
        return KVGeometry("mla", layers, latent_size=latent, rope_size=rope), ""
    heads = _read(config, "num_attention_heads")
    kv_heads = _read(config, "num_key_value_heads") or heads
    hidden = _read(config, "hidden_size")
    head_size = _read(config, "head_dim") or (
        hidden // heads if hidden and heads and hidden % heads == 0 else None
    )
    value_size = _read(config, "v_head_dim") or head_size
    if not kv_heads or not head_size:
        return None, "its KV geometry is not in its config"
    if value_size != head_size:
        return None, "its keys and values have different head sizes"
    return KVGeometry("attention", layers, heads=kv_heads, head_size=head_size), ""


def check_manifest(manifest: wire.Manifest, geometry: KVGeometry) -> None:
    """Raise HandoffError unless every layer of `manifest` fits the local model's caches."""
    if len(manifest.layers) != geometry.layers:
        raise HandoffError(
            f"the handoff has {len(manifest.layers)} layers; "
            f"the local model has {geometry.layers}"
        )
    for layer in manifest.layers:
        if layer.kind != geometry.kind:
            raise HandoffError(
                f"layer {layer.index} is an {layer.kind} layer; "
                f"the local model has {geometry.kind}"
            )
        if layer.kind == "mla":
            if (layer.latent_size, layer.rope_size) != (
                geometry.latent_size,
                geometry.rope_size,
            ):
                raise HandoffError(
                    f"layer {layer.index} has latent {layer.latent_size} and rope "
                    f"{layer.rope_size}; the local model has {geometry.latent_size} "
                    f"and {geometry.rope_size}"
                )
            continue
        if (layer.total_heads, layer.head_size) != (geometry.heads, geometry.head_size):
            raise HandoffError(
                f"layer {layer.index} holds {layer.total_heads} KV heads of "
                f"{layer.head_size}; the local model has {geometry.heads} of "
                f"{geometry.head_size}"
            )
