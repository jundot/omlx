# SPDX-License-Identifier: Apache-2.0
"""Model-family normalization for SwitchGLU expert streaming."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MoELayerAdapter:
    """A routed layer normalized across the supported model families."""

    layer_id: int
    layer: Any
    container_name: str
    moe: Any
    router_name: str
    router: Any
    switch_mlp: Any
    num_experts: int
    top_k: int

    def replace_switch(self, replacement: Any) -> None:
        self.moe.switch_mlp = replacement

    def replace_router(self, replacement: Any) -> None:
        if not self.router_name:
            raise ValueError(f"Layer {self.layer_id} does not expose a router")
        setattr(self.moe, self.router_name, replacement)


def _main_layers(model: Any) -> list[Any]:
    candidates = (
        getattr(
            getattr(getattr(model, "language_model", None), "model", None),
            "layers",
            None,
        ),
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(model, "model", None), "pipeline_layers", None),
        getattr(model, "layers", None),
        getattr(model, "pipeline_layers", None),
    )
    for layers in candidates:
        if isinstance(layers, (list, tuple)) and layers:
            return list(layers)
    raise ValueError("Could not locate the model's routed MoE layers")


def _first_int(objects: tuple[Any, ...], names: tuple[str, ...]) -> int:
    for obj in objects:
        if obj is None:
            continue
        for name in names:
            value = getattr(obj, name, None)
            if value is not None:
                try:
                    result = int(value)
                except (TypeError, ValueError):
                    continue
                if result > 0:
                    return result
    return 0


def discover_moe_layers(model: Any) -> list[MoELayerAdapter]:
    """Find Qwen, DeepSeek, GLM, MiMo, Hy, Laguna and similar SwitchGLUs."""

    result: list[MoELayerAdapter] = []
    for layer_id, layer in enumerate(_main_layers(model)):
        if layer is None:
            continue
        container_name = ""
        moe = None
        for candidate in ("mlp", "ffn"):
            value = getattr(layer, candidate, None)
            if getattr(value, "switch_mlp", None) is not None:
                container_name = candidate
                moe = value
                break
        if moe is None:
            continue
        switch = moe.switch_mlp
        projection = getattr(switch, "up_proj", None) or getattr(
            switch, "gate_up_proj", None
        )
        projection_experts = 0
        if projection is not None:
            projection_experts = _first_int(
                (projection,), ("num_experts",)
            )
            if not projection_experts:
                weight = getattr(projection, "weight", None)
                if weight is not None and len(weight.shape) >= 3:
                    projection_experts = int(weight.shape[0])
        router_name = ""
        gate = None
        for candidate in ("gate", "router"):
            value = getattr(moe, candidate, None)
            if value is not None:
                router_name = candidate
                gate = value
                break
        config = getattr(moe, "config", None) or getattr(moe, "args", None)
        num_experts = _first_int(
            (moe, gate, config),
            ("num_experts", "n_routed_experts", "n_routed"),
        ) or projection_experts
        top_k = _first_int(
            (moe, gate, config),
            ("top_k", "num_experts_per_tok", "num_experts_per_token"),
        )
        if getattr(moe, "pack_shared_expert", False):
            top_k += 1
        if not num_experts or not top_k:
            raise ValueError(
                f"Layer {layer_id} does not expose compatible MoE geometry"
            )
        result.append(
            MoELayerAdapter(
                layer_id=layer_id,
                layer=layer,
                container_name=container_name,
                moe=moe,
                router_name=router_name,
                router=gate,
                switch_mlp=switch,
                num_experts=num_experts,
                top_k=top_k,
            )
        )
    if not result:
        raise ValueError("The selected model has no compatible SwitchGLU MoE layers")
    geometry = {(target.num_experts, target.top_k) for target in result}
    if len(geometry) != 1:
        raise ValueError(f"Mixed routed-expert geometry is unsupported: {geometry}")
    return result


def projection_schema(switch_mlp: Any) -> dict[str, dict[str, Any]]:
    """Describe separate or fused affine/MXFP/NVFP expert projections."""

    metadata: dict[str, dict[str, Any]] = {}
    gate_up = getattr(switch_mlp, "gate_up_proj", None)
    for projection in ("gate_proj", "up_proj", "down_proj"):
        module = getattr(switch_mlp, projection, None)
        fused_half = None
        if module is None and projection in {"gate_proj", "up_proj"}:
            module = gate_up
            fused_half = "gate" if projection == "gate_proj" else "up"
        if module is None or not all(
            hasattr(module, name) for name in ("weight", "scales")
        ):
            raise ValueError(
                "Expert streaming requires stacked quantized SwitchGLU weights"
            )
        parts = ["weight", "scales"]
        if getattr(module, "biases", None) is not None:
            parts.append("biases")
        if getattr(module, "bias", None) is not None:
            parts.append("bias")
        metadata[projection] = {
            "group_size": int(getattr(module, "group_size", 64)),
            "bits": int(getattr(module, "bits", 4)),
            "mode": str(getattr(module, "mode", "affine")),
            "parts": tuple(parts),
            "source_projection": "gate_up_proj" if fused_half else projection,
            "fused_half": fused_half,
            "arrays": {part: getattr(module, part) for part in parts},
        }
    return metadata
