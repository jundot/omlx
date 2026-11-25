# SPDX-License-Identifier: Apache-2.0
"""Disk-backed routed experts for converted Inkling 2 checkpoints."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ...oqh.format import is_oqh_config
from ..moe_expert_offload import (
    CheckpointExpertStore,
    ExpertCache,
    _GLUStoreView,
    _OffloadExpertRoutes,
)
from ..qwen35_verify_qmm import is_row_exact_armed
from .model import Inkling2Experts, TextArgs, _expert_specs, _softcapped_swiglu

logger = logging.getLogger(__name__)
_PROJECTIONS = ("w13", "w2")


def _expert_layout(
    model_dir: Path, store: CheckpointExpertStore
) -> tuple[TextArgs, int, bool]:
    """Validate every physical layer before replacing modules or counting savings.

    Returns the text args, one expert's bytes across layers, and whether the
    checkpoint stores one tensor per expert (``experts.<e>.w13.weight``, the
    offload repack) instead of stacked ``[E, ...]`` tables."""
    config = json.loads((model_dir / "config.json").read_text())
    text = TextArgs.from_dict(config["text_config"])
    quant = config.get("quantization")
    if not isinstance(quant, dict) or is_oqh_config(config):
        raise ValueError(
            "Inkling 2 expert offload requires a converted affine checkpoint."
        )
    count, hidden, inter = (
        text.n_routed_experts,
        text.hidden_size,
        text.intermediate_size,
    )
    if (
        min(count, hidden, inter, text.num_hidden_layers) <= 0
        or not 0 < text.num_experts_per_tok <= count
    ):
        raise ValueError("Invalid Inkling 2 expert geometry.")

    dims = ((2 * inter, hidden), (hidden, inter))
    per_expert = store.has("model.layers.0.mlp.experts.0.w13.weight")
    expert_bytes = 0
    for layer_id in range(text.num_hidden_layers):
        prefix = f"model.layers.{layer_id}.mlp.experts"
        for proj, (output, width), spec in zip(
            _PROJECTIONS, dims, _expert_specs(text, quant, layer_id)
        ):
            bits, group = spec["bits"], spec["group_size"]
            if (
                spec["mode"] != "affine"
                or bits not in (2, 3, 4, 5, 6, 8)
                or group not in (32, 64, 128)
                or width % group
                or width * bits % 32
            ):
                raise ValueError(
                    f"Unsupported Inkling 2 expert quantization: {prefix}.{proj}"
                )
            for field in ("weight", "scales", "biases"):
                is_weight = field == "weight"
                shape = (output, width * bits // 32 if is_weight else width // group)
                dtypes = {"U32"} if is_weight else {"BF16", "F16", "F32"}
                if per_expert:
                    names = [f"{prefix}.{e}.{proj}.{field}" for e in range(count)]
                else:
                    names = [f"{prefix}.{proj}.{field}"]
                    shape = (count,) + shape
                for name in names:
                    if not store.has(name):
                        raise ValueError(
                            f"Checkpoint is missing Inkling 2 expert tensor: {name}"
                        )
                    actual_shape, dtype = store.spec(name)
                    if actual_shape != shape or dtype not in dtypes:
                        raise ValueError(f"Unsupported Inkling 2 expert tensor: {name}")
                plan = (
                    store.plan_tensor(names[0])
                    if per_expert
                    else store.plan_expert(names[0], 0)
                )
                expert_bytes += plan.nbytes
    return text, expert_bytes, per_expert


def _capacity(text: TextArgs, resident_fraction: float) -> int:
    count = text.n_routed_experts
    return min(
        count, max(8, text.num_experts_per_tok, round(count * resident_fraction))
    )


def estimate_inkling2_expert_savings(model_dir: Path, resident_fraction: float) -> int:
    text, expert_bytes, _ = _expert_layout(model_dir, CheckpointExpertStore(model_dir))
    return expert_bytes * (text.n_routed_experts - _capacity(text, resident_fraction))


class OffloadInkling2Experts(_OffloadExpertRoutes):
    def __init__(
        self, experts: Inkling2Experts, capacity: int, disk: _GLUStoreView, floor: int
    ):
        super().__init__()
        self.cache = ExpertCache(
            experts,
            capacity,
            disk,
            floor,
            projections=_PROJECTIONS,
            direct_reads=True,
            # Start the resident routes as soon as routing is known: with
            # reads landing in place, waiting for a fast miss only idles the GPU.
            overlap_grace_s=0.0,
        )
        self._gcap = experts._gcap
        self._ucap = experts._ucap
        self._overlap = True

    def _glu(self, x, slots, sorted_indices=False):
        y = self.cache.qmm("w13", x, slots, sorted_indices)
        act = _softcapped_swiglu(y[..., 0::2], y[..., 1::2], self._gcap, self._ucap)
        return self.cache.qmm("w2", act, slots, sorted_indices)

    def unreduced(self, x, inds, weights):
        do_sort = inds.size >= 64 and not is_row_exact_armed()
        return self._run_routes(x, inds, do_sort), weights


def apply_inkling2_expert_offload(
    model, model_dir: Path, store: CheckpointExpertStore, resident_fraction: float
) -> int:
    text, expert_bytes, per_expert = _expert_layout(model_dir, store)
    layers = model.layers
    if len(layers) != text.num_hidden_layers or any(
        not isinstance(layer.mlp.experts, Inkling2Experts) for layer in layers
    ):
        raise ValueError("Inkling 2 expert modules do not match the checkpoint layout.")

    capacity = _capacity(text, resident_fraction)
    for i, layer in enumerate(layers):
        # The per-expert view names tensors under the experts' parent.
        prefix = f"model.layers.{i}.mlp" + ("" if per_expert else ".experts")
        disk = _GLUStoreView(store, prefix, per_expert=per_expert)
        layer.mlp.experts = OffloadInkling2Experts(
            layer.mlp.experts, capacity, disk, max(8, text.num_experts_per_tok)
        )
    logger.info(
        "Inkling 2 expert offload: wrapped %d layers with %d/%d slots each "
        "(%.2f GiB of expert slots)",
        len(layers),
        capacity,
        text.n_routed_experts,
        expert_bytes * capacity / 2**30,
    )
    return len(layers)
