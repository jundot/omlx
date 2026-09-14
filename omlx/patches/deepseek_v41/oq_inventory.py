# SPDX-License-Identifier: MIT
"""Read logical projection sizes without materializing source weights."""

import json
import math
import re
import struct
from pathlib import Path

from .convert import mapped


def projection_inventory(source, config, mapping, *, preserve_mtp=False):
    headers = {}
    source = Path(source)
    for filename in dict.fromkeys(mapping.values()):
        with (source / filename).open("rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            headers[filename] = json.loads(handle.read(length))
    inventory = {}
    text_config = config.get("text_config", config)
    for key, filename in mapping.items():
        if not key.endswith(".weight") or ".engram.embed." in key:
            continue
        if key.startswith("mtp.") and not preserve_mtp:
            continue
        entry = headers[filename][key]
        shape = list(entry["shape"])
        dtype = entry["dtype"]
        if dtype in ("I8", "U8"):
            shape[-1] *= 2
        match = re.match(
            r"((?:layers|mtp)\.\d+\.ffn\.experts)\.\d+\.(w[123])\.weight$", key
        )
        prefix = mapped(key.removesuffix(".weight"))
        if match:
            base, projection = match.groups()
            prefix = mapped(f"{base}.{projection}")
            if prefix in inventory:
                continue
            experts = text_config["n_routed_experts"]
            if base.startswith("mtp."):
                experts = text_config.get("dspark_n_routed_experts") or experts
            shape.insert(0, experts)
        elements = math.prod(shape)
        forced_dense = key.endswith(
            ("wo_a.weight", ".markov_head.embed.weight", ".markov_head.head.weight")
        ) or key in ("embed.weight", "head.weight")
        quantization = None
        if dtype in ("I8", "U8") or dtype.startswith("F8_E4M3"):
            bits = 4 if dtype in ("I8", "U8") else 8
            if forced_dense:
                byte_count = elements * 2
            else:
                byte_count = elements * bits // 8 + elements // 32
                quantization = {"bits": bits, "group_size": 32, "mode": f"mxfp{bits}"}
        else:
            byte_count = entry["data_offsets"][1] - entry["data_offsets"][0]
            if match:
                byte_count *= experts
        inventory[prefix] = dict(
            shape=tuple(shape),
            source_bytes=byte_count,
            source_quantization=quantization,
        )
    return inventory


def build_affine_plan(inventory, budget, config, imatrix, sensitivity, *, target, cap):
    """Price preserved source tensors separately from eligible affine projections."""
    import numpy as np

    from ...oq import (
        _build_quant_plan,
        _tensor_quantized_bytes,
        universal_quant_predicate,
    )

    if not sensitivity or any(
        not math.isfinite(value) or value < 0 for value in sensitivity.values()
    ):
        raise ValueError("V4.1 allocation requires measured finite layer sensitivity")
    config = {
        **config,
        "_oq_use_budget_plan": True,
        "_oq_sensitivity_map": {str(key): value for key, value in sensitivity.items()},
    }
    shapes = {}
    preserved = {}
    unobserved = {}
    for name, info in inventory.items():
        shape = tuple(info["shape"])
        entry = imatrix.entries.get(name)
        if name.startswith(("vision.", "aligner.", "image_")):
            preserved[name] = "precision_policy"
        elif len(shape) < 2 or shape[-1] % 64:
            preserved[name] = "ineligible_shape"
        elif ".engram." in name:
            preserved[name] = "outside_layer_sensitivity"
        elif entry is None:
            preserved[name] = "uncalibrated"
        elif universal_quant_predicate(name, None, config, 3) is False:
            preserved[name] = "precision_policy"
        else:
            shapes[name] = shape
            missing = np.flatnonzero(entry.counts <= 0).tolist()
            if missing:
                unobserved[name] = missing
    if not shapes:
        raise ValueError("V4.1 oQ3e has no calibrated eligible projections")
    total_params = budget["remaining_weights"]["logical_parameters"]
    eligible_params = sum(math.prod(shape) for shape in shapes.values())
    fixed_bytes = budget["remaining_weights"]["tensor_bytes"] - sum(
        inventory[name]["source_bytes"] for name in shapes
    )
    plan = _build_quant_plan(
        shapes,
        config,
        3,
        target_bpw=(target * total_params - 8 * fixed_bytes) / eligible_params,
        hard_cap_bpw=(cap * total_params - 8 * fixed_bytes) / eligible_params,
        supported_bits=(2, 3, 4, 6, 8),
    )
    specs = {
        name: plan.boost_map.get(name, dict(bits=3, group_size=64, mode="affine"))
        for name in shapes
    }
    if any(spec["bits"] not in (2, 3, 4, 6, 8) for spec in specs.values()):
        raise ValueError("V4.1 allocation selected an unsupported affine width")
    output_bytes = fixed_bytes + sum(
        _tensor_quantized_bytes(
            shapes[name], spec["bits"], spec["group_size"], spec["mode"]
        )
        for name, spec in specs.items()
    )
    effective_bpw = 8 * output_bytes / total_params
    if effective_bpw > cap:
        raise ValueError(
            f"V4.1 calibrated allocation requires {effective_bpw:.4f} bpw, above {cap:.4f} cap"
        )
    return specs, dict(
        tensor_bytes=output_bytes,
        effective_bpw=effective_bpw,
        fixed_source_bytes=fixed_bytes,
        preserved_modules=preserved,
        unobserved_experts=unobserved,
        unobserved_expert_policy="uniform_importance_at_allocated_bits",
    )
