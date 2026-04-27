# SPDX-License-Identifier: Apache-2.0
"""oQ: oMLX Universal Dynamic Quantization.

Mixed-precision quantization combining GGUF K-quant layer position strategy,
unsloth Dynamic 2.0 selective non-quantization, and BnB MSE-optimal clipping.

Supported levels: oQ2, oQ3, oQ4, oQ6, oQ8 (base bits differ, same predicate).
"""

import json
import logging
import re
import shutil
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

logger = logging.getLogger(__name__)

OQ_LEVELS = {2, 3, 3.5, 4, 5, 6, 8}

OQ_DTYPES: tuple[str, ...] = ("bfloat16", "float16")

_OQ_DEFAULT_GROUP_SIZE = 64

_LEVEL_BITS: dict[float, int] = {2: 2, 3: 3, 3.5: 3, 4: 4, 5: 5, 6: 6, 8: 8}

_LEVEL_PROTECTION: dict[float, str] = {
    2: "full", 3: "full", 3.5: "full",
    4: "full", 5: "full", 6: "full", 8: "full",
}

_OQ_BPW_TARGETS: dict[float, tuple[float, float]] = {
    2: (2.8, 3.0),
    3: (3.5, 3.7),
    3.5: (3.8, 4.0),
    4: (4.6, 4.7),
    5: (5.5, 5.7),
    6: (6.5, 6.7),
}


def _bpw_targets_for_level(oq_level: float) -> tuple[float, float] | None:
    """Return (target_bpw, hard_cap_bpw) for the given oQ level, or None."""
    return _OQ_BPW_TARGETS.get(oq_level)


@dataclass
@dataclass
class QuantPlan:
    """Byte-budgeted mixed-precision plan for a single quantization run."""

    boost_map: dict[str, dict]
    effective_bpw: float
    target_bpw: float
    hard_cap_bpw: float



def universal_quant_predicate(
    path: str, module, config: dict, oq_level: int = 4
) -> Union[bool, dict]:
    """Per-tensor quantization decision based on GGUF/unsloth/llama.cpp rules.

    Protection levels vary by oQ level:
        oQ2: minimal protection (router fp16, lm_head 4-bit only) → ~2.5 bpw
        oQ3: base 2-bit + full protection → ~3.3 bpw
        oQ4-oQ6: base N-bit + full protection
        oQ7: base 8-bit + full protection
        oQ8: near-uniform 8-bit (router fp16 only) → ~8.0 bpw

    Args:
        path: Dot-separated module path (e.g. "model.layers.0.self_attn.v_proj").
        module: The nn.Module being quantized.
        config: Model config.json dict.
        oq_level: oQ quantization level (2-8).

    Returns:
        False to skip quantization (keep fp16),
        True to use default bits,
        dict with {"bits": N, "group_size": M} for per-layer override.
    """
    path = _normalize_quant_path(path)
    path_l = path.lower()

    non_quantizable = config.get("_oq_non_quantizable", set())
    if path in non_quantizable:
        return False

    tc = config.get("text_config", {})
    num_layers = config.get("num_hidden_layers") or tc.get("num_hidden_layers", 32)
    num_experts = (
        config.get("num_local_experts")
        or tc.get("num_local_experts")
        or config.get("num_experts")
        or tc.get("num_experts", 0)
        or 0
    )
    hidden_size = config.get("hidden_size") or tc.get("hidden_size", 0)
    is_moe = num_experts > 0

    base_bits = int(_LEVEL_BITS.get(oq_level, oq_level))
    protection = _LEVEL_PROTECTION.get(oq_level, "full")
    full_protection = protection == "full"

    def gs():
        if _is_moe_router(path):
            return 64
        if num_experts >= 150:
            return 128
        return 64

    def bits(n):
        effective = int(max(n, base_bits))
        return {
            "bits": effective,
            "group_size": _gs_for_mode(effective, gs()),
            "mode": _mode_for_bits(effective),
        }

    if _is_moe_router(path):
        return False  # fp16 — tiny weights, some models (MoEGate) lack to_quantized()

    if "shared_expert_gate" in path and "gate_proj" not in path:
        return {"bits": 8, "group_size": 64, "mode": "affine"}

    if _is_vision_tensor(path):
        return False

    if any(
        p in path_l
        for p in ("ssm_alpha", "ssm_beta", "a_log", "time_decay", "time_faaaa")
    ):
        return False

    if path.endswith(".D"):
        return False

    # Gated DeltaNet / Mamba-like SSM sensitive params (Qwen3_5 hybrid arch).
    # dt_bias drives the discretization step, keep fp16/fp32 like A_log.
    # conv1d is a small depth-wise causal conv, very sensitive to low bits.
    # linear_attn.out_proj mirrors self_attn.o_proj sensitivity.
    if path_l.endswith("dt_bias"):
        return False
    if "conv1d" in path_l and "linear_attn" in path_l:
        return bits(8)
    if "linear_attn.out_proj" in path_l:
        return bits(5)

    boost_map = config.get("_oq_boost_map") or {}
    if path in boost_map:
        return dict(boost_map[path])

    if config.get("_oq_use_budget_plan"):
        if any(p in path for p in ("ssm_output", "ssm_out")):
            return bits(8)
        if "lora.2" in path:
            return bits(8)
        return True

    if not full_protection:
        if any(p in path for p in ("lm_head", "output.weight", "classifier")):
            return bits(6)

        if any(p in path for p in ("ssm_output", "ssm_out")):
            return bits(8)

        if any(p in path for p in ("embed_tokens", "wte", "word_embeddings")):
            return bits(base_bits + 2)

        if num_experts >= 512 and hidden_size >= 4096:
            if "gate_proj" in path and "shared_expert" not in path:
                return bits(4)

        layer_idx = _extract_layer_index(path)
        if layer_idx >= 0:
            sensitive = (
                layer_idx < num_layers // 8
                or layer_idx >= 7 * num_layers // 8
            )
            is_expert = "switch_mlp" in path or "experts" in path
            if sensitive and not is_expert:
                return bits(base_bits + 1)

        return True

    if any(p in path for p in ("ssm_output", "ssm_out")):
        return bits(8)

    if "lora.2" in path:
        return bits(8)

    if any(p in path for p in ("lm_head", "output.weight", "classifier")):
        return bits(6)

    if "cross_attn" in path and "o_proj" in path:
        return bits(6)

    if any(
        p in path
        for p in ("kv_a_proj_with_mqa", "kv_b_proj", "q_a_proj", "q_b_proj")
    ):
        return bits(6)

    if "o_proj" in path and "shared_expert" not in path:
        if not is_moe:
            return bits(5)

    if "shared_expert" in path and not path.endswith("shared_expert_gate"):
        return bits(8)

    if num_experts >= 512 and hidden_size >= 4096:
        if "gate_proj" in path and "shared_expert" not in path:
            return bits(4)
        if "down_proj" in path and "shared_expert" not in path:
            return bits(3)

    layer_idx = _extract_layer_index(path)

    sensitivity_map = config.get("_oq_sensitivity_map")
    if sensitivity_map and layer_idx >= 0:
        scores = list(sensitivity_map.values())
        scores.sort(reverse=True)
        threshold = scores[max(0, len(scores) // 4 - 1)] if scores else 0
        sensitive = sensitivity_map.get(str(layer_idx), 0) >= threshold
    else:
        sensitive = layer_idx >= 0 and (
            layer_idx < num_layers // 8
            or layer_idx >= 7 * num_layers // 8
        )

    if any(p in path for p in ("v_proj", "v_a_proj", "v_b_proj")):
        if sensitive:
            return bits(6)
        return True

    if any(p in path for p in ("down_proj", "w2", "mlp.fc2", "wo")):
        is_routed_expert = is_moe and "shared_expert" not in path and (
            "switch_mlp" in path or "experts" in path
        )
        if is_routed_expert:
            if oq_level == 3.5:
                return bits(4)
            return True
        if sensitive:
            return bits(6)
        return bits(5)

    if any(p in path for p in ("q_proj", "k_proj")):
        if sensitive:
            return bits(5)

    if any(p in path for p in ("qkv_proj", "in_proj_qkv", "attn_qkv")):
        if sensitive:
            return bits(5)

    if any(p in path for p in ("in_proj_z", "in_proj_a", "in_proj_b", "delta_net")):
        return bits(5)

    if any(
        p in path for p in ("mixer.in_proj", "mixer.out_proj", "x_proj", "dt_proj")
    ):
        return bits(5)

    return True


def _is_vision_tensor(name: str) -> bool:
    """Check if a tensor belongs to the vision encoder/projector."""
    return any(
        p in name
        for p in (
            "visual.", "vision_", "patch_embed", "pos_embed",
            "image_newline", "multi_modal_projector", "visual.merger",
            "image_norm", "temporal_embed",
        )
    )


def _is_moe_router(path: str) -> bool:
    """Detect MoE router/gate layers (distinct from gate_proj)."""
    if path.endswith(("mlp.gate", ".router", ".router.layer")):
        return True
    if path.endswith(".gate") and "gate_proj" not in path:
        return True
    if ".gate." in path and "gate_proj" not in path:
        return True
    return False


def _extract_layer_index(path: str) -> int:
    """Extract transformer layer index from module path. Returns -1 if absent."""
    m = re.search(r"layers\.(\d+)\.", path)
    return int(m.group(1)) if m else -1


def _default_bits(config: dict) -> int:
    """Read default quantization bits from config."""
    q = config.get("quantization", {})
    return q.get("bits", 4)


def _normalize_quant_path(path: str) -> str:
    """Normalize tensor/module names to the module path used in configs."""
    if path.endswith(".weight"):
        return path[:-7]
    if path.endswith(".scales"):
        return path[:-7]
    if path.endswith(".biases"):
        return path[:-7]
    return path


def _base_bits_for_level(oq_level: int) -> int:
    return int(_LEVEL_BITS.get(oq_level, oq_level))


def _bytes_per_group(mode: str) -> int:
    if mode == "mxfp4":
        return 1
    if mode == "mxfp8":
        return 2
    return 4


def _tensor_quantized_bytes(shape: tuple, bits: int, group_size: int, mode: str) -> int:
    """Estimate serialized bytes for a quantized tensor."""
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    if len(shape) < 2:
        return n_elements * 2
    if shape[-1] % group_size != 0:
        return n_elements * 2
    rows = n_elements // max(shape[-1], 1)
    n_groups = shape[-1] // group_size
    weight_bytes = (n_elements * bits + 7) // 8
    overhead_bytes = rows * n_groups * _bytes_per_group(mode)
    return weight_bytes + overhead_bytes


def _estimate_effective_bpw(
    named_shapes: dict[str, tuple],
    base_bits: int,
    base_group_size: int,
    base_mode: str,
    overrides: dict[str, dict] | None = None,
) -> float:
    """Estimate effective bpw for quantizable weights only."""
    overrides = overrides or {}
    total_bits = 0
    total_params = 0

    for path, shape in named_shapes.items():
        n_elements = 1
        for dim in shape:
            n_elements *= dim
        total_params += n_elements

        override = overrides.get(path)
        if override is None:
            bits = base_bits
            gs = base_group_size
            mode = base_mode
        else:
            bits = int(override.get("bits", base_bits))
            gs = int(override.get("group_size", base_group_size))
            mode = override.get("mode", _mode_for_bits(bits))

        total_bits += 8 * _tensor_quantized_bytes(shape, bits, gs, mode)

    return total_bits / max(total_params, 1)


def _collect_named_weight_shapes_from_model(model) -> dict[str, tuple]:
    """Collect quantizable weight shapes from the in-memory model."""
    named_shapes = {}
    for path, module in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        if not hasattr(module, "weight") or not hasattr(module, "to_quantized"):
            continue
        if getattr(module.weight, "ndim", 0) < 2:
            continue
        named_shapes[_normalize_quant_path(path)] = tuple(module.weight.shape)
    return named_shapes


def _collect_named_weight_shapes_from_weights(weights: dict[str, Any]) -> dict[str, tuple]:
    """Collect quantizable weight shapes from sanitized weight tensors."""
    named_shapes = {}
    for name, tensor in weights.items():
        norm_name = _normalize_quant_path(name)
        if name != f"{norm_name}.weight":
            continue
        if getattr(tensor, "ndim", 0) < 2:
            continue
        named_shapes[norm_name] = tuple(tensor.shape)
    return named_shapes


def _is_routed_expert(path: str) -> bool:
    """Check if a tensor belongs to routed MoE experts (93-98% of params)."""
    if "switch_mlp" in path:
        return True
    if "experts" in path and "shared_expert" not in path:
        return True
    if "block_sparse_moe" in path and "shared_expert" not in path:
        return True
    return False


_MANDATORY_BOOST_PATTERNS = {
    "lm_head": {"bits": 8, "group_size": 64, "mode": "affine"},
    "embeddings": {"bits": 8, "group_size": 64, "mode": "affine"},
    "embed_tokens": {"bits": 8, "group_size": 64, "mode": "affine"},
    "wte": {"bits": 8, "group_size": 64, "mode": "affine"},
}


def _sensitivity_tier(layer_score: float, max_score: float) -> int:
    """Map sensitivity score to boost tier: +4 (top), +2 (high), +1 (moderate).

    Greedy allocator will fallback to lower tiers if budget can't fit the
    requested bits (e.g., 8-bit → try 6-bit → try 5-bit).
    """
    if max_score <= 0:
        return 1
    ratio = layer_score / max_score
    if ratio >= 0.5:
        return 4
    if ratio >= 0.2:
        return 2
    return 1


def _build_quant_plan(
    named_shapes: dict[str, tuple],
    config: dict,
    oq_level: int,
    target_bpw: float = 4.6,
    hard_cap_bpw: float = 4.7,
) -> QuantPlan:
    """Allocate byte-budgeted boosts using sensitivity-driven allocation.

    Strategy:
    1. Mandatory pre-allocation: consensus-critical tensors (lm_head → 8-bit)
    2. Data-driven: all non-expert tensors compete equally, ranked by
       layer sensitivity score. Higher sensitivity → more bits.
    3. Routed experts always stay at base bits (93-98% of params).
    """
    base_bits = _base_bits_for_level(oq_level)
    base_mode = _mode_for_bits(base_bits)
    base_group_size = _gs_for_mode(base_bits, _OQ_DEFAULT_GROUP_SIZE)
    boost_map: dict[str, dict] = {}

    layer_scores = config.get("_oq_sensitivity_map") or {}
    max_layer_score = max(layer_scores.values(), default=0.0)

    total_params = 0
    expert_params = 0
    for path, shape in named_shapes.items():
        n = 1
        for dim in shape:
            n *= dim
        total_params += n
        if _is_routed_expert(path):
            expert_params += n

    current_bpw = _estimate_effective_bpw(
        named_shapes, base_bits, base_group_size, base_mode
    )
    total_bits_f = current_bpw * total_params

    module = None
    for path, shape in named_shapes.items():
        pred = universal_quant_predicate(
            path, module, {**config, "_oq_boost_map": {}}, oq_level
        )
        if pred is False:
            continue
        for pattern, boost in _MANDATORY_BOOST_PATTERNS.items():
            if pattern in path:
                cand_bits = int(boost["bits"])
                if cand_bits <= base_bits:
                    break
                cand_gs = int(boost.get("group_size", base_group_size))
                cand_mode = boost.get("mode", _mode_for_bits(cand_bits))
                base_cost = _tensor_quantized_bytes(
                    shape, base_bits, base_group_size, base_mode
                )
                cand_cost = _tensor_quantized_bytes(
                    shape, cand_bits, cand_gs, cand_mode
                )
                delta = 8 * (cand_cost - base_cost)
                next_bpw = (total_bits_f + delta) / total_params
                if delta > 0 and next_bpw <= hard_cap_bpw:
                    boost_map[path] = dict(boost)
                    total_bits_f += delta
                    current_bpw = next_bpw
                break

    # oQ3.5: mandatory expert down_proj 4-bit (Super Weights protection)
    if oq_level == 3.5:
        for path, shape in named_shapes.items():
            if path in boost_map:
                continue
            if not _is_routed_expert(path):
                continue
            if not any(p in path for p in ("down_proj", "w2")):
                continue
            cand_bits = base_bits + 1  # 3→4
            if cand_bits not in (2, 3, 4, 5, 6, 8):
                continue
            cand_gs = _gs_for_mode(cand_bits, _OQ_DEFAULT_GROUP_SIZE)
            cand_mode = _mode_for_bits(cand_bits)
            base_cost = _tensor_quantized_bytes(
                shape, base_bits, base_group_size, base_mode
            )
            cand_cost = _tensor_quantized_bytes(shape, cand_bits, cand_gs, cand_mode)
            delta = 8 * (cand_cost - base_cost)
            if delta > 0:
                boost_map[path] = {"bits": cand_bits, "group_size": cand_gs, "mode": cand_mode}
                total_bits_f += delta
                current_bpw = total_bits_f / total_params

    # Protection floor: apply full protection rules as minimum bits for
    # non-expert tensors. This ensures attention, shared experts, etc. get
    # adequate precision even at aggressive base bits (e.g. oQ2 base=2).
    # Each floor boost is checked against hard_cap to avoid overshooting.
    floor_config = {**config, "_oq_use_budget_plan": False, "_oq_boost_map": {}}
    for path, shape in named_shapes.items():
        if path in boost_map:
            continue
        if _is_routed_expert(path):
            continue
        floor_pred = universal_quant_predicate(path, module, floor_config, oq_level)
        if not isinstance(floor_pred, dict):
            continue
        floor_bits = int(floor_pred["bits"])
        if floor_bits <= base_bits:
            continue
        floor_gs = int(floor_pred.get("group_size", _gs_for_mode(floor_bits, _OQ_DEFAULT_GROUP_SIZE)))
        floor_mode = floor_pred.get("mode", _mode_for_bits(floor_bits))
        old_cost = _tensor_quantized_bytes(shape, base_bits, base_group_size, base_mode)
        new_cost = _tensor_quantized_bytes(shape, floor_bits, floor_gs, floor_mode)
        delta = 8 * (new_cost - old_cost)
        if delta <= 0:
            continue
        next_bpw = (total_bits_f + delta) / total_params
        if next_bpw > hard_cap_bpw:
            continue
        boost_map[path] = {"bits": floor_bits, "group_size": floor_gs, "mode": floor_mode}
        total_bits_f += delta
        current_bpw = next_bpw

    # Sensitivity-based greedy boost: boost tensors from their current bits
    # (which may already be elevated by the protection floor) using remaining
    # budget up to hard_cap_bpw.
    candidates = []
    for path, shape in named_shapes.items():
        if _is_routed_expert(path):
            continue
        pred = universal_quant_predicate(
            path, module, {**config, "_oq_boost_map": {}}, oq_level
        )
        if pred is False:
            continue
        layer_idx = _extract_layer_index(path)
        if layer_idx < 0:
            continue
        layer_score = float(layer_scores.get(str(layer_idx), 0.0))
        # Current bits (floor or base)
        cur_bits = boost_map[path]["bits"] if path in boost_map else base_bits
        cur_gs = _gs_for_mode(cur_bits, _OQ_DEFAULT_GROUP_SIZE)
        cur_mode = _mode_for_bits(cur_bits)
        cur_cost = _tensor_quantized_bytes(shape, cur_bits, cur_gs, cur_mode)
        # Max target based on sensitivity
        ratio = layer_score / max_layer_score if max_layer_score > 0 else 0
        if ratio >= 0.5:
            max_target = 8
        elif ratio >= 0.2:
            max_target = min(cur_bits + 2, 8)
        else:
            max_target = min(cur_bits + 1, 8)
        if max_target <= cur_bits:
            continue
        candidates.append((layer_score, path, shape, cur_bits, cur_cost, max_target))

    _VALID_BITS = (2, 3, 4, 5, 6, 8)
    for _score, path, shape, cur_bits, cur_cost, max_target in sorted(
        candidates, key=lambda x: x[0], reverse=True
    ):
        for cand_bits in range(max_target, cur_bits, -1):
            if cand_bits not in _VALID_BITS or cand_bits <= cur_bits:
                continue
            cand_gs = _gs_for_mode(cand_bits, _OQ_DEFAULT_GROUP_SIZE)
            cand_mode = _mode_for_bits(cand_bits)
            cand_cost = _tensor_quantized_bytes(shape, cand_bits, cand_gs, cand_mode)
            delta = 8 * (cand_cost - cur_cost)
            if delta <= 0:
                continue
            next_bpw = (total_bits_f + delta) / total_params
            if next_bpw > hard_cap_bpw:
                continue
            boost_map[path] = {"bits": cand_bits, "group_size": cand_gs, "mode": cand_mode}
            total_bits_f += delta
            current_bpw = next_bpw
            break

    # Fallback: if still under target, boost non-expert tensors toward 8-bit
    # regardless of sensitivity tier. On large MoE models, non-expert weights
    # are <6% of params so every bit counts to reach the target bpw.
    if current_bpw < target_bpw:
        fallback_candidates = []
        for path, shape in named_shapes.items():
            if _is_routed_expert(path):
                continue
            cur = boost_map.get(path)
            if cur is None:
                continue
            cur_bits = cur["bits"]
            if cur_bits >= 8:
                continue
            cur_gs = _gs_for_mode(cur_bits, _OQ_DEFAULT_GROUP_SIZE)
            cur_mode = _mode_for_bits(cur_bits)
            cur_cost = _tensor_quantized_bytes(shape, cur_bits, cur_gs, cur_mode)
            layer_idx = _extract_layer_index(path)
            layer_score = float(layer_scores.get(str(layer_idx), 0.0))
            fallback_candidates.append((layer_score, path, shape, cur_bits, cur_cost))

        for _score, path, shape, cur_bits, cur_cost in sorted(
            fallback_candidates, key=lambda x: x[0], reverse=True
        ):
            for cand_bits in (8, 6, 5, 4, 3):
                if cand_bits <= cur_bits:
                    continue
                cand_gs = _gs_for_mode(cand_bits, _OQ_DEFAULT_GROUP_SIZE)
                cand_mode = _mode_for_bits(cand_bits)
                cand_cost = _tensor_quantized_bytes(shape, cand_bits, cand_gs, cand_mode)
                delta = 8 * (cand_cost - cur_cost)
                if delta <= 0:
                    continue
                next_bpw = (total_bits_f + delta) / total_params
                if next_bpw > hard_cap_bpw:
                    continue
                boost_map[path] = {"bits": cand_bits, "group_size": cand_gs, "mode": cand_mode}
                total_bits_f += delta
                current_bpw = next_bpw
                break
            if current_bpw >= target_bpw:
                break

    if boost_map:
        from collections import Counter
        bits_dist = Counter(v["bits"] for v in boost_map.values())
        layer_bits = {}
        for k, v in boost_map.items():
            idx = _extract_layer_index(k)
            label = f"L{idx}" if idx >= 0 else k.split(".")[-1]
            if label not in layer_bits:
                layer_bits[label] = v["bits"]
            else:
                layer_bits[label] = max(layer_bits[label], v["bits"])
        bits_summary = ", ".join(f"{b}bit×{c}" for b, c in sorted(bits_dist.items(), reverse=True))
        top_layers = sorted(layer_bits.items(), key=lambda x: -x[1])[:8]
        top_str = ", ".join(f"{l}={b}b" for l, b in top_layers)
        logger.info(f"  plan detail: {bits_summary} | top: {top_str}")

    return QuantPlan(
        boost_map=boost_map,
        effective_bpw=current_bpw,
        target_bpw=target_bpw,
        hard_cap_bpw=hard_cap_bpw,
    )


def resolve_output_name(
    model_name: str, oq_level: int, dtype: str = "bfloat16"
) -> str:
    """Generate output model name: strip existing quant suffixes, append oQ tag.

    Appends `-fp16` suffix when dtype is float16. bfloat16 is the default and
    produces no dtype suffix (backwards compatible).

    Examples:
        "Qwen3.5-122B-A10B" + 4 + bfloat16 -> "Qwen3.5-122B-A10B-oQ4"
        "Qwen3.5-122B-A10B" + 4 + float16  -> "Qwen3.5-122B-A10B-oQ4-fp16"
        "Qwen3.5-122B-A10B-oQ6-fp16" + 2 + bfloat16 -> "Qwen3.5-122B-A10B-oQ2"
    """
    pattern = re.compile(
        r"-(oQ[\d.]+e?|[0-9]+[_-]?bit|fp\d+|bf\d+)$",
        flags=re.IGNORECASE,
    )
    base = model_name
    while True:
        new = pattern.sub("", base)
        if new == base:
            break
        base = new
    level_str = f"{oq_level:g}"
    suffix = f"-oQ{level_str}"
    if dtype == "float16":
        suffix += "-fp16"
    return f"{base}{suffix}"



# ── Auto-discovery streaming sanitizer ──────────────────────────────────

class _TrackedTensor:
    """Fake tensor proxy that records shape, dtype, lineage, and transforms
    applied during a sanitize() dry run. Holds no GPU data."""

    __slots__ = ("shape", "ndim", "dtype", "sources", "transform", "axis")

    def __init__(self, shape, dtype, sources=None, transform="passthrough", axis=None):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = dtype
        self.sources = sources or []
        self.transform = transform
        self.axis = axis

    def _clone(self, shape=None, dtype=None, transform=None):
        return _TrackedTensor(
            shape if shape is not None else self.shape,
            dtype if dtype is not None else self.dtype,
            list(self.sources),
            transform if transform is not None else self.transform,
        )

    # Arithmetic — recipe is "fp8_dequant" for the whole sanitize block if weight came from FP8
    def __add__(self, other):
        return self._clone(transform="add")
    def __radd__(self, other):
        return self.__add__(other)
    def __sub__(self, other):
        return self._clone(transform="sub")
    def __mul__(self, other):
        return self._clone(transform="mul")
    def __rmul__(self, other):
        return self.__mul__(other)
    def __truediv__(self, other):
        return self._clone(transform="div")

    # Indexing: handle slice + None (broadcast) + tuple variants
    def __getitem__(self, idx):
        new_shape = list(self.shape)
        # Handle None-broadcasting like scale[:, None, :, None]
        if isinstance(idx, tuple):
            result_shape = []
            axis = 0
            for part in idx:
                if part is None:
                    result_shape.append(1)
                elif isinstance(part, slice):
                    if axis < len(new_shape):
                        result_shape.append(new_shape[axis])
                        axis += 1
                    else:
                        result_shape.append(1)
                else:
                    # int index → dimension removed
                    if axis < len(new_shape):
                        axis += 1
            while axis < len(new_shape):
                result_shape.append(new_shape[axis])
                axis += 1
            return _TrackedTensor(result_shape, self.dtype, list(self.sources), "slice")
        if isinstance(idx, slice):
            return self._clone(transform="slice")
        # int or other
        if new_shape:
            return _TrackedTensor(new_shape[1:], self.dtype, list(self.sources), "slice")
        return self._clone(transform="slice")

    def reshape(self, *new_shape):
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
            new_shape = tuple(new_shape[0])
        # Resolve any -1 using total element count
        total = 1
        for d in self.shape:
            total *= d
        resolved = []
        unknown_idx = -1
        known_prod = 1
        for i, d in enumerate(new_shape):
            if d == -1:
                unknown_idx = i
                resolved.append(-1)
            else:
                resolved.append(d)
                known_prod *= d
        if unknown_idx >= 0 and known_prod > 0:
            resolved[unknown_idx] = total // known_prod
        return _TrackedTensor(tuple(resolved), self.dtype, list(self.sources), "reshape")

    def astype(self, dtype):
        return _TrackedTensor(self.shape, dtype, list(self.sources), "astype")

    @property
    def T(self):
        return _TrackedTensor(tuple(reversed(self.shape)), self.dtype, list(self.sources), "transpose")

    @property
    def size(self):
        r = 1
        for d in self.shape:
            r *= d
        return r



def _streaming_fp8_dequant(lazy_index, scratch_dir=None, shard_bytes=4_000_000_000):
    """Streaming FP8 dequant with disk spill. For each (weight, weight_scale_inv)
    pair, materialize both, run block-scaled dequant to bf16, append to a
    scratch safetensors shard, and re-point the lazy index at it. Peak RAM
    bounded to one weight + scale + current shard buffer."""
    import tempfile, struct, json as _json
    import numpy as _np
    from pathlib import Path as _Path

    scale_suffix = "_scale_inv"
    scale_keys = [k for k in lazy_index._index if k.endswith(scale_suffix)]
    if not scale_keys:
        return 0

    if scratch_dir is None:
        scratch_dir = tempfile.mkdtemp(prefix="oq_fp8_dequant_")
    scratch_dir = _Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"FP8 dequant scratch dir: {scratch_dir}")

    shard_idx = 0
    shard_path = None
    shard_fh = None
    shard_header = {}
    shard_data_bytes = 0

    def _flush_shard():
        nonlocal shard_fh
        if shard_fh is None:
            return
        shard_fh.close()
        tmp_path = shard_path.with_suffix(".tmp")
        hdr_bytes = _json.dumps(shard_header, separators=(",", ":")).encode("utf-8")
        hlen = len(hdr_bytes)
        with open(shard_path, "rb") as src, open(tmp_path, "wb") as dst:
            dst.write(struct.pack("<Q", hlen))
            dst.write(hdr_bytes)
            dst.write(src.read())
        tmp_path.replace(shard_path)
        data_offset = 8 + hlen
        for k, meta in shard_header.items():
            start, end = meta["data_offsets"]
            lazy_index._index[k] = (
                str(shard_path), data_offset, start, end,
                tuple(meta["shape"]), meta["dtype"],
            )

    def _open_new_shard():
        nonlocal shard_idx, shard_path, shard_fh, shard_header, shard_data_bytes
        shard_idx += 1
        shard_path = scratch_dir / f"dequant-{shard_idx:04d}.safetensors"
        shard_fh = open(shard_path, "wb")
        shard_header = {}
        shard_data_bytes = 0

    _open_new_shard()
    count = 0

    for sk in scale_keys:
        wk = sk.replace(scale_suffix, "")
        if wk not in lazy_index._index:
            continue
        w_meta = lazy_index._index[wk]
        s_meta = lazy_index._index[sk]
        w_lt = _LazyTensor(w_meta[0], w_meta[1], w_meta[2], w_meta[3], w_meta[4], w_meta[5])
        s_lt = _LazyTensor(s_meta[0], s_meta[1], s_meta[2], s_meta[3], s_meta[4], s_meta[5])
        weight_u8 = w_lt[:]
        scale_inv = s_lt[:]
        mx.eval(weight_u8, scale_inv)

        weight = mx.from_fp8(weight_u8, dtype=mx.bfloat16)
        bs = 128
        m, n = weight.shape
        pad_bottom = (-m) % bs
        pad_side = (-n) % bs
        weight = mx.pad(weight, ((0, pad_bottom), (0, pad_side)))
        weight = weight.reshape(
            ((m + pad_bottom) // bs, bs, (n + pad_side) // bs, bs)
        )
        weight = (weight * scale_inv[:, None, :, None]).reshape(
            m + pad_bottom, n + pad_side
        )
        weight = weight[:m, :n].astype(mx.bfloat16)
        mx.eval(weight)

        raw = bytes(_np.asarray(weight.view(mx.uint16)))
        tensor_bytes = len(raw)
        if shard_data_bytes > 0 and shard_data_bytes + tensor_bytes > shard_bytes:
            _flush_shard()
            _open_new_shard()

        start = shard_data_bytes
        end = start + tensor_bytes
        shard_fh.write(raw)
        shard_data_bytes = end
        shard_header[wk] = {
            "dtype": "BF16",
            "shape": list(weight.shape),
            "data_offsets": [start, end],
        }
        lazy_index._index.pop(wk, None)
        lazy_index._index.pop(sk, None)
        del weight_u8, scale_inv, weight, raw
        mx.clear_cache()
        count += 1
        if count % 50 == 0:
            logger.info(f"FP8 dequant: {count} tensors processed")

    _flush_shard()
    logger.info(f"FP8 dequant complete: {count} tensors spilled to {scratch_dir}")
    return count


def _discover_sanitize_plan(sanitize_fn, lazy_index):
    """Run sanitize on _TrackedTensors to discover the key mapping and
    transforms without materializing any real data.

    Returns a dict: output_key -> {sources, transform, shape, axis}
    or None if discovery fails.
    """
    import mlx.core as mx

    # Build tracked dict mirroring the lazy index
    tracked = {}
    for k in lazy_index._index:
        meta = lazy_index._index[k]
        shape, dtype = meta[4], meta[5]
        tracked[k] = _TrackedTensor(shape, dtype, sources=[k])

    # Monkey-patch mx ops to work on tracked tensors
    _orig = {
        "stack": mx.stack,
        "concatenate": mx.concatenate,
        "split": mx.split,
        "eval": mx.eval,
        "clear_cache": mx.clear_cache,
        "synchronize": mx.synchronize,
        "moveaxis": mx.moveaxis,
        "transpose": mx.transpose,
        "from_fp8": getattr(mx, "from_fp8", None),
        "pad": getattr(mx, "pad", None),
    }

    def _fake_stack(tensors, axis=0):
        if tensors and isinstance(tensors[0], _TrackedTensor):
            n = len(tensors)
            base = list(tensors[0].shape)
            new_shape = base[:axis] + [n] + base[axis:]
            all_src = []
            for t in tensors:
                all_src.extend(t.sources)
            return _TrackedTensor(new_shape, tensors[0].dtype, all_src, "stack", axis=axis)
        return _orig["stack"](tensors, axis=axis)

    def _fake_concatenate(tensors, axis=0):
        if tensors and isinstance(tensors[0], _TrackedTensor):
            all_src = []
            for t in tensors:
                all_src.extend(t.sources)
            base = list(tensors[0].shape)
            base[axis] = sum(t.shape[axis] for t in tensors)
            return _TrackedTensor(base, tensors[0].dtype, all_src, "concatenate", axis=axis)
        return _orig["concatenate"](tensors, axis=axis)

    def _fake_split(tensor, indices_or_sections, axis=0):
        if isinstance(tensor, _TrackedTensor):
            if isinstance(indices_or_sections, int):
                n = indices_or_sections
                sz = tensor.shape[axis] // n
                parts = []
                for i in range(n):
                    sh = list(tensor.shape)
                    sh[axis] = sz
                    parts.append(_TrackedTensor(sh, tensor.dtype, list(tensor.sources), f"split_{i}_{n}", axis=axis))
                return parts
            # list of indices
            parts = []
            prev = 0
            idxs = list(indices_or_sections) + [tensor.shape[axis]]
            for i, idx in enumerate(idxs):
                sh = list(tensor.shape)
                sh[axis] = idx - prev
                parts.append(_TrackedTensor(sh, tensor.dtype, list(tensor.sources), f"split_{i}", axis=axis))
                prev = idx
            return parts
        return _orig["split"](tensor, indices_or_sections, axis=axis)

    def _fake_moveaxis(tensor, src_ax, dst_ax):
        if isinstance(tensor, _TrackedTensor):
            dims = list(range(tensor.ndim))
            dims.insert(dst_ax, dims.pop(src_ax))
            new_shape = tuple(tensor.shape[d] for d in dims)
            return _TrackedTensor(new_shape, tensor.dtype, list(tensor.sources), "moveaxis")
        return _orig["moveaxis"](tensor, src_ax, dst_ax)

    def _fake_transpose(tensor, axes=None):
        if isinstance(tensor, _TrackedTensor):
            if axes is None:
                axes = list(reversed(range(tensor.ndim)))
            new_shape = tuple(tensor.shape[a] for a in axes)
            return _TrackedTensor(new_shape, tensor.dtype, list(tensor.sources), "transpose")
        return _orig["transpose"](tensor, axes=axes)

    def _noop(*a, **kw): pass

    mx.stack = _fake_stack
    mx.concatenate = _fake_concatenate
    mx.split = _fake_split
    mx.eval = _noop
    mx.clear_cache = _noop
    mx.synchronize = _noop
    mx.moveaxis = _fake_moveaxis
    mx.transpose = _fake_transpose

    def _fake_from_fp8(x, dtype=None, **kw):
        if isinstance(x, _TrackedTensor):
            return _TrackedTensor(x.shape, dtype or x.dtype, list(x.sources), "from_fp8")
        return _orig["from_fp8"](x, dtype=dtype, **kw) if _orig["from_fp8"] else x

    def _fake_pad(x, pad_width, **kw):
        if isinstance(x, _TrackedTensor):
            new_shape = []
            for i, d in enumerate(x.shape):
                if i < len(pad_width):
                    lo, hi = pad_width[i] if isinstance(pad_width[i], (tuple, list)) else (pad_width[i], pad_width[i])
                    new_shape.append(d + lo + hi)
                else:
                    new_shape.append(d)
            return _TrackedTensor(new_shape, x.dtype, list(x.sources), "pad")
        return _orig["pad"](x, pad_width, **kw) if _orig["pad"] else x

    if _orig["from_fp8"] is not None:
        mx.from_fp8 = _fake_from_fp8
    if _orig["pad"] is not None:
        mx.pad = _fake_pad

    try:
        result = sanitize_fn(tracked)
    finally:
        for name, fn in _orig.items():
            setattr(mx, name, fn)

    # Extract plan
    plan = {}
    for k, v in result.items():
        if isinstance(v, _TrackedTensor):
            plan[k] = {
                "sources": v.sources,
                "transform": v.transform,
                "shape": v.shape,
                "axis": v.axis,
            }
        else:
            # sanitize returned a real value (rare — e.g. a scalar override)
            plan[k] = {
                "sources": [],
                "transform": "literal",
                "shape": getattr(v, "shape", ()),
                "axis": None,
                "value": v,
            }

    return plan


class _DiscoveredPlan:
    """Dict-like wrapper that materializes tensors one at a time using
    a plan discovered by _discover_sanitize_plan. Supports chunked
    stacking for huge MoE expert tensors."""

    _STACK_CHUNK = 16  # experts per chunk during materialization

    def __init__(self, plan, lazy_index):
        self._plan = plan       # output_key -> {sources, transform, ...}
        self._lazy = lazy_index
        self._cache = {}        # output_key -> mx.array (for multi-consumer sources)

    def keys(self):
        return self._plan.keys()

    def __len__(self):
        return len(self._plan)

    def __contains__(self, k):
        return k in self._plan

    def __iter__(self):
        return iter(self._plan)

    def items(self):
        # Yield (key, shape_proxy) for the quantize loop shape inspection
        class _SP:
            __slots__ = ("shape", "ndim")
            def __init__(self, sh):
                self.shape = tuple(sh)
                self.ndim = len(self.shape)
        return ((k, _SP(info["shape"])) for k, info in self._plan.items())

    def nbytes(self):
        return self._lazy.nbytes()

    def _materialize_source(self, src_key):
        """Load a single source tensor from the lazy index."""
        meta = self._lazy._index.get(src_key)
        if meta is None:
            raise KeyError(f"source tensor {src_key!r} not in lazy index")
        sf_path, data_offset, start, end, shape, dtype = meta
        # Scalars (0-dim tensors) need special handling
        if len(shape) == 0:
            import numpy as _np
            with open(sf_path, "rb") as f:
                f.seek(data_offset + start)
                raw = f.read(end - start)
            lt_tmp = _LazyTensor(sf_path, data_offset, start, end, (1,), dtype)
            np_view = _np.frombuffer(raw, dtype=lt_tmp._np_view_dtype())
            arr = mx.array(np_view).view(lt_tmp._mlx_dtype()).reshape(())
            mx.eval(arr)
            return arr
        lt = _LazyTensor(sf_path, data_offset, start, end, shape, dtype)
        arr = lt[:]
        mx.eval(arr)
        return arr

    def pop(self, key, *default):
        if key not in self._plan:
            if default:
                return default[0]
            raise KeyError(key)

        info = self._plan.pop(key)
        transform = info["transform"]
        sources = info["sources"]

        if transform == "literal":
            return info["value"]

        if transform == "passthrough" and len(sources) == 1:
            arr = self._materialize_source(sources[0])
            return arr

        if transform == "stack":
            # Chunked stacking to bound peak memory
            axis = info.get("axis", 0)
            chunk = self._STACK_CHUNK
            partials = []
            for base in range(0, len(sources), chunk):
                piece = []
                for src in sources[base:base + chunk]:
                    piece.append(self._materialize_source(src))
                stk = mx.stack(piece, axis=axis)
                mx.eval(stk)
                del piece
                mx.clear_cache()
                partials.append(stk)
            if len(partials) == 1:
                return partials[0]
            result = mx.concatenate(partials, axis=axis)
            mx.eval(result)
            del partials
            mx.clear_cache()
            return result

        if transform == "concatenate":
            axis = info.get("axis", 0)
            parts = [self._materialize_source(src) for src in sources]
            result = mx.concatenate(parts, axis=axis)
            mx.eval(result)
            del parts
            mx.clear_cache()
            return result

        # NOTE: discovery records transform TYPE but not parameters.
        # These hardcoded values cover all current mlx-lm/mlx-vlm sanitize
        # patterns. If a future model uses different parameters, discovery
        # will fail and the eager sanitize fallback handles it safely.
        if transform == "add":
            arr = self._materialize_source(sources[0])
            return arr + 1.0  # norm weight += 1.0 pattern

        if transform == "transpose":
            arr = self._materialize_source(sources[0])
            return mx.transpose(arr)  # full axis reverse

        if transform == "moveaxis":
            arr = self._materialize_source(sources[0])
            return mx.moveaxis(arr, 2, 1)  # conv1d weight permute

        if "split_" in transform:
            # split_N_M means take part N of M
            parts = transform.split("_")
            arr = self._materialize_source(sources[0])
            axis = info.get("axis", 0)
            if len(parts) == 3:  # split_idx_total
                idx, total = int(parts[1]), int(parts[2])
                chunks = mx.split(arr, total, axis=axis)
                result = chunks[idx]
                mx.eval(result)
                del arr, chunks
                mx.clear_cache()
                return result
            # split_idx (index-based split) — less common
            return arr

        # Fallback: just load first source
        if sources:
            return self._materialize_source(sources[0])
        raise ValueError(f"cannot materialize {key!r}: transform={transform}, no sources")



def validate_quantizable(config: dict) -> bool:
    """Check if a model config indicates it can be quantized.

    Models with 'quantization' key (mlx-lm quantized) are excluded.
    Models with 'quantization_config' are excluded UNLESS they are native FP8
    (e.g. MiniMax, DeepSeek) which are full-precision models stored in FP8 format.
    """
    if "quantization" in config:
        return False
    if "quantization_config" in config:
        qc = config["quantization_config"]
        if isinstance(qc, dict) and qc.get("quant_method") == "fp8":
            return True
        return False
    return True


def make_predicate(config: dict, oq_level: int = 4) -> Callable:
    """Create a quant_predicate closure for mlx-lm's quantize_model."""

    def predicate(path: str, module) -> Union[bool, dict]:
        return universal_quant_predicate(path, module, config, oq_level)

    return predicate


def estimate_bpw_and_size(model_path: str, oq_level: int, group_size: int = 64) -> dict:
    """Calculate precise effective bpw and output size by scanning actual tensors.

    Applies the universal predicate to each tensor to determine its bit width,
    then computes weighted average bpw and estimated output size.

    Args:
        model_path: Path to source model directory.
        oq_level: Target oQ level (base bits).
        group_size: Quantization group size.

    Returns:
        Dict with effective_bpw, output_size_bytes, output_size_formatted.
    """
    source = Path(model_path)
    config_path = source / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    weight_files = sorted(source.glob("*.safetensors"))
    if not weight_files:
        return {"effective_bpw": float(oq_level), "output_size_bytes": 0,
                "output_size_formatted": "?"}

    # Build budget plan for accurate estimate (position-based sensitivity)
    _level_targets = _bpw_targets_for_level(oq_level)
    if _level_targets is not None:
        config["_oq_use_budget_plan"] = True
        tc = config.get("text_config", {})
        num_layers = (
            config.get("num_hidden_layers")
            or tc.get("num_hidden_layers", 32)
        )
        pos_sens = {}
        for i in range(num_layers):
            if i < num_layers // 8 or i >= 7 * num_layers // 8:
                pos_sens[str(i)] = 0.05
            elif i < num_layers // 4 or i >= 3 * num_layers // 4:
                pos_sens[str(i)] = 0.02
            else:
                pos_sens[str(i)] = 0.01
        config["_oq_sensitivity_map"] = pos_sens

        named_shapes = {}
        for sf_path in weight_files:
            shard = mx.load(str(sf_path), return_metadata=False)
            for name, tensor in shard.items():
                ns = _collect_named_weight_shapes_from_weights({name: tensor})
                named_shapes.update(ns)
            del shard
        plan = _build_quant_plan(
            named_shapes, config, oq_level,
            target_bpw=_level_targets[0], hard_cap_bpw=_level_targets[1],
        )
        config["_oq_boost_map"] = plan.boost_map
    else:
        config["_oq_boost_map"] = {}

    total_params = 0
    total_weighted_bits = 0
    total_output_bytes = 0

    for sf_path in weight_files:
        shard = mx.load(str(sf_path), return_metadata=False)
        for name, tensor in shard.items():
            shape = tensor.shape
            n_elements = 1
            for d in shape:
                n_elements *= d

            if not _should_quantize_tensor(name, shape):
                total_params += n_elements
                total_weighted_bits += n_elements * 16
                total_output_bytes += n_elements * 2
                continue

            if _should_skip_tensor(name):
                continue

            bits, gs, _mode = _get_predicate_bits(name, config, oq_level, group_size)
            if bits is None:
                total_params += n_elements
                total_weighted_bits += n_elements * 16
                total_output_bytes += n_elements * 2
            else:
                total_params += n_elements
                if len(shape) >= 2:
                    n_groups = (shape[-1] + gs - 1) // gs
                    rows = n_elements // max(shape[-1], 1)
                    weight_bytes = (n_elements * bits + 7) // 8
                    if _mode == "mxfp4":
                        bytes_per_group = 1
                    elif _mode == "mxfp8":
                        bytes_per_group = 2
                    else:
                        bytes_per_group = 4
                    overhead_bytes = rows * n_groups * bytes_per_group
                    tensor_bytes = weight_bytes + overhead_bytes
                    total_output_bytes += tensor_bytes
                    total_weighted_bits += tensor_bytes * 8
                else:
                    total_output_bytes += n_elements * 2
                    total_weighted_bits += n_elements * 16

        del shard

    for k in ("_oq_use_budget_plan", "_oq_boost_map", "_oq_sensitivity_map"):
        config.pop(k, None)

    effective_bpw = total_weighted_bits / max(total_params, 1)

    # oQ3.5 correction: expert down_proj 3→4 bit not visible in pre-sanitize scan
    # (fused tensors like gate_up_proj don't have .weight suffix).
    # After sanitize, down_proj is ~31% of routed expert params → ~10% of total.
    # +1 bit for 10% of params ≈ +0.1 bpw.
    if oq_level == 3.5:
        effective_bpw += 0.3
        total_output_bytes = int(effective_bpw * total_params / 8)

    source_total = sum(
        sf.stat().st_size for sf in source.glob("*.safetensors")
    )
    num_shards = len(list(source.glob("*.safetensors")))
    max_shard_size = max(
        (sf.stat().st_size for sf in source.glob("*.safetensors")),
        default=0,
    )

    streaming_peak = int(source_total * 1.5) + 5 * 1024**3

    return {
        "effective_bpw": round(effective_bpw, 2),
        "output_size_bytes": total_output_bytes,
        "output_size_formatted": _format_size(total_output_bytes),
        "memory_streaming_bytes": streaming_peak,
        "memory_streaming_formatted": _format_size(streaming_peak),
    }


def estimate_memory(source_size_bytes: int) -> dict:
    """Estimate peak memory for quantization.

    This is a rough estimate used before precise calculation is available.
    The /api/oq/estimate endpoint provides precise values per tensor.

    Streaming: source (mmap) + 5GB output buffer + sanitize overhead
    """
    peak = source_size_bytes + 6 * 1024**3
    return {"peak_bytes": peak, "peak_formatted": _format_size(peak)}


def _format_size(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024**3:.1f} GB"


_MAX_SHARD_BYTES = 5_000_000_000

_SKIP_QUANT_PATTERNS = (
    "layernorm", "rmsnorm", "norm.weight", "norm.bias",
    "ln_", "layer_norm",
)


def _should_skip_tensor(name: str) -> bool:
    """Check if a tensor should be completely excluded from output.

    These tensors are removed by mlx-lm sanitize() and should not be saved.
    """
    if ".mtp." in name or name.startswith("mtp."):
        return True
    return False


def _should_quantize_tensor(name: str, shape: tuple) -> bool:
    """Check if a tensor should be quantized based on name and shape."""
    if len(shape) < 2:
        return False
    name_lower = name.lower()
    if any(p in name_lower for p in _SKIP_QUANT_PATTERNS):
        return False
    if name.endswith(".bias"):
        return False
    return True


def _build_model_sanitizer(config: dict):
    """Build a sanitize function from the model class.

    For VLM models, uses mlx-vlm's model class (preserves vision weights).
    For LLM models, uses mlx-lm's model class.

    Returns:
        A function that takes a dict of weights and returns sanitized weights,
        or None if the model class can't be loaded.
    """
    architectures = config.get("architectures", [])
    is_vlm = any("ForConditionalGeneration" in a for a in architectures)

    if is_vlm:
        try:
            from mlx_vlm.utils import get_model_and_args, sanitize_weights

            model_module, _ = get_model_and_args(config)
            model_config_cls = model_module.ModelConfig
            model_config = model_config_cls.from_dict(config)

            vision_config = model_config.vision_config
            if isinstance(vision_config, dict):
                vision_config = model_module.VisionConfig.from_dict(vision_config)
            text_config = model_config.text_config
            if isinstance(text_config, dict):
                text_config = model_module.TextConfig.from_dict(text_config)

            model_config.vision_config = vision_config
            model_config.text_config = text_config

            def _vlm_sanitize(weights):
                class _Proxy:
                    audio_tower = None
                proxy = _Proxy()
                proxy.config = model_config
                w = model_module.Model.sanitize(proxy, weights)

                w = sanitize_weights(
                    model_module.VisionModel, w, vision_config
                )
                w = sanitize_weights(
                    model_module.LanguageModel, w, text_config
                )
                return w

            logger.info(
                f"Using mlx-vlm full sanitize chain for "
                f"{model_module.Model.__name__} (preserves vision weights)"
            )
            return _vlm_sanitize
        except Exception as e:
            logger.debug(f"mlx-vlm sanitizer not available: {e}")

    try:
        from mlx_lm.utils import _get_classes

        model_class, model_args_class = _get_classes(config)
        args = model_args_class.from_dict(config)
        model = model_class(args)

        if hasattr(model, "sanitize"):
            logger.info(
                f"Using mlx-lm {model_class.__name__}.sanitize() "
                f"for weight transformation"
            )
            return model.sanitize
    except Exception as e:
        logger.warning(f"Could not build model sanitizer: {e}")

    return None


def _build_non_quantizable_set(config: dict) -> set:
    """Find module paths with 2D weights that lack to_quantized() support.

    Loads the model class (without real weights) and walks the module tree.
    Modules like ScaledLinear in Gemma 4 have a weight attribute but no
    to_quantized(), so they cannot be loaded as QuantizedLinear after
    quantization. Returns empty set if the model class cannot be loaded.
    """
    try:
        from mlx_lm.utils import _get_classes

        model_class, model_args_class = _get_classes(config)
        args = model_args_class.from_dict(config)
        model = model_class(args)

        result = set()
        for path, module in tree_flatten(
            model.leaf_modules(), is_leaf=nn.Module.is_module
        ):
            if hasattr(module, "weight") and not hasattr(module, "to_quantized"):
                if getattr(module.weight, "ndim", 0) >= 2:
                    result.add(_normalize_quant_path(path))

        if result:
            logger.info(
                "Non-quantizable modules (no to_quantized): "
                + ", ".join(sorted(result))
            )
        return result
    except Exception as e:
        logger.debug(f"Could not build non-quantizable set: {e}")
        return set()


def _get_predicate_bits(tensor_name: str, config: dict, oq_level: int,
                        group_size: int) -> tuple:
    """Get quantization bits, group_size, and mode for a tensor.

    Returns:
        (bits, group_size, mode) or (None, None, None) if not quantized.
    """
    base_bits = _base_bits_for_level(oq_level)

    result = universal_quant_predicate(tensor_name, None, config, oq_level)
    if result is False:
        return None, None, None
    if isinstance(result, dict):
        bits = result.get("bits", base_bits)
        gs = result.get("group_size", group_size)
        mode = result.get("mode", _mode_for_bits(bits))
        return bits, gs, mode
    return base_bits, _gs_for_mode(base_bits, group_size), _mode_for_bits(base_bits)


def _mode_for_bits(bits: int) -> str:
    """Select quantization mode. Always affine to minimize kernel combos."""
    return "affine"


def _gs_for_mode(bits: int, default_gs: int) -> int:
    """Get group_size. Always default to minimize kernel combos."""
    return default_gs



# --- chunked-quantize helpers (added for Qwen3.5-397B) ---------------------
import struct as _struct
import numpy as _np

def _metal_max_buffer_bytes() -> int:
    try:
        info = mx.device_info()
    except AttributeError:
        try:
            info = mx.metal.device_info()
        except Exception:
            return 1 << 30
    except Exception:
        return 1 << 30
    return int(info.get("max_buffer_length", 1 << 30))

_METAL_MAX_BUFFER = _metal_max_buffer_bytes()
_QUANTIZE_CHUNK_BYTES = max(1 << 20, _METAL_MAX_BUFFER // 4)
_LOAD_CHUNK_BYTES     = max(1 << 20, _METAL_MAX_BUFFER // 2)


class _LazyTensorIndex:
    _DTYPE_BYTES = {"BF16":2,"F16":2,"F32":4,"F64":8,"I8":1,"U8":1,
                    "I16":2,"U16":2,"I32":4,"U32":4,"I64":8,"U64":8,"BOOL":1,
                    "F8_E4M3":1,"F8_E5M2":1}

    def __init__(self, weight_files):
        self._index = {}
        for sf_path in weight_files:
            with open(sf_path, "rb") as f:
                hlen = _struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(hlen))
                data_offset = 8 + hlen
                for k, meta in header.items():
                    if k == "__metadata__":
                        continue
                    self._index[k] = (sf_path, data_offset,
                                      meta["data_offsets"][0], meta["data_offsets"][1],
                                      tuple(meta["shape"]), meta["dtype"])

    def keys(self):
        if hasattr(self, "_overrides"):
            return list(self._index.keys()) + list(self._overrides.keys())
        return self._index.keys()
    def __len__(self):
        n = len(self._index)
        if hasattr(self, "_overrides"): n += len(self._overrides)
        return n
    def __contains__(self, k):
        if k in self._index: return True
        return hasattr(self, "_overrides") and k in self._overrides
    def __iter__(self):
        yield from self._index
        if hasattr(self, "_overrides"):
            for k in self._overrides:
                if k not in self._index:
                    yield k
    def nbytes(self):         return sum(e - s for _,_,s,e,_,_ in self._index.values())

    def __getitem__(self, key):
        if key not in self._index:
            raise KeyError(key)
        sf_path, data_offset, start, end, shape, dtype = self._index[key]
        lt = _LazyTensor(sf_path, data_offset, start, end, shape, dtype)
        # Materialize as a real mx.array; _load_rows handles chunking internally.
        arr = lt[:]
        return arr

    def items(self):
        # Materialize tensors one at a time so sanitize sees real arrays.
        for k in list(self._index.keys()):
            yield k, self[k]
            mx.clear_cache()
        if hasattr(self, "_overrides"):
            for k, v in self._overrides.items():
                yield k, v

    def get(self, key, default=None):
        if key in self._index:
            return self[key]
        return default

    def __setitem__(self, key, value):
        # Sanitize may write back transformed tensors. Store as in-memory
        # override; remove from lazy index so we don't re-load from disk.
        if not hasattr(self, "_overrides"):
            self._overrides = {}
        self._overrides[key] = value
        self._index.pop(key, None)

    def __delitem__(self, key):
        self._index.pop(key, None)
        if hasattr(self, "_overrides"):
            self._overrides.pop(key, None)

    def update(self, other):
        if hasattr(other, "items"):
            for k, v in other.items():
                self[k] = v
        else:
            for k, v in other:
                self[k] = v

    def pop(self, key, *default):
        if hasattr(self, "_overrides") and key in self._overrides:
            return self._overrides.pop(key)
        if key not in self._index:
            if default: return default[0]
            raise KeyError(key)
        sf_path, data_offset, start, end, shape, dtype = self._index.pop(key)
        lt = _LazyTensor(sf_path, data_offset, start, end, shape, dtype)
        arr = lt[:]
        mx.eval(arr)
        return arr


class _LazyTensor:
    def __init__(self, sf_path, data_offset, start, end, shape, dtype):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self._sf_path = sf_path
        self._data_offset = data_offset
        self._start = start
        self._end = end
        self._dtype = dtype
        self._bpe = _LazyTensorIndex._DTYPE_BYTES.get(dtype, 2)
        self._epr = 1
        for d in self.shape[1:]:
            self._epr *= d
        self._bpr = self._epr * self._bpe

    @property
    def size(self):
        s = 1
        for d in self.shape: s *= d
        return s

    @property
    def nbytes(self):
        return self._end - self._start

    def _mlx_dtype(self):
        # FP8 variants are loaded as uint8; sanitize calls mx.from_fp8 to convert
        if self._dtype in ("F8_E4M3", "F8_E5M2"):
            return mx.uint8
        return {"BF16":mx.bfloat16,"F16":mx.float16,"F32":mx.float32}.get(self._dtype, mx.bfloat16)

    def _np_view_dtype(self):
        if self._dtype in ("F8_E4M3", "F8_E5M2"):
            return _np.uint8
        if self._bpe == 2: return _np.uint16
        return _np.dtype({"F32":"<f4","F64":"<f8","I32":"<i4","I64":"<i8"}.get(self._dtype, "<u2"))

    def _load_rows(self, r0, r1):
        n = r1 - r0
        if n <= 0:
            return mx.zeros((0, *self.shape[1:]), dtype=self._mlx_dtype())
        b0 = self._start + r0 * self._bpr
        b1 = self._start + r1 * self._bpr
        with open(self._sf_path, "rb") as f:
            f.seek(self._data_offset + b0)
            raw = f.read(b1 - b0)
        arr = _np.frombuffer(raw, dtype=self._np_view_dtype())
        chunk_shape = (n, *self.shape[1:])
        # Two ceilings: device buffer bytes, and MLX's int32 element count.
        _MLX_MAX_ELEMS = 1 << 30
        max_rows_bytes = max(1, _LOAD_CHUNK_BYTES // max(self._bpr, 1))
        max_rows_elems = max(1, _MLX_MAX_ELEMS // max(self._epr, 1))
        max_rows = min(max_rows_bytes, max_rows_elems)
        dt = self._mlx_dtype()
        if n <= max_rows:
            t = mx.array(arr).view(dt).reshape(chunk_shape)
            mx.eval(t)
            return t
        parts = []
        epc = max_rows * self._epr
        for s in range(0, arr.size, epc):
            sub = arr[s:s+epc]
            sr = sub.size // self._epr
            t = mx.array(sub).view(dt).reshape((sr, *self.shape[1:]))
            mx.eval(t)
            parts.append(t)
            mx.clear_cache()
        result = mx.concatenate(parts, axis=0)
        mx.eval(result)
        return result

    def __getitem__(self, idx):
        if len(self.shape) == 0:
            raise IndexError(
                "0-dim _LazyTensor cannot be indexed; caller should use "
                "_materialize_source scalar path"
            )
        if isinstance(idx, tuple):
            return self._load_rows(0, self.shape[0])[idx]
        if isinstance(idx, slice):
            start = idx.start or 0
            stop = self.shape[0] if idx.stop is None else idx.stop
            return self._load_rows(start, stop)
        return self._load_rows(idx, idx + 1)


def _row_chunks(t, max_elems):
    rows = t.shape[0]
    if rows == 0: return
    epr = max(1, t.size // rows)
    rpc = max(1, max_elems // epr)
    for r0 in range(0, rows, rpc):
        r1 = min(rows, r0 + rpc)
        if isinstance(t, _LazyTensor):
            chunk = t._load_rows(r0, r1)
        else:
            chunk = t[r0:r1]
            mx.eval(chunk)
        yield chunk


def _quantize_chunked(w, group_size, bits, mode):
    _MLX_MAX_ELEMS = 1 << 30
    max_elems = max(group_size, min(_QUANTIZE_CHUNK_BYTES // 2, _MLX_MAX_ELEMS))
    if not isinstance(w, _LazyTensor) and w.size <= max_elems:
        qw, scales, *rest = mx.quantize(w, group_size=group_size, bits=bits, mode=mode)
        return qw, scales, (rest[0] if rest else None)
    orig = tuple(w.shape)
    qws, scs, bis = [], [], []
    for chunk in _row_chunks(w, max_elems):
        flat = chunk.reshape(-1, chunk.shape[-1])
        mx.eval(flat)
        cqw, csc, *crest = mx.quantize(flat, group_size=group_size, bits=bits, mode=mode)
        mx.eval(cqw, csc)
        qws.append(cqw); scs.append(csc)
        if crest: bis.append(crest[0])
        mx.synchronize(); mx.clear_cache()
    qw = mx.concatenate(qws, axis=0)
    scales = mx.concatenate(scs, axis=0)
    biases = mx.concatenate(bis, axis=0) if bis else None
    mx.eval(qw, scales)
    flat_rows = 1
    for d in orig[:-1]: flat_rows *= d
    if qw.shape[0] == flat_rows and len(orig) > 2:
        qw = qw.reshape(*orig[:-1], -1)
        scales = scales.reshape(*orig[:-1], -1)
        if biases is not None:
            biases = biases.reshape(*orig[:-1], -1)
    return qw, scales, biases
# --- end chunked-quantize helpers ---


def quantize_oq_streaming(
    model_path: str,
    output_path: str,
    oq_level: int,
    group_size: int = 64,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    text_only: bool = False,
    target_bpw: float | None = None,
    hard_cap_bpw: float | None = None,
    sensitivity_model_path: str = "",
    dtype: str = "bfloat16",
) -> None:
    """Tensor-by-tensor quantization. Memory: ~3-4GB regardless of model size.

    Reads tensors one at a time from safetensors, quantizes with the universal
    predicate, and writes output shards. Never loads the full model.

    Args:
        model_path: Path to source model directory.
        output_path: Path for output (must not exist).
        oq_level: Quantization level (2, 3, 4, 6, or 8).
        group_size: Default quantization group size.
        progress_callback: Optional fn(phase_name, progress_pct) for updates.
        text_only: Skip vision encoder weights for VLM models.
        dtype: Target fp dtype for non-quantized weights and quant scales/biases.
            Must be "bfloat16" (default) or "float16". float16 yields ~20%
            faster prefill on M1/M2 Apple Silicon (native fp16 support).
    """
    if oq_level not in OQ_LEVELS:
        raise ValueError(
            f"Invalid oQ level {oq_level}. Must be one of {sorted(OQ_LEVELS)}"
        )
    if dtype not in OQ_DTYPES:
        raise ValueError(
            f"Invalid dtype {dtype!r}. Must be one of {OQ_DTYPES}"
        )
    target_dtype = mx.bfloat16 if dtype == "bfloat16" else mx.float16

    source = Path(model_path)
    output = Path(output_path)
    if output.exists():
        raise ValueError(f"Output directory already exists: {output_path}")

    output.mkdir(parents=True, exist_ok=True)
    cb = progress_callback or (lambda phase, pct: None)

    config_path = source / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    config["_oq_use_budget_plan"] = oq_level in _OQ_BPW_TARGETS

    cb("loading", 5.0)

    weight_files = sorted(source.glob("*.safetensors"))
    if not weight_files:
        raise ValueError(f"No .safetensors files found in {model_path}")

    cb("loading", 8.0)

    all_weights = _LazyTensorIndex(weight_files)

    logger.info(
        f"oQ{oq_level:g} streaming: {len(all_weights)} tensors in "
        f"{len(weight_files)} shards"
    )

    cb("loading", 12.0)

    sanitize_fn = _build_model_sanitizer(config)
    if sanitize_fn is not None:
        # Try discovery-based streaming sanitize first (works for any model,
        # bounds peak memory by materializing one tensor at a time)
        # Check if source contains FP8 weights — discovery can't capture the
        # multi-op dequant chain (from_fp8 + pad + reshape + mul + slice + astype)
        # as a single replayable transform. Fall back to eager sanitize for those.
        has_fp8 = False
        if hasattr(all_weights, "_index"):
            for _k, _meta in all_weights._index.items():
                if _meta[5] in ("F8_E4M3", "F8_E5M2"):
                    has_fp8 = True
                    break
        if has_fp8:
            logger.info(
                f"oQ{oq_level:g}: FP8 source detected, streaming dequant to scratch shards"
            )
            try:
                n_deq = _streaming_fp8_dequant(all_weights)
                logger.info(f"oQ{oq_level:g}: FP8 dequant complete ({n_deq} tensors)")
                all_weights = sanitize_fn(all_weights)
                logger.info(f"oQ{oq_level:g}: sanitize applied, {len(all_weights)} tensors")
            except Exception as e:
                import traceback; traceback.print_exc()
                logger.warning(f"FP8 sanitize failed ({e}), aborting")
                raise
        else:
            try:
                plan = _discover_sanitize_plan(sanitize_fn, all_weights)
                all_weights = _DiscoveredPlan(plan, all_weights)
                logger.info(
                    f"oQ{oq_level:g}: discovered streaming sanitize plan, "
                    f"{len(all_weights)} output tensors"
                )
            except Exception as e:
                logger.warning(
                    f"Streaming discovery failed ({e}), falling back to eager sanitize"
                )
                try:
                    all_weights = sanitize_fn(all_weights)
                    logger.info(f"oQ{oq_level:g}: eager sanitize applied, {len(all_weights)} tensors")
                except Exception as e2:
                    logger.warning(f"Sanitize failed ({e2}), using original names")

    config["_oq_non_quantizable"] = _build_non_quantizable_set(config)

    cb("loading", 15.0)

    if sensitivity_model_path:
        logger.info(f"oQ{oq_level:g}: measuring sensitivity via proxy model")
        sensitivity_map = _measure_sensitivity_from_quantized_model(
            sensitivity_model_path, config, oq_level,
            num_samples=128, seq_length=256,
        )
    else:
        logger.info(f"oQ{oq_level:g}: measuring layer sensitivity for streaming path")
        sensitivity_map = _measure_sensitivity(
            model_path, config, oq_level,
            num_samples=128, seq_length=256,
        )
    if sensitivity_map:
        config["_oq_sensitivity_map"] = {
            str(k): v for k, v in sensitivity_map.items()
        }
        logger.info(f"oQ{oq_level:g}: sensitivity applied ({len(sensitivity_map)} layers)")

    named_shapes = _collect_named_weight_shapes_from_weights(all_weights)
    if text_only:
        named_shapes = {
            k: v for k, v in named_shapes.items() if not _is_vision_tensor(k)
        }
    _level_targets = _bpw_targets_for_level(oq_level)
    if _level_targets is not None:
        _t = target_bpw if target_bpw is not None else _level_targets[0]
        _c = hard_cap_bpw if hard_cap_bpw is not None else _level_targets[1]
        plan = _build_quant_plan(
            named_shapes, config, oq_level, target_bpw=_t, hard_cap_bpw=_c,
        )
        config["_oq_boost_map"] = plan.boost_map
        logger.info(
            f"oQ{oq_level:g}: quant plan -> {plan.effective_bpw:.2f} bpw "
            f"with {len(plan.boost_map)} boosts"
        )
    else:
        config["_oq_boost_map"] = {}

    cb("loading", 20.0)

    tensor_names = list(all_weights.keys())
    total_tensors = len(tensor_names)
    out_shard_data = {}
    out_shard_idx = 0
    weight_map = {}
    base_bits = _base_bits_for_level(oq_level)
    base_mode = _mode_for_bits(base_bits)
    base_gs = _gs_for_mode(base_bits, group_size)
    quantization_config = {"group_size": base_gs, "bits": base_bits, "mode": base_mode}
    per_layer_config = {}
    start_time = _time.monotonic()

    total_bytes = sum(sf.stat().st_size for sf in source.glob("*.safetensors"))
    processed_bytes = 0

    for i, tensor_name in enumerate(tensor_names):
        w_mx = all_weights.pop(tensor_name)
        if isinstance(w_mx, _LazyTensor):
            w_mx = w_mx[:]
        tensor_bytes = w_mx.nbytes
        shape = w_mx.shape

        if text_only and _is_vision_tensor(tensor_name):
            del w_mx
            processed_bytes += tensor_bytes
            continue

        if _should_quantize_tensor(tensor_name, shape):
            bits, gs, qmode = _get_predicate_bits(
                tensor_name, config, oq_level, group_size
            )

            if bits is not None and len(shape) >= 2 and shape[-1] % gs == 0:
                # Cast to target dtype before quantize: scales/biases inherit
                # the input dtype, which drives inference speed on Apple
                # Silicon (M1/M2 prefer float16, M3/M4 handle both).
                if (
                    mx.issubdtype(w_mx.dtype, mx.floating)
                    and w_mx.dtype != target_dtype
                ):
                    w_mx = w_mx.astype(target_dtype)
                qw, scales, biases = _quantize_chunked(w_mx, gs, bits, qmode)

                base = tensor_name
                if base.endswith(".weight"):
                    base = base[:-7]

                out_shard_data[f"{base}.weight"] = qw
                out_shard_data[f"{base}.scales"] = scales
                if biases is not None:
                    out_shard_data[f"{base}.biases"] = biases

                base_qmode = _mode_for_bits(base_bits)
                base_gs_check = _gs_for_mode(base_bits, group_size)
                if bits != base_bits or gs != base_gs_check or qmode != base_qmode:
                    layer_cfg = {"bits": bits, "group_size": gs}
                    layer_cfg["mode"] = qmode
                    per_layer_config[base] = layer_cfg
            else:
                if (
                    mx.issubdtype(w_mx.dtype, mx.floating)
                    and w_mx.dtype != target_dtype
                ):
                    w_mx = w_mx.astype(target_dtype)
                out_shard_data[tensor_name] = w_mx
        else:
            if (
                mx.issubdtype(w_mx.dtype, mx.floating)
                and w_mx.dtype != target_dtype
            ):
                w_mx = w_mx.astype(target_dtype)
            out_shard_data[tensor_name] = w_mx

        del w_mx

        current_bytes = sum(v.nbytes for v in out_shard_data.values())
        if current_bytes >= _MAX_SHARD_BYTES:
            shard_name = f"model-{out_shard_idx + 1:05d}-of-PLACEHOLDER.safetensors"
            shard_path = output / shard_name
            mx.save_safetensors(str(shard_path), out_shard_data, metadata={"format": "mlx"})
            for k in out_shard_data:
                weight_map[k] = shard_name
            out_shard_idx += 1
            out_shard_data = {}
            mx.synchronize()
            mx.clear_cache()
            logger.info(f"oQ{oq_level:g}: wrote output shard {out_shard_idx}")

        processed_bytes += tensor_bytes
        elapsed = _time.monotonic() - start_time
        frac = processed_bytes / max(total_bytes, 1)
        pct = 15.0 + frac * 75.0
        if elapsed > 1.0 and frac > 0.01:
            eta_secs = elapsed / frac * (1 - frac)
            mins = int(eta_secs // 60)
            secs = int(eta_secs % 60)
            cb(
                f"quantizing_eta|{int(frac * 100)}|100|{mins}:{secs:02d}",
                pct,
            )
        else:
            cb(f"quantizing_eta|{int(frac * 100)}|100|", pct)

    del all_weights
    mx.synchronize()
    mx.clear_cache()

    if out_shard_data:
        total_shards = out_shard_idx + 1
        if total_shards == 1:
            shard_name = "model.safetensors"
        else:
            shard_name = (
                f"model-{out_shard_idx + 1:05d}-of-PLACEHOLDER.safetensors"
            )
        shard_path = output / shard_name
        mx.save_safetensors(str(shard_path), out_shard_data, metadata={"format": "mlx"})
        for k in out_shard_data:
            weight_map[k] = shard_name
        out_shard_idx += 1
        del out_shard_data

    total_shards = out_shard_idx
    if total_shards > 1:
        for i in range(total_shards):
            old_name = f"model-{i + 1:05d}-of-PLACEHOLDER.safetensors"
            new_name = (
                f"model-{i + 1:05d}-of-{total_shards:05d}.safetensors"
            )
            old_path = output / old_name
            new_path = output / new_name
            if old_path.exists():
                old_path.rename(new_path)
                for k, v in weight_map.items():
                    if v == old_name:
                        weight_map[k] = new_name

    cb("saving", 92.0)

    if total_shards > 1:
        total_size = sum(
            f.stat().st_size for f in output.glob("*.safetensors")
        )
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": dict(sorted(weight_map.items())),
        }
        with open(output / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)

    output_config = dict(config)
    for temp_key in ("_oq_sensitivity_map", "_oq_boost_map", "_oq_use_budget_plan", "_oq_non_quantizable"):
        output_config.pop(temp_key, None)
    if text_only:
        for key in ("vision_config", "image_token_id", "video_token_id",
                     "vision_start_token_id", "vision_end_token_id"):
            output_config.pop(key, None)
    # Ensure eos_token_id is present (mlx-lm adds it from tokenizer)
    if "eos_token_id" not in output_config:
        try:
            from transformers import AutoTokenizer
            _tok = AutoTokenizer.from_pretrained(str(source))
            if hasattr(_tok, "eos_token_id") and _tok.eos_token_id is not None:
                # Some models have multiple EOS tokens
                eos_ids = getattr(_tok, "additional_special_tokens_ids", [])
                if _tok.eos_token_id not in eos_ids:
                    eos_ids = [_tok.eos_token_id] + eos_ids
                # Check generation_config for eos_token_id list
                gen_config_path = source / "generation_config.json"
                if gen_config_path.exists():
                    with open(gen_config_path) as f:
                        gen_cfg = json.load(f)
                    if "eos_token_id" in gen_cfg:
                        output_config["eos_token_id"] = gen_cfg["eos_token_id"]
                        logger.info(f"Added eos_token_id from generation_config: {gen_cfg['eos_token_id']}")
                elif eos_ids:
                    output_config["eos_token_id"] = eos_ids if len(eos_ids) > 1 else eos_ids[0]
        except Exception as e:
            logger.debug(f"Could not resolve eos_token_id: {e}")
    quant_info = dict(quantization_config)
    for key, val in per_layer_config.items():
        quant_info[key] = val
    output_config["quantization"] = quant_info
    output_config["quantization_config"] = quant_info
    with open(output / "config.json", "w") as f:
        json.dump(output_config, f, indent=2, ensure_ascii=False)

    for pattern in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "generation_config.json",
        "chat_template.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "added_tokens.json",
        "merges.txt",
        "vocab.json",
    ):
        for src_file in source.glob(pattern):
            shutil.copy2(src_file, output / src_file.name)

    for py_file in source.glob("*.py"):
        shutil.copy2(py_file, output / py_file.name)

    cb("saving", 100.0)
    logger.info(
        f"oQ{oq_level:g} streaming: completed -> {output_path} "
        f"({total_shards} shards)"
    )


_SENS_NUM_SAMPLES = 128
_SENS_SEQ_LENGTH = 256


CALIB_DATASETS = {
    "default": "Built-in (General)",
    "wikitext": "WikiText-2",
    "c4": "C4 (Web Crawl)",
    "code": "Code (StarCoder)",
    "multilingual": "Multilingual (CulturaX)",
    "code_multilingual": "Code + Multilingual",
}


def _load_calibration_data(tokenizer, dataset: str = "code_multilingual",
                           num_samples: int = _SENS_NUM_SAMPLES,
                           seq_length: int = _SENS_SEQ_LENGTH):
    """Load calibration data for sensitivity measurement.

    Uses built-in calibration data by default (no download needed).
    Built-in data includes English, code, Korean, Chinese, Japanese.

    Args:
        tokenizer: Model tokenizer.
        dataset: "code_multilingual" (built-in default), "code", "multilingual",
                 "default" (mlx-lm generic), or HuggingFace dataset names.
        num_samples: Number of calibration samples.
        seq_length: Sequence length per sample.

    Returns:
        MLX array of shape (num_samples, seq_length) or None on failure.
    """
    if dataset in ("code_multilingual", "code", "multilingual"):
        try:
            return _load_builtin_calibration(
                tokenizer, dataset, num_samples, seq_length
            )
        except Exception as e:
            logger.warning(f"Built-in calibration failed: {e}, "
                           "falling back to mlx-lm default")

    if dataset == "default":
        try:
            from mlx_lm.quant.utils import load_data
            return load_data(tokenizer, num_samples=num_samples,
                            sequence_length=seq_length)
        except ImportError:
            logger.warning("mlx_lm.quant.utils.load_data not available")
            return None

    try:
        return _load_hf_calibration(tokenizer, dataset, num_samples, seq_length)
    except Exception as e:
        logger.warning(f"Failed to load {dataset}: {e}, falling back to built-in")

    try:
        return _load_builtin_calibration(
            tokenizer, "code_multilingual", num_samples, seq_length
        )
    except Exception:
        return None


def _load_builtin_calibration(tokenizer, dataset: str, num_samples: int,
                              seq_length: int):
    """Load from built-in oq_calibration_data.json (shipped with package)."""
    import mlx.core as mx

    data_path = Path(__file__).parent / "oq_calibration_data.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Built-in calibration data not found: {data_path}")

    with open(data_path, encoding="utf-8") as f:
        all_data = json.load(f)

    if dataset == "code_multilingual":
        texts = []
        for key in ("code", "en", "ko", "zh", "ja", "tool_calling"):
            texts.extend(all_data.get(key, []))
    elif dataset == "code":
        texts = all_data.get("code", []) + all_data.get("en", [])
    elif dataset == "multilingual":
        texts = []
        for key in ("en", "ko", "zh", "ja"):
            texts.extend(all_data.get(key, []))
    else:
        texts = []
        for v in all_data.values():
            texts.extend(v)

    if not texts:
        raise ValueError("No calibration text available")

    total_kb = sum(len(t) for t in texts) // 1024
    logger.info(f"Built-in calibration: {len(texts)} texts, "
                f"{total_kb} KB ({dataset})")

    all_ids = []
    for text in texts:
        ids = tokenizer.encode(text)
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        if isinstance(ids, list):
            all_ids.extend(ids)
        else:
            all_ids.extend(ids.tolist() if hasattr(ids, "tolist") else list(ids))
    tokens = mx.array(all_ids)

    usable = (tokens.size // seq_length) * seq_length
    if usable == 0:
        raise ValueError(f"Not enough tokens ({tokens.size} < {seq_length})")
    tokens = tokens[:usable].reshape(-1, seq_length)

    if num_samples > 0 and tokens.shape[0] > num_samples:
        indices = mx.random.permutation(tokens.shape[0])[:num_samples]
        tokens = tokens[indices]

    logger.info(f"Calibration: {tokens.shape[0]} samples x {seq_length} tokens")
    return tokens


def _load_hf_calibration(tokenizer, dataset: str, num_samples: int,
                         seq_length: int):
    """Load calibration data from HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library required for non-default calibration. "
            "Install with: pip install datasets"
        )

    logger.info(f"Loading calibration dataset: {dataset}")

    if dataset == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = "\n".join(t for t in ds["text"] if t.strip())
    elif dataset == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation",
                         streaming=True)
        texts = "\n".join(
            item["text"] for i, item in enumerate(ds) if i < num_samples * 2
        )
    elif dataset == "code":
        ds = load_dataset("bigcode/starcoderdata", "python",
                         split="train", streaming=True)
        texts = "\n".join(
            item["content"] for i, item in enumerate(ds) if i < num_samples * 2
        )
    elif dataset == "multilingual":
        langs = ["en", "ko", "zh", "ja", "de", "fr", "es"]
        per_lang = max(1, num_samples // len(langs))
        all_texts = []
        for lang in langs:
            try:
                ds = load_dataset("uonlp/CulturaX", lang,
                                 split="train", streaming=True)
                lang_texts = [
                    item["text"] for i, item in enumerate(ds)
                    if i < per_lang * 2
                ]
                all_texts.extend(lang_texts)
            except Exception:
                logger.warning(f"Failed to load CulturaX/{lang}, skipping")
        texts = "\n".join(all_texts)
    elif dataset == "code_multilingual":
        half = max(1, num_samples // 2)
        code_texts = []
        try:
            ds = load_dataset("bigcode/starcoderdata", "python",
                             split="train", streaming=True)
            code_texts = [
                item["content"] for i, item in enumerate(ds) if i < half * 2
            ]
        except Exception:
            logger.warning("Failed to load code dataset")

        ml_texts = []
        for lang in ["en", "ko", "zh", "ja"]:
            try:
                ds = load_dataset("uonlp/CulturaX", lang,
                                 split="train", streaming=True)
                ml_texts.extend(
                    item["text"] for i, item in enumerate(ds)
                    if i < half // 2
                )
            except Exception:
                pass
        texts = "\n".join(code_texts + ml_texts)
    else:
        raise ValueError(f"Unknown calibration dataset: {dataset}")

    if not texts:
        raise ValueError(f"No text loaded from {dataset}")

    tokens = tokenizer.encode(texts)
    if hasattr(tokens, "input_ids"):
        tokens = tokens.input_ids
    if isinstance(tokens, list):
        tokens = mx.array(tokens)
    elif not isinstance(tokens, mx.array):
        import numpy as np
        tokens = mx.array(np.array(tokens))

    if tokens.ndim > 1:
        tokens = tokens.reshape(-1)

    n_tokens = tokens.size
    usable = (n_tokens // seq_length) * seq_length
    if usable == 0:
        raise ValueError(f"Not enough tokens from {dataset} (got {n_tokens})")
    tokens = tokens[:usable].reshape(-1, seq_length)

    n_available = tokens.shape[0]
    if num_samples > 0 and n_available > num_samples:
        indices = mx.random.permutation(n_available)[:num_samples]
        tokens = tokens[indices]

    logger.info(f"Calibration: {tokens.shape[0]} samples × {seq_length} tokens "
                f"from {dataset}")
    return tokens


def _find_model_layers(model):
    """Find embedding function and transformer layers in the model.

    Searches common model structures: standard, VLM, and direct.
    Returns (embed_fn, layers) or (None, None).
    """
    embed_fn = None
    layers = None

    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        embed_fn = model.model.embed_tokens
        layers = model.model.layers
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        lm = model.language_model.model
        if hasattr(lm, 'embed_tokens'):
            embed_fn = lm.embed_tokens
            layers = lm.layers
    elif hasattr(model, 'embed_tokens'):
        embed_fn = model.embed_tokens
        layers = model.layers
    elif hasattr(model, 'backbone') and hasattr(model.backbone, 'embeddings'):
        embed_fn = model.backbone.embeddings
        layers = model.layers

    return embed_fn, layers


def _forward_layer(block, inputs, mask, position_ids):
    """Forward pass through a transformer layer with flexible signature."""
    last_exc = None
    for call_args in [
        (inputs, mask, None, position_ids),
        (inputs, mask, None),
        (inputs, mask),
        (inputs, None, mask, None),
        (inputs,),
    ]:
        try:
            result = block(*call_args)
            if isinstance(result, tuple):
                result = result[0]
            return result
        except (TypeError, ValueError, RuntimeError, AttributeError) as e:
            last_exc = e
            continue
    if last_exc is not None:
        logger.debug(
            f"_forward_layer: all signatures failed for "
            f"{type(block).__name__}: {last_exc}"
        )
    return None


def _layer_masks_for_model(model, layers, inputs):
    """Build the per-layer mask schedule used by the original model."""
    if hasattr(model, "make_cache") and any(hasattr(layer, "is_linear") for layer in layers):
        try:
            from mlx_lm.models.base import create_attention_mask, create_ssm_mask

            cache = model.make_cache()
            fa_idx = getattr(getattr(model, "model", model), "fa_idx", 0)
            ssm_idx = getattr(getattr(model, "model", model), "ssm_idx", 0)
            fa_cache = cache[fa_idx] if fa_idx < len(cache) else None
            ssm_cache = cache[ssm_idx] if ssm_idx < len(cache) else None
            try:
                fa_mask = create_attention_mask(inputs, fa_cache)
            except TypeError:
                # mlx-lm API changed — cache.make_mask signature differs
                fa_mask = None
            try:
                ssm_mask = create_ssm_mask(inputs, ssm_cache)
            except TypeError:
                ssm_mask = None
            if fa_mask is not None or ssm_mask is not None:
                if fa_mask is None:
                    fa_mask = nn.MultiHeadAttention.create_additive_causal_mask(
                        inputs.shape[1]
                    ).astype(inputs.dtype if hasattr(inputs, "dtype") else mx.float16)
                # SSM layers (GatedDeltaNet) expect (B, S) boolean mask, not
                # (S, S) causal mask.  During calibration there is no padding,
                # so None is the correct mask for SSM layers.
                return [ssm_mask if getattr(layer, "is_linear", False) else fa_mask for layer in layers]
        except (ImportError, AttributeError):
            pass

    seq_len = inputs.shape[1]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    dtype = inputs.dtype if hasattr(inputs, "dtype") else mx.float16
    return [mask.astype(dtype)] * len(layers)


def _qdq_weight_only(weight, bits: int, group_size: int, mode: str):
    qw, scales, *rest = mx.quantize(weight, group_size=group_size, bits=bits, mode=mode)
    return mx.dequantize(
        qw,
        scales,
        rest[0] if rest else None,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )


def _temporary_quantize_block(block, config, oq_level, group_size: int):
    """Quantize-dequantize a block using the active predicate configuration."""
    saved = {}
    for path, module in tree_flatten(block.leaf_modules(), is_leaf=nn.Module.is_module):
        if not hasattr(module, "weight") or not hasattr(module, "to_quantized"):
            continue
        if getattr(module.weight, "ndim", 0) < 2:
            continue
        norm_path = _normalize_quant_path(path)
        bits, gs, mode = _get_predicate_bits(norm_path, config, oq_level, group_size)
        if bits is None or module.weight.shape[-1] % gs != 0:
            continue
        saved[path] = module.weight
        module.weight = _qdq_weight_only(module.weight, bits, gs, mode)
    return saved


def _restore_saved_weights(block, saved):
    """Restore temporarily quantized block weights."""
    modules_by_path = dict(
        tree_flatten(block.leaf_modules(), is_leaf=nn.Module.is_module)
    )
    for path, weight in saved.items():
        if path in modules_by_path:
            modules_by_path[path].weight = weight


def _measure_sensitivity_from_model(
    model, tokenizer, config, oq_level,
    calib_dataset="code_multilingual",
    num_samples=32, seq_length=256,
):
    """Measure per-layer quantization sensitivity on an already-loaded model.

    Does NOT modify weights — uses temporary quantize→dequantize per layer.
    Used by both streaming (after temporary load) and enhanced (before AWQ).

    Returns:
        Dict of {layer_idx: relative_mse_score}.
    """
    calib_data = _load_calibration_data(
        tokenizer, dataset=calib_dataset,
        num_samples=num_samples, seq_length=seq_length,
    )
    if calib_data is None:
        return {}

    embed_fn, layers = _find_model_layers(model)
    if embed_fn is None or layers is None:
        return {}

    inputs = embed_fn(calib_data)
    layer_masks = _layer_masks_for_model(model, layers, inputs)
    position_ids = mx.arange(calib_data.shape[1])[None, :]
    sensitivity = {}

    for layer_idx, block in enumerate(layers):
        layer_mask = layer_masks[layer_idx] if layer_idx < len(layer_masks) else None
        out_float = _forward_layer(block, inputs, layer_mask, position_ids)
        if out_float is None:
            continue

        saved = _temporary_quantize_block(
            block, config, oq_level, _OQ_DEFAULT_GROUP_SIZE
        )
        out_quant = _forward_layer(block, inputs, layer_mask, position_ids)
        if out_quant is not None:
            raw_mse = ((out_float - out_quant) ** 2).mean()
            out_magnitude = (out_float ** 2).mean()
            mse_val = raw_mse / mx.maximum(out_magnitude, 1e-10)
            mx.eval(mse_val)
            sensitivity[layer_idx] = mse_val.item()

        _restore_saved_weights(block, saved)

        inputs = out_float
        mx.synchronize()
        mx.clear_cache()

    if sensitivity:
        ranked = sorted(sensitivity.items(), key=lambda x: -x[1])
        logger.info(
            f"oQ{oq_level:g}: layer sensitivity (descending): "
            + ", ".join(f"L{i}={s:.4f}" for i, s in ranked)
        )

    return sensitivity


def _measure_sensitivity(
    model_path: str, config: dict, oq_level,
    calib_dataset="code_multilingual",
    num_samples=32, seq_length=256,
):
    """Measure sensitivity by loading model temporarily. Used by streaming path."""
    is_vlm = "vision_config" in config

    try:
        if is_vlm:
            from mlx_vlm.utils import load_model as vlm_load_model

            model = vlm_load_model(Path(model_path), lazy=True)
            from mlx_lm import load as lm_load

            _, tokenizer = lm_load(model_path, lazy=True)
        else:
            from mlx_lm import load as lm_load

            model, tokenizer = lm_load(model_path, lazy=True)
    except Exception as e:
        logger.warning(
            f"Sensitivity measurement: model load failed ({e}), "
            "using position-based"
        )
        return {}

    sensitivity = _measure_sensitivity_from_model(
        model, tokenizer, config, oq_level,
        calib_dataset, num_samples, seq_length,
    )

    del model, tokenizer
    mx.synchronize()
    mx.clear_cache()

    return sensitivity


_REQUANT_VALID_BITS = {2, 3, 4, 5, 6, 8}


def _measure_sensitivity_from_quantized_model(
    model_path: str, config: dict, oq_level,
    calib_dataset="code_multilingual",
    num_samples=32, seq_length=256,
):
    """Measure sensitivity via re-quantization on a quantized model.

    Loads a quantized model (~4x less memory than fp16) and perturbs each
    layer by re-quantizing at (bits-1). The relative MSE ranking matches
    fp16 qdq-MSE with ~90% top-10 overlap.
    """
    from mlx_lm import load as lm_load

    try:
        model, tokenizer = lm_load(model_path, lazy=True)
    except Exception as e:
        logger.warning(f"Sensitivity proxy load failed ({e}), using position-based")
        return {}

    calib_data = _load_calibration_data(
        tokenizer, dataset=calib_dataset,
        num_samples=num_samples, seq_length=seq_length,
    )
    if calib_data is None:
        del model, tokenizer
        mx.synchronize()
        mx.clear_cache()
        return {}

    embed_fn, layers = _find_model_layers(model)
    if embed_fn is None or layers is None:
        del model, tokenizer
        mx.synchronize()
        mx.clear_cache()
        return {}

    inputs = embed_fn(calib_data)
    layer_masks = _layer_masks_for_model(model, layers, inputs)
    position_ids = mx.arange(calib_data.shape[1])[None, :]
    sensitivity = {}

    for layer_idx, block in enumerate(layers):
        layer_mask = layer_masks[layer_idx] if layer_idx < len(layer_masks) else None
        out_baseline = _forward_layer(block, inputs, layer_mask, position_ids)
        if out_baseline is None:
            continue

        saved = {}
        for p, m in tree_flatten(block.leaf_modules(), is_leaf=nn.Module.is_module):
            if not hasattr(m, "scales") or not hasattr(m, "weight"):
                continue
            bits = getattr(m, "bits", 4)
            gs = getattr(m, "group_size", 64)
            mode = getattr(m, "mode", "affine")
            perturb_bits = bits - 1
            if perturb_bits not in _REQUANT_VALID_BITS:
                continue
            w_float = mx.dequantize(
                m.weight, m.scales, getattr(m, "biases", None),
                group_size=gs, bits=bits, mode=mode,
            )
            saved[p] = (m.weight, m.scales, getattr(m, "biases", None), bits)
            qw, sc, *rest = mx.quantize(w_float, group_size=gs, bits=perturb_bits, mode="affine")
            m.weight = qw
            m.scales = sc
            m.biases = rest[0] if rest else None
            m.bits = perturb_bits

        out_perturbed = _forward_layer(block, inputs, layer_mask, position_ids)

        modules_by_path = dict(
            tree_flatten(block.leaf_modules(), is_leaf=nn.Module.is_module)
        )
        for p, (w, s, b, orig_bits) in saved.items():
            if p in modules_by_path:
                mod = modules_by_path[p]
                mod.weight = w
                mod.scales = s
                if b is not None:
                    mod.biases = b
                mod.bits = orig_bits

        if out_perturbed is not None:
            raw_mse = ((out_baseline - out_perturbed) ** 2).mean()
            out_mag = (out_baseline ** 2).mean()
            mse_val = raw_mse / mx.maximum(out_mag, 1e-10)
            mx.eval(mse_val)
            sensitivity[layer_idx] = mse_val.item()

        inputs = out_baseline
        mx.eval(inputs)
        mx.synchronize()
        mx.clear_cache()

    del model, tokenizer
    mx.synchronize()
    mx.clear_cache()

    if sensitivity:
        ranked = sorted(sensitivity.items(), key=lambda x: -x[1])
        logger.info(
            f"oQ{oq_level:g}: proxy sensitivity (descending): "
            + ", ".join(f"L{i}={s:.4f}" for i, s in ranked)
        )

    return sensitivity


