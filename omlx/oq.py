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
from pathlib import Path
from typing import Callable, Optional, Union

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

logger = logging.getLogger(__name__)

# Allowed oQ quantization levels
OQ_LEVELS = {2, 3, 3.5, 4, 5, 6, 8}

# Bits-per-GB estimate for progress timing (seconds per GB of source weights)
_QUANT_SECONDS_PER_GB = 3.0


# =============================================================================
# Universal Quant Predicate
# =============================================================================


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
    # VLM models nest text model config under "text_config"
    tc = config.get("text_config", {})
    num_layers = config.get("num_hidden_layers") or tc.get("num_hidden_layers", 32)
    num_experts = (
        config.get("num_local_experts")
        or tc.get("num_local_experts")
        or config.get("num_experts")
        or tc.get("num_experts", 0)
    )
    hidden_size = config.get("hidden_size") or tc.get("hidden_size", 0)
    is_moe = num_experts > 0

    # oQ level → base bits + protection mode
    # oQ2: base 2, minimal protection (only safety-critical)
    # oQ3: base 3, full protection
    # oQ3.5: base 3, full protection + expert down_proj 4-bit (Super Weights)
    # oQ4-6: base = level, full protection
    # oQ8: base 8, full protection
    _LEVEL_MAP = {
        2: (2, "minimal"),
        3: (3, "full"),
        3.5: (3, "full"),
        4: (4, "full"),
        5: (5, "full"),
        6: (6, "full"),
        8: (8, "full"),
    }
    base_bits, protection = _LEVEL_MAP.get(oq_level, (oq_level, "full"))
    full_protection = protection == "full"

    def gs():
        if _is_moe_router(path):
            return 64
        if num_experts >= 150:
            return 128
        return 64

    # Helper: never assign bits below base_bits
    # Auto-select optimal mode per bit width:
    #   4-bit → mxfp4 (uint8 scales, 0.25 bpw overhead vs affine 0.50)
    #   8-bit → mxfp8 (Apple Silicon native, group_size=32)
    #   other → affine (supports 2,3,5,6-bit with flexible group_size)
    def bits(n):
        effective = int(max(n, base_bits))
        if effective == 4:
            return {"bits": 4, "group_size": 32, "mode": "mxfp4"}
        if effective == 8:
            return {"bits": 8, "group_size": 32, "mode": "mxfp8"}
        return {"bits": effective, "group_size": gs()}

    # ══════════════════════════════════════════════
    # Stage 0: Safety-critical non-quantization (ALWAYS applied)
    # ══════════════════════════════════════════════

    # MoE router: fp16 (routing decisions must be precise)
    if _is_moe_router(path):
        return False

    # shared expert gate: fp16
    # endswith check handles both module paths ("shared_expert_gate")
    # and tensor names ("shared_expert_gate.weight")
    if "shared_expert_gate" in path and "gate_proj" not in path:
        return False

    # VLM vision encoder + projector: fp16
    if _is_vision_tensor(path):
        return False

    # SSM state parameters: F32
    if any(
        p in path
        for p in ("ssm_alpha", "ssm_beta", "a_log", "time_decay", "time_faaaa")
    ):
        return False

    # Mamba D parameter: F32
    if path.endswith(".D"):
        return False

    # ══════════════════════════════════════════════
    # Minimal protection mode (oQ2, oQ8)
    # Only lm_head gets slight boost, everything else = base_bits
    # ══════════════════════════════════════════════

    if not full_protection:
        # Critical layers: 6-bit affine (64 levels, +2 GB for 5% of params)
        # 6-bit chosen over mxfp4 4-bit for 4x precision (64 vs 16 levels)
        # Cost: ~0.2 bpw overhead — negligible for significant quality gain

        # lm_head: 6-bit
        if any(p in path for p in ("lm_head", "output.weight", "classifier")):
            return bits(6)

        # SSM output: at least 8-bit
        if any(p in path for p in ("ssm_output", "ssm_out")):
            return bits(8)

        # Embedding: base+2 (error propagates to all layers, <0.6% of params)
        if any(p in path for p in ("embed_tokens", "wte", "word_embeddings")):
            return bits(base_bits + 2)

        # 512+ expert MLP asymmetry safety (prevent NaN)
        if num_experts >= 512 and hidden_size >= 4096:
            if "gate_proj" in path and "shared_expert" not in path:
                return bits(4)

        # Sensitive layers (first/last 12.5%): +1 bit for non-expert layers only
        layer_idx = _extract_layer_index(path)
        if layer_idx >= 0:
            sensitive = (
                layer_idx < num_layers // 8
                or layer_idx >= 7 * num_layers // 8
            )
            is_expert = "switch_mlp" in path or "experts" in path
            if sensitive and not is_expert:
                return bits(base_bits + 1)

        # Everything else: base_bits
        return True

    # ══════════════════════════════════════════════
    # Full protection mode (oQ3-oQ7)
    # ══════════════════════════════════════════════

    # ── High-precision protection ──

    # SSM output: Q8
    if any(p in path for p in ("ssm_output", "ssm_out")):
        return bits(8)

    # RWKV lora: Q8
    if "lora.2" in path:
        return bits(8)

    # lm_head: Q6
    if any(p in path for p in ("lm_head", "output.weight", "classifier")):
        return bits(6)

    # cross-attention output (VLM): Q6
    if "cross_attn" in path and "o_proj" in path:
        return bits(6)

    # MLA projections (DeepSeek): Q6
    if any(
        p in path
        for p in ("kv_a_proj_with_mqa", "kv_b_proj", "q_a_proj", "q_b_proj")
    ):
        return bits(6)

    # attn_output: Q5 for dense
    if "o_proj" in path and "shared_expert" not in path:
        if not is_moe:
            return bits(5)

    # ── MoE-specific ──

    # shared expert body: high-bits
    if "shared_expert" in path and not path.endswith("shared_expert_gate"):
        if "gate_proj" in path or "up_proj" in path:
            return bits(6)
        if "down_proj" in path:
            return bits(5)

    # 512+ expert MLP asymmetry
    if num_experts >= 512 and hidden_size >= 4096:
        if "gate_proj" in path and "shared_expert" not in path:
            return bits(4)
        if "down_proj" in path and "shared_expert" not in path:
            return bits(3)

    # ── Layer position strategy ──

    layer_idx = _extract_layer_index(path)
    sensitive = layer_idx >= 0 and (
        layer_idx < num_layers // 8
        or layer_idx >= 7 * num_layers // 8
    )

    # v_proj: Q6 (sensitive) / base (rest)
    if any(p in path for p in ("v_proj", "v_a_proj", "v_b_proj")):
        if sensitive:
            return bits(6)
        return True  # base_bits

    # down_proj: protected for dense/shared, base for routed experts
    if any(p in path for p in ("down_proj", "w2", "mlp.fc2", "wo")):
        is_routed_expert = is_moe and "shared_expert" not in path and (
            "switch_mlp" in path or "experts" in path
        )
        if is_routed_expert:
            # oQ3.5: boost expert down_proj to 4-bit (Super Weights protection)
            if oq_level == 3.5:
                return bits(4)
            return True  # base_bits
        if sensitive:
            return bits(6)
        return bits(5)

    # q/k_proj: Q5 (sensitive)
    if any(p in path for p in ("q_proj", "k_proj")):
        if sensitive:
            return bits(5)

    # fused QKV: Q5 (sensitive)
    if any(p in path for p in ("qkv_proj", "in_proj_qkv", "attn_qkv")):
        if sensitive:
            return bits(5)

    # ── SSM/GatedDeltaNet ──

    if any(p in path for p in ("in_proj_z", "in_proj_a", "in_proj_b", "delta_net")):
        return bits(5)

    if any(
        p in path for p in ("mixer.in_proj", "mixer.out_proj", "x_proj", "dt_proj")
    ):
        return bits(5)

    # ── Default: base_bits ──
    return True


# =============================================================================
# Helper functions
# =============================================================================


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


def resolve_output_name(model_name: str, oq_level: int,
                        enable_clip: bool = False) -> str:
    """Generate output model name: strip existing quant suffixes, append oQ tag.

    Examples:
        "Qwen3.5-122B-A10B" + 4 -> "Qwen3.5-122B-A10B-oQ4"
        "Qwen3.5-122B-A10B" + 4 + clip -> "Qwen3.5-122B-A10B-oQ4+"
        "Qwen3.5-122B-A10B-8bit" + 4 -> "Qwen3.5-122B-A10B-oQ4"
        "Qwen3.5-122B-A10B-oQ6" + 2 -> "Qwen3.5-122B-A10B-oQ2"
    """
    base = re.sub(
        r"-(oQ[\d.]+\+?|[0-9]+[_-]?bit|fp\d+|bf\d+)$",
        "",
        model_name,
        flags=re.IGNORECASE,
    )
    level_str = f"{oq_level:g}"  # 3.5 → "3.5", 4.0 → "4"
    suffix = f"oQ{level_str}+" if enable_clip else f"oQ{level_str}"
    return f"{base}-{suffix}"


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
        # Native FP8 models are quantizable (mlx-lm sanitize handles FP8→float)
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

    # Scan all tensors to get names and sizes (lazy/mmap, no memory cost)
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
                # Non-quantizable: stored as fp16
                total_params += n_elements
                total_weighted_bits += n_elements * 16
                total_output_bytes += n_elements * 2  # fp16
                continue

            if _should_skip_tensor(name):
                continue

            bits, gs, _mode = _get_predicate_bits(name, config, oq_level, group_size)
            if bits is None:
                # predicate = False → fp16
                total_params += n_elements
                total_weighted_bits += n_elements * 16
                total_output_bytes += n_elements * 2
            else:
                total_params += n_elements
                # Quantized size: weight + scales + biases overhead
                if len(shape) >= 2:
                    n_groups = (shape[-1] + gs - 1) // gs
                    rows = n_elements // max(shape[-1], 1)
                    weight_bytes = (n_elements * bits + 7) // 8
                    # mxfp4: uint8 scale (1 byte/group)
                    # mxfp8: fp16 scale (2 bytes/group)
                    # affine: fp16 scale + fp16 bias (4 bytes/group)
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

    # bpw from actual output bytes (includes scale/bias overhead, matches GGUF convention)
    effective_bpw = total_weighted_bits / max(total_params, 1)

    # Precise memory estimation based on actual tensor sizes:
    source_total = sum(
        sf.stat().st_size for sf in source.glob("*.safetensors")
    )
    num_shards = len(list(source.glob("*.safetensors")))
    max_shard_size = max(
        (sf.stat().st_size for sf in source.glob("*.safetensors")),
        default=0,
    )

    # Streaming: mx.load uses mmap (lazy). Only the current shard being
    # processed + sanitize temp + output buffer are in physical memory.
    # Peak ≈ largest_source_shard + output_shard_buffer(5GB) + sanitize_overhead
    streaming_peak = max_shard_size * 2 + 5 * 1024**3 + 2 * 1024**3

    # Clip: full model loaded via mlx_lm.load() (all weights materialized)
    # + calibration activations + quantization transition overhead
    clip_peak = source_total + total_output_bytes + 500_000_000

    return {
        "effective_bpw": round(effective_bpw, 2),
        "output_size_bytes": total_output_bytes,
        "output_size_formatted": _format_size(total_output_bytes),
        "memory_streaming_bytes": streaming_peak,
        "memory_streaming_formatted": _format_size(streaming_peak),
        "memory_clip_bytes": clip_peak,
        "memory_clip_formatted": _format_size(clip_peak),
    }


def estimate_memory(source_size_bytes: int, enable_clip: bool) -> dict:
    """Estimate peak memory for quantization.

    This is a rough estimate used before precise calculation is available.
    The /api/oq/estimate endpoint provides precise values per tensor.

    Streaming: source (mmap) + 5GB output buffer + sanitize overhead
    Clip: source (loaded) + calibration + transition overhead
    """
    if enable_clip:
        peak = source_size_bytes + int(source_size_bytes * 0.15) + 500_000_000
    else:
        peak = source_size_bytes + 6 * 1024**3  # source (mmap) + 5GB buffer + 1GB overhead
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


# =============================================================================
# Tensor-by-Tensor Streaming Quantization (Low Memory)
# =============================================================================

# Max shard size in bytes (5 GB, matching mlx-lm default)
_MAX_SHARD_BYTES = 5_000_000_000

# Tensor name patterns that should NOT be quantized (norms, biases, etc.)
_SKIP_QUANT_PATTERNS = (
    "layernorm", "rmsnorm", "norm.weight", "norm.bias",
    "ln_", "layer_norm",
)


def _should_skip_tensor(name: str) -> bool:
    """Check if a tensor should be completely excluded from output.

    These tensors are removed by mlx-lm sanitize() and should not be saved.
    """
    # MTP (multi-token prediction) layers — removed by qwen3_5 sanitize
    if ".mtp." in name or name.startswith("mtp."):
        return True
    return False


def _should_quantize_tensor(name: str, shape: tuple) -> bool:
    """Check if a tensor should be quantized based on name and shape."""
    # Skip 1D tensors (biases, norms)
    if len(shape) < 2:
        return False
    # Skip if name indicates norm/bias
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

    # For VLM models, use mlx-vlm's full sanitize chain
    # (model.sanitize → VisionModel.sanitize → LanguageModel.sanitize)
    # This preserves vision weights and handles all model-specific transforms
    if is_vlm:
        try:
            from mlx_vlm.utils import get_model_and_args, sanitize_weights

            model_module, _ = get_model_and_args(config)
            model_config_cls = model_module.ModelConfig
            model_config = model_config_cls.from_dict(config)

            # Convert nested dict configs to proper dataclasses
            vision_config = model_config.vision_config
            if isinstance(vision_config, dict):
                vision_config = model_module.VisionConfig.from_dict(vision_config)
            text_config = model_config.text_config
            if isinstance(text_config, dict):
                text_config = model_module.TextConfig.from_dict(text_config)

            # Replace dict configs in model_config with proper dataclasses
            model_config.vision_config = vision_config
            model_config.text_config = text_config

            def _vlm_sanitize(weights):
                # Step 1: Model-level sanitize (mtp removal, gate_up split, renames)
                # Can't use sanitize_weights(Model, w, config) because Model()
                # constructor fails on VisionModel init. Instead, create a
                # minimal proxy with just the config attributes sanitize needs.
                import types

                class _Proxy:
                    pass
                proxy = _Proxy()
                proxy.config = model_config
                w = model_module.Model.sanitize(proxy, weights)

                # Step 2: VisionModel sanitize (conv transpose, etc.)
                w = sanitize_weights(
                    model_module.VisionModel, w, vision_config
                )
                # Step 3: LanguageModel sanitize (norm +1, conv1d, etc.)
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

    # Fallback to mlx-lm
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


def _get_predicate_bits(tensor_name: str, config: dict, oq_level: int,
                        group_size: int) -> tuple:
    """Get quantization bits, group_size, and mode for a tensor.

    Returns:
        (bits, group_size, mode) or (None, None, None) if not quantized.
    """
    # Get base_bits from level map
    _LEVEL_MAP = {2: 2, 3: 3, 3.5: 3, 4: 4, 5: 5, 6: 6, 8: 8}
    base_bits = int(_LEVEL_MAP.get(oq_level, oq_level))

    result = universal_quant_predicate(tensor_name, None, config, oq_level)
    if result is False:
        return None, None, None
    if isinstance(result, dict):
        bits = result.get("bits", base_bits)
        gs = result.get("group_size", group_size)
        mode = result.get("mode", _mode_for_bits(bits))
        return bits, gs, mode
    # True → base bits with auto mode
    return base_bits, _gs_for_mode(base_bits, group_size), _mode_for_bits(base_bits)


def _mode_for_bits(bits: int) -> str:
    """Select optimal quantization mode for a given bit width."""
    if bits == 4:
        return "mxfp4"
    if bits == 8:
        return "mxfp8"
    return "affine"


def _gs_for_mode(bits: int, default_gs: int) -> int:
    """Get required group_size for a mode."""
    if bits == 4:
        return 32  # mxfp4 requires gs=32
    if bits == 8:
        return 32  # mxfp8 requires gs=32
    return default_gs


def quantize_oq_streaming(
    model_path: str,
    output_path: str,
    oq_level: int,
    group_size: int = 64,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    text_only: bool = False,
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
    """
    if oq_level not in OQ_LEVELS:
        raise ValueError(
            f"Invalid oQ level {oq_level}. Must be one of {sorted(OQ_LEVELS)}"
        )

    source = Path(model_path)
    output = Path(output_path)
    if output.exists():
        raise ValueError(f"Output directory already exists: {output_path}")

    output.mkdir(parents=True, exist_ok=True)
    cb = progress_callback or (lambda phase, pct: None)

    # Read config
    config_path = source / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    cb("loading", 5.0)

    # Scan all safetensors files
    weight_files = sorted(source.glob("*.safetensors"))
    if not weight_files:
        raise ValueError(f"No .safetensors files found in {model_path}")

    cb("loading", 8.0)

    # Load ALL weights as one lazy dict (mmap, no physical memory used yet)
    # Then apply the model's sanitize() to get correct names/transforms
    all_weights = {}
    for sf_path in weight_files:
        shard = mx.load(str(sf_path), return_metadata=False)
        all_weights.update(shard)
        del shard

    logger.info(
        f"oQ{oq_level:g} streaming: {len(all_weights)} tensors in "
        f"{len(weight_files)} shards"
    )

    cb("loading", 12.0)

    # Apply model's sanitize() — handles ALL model-specific transformations
    # (name renames, fused tensor splits, conv1d transpose, norm +1, etc.)
    sanitize_fn = _build_model_sanitizer(config)
    if sanitize_fn is not None:
        try:
            all_weights = sanitize_fn(all_weights)
            logger.info(f"oQ{oq_level:g}: sanitize applied, {len(all_weights)} tensors")
        except Exception as e:
            logger.warning(f"Sanitize failed ({e}), using original names")

    # Resolve model dtype (same as mlx-lm convert)
    # FP8/float32 models need dtype normalization before quantization
    tc = config.get("text_config", {})
    model_dtype_str = config.get("torch_dtype") or tc.get("dtype")
    model_dtype = getattr(mx, model_dtype_str) if model_dtype_str in (
        "float16", "bfloat16", "float32",
    ) else None

    cb("loading", 15.0)

    # Group sanitized weights into output shards for processing
    # Process in chunks to keep memory bounded
    tensor_names = list(all_weights.keys())
    total_tensors = len(tensor_names)
    out_shard_data = {}
    out_shard_idx = 0
    weight_map = {}
    _LEVEL_MAP = {2: 2, 3: 3, 3.5: 3, 4: 4, 5: 5, 6: 6, 8: 8}
    base_bits = int(_LEVEL_MAP.get(oq_level, oq_level))
    base_mode = _mode_for_bits(base_bits)
    base_gs = _gs_for_mode(base_bits, group_size)
    quantization_config = {"group_size": base_gs, "bits": base_bits, "mode": base_mode}
    per_layer_config = {}
    start_time = _time.monotonic()

    # Compute total bytes for progress tracking
    # Use source file size as proxy (avoids materializing all tensors)
    total_bytes = sum(sf.stat().st_size for sf in source.glob("*.safetensors"))
    processed_bytes = 0

    for i, tensor_name in enumerate(tensor_names):
        w_mx = all_weights.pop(tensor_name)  # pop: remove from dict to free memory
        tensor_bytes = w_mx.nbytes
        shape = w_mx.shape

        # text_only: skip vision encoder weights entirely
        if text_only and _is_vision_tensor(tensor_name):
            del w_mx
            processed_bytes += tensor_bytes
            continue

        # Normalize dtype to match mlx-lm convert behavior
        # (e.g. FP8 models sanitized to float32, then cast to target dtype)
        if model_dtype is not None and mx.issubdtype(w_mx.dtype, mx.floating):
            w_mx = w_mx.astype(model_dtype)

        if _should_quantize_tensor(tensor_name, shape):
            bits, gs, qmode = _get_predicate_bits(
                tensor_name, config, oq_level, group_size
            )

            if bits is not None and len(shape) >= 2 and shape[-1] % gs == 0:
                # Do NOT cast to float16 — bf16→f16 changes 41%+ of quantized values.
                qw, scales, *rest = mx.quantize(
                    w_mx, group_size=gs, bits=bits, mode=qmode
                )
                biases = rest[0] if rest else None

                base = tensor_name
                if base.endswith(".weight"):
                    base = base[:-7]

                out_shard_data[f"{base}.weight"] = qw
                out_shard_data[f"{base}.scales"] = scales
                if biases is not None:
                    out_shard_data[f"{base}.biases"] = biases

                # Track per-layer config for mixed precision/mode
                _LEVEL_MAP = {2: 2, 3: 3, 3.5: 3, 4: 4, 5: 5, 6: 6, 8: 8}
                base_bits = int(_LEVEL_MAP.get(oq_level, oq_level))
                base_qmode = _mode_for_bits(base_bits)
                base_gs = _gs_for_mode(base_bits, group_size)
                if bits != base_bits or gs != base_gs or qmode != base_qmode:
                    layer_cfg = {"bits": bits, "group_size": gs}
                    # Always include mode — nn.quantize defaults to affine
                    # when mode is missing from per-layer config
                    layer_cfg["mode"] = qmode
                    per_layer_config[base] = layer_cfg
            else:
                # Can't quantize or predicate=False → keep fp16
                out_shard_data[tensor_name] = w_mx
        else:
            # Non-quantizable (1D norms, biases) → keep original dtype
            out_shard_data[tensor_name] = w_mx

        # Free source tensor immediately
        del w_mx

        # Flush shard when output size exceeds limit
        current_bytes = sum(v.nbytes for v in out_shard_data.values())
        if current_bytes >= _MAX_SHARD_BYTES:
            shard_name = f"model-{out_shard_idx + 1:05d}-of-PLACEHOLDER.safetensors"
            shard_path = output / shard_name
            mx.save_safetensors(str(shard_path), out_shard_data, metadata={"format": "mlx"})
            for k in out_shard_data:
                weight_map[k] = shard_name
            out_shard_idx += 1
            out_shard_data = {}
            mx.clear_cache()
            logger.info(f"oQ{oq_level:g}: wrote output shard {out_shard_idx}")

        # Progress + ETA (bytes-based for accuracy with mixed tensor sizes)
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
    mx.clear_cache()

    # Flush remaining shard
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

    # Rename PLACEHOLDER shards to actual count
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

    # Write weight index (if multiple shards)
    if total_shards > 1:
        index = {
            "metadata": {"total_size": sum(0 for _ in weight_map)},
            "weight_map": dict(sorted(weight_map.items())),
        }
        with open(output / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)

    # Write config.json with quantization info
    # mlx-lm loader uses class_predicate:
    #   1. Check per-layer config: config["quantization"][module_path]
    #   2. Check if module has to_quantized()
    #   3. Check if {path}.scales exists in weights
    # For mixed-precision, we store per-layer overrides for layers with
    # different bits than the base config.
    output_config = dict(config)
    # text_only: strip VLM-specific config keys
    if text_only:
        for key in ("vision_config", "image_token_id", "video_token_id",
                     "vision_start_token_id", "vision_end_token_id"):
            output_config.pop(key, None)
    quant_info = dict(quantization_config)
    # Only add per-layer entries that differ from base bits/group_size
    # and deduplicate (e.g. one entry per module, not per expert)
    for key, val in per_layer_config.items():
        quant_info[key] = val
    output_config["quantization"] = quant_info
    output_config["quantization_config"] = quant_info
    with open(output / "config.json", "w") as f:
        json.dump(output_config, f, indent=2, ensure_ascii=False)

    # Copy tokenizer and other files
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

    # Copy .py files (trust_remote_code models)
    for py_file in source.glob("*.py"):
        shutil.copy2(py_file, output / py_file.name)

    cb("saving", 100.0)
    logger.info(
        f"oQ{oq_level:g} streaming: completed -> {output_path} "
        f"({total_shards} shards)"
    )


# =============================================================================
# AWQ-Style Output-MSE Clip Optimization
# =============================================================================

# Max bits to apply clip optimization (diminishing returns above 4-bit)
_CLIP_MAX_BITS = 4
# Default calibration parameters
_CLIP_NUM_SAMPLES = 128
_CLIP_SEQ_LENGTH = 512
_CLIP_N_GRID = 20
_CLIP_MAX_SHRINK = 0.5
_CLIP_N_FRAMES = 512
_CLIP_BATCH_SIZE = 64


# Available calibration datasets
CALIB_DATASETS = {
    "default": "Built-in (General)",
    "wikitext": "WikiText-2",
    "c4": "C4 (Web Crawl)",
    "code": "Code (StarCoder)",
    "multilingual": "Multilingual (CulturaX)",
    "code_multilingual": "Code + Multilingual",
}


def _load_calibration_data(tokenizer, dataset: str = "code_multilingual",
                           num_samples: int = _CLIP_NUM_SAMPLES,
                           seq_length: int = _CLIP_SEQ_LENGTH):
    """Load calibration data for clip optimization.

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
    import mlx.core as mx

    # Built-in datasets (no download needed)
    if dataset in ("code_multilingual", "code", "multilingual"):
        try:
            return _load_builtin_calibration(
                tokenizer, dataset, num_samples, seq_length
            )
        except Exception as e:
            logger.warning(f"Built-in calibration failed: {e}, "
                           "falling back to mlx-lm default")

    # mlx-lm default
    if dataset == "default":
        try:
            from mlx_lm.quant.utils import load_data
            return load_data(tokenizer, num_samples=num_samples,
                            sequence_length=seq_length)
        except ImportError:
            logger.warning("mlx_lm.quant.utils.load_data not available")
            return None

    # HuggingFace datasets (requires download)
    try:
        return _load_hf_calibration(tokenizer, dataset, num_samples, seq_length)
    except Exception as e:
        logger.warning(f"Failed to load {dataset}: {e}, falling back to built-in")

    # Final fallback: built-in
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

    # Select texts based on dataset type
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

    combined = "\n".join(texts)
    logger.info(f"Built-in calibration: {len(texts)} texts, "
                f"{len(combined) // 1024} KB ({dataset})")

    # Tokenize
    tokens = tokenizer.encode(combined)
    if hasattr(tokens, "input_ids"):
        tokens = tokens.input_ids
    if isinstance(tokens, list):
        tokens = mx.array(tokens)
    if tokens.ndim > 1:
        tokens = tokens.reshape(-1)

    # Chunk into sequences
    usable = (tokens.size // seq_length) * seq_length
    if usable == 0:
        raise ValueError(f"Not enough tokens ({tokens.size} < {seq_length})")
    tokens = tokens[:usable].reshape(-1, seq_length)

    # Random sample
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
        # CulturaX mixed languages (en, ko, zh, ja, de, fr, es)
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
        # Mix: 50% code + 50% multilingual
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

    # Tokenize and chunk
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

    # Chunk into sequences
    n_tokens = tokens.size
    usable = (n_tokens // seq_length) * seq_length
    if usable == 0:
        raise ValueError(f"Not enough tokens from {dataset} (got {n_tokens})")
    tokens = tokens[:usable].reshape(-1, seq_length)

    # Random sample
    n_available = tokens.shape[0]
    if num_samples > 0 and n_available > num_samples:
        indices = mx.random.permutation(n_available)[:num_samples]
        tokens = tokens[indices]

    logger.info(f"Calibration: {tokens.shape[0]} samples × {seq_length} tokens "
                f"from {dataset}")
    return tokens


def _search_best_clip(w, x, group_size: int, bits: int,
                      n_grid: int = _CLIP_N_GRID,
                      max_shrink: float = _CLIP_MAX_SHRINK,
                      n_frames: int = _CLIP_N_FRAMES,
                      batch_size: int = _CLIP_BATCH_SIZE):
    """Find optimal per-group weight clipping using output MSE.

    Adapted from mlx-lm AWQ search_best_clip (awq.py:307-375).
    Searches clip ratios and picks the one minimizing output reconstruction error.

    Args:
        w: Float weight tensor (out_dims, in_dims).
        x: Input activations (n_tokens, in_dims).
        group_size: Quantization group size.
        bits: Target bit width.
        n_grid: Number of grid search steps.
        max_shrink: Maximum shrinkage fraction (0.5 = test down to 50%).
        n_frames: Number of activation frames to use.
        batch_size: Batch size for weight processing.

    Returns:
        Clipped weight tensor (same shape as w).
    """
    # Subsample activations
    x = x.reshape(-1, x.shape[-1])
    stride = max(1, (x.shape[0] + n_frames - 1) // n_frames)
    x = x[::stride]
    x = x.reshape(x.shape[0], -1, group_size)

    bits = int(bits)
    def quantize_func(w_in):
        qw = mx.quantize(w_in, group_size=group_size, bits=bits)
        return mx.dequantize(*qw, group_size=group_size, bits=bits)

    w_init_shape = w.shape
    w_all = mx.flatten(w, 0, w.ndim - 2) if w.ndim > 2 else w

    w_max_all = []
    for b in range(0, w_all.shape[0], batch_size):
        w_batch = w_all[b: b + batch_size]
        group_shape = (w_batch.shape[0], w_batch.shape[-1] // group_size)
        best_error = mx.full(group_shape, float("inf"))
        best_w_max = mx.zeros((*group_shape, 1), dtype=x.dtype)

        w_shape = w_batch.shape
        w_grouped = w_batch.reshape(*w_batch.shape[:-1], -1, group_size)

        # Baseline unquantized output
        out = mx.einsum("bdg,odg->bod", x, w_grouped)
        init_max = w_grouped.abs().max(axis=-1, keepdims=True)

        # Grid search over clip ratios
        # Defer mx.eval to end of grid search for better GPU pipelining
        for i in range(int(max_shrink * n_grid)):
            p = 1 - i / n_grid
            w_max = p * init_max
            w_clipped = mx.clip(w_grouped, -w_max, w_max).reshape(w_shape)

            w_q = quantize_func(w_clipped)
            w_q = w_q.reshape(*w_q.shape[:-1], -1, group_size)

            out_q = mx.einsum("bdg,odg->bod", x, w_q)
            loss = ((out - out_q) ** 2).sum(axis=0)
            loss = loss / out.shape[0]

            improved = loss < best_error
            best_error = mx.where(improved, loss, best_error)
            best_w_max = mx.where(improved[..., None], w_max, best_w_max)

        # Evaluate once at end of grid search (not per step)
        mx.eval(best_w_max, best_error)
        w_max_all.append(best_w_max)

    best_w_max = mx.concatenate(w_max_all, axis=0)
    w_grouped = w_all.reshape(*w_all.shape[:-1], -1, group_size)
    best_w = mx.clip(w_grouped, -best_w_max, best_w_max)
    best_w = best_w.reshape(w_init_shape)
    mx.eval(best_w)
    return best_w


def _run_clip_optimization(model, tokenizer, config, oq_level,
                           progress_callback=None, clip_batch_size=1024,
                           calib_dataset="code_multilingual",
                           num_samples=_CLIP_NUM_SAMPLES,
                           seq_length=_CLIP_SEQ_LENGTH):
    """Run AWQ-style clip optimization on the model before quantization.

    Layer-by-layer forward pass with calibration data, then per-layer
    clip search for layers that will get <=4 bits. Modifies weights in-place.

    Args:
        model: Loaded model (float weights).
        tokenizer: Tokenizer for calibration data.
        config: Model config dict.
        oq_level: Target oQ level.
        progress_callback: Optional fn(phase, pct).

    Returns:
        Number of layers optimized.
    """
    cb = progress_callback or (lambda phase, pct: None)

    # Load calibration data
    calib_data = _load_calibration_data(
        tokenizer, dataset=calib_dataset,
        num_samples=num_samples, seq_length=seq_length,
    )
    if calib_data is None:
        return 0

    predicate = make_predicate(config, oq_level)
    group_size = 64

    # Identify which layers need clip optimization
    clip_targets = {}
    for path, module in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        if not hasattr(module, "to_quantized") or not hasattr(module, "weight"):
            continue
        pred_result = predicate(path, module)
        if pred_result is False:
            continue
        effective_bits = int(pred_result.get("bits", oq_level) if isinstance(pred_result, dict) else oq_level)
        if effective_bits <= _CLIP_MAX_BITS:
            clip_targets[path] = effective_bits

    if not clip_targets:
        return 0

    logger.info(f"oQ{oq_level:g}: clip optimization for {len(clip_targets)} layers")

    # Build attention mask for calib data
    seq_len = calib_data.shape[1]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    # Detect model dtype from embed_tokens (may be nested under language_model)
    mask_dtype = mx.float16
    if hasattr(model, 'embed_tokens'):
        mask_dtype = model.embed_tokens.weight.dtype
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        mask_dtype = model.language_model.model.embed_tokens.weight.dtype
    elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        mask_dtype = model.model.embed_tokens.weight.dtype
    mask = mask.astype(mask_dtype)

    # Embed calibration tokens — search common model structures
    embed_fn = None
    layers = None

    # Standard: model.model.embed_tokens (Llama, Qwen dense)
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        embed_fn = model.model.embed_tokens
        layers = model.model.layers
    # VLM: model.language_model.model.embed_tokens (Qwen3.5 VLM)
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        lm = model.language_model.model
        if hasattr(lm, 'embed_tokens'):
            embed_fn = lm.embed_tokens
            layers = lm.layers
    # Direct: model.embed_tokens
    elif hasattr(model, 'embed_tokens'):
        embed_fn = model.embed_tokens
        layers = model.layers

    if embed_fn is None or layers is None:
        logger.warning("Cannot find embedding layer, skipping clip optimization")
        return 0

    inputs = embed_fn(calib_data)

    # Build position_ids for models that require it (e.g. Qwen3.5 VLM)
    seq_len = calib_data.shape[1]
    position_ids = mx.arange(seq_len)[None, :]  # (1, seq_len)

    optimized = 0
    total_layers = len(layers)

    for layer_idx, block in enumerate(layers):
        # Forward pass through this layer (unquantized)
        # Try multiple signatures until one works
        outputs = None
        for call_args in [
            (inputs, mask, None, position_ids),
            (inputs, mask, None),
            (inputs, mask),
            (inputs, None, mask, None),
            (inputs,),
        ]:
            try:
                outputs = block(*call_args)
                break
            except (TypeError, ValueError, RuntimeError, AttributeError):
                outputs = None
                continue
        if outputs is None:
            logger.warning(
                f"Clip optimization: layer {layer_idx} forward failed, skipping"
            )
            inputs = inputs  # Keep same input for next layer
            continue

        # Collect all sublayers that need clipping in this block
        x_flat = inputs.reshape(-1, inputs.shape[-1])

        # Group sublayers by (bits, input_dim) for batched processing
        from collections import defaultdict
        groups = defaultdict(list)

        for path, module in tree_flatten(block.leaf_modules(), is_leaf=nn.Module.is_module):
            if not hasattr(module, "weight") or not hasattr(module, "to_quantized"):
                continue
            pred_result = predicate(path, module)
            if pred_result is False:
                continue
            effective_bits = int(pred_result.get("bits", oq_level) if isinstance(pred_result, dict) else oq_level)
            if effective_bits > _CLIP_MAX_BITS:
                continue
            w = module.weight
            if x_flat.shape[-1] == w.shape[-1]:
                groups[(effective_bits, w.shape[-1], w.ndim)].append(module)

        # Process each group: stack weights → single clip search → unstack
        for (ebits, in_dim, ndim), modules in groups.items():
            if len(modules) == 1:
                # Single module — direct clip search
                m = modules[0]
                m.weight = _search_best_clip(
                    m.weight, x_flat,
                    group_size=group_size, bits=ebits,
                    batch_size=clip_batch_size,
                )
                optimized += 1
            else:
                # Multiple modules with same bits/dims — stack and batch
                weights = [m.weight for m in modules]
                stacked = mx.concatenate(weights, axis=0)
                clipped = _search_best_clip(
                    stacked, x_flat,
                    group_size=group_size, bits=ebits,
                    batch_size=clip_batch_size,
                )
                # Unstack and assign back
                offset = 0
                for m in modules:
                    rows = m.weight.shape[0]
                    m.weight = clipped[offset:offset + rows]
                    offset += rows
                    optimized += 1

        # Move to next layer
        inputs = outputs
        mx.clear_cache()

        # Progress + ETA
        import time as _time
        elapsed = _time.monotonic() - _opt_start if '_opt_start' in dir() else 0
        if layer_idx == 0:
            _opt_start = _time.monotonic()
            elapsed = 0
        else:
            elapsed = _time.monotonic() - _opt_start
        pct = 30.0 + ((layer_idx + 1) / total_layers) * 30.0
        if layer_idx > 0 and elapsed > 0:
            rate = (layer_idx + 1) / elapsed
            remaining = (total_layers - layer_idx - 1) / rate
            mins = int(remaining // 60)
            secs = int(remaining % 60)
            cb(f"optimizing ({layer_idx + 1}/{total_layers}, {mins}:{secs:02d} remaining)", pct)
        else:
            cb(f"optimizing ({layer_idx + 1}/{total_layers})", pct)

    logger.info(f"oQ{oq_level:g}: clip-optimized {optimized} layers")
    return optimized


# =============================================================================
# Main quantization pipeline
# =============================================================================


def quantize_oq(
    model_path: str,
    output_path: str,
    oq_level: int,
    enable_clip_optimization: bool = True,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    clip_batch_size: int = 1024,
    calib_dataset: str = "code_multilingual",
    text_only: bool = False,
    clip_num_samples: int = _CLIP_NUM_SAMPLES,
    clip_seq_length: int = _CLIP_SEQ_LENGTH,
) -> None:
    """Run oQ quantization: load -> clip-optimize -> quantize -> save.

    Pipeline:
        1. Load model (float weights)
        2. (Optional) AWQ-style clip optimization for <=4 bit layers
        3. quantize_model() with universal predicate
        4. Save

    Args:
        model_path: Path to source model directory.
        output_path: Path for output (must not exist).
        oq_level: Quantization level (2, 3, 4, 6, or 8).
        enable_clip_optimization: Run AWQ-style clip search (requires calibration data).
        progress_callback: Optional fn(phase_name, progress_pct) for updates.
    """
    from mlx_lm.utils import quantize_model

    if oq_level not in OQ_LEVELS:
        raise ValueError(
            f"Invalid oQ level {oq_level}. Must be one of {sorted(OQ_LEVELS)}"
        )

    output = Path(output_path)
    if output.exists():
        raise ValueError(f"Output directory already exists: {output_path}")

    cb = progress_callback or (lambda phase, pct: None)

    # Phase 1: Load
    cb("loading", 5.0)
    logger.info(f"oQ{oq_level:g}: loading {model_path}")

    config_path = Path(model_path) / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    is_vlm = "vision_config" in config and not text_only

    if is_vlm:
        # VLM: use mlx-vlm to load full model (vision + text)
        from mlx_vlm.utils import load_model as vlm_load_model

        model = vlm_load_model(Path(model_path), lazy=True)
        # Load tokenizer separately via mlx-lm (processor loading can fail)
        from mlx_lm import load as lm_load

        _, tokenizer = lm_load(model_path)
        logger.info(f"oQ{oq_level:g}: loaded VLM with mlx-vlm (vision weights preserved)")
    else:
        from mlx_lm import load as lm_load

        model, tokenizer = lm_load(model_path)
        if text_only and "vision_config" in config:
            logger.info(f"oQ{oq_level:g}: text-only mode, vision weights excluded")

    # Dtype normalization is NOT needed here:
    # - mlx-lm/mlx-vlm load() already handles FP8→float via sanitize
    # - bfloat16/float16 models are already in the correct dtype
    # - Accessing all parameters would defeat lazy loading and OOM for large models
    # - Streaming path handles dtype per-tensor via config.torch_dtype

    cb("loading", 25.0)

    # Phase 2: Clip optimization (AWQ-style, output MSE based)
    if enable_clip_optimization and oq_level <= _CLIP_MAX_BITS:
        cb("optimizing", 30.0)
        logger.info(f"oQ{oq_level:g}: running clip optimization")
        _run_clip_optimization(model, tokenizer, config, oq_level, cb,
                               clip_batch_size, calib_dataset,
                               clip_num_samples, clip_seq_length)

    cb("quantizing", 60.0)

    # Phase 3: Quantize with universal predicate
    logger.info(f"oQ{oq_level:g}: quantizing with universal predicate")
    predicate = make_predicate(config, oq_level)
    # oQ level → base bits
    _LEVEL_MAP = {2: 2, 3: 3, 3.5: 3, 4: 4, 5: 5, 6: 6, 8: 8}
    base_bits = int(_LEVEL_MAP.get(oq_level, oq_level))

    base_mode = _mode_for_bits(base_bits)
    base_gs = _gs_for_mode(base_bits, 64)

    model, quantized_config = quantize_model(
        model,
        config,
        group_size=base_gs,
        bits=base_bits,
        mode=base_mode,
        quant_predicate=predicate,
    )
    cb("quantizing", 90.0)

    # Phase 4: Save
    cb("saving", 92.0)
    logger.info(f"oQ{oq_level:g}: saving to {output_path}")

    if is_vlm:
        import glob
        import shutil

        from mlx_vlm.utils import save_config as vlm_save_config
        from mlx_vlm.utils import save_weights as vlm_save_weights

        # save_weights preserves vision + text weights
        vlm_save_weights(output, model, donate_weights=True)
        vlm_save_config(quantized_config, output / "config.json")
        # Copy tokenizer and other files from source
        tokenizer.save_pretrained(str(output))
        src = Path(model_path)
        for pattern in ["*.py", "generation_config.json", "preprocessor_config.json",
                        "processor_config.json", "chat_template.json"]:
            for f in glob.glob(str(src / pattern)):
                shutil.copy(f, output)
    else:
        from mlx_lm.utils import save

        save(str(output), model_path, model, tokenizer, quantized_config)

    cb("saving", 100.0)

    logger.info(f"oQ{oq_level:g}: completed -> {output_path}")
