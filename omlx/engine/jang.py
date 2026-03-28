# SPDX-License-Identifier: Apache-2.0
"""
JANG model engine for loading quantized MoE+SSM hybrid models.

This engine wraps the jang-tools package to load JANG quantized models
with mixed-precision quantization (attention 6-8-bit, experts 2-4-bit).
"""

from __future__ import annotations

import copy
import gc
import json
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ..api.tool_calling import convert_tools_for_template
from ..api.utils import clean_special_tokens, detect_and_strip_partial
from ..models.vlm import VLMModelAdapter
from .base import BaseEngine, GenerationOutput

logger = logging.getLogger(__name__)


_auto_image_processor_patched = False


def _patch_auto_image_processor():
    """Replace the AutoImageProcessor dummy with the real class.

    In PyTorch-free environments, ``transformers`` replaces
    ``AutoImageProcessor`` with a dummy that raises ``ImportError``.
    The actual image-processor classes only need PIL, so we import the
    real ``AutoImageProcessor`` from the internal auto module (which is
    not behind the backend gate) and swap it into the public namespace.

    Must be called **before** any code that does
    ``from transformers import AutoImageProcessor``.
    """
    global _auto_image_processor_patched
    if _auto_image_processor_patched:
        return
    _auto_image_processor_patched = True

    try:
        from transformers import AutoImageProcessor

        if "dummy" not in getattr(AutoImageProcessor, "__module__", ""):
            return  # Real class already available — nothing to do
    except ImportError:
        pass

    try:
        from transformers.models.auto.image_processing_auto import (
            AutoImageProcessor as _RealAutoImageProcessor,
        )
        import transformers

        transformers.AutoImageProcessor = _RealAutoImageProcessor
        logger.debug("Bypassed AutoImageProcessor backend gate for MLX")
    except ImportError:
        logger.debug("Real AutoImageProcessor not available in this transformers version")


def _infer_bits_and_group_size(
    weight_shape: tuple, scales_shape: tuple
) -> tuple[int | None, int | None]:
    """Infer quantization bits and group_size from weight/scales shapes.

    JANG models use mixed-precision quantization — different tensors have
    different bit widths.  The actual bits can be recovered from the ratio
    of packed uint32 columns (weight) to group count (scales).

    See: JANG Integration Guide § "Inferring Bit Width Per Tensor"
    """
    w_cols = weight_shape[-1]  # packed columns
    s_cols = scales_shape[-1]  # number of groups
    for bits in [2, 3, 4, 5, 6, 8]:
        elem_per_u32 = 32 // bits
        in_features = w_cols * elem_per_u32
        if s_cols > 0:
            group_size = in_features // s_cols
            if group_size > 0 and group_size * s_cols == in_features:
                return bits, group_size
    return None, None


class _TokenizerWrapper:
    """
    Wrapper for tokenizers that don't have an encode() method.

    Some VLM processors (like Qwen3VLProcessor) return a tokenizer that
    doesn't have encode() directly. This wrapper delegates to the underlying
    HF tokenizer's encode method.
    """

    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer

    def __getattr__(self, name: str) -> Any:
        # Delegate attribute access to the underlying tokenizer
        return getattr(self._tokenizer, name)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        # Try different ways to get encode
        if hasattr(self._tokenizer, "encode"):
            return self._tokenizer.encode(text)
        if hasattr(self._tokenizer, "tokenize"):
            # Fallback: use tokenize and map to ids
            tokens = self._tokenizer.tokenize(text)
            if hasattr(self._tokenizer, "convert_tokens_to_ids"):
                return self._tokenizer.convert_tokens_to_ids(tokens)
        # Try calling the tokenizer directly (e.g. HF processors)
        if callable(self._tokenizer):
            result = self._tokenizer(text)
            if isinstance(result, dict) and "input_ids" in result:
                return list(result["input_ids"])
        raise TypeError(
            f"Cannot encode text: tokenizer {type(self._tokenizer).__name__} "
            f"has no encode(), tokenize(), or __call__ returning input_ids"
        )


class JANGLoader(BaseEngine):
    """
    Engine for loading JANG quantized models via jang-tools.

    JANG models use mixed-precision quantization and require special handling:
    - Mixed bit widths per tensor (read from jang_config.json)
    - Nemotron-H weight renaming (fc1/fc2)
    - Auto bfloat16 for large expert models (512+ experts)
    - VLM support with vision encoder

    Args:
        model_name: HuggingFace model name or local path
        scheduler_config: Optional scheduler configuration
        stream_interval: Tokens to batch before streaming (1=every token)
        enable_thinking: Enable thinking mode for reasoning models
        model_settings: Optional per-model settings for post-load transforms
    """

    def __init__(
        self,
        model_name: str,
        scheduler_config: Any | None = None,
        stream_interval: int = 1,
        enable_thinking: bool | None = None,
        model_settings: Any | None = None,
    ):
        self._model_name = model_name
        self._scheduler_config = scheduler_config
        self._stream_interval = stream_interval
        self._enable_thinking = enable_thinking
        self._model_settings = model_settings

        self._model = None
        self._tokenizer = None
        self._engine = None
        self._loaded = False

        self._processor = None

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def tokenizer(self) -> Any:
        """Get the tokenizer."""
        return self._tokenizer

    @property
    def model_type(self) -> str | None:
        """Get the model type from config (e.g., 'gpt_oss', 'llama', 'qwen2')."""
        if self._model is None:
            return None
        try:
            if hasattr(self._model, 'config'):
                config = self._model.config
                if hasattr(config, 'model_type'):
                    model_type = config.model_type
                    return model_type if isinstance(model_type, str) else None
                elif isinstance(config, dict):
                    model_type = config.get('model_type')
                    return model_type if isinstance(model_type, str) else None
            if hasattr(self._model, 'args'):
                args = self._model.args
                if hasattr(args, 'model_type'):
                    model_type = args.model_type
                    return model_type if isinstance(model_type, str) else None
        except Exception as e:
            logger.debug(f"Error getting model_type: {e}")
        return None

    def _check_jang_tools_available(self) -> None:
        """Check if jang-tools is installed."""
        try:
            import jang_tools  # noqa: F401
        except ImportError:
            from ..exceptions import JANGDependencyError
            raise JANGDependencyError(
                "jang-tools package not found. Install with: pip install jang[mlx]",
                model_name=self._model_name,
            )

    def _get_config_dict(self) -> dict:
        """Get model config as a dict, handling both dict and object configs."""
        if self._model is None:
            return {}
        config = getattr(self._model, 'config', None)
        if config is None:
            return {}
        if isinstance(config, dict):
            return config
        # Config is an object — try to extract relevant fields
        result = {}
        for attr in ('architectures', 'num_local_experts', 'num_experts',
                      'n_routed_experts', 'hidden_size', 'model_type',
                      'text_config'):
            if hasattr(config, attr):
                result[attr] = getattr(config, attr)
        return result

    def _detect_nemotron_h(self) -> bool:
        """Check if model is Nemotron-H architecture."""
        if self._model is None:
            return False
        try:
            config = self._get_config_dict()
            for a in config.get('architectures', []):
                if 'Nemotron' in a:
                    return True
            return False
        except Exception:
            return False

    def _needs_bfloat16(self) -> bool:
        """Check if model needs bfloat16 (512+ experts, hidden>=4096)."""
        if self._model is None:
            return False
        try:
            config = self._get_config_dict()
            text_cfg = config.get('text_config', config)
            if isinstance(text_cfg, dict):
                cfg = text_cfg
            else:
                cfg = config
            n_experts = cfg.get('num_local_experts',
                        cfg.get('num_experts',
                        cfg.get('n_routed_experts', 0)))
            hidden_size = cfg.get('hidden_size', 0)
            if n_experts >= 512 and hidden_size >= 4096:
                return True
        except Exception as e:
            logger.debug(f"Error checking bfloat16 requirements: {e}")
        return False

    def _fix_jang_quantized_bits(self) -> None:
        """Fix per-tensor quantization bits for JANG mixed-precision models.

        After loading with uniform default bits (from config.json), each
        QuantizedLinear / QuantizedEmbedding / QuantizedSwitchLinear layer
        may have the wrong bits and group_size.  This infers the correct
        values from the actual weight and scales tensor shapes.

        See: JANG Integration Guide § "Setting Up QuantizedLinear Per Tensor"
        """
        if self._model is None:
            return
        fixed = 0
        for name, module in self._model.named_modules():
            if not (
                hasattr(module, "bits")
                and hasattr(module, "weight")
                and hasattr(module, "scales")
            ):
                continue
            bits, gs = _infer_bits_and_group_size(
                module.weight.shape, module.scales.shape
            )
            if bits is not None:
                module.bits = bits
                module.group_size = gs
                fixed += 1
        if fixed:
            logger.info(f"Fixed JANG quantization bits for {fixed} layers")

    def _fix_nemotron_h_weights(self) -> None:
        """Fix Nemotron-H weights after JANG loading.

        JANG v2 stores Nemotron-H weights with different naming and quantized
        gate weights that mlx-lm's nemotron_h.py cannot handle directly.

        The gate weights are nn.Linear in mlx-lm's model skeleton, but JANG
        stores them as quantized uint32. When jang-tools loads with strict=False,
        the gate.weight (uint32) is loaded but gate.scales and gate.biases are
        dropped because nn.Linear doesn't declare them. We must read the
        scales/biases from the original safetensors files to dequantize.
        """
        model_path = Path(self._model_name)
        use_bfloat16 = self._needs_bfloat16()
        target_dtype = mx.bfloat16 if use_bfloat16 else mx.float16

        # Read the shard index to find which files contain gate weights
        index_path = model_path / "model.safetensors.index.json"
        if not index_path.exists():
            # Try consolidated format
            index_path = model_path / "consolidated.safetensors.index.json"
        if not index_path.exists():
            logger.warning("Nemotron-H: no safetensors index found, skipping gate fixup")
            return

        with open(index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})

        # Find all gate weight/scales/biases keys in the safetensors index
        # Group by gate prefix (e.g., "backbone.layers.0.mixer.gate")
        gate_parts: dict[str, dict[str, str]] = {}  # prefix -> {suffix -> shard_file}
        for key, shard in weight_map.items():
            if ".gate." in key:
                prefix = key[:key.index(".gate.") + len(".gate")]
                suffix = key[key.index(".gate.") + len(".gate."):]
                if prefix not in gate_parts:
                    gate_parts[prefix] = {}
                gate_parts[prefix][suffix] = shard

        if not gate_parts:
            logger.info("Nemotron-H: no gate weights found in index, skipping")
            return

        # Load gate tensors from safetensors and dequantize
        dequantized_weights: list[tuple[str, mx.array]] = []
        # Cache loaded shards to avoid re-reading
        shard_cache: dict[str, dict[str, mx.array]] = {}

        for prefix, parts in gate_parts.items():
            if "weight" not in parts:
                continue
            if "scales" not in parts or "biases" not in parts:
                # Gate is not quantized (no scales/biases), skip
                continue

            # Load the required tensors from safetensors
            tensors: dict[str, mx.array] = {}
            for suffix in ("weight", "scales", "biases"):
                full_key = f"{prefix}.{suffix}"
                shard_file = parts[suffix]
                if shard_file not in shard_cache:
                    shard_cache[shard_file] = mx.load(str(model_path / shard_file))
                tensors[suffix] = shard_cache[shard_file][full_key]

            gate_weight = tensors["weight"]
            scales = tensors["scales"]
            biases = tensors["biases"]

            # Dequantize by trying bit widths (gate is typically 8-bit CRITICAL tier)
            dequantized = None
            for bits in [8, 6, 4, 3, 2]:
                elem_per_u32 = 32 // bits
                real_cols = gate_weight.shape[-1] * elem_per_u32
                gs = real_cols // scales.shape[-1]
                if gs > 0 and gs * scales.shape[-1] == real_cols:
                    dequantized = mx.dequantize(
                        gate_weight, scales, biases, gs, bits
                    )
                    dequantized = dequantized.astype(target_dtype)
                    logger.info(
                        f"Nemotron-H: dequantized {prefix}.weight "
                        f"({bits}-bit, group_size={gs}) "
                        f"{gate_weight.shape} -> {dequantized.shape}"
                    )
                    break

            if dequantized is not None:
                dequantized_weights.append((f"{prefix}.weight", dequantized))
            else:
                logger.warning(
                    f"Nemotron-H: could not dequantize {prefix}, "
                    f"weight={gate_weight.shape}, scales={scales.shape}"
                )

        # Free shard cache
        del shard_cache

        if dequantized_weights:
            self._model.load_weights(dequantized_weights, strict=False)
            logger.info(
                f"Nemotron-H: dequantized {len(dequantized_weights)} "
                f"gate weights to {target_dtype}"
            )
        else:
            logger.info("Nemotron-H: no gate weights needed dequantization")

        logger.info("Nemotron-H: weight fixup complete")

    def _load_jang_vlm_manual(self):
        """Load JANG VLM following the JANG creator's reference implementation.

        Uses mlx_vlm for model architecture + processor and jang_tools for
        weight loading + per-tensor bit fixing.  Does **not** need PyTorch
        (unlike ``load_jang_vlm_model`` which uses ``AutoImageProcessor``).

        Returns:
            ``(model, processor)`` — the VLM model and its processor.
        """
        import gc

        import mlx.nn as nn
        from jang_tools.loader import _fix_quantized_bits, _get_v2_weight_files
        from mlx_vlm.utils import (
            get_model_and_args,
            load_processor,
            skip_multimodal_module,
            update_module_configs,
        )

        from .vlm import _patch_video_processor_bug

        _patch_video_processor_bug()

        model_path = Path(self._model_name)
        vlm_config = json.loads((model_path / "config.json").read_text())

        # Build model skeleton from mlx_vlm model registry
        model_class, _ = get_model_and_args(config=vlm_config)
        model_config = model_class.ModelConfig.from_dict(vlm_config)
        modules = ["text", "vision", "perceiver", "projector", "audio"]
        model_config = update_module_configs(
            model_config, model_class, vlm_config, modules
        )

        # update_module_configs may not propagate every parameter from the
        # raw JSON into the typed config objects (dataclass fields that
        # don't exist yet are silently dropped).  Back-fill anything that
        # is still None on text_config from the raw dict — this is
        # critical for rope_theta, rope_scaling, head_dim, etc.
        raw_text_cfg = vlm_config.get("text_config", vlm_config)
        text_cfg = getattr(model_config, "text_config", None)
        if text_cfg is not None:
            for key, value in raw_text_cfg.items():
                if value is not None and not isinstance(value, dict):
                    if getattr(text_cfg, key, None) is None:
                        try:
                            setattr(text_cfg, key, value)
                        except Exception:
                            pass
            # Also check top-level VLM config as final fallback
            for key in ("rope_theta", "rope_scaling", "head_dim"):
                if getattr(text_cfg, key, None) is None and key in vlm_config:
                    try:
                        setattr(text_cfg, key, vlm_config[key])
                    except Exception:
                        pass

        model = model_class.Model(model_config)

        # Fix RoPE layers that were created without a base frequency.
        # This happens when update_module_configs / ModelConfig.from_dict
        # drops rope_theta (frozen dataclass, missing field, version skew).
        rope_theta = (
            raw_text_cfg.get("rope_theta")
            or raw_text_cfg.get("rope_parameters", {}).get("rope_theta")
            or vlm_config.get("rope_theta")
            or vlm_config.get("rope_parameters", {}).get("rope_theta")
        )
        if rope_theta:
            import mlx.nn as _nn

            for _name, _mod in model.named_modules():
                if isinstance(_mod, _nn.RoPE) and _mod._base is None:
                    _mod._base = float(rope_theta)
                    logger.debug(
                        f"Set rope_theta={rope_theta} on {_name}"
                    )

        # Pre-quantize layers that JANG will fill with mixed-bit weights.
        # Skip vision/multimodal layers (kept fp16) and MoE gate (not gate_proj).
        def _class_pred(p, m):
            if skip_multimodal_module(p):
                return False
            if "gate" in p and "gate_proj" not in p:
                return False
            return hasattr(m, "to_quantized")

        quant_cfg = vlm_config.get("quantization", {})
        qbits = quant_cfg.get("bits", 4)
        qgs = quant_cfg.get("group_size", 64)
        nn.quantize(
            model, group_size=qgs, bits=qbits, class_predicate=_class_pred
        )

        # Load JANG v2 safetensor shards
        for sf in _get_v2_weight_files(model_path):
            w = mx.load(str(sf))
            clean = {
                k: v
                for k, v in w.items()
                if not k.endswith(".importance")
                and not k.startswith("mtp.")
                and "activation_scale" not in k
                and "scale_inv" not in k
            }
            try:
                clean = model.sanitize(clean)
            except Exception:
                pass
            model.load_weights(list(clean.items()), strict=False)
            del clean, w
            gc.collect()

        # Correct per-tensor bit widths
        _fix_quantized_bits(model, {})

        # bfloat16 for numerical stability on large models
        self._model = model  # needed by _needs_bfloat16
        if self._needs_bfloat16():
            logger.info("Applying bfloat16 for large expert model")
            model.set_dtype(mx.bfloat16)

        mx.eval(model.parameters())

        model.config = model_config

        # Try loading the full processor (needed for image/OCR input).
        # Some model-specific processors (e.g. PixtralProcessor) are also
        # gated behind PyTorch in transformers.  When that happens, fall
        # back to a plain tokenizer — text inference still works.
        try:
            processor = load_processor(model_path, processor_config=vlm_config)
            logger.info("JANG VLM loaded via manual path (full processor)")
        except ImportError as proc_err:
            logger.warning(
                f"VLM processor unavailable ({proc_err}), "
                f"loading tokenizer only (text works, image input disabled)"
            )
            from mlx_lm.utils import load_tokenizer

            processor = load_tokenizer(model_path)

        return model, processor

    def _detect_mistral4_text(self) -> bool:
        """Check if model has a mistral4 text backbone (MLA + MoE).

        Mistral-Small-4 and similar models use ``mistral3`` as the top-level
        model_type (VLM wrapper) with ``text_config.model_type == "mistral4"``.
        The ``mistral4`` architecture uses Multi-Latent Attention (MLA) which
        is architecturally identical to DeepSeek-V3 but not yet supported by
        mlx-lm's ``mistral3`` module (which falls back to ``llama``).
        """
        model_path = Path(self._model_name)
        try:
            config_path = model_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                tc = config.get("text_config", {})
                return tc.get("model_type") == "mistral4"
        except Exception:
            pass
        return False

    def _load_jang_mistral4(self):
        """Load a JANG model with mistral4 text backbone using deepseek_v3.

        The mistral4 text architecture uses MLA (Multi-Latent Attention) which
        is architecturally identical to DeepSeek-V3.  Since mlx-lm does not
        yet have a native ``mistral4`` model, we load using ``deepseek_v3``
        with custom weight mapping:

        1. Build deepseek_v3 model skeleton from text_config
        2. Pre-quantize the skeleton at the config-specified bit width
        3. Load JANG v2 weights, stripping the ``language_model.`` prefix
        4. Split ``kv_b_proj`` into ``embed_q`` + ``unembed_out`` (MLA KV
           decomposition), re-quantizing at the pre-quant bits
        5. Fix per-tensor quantization bits

        Returns:
            ``(model, tokenizer)``
        """
        import time

        from jang_tools.loader import _fix_quantized_bits, _get_v2_weight_files
        from mlx_lm.models.deepseek_v3 import Model, ModelArgs
        from mlx_lm.utils import load_tokenizer

        model_path = Path(self._model_name)
        config = json.loads((model_path / "config.json").read_text())
        tc = config["text_config"]

        # Map mistral4 config → deepseek_v3 config
        tc["rope_scaling"] = tc.pop("rope_parameters", None)
        if tc.get("rope_scaling"):
            tc["rope_theta"] = tc["rope_scaling"].get("rope_theta", 10000.0)
        tc["model_type"] = "deepseek_v3"

        args = ModelArgs.from_dict(tc)
        model = Model(args)

        # Pre-quantize at config-specified bits
        quant = config.get("quantization", {"group_size": 64, "bits": 4})
        quant_bits = quant.get("bits", 4)
        quant_gs = quant.get("group_size", 64)
        nn.quantize(model, group_size=quant_gs, bits=quant_bits)

        start = time.perf_counter()
        for sf in _get_v2_weight_files(model_path):
            w = mx.load(str(sf))
            clean = {}
            for k, v in w.items():
                if (
                    k.endswith(".importance")
                    or k.startswith("mtp.")
                    or "activation_scale" in k
                    or "scale_inv" in k
                ):
                    continue
                # Strip language_model. prefix (keep model.)
                if k.startswith("language_model."):
                    k = k[len("language_model."):]
                clean[k] = v

            # Split kv_b_proj → embed_q + unembed_out for MLA
            for layer_idx in range(args.num_hidden_layers):
                pfx = f"model.layers.{layer_idx}.self_attn"
                wkey = f"{pfx}.kv_b_proj.weight"
                skey = f"{pfx}.kv_b_proj.scales"
                bkey = f"{pfx}.kv_b_proj.biases"
                if wkey not in clean or skey not in clean:
                    continue

                qw = clean.pop(wkey)
                scales = clean.pop(skey)
                biases = clean.pop(bkey, mx.zeros_like(scales))

                # Infer original quantization from shapes
                orig_bits = (qw.shape[-1] * 32) // args.kv_lora_rank
                orig_gs = args.kv_lora_rank // scales.shape[-1]

                # Dequantize → reshape → split
                dq = mx.dequantize(qw, scales, biases, orig_gs, orig_bits)
                head_dim = args.qk_nope_head_dim + args.v_head_dim
                dq = dq.reshape(args.num_attention_heads, head_dim, -1)
                wk = mx.contiguous(
                    dq[:, : args.qk_nope_head_dim, :].swapaxes(-1, -2)
                )
                wv = mx.contiguous(dq[:, args.qk_nope_head_dim :, :])

                # Re-quantize at the pre-quant bits (not original)
                wk_q, wk_s, wk_b = mx.quantize(wk, quant_gs, quant_bits)
                wv_q, wv_s, wv_b = mx.quantize(wv, quant_gs, quant_bits)

                clean[f"{pfx}.embed_q.weight"] = wk_q
                clean[f"{pfx}.embed_q.scales"] = wk_s
                clean[f"{pfx}.embed_q.biases"] = wk_b
                clean[f"{pfx}.unembed_out.weight"] = wv_q
                clean[f"{pfx}.unembed_out.scales"] = wv_s
                clean[f"{pfx}.unembed_out.biases"] = wv_b

            model.load_weights(list(clean.items()), strict=False)
            del clean, w
            gc.collect()

        _fix_quantized_bits(model, {})
        mx.eval(model.parameters())

        elapsed = time.perf_counter() - start
        logger.info(
            f"JANG mistral4→deepseek_v3 loaded in {elapsed:.1f}s: "
            f"{self._model_name}"
        )

        tokenizer = load_tokenizer(
            model_path,
            eos_token_ids=config.get("eos_token_id", None),
        )
        return model, tokenizer

    def _is_vlm_model(self) -> bool:
        """Check if this is a VLM model.

        Checks (in order):
        1. jang_config.json.architecture.has_vision (primary JANG metadata)
        2. preprocessor_config.json existence (VLM indicator)
        3. config.json.vision_config (existing MLX pattern)
        4. Model config for vision_config or VLM architectures (after load)
        """
        model_path = Path(self._model_name)

        # Check 1: JANG config architecture.has_vision (primary JANG metadata)
        try:
            for cfg_name in ("jang_config.json", "jjqf_config.json", "jang_cfg.json"):
                jang_config_path = model_path / cfg_name
                if jang_config_path.exists():
                    with open(jang_config_path) as f:
                        jang_config = json.load(f)
                    architecture = jang_config.get("architecture", {})
                    if isinstance(architecture, dict) and architecture.get("has_vision") is True:
                        return True
                    break  # Found JANG config, no need to check others
        except Exception:
            pass

        # Check 2: preprocessor_config.json existence (VLM indicator)
        try:
            preprocessor_path = model_path / "preprocessor_config.json"
            if preprocessor_path.exists():
                return True
        except Exception:
            pass

        # Check 3: config.json.vision_config (existing MLX pattern, fallback)
        try:
            config_path = model_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                if "vision_config" in config:
                    return True
                # Also check architectures for VLM hints
                arch = config.get("architectures", [])
                for a in arch:
                    if "VLM" in a or "VL" in a or "Vision" in a:
                        return True
        except Exception:
            pass

        # Check 4: Model config (after load)
        if self._model is not None:
            try:
                config = getattr(self._model, 'config', {})
                if isinstance(config, dict):
                    if "vision_config" in config:
                        return True
                    arch = config.get("architectures", [])
                    for a in arch:
                        if "VLM" in a or "VL" in a or "Vision" in a:
                            return True
            except Exception:
                pass

        return False

    async def start(self) -> None:
        """Start the engine (load JANG model if not loaded)."""
        if self._loaded:
            return

        from ..engine_core import AsyncEngineCore, EngineConfig
        from ..scheduler import SchedulerConfig

        # Check jang-tools is available
        self._check_jang_tools_available()

        # Validate that config.json and jang_config.json are consistent
        model_path = Path(self._model_name)
        config_path = model_path / "config.json"
        try:
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)

                # Check for common inconsistencies
                text_config = config.get("text_config", config)

                # Get hidden_size from text_config
                hidden_size = text_config.get("hidden_size", 0)
                num_attention_heads = text_config.get("num_attention_heads", 1)
                head_dim = text_config.get("head_dim", 0)

                # Verify consistency: hidden_size should equal num_attention_heads * head_dim
                if hidden_size > 0 and num_attention_heads > 0 and head_dim > 0:
                    expected_hidden = num_attention_heads * head_dim
                    if abs(hidden_size - expected_hidden) > 10:  # Allow small tolerance
                        logger.warning(
                            f"Potential config inconsistency: hidden_size={hidden_size} "
                            f"but num_attention_heads={num_attention_heads} * head_dim={head_dim} "
                            f"= {expected_hidden}. This may cause loading issues."
                        )

                # Check vocab_size consistency with embed_tokens
                vocab_size = text_config.get("vocab_size", 0)
                if vocab_size > 0:
                    logger.debug(f"Model vocab_size: {vocab_size}, hidden_size: {hidden_size}")
        except Exception as e:
            logger.debug(f"Config validation skipped: {e}")

        # Read the full config for jang-tools
        model_config = None
        try:
            if config_path.exists():
                with open(config_path) as f:
                    model_config = json.load(f)
                logger.info(
                    f"Model config loaded: model_type={model_config.get('model_type')}, "
                    f"architectures={model_config.get('architectures')}"
                )
                if "text_config" in model_config:
                    tc = model_config["text_config"]
                    logger.info(
                        f"text_config: hidden_size={tc.get('hidden_size')}, "
                        f"vocab_size={tc.get('vocab_size')}, "
                        f"num_hidden_layers={tc.get('num_hidden_layers')}"
                    )
        except Exception as e:
            logger.debug(f"Could not read full config: {e}")

        try:
            import jang_tools
            from jang_tools.loader import load_jang_model

            # ── Loader selection ─────────────────────────────────
            # 1. mistral4 text backbone: custom deepseek_v3-based loader
            #    (mlx-lm's mistral3 module falls back to llama which
            #    cannot handle MLA attention)
            # 2. All other JANG models: use load_jang_model (text path)
            #    even for VLMs — the text loader produces correct weights
            #    and text inference works.  The VLM path (load_jang_vlm_model
            #    / _load_jang_vlm_manual) can corrupt weights for models
            #    whose mlx_vlm model registry differs from mlx_lm.
            if self._detect_mistral4_text():
                logger.info(
                    f"Loading JANG mistral4 model via deepseek_v3: "
                    f"{self._model_name}"
                )
                self._model, self._tokenizer = self._load_jang_mistral4()
                self._processor = None
            else:
                logger.info(f"Loading JANG model: {self._model_name}")
                self._model, self._tokenizer = load_jang_model(self._model_name)
                self._processor = None

            # Wrap tokenizer if needed (some VLM processors don't have encode method)
            if self._tokenizer is not None and not hasattr(self._tokenizer, "encode"):
                if callable(self._tokenizer):
                    logger.debug("Wrapping callable tokenizer with encode() method")
                    self._tokenizer = _TokenizerWrapper(self._tokenizer)
                else:
                    logger.warning(
                        "Tokenizer has no encode() method and is not callable; "
                        "token counting may fail"
                    )

            # Apply Nemotron-H weight fixups before any dtype changes
            if self._detect_nemotron_h():
                logger.info("Detected Nemotron-H architecture, applying weight fixups")
                self._fix_nemotron_h_weights()

            # Auto-switch to bfloat16 for large expert models
            if self._needs_bfloat16():
                logger.info("Large expert model detected (512+ experts, hidden>=4096), using bfloat16")
                self._model.set_dtype(mx.bfloat16)

            # Create engine config (copy to avoid mutating the shared instance)
            scheduler_config = copy.copy(self._scheduler_config) if self._scheduler_config else SchedulerConfig()
            scheduler_config.model_name = self._model_name  # Ensure cache isolation per model
            engine_config = EngineConfig(
                model_name=self._model_name,
                scheduler_config=scheduler_config,
                stream_interval=self._stream_interval,
            )

            # Create async engine
            self._engine = AsyncEngineCore(
                model=self._model,
                tokenizer=self._tokenizer,
                config=engine_config,
            )

            await self._engine.engine.start()
            self._loaded = True
            logger.info(f"JANGLoader loaded: {self._model_name}")

        except ImportError as e:
            from ..exceptions import JANGDependencyError
            raise JANGDependencyError(
                f"Failed to import jang-tools: {e}. Install with: pip install jang[mlx]",
                model_name=self._model_name,
            )
        except Exception as e:
            from ..exceptions import JANGLoadError
            raise JANGLoadError(
                f"Failed to load JANG model {self._model_name}: {e}",
                model_name=self._model_name,
            ) from e

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        if self._engine:
            await self._engine.stop()
            self._engine.engine.close()
        self._engine = None
        self._model = None
        self._tokenizer = None
        self._loaded = False
        logger.info("JANGLoader stopped")

    def _preprocess_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Preprocess messages for model-specific formats."""
        try:
            from ..adapter.harmony import preprocess_harmony_messages
            if self.model_type == "gpt_oss":
                return preprocess_harmony_messages(messages)
        except ImportError:
            pass
        return messages

    def _apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Apply chat template to messages."""
        if hasattr(self._tokenizer, 'apply_chat_template'):
            is_partial = detect_and_strip_partial(messages)
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": not is_partial,
            }
            if is_partial:
                template_kwargs["continue_final_message"] = True
            if tools:
                template_kwargs["tools"] = tools
            if self._enable_thinking is not None:
                template_kwargs["enable_thinking"] = self._enable_thinking
            if chat_template_kwargs:
                template_kwargs.update(chat_template_kwargs)

            try:
                return self._tokenizer.apply_chat_template(messages, **template_kwargs)
            except TypeError:
                if chat_template_kwargs:
                    for key in chat_template_kwargs:
                        template_kwargs.pop(key, None)
                template_kwargs.pop("tools", None)
                template_kwargs.pop("enable_thinking", None)
                return self._tokenizer.apply_chat_template(messages, **template_kwargs)
            except Exception as e:
                logger.error(f"Chat template rendering failed: {e}")
                raise
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            return prompt + "\nassistant:"

    def count_chat_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> int:
        """Count prompt tokens for chat messages."""
        messages = self._preprocess_messages(messages)
        template_tools = convert_tools_for_template(tools) if tools else None
        prompt = self._apply_chat_template(
            messages, template_tools, chat_template_kwargs=chat_template_kwargs
        )
        return len(self._tokenizer.encode(prompt))

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """Generate a complete response (non-streaming)."""
        if not self._loaded:
            await self.start()

        from ..request import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            stop=stop or [],
            thinking_budget=kwargs.get("thinking_budget", None),
        )

        output = await self._engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
        )

        text = clean_special_tokens(output.output_text)

        return GenerationOutput(
            text=text,
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens,
            finish_reason=output.finish_reason,
            tool_calls=output.tool_calls,
            cached_tokens=output.cached_tokens,
        )

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs,
    ) -> Any:
        """Stream generation token by token."""
        if not self._loaded:
            await self.start()

        from ..request import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            stop=stop or [],
            thinking_budget=kwargs.get("thinking_budget", None),
        )

        request_id = await self._engine.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
        )

        finished_normally = False
        try:
            async for output in self._engine.stream_outputs(request_id):
                text = clean_special_tokens(output.output_text)

                if output.finished:
                    finished_normally = True

                yield GenerationOutput(
                    text=text,
                    new_text=output.new_text,
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    finished=output.finished,
                    finish_reason=output.finish_reason,
                    tool_calls=output.tool_calls,
                    cached_tokens=output.cached_tokens,
                )
        except GeneratorExit:
            logger.info(f"[stream_generate] GeneratorExit caught for request {request_id}")
        finally:
            if not finished_normally:
                logger.info(f"[stream_generate] Aborting request {request_id}")
                await self._engine.abort_request(request_id)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """Chat completion (non-streaming)."""
        if not self._loaded:
            await self.start()

        messages = self._preprocess_messages(messages)
        template_tools = convert_tools_for_template(tools) if tools else None
        ct_kwargs = kwargs.pop("chat_template_kwargs", None)
        prompt = self._apply_chat_template(
            messages, template_tools, chat_template_kwargs=ct_kwargs
        )

        return await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            **kwargs,
        )

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> Any:
        """Stream chat completion token by token."""
        if not self._loaded:
            await self.start()

        messages = self._preprocess_messages(messages)
        template_tools = convert_tools_for_template(tools) if tools else None
        ct_kwargs = kwargs.pop("chat_template_kwargs", None)
        prompt = self._apply_chat_template(
            messages, template_tools, chat_template_kwargs=ct_kwargs
        )

        async for output in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            **kwargs,
        ):
            yield output

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        stats = {
            "engine_type": "jang",
            "model_name": self._model_name,
            "loaded": self._loaded,
            "stream_interval": self._stream_interval,
        }
        if self._engine:
            stats.update(self._engine.get_stats())
        return stats

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get cache statistics."""
        if self._engine:
            return self._engine.get_cache_stats()
        return None
