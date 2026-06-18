# SPDX-License-Identifier: Apache-2.0
"""
JANG model engine for loading quantized MoE+SSM hybrid models.

This engine wraps the jang-tools package to load JANG quantized models
with mixed-precision quantization (attention 6-8-bit, experts 2-4-bit).
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx

from ..api.tool_calling import convert_tools_for_template
from ..api.utils import clean_special_tokens, detect_and_strip_partial
from ..model_discovery import JANG_CONFIG_FILES, _get_jang_has_vision, detect_model_type
from ..models.vlm import VLMModelAdapter
from .base import BaseEngine, GenerationOutput

logger = logging.getLogger(__name__)


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


def _load_jang_config(model_path: Path) -> dict | None:
    """Return the first *parseable* JANG config dict found in model_path, or None.

    Iterates JANG_CONFIG_FILES in order and returns the parsed dict from the
    first file that exists and contains valid JSON. Files that exist but contain
    invalid JSON are logged and skipped so that a parseable candidate later in
    the list can still be found. Returns None only if no file parses successfully.
    """
    for config_name in JANG_CONFIG_FILES:
        config_path = model_path / config_name
        if not config_path.exists():
            continue
        try:
            with open(config_path) as f:
                return json.load(f)
        except Exception as exc:
            logger.warning(
                "JANG config %s exists but failed to parse (%s: %s); "
                "trying next candidate",
                config_path,
                type(exc).__name__,
                exc,
            )
            continue
    return None


class JANGLoader(BaseEngine):
    """
    Engine for loading JANG quantized models via jang-tools.

    JANG models use mixed-precision quantization and require special handling:
    - Mixed bit widths per tensor (read from jang_config.json)
    - Nemotron-H weight renaming (fc1/fc2)
    - Auto bfloat16 for large expert models (512+ experts)
    - VLM support with vision encoder

    Args:
        model_name: Local model directory path (JANG config detection uses
            Path(model_name).exists(), so remote HuggingFace repo IDs will
            not be detected as JANG models).
        scheduler_config: Optional scheduler configuration
        stream_interval: Tokens to batch before streaming (1=every token)
        enable_thinking: Enable thinking mode for reasoning models
        model_settings: Per-model settings accepted for engine-pool constructor
            parity. JANG loads via jang-tools and does not currently consume
            per-model settings (no dflash/MTP/speculative support); non-default
            values are noted in debug logging and otherwise ignored.
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
        if model_settings is not None:
            logger.debug(
                "JANGLoader: model_settings provided but not consumed "
                "(JANG has no dflash/MTP/speculative support); accepted for engine-pool parity"
            )

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

    def _is_jang_v2(self) -> bool:
        """Return True when the model uses the JANG v2 format."""
        cfg = _load_jang_config(Path(self._model_name))
        if cfg is None:
            return False
        try:
            version = str(cfg.get("format_version", "1.0"))
            return int(version.split(".")[0]) >= 2
        except Exception:
            return False

    def _is_jangtq(self) -> bool:
        """Return True when the model uses JANGTQ (TurboQuant) format."""
        cfg = _load_jang_config(Path(self._model_name))
        if cfg is None:
            return False
        return cfg.get("weight_format") == "mxtq" or "JANGTQ" in cfg.get("profile", "")

    def _get_jang_has_vision(self) -> bool | None:
        """Return explicit has_vision metadata from JANG config when present.

        Delegates to model_discovery._get_jang_has_vision so the two copies
        stay in sync. JANG_CONFIG_FILES and the parsing logic live in one place.
        """
        return _get_jang_has_vision(Path(self._model_name))

    def _should_use_vlm_loader(self) -> bool:
        """Choose the JANG loader path, trusting shared discovery rules."""
        explicit_has_vision = self._get_jang_has_vision()
        if explicit_has_vision is not None:
            return explicit_has_vision
        return detect_model_type(Path(self._model_name)) == "vlm"

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


    def _fix_jang_switch_quantization(self) -> None:
        """Fix QuantizedSwitchLinear metadata when jang-tools creates them with
        wrong bits/group_size from config.json vs actual packed weight data.

        JANG v2 models have mixed-precision MoE experts (2-bit routed, 8-bit attention).
        jang-tools' _upgrade_switch_to_quantized uses config.json bits (8) for ALL layers,
        but the actual weights are 2-bit packed uint32. Instead of dequantizing (OOM on
        large models), we just fix the bits/group_size metadata on the QuantizedSwitchLinear
        so gather_qmm unpacks correctly.
        """
        from mlx_lm.models.switch_layers import QuantizedSwitchLinear

        config_path = Path(self._model_name) / "config.json"
        if not config_path.exists():
            return
        with open(config_path) as f:
            model_cfg = json.load(f)
        tc = model_cfg.get("text_config", model_cfg)
        hidden_size = tc.get("hidden_size", 0)
        intermediate_size = tc.get("intermediate_size", 0)
        if not hidden_size or not intermediate_size:
            return

        config_bits = model_cfg.get("quantization", {}).get("bits", 8)

        fixed = 0
        for name, module in self._model.named_modules():
            if not isinstance(module, QuantizedSwitchLinear):
                continue

            w = module.weight  # uint32
            if len(w.shape) != 3:
                continue

            num_experts, out_dim, packed_dim = w.shape
            module_bits = getattr(module, "bits", config_bits)
            module_gs = getattr(module, "group_size", 64)

            # Determine expected input_dims from architecture
            if out_dim == intermediate_size:
                in_dim = hidden_size      # gate_proj, up_proj
            elif out_dim == hidden_size:
                in_dim = intermediate_size # down_proj
            else:
                logger.error(
                    f"JANG switch layer {name}: out_dim={out_dim} matches neither "
                    f"intermediate_size={intermediate_size} nor hidden_size={hidden_size}; "
                    f"cannot determine in_dim — corrupt or unexpected shard"
                )
                raise ValueError(
                    f"JANG switch layer {name}: out_dim={out_dim} is unclassifiable "
                    f"(intermediate_size={intermediate_size}, hidden_size={hidden_size})"
                )

            # Reverse-engineer actual bits by trying each plausible bit-width.
            # For power-of-2 widths (2/4/8) the mapping is exact; for 3/6-bit
            # MLX pads the packed dim, so we accept recovered_in_dim in the
            # range [in_dim, in_dim + elem_per_u32).  If no width matches the
            # layer is genuinely corrupt.
            actual_bits = None
            for candidate_bits in (2, 3, 4, 6, 8):
                elem_per_u32 = 32 // candidate_bits
                recovered = packed_dim * elem_per_u32
                if in_dim <= recovered < in_dim + elem_per_u32:
                    actual_bits = candidate_bits
                    break
            if actual_bits is None:
                raise ValueError(
                    f"JANG switch layer {name}: packed_dim={packed_dim} is not "
                    f"consistent with in_dim={in_dim} for any supported bit-width "
                    f"(2/3/4/6/8) — corrupt or mismatched shard"
                )

            if actual_bits == module_bits:
                continue  # No mismatch

            # Determine actual group_size from scales shape.
            # Non-exact division means the scale tensor is inconsistent with in_dim.
            actual_gs = module_gs
            if module.scales is not None:
                scale_groups = module.scales.shape[-1]
                if scale_groups > 0:
                    if in_dim % scale_groups != 0:
                        raise ValueError(
                            f"JANG switch layer {name}: in_dim={in_dim} is not "
                            f"divisible by scale_groups={scale_groups} — corrupt shard"
                        )
                    actual_gs = in_dim // scale_groups

            logger.info(
                f"Fixing {name}: bits {module_bits}->{actual_bits}, "
                f"group_size {module_gs}->{actual_gs}"
            )

            # Just update the metadata — weights are already correctly packed
            module.bits = actual_bits
            module.group_size = actual_gs
            fixed += 1

        if fixed:
            logger.info(f"JANG switch layer fix: corrected {fixed} layer metadata")

    async def start(self) -> None:
        """Start the engine (load JANG model if not loaded)."""
        if self._loaded:
            return

        from ..engine_core import AsyncEngineCore, EngineConfig
        from ..scheduler import SchedulerConfig

        # Check jang-tools is available
        self._check_jang_tools_available()

        # Read config for lightweight diagnostics only. Some architectures
        # legitimately use attention widths that differ from hidden_size.
        model_path = Path(self._model_name)
        config_path = model_path / "config.json"
        try:
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)

                text_config = config.get("text_config", config)
                hidden_size = text_config.get("hidden_size", 0)
                vocab_size = text_config.get("vocab_size", 0)
                if vocab_size > 0:
                    logger.debug(f"Model vocab_size: {vocab_size}, hidden_size: {hidden_size}")
        except Exception as e:
            logger.debug(f"Config validation skipped: {e}")

        is_vlm = self._should_use_vlm_loader()
        is_jangtq = self._is_jangtq()

        # F1: JANGTQ + VLM is an unsupported combination.  load_jangtq_model is
        # text-only and would silently drop the vision encoder.  Fail loudly here
        # rather than let the model load and produce wrong outputs.
        if is_jangtq and is_vlm:
            from ..exceptions import JANGLoadError
            raise JANGLoadError(
                f"JANGTQ + VLM is not supported: {self._model_name} is "
                "mxtq-quantized AND has vision; load_jangtq_model is text-only "
                "and would silently drop the vision encoder",
                model_name=self._model_name,
            )

        # Import jang_tools in its own narrow try so that ImportError from the
        # jang_tools import itself maps to JANGDependencyError ("install jang[mlx]"),
        # while ImportError raised inside the loader functions (e.g. a missing
        # transformers submodule) propagates to the outer except Exception handler
        # and is reported as a JANGLoadError with the real underlying message.
        try:
            import jang_tools
            from jang_tools.loader import load_jang_model, load_jang_vlm_model
        except ImportError as e:
            from ..exceptions import JANGDependencyError
            raise JANGDependencyError(
                f"Failed to import jang-tools: {e}. Install with: pip install jang[mlx]",
                model_name=self._model_name,
            )

        try:
            # Dispatch to the correct loader.  JANGTQ models use TurboQuantLinear
            # (Metal kernel, no dequant) and load_jangtq_model auto-caps the MLX
            # wired_limit to prevent Metal OOM — must be checked before the
            # standard VLM/LLM paths.
            if is_jangtq:
                from jang_tools.load_jangtq import load_jangtq_model
                logger.info(f"Loading JANGTQ model: {self._model_name}")
                self._model, self._tokenizer = load_jangtq_model(self._model_name)
                self._processor = None
            elif is_vlm:
                logger.info(f"Loading JANG VLM model: {self._model_name}")
                self._model, self._processor = load_jang_vlm_model(self._model_name)
                # Extract tokenizer from processor for token counting
                if hasattr(self._processor, "tokenizer"):
                    self._tokenizer = self._processor.tokenizer
                else:
                    self._tokenizer = self._processor
                # Wrap model with VLMModelAdapter for BatchGenerator compatibility
                logger.info("Wrapping VLM model with VLMModelAdapter")
                self._model = VLMModelAdapter(self._model)
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

            # Dequantize mixed-precision switch layers.
            # JANG models may have 2-bit routed experts while config.json
            # reports bits=8. QuantizedSwitchLinear created by jang-tools
            # uses config bits, causing shape mismatches at inference.
            # Fix: detect and correct mismatched layer metadata.
            # Do NOT catch exceptions here — a failed quant-fix means the model
            # would produce garbage tokens silently; let it propagate to the
            # outer JANGLoadError handler so the load fails loudly.
            #
            # JANGTQ skips this fix: load_jangtq_model uses TurboQuantLinear,
            # not QuantizedSwitchLinear, so there are no switch-layer metadata
            # mismatches to correct — and named_modules() would find no instances.
            if not is_jangtq:
                self._fix_jang_switch_quantization()

            # jang-tools already handles Nemotron-H repair in the v2 path.
            if self._detect_nemotron_h() and not self._is_jang_v2():
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

        except Exception as e:
            from ..exceptions import JANGLoadError
            # Best-effort cleanup to avoid leaking Metal buffers on failed load.
            # Wrapped in its own try/except so cleanup failure doesn't mask the
            # original load error.
            try:
                await self.stop()
            except Exception as cleanup_exc:
                logger.debug(f"Cleanup after failed JANG load raised: {cleanup_exc}")
            try:
                mx.clear_cache()
            except Exception as cache_exc:
                logger.debug(f"mx.clear_cache() after failed JANG load raised: {cache_exc}")
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
        is_partial: bool | None = None,
    ) -> str:
        """Apply chat template to messages."""
        if hasattr(self._tokenizer, 'apply_chat_template'):
            if is_partial is None:
                is_partial = detect_and_strip_partial(messages)
            else:
                # Caller passed an explicit value; still strip the sentinel if present.
                detect_and_strip_partial(messages)
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
        is_partial: bool | None = None,
    ) -> int:
        """Count prompt tokens for chat messages.

        `is_partial` is forwarded to `_apply_chat_template` so that partial
        (assistant-prefill) messages are rendered without a generation-prompt
        suffix, giving an accurate token count for the actual prompt string.
        """
        messages = self._preprocess_messages(messages)
        template_tools = convert_tools_for_template(tools) if tools else None
        prompt = self._apply_chat_template(
            messages, template_tools, chat_template_kwargs=chat_template_kwargs,
            is_partial=is_partial,
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
