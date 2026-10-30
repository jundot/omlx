import json
import logging
import re

import mlx.core as mx
import mlx.nn as nn

from omlx.patches.mlx_vlm_mtp.qwen38_fp8 import dequantize_fp8_weights

from ..qwen3_5 import Model as Qwen3_5Model
from ..qwen3_5.qwen3_5 import sanitize_key
from .config import ModelConfig
from .language import (
    LanguageModel,
    Qwen4ExpMTPModule,
    Qwen4ExpRMSNorm,
    compile_hyper_connections,
    fuse_hyper_connection_projections,
    fuse_resident_ple_embeddings,
    get_mtp_runtime,
    get_ple_runtime_mode,
)
from .vision import VisionModel

logger = logging.getLogger(__name__)

_NGRAM_SHARD_RE = re.compile(r"\.ngram_embedding\.shard_(\d+)(?=\.)")
_NGRAM_STORAGE_RE = re.compile(
    r"\.ple\.ple_embedding\.ngram_embedding\."
    r"(?:shard_\d+|shards\.\d+)\.(?:weight|scales|biases)$"
)
_MTP_PREFIXES = (
    "model.language_model.mtp.",
    "language_model.mtp.",
    "model.mtp.",
    "mtp.",
)
_RMSNORM_CENTER_ANCHOR_RE = re.compile(
    r"^language_model\.model\.layers\.\d+"
    r"\.attn_hyper_connection\.hc_norm\.weight$"
)
_RMSNORM_CENTER_MIN_ANCHORS = 8
_RMSNORM_ONES_CENTERED_VOTE = 0.9
_RMSNORM_ONES_CENTERED_MEDIAN_MIN = 0.75
_RMSNORM_ONES_CENTERED_MEDIAN_MAX = 1.5
_RMSNORM_ZERO_CENTERED_VOTE = 0.1
_RMSNORM_ZERO_CENTERED_MEDIAN_MIN = -0.5
_RMSNORM_ZERO_CENTERED_MEDIAN_MAX = 0.25


def _recenter_rmsnorm_weights(model, weights, *, mtp_only=None):
    """Recenter selected direct-gamma norms without losing BF16 precision."""
    target_keys = {
        f"{path}.weight"
        for path, module in model.named_modules()
        if isinstance(module, Qwen4ExpRMSNorm)
        and (mtp_only is None or path.startswith("mtp.") == mtp_only)
    }
    normalized = 0
    for key in target_keys:
        value = weights.get(key)
        if not isinstance(value, mx.array) or not mx.issubdtype(
            value.dtype, mx.floating
        ):
            continue
        # BF16 can lose information below 0.5. Keep it only when exact, so
        # fused BF16 kernels remain available and 1 + weight stays unchanged.
        residual = value.astype(mx.float32) - 1.0
        narrow = residual.astype(value.dtype)
        exact = mx.array_equal(narrow.astype(mx.float32), residual).item()
        weights[key] = narrow if exact else residual
        normalized += 1
    return normalized


def _normalize_ones_centered_rmsnorm_weights(model, weights, *, separate_mtp=False):
    """Canonicalize legacy direct-gamma Qwen4 RMSNorm checkpoints.

    Qwen4 stores most RMSNorm parameters as a residual around zero and applies
    ``1 + weight`` at runtime. Some early community MLX conversions instead
    stored the direct gamma around one. Use the stable per-layer attention
    hyper-connection norms to identify that checkpoint-wide conversion, then
    recenter only modules backed by :class:`Qwen4ExpRMSNorm`.

    An embedded MTP head shares that decision. A standalone head does not:
    its trained norm means cannot reliably identify its own convention.
    """
    named_modules = getattr(model, "named_modules", None)
    if named_modules is None:
        return

    anchors = [
        value
        for key, value in weights.items()
        if _RMSNORM_CENTER_ANCHOR_RE.fullmatch(key)
        and isinstance(value, mx.array)
        and mx.issubdtype(value.dtype, mx.floating)
    ]
    if len(anchors) < _RMSNORM_CENTER_MIN_ANCHORS:
        return

    means = mx.stack([mx.mean(value.astype(mx.float32)) for value in anchors])
    median = mx.median(means)
    ones_vote = mx.mean((means > 0.5).astype(mx.float32))
    mx.eval(median, ones_vote)
    median_value = float(median.item())
    ones_vote_value = float(ones_vote.item())

    ones_centered = (
        ones_vote_value >= _RMSNORM_ONES_CENTERED_VOTE
        and _RMSNORM_ONES_CENTERED_MEDIAN_MIN
        <= median_value
        <= _RMSNORM_ONES_CENTERED_MEDIAN_MAX
    )
    zero_centered = (
        ones_vote_value <= _RMSNORM_ZERO_CENTERED_VOTE
        and _RMSNORM_ZERO_CENTERED_MEDIAN_MIN
        <= median_value
        <= _RMSNORM_ZERO_CENTERED_MEDIAN_MAX
    )
    if not ones_centered:
        if not zero_centered:
            logger.warning(
                "Qwen4-Exp RMSNorm checkpoint centering is ambiguous "
                "(%d anchors, median %.4f, ones-centered vote %.1f%%); "
                "leaving weights unchanged",
                len(anchors),
                median_value,
                ones_vote_value * 100.0,
            )
        return

    normalized = _recenter_rmsnorm_weights(
        model, weights, mtp_only=False if separate_mtp else None
    )

    logger.info(
        "Canonicalized %d ones-centered Qwen4-Exp RMSNorm tensors "
        "(%d anchors, median %.4f)",
        normalized,
        len(anchors),
        median_value,
    )


class Model(Qwen3_5Model):
    def __init__(self, config: ModelConfig):
        nn.Module.__init__(self)
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)
        mtp_runtime = get_mtp_runtime()
        self._mtp_sidecar_path = mtp_runtime.sidecar_path
        if mtp_runtime.enabled:
            self.mtp = Qwen4ExpMTPModule(config.text_config)
            self.language_model.bind_mtp_owner(self)

    def sanitize(self, weights):
        mtp_runtime = get_mtp_runtime()
        sidecar_path = getattr(self, "_mtp_sidecar_path", None)
        loaded_sidecar = False
        sidecar_norm_format = "zero_centered"
        if (
            mtp_runtime.enabled
            and not mtp_runtime.checkpoint_prefix
            and sidecar_path is not None
        ):
            # Native qwen4_exp_mtp checkpoints use zero-centered RMSNorm
            # weights. A legacy direct-gamma conversion must declare its format.
            sidecar_config = json.loads(
                (sidecar_path.parent / "config.json").read_text()
            )
            sidecar_norm_format = sidecar_config.get(
                "rms_norm_weight_format", "zero_centered"
            )
            if sidecar_norm_format not in ("zero_centered", "ones_centered"):
                raise ValueError(
                    f"Unknown Qwen4 MTP RMSNorm format {sidecar_norm_format!r}"
                )
            logger.info(
                "Loading Qwen4 MTP sidecar with %s RMSNorm weights%s",
                sidecar_norm_format,
                " (native default)"
                if "rms_norm_weight_format" not in sidecar_config
                else "",
            )
            raw_sidecar = mx.load(str(sidecar_path))
            external_quant = sidecar_config.get("quantization_config")
            if not sidecar_config.get("quantization") and isinstance(
                external_quant, dict
            ):
                method = external_quant.get("quant_method")
                if method and (
                    method != "fp8"
                    or any(key.endswith(".scales") for key in raw_sidecar)
                ):
                    raise ValueError(
                        f"Qwen4 MTP sidecar quantization {method!r} is not a native "
                        "MLX recipe; convert the head to MLX format first"
                    )
            sidecar_weights = {}
            for key, value in raw_sidecar.items():
                prefix = next(
                    (prefix for prefix in _MTP_PREFIXES if key.startswith(prefix)),
                    "",
                )
                key = "mtp." + key[len(prefix) :]
                if key in sidecar_weights:
                    raise ValueError(f"Qwen4 MTP sidecar has duplicate tensor {key!r}")
                sidecar_weights[key] = value
            weights.update(sidecar_weights)
            loaded_sidecar = True
        if get_ple_runtime_mode() == "mmap" and not getattr(
            self, "_omlx_preserve_qwen4_ple_for_quantization", False
        ):
            for key in [key for key in weights if _NGRAM_STORAGE_RE.search(key)]:
                weights.pop(key)
        weights = dequantize_fp8_weights(weights, copy_weights=False)
        for layer_id in getattr(self.config.text_config, "ple_layer_ids", ()):
            source_scale_key = (
                f"model.language_model.layers.{int(layer_id) - 1}.ple."
                "ple_embedding.ngram_embedding.weight_scale"
            )
            runtime_scale_key = (
                f"language_model.model.layers.{int(layer_id) - 1}.ple."
                "ple_embedding.ngram_embedding.weight_scale"
            )
            # Converted MLX checkpoints already use the runtime prefix. Do not
            # add the raw-HF default as a second key: sanitize_key() maps both
            # spellings to runtime_scale_key, and the default would otherwise
            # overwrite a real shared FP8 PLE scale during normalization.
            if (
                source_scale_key not in weights
                and runtime_scale_key not in weights
            ):
                weights[source_scale_key] = mx.ones((1,), dtype=mx.bfloat16)
        mtp_enabled = mtp_runtime.enabled

        normalized = {}
        for key, value in weights.items():
            mtp_key = next(
                (prefix for prefix in _MTP_PREFIXES if key.startswith(prefix)),
                None,
            )
            if mtp_key is not None:
                if not mtp_enabled:
                    continue
                key = "mtp." + key[len(mtp_key) :]
            normalized[key] = value
        weights = normalized

        if self.config.text_config.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        num_experts = int(getattr(self.config.text_config, "num_experts", 0) or 0)

        def stack_experts(prefix, expert_count):
            if f"{prefix}.switch_mlp.gate_proj.weight" in weights:
                return

            gate_up_key = next(
                (
                    key
                    for key in (
                        f"{prefix}.experts.gate_up_proj",
                        f"{prefix}.experts.gate_up_proj.weight",
                    )
                    if key in weights
                ),
                None,
            )
            if gate_up_key is not None:
                up_key = f"{prefix}.switch_mlp.up_proj.weight"
                down_key = next(
                    (
                        key
                        for key in (
                            f"{prefix}.experts.down_proj",
                            f"{prefix}.experts.down_proj.weight",
                        )
                        if key in weights
                    ),
                    None,
                )
                if loaded_sidecar and prefix.startswith("mtp."):
                    for key in (up_key, f"{prefix}.switch_mlp.down_proj.weight"):
                        if key in weights and (key == up_key or down_key is not None):
                            raise ValueError(
                                f"Qwen4 MTP sidecar has duplicate tensor {key!r}"
                            )
                gate, up = mx.split(weights.pop(gate_up_key), 2, axis=-2)
                weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate
                weights[up_key] = up
                if down_key is not None:
                    weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(
                        down_key
                    )
                return

            if f"{prefix}.experts.0.gate_proj.weight" not in weights:
                return
            for projection in ("gate_proj", "up_proj", "down_proj"):
                for suffix in ("weight", "scales", "biases"):
                    first = f"{prefix}.experts.0.{projection}.{suffix}"
                    if first not in weights:
                        continue
                    weights[f"{prefix}.switch_mlp.{projection}.{suffix}"] = mx.stack(
                        [
                            weights.pop(
                                f"{prefix}.experts.{expert}.{projection}.{suffix}"
                            )
                            for expert in range(expert_count)
                        ]
                    )

        for layer_idx in range(self.config.text_config.num_hidden_layers):
            stack_experts(f"model.language_model.layers.{layer_idx}.mlp", num_experts)

        if mtp_enabled:
            mtp_num_experts = getattr(self.config.text_config, "mtp_num_experts", None)
            if mtp_num_experts is None:
                mtp_num_experts = num_experts
            mtp_layer_indices = sorted(
                {
                    int(key.split(".")[2])
                    for key in weights
                    if key.startswith("mtp.layers.")
                    and len(key.split(".")) > 2
                    and key.split(".")[2].isdigit()
                }
            )
            for layer_idx in mtp_layer_indices:
                stack_experts(f"mtp.layers.{layer_idx}.mlp", mtp_num_experts)

        sanitized = {}
        for key, value in weights.items():
            key = sanitize_key(key)
            key = _NGRAM_SHARD_RE.sub(r".ngram_embedding.shards.\1", key)
            if "conv1d.weight" in key and value.shape[-1] != 1:
                value = value.moveaxis(2, 1)
            sanitized[key] = value
        if loaded_sidecar:
            # mlx-vlm prioritizes a per-module override over the presence of
            # scales. Dense head tensors must not inherit stale target
            # overrides; quantized head tensors use their own recipe when given.
            sidecar_quant = sidecar_config.get("quantization") or sidecar_config.get(
                "quantization_config"
            )
            standalone_recipes = {}
            for key in sanitized:
                if not key.startswith("mtp.") or not key.endswith(".weight"):
                    continue
                path = key.removesuffix(".weight")
                recipe = None
                if path + ".scales" in sanitized and isinstance(sidecar_quant, dict):
                    head_path = path.removeprefix("mtp.")
                    names = (
                        head_path,
                        *(prefix + head_path for prefix in _MTP_PREFIXES),
                    )
                    layer_quant = next(
                        (
                            sidecar_quant[name]
                            for name in names
                            if name in sidecar_quant
                        ),
                        None,
                    )
                    if layer_quant is False:
                        raise ValueError(
                            f"Qwen4 MTP sidecar has scales for disabled module {path!r}"
                        )
                    if isinstance(layer_quant, dict):
                        recipe = layer_quant
                    else:
                        recipe = {
                            name: sidecar_quant[name]
                            for name in ("group_size", "bits", "mode", "quantize_input")
                            if name in sidecar_quant
                        }
                if path + ".scales" in sanitized and not isinstance(
                    getattr(self.config, "quantization", None), dict
                ):
                    if not recipe:
                        raise ValueError(
                            "A quantized Qwen4 MTP sidecar on a dense target requires "
                            "its own MLX quantization recipe"
                        )
                    standalone_recipes[path.removeprefix("mtp.")] = dict(recipe)
                for attr in ("quantization", "quantization_config"):
                    quantization = getattr(self.config, attr, None)
                    if not isinstance(quantization, dict):
                        continue
                    if recipe:
                        quantization[path] = dict(recipe)
                    elif (
                        path + ".scales" in sanitized
                        and quantization.get(path) is False
                    ):
                        # A disabled target-head override cannot describe the
                        # separate packed head; use the target's global recipe.
                        quantization.pop(path)
                    elif path + ".scales" not in sanitized and path in quantization:
                        quantization[path] = False
            if standalone_recipes:
                nn.quantize(
                    self.mtp,
                    class_predicate=lambda path, _: standalone_recipes.get(path, False),
                )
        _normalize_ones_centered_rmsnorm_weights(
            self, sanitized, separate_mtp=loaded_sidecar
        )
        if loaded_sidecar and sidecar_norm_format == "ones_centered":
            normalized = _recenter_rmsnorm_weights(self, sanitized, mtp_only=True)
            logger.info(
                "Canonicalized %d ones-centered Qwen4 MTP RMSNorm tensors", normalized
            )
        return sanitized

    def load_weights(self, weights, strict=True):
        result = super().load_weights(weights, strict=strict)
        mtp_enabled = get_mtp_runtime().enabled
        hybrid = 0 if mtp_enabled else fuse_hyper_connection_projections(self)
        fused_ple = fuse_resident_ple_embeddings(self)
        compiled = compile_hyper_connections(self)
        if mtp_enabled:
            logger.info(
                "Skipped Qwen4-Exp exact hybrid projections while "
                "Lightning MTP target verification is enabled"
            )
        logger.info(
            "Enabled Qwen4-Exp hyper-connection optimizations: "
            "%d exact hybrid projection pairs, %d compiled decode paths",
            hybrid,
            compiled,
        )
        if fused_ple:
            logger.info(
                "Fused %d resident Qwen4-Exp PLE table into one packed "
                "device-side embedding",
                fused_ple,
            )
        return result

    def close(self):
        """Release external PLE mmap handles during oMLX model unload."""
        for layer in self.language_model.model.layers:
            ple = getattr(layer, "ple", None)
            embedding = getattr(
                getattr(ple, "ple_embedding", None), "ngram_embedding", None
            )
            close = getattr(embedding, "close", None)
            if close is not None:
                close()

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
