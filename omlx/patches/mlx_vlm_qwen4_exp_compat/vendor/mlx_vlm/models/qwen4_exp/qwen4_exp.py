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


def _normalize_ones_centered_rmsnorm_weights(model, weights):
    """Canonicalize legacy direct-gamma Qwen4 RMSNorm checkpoints.

    Qwen4 stores most RMSNorm parameters as a residual around zero and applies
    ``1 + weight`` at runtime. Some early community MLX conversions instead
    stored the direct gamma around one. Use the stable per-layer attention
    hyper-connection norms to identify that checkpoint-wide conversion, then
    recenter only modules backed by :class:`Qwen4ExpRMSNorm`.

    The MTP head is normalized using the base-model decision. Its trained norm
    means are not independently centered near zero, so sampling the MTP head
    alone would be ambiguous.
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

    target_keys = {
        f"{path}.weight"
        for path, module in named_modules()
        if isinstance(module, Qwen4ExpRMSNorm)
    }
    normalized = 0
    for key in target_keys:
        value = weights.get(key)
        if not isinstance(value, mx.array) or not mx.issubdtype(
            value.dtype, mx.floating
        ):
            continue
        # Keep the residual in FP32. Subtracting in BF16 can lose information
        # for direct gamma values below 0.5, which occur in the trained MTP head.
        weights[key] = value.astype(mx.float32) - 1.0
        normalized += 1

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
        if get_mtp_runtime().enabled:
            self.mtp = Qwen4ExpMTPModule(config.text_config)
            self.language_model.bind_mtp_owner(self)

    def sanitize(self, weights):
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
        mtp_enabled = get_mtp_runtime().enabled

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

        def stack_experts(prefix):
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
                gate, up = mx.split(weights.pop(gate_up_key), 2, axis=-2)
                weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate
                weights[f"{prefix}.switch_mlp.up_proj.weight"] = up
                for down_key in (
                    f"{prefix}.experts.down_proj",
                    f"{prefix}.experts.down_proj.weight",
                ):
                    if down_key in weights:
                        weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(
                            down_key
                        )
                        break
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
                            for expert in range(num_experts)
                        ]
                    )

        for layer_idx in range(self.config.text_config.num_hidden_layers):
            stack_experts(f"model.language_model.layers.{layer_idx}.mlp")

        if mtp_enabled:
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
                stack_experts(f"mtp.layers.{layer_idx}.mlp")

        sanitized = {}
        for key, value in weights.items():
            key = sanitize_key(key)
            key = _NGRAM_SHARD_RE.sub(r".ngram_embedding.shards.\1", key)
            if "conv1d.weight" in key and value.shape[-1] != 1:
                value = value.moveaxis(2, 1)
            sanitized[key] = value
        _normalize_ones_centered_rmsnorm_weights(self, sanitized)
        return sanitized

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
