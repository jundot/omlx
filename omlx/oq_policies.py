# SPDX-License-Identifier: Apache-2.0
"""Quantization policy registry.

Each QuantizationPolicy subclass encapsulates a distinct domain of
quantization decisions (MoE, SSM/Mamba, VLM, audio, default).  The
registry maps model-architecture keys to the policy that handles them.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Optional, Union


class QuantizationPolicy(ABC):
    """Base class for quantization policies.

    Subclasses implement ``evaluate`` to return one of:
        - ``False``          → skip quantization (keep fp16)
        - ``True``           → use default bits from config
        - ``dict``           → per-layer override ``{"bits": N, "group_size": M, "mode": ...}``
    """

    @abstractmethod
    def evaluate(
        self,
        path: str,
        config: dict,
        oq_level: float,
        _bits_fn: Any,
    ) -> Union[bool, dict]:
        ...


# ---------------------------------------------------------------------------
# Policy implementations
# ---------------------------------------------------------------------------


class MoEPolicy(QuantizationPolicy):
    """Handles quantization for MoE (Mixture-of-Experts) models.

    Key rules:
    - Router / gate layers are never quantized (too small, risk of crash).
    - Shared-expert paths get 8-bit protection.
    - High-parameter experts (>= 512) get tighter quantization.
    """

    # Patterns that identify router or gate-only components
    _ROUTER_PATTERNS = (
        "mlp.gate",
        ".router",
        ".router.layer",
    )

    def _is_router(self, path: str) -> bool:
        if path.endswith(".gate") and "gate_proj" not in path:
            return True
        if ".gate." in path and "gate_proj" not in path:
            return True
        return False

    def evaluate(
        self,
        path: str,
        config: dict,
        oq_level: float,
        _bits_fn: Any,
    ) -> Union[bool, dict]:
        num_experts = (
            config.get("num_local_experts")
            or config.get("num_experts", 0)
            or 0
        )

        # Shared expert (non-gate) gets 8-bit protection
        if "shared_expert" in path and not path.endswith("shared_expert_gate"):
            return {"bits": 8, "group_size": 64, "mode": "affine"}

        # MoE routers and gate-only paths stay fp16
        if any(p in path for p in self._ROUTER_PATTERNS):
            return False

        # High-parameter expert paths get tighter quantization
        if num_experts >= 512:
            if "gate_proj" in path and "shared_expert" not in path:
                bits_fn = _bits_fn or (lambda n: {"bits": n, "group_size": 64, "mode": "affine"})
                return bits_fn(4)
            if "down_proj" in path and "shared_expert" not in path:
                bits_fn = _bits_fn or (lambda n: {"bits": n, "group_size": 64, "mode": "affine"})
                return bits_fn(3)

        return True


class SsmPolicy(QuantizationPolicy):
    """Handles quantization for SSM / Mamba-like architectures.

    Key rules:
    - SSM alpha/beta, a_log, time_decay, time_faaaa stay fp16.
    - dt_bias stays fp16 (discretization step sensitivity).
    - conv1d + linear_attn → 8-bit.
    - linear_attn.out_proj → 5-bit.
    - SSM output projections get 8-bit protection.
    """

    _SSM_SENSITIVE = (
        "ssm_alpha",
        "ssm_beta",
        "a_log",
        "time_decay",
        "time_faaaa",
    )

    def evaluate(
        self,
        path: str,
        config: dict,
        oq_level: float,
        _bits_fn: Any,
    ) -> Union[bool, dict]:
        path_l = path.lower()

        # SSM-sensitive parameters stay fp16
        if any(p in path_l for p in self._SSM_SENSITIVE):
            return False

        # dt_bias: discretization step sensitivity
        if path_l.endswith("dt_bias"):
            return False

        # conv1d inside linear_attn → 8-bit
        if "conv1d" in path_l and "linear_attn" in path_l:
            bits_fn = _bits_fn or (lambda n: {"bits": n, "group_size": 64, "mode": "affine"})
            return bits_fn(8)

        # linear_attn.out_proj → 5-bit
        if "linear_attn.out_proj" in path_l:
            bits_fn = _bits_fn or (lambda n: {"bits": n, "group_size": 64, "mode": "affine"})
            return bits_fn(5)

        return True


class QwenMoEPolicy(QuantizationPolicy):
    """Quantization policy for Qwen-MoE models (e.g. Qwen3.5 122B-A10B).

    Extends MoEPolicy with Qwen-specific sensitivity handling:
    - Qwen3_5 hybrid arch has gated dt_bias (discretization sensitivity)
    - linear_attn.out_proj mirrors self_attn.o_proj sensitivity
    """

    _QWEN_MOE_SENSITIVE = (
        "dt_bias",
        "conv1d",
        "linear_attn.out_proj",
    )

    def evaluate(
        self,
        path: str,
        config: dict,
        oq_level: float,
        _bits_fn: Any,
    ) -> Union[bool, dict]:
        path_l = path.lower()

        # Qwen3_5 hybrid: dt_bias drives discretization, keep fp16
        if "dt_bias" in path_l:
            return False

        # conv1d inside linear_attn → 8-bit
        if "conv1d" in path_l and "linear_attn" in path_l:
            bits_fn = _bits_fn or (lambda n: {"bits": n, "group_size": 64, "mode": "affine"})
            return bits_fn(8)

        # linear_attn.out_proj mirrors self_attn.o_proj sensitivity
        if "linear_attn.out_proj" in path_l:
            bits_fn = _bits_fn or (lambda n: {"bits": n, "group_size": 64, "mode": "affine"})
            return bits_fn(5)

        return True


class VlmPolicy(QuantizationPolicy):
    """Handles quantization for Vision-Language Models.

    Key rules:
    - Vision encoder / projector tensors are never quantized.
    - Covers visual., vision_, patch_embed, pos_embed, image_newline,
      multi_modal_projector, visual.merger, image_norm, temporal_embed.
    """

    _VISION_PATTERNS = (
        "visual.",
        "vision_",
        "patch_embed",
        "pos_embed",
        "image_newline",
        "multi_modal_projector",
        "visual.merger",
        "image_norm",
        "temporal_embed",
    )

    def evaluate(
        self,
        path: str,
        config: dict,
        oq_level: float,
        _bits_fn: Any,
    ) -> Union[bool, dict]:
        if any(p in path for p in self._VISION_PATTERNS):
            return False
        return True


class AudioPolicy(QuantizationPolicy):
    """Handles quantization for audio encoder components.

    Only audio_tower paths are skipped; embed_audio projections are
    quantized like embed_vision.embedding_projection.
    """

    _AUDIO_PATTERNS = ("audio_tower",)

    def evaluate(
        self,
        path: str,
        config: dict,
        oq_level: float,
        _bits_fn: Any,
    ) -> Union[bool, dict]:
        if any(p in path for p in self._AUDIO_PATTERNS):
            return False
        return True


class DefaultPolicy(QuantizationPolicy):
    """Default quantization policy for standard LLM / causal models.

    Key rules:
    - lm_head, output.weight, classifier → 6-bit (output sensitivity).
    - embed_tokens / wte / word_embeddings → base_bits + 2.
    - o_proj (non-MoE) → 5-bit.
    - cross_attn o_proj → 6-bit.
    - kv_a_proj_with_mqa, kv_b_proj, q_a_proj, q_b_proj → 6-bit.
    - v_proj, v_a_proj, v_b_proj: sensitive → 6-bit, else True.
    - down_proj, w2, mlp.fc2, wo: sensitive → 6-bit, else 5-bit.
    - q_proj, k_proj: sensitive → 5-bit.
    - qkv_proj, in_proj_qkv, attn_qkv: sensitive → 5-bit.
    - in_proj_z/a/b, delta_net → 5-bit.
    - mixer.in_proj, mixer.out_proj, x_proj, dt_proj → 5-bit.
    """

    _OUTPUT_SENSITIVE = (
        "lm_head",
        "output.weight",
        "classifier",
    )

    _EMBED_SENSITIVE = (
        "embed_tokens",
        "wte",
        "word_embeddings",
    )

    _ATTN_V_SENSITIVE = (
        "v_proj",
        "v_a_proj",
        "v_b_proj",
    )

    _ATTN_Q_SENSITIVE = (
        "q_proj",
        "k_proj",
    )

    _QKV_SENSITIVE = (
        "qkv_proj",
        "in_proj_qkv",
        "attn_qkv",
    )

    _DELTA_SENSITIVE = (
        "in_proj_z",
        "in_proj_a",
        "in_proj_b",
        "delta_net",
    )

    _MIXER_SENSITIVE = (
        "mixer.in_proj",
        "mixer.out_proj",
        "x_proj",
        "dt_proj",
    )

    def evaluate(
        self,
        path: str,
        config: dict,
        oq_level: float,
        _bits_fn: Any,
    ) -> Union[bool, dict]:
        path_l = path.lower()
        num_experts = (
            config.get("num_local_experts")
            or config.get("num_experts", 0)
            or 0
        )
        is_moe = num_experts > 0
        base_bits = int(_LEVEL_BITS.get(oq_level, oq_level))
        protection = _LEVEL_PROTECTION.get(oq_level, "full")
        full_protection = protection == "full"

        # Layer sensitivity tracking
        layer_idx = _extract_layer_index(path)
        num_layers = config.get("num_hidden_layers") or config.get("text_config", {}).get("num_hidden_layers", 32)
        layer_idx = _extract_layer_index(path)
        sensitive = (
            layer_idx >= 0 and (
                layer_idx < num_layers // 8
                or layer_idx >= 7 * num_layers // 8
            )
        )

        if not full_protection:
            # Output sensitivity
            if any(p in path for p in self._OUTPUT_SENSITIVE):
                bits_fn = _bits_fn or (lambda n: {"bits": n, "group_size": 64, "mode": "affine"})
                return bits_fn(6)

            # Embed sensitivity
            if any(p in path for p in self._EMBED_SENSITIVE):
                bits_fn = _bits_fn or (lambda n: {"bits": n, "group_size": 64, "mode": "affine"})
                return bits_fn(base_bits + 2)

            # MoE high-parameter expert paths
            if num_experts >= 512:
                if "gate_proj" in path and "shared_expert" not in path:
                    bits_fn = _bits_fn or (lambda n: {"bits": n, "group_size": 64, "mode": "affine"})
                    return bits_fn(4)

            # Layer sensitivity
            if layer_idx >= 0:
                if sensitive and not any(
                    p in path for p in ("switch_mlp", "experts")
                ):
                    bits_fn = _bits_fn or (lambda n: {"bits": n, "group_size": 64, "mode": "affine"})
                    return bits_fn(base_bits + 1)

            return True

        # Full protection paths
        # SSM output projections → 8-bit
        if any(p in path for p in ("ssm_output", "ssm_out")):
            bits_fn = _bits_fn or (lambda n: {"bits": n, "group_size": 64, "mode": "affine"})
            return bits_fn(8)

        # lora.2 → 8-bit
        if "lora.2" in path:
            bits_fn = _bits_fn or (lambda n: {"bits": n, "group_size": 64, "mode": "affine"})
            return bits_fn(8)

        # Output sensitivity
        if any(p in path for p in self._OUTPUT_SENSITIVE):
            bits_fn = _bits_fn or (lambda n: {"bits": n, "group_size": 64, "mode": "affine"})
            return bits_fn(6)

        # Cross-attn o_proj → 6-bit
        if "cross_attn" in path and "o_proj" in path:
            bits_fn = _bits_fn or (lambda n: {"bits": n, "group_size": 64, "mode": "affine"})
            return bits_fn(6)

        # KV projection → 6-bit
        if any(p in path for p in ("kv_a_proj_with_mqa", "kv_b_proj", "q_a_proj", "q_b_proj")):
            bits_fn = _bits_fn or (lambda n: {"bits": n, "group_size": 64, "mode": "affine"})
            return bits_fn(6)

        # o_proj (non-MoE) → 5-bit
        if "o_proj" in path and "shared_expert" not in path:
            if not is_moe:
                bits_fn = _bits_fn or (lambda n: {"bits": n, "group_size": 64, "mode": "affine"})
                return bits_fn(5)

        # Shared expert → 8-bit
        if "shared_expert" in path and not path.endswith("shared_expert_gate"):
            bits_fn = _bits_fn or (lambda n: {"bits": n, "group_size": 64, "mode": "affine"})
            return bits_fn(8)

        # High-parameter expert paths
        if num_experts >= 512:
            if "gate_proj" in path and "shared_expert" not in path:
                bits_fn = _bits_fn or (lambda n: {"bits": n, "group_size": 64, "mode": "affine"})
                return bits_fn(4)
            if "down_proj" in path and "shared_expert" not in path:
                bits_fn = _bits_fn or (lambda n: {"bits": n, "group_size": 64, "mode": "affine"})
                return bits_fn(3)

        # Attention v_proj → sensitive → 6-bit, else True
        if any(p in path for p in self._ATTN_V_SENSITIVE):
            if sensitive:
                bits_fn = _bits_fn or (lambda n: {"bits": n, "group_size": 64, "mode": "affine"})
                return bits_fn(6)
            return True

        # down_proj, w2, mlp.fc2, wo
        if any(p in path for p in ("down_proj", "w2", "mlp.fc2", "wo")):
            is_routed_expert = is_moe and "shared_expert" not in path and (
                "switch_mlp" in path or "experts" in path
            )
            if is_routed_expert:
                if oq_level == 3.5:
                    bits_fn = _bits_fn or (lambda n: {"bits": n, "group_size": 64, "mode": "affine"})
                    return bits_fn(4)
                return True
            if sensitive:
                bits_fn = _bits_fn or (lambda n: {"bits": n, "group_size": 64, "mode": "affine"})
                return bits_fn(6)
            return bits_fn(5)

        # q_proj, k_proj → sensitive → 5-bit
        if any(p in path for p in self._ATTN_Q_SENSITIVE):
            if sensitive:
                bits_fn = _bits_fn or (lambda n: {"bits": n, "group_size": 64, "mode": "affine"})
                return bits_fn(5)

        # qkv_proj, in_proj_qkv, attn_qkv → sensitive → 5-bit
        if any(p in path for p in self._QKV_SENSITIVE):
            if sensitive:
                bits_fn = _bits_fn or (lambda n: {"bits": n, "group_size": 64, "mode": "affine"})
                return bits_fn(5)

        # in_proj_z/a/b, delta_net → 5-bit
        if any(p in path for p in self._DELTA_SENSITIVE):
            bits_fn = _bits_fn or (lambda n: {"bits": n, "group_size": 64, "mode": "affine"})
            return bits_fn(5)

        # mixer.in_proj, mixer.out_proj, x_proj, dt_proj → 5-bit
        if any(p in path for p in self._MIXER_SENSITIVE):
            bits_fn = _bits_fn or (lambda n: {"bits": n, "group_size": 64, "mode": "affine"})
            return bits_fn(5)

        return True


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, QuantizationPolicy] = {
    "moe": MoEPolicy(),
    "ssm": SsmPolicy(),
    "qwen_moe": QwenMoEPolicy(),
    "vlm": VlmPolicy(),
    "audio": AudioPolicy(),
    "default": DefaultPolicy(),
}


# Re-export _bits_fn_factory for use by oq.py
from .oq_constants import _bits_fn_factory


def get_policy_for_model(
    model_type: str | None,
    config: dict,
) -> QuantizationPolicy:
    """Return the quantization policy for the given model architecture.

    The model_type is matched against known architecture keys; if no
    direct match is found the default policy is returned.
    """
    # MoE detection: num_local_experts or num_experts > 0
    num_experts = (
        config.get("num_local_experts")
        or config.get("num_experts", 0)
        or 0
    )
    if num_experts > 0:
        return MoEPolicy()

    mt = model_type.lower() if model_type else ""

    # Direct registry match
    if mt in _REGISTRY:
        return _REGISTRY[mt]

    # Qwen-MoE detection: contains both "qwen" and MoE indicators
    if "qwen" in mt and "moe" in mt:
        return QwenMoEPolicy()

    # SSM detection
    if any(k in mt for k in ("mamba", "ssm", "mamba2", "ssm", "state_space")):
        return SsmPolicy()

    # VLM / vision-language detection
    if "vision" in mt or "vlm" in mt or "image" in mt:
        return VlmPolicy()

    # Audio detection
    if "audio" in mt or "whisper" in mt or "wav2vec" in mt:
        return AudioPolicy()

    return DefaultPolicy()


# ---------------------------------------------------------------------------
# Compatibility helpers (imported by oq.py)
# ---------------------------------------------------------------------------

from .oq_constants import (
    _LEVEL_BITS,
    _LEVEL_PROTECTION,
    _OQ_DEFAULT_GROUP_SIZE,
    _extract_layer_index,
    _mode_for_bits,
)

__all__ = [
    "QuantizationPolicy",
    "MoEPolicy",
    "SsmPolicy",
    "QwenMoEPolicy",
    "VlmPolicy",
    "AudioPolicy",
    "DefaultPolicy",
    "get_policy_for_model",
]
