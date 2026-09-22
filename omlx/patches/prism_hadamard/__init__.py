# SPDX-License-Identifier: Apache-2.0
"""Runtime-only optimizations for mlx-vlm's native Prism Hadamard model.

The upstream loader owns the checkpoint format, transforms and processor. This
module only preserves singleton KV capacity and optionally narrows activations.
"""

import logging
import os

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.prism_hadamard_qwen35.prism_hadamard_qwen35 import (
    HadamardQuantizedEmbedding,
    HadamardQuantizedLinear,
)
from mlx_vlm.models.qwen3_5.language import Qwen3_5Model

from .decode import CapacityPreservingModel

logger = logging.getLogger(__name__)
MODEL_TYPE = "prism_hadamard_qwen35"


class _FP16Linear(HadamardQuantizedLinear):
    def __call__(self, x):
        return super().__call__(x.astype(mx.float16)).astype(mx.float16)


class _FP16Embedding(HadamardQuantizedEmbedding):
    def as_linear(self, x):
        return super().as_linear(x.astype(mx.float16)).astype(mx.float16)


class _ActivationRMSNorm(nn.RMSNorm):
    def __call__(self, x):
        return super().__call__(x).astype(x.dtype)


class _ActivationConv1d(nn.Conv1d):
    def __call__(self, x):
        return super().__call__(x).astype(x.dtype)


def _enable_fp16_activations(model):
    # Preserve the native transforms, checkpoint weights and recurrent state.
    # Exact type checks avoid replacing another extension's module subclass.
    replacements = {
        HadamardQuantizedLinear: _FP16Linear,
        HadamardQuantizedEmbedding: _FP16Embedding,
        nn.RMSNorm: _ActivationRMSNorm,
        nn.Conv1d: _ActivationConv1d,
    }
    for _, module in model.named_modules():
        replacement = replacements.get(type(module))
        if replacement is not None:
            module.__class__ = replacement


def apply_runtime_patches(model):
    """Patch a loaded native Prism VLM, leaving other model classes untouched."""
    if getattr(getattr(model, "config", None), "model_type", None) != MODEL_TYPE:
        return False
    language_model = getattr(model, "language_model", None)
    backbone = getattr(language_model, "model", None)
    if type(backbone) is CapacityPreservingModel:
        return True
    if type(backbone) is not Qwen3_5Model:
        logger.warning("Prism runtime patches skipped: unfamiliar decoder class")
        return False

    backbone.__class__ = CapacityPreservingModel
    fp16 = os.environ.get("OMLX_PRISM_FP16_ACTIVATIONS", "0") == "1"
    if fp16:
        _enable_fp16_activations(language_model)
        # Native mlx-vlm loading is a new implementation boundary. Do not reuse
        # FP32 caches or FP16 caches from the previous custom-loader adapter.
        model._omlx_prism_activation_signature = "prism_fp16_v2"
    logger.info("Prism singleton KV capacity enabled; FP16 activations=%s", fp16)
    return True
