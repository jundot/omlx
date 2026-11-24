# SPDX-License-Identifier: Apache-2.0
"""Regressions for the DFlash drafter's dtype sources.

Two load-time dtype decisions used to read a value that is not a compute
dtype:

* the target's token embedding was looked up only as ``embed_tokens`` on the
  inner model, so a Nemotron-H target (``backbone.embeddings``) returned None
  and ``match_target_dtype`` evaluated ``None != mx.float16``, which raises
  ``TypeError: __eq__(): incompatible function arguments``;
* the draft activation/KV-ring dtype came from ``fc.scales.dtype``, which is
  ``uint8`` (E4M3 block scales) on a ModelOpt NVFP4 draft, so the ring and the
  residual were allocated in ``uint8`` and every context row was truncated to
  an integer.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.speculative.dflash import (
    _target_float_dtype,
    match_target_dtype,
)
from omlx.speculative.dflash.drafter import _activation_dtype


def _embedding(dtype):
    embed = nn.Embedding(16, 8)
    embed.weight = embed.weight.astype(dtype)
    return embed


def _nvfp4_carrier():
    """A ModelOpt NVFP4 projection: uint32 packed weight, uint8 E4M3 scales."""
    return SimpleNamespace(
        weight=mx.zeros((8, 4), dtype=mx.uint32),
        scales=mx.zeros((8, 1), dtype=mx.uint8),
        weight_scale_2=mx.array(1.0, dtype=mx.float32),
    )


# ---------------------------------------------------------------------------
# target embedding dtype
# ---------------------------------------------------------------------------


def test_target_float_dtype_reads_standard_embed_tokens():
    inner = SimpleNamespace(embed_tokens=_embedding(mx.float16))
    target = SimpleNamespace(language_model=SimpleNamespace(model=inner))
    assert _target_float_dtype(target) == mx.float16


def test_target_float_dtype_reads_a_nemotron_backbone_embedding():
    # Nemotron-H keeps the table on the backbone and has no ``embed_tokens``.
    language = SimpleNamespace(
        backbone=SimpleNamespace(embeddings=_embedding(mx.bfloat16))
    )
    target = SimpleNamespace(language_model=language)
    assert _target_float_dtype(target) == mx.bfloat16


def test_target_float_dtype_reads_a_quantized_embedding_by_its_scales():
    embed = SimpleNamespace(
        weight=mx.zeros((16, 2), dtype=mx.uint32),
        scales=mx.zeros((16, 1), dtype=mx.bfloat16),
    )
    target = SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(embed_tokens=embed))
    )
    assert _target_float_dtype(target) == mx.bfloat16


def test_target_float_dtype_is_none_without_an_embedding():
    assert _target_float_dtype(SimpleNamespace(language_model=SimpleNamespace())) is None


def test_match_target_dtype_tolerates_a_target_without_an_embedding():
    # The None dtype used to reach ``None != mx.float16`` and raise.
    class _Drafter(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4)
            self.proj.weight = self.proj.weight.astype(mx.bfloat16)

    bare_target = SimpleNamespace(language_model=SimpleNamespace())
    assert match_target_dtype(_Drafter(), bare_target) == 0


# ---------------------------------------------------------------------------
# draft activation dtype
# ---------------------------------------------------------------------------


def test_activation_dtype_ignores_the_uint8_scales_of_an_nvfp4_carrier():
    model = SimpleNamespace(
        hidden_norm=nn.RMSNorm(8),
        fc=_nvfp4_carrier(),
    )
    model.hidden_norm.weight = model.hidden_norm.weight.astype(mx.bfloat16)
    assert _activation_dtype(model) == mx.bfloat16


def test_activation_dtype_follows_a_fp16_drafter():
    model = SimpleNamespace(fc=SimpleNamespace(weight=mx.zeros((8, 8), dtype=mx.float16)))
    assert _activation_dtype(model) == mx.float16


def test_activation_dtype_reads_an_encoder_projection():
    # Muse Glimmer's context projection lives on ``encoder``.
    encoder = SimpleNamespace(fc=SimpleNamespace(weight=mx.zeros((8, 8), dtype=mx.float16)))
    model = SimpleNamespace(encoder=encoder, embed_tokens=_embedding(mx.bfloat16))
    assert _activation_dtype(model) == mx.float16


def test_activation_dtype_defaults_to_bfloat16():
    assert _activation_dtype(nn.Module()) == mx.bfloat16


def test_activation_dtype_is_a_compute_type_for_an_nvfp4_model_without_norms():
    # Even without any floating projection, the carrier's uint8 scales must
    # never become the activation dtype.
    model = SimpleNamespace(fc=_nvfp4_carrier())
    assert _activation_dtype(model) == mx.bfloat16
    assert not mx.issubdtype(_activation_dtype(model), mx.integer)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
