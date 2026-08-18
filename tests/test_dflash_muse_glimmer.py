# SPDX-License-Identifier: Apache-2.0
"""Muse Glimmer DFlash integration tests (oMLX side).

The heavy drafter/backend unit tests live in the dflash-mlx fork
(tests/test_muse_glimmer_draft.py, tests/test_target_muse_glimmer.py).
This file guards the oMLX-side integration surfaces:

- cross-implementation drift between dflash-mlx's text-only mlx-lm module
  and the vendored mlx-vlm port (the two must stay numerically identical
  or DFlash verify logits diverge from serving logits),
- independence from oMLX's DFlashDraftModelArgs.from_dict normalizer
  wrapper (issue #2317) — the muse drafter does its own root-key
  normalization and must keep working with the wrapper installed,
- drafter discovery classification (config_model_type payload the
  dashboard's DFlash drafter set keys on).
"""

from __future__ import annotations

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    import dflash_mlx  # noqa: F401

    HAS_DFLASH = True
except ImportError:
    HAS_DFLASH = False

pytestmark = pytest.mark.skipif(
    not (HAS_MLX and HAS_DFLASH), reason="MLX or dflash-mlx not available"
)

_TINY_TEXT_KWARGS = dict(
    vocab_size=64,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=4,
    max_position_embeddings=256,
    sliding_window=8,
)


def _fork_model():
    from dflash_mlx.models.muse_glimmer import Model, ModelArgs

    mx.random.seed(0)
    model = Model(ModelArgs(**_TINY_TEXT_KWARGS))
    model.set_dtype(mx.bfloat16)
    return model


def _vendor_language_model():
    from omlx.patches.mlx_vlm_muse_glimmer_compat import (
        apply_mlx_vlm_muse_glimmer_compat_patch,
    )

    apply_mlx_vlm_muse_glimmer_compat_patch()
    from mlx_vlm.models.muse_glimmer.config import TextConfig
    from mlx_vlm.models.muse_glimmer.language import LanguageModel

    mx.random.seed(0)
    model = LanguageModel(TextConfig(rms_norm_eps=1e-5, **_TINY_TEXT_KWARGS))
    model.set_dtype(mx.bfloat16)
    return model


class TestCrossImplementationParity:
    """Fork text module vs vendored mlx-vlm port on identical weights."""

    def _sync_weights(self, fork_model, vendor_lm):
        from mlx.utils import tree_flatten, tree_unflatten

        vendor_weights = dict(tree_flatten(vendor_lm.parameters()))
        # Vendor paths are model.<...>/lm_head.<...>; the fork uses the
        # same layout, so the mapping is the identity.
        fork_model.update(tree_unflatten(list(vendor_weights.items())))

    def test_logits_match_bit_exact(self):
        fork_model = _fork_model()
        vendor_lm = _vendor_language_model()
        self._sync_weights(fork_model, vendor_lm)

        ids = mx.array([[(i * 7) % 60 for i in range(24)]])
        fork_logits = fork_model(ids)
        vendor_logits = vendor_lm(ids).logits
        mx.eval(fork_logits, vendor_logits)
        assert bool(mx.array_equal(fork_logits, vendor_logits))

    def test_cache_layout_matches(self):
        fork_model = _fork_model()
        vendor_lm = _vendor_language_model()
        fork_kinds = [type(c).__name__ for c in fork_model.make_cache()]
        vendor_kinds = [type(c).__name__ for c in vendor_lm.make_cache()]
        assert fork_kinds == vendor_kinds

    def test_backend_capture_matches_vendor_forward(self):
        from dflash_mlx.engine.target_muse_glimmer import MuseGlimmerTargetOps

        fork_model = _fork_model()
        vendor_lm = _vendor_language_model()
        self._sync_weights(fork_model, vendor_lm)

        ids = mx.array([[(i * 5) % 60 for i in range(16)]])
        ops = MuseGlimmerTargetOps()
        logits, _ = ops.forward_with_hidden_capture(
            fork_model,
            input_ids=ids,
            cache=ops.make_cache(fork_model, enable_speculative_linear_cache=False),
            capture_layer_ids={0},
        )
        vendor_logits = vendor_lm(ids, cache=vendor_lm.make_cache()).logits
        mx.eval(logits, vendor_logits)
        assert bool(mx.allclose(logits, vendor_logits, atol=1e-5))


class TestDraftConfig:
    def test_muse_from_dict_supports_nested_rope_config(self):
        from dflash_mlx.models.muse_glimmer_draft import MuseGlimmerDraftModelArgs

        args = MuseGlimmerDraftModelArgs.from_dict(
            {
                "model_type": "muse_glimmer_assistant",
                "hidden_size": 32,
                "num_hidden_layers": 1,
                "intermediate_size": 64,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "rms_norm_eps": 1e-5,
                "max_position_embeddings": 4096,
                "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
                "layer_types": ["sliding_attention"],
                "sliding_window": 16,
                "block_size": 4,
                "target_layer_ids": [1],
                "mask_token_id": 99,
            }
        )
        assert args.rope_theta == 500000.0
        assert args.dflash_config["mask_token_id"] == 99

    def test_base_dispatch_unaffected(self):
        from dflash_mlx.model import DFlashDraftModel
        from dflash_mlx.runtime.loading import _get_dflash_model_classes

        model_cls, _ = _get_dflash_model_classes({"model_type": "qwen3"})
        assert model_cls is DFlashDraftModel


class TestDrafterClassification:
    def test_assistant_is_helper_not_servable(self):
        from omlx.model_discovery import (
            is_helper_config_model_type,
            is_helper_model_config,
        )

        assert is_helper_config_model_type("muse_glimmer_assistant")
        assert is_helper_model_config(
            {
                "model_type": "muse_glimmer_assistant",
                "architectures": ["MuseGlimmerAssistantModel"],
            }
        )
