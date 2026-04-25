# SPDX-License-Identifier: Apache-2.0
"""Real-model end-to-end integration test for PlanarQuant3."""

from __future__ import annotations

import pytest
import mlx.core as mx

MODEL_ID = "mlx-community/Qwen3.5-4B-MLX-4bit"


def _try_load_model():
    try:
        from mlx_lm import load
    except ImportError:
        pytest.skip("mlx_lm not installed")
    try:
        return load(MODEL_ID)
    except Exception as e:
        pytest.skip(f"Could not load {MODEL_ID}: {e}")


@pytest.fixture(scope="module")
def model_and_tokenizer():
    return _try_load_model()


@pytest.mark.slow
def test_forward_pass_with_planarquant_cache_matches_fp16_within_tolerance(
    model_and_tokenizer,
):
    model, tokenizer = model_and_tokenizer
    from mlx_lm.models import cache as mlx_cache_mod

    from omlx.cache.planarquant.kv_cache import PlanarQuantKVCache
    from omlx.patches.planarquant_cache import (
        disable_planarquant_cache,
        enable_planarquant_cache,
    )
    from omlx.patches.turboquant_attention import apply_turboquant_attention_patch

    apply_turboquant_attention_patch()

    prompt = "The capital of France is"
    tokens = mx.array(tokenizer.encode(prompt))
    tokens_2d = tokens[None, :]

    # Baseline: FP16 KV cache path
    disable_planarquant_cache()
    fp16_cache = mlx_cache_mod.make_prompt_cache(model)
    logits_fp16 = model(tokens_2d, cache=fp16_cache)
    last_fp16 = logits_fp16[0, -1, :]

    # PlanarQuant path
    enable_planarquant_cache(3.0)
    pq_cache = mlx_cache_mod.make_prompt_cache(model)
    n_planar = sum(1 for c in pq_cache if isinstance(c, PlanarQuantKVCache))
    assert n_planar > 0, f"No PlanarQuant caches created; got {[type(c).__name__ for c in pq_cache]}"

    logits_pq = model(tokens_2d, cache=pq_cache)
    last_pq = logits_pq[0, -1, :]

    disable_planarquant_cache()

    dot = float(mx.sum(last_fp16.astype(mx.float32) * last_pq.astype(mx.float32)).item())
    norm_fp = float(mx.sqrt(mx.sum(last_fp16.astype(mx.float32) ** 2)).item())
    norm_pq = float(mx.sqrt(mx.sum(last_pq.astype(mx.float32) ** 2)).item())
    cos_sim = dot / (norm_fp * norm_pq + 1e-10)

    print(f"\nPlanarQuant3 integration: cos_sim = {cos_sim:.6f}")
    print(f"  n_planar_caches = {n_planar}/{len(pq_cache)}")
    assert cos_sim > 0.95, f"Logit cos sim too low: {cos_sim}"
