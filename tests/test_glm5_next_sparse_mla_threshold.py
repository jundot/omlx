# SPDX-License-Identifier: Apache-2.0
"""The native sparse-MLA kernel only pays off above a context-length crossover.

Measured on M1 Ultra with real GLM-5.3-Flash shapes (64 heads, latent 512,
top-k 2048), median of 7 runs:

    ctx  4096: dense  28.13 ms   native  82.45 ms   (native 3x slower)
    ctx  8192: dense 107.62 ms   native 165.12 ms
    ctx 16384: dense 445.53 ms   native 334.38 ms   (native 1.33x faster)
    ctx 32768: dense 909.99 ms   native 343.39 ms   (native 2.65x faster)

The previous gate (4096) enabled the kernel exactly where it loses. These
tests pin the decision — not the kernel — so they run without Metal.
"""

import pytest

try:
    import mlx.core as mx

    from omlx.patches.mlx_vlm_glm5_next_compat import (
        apply_mlx_vlm_glm5_next_compat_patch,
    )

    apply_mlx_vlm_glm5_next_compat_patch()
    from mlx_vlm.models.base import create_attention_mask
    from mlx_vlm.models.glm5_next import language as lang
    from tests.test_mlx_vlm_glm5_next_compat import _tiny_config

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


def _sparse_layer():
    mx.random.seed(3)
    cfg = _tiny_config().text_config
    attn = lang.Glm5NextSparseAttention(cfg)
    mx.eval(attn.parameters())
    from mlx_lm.models.cache import PoolingCache

    cache = lang.CacheList(lang.KVCache(), PoolingCache(attn.indexer.index_kpool))
    return attn, cache, cfg


def _run_prefill(attn, cache, cfg, ctx):
    x = mx.random.normal((1, ctx, cfg.hidden_size)).astype(mx.float16)
    out = attn(x, create_attention_mask(x, cache[0], return_array=True), cache)
    mx.eval(out)
    return out


def test_default_threshold_is_the_measured_crossover():
    assert lang._SPARSE_MLA_MIN_KV == 16384
    src = open(lang.__file__, encoding="utf-8").read()
    assert "OMLX_GLM5_SPARSE_MLA_MIN_KV" in src, "must stay environment-overridable"


@pytest.mark.parametrize("threshold, expect_native", [(10**9, False), (32, True)])
def test_native_sparse_mla_engages_only_above_the_threshold(
    monkeypatch, threshold, expect_native
):
    attn, cache, cfg = _sparse_layer()
    calls = []

    def spy(*args, **kwargs):
        calls.append(args[2].shape[2])  # Kv seen by the kernel
        return None  # "kernel unavailable": the dense path must still answer

    monkeypatch.setattr(lang, "sparse_mla_attention", spy)
    monkeypatch.setattr(lang, "_SPARSE_MLA_MIN_KV", threshold)

    out = _run_prefill(attn, cache, cfg, ctx=64)  # L=64 > 8: the wide path
    assert out.shape == (1, 64, cfg.hidden_size)
    assert bool(calls) is expect_native, (
        f"threshold={threshold}: native kernel {'called' if calls else 'skipped'} at Kv=64"
    )
