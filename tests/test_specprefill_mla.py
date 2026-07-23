# SPDX-License-Identifier: Apache-2.0
"""Tests for the MLA (Multi-head Latent Attention) SpecPrefill query extractor.

Covers DeepSeek/GLM-family draft models (e.g. glm4_moe_lite / GLM-4.7-Flash),
whose attention has no plain ``q_proj`` contract: queries go through an
optional low-rank projection and only the rope slice of the head dim is
rotated, and the KV cache stores the compressed latent (keys) plus the roped
positional slice (values).
"""

from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from omlx.patches.specprefill import (
    _detect_query_extractor,
    _is_mla_attention,
    _llama_extract_queries,
    _mla_extract_queries,
)


class TestMlaDetection:
    def test_detect_mla_low_rank_q(self):
        attn = MagicMock(
            spec=[
                "kv_a_proj_with_mqa",
                "q_a_proj",
                "q_a_layernorm",
                "q_b_proj",
                "embed_q",
                "rope",
                "num_heads",
                "q_head_dim",
                "qk_nope_head_dim",
                "scale",
            ]
        )
        assert _is_mla_attention(attn)
        assert _detect_query_extractor(attn) is _mla_extract_queries

    def test_detect_mla_full_q_wins_over_llama_fallback(self):
        # q_lora_rank=None MLA variants expose a plain q_proj; MLA detection
        # must still win, since the cache layout is latent, not per-head.
        attn = MagicMock(
            spec=[
                "kv_a_proj_with_mqa",
                "q_proj",
                "embed_q",
                "rope",
                "num_heads",
                "q_head_dim",
                "qk_nope_head_dim",
                "scale",
            ]
        )
        assert _detect_query_extractor(attn) is _mla_extract_queries

    def test_non_mla_dispatch_unchanged(self):
        attn = MagicMock(spec=["q_proj", "rope", "num_heads"])
        assert not _is_mla_attention(attn)
        assert _detect_query_extractor(attn) is _llama_extract_queries


class TestMlaExtractorNumerics:
    """Validate the extractor against the module's own math.

    glm4_moe_lite computes attention in two equivalent formulations: absorbed
    (decode: q_nope projected into latent space via embed_q) and expanded
    (prefill: latent keys projected into q_nope space via embed_q with
    transpose=False). The extractor uses the absorbed form against the cached
    latent; the reference below uses the module's expanded form with the same
    weights. Agreement validates layout, RoPE offset handling, and the
    pre-scaling that maps the scorer's generic 1/sqrt(d) onto the module's
    true attention scale.
    """

    def _tiny_args(self, glm, q_lora_rank):
        return glm.ModelArgs(
            hidden_size=64,
            num_attention_heads=2,
            num_key_value_heads=2,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=16,
            qk_rope_head_dim=8,
            qk_nope_head_dim=16,
            v_head_dim=16,
            num_hidden_layers=1,
            intermediate_size=128,
            moe_intermediate_size=32,
        )

    @pytest.mark.parametrize("q_lora_rank", [32, None])
    def test_scores_match_module_reference(self, q_lora_rank):
        glm = pytest.importorskip("mlx_lm.models.glm4_moe_lite")
        from mlx_lm.models.cache import KVCache

        args = self._tiny_args(glm, q_lora_rank)
        attn = glm.Glm4MoeLiteAttention(args)
        cache = KVCache()
        n_prompt = 5
        x_prompt = mx.random.normal((1, n_prompt, args.hidden_size))
        attn(x_prompt, mask=None, cache=cache)
        assert cache.offset == n_prompt

        x_step = mx.random.normal((1, 1, args.hidden_size))
        queries = _mla_extract_queries(attn, x_step, cache=cache)
        assert queries.shape == (
            1,
            attn.num_heads,
            1,
            args.kv_lora_rank + args.qk_rope_head_dim,
        )

        # Production scoring path: q_stack @ concat(keys, values)^T * 1/sqrt(d)
        latent = cache.keys[..., :n_prompt, :]
        k_pe = cache.values[..., :n_prompt, :]
        keys_cat = mx.concatenate(
            [
                mx.broadcast_to(
                    latent, (1, attn.num_heads, n_prompt, latent.shape[-1])
                ),
                mx.broadcast_to(k_pe, (1, attn.num_heads, n_prompt, k_pe.shape[-1])),
            ],
            axis=-1,
        )
        ext_scores = (queries @ keys_cat.swapaxes(-1, -2)) * (
            keys_cat.shape[-1] ** -0.5
        )

        # Reference: the module's own expanded (prefill-branch) formulation.
        if q_lora_rank is None:
            q_full = attn.q_proj(x_step)
        else:
            q_full = attn.q_b_proj(attn.q_a_layernorm(attn.q_a_proj(x_step)))
        q_full = q_full.reshape(1, 1, attn.num_heads, attn.q_head_dim).transpose(
            0, 2, 1, 3
        )
        q_nope, q_pe = mx.split(q_full, [attn.qk_nope_head_dim], axis=-1)
        q_pe = attn.rope(q_pe, cache.offset)
        k_expanded = attn.embed_q(latent, transpose=False)
        ref_scores = (
            q_nope @ k_expanded.swapaxes(-1, -2)
            + q_pe
            @ mx.broadcast_to(
                k_pe, (1, attn.num_heads, n_prompt, k_pe.shape[-1])
            ).swapaxes(-1, -2)
        ) * attn.scale

        assert mx.allclose(ext_scores, ref_scores, atol=1e-4, rtol=1e-4)
