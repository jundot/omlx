# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DeepSeek-V4.1 Flash omlx patch (no full weights required)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import pytest


@pytest.fixture(scope="module")
def applied_patch():
    from omlx.patches.deepseek_v41 import apply_deepseek_v41_patch

    apply_deepseek_v41_patch()
    return True


class TestPredicates:
    def test_v4_excludes_v41(self):
        from omlx.patches.deepseek_v41.predicates import (
            is_deepseek_v4,
            is_deepseek_v41,
            is_deepseek_v4_family,
        )

        assert is_deepseek_v41("deepseek_v41")
        assert is_deepseek_v41("deepseek_v41_text")
        assert not is_deepseek_v4("deepseek_v41")
        assert not is_deepseek_v4("deepseek_v41_text")
        assert is_deepseek_v4("deepseek_v4")
        assert is_deepseek_v4("deepseek_v4_mtp")
        assert is_deepseek_v4_family("deepseek_v41")
        assert is_deepseek_v4_family("deepseek_v4")

    def test_startswith_collision(self):
        # Document the bug we fixed
        assert "deepseek_v41".startswith("deepseek_v4")
        from omlx.patches.deepseek_v41.predicates import is_deepseek_v4

        assert not is_deepseek_v4("deepseek_v41")


class TestModelArgs:
    def test_flatten_text_config(self, applied_patch):
        from mlx_lm.models.deepseek_v41 import ModelArgs

        raw = {
            "model_type": "deepseek_v41",
            "text_config": {
                "hidden_size": 5120,
                "num_hidden_layers": 4,
                "moe_intermediate_size": 2304,
                "n_routed_experts": 8,
                "num_experts_per_tok": 2,
                "rms_norm_eps": 1e-20,
                "compress_ratios": [0, 2, 1, 0],
                "kv_source_layer_ids": [1],
                "index_source_layer_ids": [1],
                "index_n_heads": 32,
                "hc_mult": 4,
                "engram_layer_ids": [1],
                "engram_num_embeddings": [1024],
                "engram_max_ngram_size": 4,
                "engram_n_heads": 8,
                "engram_head_dim": 256,
                "dspark_target_layer_ids": [37, 38, 39],
                "dspark_block_size": 5,
            },
        }
        args = ModelArgs.from_dict(raw)
        assert args.model_type == "deepseek_v41"
        assert args.hidden_size == 5120
        assert args.num_hidden_layers == 4
        assert args.rms_norm_eps == 1e-20
        assert args.compress_ratios == [0, 2, 1, 0]
        assert args.index_n_heads == 32
        assert args.kv_source_layer_ids == [1]
        assert args.dspark_target_layer_ids == [37, 38, 39]

    def test_compress_ratios_validation(self, applied_patch):
        from mlx_lm.models.deepseek_v41 import ModelArgs

        with pytest.raises(ValueError, match="compress ratios"):
            ModelArgs(
                num_hidden_layers=3,
                compress_ratios=[0, 4, 128],
            )

        ok = ModelArgs(num_hidden_layers=3, compress_ratios=[0, 1, 2])
        assert ok.compress_ratios == [0, 1, 2]

    def test_official_config_json(self, applied_patch):
        cfg_path = Path("/tmp/dsv41-work/config.json")
        if not cfg_path.exists():
            pytest.skip("official config not available")
        from mlx_lm.models.deepseek_v41 import ModelArgs

        raw = json.loads(cfg_path.read_text())
        args = ModelArgs.from_dict(raw)
        assert args.num_hidden_layers == 40
        assert set(args.compress_ratios) <= {0, 1, 2}
        import os
        if os.environ.get("OMLX_DSV41_ENGRAM", "stub").strip().lower() == "full":
            assert args.engram_layer_ids == [1, 14]
        else:
            assert args.engram_layer_ids == []  # stub mode strips Engram hooks
        assert args.index_n_heads == 32


class TestSelectCandidateBlocks:
    def test_shapes_and_pin_last(self):
        from omlx.patches.deepseek_v41.select_candidates import select_candidate_blocks

        # 1 query, 16 positions, block_size=4 -> 4 blocks
        positions = mx.arange(16)
        logits = mx.where(positions == 0, mx.array(100.0), mx.array(-10.0))
        logits = logits[None, :]
        mask = select_candidate_blocks(logits, compress_lens=16, topk_blocks=2, block_size=4)
        assert mask.shape == (1, 16)
        assert mask.dtype == mx.bool_
        # last block (positions 12-15) pinned + block 0
        assert bool(mask[0, 0])
        assert bool(mask[0, 15])


class TestSharedAttentionRuntime:
    def test_pool_identity(self, applied_patch):
        from omlx.patches.deepseek_v4 import apply_pooling_cache_support

        apply_pooling_cache_support()
        from mlx_lm.models.cache import PoolingCache
        from omlx.patches.deepseek_v41.shared_runtime import SharedAttentionRuntime

        rt = SharedAttentionRuntime()
        p1 = rt.register_kv_pool(2, PoolingCache(2))
        p2 = rt.register_kv_pool(2, PoolingCache(2))
        assert p1 is p2
        assert rt.kv_source_for(5, [2, 8, 14]) == 2
        assert rt.kv_source_for(10, [2, 8, 14]) == 8


class TestDeferredHC:
    def test_mix_shapes(self, applied_patch):
        from mlx_lm.models.deepseek_v41 import ModelArgs
        from omlx.patches.deepseek_v41.deferred_hc import (
            DeferredHyperConnection,
            hc_collapse,
            make_identity_pre_mix,
        )

        args = ModelArgs(
            num_hidden_layers=2,
            hidden_size=64,
            hc_mult=4,
            compress_ratios=[0, 0],
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            num_attention_heads=4,
            head_dim=16,
            q_lora_rank=32,
            o_lora_rank=32,
            o_groups=2,
            index_n_heads=4,
            index_head_dim=16,
        )
        hc = DeferredHyperConnection(args)
        B, L, H, D = 1, 3, 4, 64
        x = mx.random.normal((B, L, H, D))
        pre, post, comb = hc.mixes(x)
        assert pre.shape == (B, L, H)
        assert post.shape == (B, L, H)
        assert comb.shape == (B, L, H, H)
        identity = make_identity_pre_mix(x, H)
        collapsed = hc_collapse(x, identity)
        assert collapsed.shape == (B, L, D)


class TestSanitize:
    def test_key_remaps(self, applied_patch):
        from mlx_lm.models.deepseek_v41 import Model, ModelArgs

        args = ModelArgs(
            num_hidden_layers=2,
            hidden_size=64,
            vocab_size=128,
            compress_ratios=[0, 0],
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            num_attention_heads=4,
            head_dim=16,
            q_lora_rank=32,
            o_lora_rank=32,
            o_groups=2,
            index_n_heads=4,
            hc_mult=4,
        )
        model = Model(args)
        fake = {
            "embed.weight": mx.zeros((128, 64)),
            "head.weight": mx.zeros((128, 64)),
            "norm.weight": mx.zeros((64,)),
            "hc_head_fn": mx.zeros((4, 256)),
            "layers.0.ffn.gate.bias": mx.zeros((4,)),
            "mtp.0.main_proj.weight": mx.zeros((64, 64)),
        }
        out = model.sanitize(fake)
        assert "model.embed_tokens.weight" in out
        assert "lm_head.weight" in out
        assert "model.hc_head.fn" in out
        assert "model.layers.0.ffn.gate.e_score_correction_bias" in out
        assert "mtp.0.main_proj.weight" not in out


class TestSmokeImport:
    def test_apply_patch_import(self):
        from omlx.patches.deepseek_v41 import apply_deepseek_v41_patch

        apply_deepseek_v41_patch()
        assert "mlx_lm.models.deepseek_v41" in sys.modules

    def test_make_cache_shared_pools(self, applied_patch):
        from mlx_lm.models.deepseek_v41 import Model, ModelArgs

        args = ModelArgs(
            num_hidden_layers=6,
            hidden_size=64,
            vocab_size=128,
            compress_ratios=[0, 0, 2, 2, 2, 0],
            kv_source_layer_ids=[2],
            index_source_layer_ids=[2],
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            num_attention_heads=4,
            head_dim=16,
            q_lora_rank=32,
            o_lora_rank=32,
            o_groups=2,
            index_n_heads=4,
            hc_mult=4,
        )
        model = Model(args)
        caches = model.make_cache()
        assert len(caches) == 6
        # layers 3,4 should reuse layer 2's kv pool
        from mlx_lm.models.cache import CacheList

        assert isinstance(caches[2], CacheList)
        assert isinstance(caches[3], CacheList)
        assert caches[2][1] is caches[3][1]
