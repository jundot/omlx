# SPDX-License-Identifier: Apache-2.0
"""Tests for the GLM-5.2 (glm_moe_dsa) native MTP patch."""

import sys

import mlx.core as mx
import mlx.utils as mu
import pytest

from omlx.patches.glm_moe_dsa import apply_glm_moe_dsa_patch
from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch, set_mtp_active


@pytest.fixture(scope="module")
def glm():
    apply_glm_moe_dsa_patch()
    apply_mlx_lm_mtp_patch()
    return sys.modules["mlx_lm.models.glm_moe_dsa"]


@pytest.fixture()
def mtp_active():
    set_mtp_active(True)
    yield
    set_mtp_active(False)


TINY_CFG = dict(
    model_type="glm_moe_dsa",
    vocab_size=128,
    hidden_size=64,
    index_head_dim=32,
    index_n_heads=4,
    index_topk=16,
    intermediate_size=96,
    moe_intermediate_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    n_shared_experts=1,
    n_routed_experts=4,
    routed_scaling_factor=1.0,
    kv_lora_rank=32,
    q_lora_rank=48,
    qk_rope_head_dim=16,
    v_head_dim=32,
    qk_nope_head_dim=24,
    topk_method="noaux_tc",
    scoring_func="sigmoid",
    norm_topk_prob=True,
    n_group=1,
    topk_group=1,
    num_experts_per_tok=2,
    moe_layer_freq=1,
    first_k_dense_replace=1,
    max_position_embeddings=512,
    rms_norm_eps=1e-5,
    rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
    attention_bias=False,
    index_topk_freq=4,
    index_skip_topk_offset=3,
    indexer_types=["full", "shared"],
    num_nextn_predict_layers=1,
)


def _raw_hf_weights(glm, model):
    """Rebuild a raw-HF-layout weights dict from a built model's params.

    Inverts the sanitize transforms for the MTP layer (switch stacking,
    gate_up fusion, embed_q/unembed_out) so sanitize can be exercised on
    checkpoint-shaped input.
    """
    cfg = TINY_CFG
    flat = dict(mu.tree_flatten(model.parameters()))
    weights = {}
    for k, v in flat.items():
        if k.startswith("mtp.0."):
            rest = k[len("mtp.0."):]
            if rest.startswith("block."):
                rk = "model.layers.2." + rest[len("block."):]
            elif rest == "norm.weight":
                rk = "model.layers.2.shared_head.norm.weight"
            else:
                rk = "model.layers.2." + rest
            weights[rk] = v
        else:
            weights[k] = v

    raw = {}
    for k, v in weights.items():
        if ".mlp.switch_mlp.gate_up_proj.weight" in k:
            base = k.split(".mlp.switch_mlp.")[0]
            gate, up = mx.split(v, 2, axis=1)
            for e in range(v.shape[0]):
                raw[f"{base}.mlp.experts.{e}.gate_proj.weight"] = gate[e]
                raw[f"{base}.mlp.experts.{e}.up_proj.weight"] = up[e]
        elif ".mlp.switch_mlp.down_proj.weight" in k:
            base = k.split(".mlp.switch_mlp.")[0]
            for e in range(v.shape[0]):
                raw[f"{base}.mlp.experts.{e}.down_proj.weight"] = v[e]
        elif ".self_attn.embed_q.weight" in k:
            continue  # regenerated from the fabricated kv_b_proj below
        elif ".self_attn.unembed_out.weight" in k:
            base = k.split(".self_attn.")[0]
            nh = cfg["num_attention_heads"]
            hd = cfg["qk_nope_head_dim"] + cfg["v_head_dim"]
            raw[f"{base}.self_attn.kv_b_proj.weight"] = mx.random.normal(
                (nh * hd, cfg["kv_lora_rank"])
            )
        else:
            raw[k] = v
    return raw


class TestModelArgs:
    def test_nextn_count_and_indexer_extension(self, glm):
        args = glm.ModelArgs.from_dict(TINY_CFG)
        assert args.num_nextn_predict_layers == 1
        # freq=4/offset=3: layer 2 -> max(0,0)%4==0 -> "full"
        assert args.indexer_types == ["full", "shared", "full"]

    def test_no_nextn_is_untouched(self, glm):
        cfg = dict(TINY_CFG, num_nextn_predict_layers=0)
        args = glm.ModelArgs.from_dict(cfg)
        assert args.num_nextn_predict_layers == 0
        assert args.indexer_types == ["full", "shared"]


class TestModelInit:
    def test_mtp_attached_when_active(self, glm, mtp_active):
        args = glm.ModelArgs.from_dict(TINY_CFG)
        model = glm.Model(args)
        assert hasattr(model, "mtp") and len(model.mtp) == 1
        assert model._omlx_mtp_decode_enabled
        assert model._omlx_mtp_chain
        assert model._omlx_mtp_head_clone is False

    def test_mtp_skipped_when_inactive(self, glm):
        set_mtp_active(False)
        args = glm.ModelArgs.from_dict(TINY_CFG)
        model = glm.Model(args)
        assert not hasattr(model, "mtp")
        assert model._omlx_mtp_decode_enabled is False


class TestSanitize:
    def test_raw_hf_remap_and_strict_load(self, glm, mtp_active):
        mx.random.seed(0)
        args = glm.ModelArgs.from_dict(TINY_CFG)
        model = glm.Model(args)
        raw = _raw_hf_weights(glm, model)

        out = model.sanitize(raw)
        assert not any(".layers.2." in k for k in out)
        for expected in (
            "mtp.0.eh_proj.weight",
            "mtp.0.enorm.weight",
            "mtp.0.hnorm.weight",
            "mtp.0.norm.weight",
            "mtp.0.block.mlp.switch_mlp.gate_up_proj.weight",
            "mtp.0.block.self_attn.embed_q.weight",
            "mtp.0.block.self_attn.indexer.wk.weight",
        ):
            assert expected in out, expected
        model.load_weights(list(out.items()), strict=True)

    def test_layer_count_restored_after_sanitize(self, glm, mtp_active):
        args = glm.ModelArgs.from_dict(TINY_CFG)
        model = glm.Model(args)
        raw = _raw_hf_weights(glm, model)
        model.sanitize(raw)
        assert model.args.num_hidden_layers == TINY_CFG["num_hidden_layers"]

    def test_presanitized_passthrough(self, glm, mtp_active):
        """oQ-style checkpoints (already mtp.*) survive a second sanitize."""
        mx.random.seed(0)
        args = glm.ModelArgs.from_dict(TINY_CFG)
        model = glm.Model(args)
        once = model.sanitize(_raw_hf_weights(glm, model))
        twice = model.sanitize(dict(once))
        assert sorted(twice) == sorted(once)
        model.load_weights(list(twice.items()), strict=True)

    def test_mtp_off_drops_all_mtp_keys(self, glm, mtp_active):
        mx.random.seed(0)
        args = glm.ModelArgs.from_dict(TINY_CFG)
        model = glm.Model(args)
        sanitized = model.sanitize(_raw_hf_weights(glm, model))

        set_mtp_active(False)
        model_off = glm.Model(args)
        out = model_off.sanitize(dict(sanitized))
        assert not any(k.startswith("mtp.") for k in out)
        model_off.load_weights(list(out.items()), strict=True)

    def test_missing_head_weights_degrades_gracefully(self, glm, mtp_active):
        mx.random.seed(0)
        args = glm.ModelArgs.from_dict(TINY_CFG)
        model = glm.Model(args)
        sanitized = model.sanitize(_raw_hf_weights(glm, model))
        stripped = {k: v for k, v in sanitized.items() if not k.startswith("mtp.")}

        model2 = glm.Model(args)
        out = model2.sanitize(stripped)
        assert not hasattr(model2, "mtp")
        assert model2._omlx_mtp_decode_enabled is False
        model2.load_weights(list(out.items()), strict=True)


class TestIndexerFusion:
    def test_mtp_indexer_fused_alongside_backbone(self, glm, mtp_active):
        """MTP indexer fusion must happen before the stock sanitize: its
        backbone fusion pass drops every unfused ``.indexer.wk`` /
        ``.weights_proj`` key by substring, MTP keys included."""
        cfg = dict(TINY_CFG)
        q8 = {"bits": 8, "group_size": 64, "mode": "affine"}
        cfg["quantization"] = {
            "group_size": 64,
            "bits": 4,
            "mode": "affine",
            "model.layers.0.self_attn.indexer.wk": dict(q8),
            "model.layers.0.self_attn.indexer.weights_proj": dict(q8),
        }
        args = glm.ModelArgs.from_dict(cfg)
        model = glm.Model(args)
        assert model.mtp[0].block.self_attn.indexer.wk_weights_proj is not None

        h = TINY_CFG["hidden_size"]
        hd = TINY_CFG["index_head_dim"]
        nh = TINY_CFG["index_n_heads"]
        weights = {}
        for prefix in (
            "model.layers.0.self_attn.indexer",
            "mtp.0.block.self_attn.indexer",
        ):
            for suffix in ("weight", "scales", "biases"):
                weights[f"{prefix}.wk.{suffix}"] = mx.zeros((hd, 4))
                weights[f"{prefix}.weights_proj.{suffix}"] = mx.zeros((nh, 4))

        out = model.sanitize(weights)
        for prefix in (
            "model.layers.0.self_attn.indexer",
            "mtp.0.block.self_attn.indexer",
        ):
            assert f"{prefix}.wk_weights_proj.weight" in out, prefix
            assert f"{prefix}.wk.weight" not in out
            assert f"{prefix}.weights_proj.weight" not in out
        assert out["mtp.0.block.self_attn.indexer.wk_weights_proj.weight"].shape == (
            hd + nh,
            4,
        )


class TestForward:
    @pytest.fixture()
    def loaded(self, glm, mtp_active):
        mx.random.seed(0)
        args = glm.ModelArgs.from_dict(TINY_CFG)
        model = glm.Model(args)
        out = model.sanitize(_raw_hf_weights(glm, model))
        model.load_weights(list(out.items()), strict=True)
        mx.eval(model.parameters())
        return model

    def test_return_hidden_and_mtp_cycle(self, loaded):
        model = loaded
        cache = model.make_cache()
        toks = mx.array([[1, 2, 3, 4]])
        logits, hidden = model(toks, cache=cache, return_hidden=True)
        mx.eval(logits, hidden)
        assert logits.shape == (1, 4, TINY_CFG["vocab_size"])
        assert hidden.shape == (1, 4, TINY_CFG["hidden_size"])

        # hidden is pre-norm: normed hidden feeds the head (post-norm contract)
        post = model.model.norm(hidden)
        mtp_cache = model.make_mtp_cache()
        assert isinstance(mtp_cache, list) and len(mtp_cache) == 2

        lg, hh = model.mtp_forward(
            post, toks, mtp_cache, return_hidden=True, logits_keep=1
        )
        mx.eval(lg, hh)
        assert lg.shape == (1, 1, TINY_CFG["vocab_size"])
        assert hh.shape == (1, 4, TINY_CFG["hidden_size"])
        assert mtp_cache[0].offset == 4 and mtp_cache[1].offset == 4

        # chained draft step + rollback trim
        lg2, _ = model.mtp_forward(
            hh[:, -1:], mx.array([[7]]), mtp_cache, return_hidden=True
        )
        mx.eval(lg2)
        assert mtp_cache[0].offset == 5

        from omlx.patches.mlx_lm_mtp.batch_generator import _mtp_head_trim_to

        _mtp_head_trim_to(mtp_cache, 4)
        assert mtp_cache[0].offset == 4 and mtp_cache[1].offset == 4

    def test_partial_rollback_trims_verify_window(self, loaded):
        model = loaded
        cache = model.make_cache()
        logits, _ = model(mx.array([[1, 2, 3]]), cache=cache, return_hidden=True)
        mx.eval(logits, *(c[0].keys for c in cache))
        base = cache[0][0].offset

        # verify window: num_drafts + 1 rows, accept 1 of 3 drafts
        logits, _ = model(
            mx.array([[4, 5, 6, 7]]), cache=cache, return_hidden=True
        )
        mx.eval(logits, *(c[0].keys for c in cache))
        assert cache[0][0].offset == base + 4

        assert model.mtp_partial_rollback(cache, 1, 3)
        for c in cache:
            for sub in c.caches:  # latent KV (+ indexer KV on full layers)
                assert sub.offset == base + 2  # next_main + 1 accepted draft

    def test_n_confirmed_accepted(self, loaded):
        cache = loaded.make_cache()
        logits, _ = loaded(
            mx.array([[1, 2]]), cache=cache, return_hidden=True, n_confirmed=1
        )
        mx.eval(logits)
        assert logits.shape == (1, 2, TINY_CFG["vocab_size"])


class TestSmallLRouting:
    def test_absorbed_matches_materialized(self, glm, mtp_active):
        """The widened L<=8 absorbed path equals the legacy materialize path."""
        import omlx.patches.glm_moe_dsa.glm_moe_dsa_model as gm
        from mlx_lm.models.base import create_attention_mask
        from mlx_lm.models.cache import KVCache

        mx.random.seed(3)
        args = glm.ModelArgs.from_dict(TINY_CFG)
        attn = glm.GlmMoeDsaAttention(args, 0)
        mx.eval(attn.parameters())

        def run(L, max_l):
            mx.random.seed(11)
            cache = [KVCache(), KVCache()]
            x_pre = mx.random.normal((1, 12, TINY_CFG["hidden_size"]))
            mask = create_attention_mask(x_pre, cache[0], return_array=True)
            out, _ = attn(x_pre, mask, cache, None)
            mx.eval(out)
            x = mx.random.normal((1, L, TINY_CFG["hidden_size"]))
            mask = create_attention_mask(x, cache[0], return_array=True)
            saved = gm._ABSORBED_DECODE_MAX_L
            gm._ABSORBED_DECODE_MAX_L = max_l
            try:
                out, _ = attn(x, mask, cache, None)
                mx.eval(out)
            finally:
                gm._ABSORBED_DECODE_MAX_L = saved
            return out

        for L in (2, 3, 4, 8):
            legacy = run(L, 1)
            absorbed = run(L, 8)
            diff = float(mx.abs(legacy - absorbed).max())
            assert diff < 2e-5, f"L={L}: {diff}"

    def test_topk_gather_matches_masked_reference(self, glm, mtp_active):
        """With the DSA indexer active (K > index_topk), the decode-shape
        per-row gather path must equal the legacy masked full-K path."""
        import omlx.patches.glm_moe_dsa.glm_moe_dsa_model as gm
        from mlx_lm.models.base import create_attention_mask
        from mlx_lm.models.cache import KVCache

        mx.random.seed(5)
        args = glm.ModelArgs.from_dict(TINY_CFG)
        attn = glm.GlmMoeDsaAttention(args, 0)
        mx.eval(attn.parameters())

        def run(L, max_l):
            mx.random.seed(17)
            cache = [KVCache(), KVCache()]
            # Prefill past index_topk (16) so the indexer emits topk state.
            x_pre = mx.random.normal((1, 24, TINY_CFG["hidden_size"]))
            mask = create_attention_mask(x_pre, cache[0], return_array=True)
            out, _ = attn(x_pre, mask, cache, None)
            mx.eval(out)
            x = mx.random.normal((1, L, TINY_CFG["hidden_size"]))
            mask = create_attention_mask(x, cache[0], return_array=True)
            saved = gm._ABSORBED_DECODE_MAX_L
            gm._ABSORBED_DECODE_MAX_L = max_l
            try:
                out, state = attn(x, mask, cache, None)
                mx.eval(out)
            finally:
                gm._ABSORBED_DECODE_MAX_L = saved
            return out, state

        for L in (2, 3, 4):
            legacy, legacy_state = run(L, 1)  # masked materialize fallback
            gathered, state = run(L, 8)  # per-row topk gather
            idx, prefix = gm._parse_topk_state(state)
            assert idx is not None and idx.shape[2] == L and prefix == 0
            diff = float(mx.abs(legacy - gathered).max())
            assert diff < 2e-5, f"L={L}: {diff}"
