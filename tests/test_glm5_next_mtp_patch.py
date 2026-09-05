# SPDX-License-Identifier: Apache-2.0
"""Tests for the GLM-5.3 (glm5_next, mlx-vlm vendored) native MTP patch."""

import sys

import mlx.core as mx
import pytest

from omlx.patches import mlx_vlm_glm5_next_compat as compat
from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch, set_mtp_active


@pytest.fixture(scope="module")
def glm():
    compat.apply_mlx_vlm_glm5_next_compat_patch()
    apply_mlx_lm_mtp_patch()
    return sys.modules["mlx_vlm.models.glm5_next"]


@pytest.fixture()
def mtp_active():
    set_mtp_active(True)
    yield
    set_mtp_active(False)


def _headful_text_config():
    from tests.test_mlx_vlm_glm5_next_compat import _tiny_config

    text = _tiny_config(with_vision=False).text_config
    text.num_nextn_predict_layers = 1
    text.layer_types = [
        "linear_attention",
        "deepseek_sparse_attention",
        "deepseek_sparse_attention",
    ]
    text.mlp_layer_types = ["dense", "dense", "sparse"]
    text.first_k_dense_replace = 0
    text.n_routed_experts = 4
    text.n_shared_experts = 1
    return text


class TestPatchApply:
    def test_patch_applies_and_noops_without_module(self, glm):
        from omlx.patches.mlx_lm_mtp import glm5_next_model

        assert glm5_next_model.apply() is True
        # Marker present on the patched class.
        assert glm.LanguageModel.__dict__.get("_omlx_mtp_patched") == "patch"

    def test_headless_init_leaves_model_stock(self, glm):
        text = _headful_text_config()
        lm = glm.LanguageModel(text)
        assert getattr(lm, "mtp", None) is None
        assert lm._omlx_mtp_decode_enabled is False
        assert not hasattr(lm, "_omlx_mtp_chain")


class TestHeadAttach:
    def test_active_load_attaches_chain_markers(self, glm, mtp_active):
        text = _headful_text_config()
        lm = glm.LanguageModel(text)
        assert lm.mtp is not None and len(lm.mtp) == 1
        assert lm._omlx_mtp_chain is True
        assert lm._omlx_mtp_depth >= 1
        assert lm._omlx_mtp_head_clone is False
        # GLM-5.2 head-input convention: the return_hidden wrapper feeds
        # the head the POST-final-norm hidden (hnorm re-normalises inside
        # the block), and the hidden-normed marker keeps the chain's
        # _trunk_norm_module the identity (no double norm).
        assert lm._omlx_mtp_head_prenorm is False
        assert lm._omlx_mtp_head_hidden_normed is True
        assert lm._omlx_mtp_decode_enabled is True

    def test_block_layout_matches_nextn_surface(self, glm, mtp_active):
        text = _headful_text_config()
        lm = glm.LanguageModel(text)
        block = lm.mtp[0]
        assert hasattr(block, "enorm") and hasattr(block, "hnorm")
        assert hasattr(block, "eh_proj") and hasattr(block, "norm")
        # Draft runs the full sparse-attention decoder layer (DSA indexer
        # + 288-expert MoE in the real checkpoint).
        assert type(block.block).__name__ == "Glm5NextDecoderLayer"
        assert block.block.is_linear is False
        assert type(block.block.mlp).__name__ == "Glm5NextMoE"

    def test_mtp_cache_is_flat_with_pooling(self, glm, mtp_active):
        text = _headful_text_config()
        lm = glm.LanguageModel(text)
        caches = lm.make_mtp_cache()
        # FLAT pair per draft block (not CacheList): _mtp_head_trim_to
        # reads offset on each entry directly. The pooling entry rides a
        # token-offset adapter (native offset counts pool windows).
        assert len(caches) == 2
        assert type(caches[0]).__name__ == "KVCache"
        assert type(caches[1]).__name__ == "_HeadPoolAdapter"
        pool = caches[1]._pool
        assert type(pool).__name__ == "PoolingCache"
        # adapter offset is token-space: pool_len * ratio + remainder
        assert caches[1].offset == pool.size() * pool.ratio + pool.remainder


class TestForward:
    def test_return_hidden_logits_bit_identical_to_stock(self, glm, mtp_active):
        text = _headful_text_config()
        mx.random.seed(11)
        lm = glm.LanguageModel(text)
        mx.eval(lm.parameters())
        tokens = mx.array([[1, 2, 3, 4, 5]])
        plain = lm(tokens, cache=None, num_logits_to_keep=2)
        rh = lm(tokens, cache=lm.make_cache(), return_hidden=True, num_logits_to_keep=2)
        assert mx.all(mx.equal(plain.logits, rh.logits)).item() is True
        # hidden is the pre-norm variant: applying the trunk's final norm
        # re-derives the post-norm hidden the stock path projects.
        post = lm.model.norm(rh.hidden_states)
        assert post.shape == rh.hidden_states.shape

    def test_mtp_forward_shapes_and_finiteness(self, glm, mtp_active):
        text = _headful_text_config()
        mx.random.seed(13)
        lm = glm.LanguageModel(text)
        mx.eval(lm.parameters())
        tokens = mx.array([[1, 2, 3, 4]])
        out = lm(tokens, cache=lm.make_cache(), return_hidden=True)
        h = out.hidden_states[:, -1:, :]
        caches = lm.make_mtp_cache()
        logits, head_hidden = lm.mtp_forward(
            h, mx.array([[5]]), caches, return_hidden=True
        )
        assert logits.shape == (1, 1, text.vocab_size)
        assert head_hidden.shape == (1, 1, text.hidden_size)
        assert mx.all(mx.isfinite(logits)).item() is True
        # Chained depth-2 step: the head's raw output feeds back as h.
        logits2, h2 = lm.mtp_forward(
            head_hidden, mx.array([[6]]), caches, return_hidden=True
        )
        assert logits2.shape == (1, 1, text.vocab_size)
        assert mx.all(mx.isfinite(logits2)).item() is True

    def test_get_mtp_module_via_adapter_contract(self, glm, mtp_active):
        text = _headful_text_config()
        lm = glm.LanguageModel(text)
        assert lm.get_mtp_module() is lm.mtp
        # get_mtp_module is a plain attr reader: delete the head and it
        # reports None (the adapter's VLMModelAdapter.mtp contract).
        del lm.mtp
        assert lm.get_mtp_module() is None


class TestQuantOverrideRemap:
    def test_nextn_overrides_copied_to_runtime_paths(self, glm, mtp_active):
        from omlx.patches.mlx_lm_mtp.glm5_next_model import (
            remap_mtp_quant_overrides,
        )

        three_bit = {"group_size": 128, "bits": 2}
        params = {
            "quantization": {
                "model.layers.2.mlp.switch_mlp.down_proj": dict(three_bit),
                "model.layers.2.eh_proj": dict(three_bit),
                "model.layers.2.self_attn.indexer.wk": {"group_size": 64, "bits": 8},
                "model.layers.2.shared_head.head": {"group_size": 64, "bits": 4},
                "model.layers.1.mlp.switch_mlp.down_proj": dict(three_bit),
            }
        }
        remap_mtp_quant_overrides(params, n_main=2, n_mtp=1)
        q = params["quantization"]
        assert q["mtp.0.block.mlp.switch_mlp.down_proj"] == three_bit
        assert q["mtp.0.eh_proj"] == three_bit
        assert q["mtp.0.block.self_attn.indexer.wk"] == {
            "group_size": 64,
            "bits": 8,
        }
        # Shared lm_head duplicate dropped; no runtime copy.
        assert not any("shared_head" in k for k in q if k.startswith("mtp."))
        # Backbone overrides untouched.
        assert "model.layers.1.mlp.switch_mlp.down_proj" in q
        assert not any("layers.1" in k for k in q if k.startswith("mtp."))

    def test_no_quant_block_is_inert(self, glm, mtp_active):
        from omlx.patches.mlx_lm_mtp.glm5_next_model import (
            remap_mtp_quant_overrides,
        )

        params = {"quantization": None}
        remap_mtp_quant_overrides(params, n_main=2, n_mtp=1)  # must not raise
        assert params["quantization"] is None


class TestPartialRollback:
    def test_trim_shortfall_refuses(self, glm, mtp_active):
        class _ShortCache:
            def is_trimmable(self):
                return True

            def trim(self, n):
                return 0

        lm = glm.LanguageModel(_headful_text_config())
        assert lm.mtp_partial_rollback([_ShortCache()], accepted=0, num_drafts=2) is False

    def test_no_rollback_needed_accepts(self, glm, mtp_active):
        lm = glm.LanguageModel(_headful_text_config())
        assert lm.mtp_partial_rollback([], accepted=2, num_drafts=2) is True

class TestVerifyCycleRollback:
    """Full depth-3 cycle on a linear+sparse hybrid trunk: fold history,
    verify window forward, then partial rollback at every accept count."""

    def _verify_and_rollback(self, lm, accepted, num_drafts=3):
        prompt = mx.array([[1, 2, 3, 4, 5, 6]])
        cache = lm.make_cache()
        lm(prompt, cache=cache)
        window = mx.array([[7, 8, 9, 10]])[:, : 1 + num_drafts]
        out = lm(window, cache=cache, return_hidden=True, n_confirmed=1)
        mx.eval(out.logits, out.hidden_states)
        ok = lm.mtp_partial_rollback(cache, accepted=accepted, num_drafts=num_drafts)
        # Expected state: prompt + confirmed + accepted drafts, replayed
        # one token at a time on a fresh cache.
        exp = lm.make_cache()
        lm(prompt, cache=exp)
        kept = [7] + [8, 9, 10][:accepted]
        lm(mx.array([kept]), cache=exp)
        lin = [i for i, l in enumerate(lm.model.layers) if l.is_linear]
        worst = 0.0
        for i in lin:
            worst = max(
                worst,
                mx.abs(cache[i][0] - exp[i][0]).max().item(),
                mx.abs(cache[i][1] - exp[i][1]).max().item(),
            )
        spa = [i for i, l in enumerate(lm.model.layers) if not l.is_linear]
        kv_ok = all(cache[i][0].offset == exp[i][0].offset for i in spa)
        pool_ok = all(cache[i][1].offset == exp[i][1].offset for i in spa)
        # Continuity: the next decode step must agree with the reference.
        nxt = 11 if accepted == num_drafts else 7 + 1 + accepted + 1
        o1 = lm(mx.array([[nxt]]), cache=cache)
        o2 = lm(mx.array([[nxt]]), cache=exp)
        mx.eval(o1.logits, o2.logits)
        cont = mx.abs(o1.logits - o2.logits).max().item()
        return ok, worst, kv_ok, pool_ok, cont

    def test_all_accept_counts_recover_exact_state(self, glm, mtp_active):
        text = _headful_text_config()
        mx.random.seed(7)
        lm = glm.LanguageModel(text)
        mx.eval(lm.parameters())
        for accepted in (0, 1, 2):
            ok, worst, kv_ok, pool_ok, cont = self._verify_and_rollback(
                lm, accepted
            )
            # ULP-level drift is inherent to windowed recurrent kernels
            # (same regime the qwen35 unsplit verify accepts); anything
            # larger indicates a stale stash or wrong prefix replay.
            assert ok is True
            assert worst < 1e-4, (accepted, worst)
            assert kv_ok and pool_ok
            assert cont < 1e-4, (accepted, cont)

    def test_unsplit_verify_forward_matches_sequential_logits(self, glm, mtp_active):
        text = _headful_text_config()
        mx.random.seed(19)
        lm = glm.LanguageModel(text)
        mx.eval(lm.parameters())
        prompt = mx.array([[1, 2, 3, 4]])
        c_seq = lm.make_cache()
        lm(prompt, cache=c_seq)
        last = None
        for t in (5, 6, 7):
            last = lm(mx.array([[t]]), cache=c_seq)
            mx.eval(last.logits)
        c_win = lm.make_cache()
        lm(prompt, cache=c_win)
        win = lm(mx.array([[5, 6, 7]]), cache=c_win, num_logits_to_keep=1)
        mx.eval(win.logits)
        d = mx.abs(last.logits - win.logits).max().item()
        assert d < 1e-4, d

    def test_clamp_accept_bounds_to_rollback_support(self, glm, mtp_active):
        text = _headful_text_config()
        lm = glm.LanguageModel(text)
        prompt = mx.array([[1, 2, 3, 4]])
        cache = lm.make_cache()
        lm(prompt, cache=cache)
        out = lm(mx.array([[5, 6, 7, 8]]), cache=cache, return_hidden=True, n_confirmed=1)
        mx.eval(out.logits)
        # Stash-backed rollback supports any partial accept while the
        # verify window is still stashed.
        assert lm.mtp_clamp_accept(cache, accepted=2, num_drafts=3) == 2
        # Full accept needs no rollback (n=0) and is always viable.
        assert lm.mtp_clamp_accept(cache, accepted=3, num_drafts=3) == 3
        # A consumed stash (cleared by an earlier rollback/full accept)
        # cannot replay a partial prefix: every m < num_drafts fails the
        # validation and the accept clamps to 0 (full reject semantics).
        for c in cache:
            if hasattr(c, "_mtp_draft_stash") and c._mtp_draft_stash is not None:
                c._mtp_draft_stash = None
                c.rollback_state = None
        assert lm.mtp_clamp_accept(cache, accepted=2, num_drafts=3) == 0


class TestHeadPoolAdapterTokens:
    def test_adapter_offset_is_token_space(self, glm, mtp_active):
        text = _headful_text_config()
        lm = glm.LanguageModel(text)
        hc = lm.make_mtp_cache()
        pool = hc[1]._pool
        # Fold three history rows: offset counts tokens, not windows.
        h = mx.zeros((1, 3, text.hidden_size))
        lm.mtp_forward(h, mx.array([[1, 2, 3]]), hc, logits_keep=1)
        mx.eval(hc[0].state[0]) if hasattr(hc[0], "state") else None
        assert hc[1].offset == pool.size() * pool.ratio + pool.remainder
        assert hc[1].offset == 3

    def test_speculative_tail_trims_back_to_history(self, glm, mtp_active):
        from omlx.patches.mlx_lm_mtp.batch_generator import _mtp_head_trim_to

        text = _headful_text_config()
        mx.random.seed(23)
        lm = glm.LanguageModel(text)
        mx.eval(lm.parameters())
        hc = lm.make_mtp_cache()
        h = mx.zeros((1, 1, text.hidden_size))
        lm.mtp_forward(h, mx.array([[5]]), hc, logits_keep=1)
        h2 = mx.zeros((1, 2, text.hidden_size))
        lm.mtp_forward(h2, mx.array([[6, 7]]), hc, logits_keep=1)
        assert hc[0].offset == 3 and hc[1].offset == 3
        _mtp_head_trim_to(hc, 1)
        assert hc[0].offset == 1
        assert hc[1].offset == 1

class TestPromptPriming:
    """Chunked prefill folds the head history through maybe_capture and
    the seam finishes via take_primed — the head KV must match a one-shot
    oracle fold of the same timeline."""

    def _prefill_chunked(self, lm, cache, tokens, chunks):
        i = 0
        for c in chunks:
            out = lm(tokens[:, i : i + c], cache=cache)
            mx.eval(out.logits)
            i += c

    def test_chunked_prefill_primes_and_seam_matches_oracle(self, glm, mtp_active):
        from omlx.patches.mlx_lm_mtp import prompt_priming

        text = _headful_text_config()
        mx.random.seed(41)
        lm = glm.LanguageModel(text)
        mx.eval(lm.parameters())

        n = 9
        tokens = mx.array([[11, 12, 13, 14, 15, 16, 17, 18, 19]])
        main_tok = mx.array([20])
        cache = lm.make_cache()
        self._prefill_chunked(lm, cache, tokens, [6, 3])
        stats = prompt_priming.prime_ctx_stats(lm)
        assert stats == n - 1

        # Activation forward (return_hidden): capture skips it.
        lm(main_tok[None, :], cache=cache, return_hidden=True)
        primed = prompt_priming.take_primed(lm, cache, main_tok)
        assert primed is not None
        mtp_cache, hist_offset = primed
        assert hist_offset == n
        assert prompt_priming._find_ctx(lm) is None

        # Oracle: one-shot prefill over the same tokens, head cache from a
        # single fold covering the whole history (incl. the seam token).
        ref_cache = lm.make_cache()
        out = lm(tokens, cache=ref_cache, return_hidden=True)
        mx.eval(out.hidden_states)
        ref_head = lm.make_mtp_cache()
        lm.mtp_forward(
            out.hidden_states[:, :-1, :],
            tokens[:, 1:],
            ref_head,
            logits_keep=1,
        )
        lm.mtp_forward(
            out.hidden_states[:, -1:, :], mx.array([[20]]), ref_head, logits_keep=1
        )
        kv_a = mtp_cache[0]
        kv_b = ref_head[0]
        assert kv_a.offset == kv_b.offset
        mx.eval(kv_a.state[0], kv_b.state[0])
        assert mx.allclose(kv_a.state[0], kv_b.state[0], rtol=1e-4, atol=1e-4)
        assert mtp_cache[1].offset == ref_head[1].offset

class TestDraftKvFusion:
    def test_unquantized_draft_kv_fuses_to_embed_unembed(self, glm, mtp_active):
        text = _headful_text_config()
        lm = glm.LanguageModel(text)
        heads, qk, vd, rank = 2, 8, 8, 8
        mx.random.seed(7)
        w = mx.random.normal((heads * (qk + vd), rank))
        out = lm.sanitize({"mtp.0.block.self_attn.kv_b_proj.weight": w})
        assert "mtp.0.block.self_attn.kv_b_proj.weight" not in out
        eq = out["mtp.0.block.self_attn.embed_q.weight"]
        uo = out["mtp.0.block.self_attn.unembed_out.weight"]
        assert tuple(eq.shape) == (heads, qk, rank)
        assert tuple(uo.shape) == (heads, vd, rank)
        mx.eval(eq, uo, w)
        v = w.reshape(heads, qk + vd, rank)
        assert mx.allclose(eq, mx.contiguous(v[:, :qk, :].swapaxes(-1, -2)))
        assert mx.allclose(uo, mx.contiguous(v[:, qk:, :]))



class TestDraftIdentityHC:
    def test_draft_hc_is_parameter_free(self, glm, mtp_active):
        text = _headful_text_config()
        lm = glm.LanguageModel(text)
        for name in ("attn_hc", "ffn_hc"):
            hc = getattr(lm.mtp[0].block, name)
            assert type(hc).__name__ == "_IdentityHC"
            assert dict(hc.parameters()) == {}

    def test_identity_hc_is_plain_residual(self, glm, mtp_active):
        from omlx.patches.deepseek_v4.hyper_connection import hc_expand

        text = _headful_text_config()
        lm = glm.LanguageModel(text)
        hc = lm.mtp[0].block.attn_hc
        mx.random.seed(13)
        base = mx.random.normal((1, 3, 32))
        x = mx.contiguous(mx.broadcast_to(base[:, :, None, :], (1, 3, 2, 32)))
        xc, post, comb = hc(x)
        mx.eval(xc, post, comb)
        assert tuple(xc.shape) == (1, 3, 32)
        assert mx.allclose(xc, base)
        r = mx.random.normal((1, 3, 32))
        y = hc_expand(r, x, post, comb)
        mx.eval(y)
        assert tuple(y.shape) == (1, 3, 2, 32)
        assert mx.allclose(y[:, :, 0, :], r + base)
        assert mx.allclose(y[:, :, 1, :], r + base)


class TestNextnQuantVariants:
    def test_expand_adds_mtp_runtime_keys(self):
        from omlx.utils.model_loading import expand_per_layer_quant_keys

        cfg = {
            "num_hidden_layers": 2,
            "text_config": {"num_nextn_predict_layers": 1},
            "quantization": {
                "group_size": 64,
                "bits": 8,
                "model.layers.2.mlp.switch_mlp.gate_proj": {
                    "group_size": 64,
                    "bits": 2,
                },
                "model.layers.2.eh_proj": {"group_size": 64, "bits": 8},
                "model.layers.1.mlp.switch_mlp.gate_proj": {
                    "group_size": 64,
                    "bits": 2,
                },
            },
        }
        expand_per_layer_quant_keys(cfg)
        q = cfg["quantization"]
        assert q["language_model.mtp.0.block.mlp.switch_mlp.gate_proj"] == {
            "group_size": 64,
            "bits": 2,
            "mode": "affine",
        }
        assert "language_model.mtp.0.eh_proj" in q
        assert not any(k.startswith("language_model.mtp.1.") for k in q)
        trunk_keys = [k for k in q if "mtp" in k and "layers.1." in k]
        assert trunk_keys == []


class TestDraftStreamingKeys:
    def test_candidate_keys_include_nextn_layout(self):
        from omlx.patches.expert_streaming import _mtp_candidate_stacked_keys

        ks = _mtp_candidate_stacked_keys(0, "gate_proj", "weight", trunk_layers=2)
        assert "model.layers.2.mlp.switch_mlp.gate_proj.weight" in ks
        assert "language_model.mtp.0.block.mlp.switch_mlp.gate_proj.weight" in ks

    def test_dsv4_candidates_unchanged_without_trunk_layers(self):
        from omlx.patches.expert_streaming import _mtp_candidate_stacked_keys

        assert _mtp_candidate_stacked_keys(0, "gate_proj", "weight") == [
            "mtp.0.ffn.switch_mlp.gate_proj.weight",
            "mtp.0.block.ffn.switch_mlp.gate_proj.weight",
        ]




