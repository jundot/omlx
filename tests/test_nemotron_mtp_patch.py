from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nh = pytest.importorskip("mlx_lm.models.nemotron_h")

from omlx.patches.mlx_lm_mtp import (  # noqa: E402
    nemotron_h_chain,
    nemotron_h_model,
    set_mtp_active,
)

TINY_CONFIG = {
    "model_type": "nemotron_h",
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "max_position_embeddings": 256,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "attention_bias": False,
    "mamba_num_heads": 4,
    "mamba_head_dim": 16,
    "mamba_proj_bias": False,
    "ssm_state_size": 32,
    "conv_kernel": 4,
    "n_groups": 2,
    "mlp_bias": False,
    "layer_norm_epsilon": 1e-5,
    "use_bias": False,
    "use_conv_bias": True,
    "hybrid_override_pattern": ["M", "*"],
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 32,
    "moe_shared_expert_intermediate_size": 32,
    "n_shared_experts": 1,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "routed_scaling_factor": 1.0,
    "num_nextn_predict_layers": 1,
}


@pytest.fixture(autouse=True)
def _apply_patches():
    assert nemotron_h_model.apply()
    assert nemotron_h_chain.apply()
    yield
    set_mtp_active(False)


class TestLoaderGate:
    def test_nemotron_h_is_mtp_compatible(self):
        # The stock loader must route nemotron_h through the MTP patch;
        # without this gate the whole feature is inert on a stock server.
        from omlx.utils.model_loading import _is_mtp_compatible

        assert _is_mtp_compatible({"num_nextn_predict_layers": 1}, "nemotron_h")
        assert not _is_mtp_compatible({}, "nemotron_h")


class TestApply:
    def test_idempotent(self):
        mixer_call = nh.NemotronHMamba2Mixer.__call__
        assert nemotron_h_model.apply()
        assert nemotron_h_chain.apply()
        assert nh.NemotronHMamba2Mixer.__call__ is mixer_call

    def test_markers(self):
        assert getattr(nh.NemotronHMamba2Mixer.__call__, "_omlx_nh_chain", False)
        assert getattr(nh.Model.mtp_forward, "_omlx_nh_chain", False)
        assert callable(getattr(nh.Model, "mtp_partial_rollback", None))


class TestModelStamps:
    def test_mtp_attached_and_flagged_when_active(self):
        set_mtp_active(True)
        model = nh.Model(nh.ModelArgs.from_dict(TINY_CONFIG))
        assert hasattr(model, "mtp")
        assert model._omlx_mtp_decode_enabled
        assert model._omlx_mtp_chain
        assert model._omlx_mtp_head_hidden_normed

    def test_no_mtp_when_inactive(self):
        set_mtp_active(False)
        model = nh.Model(nh.ModelArgs.from_dict(TINY_CONFIG))
        assert not hasattr(model, "mtp")
        assert not model._omlx_mtp_decode_enabled

    def test_mtp_forward_return_hidden(self):
        set_mtp_active(True)
        model = nh.Model(nh.ModelArgs.from_dict(TINY_CONFIG))
        hidden = mx.zeros((1, 1, TINY_CONFIG["hidden_size"]))
        ids = mx.zeros((1, 1), dtype=mx.uint32)
        cache = model.make_mtp_cache()
        logits, head_hidden = model.mtp_forward(
            hidden, ids, cache, return_hidden=True, logits_keep=1
        )
        mx.eval(logits, head_hidden)
        assert logits.shape == (1, 1, TINY_CONFIG["vocab_size"])
        assert head_hidden.shape == hidden.shape


class TestVerifyCapture:
    def _mixer_and_cache(self):
        from mlx_lm.models.cache import ArraysCache

        args = nh.ModelArgs.from_dict(TINY_CONFIG)
        mixer = nh.NemotronHMamba2Mixer(args)
        mx.eval(mixer.parameters())
        return mixer, ArraysCache(size=2)

    def test_per_position_restore_matches_prefix_recompute(self):
        mx.random.seed(0)
        mixer, cache = self._mixer_and_cache()
        prefix = mx.random.normal((1, 3, TINY_CONFIG["hidden_size"]))
        window = mx.random.normal((1, 4, TINY_CONFIG["hidden_size"]))
        mx.eval(prefix, window)

        # Establish state, then run a verify window with capture armed.
        mixer(prefix, None, cache)
        mixer(window, None, cache, n_confirmed=1)
        assert cache._mtp_pos_states is not None
        assert len(cache._mtp_pos_states) == 4

        # Restore to keep=2 (confirmed + 1 accepted draft).
        conv_m, ssm_m = cache._mtp_pos_states[1]

        # Reference: fresh cache, prefix + the kept 2 window tokens.
        mixer2, cache2 = self._mixer_and_cache()
        mixer2.update(mixer.parameters())
        mx.eval(mixer2.parameters())
        mixer2(prefix, None, cache2)
        mixer2(window[:, :2], None, cache2)
        mx.eval(conv_m, ssm_m, cache2[0], cache2[1])

        assert mx.allclose(conv_m, cache2[0], atol=1e-5).item()
        assert mx.allclose(ssm_m, cache2[1], atol=1e-4, rtol=1e-3).item()
