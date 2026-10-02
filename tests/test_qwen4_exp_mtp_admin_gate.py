# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the admin Lightning-MTP gates with qwen4_exp.

Qwen3.8 Flash Next (``model_type == "qwen4_exp"``) attaches its Lightning
MTP head through the dedicated VLM path in ``omlx.utils.model_loading``
(vendored mlx-vlm qwen4_exp model + ``mlx_lm_mtp`` dispatch patch) and is
deliberately absent from the mlx-lm ``_is_mtp_compatible`` whitelist, which
is the runtime gate for the *generic* text-model patch.

Both admin gates reused that whitelist, so the Lightning MTP toggle reported
"model_type='qwen4_exp' is not on the MTP whitelist" and saving the setting
returned 400 even though the runtime supports the head as shipped by the
#3174 converter. The admin gates must instead accept qwen4_exp and fall
through to the embedded ``mtp.*`` weight check — the same condition the
runtime path applies.
"""

import json

import pytest

from omlx.admin.routes import _mtp_compat_for_model

QWEN4_EXP_CONFIG = {
    "model_type": "qwen4_exp",
    "text_config": {
        "model_type": "qwen4_exp_text",
        "num_hidden_layers": 48,
        "mtp_num_hidden_layers": 1,
        "mtp": {"hybrid": True, "num_hidden_layers": 1},
    },
}


def _make_checkpoint(
    tmp_path,
    config,
    mtp_weights,
    name="Qwen3.8-Flash-Next",
    nextn_weights=False,
):
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(config))
    weight_map = {"model.layers.0.mlp.down.weight": "model.safetensors"}
    if mtp_weights:
        weight_map["mtp.fc_hidden.weight"] = "model.safetensors"
    if nextn_weights:
        weight_map["model.layers.48.self_attn.q_proj.weight"] = "model.safetensors"
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    return model_dir


class TestMtpCompatForModelQwen4Exp:
    def test_qwen4_exp_with_embedded_mtp_weights_is_compatible(self, tmp_path):
        model_dir = _make_checkpoint(tmp_path, QWEN4_EXP_CONFIG, mtp_weights=True)
        ok, reason = _mtp_compat_for_model({"model_path": str(model_dir)})
        assert ok, f"qwen4_exp with mtp.* weights must pass: {reason}"
        assert reason == ""

    def test_qwen4_exp_without_mtp_weights_is_blocked_on_weights(self, tmp_path):
        model_dir = _make_checkpoint(tmp_path, QWEN4_EXP_CONFIG, mtp_weights=False)
        ok, reason = _mtp_compat_for_model({"model_path": str(model_dir)})
        assert not ok
        # The rejection must come from the missing-weight check, not the
        # whitelist: the weights check is what the runtime path enforces.
        assert "whitelist" not in reason
        assert "mtp.* tensors" in reason

    def test_qwen4_exp_with_only_nextn_weights_is_blocked(self, tmp_path):
        config = json.loads(json.dumps(QWEN4_EXP_CONFIG))
        config["text_config"]["num_nextn_predict_layers"] = 1
        model_dir = _make_checkpoint(
            tmp_path,
            config,
            mtp_weights=False,
            nextn_weights=True,
        )
        ok, reason = _mtp_compat_for_model({"model_path": str(model_dir)})
        assert not ok
        assert "native nextn layers are not supported" in reason

    def test_qwen4_exp_without_mtp_heads_is_blocked(self, tmp_path):
        config = {"model_type": "qwen4_exp", "text_config": {}}
        model_dir = _make_checkpoint(tmp_path, config, mtp_weights=True)
        ok, reason = _mtp_compat_for_model({"model_path": str(model_dir)})
        assert not ok
        assert "no MTP heads" in reason

    def test_unsupported_model_type_still_hits_the_whitelist(self, tmp_path):
        config = {"model_type": "llama", "mtp_num_hidden_layers": 1}
        model_dir = _make_checkpoint(tmp_path, config, mtp_weights=True)
        ok, reason = _mtp_compat_for_model({"model_path": str(model_dir)})
        assert not ok
        assert "not on the MTP whitelist" in reason

    @pytest.mark.parametrize("model_type", ["qwen3_5_moe", "qwen3_6", "deepseek_v4"])
    def test_generic_whitelisted_types_still_pass(self, tmp_path, model_type):
        config = {"model_type": model_type, "mtp_num_hidden_layers": 1}
        model_dir = _make_checkpoint(tmp_path, config, mtp_weights=True, name="m")
        ok, reason = _mtp_compat_for_model({"model_path": str(model_dir)})
        assert ok, f"{model_type} must remain compatible: {reason}"
