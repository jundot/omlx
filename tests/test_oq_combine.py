# SPDX-License-Identifier: Apache-2.0
"""Tests for the MTP combine steps in omlx.oq (gemma4 assistant merge and
the native Qwen3.5/3.6 donor head graft).

Uses tiny synthetic checkpoints on disk — no model loading, no GPU work
beyond a few small mx arrays.
"""

from __future__ import annotations

import json

import mlx.core as mx
import pytest

from omlx.oq import (
    GEMMA4_ASSISTANT_MTP_PREFIX,
    GEMMA4_ASSISTANT_MTP_SHARD,
    MTPLX_RUNTIME_FILE,
    MTPLX_SIDECAR_SHARD,
    combine_gemma4_assistant_mtp,
    combine_mtp_donor,
    combine_mtp_into_output,
    import_mtplx_sidecar,
    validate_gemma4_assistant_pair,
    validate_mtp_donor_pair,
)

BASE_CONFIG = {
    "model_type": "gemma4",
    "vision_config": {},
    "text_config": {"model_type": "gemma4_text", "hidden_size": 24},
    "quantization": {"group_size": 64, "bits": 4},
}

ASSISTANT_CONFIG = {
    "model_type": "gemma4_assistant",
    "backbone_hidden_size": 24,
    "tie_word_embeddings": True,
    "text_config": {"model_type": "gemma4_text", "hidden_size": 8, "num_hidden_layers": 2},
}


def _write_base_output(tmp_path):
    out = tmp_path / "base-oQ4"
    out.mkdir()
    (out / "config.json").write_text(json.dumps(BASE_CONFIG))
    weights = {"language_model.model.embed_tokens.weight": mx.zeros((4, 24))}
    mx.save_safetensors(str(out / "model-00001-of-00001.safetensors"), weights)
    index = {
        "metadata": {"total_size": 100},
        "weight_map": {
            k: "model-00001-of-00001.safetensors" for k in weights
        },
    }
    (out / "model.safetensors.index.json").write_text(json.dumps(index))
    return out


def _write_assistant(tmp_path, config=None):
    asst = tmp_path / "assistant"
    asst.mkdir()
    (asst / "config.json").write_text(json.dumps(config or ASSISTANT_CONFIG))
    weights = {
        "model.embed_tokens.weight": mx.ones((4, 8)),
        "pre_projection.weight": mx.ones((8, 48)),
        "post_projection.weight": mx.ones((24, 8)),
    }
    mx.save_safetensors(str(asst / "model.safetensors"), weights)
    return asst


def test_combine_writes_shard_index_and_config(tmp_path):
    out = _write_base_output(tmp_path)
    asst = _write_assistant(tmp_path)

    combine_gemma4_assistant_mtp(out, asst)

    shard = out / GEMMA4_ASSISTANT_MTP_SHARD
    assert shard.exists()
    merged = mx.load(str(shard))
    assert set(merged) == {
        GEMMA4_ASSISTANT_MTP_PREFIX + "model.embed_tokens.weight",
        GEMMA4_ASSISTANT_MTP_PREFIX + "pre_projection.weight",
        GEMMA4_ASSISTANT_MTP_PREFIX + "post_projection.weight",
    }

    index = json.loads((out / "model.safetensors.index.json").read_text())
    for key in merged:
        assert index["weight_map"][key] == GEMMA4_ASSISTANT_MTP_SHARD
    # Base entries survive and total_size grows by the mtp shard bytes.
    assert (
        index["weight_map"]["language_model.model.embed_tokens.weight"]
        == "model-00001-of-00001.safetensors"
    )
    mtp_bytes = sum(v.nbytes for v in merged.values())
    assert index["metadata"]["total_size"] == 100 + mtp_bytes

    config = json.loads((out / "config.json").read_text())
    tc = config["text_config"]
    assert tc["mtp_num_hidden_layers"] == 2
    assert tc["mtp_assistant_config"] == ASSISTANT_CONFIG
    # Base fields untouched.
    assert config["quantization"] == BASE_CONFIG["quantization"]
    assert tc["hidden_size"] == 24


def test_combine_rejects_non_assistant_model(tmp_path):
    out = _write_base_output(tmp_path)
    wrong = dict(ASSISTANT_CONFIG)
    wrong["model_type"] = "gemma4"
    asst = _write_assistant(tmp_path, config=wrong)
    with pytest.raises(ValueError, match="gemma4_assistant"):
        combine_gemma4_assistant_mtp(out, asst)


def test_validate_rejects_hidden_size_mismatch():
    mismatched = dict(ASSISTANT_CONFIG)
    mismatched["backbone_hidden_size"] = 32
    with pytest.raises(ValueError, match="backbone_hidden_size"):
        validate_gemma4_assistant_pair(BASE_CONFIG, mismatched)


def test_validate_rejects_non_gemma4_base():
    base = dict(BASE_CONFIG)
    base["model_type"] = "qwen3_5"
    with pytest.raises(ValueError, match="gemma4 base"):
        validate_gemma4_assistant_pair(base, ASSISTANT_CONFIG)


def test_validate_rejects_headless_assistant():
    headless = dict(ASSISTANT_CONFIG)
    headless["text_config"] = {"model_type": "gemma4_text"}
    with pytest.raises(ValueError, match="num_hidden_layers"):
        validate_gemma4_assistant_pair(BASE_CONFIG, headless)


# ── Native Qwen3.5/3.6 donor head graft ─────────────────────────────────

QWEN_GEOMETRY = {
    "vocab_size": 16,
    "hidden_size": 8,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 4,
    "intermediate_size": 16,
    "rms_norm_eps": 1e-06,
    "rope_theta": 10000,
}

TOKENIZER_BYTES = b'{"version": "qwen-test-tokenizer"}'


def _qwen_config(*, vlm: bool, **scope_overrides):
    scope = {"num_hidden_layers": 2, **QWEN_GEOMETRY}
    scope.update(scope_overrides)
    if vlm:
        scope.setdefault("model_type", "qwen3_5_text")
        return {"model_type": "qwen3_5", "vision_config": {}, "text_config": scope}
    scope.setdefault("model_type", "qwen3_5")
    return scope


def _write_qwen_output(tmp_path, *, vlm=False, rope_nested=False):
    out = tmp_path / ("qwen-vlm-oQ6" if vlm else "qwen-oQ6")
    out.mkdir()
    config = _qwen_config(vlm=vlm)
    scope = config["text_config"] if vlm else config
    if rope_nested:
        scope["rope_parameters"] = {"rope_theta": scope.pop("rope_theta")}
    quant = {"group_size": 64, "bits": 6, "mode": "affine"}
    config["quantization"] = dict(quant)
    config["quantization_config"] = dict(quant)
    (out / "config.json").write_text(json.dumps(config))
    prefix = "language_model." if vlm else ""
    weights = {prefix + "model.embed_tokens.weight": mx.zeros((16, 8))}
    mx.save_safetensors(str(out / "model-00001-of-00001.safetensors"), weights)
    index = {
        "metadata": {"total_size": 100},
        "weight_map": {k: "model-00001-of-00001.safetensors" for k in weights},
    }
    (out / "model.safetensors.index.json").write_text(json.dumps(index))
    (out / "tokenizer.json").write_bytes(TOKENIZER_BYTES)
    return out


def _write_qwen_donor(
    tmp_path,
    *,
    vlm=False,
    quantized=False,
    headless=False,
    tokenizer=TOKENIZER_BYTES,
    **scope_overrides,
):
    donor = tmp_path / ("qwen-donor-vlm" if vlm else "qwen-donor")
    donor.mkdir()
    scope_overrides.setdefault("mtp_num_hidden_layers", 1)
    config = _qwen_config(vlm=vlm, **scope_overrides)
    prefix = "language_model." if vlm else ""
    bf16 = mx.bfloat16
    weights = {
        prefix + "model.embed_tokens.weight": mx.zeros((16, 8), dtype=bf16),
    }
    if not headless:
        weights.update(
            {
                prefix + "mtp.fc.weight": mx.arange(128, dtype=mx.float32)
                .reshape(8, 16)
                .astype(bf16),
                prefix + "mtp.norm.weight": mx.ones((8,), dtype=bf16),
                prefix + "mtp.pre_fc_norm_embedding.weight": mx.ones((8,), dtype=bf16),
                prefix + "mtp.pre_fc_norm_hidden.weight": mx.ones((8,), dtype=bf16),
                prefix
                + "mtp.layers.0.input_layernorm.weight": mx.ones((8,), dtype=bf16),
            }
        )
        if quantized:
            weights.update(
                {
                    prefix
                    + "mtp.layers.0.self_attn.q_proj.weight": mx.full(
                        (8, 1), 7, dtype=mx.uint32
                    ),
                    prefix
                    + "mtp.layers.0.self_attn.q_proj.scales": mx.ones(
                        (8, 1), dtype=bf16
                    ),
                    prefix
                    + "mtp.layers.0.self_attn.q_proj.biases": mx.zeros(
                        (8, 1), dtype=bf16
                    ),
                    prefix
                    + "mtp.layers.0.mlp.gate_proj.weight": mx.full(
                        (16, 1), 3, dtype=mx.uint32
                    ),
                    prefix
                    + "mtp.layers.0.mlp.gate_proj.scales": mx.ones((16, 1), dtype=bf16),
                    prefix
                    + "mtp.layers.0.mlp.gate_proj.biases": mx.zeros(
                        (16, 1), dtype=bf16
                    ),
                }
            )
            quant = {"group_size": 64, "bits": 4, "mode": "affine"}
            # One module rides a per-layer override, the other the global.
            quant[prefix + "mtp.layers.0.self_attn.q_proj"] = {
                "group_size": 32,
                "bits": 8,
                "mode": "affine",
            }
            config["quantization"] = quant
        else:
            weights.update(
                {
                    prefix
                    + "mtp.layers.0.self_attn.q_proj.weight": mx.ones(
                        (8, 8), dtype=bf16
                    ),
                    prefix
                    + "mtp.layers.0.mlp.gate_proj.weight": mx.ones((16, 8), dtype=bf16),
                }
            )
    (donor / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(donor / "model.safetensors"), weights)
    index = {
        "metadata": {"total_size": sum(v.nbytes for v in weights.values())},
        "weight_map": {k: "model.safetensors" for k in weights},
    }
    (donor / "model.safetensors.index.json").write_text(json.dumps(index))
    (donor / "tokenizer.json").write_bytes(tokenizer)
    return donor


def test_graft_bf16_donor_writes_shard_index_config(tmp_path):
    out = _write_qwen_output(tmp_path)
    donor = _write_qwen_donor(tmp_path)

    combine_mtp_donor(out, donor)

    shard = out / GEMMA4_ASSISTANT_MTP_SHARD
    assert shard.exists()
    merged = mx.load(str(shard))
    assert merged, "no mtp tensors grafted"
    assert all(k.startswith("mtp.") for k in merged)
    assert "mtp.fc.weight" in merged

    donor_weights = mx.load(str(donor / "model.safetensors"))
    assert mx.array_equal(merged["mtp.fc.weight"], donor_weights["mtp.fc.weight"])
    assert merged["mtp.fc.weight"].dtype == mx.bfloat16

    index = json.loads((out / "model.safetensors.index.json").read_text())
    for key in merged:
        assert index["weight_map"][key] == GEMMA4_ASSISTANT_MTP_SHARD
    mtp_bytes = sum(v.nbytes for v in merged.values())
    assert index["metadata"]["total_size"] == 100 + mtp_bytes

    config = json.loads((out / "config.json").read_text())
    assert config["mtp_num_hidden_layers"] == 1
    # bf16 donor adds zero quantization entries.
    assert config["quantization"] == {"group_size": 64, "bits": 6, "mode": "affine"}
    assert config["quantization_config"] == config["quantization"]


def test_graft_quantized_donor_synthesizes_per_layer_entries(tmp_path):
    out = _write_qwen_output(tmp_path)
    donor = _write_qwen_donor(tmp_path, quantized=True)

    combine_mtp_donor(out, donor)

    config = json.loads((out / "config.json").read_text())
    for section in ("quantization", "quantization_config"):
        quant = config[section]
        # Recipient global untouched.
        assert quant["bits"] == 6
        # Donor per-layer override wins for the overridden module.
        assert quant["mtp.layers.0.self_attn.q_proj"] == {
            "group_size": 32,
            "bits": 8,
            "mode": "affine",
        }
        # Global-riding donor modules get the donor global, explicitly.
        assert quant["mtp.layers.0.mlp.gate_proj"] == {
            "group_size": 64,
            "bits": 4,
            "mode": "affine",
        }
        # fc ships bf16 without scales — no entry, stays float on load.
        assert "mtp.fc" not in quant

    merged = mx.load(str(out / GEMMA4_ASSISTANT_MTP_SHARD))
    donor_weights = mx.load(str(donor / "model.safetensors"))
    key = "mtp.layers.0.self_attn.q_proj.weight"
    assert mx.array_equal(merged[key], donor_weights[key])
    assert merged[key].dtype == mx.uint32
    assert mx.array_equal(
        merged["mtp.layers.0.self_attn.q_proj.scales"],
        donor_weights["mtp.layers.0.self_attn.q_proj.scales"],
    )


def test_graft_remaps_vlm_donor_prefix_into_text_recipient(tmp_path):
    out = _write_qwen_output(tmp_path)
    donor = _write_qwen_donor(tmp_path, vlm=True, quantized=True)

    combine_mtp_donor(out, donor)

    merged = mx.load(str(out / GEMMA4_ASSISTANT_MTP_SHARD))
    assert all(k.startswith("mtp.") for k in merged)
    config = json.loads((out / "config.json").read_text())
    assert config["mtp_num_hidden_layers"] == 1
    # Quant entries land under the recipient's bare naming.
    assert "mtp.layers.0.self_attn.q_proj" in config["quantization"]
    assert "language_model.mtp.layers.0.self_attn.q_proj" not in config["quantization"]


def test_graft_remaps_text_donor_prefix_into_vlm_recipient(tmp_path):
    out = _write_qwen_output(tmp_path, vlm=True, rope_nested=True)
    donor = _write_qwen_donor(tmp_path, quantized=True)

    combine_mtp_donor(out, donor)

    merged = mx.load(str(out / GEMMA4_ASSISTANT_MTP_SHARD))
    assert all(k.startswith("language_model.mtp.") for k in merged)
    config = json.loads((out / "config.json").read_text())
    assert config["text_config"]["mtp_num_hidden_layers"] == 1
    assert "language_model.mtp.layers.0.self_attn.q_proj" in config["quantization"]


def test_validate_rejects_tokenizer_mismatch(tmp_path):
    out = _write_qwen_output(tmp_path)
    donor = _write_qwen_donor(tmp_path, tokenizer=b'{"version": "other"}')
    with pytest.raises(ValueError, match="byte-identical"):
        validate_mtp_donor_pair(out, donor)


def test_validate_rejects_missing_tokenizer(tmp_path):
    out = _write_qwen_output(tmp_path)
    donor = _write_qwen_donor(tmp_path)
    (donor / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match="tokenizer.json missing"):
        validate_mtp_donor_pair(out, donor)


def test_validate_rejects_geometry_mismatch(tmp_path):
    out = _write_qwen_output(tmp_path)
    donor = _write_qwen_donor(tmp_path, hidden_size=12)
    with pytest.raises(ValueError, match="hidden_size"):
        validate_mtp_donor_pair(out, donor)


def test_validate_rejects_moe_donor_into_dense_recipient(tmp_path):
    out = _write_qwen_output(tmp_path)
    donor = _write_qwen_donor(tmp_path, num_experts=4, moe_intermediate_size=8)
    with pytest.raises(ValueError, match="num_experts"):
        validate_mtp_donor_pair(out, donor)


def test_validate_rejects_family_mismatch(tmp_path):
    out = _write_qwen_output(tmp_path)
    donor = _write_qwen_donor(tmp_path, model_type="qwen3_6")
    with pytest.raises(ValueError, match="does not match the recipient"):
        validate_mtp_donor_pair(out, donor)


def test_validate_rejects_non_qwen_recipient(tmp_path):
    out = _write_base_output(tmp_path)
    donor = _write_qwen_donor(tmp_path)
    with pytest.raises(ValueError, match="Qwen3.5/Qwen3.6 recipients"):
        validate_mtp_donor_pair(out, donor)


def test_validate_rejects_headless_donor(tmp_path):
    out = _write_qwen_output(tmp_path)
    donor = _write_qwen_donor(tmp_path, headless=True)
    with pytest.raises(ValueError, match="no MTP head"):
        validate_mtp_donor_pair(out, donor)


def test_validate_rejects_undeclared_donor(tmp_path):
    out = _write_qwen_output(tmp_path)
    donor = _write_qwen_donor(tmp_path, mtp_num_hidden_layers=0)
    with pytest.raises(ValueError, match="no MTP head"):
        validate_mtp_donor_pair(out, donor)


def test_combine_dispatch_routes_gemma4_assistant(tmp_path):
    out = _write_base_output(tmp_path)
    asst = _write_assistant(tmp_path)
    combine_mtp_into_output(out, asst)
    config = json.loads((out / "config.json").read_text())
    assert config["text_config"]["mtp_assistant_config"] == ASSISTANT_CONFIG


def test_combine_dispatch_routes_qwen_donor(tmp_path):
    out = _write_qwen_output(tmp_path)
    donor = _write_qwen_donor(tmp_path)
    combine_mtp_into_output(out, donor)
    config = json.loads((out / "config.json").read_text())
    assert config["mtp_num_hidden_layers"] == 1
    assert "mtp_assistant_config" not in config


# ── MTPLX side-car import ───────────────────────────────────────────────


def _write_qwen_mtplx_sidecar_model(
    tmp_path,
    *,
    vlm=False,
    bad_contract=False,
    with_contract=True,
    sidecar_rel=MTPLX_SIDECAR_SHARD,
):
    out = _write_qwen_output(tmp_path, vlm=vlm)
    config = json.loads((out / "config.json").read_text())
    config["mtplx_mtp_payload_audit"] = {"passed": True, "payload_tensor_count": 8}
    if with_contract:
        config["mtplx_mtp_contract"] = {
            "base_hidden_variant": "post_norm",
            "hidden_variant": "post_norm",
            "concat_order": "embedding_hidden",
            "mtp_position_mode": "local",
        }
    if sidecar_rel != MTPLX_SIDECAR_SHARD:
        config["mlx_lm_extra_tensors"] = {"mtp_file": sidecar_rel}
    (out / "config.json").write_text(json.dumps(config))

    runtime = {"arch_id": "qwen3-next-mtp", "mtp_depth_max": 3}
    if with_contract:
        runtime["mtp_contract"] = {
            "base_hidden_variant": "post_norm",
            "hidden_variant": "post_norm",
            "concat_order": "embedding_hidden",
            "mtp_position_mode": "local",
        }
    if bad_contract:
        runtime["mtp_contract"]["hidden_variant"] = "pre_norm"
    (out / MTPLX_RUNTIME_FILE).write_text(json.dumps(runtime))

    bf16 = mx.bfloat16
    sidecar_weights = {
        "mtp.fc.weight": mx.ones((8, 16), dtype=bf16),
        "mtp.norm.weight": mx.ones((8,), dtype=bf16),
        "mtp.pre_fc_norm_embedding.weight": mx.ones((8,), dtype=bf16),
        "mtp.pre_fc_norm_hidden.weight": mx.ones((8,), dtype=bf16),
        "mtp.layers.0.input_layernorm.weight": mx.ones((8,), dtype=bf16),
        "mtp.layers.0.self_attn.q_proj.weight": mx.ones((8, 8), dtype=bf16),
        "mtp.layers.0.mlp.gate_proj.weight": mx.ones((16, 8), dtype=bf16),
    }
    sidecar_path = out / sidecar_rel
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(sidecar_path), sidecar_weights, metadata={"format": "mlx"})
    return out


def test_import_mtplx_sidecar_remaps_vlm_prefix(tmp_path):
    out = _write_qwen_mtplx_sidecar_model(tmp_path, vlm=True)

    result = import_mtplx_sidecar(out)

    assert result["merge_mode"] == "remap"
    shard = out / GEMMA4_ASSISTANT_MTP_SHARD
    assert shard.exists()
    merged = mx.load(str(shard))
    assert all(k.startswith("language_model.mtp.") for k in merged)

    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert (
        index["weight_map"]["language_model.mtp.fc.weight"]
        == GEMMA4_ASSISTANT_MTP_SHARD
    )

    config = json.loads((out / "config.json").read_text())
    assert config["text_config"]["mtp_num_hidden_layers"] == 1

    # The consumed root side-car moves out of the *.safetensors glob so
    # loaders stop reading the bare-key duplicate on every load.
    assert not (out / MTPLX_SIDECAR_SHARD).exists()
    assert (out / (MTPLX_SIDECAR_SHARD + ".orig")).exists()


def test_import_mtplx_sidecar_renames_when_keys_align(tmp_path):
    out = _write_qwen_mtplx_sidecar_model(tmp_path, vlm=False)

    result = import_mtplx_sidecar(out)

    # Bare keys already match: the side-car is renamed onto the shard name
    # mlx_lm's model*.safetensors glob actually opens. No duplicate bytes.
    assert result["merge_mode"] == "rename"
    assert (out / GEMMA4_ASSISTANT_MTP_SHARD).exists()
    assert not (out / MTPLX_SIDECAR_SHARD).exists()

    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert index["weight_map"]["mtp.fc.weight"] == GEMMA4_ASSISTANT_MTP_SHARD


def test_import_mtplx_sidecar_resolves_mtp_file_override(tmp_path):
    # Official MTPLX exports keep the side-car at mtp/weights.safetensors,
    # declared via mlx_lm_extra_tensors.mtp_file, and pre-calibration
    # exports omit mtp_contract entirely (documented defaults apply).
    out = _write_qwen_mtplx_sidecar_model(
        tmp_path,
        vlm=True,
        with_contract=False,
        sidecar_rel="mtp/weights.safetensors",
    )

    result = import_mtplx_sidecar(out)

    assert result["merge_mode"] == "remap"
    assert (out / GEMMA4_ASSISTANT_MTP_SHARD).exists()
    # Sub-directory side-cars are invisible to the loader globs and stay put.
    assert (out / "mtp" / "weights.safetensors").exists()


def test_import_mtplx_sidecar_is_idempotent(tmp_path):
    out = _write_qwen_mtplx_sidecar_model(tmp_path, vlm=True)

    import_mtplx_sidecar(out)
    index_before = (out / "model.safetensors.index.json").read_text()

    result = import_mtplx_sidecar(out)

    assert result["merge_mode"] == "noop"
    assert (out / "model.safetensors.index.json").read_text() == index_before


def test_import_mtplx_sidecar_rejects_contract_mismatch(tmp_path):
    out = _write_qwen_mtplx_sidecar_model(tmp_path, vlm=True, bad_contract=True)

    before_index = (out / "model.safetensors.index.json").read_text()
    before_config = (out / "config.json").read_text()

    with pytest.raises(ValueError, match="Unsupported MTPLX contract"):
        import_mtplx_sidecar(out)

    assert (out / "model.safetensors.index.json").read_text() == before_index
    assert (out / "config.json").read_text() == before_config


def test_import_mtplx_sidecar_requires_runtime_file(tmp_path):
    out = _write_qwen_mtplx_sidecar_model(tmp_path, vlm=True)
    (out / MTPLX_RUNTIME_FILE).unlink()

    with pytest.raises(ValueError, match="Missing required runtime contract"):
        import_mtplx_sidecar(out)


def test_import_mtplx_sidecar_rejects_failed_audit(tmp_path):
    out = _write_qwen_mtplx_sidecar_model(tmp_path, vlm=True)
    config = json.loads((out / "config.json").read_text())
    config["mtplx_mtp_payload_audit"] = {"passed": False}
    (out / "config.json").write_text(json.dumps(config))

    with pytest.raises(ValueError, match="payload_audit"):
        import_mtplx_sidecar(out)

