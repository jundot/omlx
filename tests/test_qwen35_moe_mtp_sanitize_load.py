# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-MoE VLMs whose MTP head ships per-expert weights (issue #3688).

mlx-vlm runs ``Model.sanitize`` only for checkpoints that do not declare
``format=mlx``. An oQ conversion declares it, so a merged MTP head that
still keys its routed experts as ``experts.<i>.*`` never gets stacked into
``switch_mlp.*`` and strict ``load_weights`` rejects it. The engine then
falls back to a text-only LLM and images stop reaching the model.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from safetensors.numpy import save_file  # noqa: E402

from omlx.engine import vlm as vlm_module  # noqa: E402
from omlx.patches import mlx_lm_mtp as lm_mtp  # noqa: E402
from omlx.patches.mlx_vlm_mtp import set_mtp_attach_enabled  # noqa: E402
from omlx.utils.model_loading import maybe_apply_pre_load_patches  # noqa: E402

NUM_EXPERTS = 4

TINY_CONFIG = {
    "architectures": ["Qwen3_5_MoeForConditionalGeneration"],
    "model_type": "qwen3_5_moe",
    "tie_word_embeddings": False,
    "image_token_id": 300,
    "video_token_id": 301,
    "vision_start_token_id": 302,
    "vision_end_token_id": 303,
    "vocab_size": 320,
    "text_config": {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "num_experts": NUM_EXPERTS,
        "num_experts_per_tok": 2,
        "shared_expert_intermediate_size": 32,
        "moe_intermediate_size": 32,
        "rms_norm_eps": 1e-6,
        "vocab_size": 320,
        "max_position_embeddings": 4096,
        "full_attention_interval": 2,
        "layer_types": ["linear_attention", "full_attention"],
        "mtp_num_hidden_layers": 1,
        "eos_token_id": 2,
    },
    "vision_config": {
        "model_type": "qwen3_5_moe_vision",
        "depth": 2,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_heads": 2,
        "in_channels": 3,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "out_hidden_size": 64,
        "num_position_embeddings": 64,
        "deepstack_visual_indexes": [],
    },
}


@pytest.fixture(autouse=True)
def _reset_mtp_flags():
    """Keep the process-wide MTP switches at their documented defaults."""
    set_mtp_attach_enabled(True)
    lm_mtp.set_mtp_active(False)
    yield
    lm_mtp.set_mtp_active(False)
    set_mtp_attach_enabled(True)


def _write_shard(model_dir, weights, *, mlx_format=True):
    save_file(
        {
            k: np.zeros((1,), dtype=np.float16) if v is None else v
            for k, v in weights.items()
        },
        str(model_dir / "model.safetensors"),
        metadata={"format": "mlx"} if mlx_format else None,
    )


def _write_checkpoint(
    model_dir, *, model_type="qwen3_5_moe", per_expert=True, mlx_format=True
):
    """Minimal on-disk checkpoint: config plus one shard of named tensors."""
    config = {"model_type": model_type, "text_config": {"mtp_num_hidden_layers": 1}}
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    keys = ["language_model.model.embed_tokens.weight"]
    if per_expert:
        keys += [
            f"language_model.mtp.layers.0.mlp.experts.{e}.gate_proj.weight"
            for e in range(2)
        ]
    else:
        keys.append("language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.weight")
    _write_shard(model_dir, dict.fromkeys(keys), mlx_format=mlx_format)


def _format_of(model_dir):
    import safetensors

    with safetensors.safe_open(
        str(model_dir / "model.safetensors"), framework="np"
    ) as f:
        return (f.metadata() or {}).get("format")


class TestSanitizeForcingGate:
    """``format=mlx`` may only be hidden for checkpoints that need sanitize."""

    def test_hidden_for_a_per_expert_mtp_head(self, tmp_path):
        _write_checkpoint(tmp_path)
        assert _format_of(tmp_path) == "mlx"

        with vlm_module._force_qwen35_moe_mtp_sanitize_on_load(tmp_path):
            assert _format_of(tmp_path) is None

        assert _format_of(tmp_path) == "mlx"

    def test_untouched_when_the_mtp_head_is_already_stacked(self, tmp_path):
        # The oQ builds that work today write the head in switch_mlp form;
        # they must keep skipping sanitize exactly as before.
        _write_checkpoint(tmp_path, per_expert=False)

        with vlm_module._force_qwen35_moe_mtp_sanitize_on_load(tmp_path):
            assert _format_of(tmp_path) == "mlx"

    def test_untouched_for_another_model_type(self, tmp_path):
        _write_checkpoint(tmp_path, model_type="gemma4")

        with vlm_module._force_qwen35_moe_mtp_sanitize_on_load(tmp_path):
            assert _format_of(tmp_path) == "mlx"

    def test_untouched_when_the_checkpoint_is_not_mlx_format(self, tmp_path):
        # A raw HF export already runs sanitize; nothing to force.
        _write_checkpoint(tmp_path, mlx_format=False)

        with vlm_module._force_qwen35_moe_mtp_sanitize_on_load(tmp_path):
            assert _format_of(tmp_path) is None

    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("language_model.mtp.layers.0.mlp.experts.3.up_proj.weight", True),
            ("mtp.layers.1.mlp.experts.0.down_proj.scales", True),
            ("language_model.mtp.layers.0.mlp.switch_mlp.up_proj.weight", False),
            ("language_model.model.layers.0.mlp.experts.3.up_proj.weight", False),
            ("language_model.mtp.fc.weight", False),
        ],
    )
    def test_key_classification(self, key, expected):
        assert vlm_module._is_unsanitized_mtp_expert_key(key) is expected


def _build_tiny_checkpoint(model_dir):
    """Write a tiny MLX-format VLM whose MTP experts are stored per expert.

    The weights are taken from the model's own parameter tree, so the only
    thing that can make the load fail is the expert layout.
    """
    (model_dir / "config.json").write_text(json.dumps(TINY_CONFIG), encoding="utf-8")
    # Attachment of MTPModule is gated on the checkpoint declaring mtp.*
    # weights; the real checkpoint's index declares them, so do the same.
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {"language_model.mtp.fc.weight": "model.safetensors"},
            }
        ),
        encoding="utf-8",
    )
    maybe_apply_pre_load_patches(
        str(model_dir),
        model_settings=SimpleNamespace(mtp_enabled=False),
        for_vlm=True,
    )

    from mlx_vlm.utils import get_model_and_args, update_module_configs

    raw = json.loads((model_dir / "config.json").read_text())
    model_class, _ = get_model_and_args(config=dict(raw))
    for section in ("text_config", "vision_config", "audio_config"):
        raw.setdefault(section, {})
    model_config = model_class.ModelConfig.from_dict(raw)
    model_config = update_module_configs(
        model_config,
        model_class,
        raw,
        ["text", "vision", "perceiver", "projector", "audio"],
    )
    model = model_class.Model(model_config)

    params = dict(tree_flatten(model.parameters()))
    assert any(".mtp." in key for key in params), "MTPModule was not attached"
    weights = {
        key: np.asarray(mx.zeros(value.shape), dtype=np.float16)
        for key, value in params.items()
    }

    unstacked = 0
    for key in [k for k in list(weights) if ".mtp." in k and ".switch_mlp." in k]:
        prefix, _, rest = key.partition(".switch_mlp.")
        projection = rest.split(".")[0]
        stacked = weights.pop(key)
        for expert in range(NUM_EXPERTS):
            weights[f"{prefix}.experts.{expert}.{projection}.weight"] = stacked[expert]
        unstacked += 1
    assert unstacked, "fixture did not produce a per-expert MTP head"

    save_file(weights, str(model_dir / "model.safetensors"), metadata={"format": "mlx"})


def test_per_expert_mtp_head_loads_on_an_mlx_checkpoint(tmp_path):
    """The whole point: the VLM has to load, so the engine keeps its vision.

    Without the load hook mlx-vlm skips sanitize, the per-expert MTP tensors
    have nothing to bind to, and ``VLMBatchedEngine.start()`` raises — which
    is what makes ``EnginePool`` serve the model as a text-only LLM (#3688).
    """
    from mlx_vlm.utils import load_model

    _build_tiny_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="parameters not in model"):
        load_model(tmp_path, lazy=True)

    with vlm_module._force_qwen35_moe_mtp_sanitize_on_load(tmp_path):
        model = load_model(tmp_path, lazy=True)

    assert hasattr(model.language_model, "mtp")
