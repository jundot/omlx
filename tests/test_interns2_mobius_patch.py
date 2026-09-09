# SPDX-License-Identifier: Apache-2.0
"""Tests for the Intern-S2-Mobius mlx-lm pre-load patch (mlx-lm#1771 port) and
its native MTP companion patch.

Ordering note: the base-model tests (registration, cache layout, sanitize,
dispatch) must run before ``test_mtp_patch_exposes_contract_surface`` — that test
applies the ``mlx_lm_mtp`` patch, which replaces the shared module's classes
in-place (including ``sanitize``, which then *keeps* ``mtp.*``). pytest runs
tests in definition order within a file, so the MTP test is defined last.
"""

import importlib
import json
import sys

import mlx.core as mx


def _minimal_config(**overrides):
    config = {
        "model_type": "interns2_mobius",
        "architectures": ["InternS2MobiusForCausalLM"],
        "vocab_size": 1000,
        "hidden_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_conv_kernel_dim": 4,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "num_blocks": 2,
        "moe_intermediate_size": 64,
        "shared_expert_intermediate_size": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.5,
        "max_position_embeddings": 1000,
        "full_attention_interval": 4,
        "tie_word_embeddings": False,
    }
    config.update(overrides)
    return config


def _load_patch_module():
    from omlx.patches.interns2_mobius import apply_interns2_mobius_patch

    apply_interns2_mobius_patch()
    return importlib.import_module("mlx_lm.models.interns2_mobius")


def test_apply_registers_interns2_mobius_module():
    module = _load_patch_module()

    assert module.__package__ == "mlx_lm.models"
    assert sys.modules["mlx_lm.models.interns2_mobius"] is module

    import mlx_lm.models as models_pkg

    assert models_pkg.interns2_mobius is module


def test_apply_is_idempotent():
    from omlx.patches.interns2_mobius import apply_interns2_mobius_patch, is_applied

    first = apply_interns2_mobius_patch()
    second = apply_interns2_mobius_patch()

    assert is_applied() is True
    assert second is False
    assert first in (True, False)


def test_get_classes_resolves_interns2_mobius():
    _load_patch_module()

    from mlx_lm.utils import _get_classes

    model_cls, args_cls = _get_classes(_minimal_config())

    assert model_cls.__name__ == "Model"
    assert args_cls.__name__ == "ModelArgs"


def test_make_cache_types_match_hybrid_layout():
    interns2 = _load_patch_module()

    model = interns2.Model(interns2.ModelArgs.from_dict(_minimal_config()))
    cache = model.make_cache()

    # full_attention_interval=4: layers 0,1,2 are Gated Delta Net (ArraysCache),
    # layer 3 is full attention (KVCache). Both are string-registered types oMLX
    # recognizes — the concrete basis for "no cache-stack changes".
    assert [type(c).__name__ for c in cache] == [
        "ArraysCache",
        "ArraysCache",
        "ArraysCache",
        "KVCache",
    ]


def test_forward_and_batch_generator():
    interns2 = _load_patch_module()
    from mlx_lm.generate import BatchGenerator

    model = interns2.Model(interns2.ModelArgs.from_dict(_minimal_config()))
    cache = model.make_cache()

    prefill = model(mx.array([[1, 2, 3], [4, 5, 6]]), cache=cache)
    decode = model(mx.array([[7], [8]]), cache=cache)
    mx.eval(prefill, decode)

    assert prefill.shape == (2, 3, 1000)
    assert decode.shape == (2, 1, 1000)

    generator = BatchGenerator(
        model,
        max_tokens=2,
        prefill_batch_size=2,
        completion_batch_size=2,
        sampler=lambda logits: mx.argmax(logits, axis=-1),
    )
    uids = generator.insert([[1, 2, 3], [4, 5, 6]], max_tokens=[2, 2])
    finished = []
    for _ in range(8):
        _, generation_responses = generator.next()
        finished.extend(
            response
            for response in generation_responses
            if response.finish_reason is not None
        )
        if len(finished) == 2:
            break

    assert uids == [0, 1]
    assert {response.uid for response in finished} == {0, 1}
    assert all(response.finish_reason == "length" for response in finished)


def test_sanitize_drops_visual_and_mtp_and_stacks_expert_banks():
    interns2 = _load_patch_module()
    config = _minimal_config()
    model = interns2.Model(interns2.ModelArgs.from_dict(config))

    n_experts = config["num_experts"]  # 4 routed
    hidden = config["hidden_size"]  # 128
    moe = config["moe_intermediate_size"]  # 64

    weights = {
        "visual.patch_embed.weight": mx.ones((1,)),
        "mtp.layers.0.fc.weight": mx.ones((1,)),
    }
    # Pre-fused routed experts per bank (gate_up already stacked).
    for bank in range(config["num_blocks"]):
        weights[f"model.meta_mlp.{bank}.experts.gate_up_proj"] = mx.ones(
            (n_experts, 2 * moe, hidden)
        )
        weights[f"model.meta_mlp.{bank}.experts.down_proj"] = mx.ones(
            (n_experts, hidden, moe)
        )
    # Each layer contributes one shared expert (split gate/up) into its bank.
    for i in range(config["num_hidden_layers"]):
        base = f"model.layers.{i}.mlp.shared_expert"
        weights[f"{base}.gate_proj.weight"] = mx.ones((moe, hidden))
        weights[f"{base}.up_proj.weight"] = mx.ones((moe, hidden))
        weights[f"{base}.down_proj.weight"] = mx.ones((hidden, moe))

    sanitized = model.sanitize(weights)

    # Vision and MTP weights dropped on the base (unpatched) path. The MTP
    # companion patch overrides sanitize to KEEP mtp.* — that path is covered by
    # test_mtp_patch_exposes_contract_surface, which runs after this test.
    assert not any(key.startswith(("visual.", "mtp.")) for key in sanitized)
    # bank 0 serves layers 0 and 2 -> 4 routed + 2 shared = 6 experts stacked.
    gate_up = sanitized["model.meta_mlp.0.switch_mlp.gate_up_proj.weight"]
    down = sanitized["model.meta_mlp.0.switch_mlp.down_proj.weight"]
    assert gate_up.shape == (n_experts + 2, 2 * moe, hidden)
    assert down.shape == (n_experts + 2, hidden, moe)


def test_pre_load_dispatch_calls_interns2_mobius_patch(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "omlx.patches.interns2_mobius.apply_interns2_mobius_patch",
        lambda: calls.append(True) or True,
    )
    (tmp_path / "config.json").write_text(json.dumps(_minimal_config()))

    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    maybe_apply_pre_load_patches(str(tmp_path))

    assert calls == [True]


def test_interns2_mobius_routed_to_text_engine(tmp_path):
    # The 4-bit checkpoint ships vision token ids but no vision_config, and the
    # model_type is registered text-only, so discovery must pick the LLM engine.
    from omlx.model_discovery import (
        MLX_LM_TEXT_ONLY_MODEL_TYPES,
        detect_model_type,
    )

    assert "interns2_mobius" in MLX_LM_TEXT_ONLY_MODEL_TYPES
    (tmp_path / "config.json").write_text(
        json.dumps(_minimal_config(vision_start_token_id=1, image_token_id=2))
    )
    assert detect_model_type(tmp_path) == "llm"


def test_mtp_patch_exposes_contract_surface():
    # Runs last: applying the MTP patch mutates the shared module classes.
    _load_patch_module()
    from omlx.patches.mlx_lm_mtp import interns2_mobius_model as mtp_patch

    applied = mtp_patch.apply()
    assert applied is True

    interns2 = importlib.import_module("mlx_lm.models.interns2_mobius")

    # Driver contract methods land on the Model class.
    for method in ("mtp_forward", "make_mtp_cache", "mtp_partial_rollback"):
        assert hasattr(interns2.Model, method)
    assert getattr(interns2.Model, "_omlx_mtp_init_wrapped", False) is True
    # Patched __call__ carries the marker the generic batch_generator dispatches on.
    assert (
        getattr(interns2.Model.__dict__["__call__"], "_omlx_mtp_call_marker", False)
        is True
    )
    # Head classes exist on the vendored module.
    for cls in (
        "InternS2MobiusMTP",
        "InternS2MobiusMTPLayer",
        "InternS2MobiusMTPMoeBlock",
    ):
        assert hasattr(interns2, cls)
