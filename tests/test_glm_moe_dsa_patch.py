# SPDX-License-Identifier: Apache-2.0
"""Tests for the GLM-5.2 glm_moe_dsa monkey-patch."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from omlx.utils import model_loading
from omlx.utils.model_loading import maybe_apply_pre_load_patches


def _write_config(tmp_path, body: str) -> str:
    (tmp_path / "config.json").write_text(body)
    return str(tmp_path)


def _load_patched_glm_module():
    from omlx.patches.glm_moe_dsa import apply_glm_moe_dsa_patch

    apply_glm_moe_dsa_patch()
    from mlx_lm.models import glm_moe_dsa

    return glm_moe_dsa


def _small_glm_args(glm_moe_dsa):
    return glm_moe_dsa.ModelArgs(
        model_type="glm_moe_dsa",
        vocab_size=1024,
        hidden_size=128,
        index_head_dim=16,
        index_n_heads=4,
        index_topk=4,
        intermediate_size=256,
        moe_intermediate_size=256,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=4,
        routed_scaling_factor=2.5,
        kv_lora_rank=16,
        q_lora_rank=24,
        qk_rope_head_dim=16,
        v_head_dim=32,
        qk_nope_head_dim=16,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        norm_topk_prob=True,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=2,
        moe_layer_freq=1,
        first_k_dense_replace=1,
        max_position_embeddings=1024,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_theta": 10000.0},
        attention_bias=False,
        index_topk_pattern="FSFSFS",
    )


def _wait_for_pending_writes(manager):
    import time

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        with manager._pending_write_hashes_lock:
            if not manager._pending_write_hashes:
                return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for pending SSD cache writes")


def test_pre_load_dispatch_applies_glm_patch(tmp_path, monkeypatch):
    monkeypatch.setattr(model_loading, "_patch_mlx_lm_load_config", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "omlx.patches.mlx_lm_mtp",
        MagicMock(set_mtp_active=MagicMock()),
    )
    apply_mock = MagicMock(return_value=True)
    monkeypatch.setitem(
        sys.modules,
        "omlx.patches.glm_moe_dsa",
        MagicMock(apply_glm_moe_dsa_patch=apply_mock),
    )

    path = _write_config(tmp_path, '{"model_type": "glm_moe_dsa"}')
    maybe_apply_pre_load_patches(path)

    apply_mock.assert_called_once_with()


def test_glm_fused_gate_up_quant_spec_expanded_for_mxfp4_config():
    quant = {
        "group_size": 64,
        "bits": 8,
        "mode": "affine",
        "model.layers.1.mlp.switch_mlp.gate_proj": {
            "bits": 4,
            "group_size": 32,
            "mode": "mxfp4",
        },
        "model.layers.1.mlp.switch_mlp.up_proj": {
            "bits": 4,
            "group_size": 32,
            "mode": "mxfp4",
        },
        "model.layers.1.mlp.switch_mlp.down_proj": {
            "bits": 4,
            "group_size": 32,
            "mode": "mxfp4",
        },
    }
    cfg = {"model_type": "glm_moe_dsa", "quantization": dict(quant)}

    model_loading.expand_glm_moe_dsa_fused_quant_keys(cfg)

    assert cfg["quantization"]["model.layers.1.mlp.switch_mlp.gate_up_proj"] == {
        "bits": 4,
        "group_size": 32,
        "mode": "mxfp4",
    }
    assert "model.layers.1.mlp.switch_mlp.gate_proj" in cfg["quantization"]
    assert "model.layers.1.mlp.switch_mlp.up_proj" in cfg["quantization"]


def test_glm_mxfp4_fused_gate_up_quant_spec_avoids_bias_parameter():
    pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    from mlx.utils import tree_flatten

    glm_moe_dsa = _load_patched_glm_module()
    args = _small_glm_args(glm_moe_dsa)
    gate_path = "model.layers.1.mlp.switch_mlp.gate_up_proj"
    base_quant = {
        "group_size": 64,
        "bits": 8,
        "mode": "affine",
        "model.layers.1.mlp.switch_mlp.gate_proj": {
            "bits": 4,
            "group_size": 32,
            "mode": "mxfp4",
        },
        "model.layers.1.mlp.switch_mlp.up_proj": {
            "bits": 4,
            "group_size": 32,
            "mode": "mxfp4",
        },
        "model.layers.1.mlp.switch_mlp.down_proj": {
            "bits": 4,
            "group_size": 32,
            "mode": "mxfp4",
        },
    }
    weights = {f"{gate_path}.scales": object()}

    def gate_up_params(quantization):
        args.quantization = quantization
        model = glm_moe_dsa.Model(args)

        def class_predicate(path, module):
            if path in quantization:
                return quantization[path]
            if not hasattr(module, "to_quantized"):
                return False
            return f"{path}.scales" in weights

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
            class_predicate=class_predicate,
        )
        return {
            name
            for name, _ in tree_flatten(model.parameters())
            if name.startswith(gate_path)
        }

    before = gate_up_params(dict(base_quant))
    fixed_cfg = {"model_type": "glm_moe_dsa", "quantization": dict(base_quant)}
    model_loading.expand_glm_moe_dsa_fused_quant_keys(fixed_cfg)
    after = gate_up_params(fixed_cfg["quantization"])

    assert f"{gate_path}.biases" in before
    assert f"{gate_path}.weight" in after
    assert f"{gate_path}.scales" in after
    assert f"{gate_path}.biases" not in after


def test_glm_adaptive_prefill_config_defaults_and_gates(monkeypatch):
    from omlx.patches.glm_moe_dsa.generate_patch import (
        _glm_dsa_adaptive_prefill_config,
        _prefill_step_size_for_progress,
    )

    env_names = [
        "MLX_LM_GLM_DSA_ADAPTIVE_PREFILL_STEP",
        "MLX_LM_GLM_DSA_ADAPTIVE_PREFILL_STEP_SIZE",
        "MLX_LM_GLM_DSA_ADAPTIVE_PREFILL_AFTER",
        "MLX_LM_GLM_DSA_ADAPTIVE_PREFILL_MIN_REMAINING",
    ]
    for name in env_names:
        monkeypatch.delenv(name, raising=False)

    model = SimpleNamespace(model_type="glm_moe_dsa")
    cfg = _glm_dsa_adaptive_prefill_config(model, 2048)
    assert cfg is not None
    assert cfg.step_size == 8192
    assert cfg.after == 0
    assert cfg.min_remaining == 0
    assert _prefill_step_size_for_progress(2048, 0, 8192, cfg) == 8192

    assert _glm_dsa_adaptive_prefill_config(model, 1024) is None
    assert (
        _glm_dsa_adaptive_prefill_config(
            SimpleNamespace(model_type="deepseek_v32"), 2048
        )
        is None
    )

    monkeypatch.setenv("MLX_LM_GLM_DSA_ADAPTIVE_PREFILL_STEP", "0")
    assert _glm_dsa_adaptive_prefill_config(model, 2048) is None


def test_glm_adaptive_prefill_config_env_overrides(monkeypatch):
    from omlx.patches.glm_moe_dsa.generate_patch import (
        _glm_dsa_adaptive_prefill_config,
        _prefill_step_size_for_progress,
    )

    monkeypatch.setenv("MLX_LM_GLM_DSA_ADAPTIVE_PREFILL_STEP", "1")
    monkeypatch.setenv("MLX_LM_GLM_DSA_ADAPTIVE_PREFILL_STEP_SIZE", "4096")
    monkeypatch.setenv("MLX_LM_GLM_DSA_ADAPTIVE_PREFILL_AFTER", "8192")
    monkeypatch.setenv("MLX_LM_GLM_DSA_ADAPTIVE_PREFILL_MIN_REMAINING", "2048")

    cfg = _glm_dsa_adaptive_prefill_config(
        SimpleNamespace(args=SimpleNamespace(model_type="glm_moe_dsa")), 2048
    )
    assert cfg is not None
    assert cfg.step_size == 4096
    assert cfg.after == 8192
    assert cfg.min_remaining == 2048
    assert _prefill_step_size_for_progress(2048, 4096, 4096, cfg) == 2048
    assert _prefill_step_size_for_progress(2048, 8192, 1024, cfg) == 2048
    assert _prefill_step_size_for_progress(2048, 8192, 2048, cfg) == 4096


def test_glm_patch_keeps_vendored_helpers_private():
    glm_moe_dsa = _load_patched_glm_module()

    from omlx.patches.glm_moe_dsa import deepseek_v32 as vendored_deepseek_v32
    from mlx_lm.models import deepseek_v32 as upstream_deepseek_v32

    assert getattr(glm_moe_dsa, "_OMLX_GLM_DSA_OPTIMIZED", False)
    assert sys.modules["mlx_lm.models.glm_moe_dsa"] is glm_moe_dsa
    assert glm_moe_dsa.DeepseekV32Model is vendored_deepseek_v32.DeepseekV32Model
    assert upstream_deepseek_v32 is not vendored_deepseek_v32


def test_glm_patch_installs_native_indexer_schedule():
    glm_moe_dsa = _load_patched_glm_module()

    fields = glm_moe_dsa.ModelArgs.__dataclass_fields__
    assert "indexer_types" in fields
    assert hasattr(glm_moe_dsa, "GlmMoeDsaModel")

    args = _small_glm_args(glm_moe_dsa)
    assert args.indexer_types == [
        "full",
        "shared",
        "full",
        "shared",
        "full",
        "shared",
    ]

    model = glm_moe_dsa.Model(args)
    assert [layer.self_attn.indexer is not None for layer in model.model.layers] == [
        True,
        False,
        True,
        False,
        True,
        False,
    ]
    assert [len(c.caches) for c in model.make_cache()] == [2, 1, 2, 1, 2, 1]


def test_glm_indexer_rope_interleave_matches_upstream_contract(monkeypatch):
    glm_moe_dsa = _load_patched_glm_module()

    from omlx.patches.glm_moe_dsa import deepseek_v32 as vendored_deepseek_v32

    glm_fields = glm_moe_dsa.ModelArgs.__dataclass_fields__
    dsv32_fields = vendored_deepseek_v32.ModelArgs.__dataclass_fields__

    assert glm_fields["indexer_rope_interleave"].default is True
    assert dsv32_fields["indexer_rope_interleave"].default is False

    calls = []

    def fake_initialize_rope(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(vendored_deepseek_v32, "initialize_rope", fake_initialize_rope)

    args = _small_glm_args(glm_moe_dsa)
    assert args.indexer_rope_interleave is True
    vendored_deepseek_v32.Indexer(args)

    assert calls[-1]["traditional"] is True


def test_glm_direct_sparse_mla_uses_fork_default_threshold(monkeypatch):
    from omlx.patches.glm_moe_dsa import glm_moe_dsa_model

    monkeypatch.setattr(
        glm_moe_dsa_model.glm_fast,
        "has",
        lambda name: name == "glm_dsa_sparse_mla_attention",
    )

    assert glm_moe_dsa_model._native_sparse_mla_default_min_k() == "11264"


def test_glm_native_fused_kernels_match_reference(monkeypatch):
    mx = pytest.importorskip("mlx.core")

    try:
        from omlx.custom_kernels.glm_moe_dsa import fast
    except Exception as exc:  # pragma: no cover - depends on local native build
        pytest.skip(f"omlx.custom_kernels.glm_moe_dsa is unavailable: {exc}")

    if not fast.is_native_available():
        pytest.skip("GLM MoE DSA native extension is unavailable")

    mx.random.seed(7)

    tokens, dims = 8, 64
    for topk in (8, 6):
        x_sorted = mx.random.normal((tokens * topk, 1, dims), dtype=mx.float16)
        inv_order = mx.array(
            list(range(tokens * topk - 1, -1, -1)), dtype=mx.uint32
        )
        scores = mx.softmax(
            mx.random.normal((tokens, topk), dtype=mx.float32),
            axis=-1,
        )
        y_native = fast.glm_moe_weighted_sum(x_sorted, inv_order, scores)
        x_ref = mx.squeeze(x_sorted, -2)
        x_ref = mx.take(x_ref, inv_order, axis=0)
        x_ref = mx.reshape(x_ref, scores.shape + (dims,))
        y_ref = mx.sum(x_ref * mx.expand_dims(scores, -1), axis=-2).astype(
            mx.float16
        )
        mx.eval(y_native, y_ref)
        assert float(mx.max(mx.abs(y_native - y_ref)).item()) == 0.0

    batch, heads, length, latent, values = 1, 64, 1, 512, 256
    x = mx.random.normal((batch, heads, length, latent), dtype=mx.float16)
    w_float = mx.random.normal((heads, values, latent), dtype=mx.float16)
    weight, scales, biases = mx.quantize(
        w_float,
        group_size=64,
        bits=8,
        mode="affine",
    )
    y_native = fast.glm_dsa_q8_vup_flat(x, weight, scales, biases)
    y_ref = mx.quantized_matmul(
        x,
        weight,
        scales,
        biases,
        True,
        64,
        8,
        "affine",
    )
    y_ref = mx.transpose(y_ref, (0, 2, 1, 3))
    y_ref = mx.reshape(y_ref, (batch, length, heads * values))
    mx.eval(y_native, y_ref)
    assert float(mx.max(mx.abs(y_native - y_ref)).item()) <= 0.125

    from omlx.patches.glm_moe_dsa.sparse_mla import fused_indexer_scores

    def assert_padded_indexer_scores_match(L, K, offset_view=False):
        B, H, D = 1, 32, 128
        if offset_view:
            q_base = mx.random.normal((B, H, L + 2, D), dtype=mx.float16)
            k_base = mx.random.normal((B, 1, K + 2, D), dtype=mx.float16)
            w_base = mx.random.normal((B, L + 2, H), dtype=mx.float16)
            q = q_base[:, :, 1 : L + 1, :]
            k = k_base[:, :, 1 : K + 1, :]
            w = w_base[:, 1 : L + 1, :]
        else:
            q = mx.random.normal((B, H, L, D), dtype=mx.float16)
            k = mx.random.normal((B, 1, K, D), dtype=mx.float16)
            w = mx.random.normal((B, L, H), dtype=mx.float16)
        y_native = fused_indexer_scores(q, k, w, causal=True)
        head_scores = q @ k.swapaxes(-1, -2)
        y_ref = mx.maximum(head_scores, 0)
        y_ref = mx.sum(
            y_ref * w.swapaxes(-1, -2)[..., None],
            axis=1,
            keepdims=True,
        )
        q_pos = mx.arange(K - L, K, dtype=mx.uint32).reshape(1, 1, L, 1)
        k_pos = mx.arange(0, K, dtype=mx.uint32).reshape(1, 1, 1, K)
        y_ref = mx.where(
            k_pos <= q_pos,
            y_ref,
            mx.array(-float("inf"), dtype=y_ref.dtype),
        )
        mx.eval(y_native, y_ref)
        valid = mx.isfinite(y_ref)
        future_finite = mx.sum(
            mx.where(~valid, mx.isfinite(y_native), mx.array(False))
        )
        diff = mx.max(
            mx.where(
                valid,
                mx.abs(y_native.astype(mx.float32) - y_ref.astype(mx.float32)),
                mx.array(0.0),
            )
        )
        assert int(future_finite.item()) == 0
        assert float(diff.item()) <= 0.5

    assert_padded_indexer_scores_match(128, 4210)
    assert_padded_indexer_scores_match(100, 4200)
    assert_padded_indexer_scores_match(128, 4210, offset_view=True)

    assert not fast.has_symbol("glm_moe_swiglu_down")

    batch, heads, q_len, k_len, latent, pe = 1, 64, 2, 32, 512, 64
    scale = 0.05
    q_latent = mx.random.normal((batch, heads, q_len, latent), dtype=mx.float16)
    q_pe = mx.random.normal((batch, heads, q_len, pe), dtype=mx.float16)
    kv_latent = mx.random.normal((batch, 1, k_len, latent), dtype=mx.float16)
    k_pe = mx.random.normal((batch, 1, k_len, pe), dtype=mx.float16)
    topk_indices = mx.broadcast_to(
        mx.reshape(mx.arange(0, k_len, dtype=mx.uint32), (1, 1, 1, k_len)),
        (batch, 1, q_len, k_len),
    )
    y_native = fast.glm_dsa_sparse_mla_attention(
        q_latent,
        q_pe,
        kv_latent,
        k_pe,
        topk_indices,
        scale,
        causal=True,
    )
    scores = mx.sum(
        mx.expand_dims(q_latent, 3) * mx.expand_dims(kv_latent, 1),
        axis=-1,
    )
    scores = scores + mx.sum(
        mx.expand_dims(q_pe, 3) * mx.expand_dims(k_pe, 1),
        axis=-1,
    )
    scores = scores * scale
    q_pos = mx.reshape(
        mx.arange(k_len - q_len, k_len, dtype=mx.uint32),
        (1, 1, q_len, 1),
    )
    k_pos = mx.reshape(mx.arange(0, k_len, dtype=mx.uint32), (1, 1, 1, k_len))
    scores = mx.where(k_pos <= q_pos, scores, mx.array(-65504.0, scores.dtype))
    probs = mx.softmax(scores, axis=-1)
    y_ref = mx.sum(
        mx.expand_dims(probs, -1) * mx.expand_dims(kv_latent, 1),
        axis=3,
    )
    mx.eval(y_native, y_ref)
    assert float(mx.max(mx.abs(y_native - y_ref)).item()) <= 0.02

    batch, heads, q_len, k_len, latent, pe, topk = 1, 64, 64, 64, 512, 64, 16
    scale = 0.05
    q_latent = mx.random.normal((batch, heads, q_len, latent), dtype=mx.float16)
    q_pe = mx.random.normal((batch, heads, q_len, pe), dtype=mx.float16)
    kv_latent = mx.random.normal((batch, 1, k_len, latent), dtype=mx.float16)
    k_pe = mx.random.normal((batch, 1, k_len, pe), dtype=mx.float16)
    rows = []
    dense_rows = []
    for q_pos in range(q_len):
        start = max(0, q_pos - topk + 1)
        ids = list(range(start, q_pos + 1))
        rows.append(ids + ([0] * (topk - len(ids))))
        selected = set(ids)
        dense_rows.append([j in selected and j <= q_pos for j in range(k_len)])
    topk_indices = mx.array([[rows]], dtype=mx.uint32)
    y_native = fast.glm_dsa_sparse_mla_attention(
        q_latent,
        q_pe,
        kv_latent,
        k_pe,
        topk_indices,
        scale,
        causal=True,
        topk_valid_prefix=True,
        causal_prefix_indices=True,
    )
    scores = mx.sum(
        mx.expand_dims(q_latent, 3) * mx.expand_dims(kv_latent, 1),
        axis=-1,
    )
    scores = scores + mx.sum(
        mx.expand_dims(q_pe, 3) * mx.expand_dims(k_pe, 1),
        axis=-1,
    )
    scores = scores * scale
    dense_mask = mx.array([[dense_rows]], dtype=mx.bool_)
    scores = mx.where(dense_mask, scores, mx.array(-65504.0, scores.dtype))
    probs = mx.softmax(scores, axis=-1)
    y_ref = mx.sum(
        mx.expand_dims(probs, -1) * mx.expand_dims(kv_latent, 1),
        axis=3,
    )
    mx.eval(y_native, y_ref)
    subset_diff = mx.max(
        mx.abs(y_native.astype(mx.float32) - y_ref.astype(mx.float32))
    )
    assert float(subset_diff.item()) <= 0.02

    batch, heads, q_len, k_len, latent, pe, topk = 1, 64, 32, 64, 512, 64, 16
    prefix_rows = 16
    scale = 0.05
    q_latent = mx.random.normal((batch, heads, q_len, latent), dtype=mx.float16)
    q_pe = mx.random.normal((batch, heads, q_len, pe), dtype=mx.float16)
    kv_latent = mx.random.normal((batch, 1, k_len, latent), dtype=mx.float16)
    k_pe = mx.random.normal((batch, 1, k_len, pe), dtype=mx.float16)
    rows = []
    for q_pos in range(q_len):
        q_abs = k_len - q_len + q_pos
        if q_pos < prefix_rows:
            rows.append(list(range(topk)))
        else:
            rows.append(list(range(q_abs - topk + 1, q_abs + 1)))
    full_topk = mx.array([[rows]], dtype=mx.uint32)
    suffix_topk = full_topk[:, :, prefix_rows:, :]
    y_full = fast.glm_dsa_sparse_mla_attention(
        q_latent,
        q_pe,
        kv_latent,
        k_pe,
        full_topk,
        scale,
        causal=True,
        topk_valid_prefix=True,
        causal_prefix_indices=True,
    )
    y_compact = fast.glm_dsa_sparse_mla_attention(
        q_latent,
        q_pe,
        kv_latent,
        k_pe,
        suffix_topk,
        scale,
        causal=True,
        topk_valid_prefix=True,
        causal_prefix_indices=True,
        causal_prefix_rows=prefix_rows,
    )
    mx.eval(y_full, y_compact)
    compact_diff = mx.max(
        mx.abs(y_full.astype(mx.float32) - y_compact.astype(mx.float32))
    )
    assert float(compact_diff.item()) <= 5e-4

    if not fast.has_symbol("glm_dsa_exact_block_attention"):
        pytest.skip("GLM exact block-token attention native kernel is unavailable")

    from omlx.patches.glm_moe_dsa.sparse_mla import topk_indices_to_block_masks

    batch, heads, q_len, k_len, dims, topk = 1, 2, 32, 32, 256, 8
    scale = dims**-0.5
    q = mx.random.normal((batch, heads, q_len, dims), dtype=mx.float16)
    k = mx.random.normal((batch, heads, k_len, dims), dtype=mx.float16)
    v = mx.random.normal((batch, heads, k_len, dims), dtype=mx.float16)
    rows = []
    dense_rows = []
    for i in range(q_len):
        start = max(0, i - topk + 1)
        ids = list(range(start, i + 1))
        rows.append(([0] * (topk - len(ids))) + ids)
        selected = set(ids)
        dense_rows.append([j in selected and j <= i for j in range(k_len)])
    topk_indices = mx.array([[rows]], dtype=mx.uint32)
    block_masks = topk_indices_to_block_masks(
        topk_indices,
        L=q_len,
        K=k_len,
        q_block_size=16,
        k_block_size=8,
    )
    assert block_masks is not None
    block_mask, block_token_mask = block_masks
    y_native = fast.glm_dsa_exact_block_attention(
        q,
        k,
        v,
        block_mask,
        block_token_mask,
        scale,
        causal=True,
    )
    dense_mask = mx.array([[dense_rows]], dtype=mx.bool_)
    y_ref = mx.fast.scaled_dot_product_attention(
        q,
        k,
        v,
        scale=scale,
        mask=dense_mask,
    )
    mx.eval(y_native, y_ref)
    diff = mx.max(mx.abs(y_native.astype(mx.float32) - y_ref.astype(mx.float32)))
    assert float(diff.item()) <= 2e-3

    scores = mx.random.normal((1, 1, 2, 2048), dtype=mx.float16)
    topk_indices = fast.dsa_topk_indices(
        scores,
        2048,
        bucketed=False,
        causal_valid_prefix=True,
    )
    mx.eval(topk_indices)
    assert topk_indices.shape == (1, 1, 2, 2048)


def test_deepseek_affine_block_moe_kernels_match_gather_qmm():
    mx = pytest.importorskip("mlx.core")

    try:
        from omlx.custom_kernels.glm_moe_dsa import fast
    except Exception as exc:  # pragma: no cover - depends on local native build
        pytest.skip(f"omlx.custom_kernels.glm_moe_dsa is unavailable: {exc}")

    if not fast.is_native_available():
        pytest.skip("GLM MoE DSA native extension is unavailable")
    if not fast.has_symbol("deepseek_affine_gather_qmm_blocks"):
        pytest.skip("DeepSeek affine block-list kernels are unavailable")

    from omlx.patches.deepseek_v4.switch_layers import (
        _build_mxfp4_blocks,
        _mxfp4_block_config,
    )

    mx.random.seed(11)
    experts, output_dims, input_dims, routes = 8, 64, 128, 192
    indices = mx.array(
        sorted((i * 7) % experts for i in range(routes)),
        dtype=mx.int32,
    )
    block_bm, block_variant = _mxfp4_block_config(indices.size)
    block_meta, block_count = _build_mxfp4_blocks(indices, experts, block_bm)

    for dtype in (mx.bfloat16, mx.float16):
        x = mx.random.normal((routes, 1, input_dims), dtype=dtype)
        for bits in (2, 3):
            w0 = mx.random.normal(
                (experts, output_dims, input_dims),
                dtype=dtype,
            )
            w1 = mx.random.normal(
                (experts, output_dims, input_dims),
                dtype=dtype,
            )
            q0, s0, b0 = mx.quantize(
                w0,
                group_size=64,
                bits=bits,
                mode="affine",
            )
            q1, s1, b1 = mx.quantize(
                w1,
                group_size=64,
                bits=bits,
                mode="affine",
            )

            y_ref = mx.gather_qmm(
                x,
                q0,
                s0,
                b0,
                rhs_indices=indices,
                transpose=True,
                group_size=64,
                bits=bits,
                mode="affine",
                sorted_indices=True,
            )
            y_native = fast.deepseek_affine_gather_qmm_blocks(
                x,
                q0,
                s0,
                b0,
                block_meta,
                block_count,
                64,
                bits,
                block_variant,
            )
            y_pair = fast.deepseek_affine_gather_qmm_pair_concat_blocks(
                x,
                q0,
                s0,
                b0,
                q1,
                s1,
                b1,
                block_meta,
                block_count,
                64,
                bits,
                block_variant,
            )
            y1_ref = mx.gather_qmm(
                x,
                q1,
                s1,
                b1,
                rhs_indices=indices,
                transpose=True,
                group_size=64,
                bits=bits,
                mode="affine",
                sorted_indices=True,
            )

            y0_pair = y_pair[..., :output_dims]
            y1_pair = y_pair[..., output_dims:]
            mx.eval(y_ref, y_native, y0_pair, y1_ref, y1_pair)
            assert float(mx.max(mx.abs(y_ref - y_native)).item()) == 0.0
            assert float(mx.max(mx.abs(y_ref - y0_pair)).item()) == 0.0
            assert float(mx.max(mx.abs(y1_ref - y1_pair)).item()) == 0.0


def test_deepseek_switchglu_uses_affine_block_kernels(monkeypatch):
    mx = pytest.importorskip("mlx.core")

    try:
        from omlx.custom_kernels.glm_moe_dsa import fast
    except Exception as exc:  # pragma: no cover - depends on local native build
        pytest.skip(f"omlx.custom_kernels.glm_moe_dsa is unavailable: {exc}")

    if not fast.is_native_available():
        pytest.skip("GLM MoE DSA native extension is unavailable")
    if not fast.has_symbol("deepseek_affine_gather_qmm_pair_concat_blocks"):
        pytest.skip("DeepSeek affine block-list kernels are unavailable")

    from omlx.patches.deepseek_v4.switch_layers import SwitchGLU

    mx.random.seed(13)

    def quantized_affine(layer):
        layer = layer.to_quantized(
            group_size=64,
            bits=3,
            mode="affine",
        )
        layer.scales = layer.scales.astype(mx.bfloat16)
        layer.biases = layer.biases.astype(mx.bfloat16)
        return layer

    model = SwitchGLU(128, 64, 8)
    model.gate_proj = quantized_affine(model.gate_proj)
    model.up_proj = quantized_affine(model.up_proj)
    model.down_proj = quantized_affine(model.down_proj)

    calls = {"pair": 0, "single": 0}
    orig_pair = fast.deepseek_affine_gather_qmm_pair_concat_blocks
    orig_single = fast.deepseek_affine_gather_qmm_blocks

    def pair_spy(*args, **kwargs):
        calls["pair"] += 1
        return orig_pair(*args, **kwargs)

    def single_spy(*args, **kwargs):
        calls["single"] += 1
        return orig_single(*args, **kwargs)

    monkeypatch.setattr(fast, "deepseek_affine_gather_qmm_pair_concat_blocks", pair_spy)
    monkeypatch.setattr(fast, "deepseek_affine_gather_qmm_blocks", single_spy)

    x = mx.random.normal((1, 32, 128), dtype=mx.bfloat16)
    indices = mx.array(
        [[[(i + j) % 8 for j in range(2)] for i in range(32)]],
        dtype=mx.int32,
    )
    y = model(x, indices)
    mx.eval(y)

    assert y.shape == (1, 32, 2, 128)
    assert calls == {"pair": 1, "single": 1}


def test_deepseek_switchglu_uses_fp16_affine_blocks_for_bf16_inputs(monkeypatch):
    mx = pytest.importorskip("mlx.core")

    try:
        from omlx.custom_kernels.glm_moe_dsa import fast
    except Exception as exc:  # pragma: no cover - depends on local native build
        pytest.skip(f"omlx.custom_kernels.glm_moe_dsa is unavailable: {exc}")

    if not fast.is_native_available():
        pytest.skip("GLM MoE DSA native extension is unavailable")
    if not fast.has_symbol("deepseek_affine_gather_qmm_pair_concat_blocks"):
        pytest.skip("DeepSeek affine block-list kernels are unavailable")

    from omlx.patches.deepseek_v4.switch_layers import SwitchGLU

    mx.random.seed(19)

    def quantized_affine(layer):
        layer = layer.to_quantized(
            group_size=64,
            bits=3,
            mode="affine",
        )
        layer.scales = layer.scales.astype(mx.float16)
        layer.biases = layer.biases.astype(mx.float16)
        return layer

    model = SwitchGLU(128, 64, 8)
    model.gate_proj = quantized_affine(model.gate_proj)
    model.up_proj = quantized_affine(model.up_proj)
    model.down_proj = quantized_affine(model.down_proj)

    calls = {"pair": 0, "single": 0, "pair_dtype": None, "single_dtype": None}
    orig_pair = fast.deepseek_affine_gather_qmm_pair_concat_blocks
    orig_single = fast.deepseek_affine_gather_qmm_blocks

    def pair_spy(x, *args, **kwargs):
        calls["pair"] += 1
        calls["pair_dtype"] = x.dtype
        return orig_pair(x, *args, **kwargs)

    def single_spy(x, *args, **kwargs):
        calls["single"] += 1
        calls["single_dtype"] = x.dtype
        return orig_single(x, *args, **kwargs)

    monkeypatch.setattr(fast, "deepseek_affine_gather_qmm_pair_concat_blocks", pair_spy)
    monkeypatch.setattr(fast, "deepseek_affine_gather_qmm_blocks", single_spy)

    x = mx.random.normal((1, 32, 128), dtype=mx.bfloat16)
    indices = mx.array(
        [[[(i + j) % 8 for j in range(2)] for i in range(32)]],
        dtype=mx.int32,
    )
    y = model(x, indices)
    mx.eval(y)

    assert y.dtype == mx.bfloat16
    assert y.shape == (1, 32, 2, 128)
    assert calls == {
        "pair": 1,
        "single": 1,
        "pair_dtype": mx.float16,
        "single_dtype": mx.float16,
    }


def test_deepseek_switchglu_does_not_use_native_weighted_sum(monkeypatch):
    mx = pytest.importorskip("mlx.core")

    from omlx.custom_kernels.glm_moe_dsa import fast
    from omlx.patches.deepseek_v4.switch_layers import SwitchGLU

    orig_has_symbol = fast.has_symbol
    calls = {"weighted_sum": 0}

    def has_symbol(name):
        if name == "glm_moe_weighted_sum":
            return True
        return orig_has_symbol(name)

    def weighted_sum_spy(*args, **kwargs):
        calls["weighted_sum"] += 1
        raise AssertionError("DeepSeek V4 must use the reference scatter path")

    monkeypatch.setattr(fast, "has_symbol", has_symbol)
    monkeypatch.setattr(fast, "glm_moe_weighted_sum", weighted_sum_spy)

    mx.random.seed(17)
    model = SwitchGLU(16, 8, 8)
    x = mx.random.normal((1, 11, 16), dtype=mx.bfloat16)
    indices = mx.array(
        [[[(i + j) % 8 for j in range(6)] for i in range(11)]],
        dtype=mx.int32,
    )
    scores = mx.softmax(
        mx.random.normal(indices.shape, dtype=mx.float32),
        axis=-1,
    )

    y = model(x, indices, scores=scores)
    mx.eval(y)

    assert y.shape == (1, 11, 6, 16)
    assert calls["weighted_sum"] == 0


def test_glm_direct_sparse_mla_threshold_requires_native(monkeypatch):
    glm_moe_dsa = _load_patched_glm_module()

    monkeypatch.setattr(
        glm_moe_dsa,
        "glm_fast",
        SimpleNamespace(has=lambda name: False),
    )

    assert int(glm_moe_dsa._native_sparse_mla_default_min_k()) > 10**12

    monkeypatch.setattr(
        glm_moe_dsa,
        "glm_fast",
        SimpleNamespace(has=lambda name: name == "glm_dsa_sparse_mla_attention"),
    )
    assert glm_moe_dsa._native_sparse_mla_default_min_k() == "11264"


def test_glm_sparse_topk_mask_fallback_matches_pure_mlx():
    mx = pytest.importorskip("mlx.core")
    glm_moe_dsa = _load_patched_glm_module()

    topk_indices = mx.array(
        [[[[1, 3], [0, 2], [2, 4]]]],
        dtype=mx.uint32,
    )
    mask = glm_moe_dsa._apply_sparse_topk_mask(
        None,
        topk_indices,
        0,
        key_length=5,
        query_length=3,
    )
    expected = mx.array(
        [
            [
                [
                    [False, True, False, True, False],
                    [True, False, True, False, False],
                    [False, False, True, False, True],
                ]
            ]
        ],
        dtype=mx.bool_,
    )
    mx.eval(mask, expected)
    assert mx.all(mask == expected).item()

    compact_indices = mx.array([[[[4, 5], [3, 5]]]], dtype=mx.uint32)
    compact_mask = glm_moe_dsa._apply_sparse_topk_mask(
        None,
        compact_indices,
        2,
        key_length=6,
        query_length=4,
    )
    compact_expected = mx.array(
        [
            [
                [
                    [True, True, True, False, False, False],
                    [True, True, True, True, False, False],
                    [False, False, False, False, True, True],
                    [False, False, False, True, False, True],
                ]
            ]
        ],
        dtype=mx.bool_,
    )
    mx.eval(compact_mask, compact_expected)
    assert mx.all(compact_mask == compact_expected).item()


def test_glm_patch_forward_sparse_path_and_cache_state():
    mx = pytest.importorskip("mlx.core")
    glm_moe_dsa = _load_patched_glm_module()

    args = _small_glm_args(glm_moe_dsa)
    model = glm_moe_dsa.Model(args)
    cache = model.make_cache()

    prompt = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    logits = model(prompt, cache=cache)
    assert logits.shape == (1, 8, args.vocab_size)

    nxt = mx.argmax(logits[0, -1:, :], keepdims=True)
    logits = model(nxt, cache=cache)
    assert logits.shape == (1, 1, args.vocab_size)
    assert mx.all(mx.isfinite(logits)).item()

    mx.eval([c.state for c in cache])
    full_state = cache[0].state
    shared_state = cache[1].state
    assert len(full_state) == 2
    assert len(shared_state) == 1
    assert full_state[1][1].shape[-1] == 0


def test_glm_cachelist_hot_and_cold_round_trip(tmp_path):
    mx = pytest.importorskip("mlx.core")
    glm_moe_dsa = _load_patched_glm_module()

    from omlx.cache.paged_cache import PagedCacheManager
    from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
    from omlx.cache.prefix_cache import BlockAwarePrefixCache
    from omlx.scheduler import Scheduler

    args = _small_glm_args(glm_moe_dsa)
    model = glm_moe_dsa.Model(args)
    cache = model.make_cache()
    logits = model(mx.array([[1, 2, 3, 4, 5, 6, 7, 8]]), cache=cache)
    mx.eval(logits, [c.state for c in cache])

    scheduler = MagicMock(spec=Scheduler)
    scheduler.model_name = "glm-test"
    scheduler._normalize_rotating_snapshot_state = (
        Scheduler._normalize_rotating_snapshot_state.__get__(scheduler, Scheduler)
    )
    scheduler._extract_cache_states = Scheduler._extract_cache_states.__get__(
        scheduler, Scheduler
    )
    extracted, model_cache_config = scheduler._extract_cache_states(cache)
    assert model_cache_config is not None
    assert model_cache_config.get_type_names() == ["CacheList"] * args.num_hidden_layers

    prefix_cache = BlockAwarePrefixCache(
        model=model,
        paged_cache_manager=PagedCacheManager(
            block_size=4,
            max_blocks=16,
            model_name="glm-test",
            initial_blocks=16,
        ),
    )
    block_data = prefix_cache._extract_block_tensor_slice(
        extracted,
        0,
        4,
        model_cache_config=model_cache_config,
        is_last_block=False,
    )
    assert block_data is not None
    assert block_data[0][0] == "__cache_list__"
    assert len(block_data[0][1]) == 2
    assert len(block_data[1][1]) == 1
    assert block_data[0][1][1][1].shape[-1] == 0

    block_hash = b"glm_moe_dsa_cache"
    layer_types = model_cache_config.get_type_names()
    layer_meta = model_cache_config.get_meta_states(cache)
    cache_dir = tmp_path / "glm_cache"

    manager = PagedSSDCacheManager(
        cache_dir=cache_dir,
        max_size_bytes=64 * 1024**2,
        hot_cache_max_bytes=16 * 1024**2,
    )
    try:
        assert manager.save_block(
            block_hash,
            block_data,
            token_count=4,
            model_name="glm-test",
            layer_cache_types=layer_types,
            layer_meta_states=layer_meta,
        )
        assert manager._hot_cache_get(block_hash) is not None
        hot_loaded = manager.load_block(block_hash)
        assert hot_loaded is not None
        assert len(hot_loaded[0]) == 2
        assert len(hot_loaded[1]) == 1
        assert hot_loaded[0][1][1].shape[-1] == 0
    finally:
        manager.close()

    cold_manager = PagedSSDCacheManager(
        cache_dir=cache_dir,
        max_size_bytes=64 * 1024**2,
        hot_cache_max_bytes=16 * 1024**2,
    )
    try:
        _wait_for_pending_writes(cold_manager)
        cold_loaded = cold_manager.load_block(block_hash)
        assert cold_loaded is not None
        assert len(cold_loaded[0]) == 2
        assert len(cold_loaded[1]) == 1
        assert cold_loaded[0][1][1].shape[-1] == 0
        assert cold_manager._hot_cache_get(block_hash) is not None
    finally:
        cold_manager.close()


def test_glm_indexer_decode_rows_skip_fused_scores_kernel(monkeypatch):
    """MTP verify forwards (tiny multi-row decode) must not enter the
    prefill-shaped fused indexer scores kernel (issue #2160): it runs ~5x
    slower than the matmul + fused reduce fallback at tiny row counts and
    the gap grows with context length."""
    mx = pytest.importorskip("mlx.core")
    glm_moe_dsa = _load_patched_glm_module()

    from mlx_lm.models.cache import KVCache

    from omlx.patches.glm_moe_dsa import deepseek_v32 as dsv32

    assert dsv32._FUSED_SCORES_MIN_S == 16

    args = _small_glm_args(glm_moe_dsa)
    indexer = dsv32.Indexer(args)
    mx.eval(indexer.parameters())

    fused_calls = []
    real_fused = dsv32.fused_indexer_scores

    def counting_fused(*a, **kw):
        fused_calls.append(a[0].shape)
        return real_fused(*a, **kw)

    monkeypatch.setattr(dsv32, "fused_indexer_scores", counting_fused)

    def causal_mask(s, total):
        q_pos = mx.arange(total - s, total)[:, None]
        k_pos = mx.arange(total)[None, :]
        return k_pos <= q_pos

    cache = KVCache()
    hidden = args.hidden_size

    # Prefill-shaped call (s >= floor) still routes through the fused kernel.
    s0 = 16
    x0 = mx.random.normal((1, s0, hidden)).astype(mx.bfloat16)
    qr0 = mx.random.normal((1, s0, args.q_lora_rank)).astype(mx.bfloat16)
    out0 = indexer(x0, qr0, causal_mask(s0, s0), cache=cache)
    mx.eval(out0 if not isinstance(out0, tuple) else out0[0])
    assert len(fused_calls) == 1

    # Decode-verify-shaped call (1 < s < floor) must skip the fused kernel
    # and still produce causally valid top-k indices.
    s1 = 3
    x1 = mx.random.normal((1, s1, hidden)).astype(mx.bfloat16)
    qr1 = mx.random.normal((1, s1, args.q_lora_rank)).astype(mx.bfloat16)
    total = s0 + s1
    out1 = indexer(x1, qr1, causal_mask(s1, total), cache=cache)
    assert len(fused_calls) == 1

    idx = out1[0] if isinstance(out1, tuple) else out1
    assert idx is not None
    mx.eval(idx)
    assert idx.shape == (1, 1, s1, args.index_topk)
    for row in range(s1):
        row_pos = total - s1 + row
        assert max(idx[0, 0, row].tolist()) <= row_pos
