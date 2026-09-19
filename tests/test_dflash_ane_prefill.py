# SPDX-License-Identifier: Apache-2.0
"""Qwen prefill projection dispatch composed with real DFlash rollback hooks."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest


@pytest.fixture
def qwen_dflash_prefill(monkeypatch):
    """Use real GDN/cache math without compiling private ANE programs."""
    dflash = pytest.importorskip("dflash_mlx.engine.target_qwen_gdn")
    from mlx_lm.models import qwen3_5 as qwen

    from omlx.engine.dflash import _prepare_qwen_prefill_for_dflash
    from omlx.patches import dflash_lifecycle as lifecycle
    from omlx.patches import qwen35_q4_mlp as projections

    class IsolatedGatedDeltaNet(qwen.GatedDeltaNet):
        _dflash_speculative_call_installed = False

    # Production classes are process-global; exercise the real installer on a
    # disposable subclass so these tests cannot leak hooks into other models.
    monkeypatch.setattr(qwen, "GatedDeltaNet", IsolatedGatedDeltaNet)
    monkeypatch.setattr(qwen, "Attention", None)
    monkeypatch.setattr(lifecycle, "_DFLASH_BACKUP", {})
    monkeypatch.setattr(projections, "_LM_LINEAR_PATCHED", False)
    monkeypatch.setattr(projections, "_LM_GDN_PREFILL_BACKEND", None)
    monkeypatch.setattr(projections, "_has_native_qmm", lambda: True)
    monkeypatch.setenv("OMLX_QWEN35_Q4_LM_LINEAR", "1")
    monkeypatch.setenv("OMLX_QWEN35_Q4_LINEAR_MIN_TOKENS", "4")
    # Record the existing module attributes so monkeypatch restores them even
    # when the real lifecycle installer replaces them below.
    installer = "_install_speculative_linear_cache_hook"
    monkeypatch.setattr(dflash, installer, getattr(dflash, installer))
    marker = "_omlx_wrapped_" + installer
    monkeypatch.setattr(dflash, marker, getattr(dflash, marker, False), raising=False)

    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        model = IsolatedGatedDeltaNet(
            qwen.TextModelArgs(
                hidden_size=8,
                num_attention_heads=2,
                linear_num_value_heads=2,
                linear_num_key_heads=1,
                linear_key_head_dim=4,
                linear_value_head_dim=4,
                linear_conv_kernel_dim=4,
            )
        )
        # DFlash's Python recurrence still records the rollback tape. No Metal
        # or private ANE kernel compilation is needed for this dispatch test.
        model.train(True)
        original_call = IsolatedGatedDeltaNet.__call__
        routed = []

        def projection_backend(module, inputs, target_verify):
            routed.append((inputs.shape[1], target_verify))
            return (
                module.in_proj_qkv(inputs),
                module.in_proj_z(inputs),
                module.in_proj_b(inputs),
                module.in_proj_a(inputs),
            )

        assert _prepare_qwen_prefill_for_dflash()

        def activate_backend():
            projections.register_qwen35_lm_gdn_prefill_backend(projection_backend)

        assert lifecycle._wrap_installer(
            dflash, installer, "_dflash_speculative_call_installed"
        )
        yield model, original_call, routed, dflash, lifecycle, activate_backend
    finally:
        lifecycle.restore_dflash_class_patches()
        mx.set_default_device(previous_device)


@pytest.mark.parametrize("delayed_backend", [False, True])
def test_prefill_dispatch_preserves_dflash_verification_and_reload(
    qwen_dflash_prefill, delayed_backend
):
    from dflash_mlx.recurrent_rollback_cache import RecurrentRollbackCache

    from omlx.engine.dflash import _prepare_qwen_prefill_for_dflash

    model, original_call, routed, dflash, lifecycle, activate_backend = (
        qwen_dflash_prefill
    )
    inputs = mx.arange(64, dtype=mx.float32).reshape(1, 8, 8) / 64
    expected_cache = RecurrentRollbackCache(2)
    expected = original_call(model, inputs, cache=expected_cache)
    mx.eval(expected, expected_cache.state)

    for load in range(2):
        assert _prepare_qwen_prefill_for_dflash()
        if not delayed_backend:
            activate_backend()
        dflash._install_speculative_linear_cache_hook(model)
        installed_call = type(model).__call__
        assert _prepare_qwen_prefill_for_dflash()
        assert type(model).__call__ is installed_call

        if delayed_backend and load == 0:
            # The first resident target has ANE off. A later target may enable
            # its backend, but preparing it must preserve the existing guard.
            plain = model(inputs, cache=RecurrentRollbackCache(2))
            mx.eval(plain)
            assert bool(mx.allclose(plain, expected, atol=1e-6))
            assert routed == []
            activate_backend()

        cache = RecurrentRollbackCache(2)
        actual = model(inputs, cache=cache)
        mx.eval(actual, cache.state)

        assert routed == [(8, False)] * (load + 1)
        assert bool(mx.allclose(actual, expected, atol=1e-6))
        for actual_state, expected_state in zip(cache.state, expected_cache.state):
            assert bool(mx.allclose(actual_state, expected_state, atol=1e-6))

        # At the dispatch threshold, shape alone cannot keep verification out
        # of the projection backend. DFlash must retain its armed-cache path.
        expected_verify_cache = RecurrentRollbackCache(2)
        expected_verify_cache.state = list(expected_cache.state)
        expected_verified = original_call(
            model, inputs[:, :4, :], cache=expected_verify_cache
        )
        cache.arm_rollback()
        verified = model(inputs[:, :4, :], cache=cache)
        mx.eval(verified, cache.state, expected_verified, expected_verify_cache.state)
        assert routed == [(8, False)] * (load + 1)
        assert verified.shape == (1, 4, 8)
        assert bool(mx.allclose(verified, expected_verified, atol=1e-6))
        for actual_state, expected_state in zip(
            cache.state, expected_verify_cache.state
        ):
            assert bool(mx.allclose(actual_state, expected_state, atol=1e-6))
        assert cache._tape is not None
        assert cache._tape.shape[1] == 4
        assert cache._tape_qkv.shape[1] == 4
        assert cache._snapshot is not None

        # A one-token decode also leaves prefill projection dispatch untouched.
        cache.clear_transients()
        decoded = model(inputs[:, :1, :], cache=cache)
        mx.eval(decoded)
        assert routed == [(8, False)] * (load + 1)

        lifecycle.restore_dflash_class_patches()
        assert type(model) not in lifecycle.get_backup_classes()
        assert "_dflash_speculative_call_installed" not in type(model).__dict__


def test_preparation_rejects_guard_without_projection_base_without_mutation(
    qwen_dflash_prefill, monkeypatch
):
    from omlx.engine.dflash import _prepare_qwen_prefill_for_dflash

    model, original_call, _, dflash, lifecycle, _ = qwen_dflash_prefill
    # Simulate a DFlash hook installed outside the oMLX loading path, before
    # the prefill projection wrapper was available.
    monkeypatch.setattr(type(model), "__call__", original_call)
    dflash._install_speculative_linear_cache_hook(model)
    installed_call = type(model).__call__
    projection_wrapper = type(model)._omlx_q4_lm_gdn_wrapper
    backup = lifecycle.get_dflash_guard_base(type(model))
    assert backup is original_call

    assert _prepare_qwen_prefill_for_dflash() is False
    assert type(model).__call__ is installed_call
    assert type(model)._omlx_q4_lm_gdn_wrapper is projection_wrapper
    assert lifecycle.get_dflash_guard_base(type(model)) is backup


@pytest.fixture
def ane_projection_routes(monkeypatch):
    from omlx.patches import qwen35_ane_prefill as ane

    # A deliberately low padding threshold would otherwise route even a
    # short verification block, so suppression cannot rely on block length.
    config = ane._AnePrefillConfig(
        sequence_length=1024,
        fraction=0.5,
        variant=8,
        tail_padding_min_tokens=1,
    )
    mlp = SimpleNamespace(_omlx_ane_prefill_config=config)
    gdn = SimpleNamespace(_omlx_ane_gdn_config=config)
    calls = []

    def mlp_exact(module, inputs, target_verify=False):
        calls.append("mlp")
        return inputs

    def gdn_exact(module, inputs, target_verify=False):
        calls.append("gdn")
        return (inputs,) * 4

    monkeypatch.setattr(ane, "_backend_exact", mlp_exact)
    monkeypatch.setattr(ane, "_gdn_backend_exact", gdn_exact)

    def forward(*, inputs):
        return ane._backend(mlp, inputs), ane._gdn_backend(gdn, inputs)

    return forward, calls


@pytest.mark.parametrize("method", ["verify_block", "verify_tree_block"])
@pytest.mark.parametrize("tokens", [16, 1024])
def test_target_ops_bypass_ane_for_verify_full_tiles_and_padded_tails(
    method, tokens, ane_projection_routes
):
    from omlx.engine.dflash import _AnePrefillTargetOps

    forward, calls = ane_projection_routes
    target = SimpleNamespace(
        forward_with_hidden_capture=forward,
        verify_block=forward,
        verify_tree_block=forward,
    )
    ops = _AnePrefillTargetOps(target)
    inputs = mx.zeros((1, tokens, 8), dtype=mx.float16)

    prefill_mlp, prefill_gdn = ops.forward_with_hidden_capture(inputs=inputs)
    assert prefill_mlp.shape == inputs.shape
    assert prefill_gdn[0].shape == inputs.shape
    assert calls == ["mlp", "gdn"]

    assert getattr(ops, method)(inputs=inputs) == (None, None)
    assert calls == ["mlp", "gdn"]

    ops.forward_with_hidden_capture(inputs=inputs)
    assert calls == ["mlp", "gdn", "mlp", "gdn"]


@pytest.mark.parametrize("method", ["verify_block", "verify_tree_block"])
def test_target_ops_restore_ane_dispatch_after_verify_failure(
    method, ane_projection_routes
):
    from omlx.engine.dflash import _AnePrefillTargetOps

    forward, calls = ane_projection_routes

    def failing_verify(*, inputs):
        assert forward(inputs=inputs) == (None, None)
        raise RuntimeError("verification failed")

    target = SimpleNamespace(**{method: failing_verify})
    ops = _AnePrefillTargetOps(target)
    inputs = mx.zeros((1, 1024, 8), dtype=mx.float16)

    with pytest.raises(RuntimeError, match="verification failed"):
        getattr(ops, method)(inputs=inputs)
    assert calls == []

    forward(inputs=inputs)
    assert calls == ["mlp", "gdn"]


@pytest.mark.parametrize(
    "tile, expected_step", [(0, 2048), (1024, 2048), (2048, 2048), (4096, 4096)]
)
def test_ane_tiles_preserve_the_default_prefill_chunk_floor(tile, expected_step):
    from omlx.engine.dflash import DFlashEngine

    engine = DFlashEngine(model_name="test-target", draft_model_path="test-draft")
    engine._ane_prefill_sequence_length = tile
    assert engine._build_runtime_context().runtime.prefill_step_size == expected_step


def test_unset_fraction_uses_qwen_default(monkeypatch):
    from omlx.custom_kernels.qwen35_prefill import fast
    from omlx.engine.dflash import _enable_qwen35_ane_prefill_for_dflash
    from omlx.model_settings import ModelSettings
    from omlx.patches import qwen35_q4_mlp

    monkeypatch.setattr(
        qwen35_q4_mlp, "apply_qwen35_q4_lm_prefill_linear_patch", lambda: True
    )
    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: False)
    assert (
        _enable_qwen35_ane_prefill_for_dflash(
            SimpleNamespace(), ModelSettings(qwen35_ane_prefill_enabled=True)
        )
        == 0
    )


def test_partial_ane_setup_failure_releases_attached_state(monkeypatch):
    from omlx.engine.dflash import _enable_qwen35_ane_prefill_for_dflash
    from omlx.model_settings import ModelSettings
    from omlx.patches import qwen35_ane_prefill as ane

    layer = SimpleNamespace()
    target = SimpleNamespace(modules=lambda: [layer])
    failure = RuntimeError("compilation failed after the first layer")

    def partial_enable(model, **kwargs):
        layer._omlx_ane_prefill_state = object()
        layer._omlx_ane_prefill_cache = {"partial": layer._omlx_ane_prefill_state}
        model._omlx_ane_mlp_prefill_count = 1
        model._omlx_ane_resident_program_count = 1
        raise failure

    monkeypatch.setattr(ane, "enable_qwen35_ane_prefill", partial_enable)
    with pytest.raises(RuntimeError) as exc:
        _enable_qwen35_ane_prefill_for_dflash(
            target, ModelSettings(qwen35_ane_prefill_enabled=True)
        )

    assert exc.value is failure
    assert layer._omlx_ane_prefill_state is None
    assert layer._omlx_ane_prefill_cache == {}
    assert layer._omlx_ane_prefill_failed is True
    assert target._omlx_ane_mlp_prefill_count == 0
    assert target._omlx_ane_resident_program_count == 0
