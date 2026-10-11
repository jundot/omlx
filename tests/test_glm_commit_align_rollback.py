# SPDX-License-Identifier: Apache-2.0
"""GLM shared MTP verification with boundary-aligned row rollback."""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from test_mlx_lm_mtp_patch import _cache_rows, _model, generate

from omlx.patches import mlx_lm_mtp
from omlx.patches.mlx_lm_mtp import batch_generator as bg
from omlx.patches.mlx_lm_mtp import fused_batch


@pytest.mark.parametrize("batch_size,late_join", [(2, False), (3, False), (3, True)])
def test_glm_commit_align_vector_rollback_preserves_boundary_tokens(
    monkeypatch, batch_size, late_join
):
    previous = mlx_lm_mtp.is_mtp_active()
    previous_depth = mlx_lm_mtp.get_mtp_depth()
    mlx_lm_mtp.set_mtp_active(True)
    mlx_lm_mtp.set_mtp_depth(2)
    monkeypatch.setattr(bg, "_DepthController", lambda *args, **kwargs: None)
    monkeypatch.setattr(bg, "_batch_policy_for_next", lambda batch: None)
    try:
        mx.random.seed(173)
        model = _model("glm5")
        host = model._language_model
        mx.eval(model.parameters())
        prompts = [[3, 4, 5, 6, 7], [3, 6, 7, 8, 4, 5, 6], [4, 5, 6]]
        prompts = prompts[:batch_size]
        limits = [18, 22, 17][:batch_size]
        host._omlx_mtp_decode_enabled = False
        expected, _ = generate(model, prompts, limits, late_join=late_join)
        host._omlx_mtp_decode_enabled = True
        # Force boundary crossings within the short generation.
        model._omlx_mtp_commit_align = 4
        vector_commits = []
        original_rollback = host.rollback_speculative_cache

        def rollback(caches, captures, accepted, block_size):
            if isinstance(accepted, list):
                vector_commits.append(tuple(accepted))
            return original_rollback(caches, captures, accepted, block_size)

        monkeypatch.setattr(host, "rollback_speculative_cache", rollback)
        monkeypatch.setattr(
            fused_batch,
            "copy",
            SimpleNamespace(
                deepcopy=lambda cache: pytest.fail(
                    "Supported GLM shared verification must retain its batch cache"
                )
            ),
        )
        materialized = []
        last_verify_rows = [0]
        original_materialize = bg._materialize_mtp_boundary_emit

        def materialize(row, state):
            materialized.append(last_verify_rows[0])
            return original_materialize(row, state)

        original_backbone = bg._call_backbone

        def backbone(target, inputs, cache, n_confirmed=0):
            rows = _cache_rows(cache)
            assert rows is None or rows == int(inputs.shape[0])
            if n_confirmed:
                last_verify_rows[0] = int(inputs.shape[0])
            return original_backbone(target, inputs, cache, n_confirmed)

        monkeypatch.setattr(bg, "_materialize_mtp_boundary_emit", materialize)
        monkeypatch.setattr(bg, "_call_backbone", backbone)
        actual, _ = generate(model, prompts, limits, late_join=late_join)
        assert actual == expected
        assert any(rows > 1 for rows in materialized)
        assert vector_commits
    finally:
        mlx_lm_mtp.set_mtp_active(previous)
        mlx_lm_mtp.set_mtp_depth(previous_depth)


@pytest.mark.parametrize("commit_align", [0, 4])
def test_glm_shared_verify_unsupported_rollback_width_falls_back(
    monkeypatch, commit_align
):
    previous = mlx_lm_mtp.is_mtp_active()
    previous_depth = mlx_lm_mtp.get_mtp_depth()
    mlx_lm_mtp.set_mtp_active(True)
    mlx_lm_mtp.set_mtp_depth(2)
    monkeypatch.setattr(bg, "_DepthController", lambda *args, **kwargs: None)
    monkeypatch.setattr(bg, "_batch_policy_for_next", lambda batch: None)
    try:
        mx.random.seed(173)
        model = _model("glm5")
        host = model._language_model
        mx.eval(model.parameters())
        prompts = [[3, 4, 5, 6, 7], [3, 6, 7, 8, 4, 5, 6]]
        limits = [18, 22]
        # A DFlash-sized block exceeds BatchPoolingCache's undo window.
        host._omlx_mtp_depth = 15
        # External drafters are not bounded by the built-in head's depth cap.
        monkeypatch.setattr(
            bg, "_resolve_mtp_chain_depth", lambda target: (True, 15, True)
        )
        model._omlx_mtp_commit_align = commit_align
        shared_widths = []
        scalar_commits = []
        original_backbone = bg._call_backbone
        original_rollback = host.rollback_speculative_cache

        def backbone(target, inputs, cache, n_confirmed=0):
            if n_confirmed and int(inputs.shape[0]) > 1:
                shared_widths.append(int(inputs.shape[1]))
            return original_backbone(target, inputs, cache, n_confirmed)

        def rollback(caches, captures, accepted, block_size):
            assert not isinstance(
                accepted, list
            ), "Unsupported vector rollback selected"
            scalar_commits.append(block_size)
            return original_rollback(caches, captures, accepted, block_size)

        monkeypatch.setattr(bg, "_call_backbone", backbone)
        monkeypatch.setattr(host, "rollback_speculative_cache", rollback)
        original_group = fused_batch._advance_group

        class CheckedScalarFallbackError(Exception):
            pass

        def advance_group(batch, depth, rows, replacements, **kwargs):
            if depth != 15:
                return original_group(batch, depth, rows, replacements, **kwargs)
            # The existing scalar path can refuse this width. Its fallback
            # signal, rather than a vector-rollback error, is the contract.
            with pytest.raises(bg._MtpStepFallback, match="rejects scalar rollback"):
                original_group(batch, depth, rows, replacements, **kwargs)
            raise CheckedScalarFallbackError

        monkeypatch.setattr(fused_batch, "_advance_group", advance_group)
        with pytest.raises(CheckedScalarFallbackError):
            generate(model, prompts, limits)
        assert 16 in shared_widths
        assert 16 in scalar_commits
    finally:
        mlx_lm_mtp.set_mtp_active(previous)
        mlx_lm_mtp.set_mtp_depth(previous_depth)
