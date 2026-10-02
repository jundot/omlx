# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the MiniMax M3 batched sparse attention patch."""

import mlx.core as mx


def test_storage_q_positions_adds_left_padding_for_minimax_2d_positions():
    from omlx.patches.minimax_m3_sparse_attention import _storage_q_positions

    positions = mx.array([[2048], [1960], [1984]], dtype=mx.int32)
    left_padding = mx.array([0, 88, 64], dtype=mx.int32)

    adjusted = _storage_q_positions(positions, left_padding, 3, 1)
    assert adjusted.tolist() == [[2048], [2048], [2048]]


def test_storage_q_positions_handles_decode_vector_positions():
    from omlx.patches.minimax_m3_sparse_attention import _storage_q_positions

    positions = mx.array([2048, 1960, 1984], dtype=mx.int32)
    left_padding = mx.array([0, 88, 64], dtype=mx.int32)

    adjusted = _storage_q_positions(positions, left_padding, 3, 1)
    assert adjusted.tolist() == [2048, 2048, 2048]


def test_storage_q_positions_leaves_absent_padding_unchanged():
    from omlx.patches.minimax_m3_sparse_attention import _storage_q_positions

    positions = mx.array([[11, 12]], dtype=mx.int32)

    assert _storage_q_positions(positions, None, 1, 2) is positions


def test_preload_dispatches_minimax_m3_sparse_patch(tmp_path, monkeypatch):
    import omlx.patches.minimax_m3_sparse_attention as patch
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    calls = []

    def fake_apply():
        calls.append(True)
        return True

    monkeypatch.setattr(patch, "apply_minimax_m3_sparse_attention_patch", fake_apply)
    (tmp_path / "config.json").write_text('{"model_type": "minimax_m3_vl"}')

    maybe_apply_pre_load_patches(str(tmp_path), for_vlm=True)

    assert calls == [True]


def test_preload_skips_minimax_m3_sparse_patch_for_llm_path(tmp_path, monkeypatch):
    import omlx.patches.minimax_m3_sparse_attention as patch
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    calls = []

    def fake_apply():
        calls.append(True)
        return True

    monkeypatch.setattr(patch, "apply_minimax_m3_sparse_attention_patch", fake_apply)
    (tmp_path / "config.json").write_text('{"model_type": "minimax_m3_vl"}')

    maybe_apply_pre_load_patches(str(tmp_path), for_vlm=False)

    assert calls == []


def test_minimax_m3_sparse_patch_is_idempotent_when_available():
    from omlx.patches.minimax_m3_sparse_attention import (
        apply_minimax_m3_sparse_attention_patch,
    )

    first = apply_minimax_m3_sparse_attention_patch()
    second = apply_minimax_m3_sparse_attention_patch()

    assert first in (True, False)
    assert second is False
