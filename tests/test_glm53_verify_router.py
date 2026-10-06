# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 MTP verify blocks route each row like its one-token decode step."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.custom_kernels.nax import is_nax_available
from omlx.patches import mlx_vlm_glm5_next_compat as compat
from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk


@pytest.fixture(autouse=True)
def _apply_glm5_next_compat():
    compat.apply_mlx_vlm_glm5_next_compat_patch()


# One-row routing tests require the GLM NAX GPU kernels.
@pytest.fixture
def glm5_fused_decode():
    """Require NAX kernels for verification-to-one-token routing parity."""
    if not is_nax_available():
        pytest.skip("the fused GLM-5.3 decode kernels run on M5 (NAX) GPUs")
    compat.apply_mlx_vlm_glm5_next_compat_patch()
    from mlx_vlm.models.glm5_next import language

    assert language._DECODE_FUSION
    return language


def _language():
    from mlx_vlm.models.glm5_next import language

    return language


def _bits(a: mx.array) -> mx.array:
    view = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return a.view(view)


def _mismatches(a: mx.array, b: mx.array) -> int:
    assert a.shape == b.shape and a.dtype == b.dtype
    return int(mx.sum(_bits(a) != _bits(b)).item())


def _router(experts=288, hidden=4096, seed=0):
    language = _language()
    mx.random.seed(seed)
    cfg = SimpleNamespace(
        num_experts_per_tok=8,
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=2.5,
        n_routed_experts=experts,
        hidden_size=hidden,
    )
    gate = language.Glm5NextMoEGate(cfg)
    gate.weight = mx.random.normal((experts, hidden)) * 0.02
    gate.e_score_correction_bias = mx.random.normal((experts,)) * 0.01
    return gate


@pytest.mark.usefixtures("glm5_fused_decode")
@pytest.mark.parametrize("rows", [2, 3, 4, 8])
def test_verify_rows_route_like_their_decode_step(rows):
    gate = _router(seed=rows)
    for trial in range(3):
        x = (mx.random.normal((1, rows, 4096)) * (0.3 + trial)).astype(mx.bfloat16)
        before = dk.STATS["router"]
        indices, scores = gate(x)
        assert dk.STATS["router"] == before + 1
        assert indices.shape == (1, rows, 8) and scores.shape == (1, rows, 8)
        for r in range(rows):
            one_idx, one_scores = gate(x[:, r : r + 1])
            assert mx.array_equal(indices[:, r : r + 1], one_idx).item(), (trial, r)
            assert _mismatches(scores[:, r : r + 1], one_scores) == 0, (trial, r)


@pytest.mark.usefixtures("glm5_fused_decode")
def test_verify_rows_select_the_reference_experts(monkeypatch):
    """Same selected set as the composed router on the same fp32 logits."""
    language = compat_language()
    gate = _router(seed=11)
    x = (mx.random.normal((1, 4, 4096)) * 0.7).astype(mx.bfloat16)
    indices, scores = gate(x)
    rows = [gate(x[:, r : r + 1]) for r in range(4)]
    monkeypatch.setattr(language, "_DECODE_FUSION", False)
    for r in range(4):
        ref_idx, ref_scores = gate(x[:, r : r + 1])
        assert mx.array_equal(rows[r][0], ref_idx).item()
        assert _mismatches(rows[r][1], ref_scores) == 0
        assert sorted(indices[0, r].tolist()) == sorted(ref_idx[0, 0].tolist())


@pytest.mark.usefixtures("glm5_fused_decode")
@pytest.mark.parametrize("batch,rows,groups", [(2, 4, 1), (1, 9, 1), (1, 4, 2)])
def test_ineligible_blocks_keep_the_reference(batch, rows, groups, monkeypatch):
    language = compat_language()
    gate = _router(experts=128, hidden=1024, seed=5)
    gate.n_group = groups
    x = (mx.random.normal((batch, rows, 1024)) * 0.7).astype(mx.bfloat16)
    before = dk.STATS["router"]
    indices, scores = gate(x)
    assert dk.STATS["router"] == before
    monkeypatch.setattr(language, "_DECODE_FUSION", False)
    ref_idx, ref_scores = gate(x)
    assert mx.array_equal(indices, ref_idx).item()
    assert _mismatches(scores, ref_scores) == 0


@pytest.mark.usefixtures("glm5_fused_decode")
def test_disabled_fusion_keeps_the_block_reference(monkeypatch):
    language = compat_language()
    monkeypatch.setattr(language, "_DECODE_FUSION", False)
    gate = _router(seed=6)
    x = mx.random.normal((1, 4, 4096)).astype(mx.bfloat16)
    before = dk.STATS["router"]
    indices, scores = gate(x)
    ref_idx, ref_scores = language.group_expert_select(
        x.astype(mx.float32) @ gate.weight.astype(mx.float32).T,
        gate.e_score_correction_bias,
        gate.top_k,
        gate.n_group,
        gate.topk_group,
        gate.routed_scaling_factor,
        gate.norm_topk_prob,
    )
    assert dk.STATS["router"] == before
    assert mx.array_equal(indices, ref_idx).item()
    assert _mismatches(scores, ref_scores) == 0


@pytest.mark.usefixtures("glm5_fused_decode")
def test_declined_kernel_keeps_the_block_reference(monkeypatch):
    language = compat_language()
    # K >= 16 * E selects an unsupported split-K GEMV configuration.
    gate = _router(experts=16, hidden=1024, seed=7)
    x = mx.random.normal((1, 4, 1024)).astype(mx.bfloat16)
    before = dk.STATS["router"]
    indices, scores = gate(x)
    assert dk.STATS["router"] == before
    monkeypatch.setattr(language, "_DECODE_FUSION", False)
    ref_idx, ref_scores = gate(x)
    assert mx.array_equal(indices, ref_idx).item()
    assert _mismatches(scores, ref_scores) == 0


def compat_language():
    from mlx_vlm.models.glm5_next import language

    return language
