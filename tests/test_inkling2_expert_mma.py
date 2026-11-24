# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 expert-grouped 3-bit prefill matmul: every (token, expert) pair
keeps the bits the sorted ``gather_qmm`` gives it."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.inkling2 import expert_mma
from omlx.patches.inkling2 import model as inkling2

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

_EXPERTS, _TOP_K = 16, 4


def _proj(out_dims, in_dims, dtype, experts=_EXPERTS, bits=3, seed=0):
    spec = {"mode": "affine", "bits": bits, "group_size": 64}
    proj = inkling2.ExpertProj(experts, out_dims, in_dims, spec)
    mx.random.seed(seed)
    w = (mx.random.normal((experts, out_dims, in_dims)) * 0.05).astype(dtype)
    proj.weight, proj.scales, proj.biases = mx.quantize(w, group_size=64, bits=bits)
    mx.eval(proj.parameters())
    return proj


def _routing(tokens, seed, experts=_EXPERTS):
    """Skewed like Inkling 2: a few experts take most pairs (several 16-pair
    tiles), some take one, others none."""
    rng = np.random.default_rng(seed)
    p = 1.0 / np.arange(1, experts + 1) ** 1.5
    p[-2:] = 0  # never routed
    p = p[rng.permutation(experts)] / p.sum()
    ids = np.stack(
        [rng.choice(experts, _TOP_K, replace=False, p=p) for _ in range(tokens)]
    )
    return mx.array(ids.astype(np.uint32))


def _sorted_reference(proj, x, inds, per_row):
    """The model's sorted path: _gather_sort, gather_qmm, _scatter_unsort."""
    xe = mx.expand_dims(x, (-2, -3)) if per_row > 1 else x[:, :, None, :]
    if per_row > 1:
        xs, idx, inv = inkling2._gather_sort(xe, inds)
    else:
        # w2 rows are one per pair: [N, K, 1, I] sorted like the model sorts.
        flat = inds.reshape(-1)
        order = mx.argsort(flat)
        xs, idx, inv = (
            x.reshape(-1, 1, x.shape[-1])[order],
            flat[order],
            mx.argsort(order),
        )
    z = proj(xs, idx, True)
    return inkling2._scatter_unsort(z, inv, inds.shape).reshape(-1, z.shape[-1])


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize(
    "out_dims, in_dims", [(2304, 2304), (2304, 1152), (128, 256)], ids=str
)
@pytest.mark.parametrize("tokens", [16, 70, 200])
def test_pairs_keep_sorted_gather_qmm_bits(dtype, out_dims, in_dims, tokens):
    dtype = getattr(mx, dtype)
    proj = _proj(out_dims, in_dims, dtype)
    assert expert_mma.supported(proj, dtype)
    inds = _routing(tokens, tokens)
    pairs = inds.reshape(-1)
    assert expert_mma.applies(pairs.size, _EXPERTS)
    routing = expert_mma.plan(pairs, _EXPERTS)
    # w13: one row per token for all its experts.
    x = (mx.random.normal((tokens, in_dims)) * 0.7).astype(dtype)
    got = expert_mma.gather(proj, x, _TOP_K, routing)
    assert mx.array_equal(got, _sorted_reference(proj, x, inds, _TOP_K))
    # w2: one row per pair.
    xp = (mx.random.normal((tokens, _TOP_K, in_dims)) * 0.7).astype(dtype)
    got = expert_mma.gather(proj, xp.reshape(-1, in_dims), 1, routing)
    assert mx.array_equal(got, _sorted_reference(proj, xp, inds, 1))


def test_applies_follows_gather_qmm_matrix_dispatch():
    # Four pairs per expert, at least 16, up to MAX_PAIRS.
    assert not expert_mma.applies(511, 128)
    assert expert_mma.applies(512, 128)
    assert not expert_mma.applies(15, 2)
    assert expert_mma.applies(16, 2)
    assert expert_mma.applies(expert_mma.MAX_PAIRS, 128)
    assert not expert_mma.applies(expert_mma.MAX_PAIRS + 1, 128)


@pytest.mark.parametrize(
    "case", ["bits4", "float32", "scales_dtype", "nax", "cpu", "nvfp4"]
)
def test_other_formats_keep_gather_qmm(case, monkeypatch):
    if case == "nvfp4":
        proj = inkling2.ExpertProj(
            _EXPERTS, 128, 256, inkling2.expert_quant_spec("nvfp4")
        )
        assert not expert_mma.supported(proj, mx.bfloat16)
        return
    proj = _proj(128, 256, mx.float16, bits=4 if case == "bits4" else 3)
    dtype = mx.float32 if case == "float32" else mx.float16
    if case == "scales_dtype":
        proj.scales = proj.scales.astype(mx.bfloat16)
    if case == "nax":
        monkeypatch.setattr(expert_mma, "is_nax_available", lambda: True)
    if case == "cpu":
        monkeypatch.setattr(mx, "default_device", lambda: mx.cpu)
    assert not expert_mma.supported(proj, dtype)


def test_inkling2_prefill_keeps_its_bits(monkeypatch):
    """A tiny model with 3-bit affine experts: prefill chunks through the
    grouped kernel give the logits of the sorted gather_qmm path."""
    from tests.test_inkling_vlm_mtp import _bf16, _inkling2_tiny

    fmt = {"mode": "affine", "bits": 3, "group_size": 64}
    ids = mx.random.randint(0, 300, (1, 80))

    def run():
        model = _bf16(
            _inkling2_tiny(formats=(fmt,) * 12, hidden_size=256, intermediate_size=128)
        )
        cache = model.make_cache()
        outs = [model(ids[:, :40], cache=cache), model(ids[:, 40:76], cache=cache)]
        outs += [model(ids[:, t : t + 1], cache=cache) for t in (76, 77)]
        mx.eval(outs)
        return outs

    calls = []
    gather = expert_mma.gather
    monkeypatch.setattr(
        expert_mma, "gather", lambda *a, **k: calls.append(1) or gather(*a, **k)
    )
    got = run()
    # Both chunks sort their pairs (>= 64) and route >= 4 per expert (8
    # experts, top-2): w13 + w2 in each of 12 layers.
    assert len(calls) == 2 * 2 * 12
    monkeypatch.setattr(expert_mma, "supported", lambda *a: False)
    calls.clear()
    want = run()
    assert not calls
    for a, b in zip(got, want):
        assert mx.array_equal(a, b)
