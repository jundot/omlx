# SPDX-License-Identifier: Apache-2.0
"""Expert offload for Inkling 2 (omlx/patches/inkling2/moe_offload.py).

A tiny model with 32 experts, top-8, at the 8-slot floor: prefill routes to
more experts than the cache holds (expert-major chunks), decode runs the
compiled-layer path eagerly and evicts. Logits match the resident model at
rounding scale: chunked gathers and the eager post-attention stage round
differently from the compiled resident path.
"""

import json

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from omlx.patches.inkling2.moe_offload import OffloadedInkling2Experts
from omlx.patches.moe_expert_offload import (
    CheckpointExpertStore,
    apply_moe_expert_offload,
    materialize_offload_state,
    moe_offload_stats,
)
from tests.test_inkling_vlm_mtp import _TINY_LOOP, _inkling2_tiny

_DIMS = dict(n_routed_experts=32, num_experts_per_tok=8)


def _offloaded(tmp_path, dims):
    reference = _inkling2_tiny(**dims, **_DIMS)
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(reference.parameters()))
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "inkling2_loop", "text_config": _DIMS})
    )
    model = _inkling2_tiny(**dims, **_DIMS)
    assert apply_moe_expert_offload(model, tmp_path, 0.25) == len(model.model.layers)
    materialize_offload_state(model)
    return reference, model


@pytest.mark.parametrize("dims", [{}, _TINY_LOOP], ids=["mxfp8+nvfp4", "looped"])
def test_offloaded_logits_match_resident(tmp_path, dims):
    reference, model = _offloaded(tmp_path, dims)
    assert all(
        isinstance(layer.mlp.experts, OffloadedInkling2Experts)
        and layer.mlp.experts.cache.capacity == 8
        for layer in model.model.layers
    )

    ids = mx.random.randint(0, 300, (1, 24))
    outs = []
    for m in (reference, model):
        cache = m.make_cache()
        logits = [m(ids[:, :16], cache=cache)]  # prefill: over capacity
        for t in range(16, 24):  # decode: compiled-layer path, evictions
            logits.append(m(ids[:, t : t + 1], cache=cache))
        outs.append(mx.concatenate(logits, axis=1))
    mx.eval(outs)
    assert mx.abs(outs[0][..., :300] - outs[1][..., :300]).max().item() < 1e-4
    assert moe_offload_stats(model)["misses"] > 8 * len(model.model.layers)


def test_over_capacity_call_reads_only_missing_experts(tmp_path):
    """Resident experts run first, so a call touching every expert reads only
    the ones not already in the cache."""
    _, model = _offloaded(tmp_path, {})
    experts = model.model.layers[0].mlp.experts
    c = experts.cache
    x = mx.random.normal((64, 128)).astype(mx.bfloat16)
    inds = mx.arange(64 * 8).reshape(64, 8) % 32  # all 32 experts
    w = mx.ones((64, 8))
    experts(x, inds, w)  # warm: leaves the cache full
    before = c.misses
    experts(x, inds, w)
    assert c.misses - before == 32 - c.capacity


def test_decode_overlap_matches_serial(tmp_path, monkeypatch):
    """With slow reads, decode steps run the resident routes while the misses
    are read and keep the GPU busy; outputs match the resident module bit for
    bit and the cache ends in the same state as with the overlap off."""
    import time

    reference, _ = _offloaded(tmp_path, {})
    rng = np.random.default_rng(0)
    steps = [
        (
            mx.random.normal((1, 128)).astype(mx.bfloat16),
            mx.array(rng.choice(32, (1, 8), replace=False), dtype=mx.int32),
            mx.random.uniform(shape=(1, 8)),
        )
        for _ in range(8)
    ]
    ref_experts = reference.model.layers[0].mlp.experts
    refs = [ref_experts(*step) for step in steps]
    mx.eval(refs)
    read, async_eval = CheckpointExpertStore.read, mx.async_eval

    def run(overlap):
        model = _inkling2_tiny(**_DIMS)
        apply_moe_expert_offload(model, tmp_path, 0.25)
        materialize_offload_state(model)
        experts = model.model.layers[0].mlp.experts
        experts._overlap = overlap
        dispatched, outs = [], []
        with monkeypatch.context() as patch:
            patch.setattr(
                CheckpointExpertStore,
                "read",
                staticmethod(lambda p: time.sleep(0.003) or read(p)),
            )
            patch.setattr(
                mx, "async_eval", lambda *a: dispatched.append(1) or async_eval(*a)
            )
            for step in steps:
                outs.append(experts(*step))
                mx.eval(outs[-1])
        c = experts.cache
        return outs, (c.hits, c.misses, list(c.slot_of.items())), len(dispatched)

    on, on_state, n_on = run(True)
    off, off_state, n_off = run(False)
    assert on_state == off_state and on_state[1] > 8  # misses evicted
    assert n_on > 0 and n_off == 0
    for ref, a, b in zip(refs, on, off):
        assert bool(mx.array_equal(ref, a)) and bool(mx.array_equal(ref, b))


def test_offloaded_row_exact_verify_matches_resident(tmp_path):
    """MTP verify rows run the row-exact compiled stage (``_post_exact``); the
    offloaded MoE syncs routes to the host, so that stage must leave it out of
    its compiled trace like the decode stage does."""
    from omlx.patches.qwen35_verify_qmm import set_verify_qmm_armed

    reference, model = _offloaded(tmp_path, {})
    ids = mx.random.randint(0, 300, (1, 19))
    blocks = []
    for m in (reference, model):
        cache = m.make_cache()
        m(ids[:, :16], cache=cache)
        set_verify_qmm_armed(True, row_exact=True)
        try:
            blocks.append(m(ids[:, 16:], cache=cache))
        finally:
            set_verify_qmm_armed(False)
    mx.eval(blocks)
    assert mx.abs(blocks[0][..., :300] - blocks[1][..., :300]).max().item() < 1e-4
