# SPDX-License-Identifier: Apache-2.0
"""Expert offload for the GLM DSA MoE block (omlx/patches/glm_moe_dsa/moe_offload.py).

The adapter swaps the module's projection tensors for resident slots and
runs the module's own forward on slot indices, so every path the resident
model takes (unsorted decode, sorted prefill, the native weighted sum) is
compared against the untouched module on the same routes. Bit-exact where
the kernel path and the per-row inputs are identical; rounding-scale only
where an over-capacity prefill reassembles routes the model's fallback way.
"""

import json

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches.glm_moe_dsa import moe_offload as glm
from omlx.patches.glm_moe_dsa.switch_layers import SwitchGLU
from omlx.patches.moe_expert_offload import (
    apply_moe_expert_offload,
    estimate_offload_admission_bytes,
    materialize_offload_state,
    moe_offload_stats,
)

# top-8 like the flagship: the native weighted-sum kernel accepts top-k 6 or 8
# and half-precision activations only, which is what the real model feeds it.
E, D, INTER, K, GROUP = 32, 64, 32, 8, 32
PREFIX = "model.layers.0.mlp.switch_mlp"


def _make_pair(seed=0, e=E, d=D, inter=INTER, group=GROUP):
    """A split GLM SwitchGLU and its fused twin with identical weights."""
    mx.random.seed(seed)
    split = SwitchGLU(d, inter, e, fused_gate_up=False, inverse_scatter=True)
    fused = SwitchGLU(d, inter, e, fused_gate_up=True, inverse_scatter=True)
    # bf16 weights, as shipped: quantizing them yields bf16 scales and biases,
    # so bf16 activations stay bf16 through gather_qmm (float32 scales would
    # upcast the outputs, which the native weighted-sum kernel rejects).
    for module in (split, fused):
        for lin in module.values():
            if isinstance(lin, nn.Module) and "weight" in lin:
                lin.weight = lin.weight.astype(mx.bfloat16)
        nn.quantize(module, group_size=group, bits=4)
    for field in ("weight", "scales", "biases"):
        setattr(
            fused.gate_up_proj,
            field,
            mx.concatenate([split.gate_proj[field], split.up_proj[field]], axis=1),
        )
        setattr(fused.down_proj, field, split.down_proj[field])
    mx.eval(split.parameters(), fused.parameters())
    return split, fused


def _tensors(split, prefix=PREFIX):
    out = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for field in ("weight", "scales", "biases"):
            out[f"{prefix}.{proj}.{field}"] = getattr(split, proj)[field]
    return out


def _write(tmp_path, tensors, top_k=K):
    mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "glm_moe_dsa", "num_experts_per_tok": top_k})
    )
    return tmp_path


class _MLP(nn.Module):
    def __init__(self, glu):
        super().__init__()
        self.switch_mlp = glu


class _Layer(nn.Module):
    def __init__(self, glu):
        super().__init__()
        self.mlp = _MLP(glu)


class _Inner(nn.Module):
    def __init__(self, glus):
        super().__init__()
        self.layers = [_Layer(g) for g in glus]


class _Model(nn.Module):
    def __init__(self, glus):
        super().__init__()
        self.model = _Inner(glus)


def _copy(glu):
    """A second module instance sharing no arrays with ``glu``."""
    fused = "gate_up_proj" in glu
    twin = SwitchGLU(D, INTER, E, fused_gate_up=fused, inverse_scatter=True)
    nn.quantize(twin, group_size=GROUP, bits=4)
    for lin_name in (
        ("gate_up_proj", "down_proj")
        if fused
        else ("gate_proj", "up_proj", "down_proj")
    ):
        for field in ("weight", "scales", "biases"):
            setattr(twin[lin_name], field, mx.array(glu[lin_name][field]))
    mx.eval(twin.parameters())
    return twin


def _wrapped(tmp_path, reference, fraction):
    model = _Model([_copy(reference)])
    n = glm.apply_glm_moe_expert_offload(model, tmp_path, fraction)
    assert n == 1
    return model.model.layers[0].mlp.switch_mlp


def _routes(shape, e=E, seed=1):
    mx.random.seed(seed)
    return mx.random.randint(0, e, shape)


def _x(*shape):
    return mx.random.normal(shape).astype(mx.bfloat16)


def _scores(indices):
    s = mx.random.uniform(shape=indices.shape)
    return s / s.sum(axis=-1, keepdims=True)


@pytest.fixture(params=["native", "fallback"])
def kernels(request, monkeypatch):
    """With the native GLM kernels, or without them as on a CI runner: the
    module then returns sorted routes unsummed and the caller applies the
    scores, and the adapter must follow the same rule."""
    from omlx.patches.glm_moe_dsa import kernels as k

    if request.param == "fallback":
        monkeypatch.setattr(k, "_native_fast", None)
    elif not k.fast.has("glm_moe_weighted_sum"):
        pytest.skip("native GLM kernels are not built here")
    return request.param


@pytest.fixture(params=["fused", "split"])
def reference(request, tmp_path):
    split, fused = _make_pair()
    _write(tmp_path, _tensors(split))
    return fused if request.param == "fused" else split


def test_wrap_replaces_module_and_keeps_only_slots(tmp_path, reference):
    wrapped = _wrapped(tmp_path, reference, 0.25)
    assert isinstance(wrapped, glm.OffloadedSwitchGLU)
    cache = wrapped.cache
    assert cache.capacity == 8 and cache.n_experts == E
    for lin_name, _ in cache.layout:
        lin = cache.glu[lin_name]
        for field in ("weight", "scales", "biases"):
            assert lin[field].shape[0] == 8
            assert lin[field].shape[1:] == reference[lin_name][field].shape[1:]
    # the wrapper registers no parameters of its own: the slots live off-tree
    assert not wrapped.parameters()
    assert materialize_offload_state(_Model([wrapped])) == 1


def test_decode_bit_exact_at_quarter_residency(tmp_path, reference):
    wrapped = _wrapped(tmp_path, reference, 0.25)
    x = _x(4, 1, D)
    i = _routes((4, 1, K))
    ref, got = reference(x, i), wrapped(x, i)
    mx.eval(ref, got)
    assert bool(mx.array_equal(ref, got))
    assert moe_offload_stats(_Model([wrapped]))["misses"] == len(
        set(i.reshape(-1).tolist())
    )


def test_sorted_prefill_bit_exact_at_full_residency(tmp_path, reference, kernels):
    wrapped = _wrapped(tmp_path, reference, 1.0)
    x = _x(2, 40, D)
    i = _routes((2, 40, K))  # 160 routes: the sorted path
    for weighted in (False, True):
        s = _scores(i)
        ref = reference(x, i, scores=s, weighted_sum=weighted)
        got = wrapped(x, i, scores=s, weighted_sum=weighted)
        mx.eval(ref, got)
        assert ref.shape == got.shape
        assert ref.ndim == (3 if weighted and kernels == "native" else 4)
        assert bool(mx.array_equal(ref, got)), f"weighted_sum={weighted}"


def test_sorted_prefill_within_capacity_bit_exact(tmp_path, reference):
    """Routes that fit the cache take the module's own sorted path, keyed by
    slot instead of expert: every row still meets its own expert."""
    wrapped = _wrapped(tmp_path, reference, 0.5)  # 16 slots
    x = _x(1, 48, D)
    i = _routes((1, 48, K), e=12)  # 96 routes over 12 distinct experts
    s = _scores(i)
    ref = reference(x, i, scores=s, weighted_sum=True)
    got = wrapped(x, i, scores=s, weighted_sum=True)
    mx.eval(ref, got)
    assert bool(mx.array_equal(ref, got))


@pytest.mark.parametrize("weighted", [False, True])
def test_over_capacity_prefill_rounding_bounded(tmp_path, reference, kernels, weighted):
    wrapped = _wrapped(tmp_path, reference, 0.25)  # 8 slots
    x = _x(2, 64, D)
    i = _routes((2, 64, K))  # far more distinct experts than slots
    s = _scores(i)
    ref = reference(x, i, scores=s, weighted_sum=weighted)
    got = wrapped(x, i, scores=s, weighted_sum=weighted)
    mx.eval(ref, got)
    assert ref.shape == got.shape
    assert ref.ndim == (3 if weighted and kernels == "native" else 4)
    assert float(mx.abs(ref - got).max()) < 2e-2
    # every distinct expert was installed exactly once for this call, and
    # each install read exactly one expert's worth of bytes
    assert wrapped.cache.misses == len(set(i.reshape(-1).tolist()))
    assert (
        wrapped.cache.fetched_bytes == wrapped.cache.misses * wrapped.cache.expert_bytes
    )


def test_lru_eviction_and_counters(tmp_path, reference):
    wrapped = _wrapped(tmp_path, reference, 0.25)  # 8 slots
    x = _x(1, 1, D)
    first = mx.arange(8).reshape(1, 1, 8)
    wrapped(x, first)
    assert (wrapped.cache.hits, wrapped.cache.misses) == (0, 8)
    wrapped(x, first)
    assert (wrapped.cache.hits, wrapped.cache.misses) == (8, 8)
    wrapped(x, mx.array([[[8, 9]]]))  # evicts the two least recently used
    assert wrapped.cache.misses == 10 and len(wrapped.cache.slot_of) == 8
    assert 0 not in wrapped.cache.slot_of and 1 not in wrapped.cache.slot_of
    got = wrapped(x, mx.array([[[0, 9]]]))
    ref = reference(x, mx.array([[[0, 9]]]))
    mx.eval(got, ref)
    assert bool(mx.array_equal(ref, got))


def test_uncovered_checkpoint_is_skipped(tmp_path):
    split, fused = _make_pair()
    tensors = _tensors(split)
    tensors.pop(f"{PREFIX}.up_proj.scales")
    _write(tmp_path, tensors)
    model = _Model([_copy(fused)])
    assert glm.apply_glm_moe_expert_offload(model, tmp_path, 0.25) == 0
    assert isinstance(model.model.layers[0].mlp.switch_mlp, SwitchGLU)


def test_kill_switch(tmp_path, reference, monkeypatch):
    monkeypatch.setenv("OMLX_MOE_EXPERT_OFFLOAD", "0")
    model = _Model([_copy(reference)])
    assert glm.apply_glm_moe_expert_offload(model, tmp_path, 0.25) == 0
    assert apply_moe_expert_offload(model, tmp_path, 0.25) == 0


def test_common_entry_point_dispatches_glm(tmp_path, reference):
    """The engine calls apply_moe_expert_offload; GLM blocks are wrapped by
    their adapter, counted once, and seen by the shared walkers."""
    model = _Model([_copy(reference)])
    assert apply_moe_expert_offload(model, tmp_path, 0.25) == 1
    wrapped = model.model.layers[0].mlp.switch_mlp
    assert isinstance(wrapped, glm.OffloadedSwitchGLU)
    assert materialize_offload_state(model) == 1
    x = _x(1, 1, D)
    wrapped(x, mx.array([[[3, 5]]]))
    assert moe_offload_stats(model) == {
        "layers": 1,
        "hits": 0,
        "misses": 2,
        "hit_rate": 0.0,
    }


def test_admission_estimate_counts_glm_experts(tmp_path):
    split, _ = _make_pair()
    tensors = _tensors(split)
    tensors["model.embed_tokens.weight"] = mx.zeros((16, D), dtype=mx.float16)
    _write(tmp_path, tensors)
    expert_bytes = sum(
        v.size * v.dtype.size for k, v in tensors.items() if ".switch_mlp." in k
    )
    full = 10**9
    assert estimate_offload_admission_bytes(tmp_path, full, 0.25) == full - int(
        expert_bytes * 0.75
    )
