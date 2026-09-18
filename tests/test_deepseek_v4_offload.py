# SPDX-License-Identifier: Apache-2.0
"""Expert offload for the DeepSeek V4 / glm5_next MoE block.

(omlx/patches/deepseek_v4/moe_offload.py)

The adapter swaps the module's projection tensors for resident slots and
runs the module's own forward on slot indices, so every path the resident
model takes (unsorted decode, sorted prefill through the native block/pair
kernels, the native weighted sum) is compared against the untouched module
on the same routes. Bit-exact where the kernel path and the per-row inputs
are identical; rounding-scale only where an over-capacity prefill
reassembles routes the model's fallback way.
"""

import json

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches.deepseek_v4 import moe_offload as dsv4
from omlx.patches.deepseek_v4.switch_layers import SwitchGLU
from omlx.patches.moe_expert_offload import (
    apply_moe_expert_offload,
    estimate_offload_admission_bytes,
    materialize_offload_state,
    moe_offload_stats,
)

# top-6 like DeepSeek V4 Flash: the native weighted-sum kernel accepts
# top-k 6 or 8 on half-precision activations, which is what the real model
# feeds it.
E, D, INTER, K, GROUP = 32, 64, 32, 6, 32
PREFIX = "model.layers.0.ffn.switch_mlp"


def _make_glu(seed=0, e=E, d=D, inter=INTER, group=GROUP):
    mx.random.seed(seed)
    glu = SwitchGLU(d, inter, e)
    # bf16 weights, as shipped: quantizing them yields bf16 scales and
    # biases, so bf16 activations stay bf16 through gather_qmm.
    for lin in glu.values():
        if isinstance(lin, nn.Module) and "weight" in lin:
            lin.weight = lin.weight.astype(mx.bfloat16)
    nn.quantize(glu, group_size=group, bits=4)
    mx.eval(glu.parameters())
    return glu


def _tensors(glu, prefix=PREFIX):
    out = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for field in ("weight", "scales", "biases"):
            out[f"{prefix}.{proj}.{field}"] = getattr(glu, proj)[field]
    return out


def _write(tmp_path, tensors, top_k=K, model_type="deepseek_v4"):
    mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": model_type, "num_experts_per_tok": top_k})
    )
    return tmp_path


class _FFN(nn.Module):
    def __init__(self, glu):
        super().__init__()
        self.switch_mlp = glu


class _Layer(nn.Module):
    def __init__(self, glu):
        super().__init__()
        self.ffn = _FFN(glu)


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
    twin = _make_glu(seed=99)
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for field in ("weight", "scales", "biases"):
            setattr(twin[proj], field, mx.array(glu[proj][field]))
    mx.eval(twin.parameters())
    return twin


def _wrapped(tmp_path, reference, fraction):
    model = _Model([_copy(reference)])
    n = dsv4.apply_deepseek_v4_moe_expert_offload(model, tmp_path, fraction)
    assert n == 1
    return model.model.layers[0].ffn.switch_mlp


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
    """With the native GLM/DSv4 kernels, or without them as on a CI runner:
    the module then returns sorted routes unsummed and the caller applies the
    scores, and the adapter must follow the same rule."""
    from omlx.custom_kernels.glm_moe_dsa import fast as k

    if request.param == "fallback":
        monkeypatch.setattr(k, "_ext", None)
        monkeypatch.setattr(k, "has_symbol", lambda name: False)
    elif not k.has_symbol("glm_moe_weighted_sum"):
        pytest.skip("native GLM kernels are not built here")
    return request.param


@pytest.fixture()
def reference(tmp_path):
    glu = _make_glu()
    _write(tmp_path, _tensors(glu))
    return glu


def test_wrap_replaces_module_and_keeps_only_slots(tmp_path, reference):
    wrapped = _wrapped(tmp_path, reference, 0.25)
    assert isinstance(wrapped, dsv4.OffloadedSwitchGLU)
    cache = wrapped.cache
    assert cache.capacity == 8 and cache.n_experts == E
    for proj in ("gate_proj", "up_proj", "down_proj"):
        lin = cache.glu[proj]
        for field in ("weight", "scales", "biases"):
            assert lin[field].shape[0] == 8
            assert lin[field].shape[1:] == reference[proj][field].shape[1:]
    # num_experts is a property of the weight shape: it follows the slots
    assert cache.glu.gate_proj.num_experts == 8
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
    glu = _make_glu()
    tensors = _tensors(glu)
    tensors.pop(f"{PREFIX}.up_proj.scales")
    _write(tmp_path, tensors)
    model = _Model([_copy(glu)])
    assert dsv4.apply_deepseek_v4_moe_expert_offload(model, tmp_path, 0.25) == 0
    assert isinstance(model.model.layers[0].ffn.switch_mlp, SwitchGLU)


def test_kill_switch(tmp_path, reference, monkeypatch):
    monkeypatch.setenv("OMLX_MOE_EXPERT_OFFLOAD", "0")
    model = _Model([_copy(reference)])
    assert dsv4.apply_deepseek_v4_moe_expert_offload(model, tmp_path, 0.25) == 0
    assert apply_moe_expert_offload(model, tmp_path, 0.25) == 0


def test_common_entry_point_dispatches_dsv4(tmp_path, reference):
    """The engine calls apply_moe_expert_offload; DeepSeek V4 blocks are
    wrapped by their adapter, counted once, and seen by the shared walkers."""
    model = _Model([_copy(reference)])
    assert apply_moe_expert_offload(model, tmp_path, 0.25) == 1
    wrapped = model.model.layers[0].ffn.switch_mlp
    assert isinstance(wrapped, dsv4.OffloadedSwitchGLU)
    assert materialize_offload_state(model) == 1
    x = _x(1, 1, D)
    wrapped(x, mx.array([[[3, 5]]]))
    assert moe_offload_stats(model) == {
        "layers": 1,
        "hits": 0,
        "misses": 2,
        "hit_rate": 0.0,
    }


def test_compile_ffn_layers_stay_eager_when_offloaded(tmp_path, reference):
    """glm5_next decoder layers compile their FFN block at decode shapes
    (mlx_vlm language.py ``compile_ffn``). The offloaded block manages
    slots on the host and cannot be traced — ``tolist()`` inside
    ``mx.compile`` dies with "eval during function transformations". The
    wrap must turn that compilation off, and the layer must then run the
    offloaded block eagerly."""

    class _MoEHost(nn.Module):
        def __init__(self, glu):
            super().__init__()
            self.switch_mlp = glu

    class _CompilingLayer(nn.Module):
        def __init__(self, glu):
            super().__init__()
            self.mlp = _MoEHost(glu)
            self.compile_ffn = True
            self._ffn_c = None

        def __call__(self, x):
            if self.compile_ffn:
                if self._ffn_c is None:
                    self._ffn_c = mx.compile(self._ffn_block)
                return self._ffn_c(x)
            return self._ffn_block(x)

        def _ffn_block(self, x):
            return self.mlp.switch_mlp(x, mx.zeros((1, 1, K), dtype=mx.int32))

    glu = _copy(reference)
    layer = _CompilingLayer(glu)
    model = nn.Module()
    model.layers = [layer]
    prefix = "layers.0.mlp.switch_mlp"
    _write(tmp_path, _tensors(glu, prefix=prefix))

    assert dsv4.apply_deepseek_v4_moe_expert_offload(model, tmp_path, 0.25) == 1
    assert layer.compile_ffn is False
    x = _x(1, 1, D)
    got = layer(x)  # would raise the eval-during-trace error if compiled
    ref = reference(x, mx.zeros((1, 1, K), dtype=mx.int32))
    mx.eval(got, ref)
    assert bool(mx.array_equal(ref, got))


def test_admission_estimate_counts_dsv4_experts(tmp_path):
    glu = _make_glu()
    tensors = _tensors(glu)
    tensors["model.embed_tokens.weight"] = mx.zeros((16, D), dtype=mx.float16)
    _write(tmp_path, tensors)
    expert_bytes = sum(
        v.size * v.dtype.size for k, v in tensors.items() if ".switch_mlp." in k
    )
    full = 10**9
    assert estimate_offload_admission_bytes(tmp_path, full, 0.25) == full - int(
        expert_bytes * 0.75
    )
