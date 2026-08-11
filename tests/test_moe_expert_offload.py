# SPDX-License-Identifier: Apache-2.0
"""Tests for MoE expert offloading (omlx/patches/moe_expert_offload.py).

Assertion policy (measured against the pinned mlx-lm, see module docstring):
decode and unsorted/chunked prefill are BIT-EXACT at any residency; the
sorted prefill kernel is presentation-invariant at real model dimensions, so
full-residency prefill is bit-exact there too. Where partial residency
legitimately chunks below the sort threshold, the sorted and unsorted
gather_qmm kernels differ by ~4e-3 absolute (measured at gemma-26B geometry,
output magnitude ~5), so those cases assert a rounding-scale tolerance —
head-room for kernel choice, not for wrong experts, which show as O(1).
"""

import pytest

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import SwitchGLU

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")

if HAS_MLX:
    from omlx.patches.moe_expert_offload import (
        CheckpointExpertStore,
        OffloadSwitchGLU,
        apply_moe_expert_offload,
        moe_offload_stats,
    )

# toy geometry: E large enough that a 25% fraction clears the capacity floor
E, D, INTER, K, GROUP = 32, 64, 32, 2, 32


def _make_glu(seed=0, e=E, d=D, inter=INTER, group=GROUP):
    mx.random.seed(seed)
    glu = SwitchGLU(d, inter, e)
    nn.quantize(glu, group_size=group, bits=4)
    return glu


def _glu_tensors(glu, prefix):
    out = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        lin = getattr(glu, proj)
        for field in ("weight", "scales", "biases"):
            if lin.get(field) is not None:
                out[f"{prefix}.{proj}.{field}"] = lin[field]
    return out


def _save_checkpoint(tmp_path, tensors):
    mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)
    return tmp_path


class _Experts(nn.Module):
    def __init__(self, glu):
        super().__init__()
        self.switch_glu = glu


class _Layer(nn.Module):
    def __init__(self, glu):
        super().__init__()
        self.experts = _Experts(glu)


class _MiniMoE(nn.Module):
    def __init__(self, glus):
        super().__init__()
        self.layers = [_Layer(g) for g in glus]

    def __call__(self, x, indices):
        for layer in self.layers:
            x = x + layer.experts.switch_glu(x, indices).sum(axis=-2)
        return x


def _ri(*shape, e=E):
    return mx.random.randint(0, e, shape)


class TestCheckpointExpertStore:
    def test_fetch_matches_source_rows(self, tmp_path):
        glu = _make_glu()
        _save_checkpoint(tmp_path, _glu_tensors(glu, "layers.0.experts.switch_glu"))
        store = CheckpointExpertStore(tmp_path)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(glu, proj)
            for field in ("weight", "scales"):
                name = f"layers.0.experts.switch_glu.{proj}.{field}"
                assert store.has(name)
                assert store.spec(name)[0] == tuple(lin[field].shape)
                for e in (0, 1, E - 1):
                    got, want = store.fetch_expert(name, e), lin[field][e]
                    mx.eval(got, want)
                    assert got.dtype == want.dtype
                    assert bool(mx.array_equal(got, want))

    def test_bf16_roundtrip(self, tmp_path):
        glu = _make_glu()
        scales = glu.gate_proj["scales"].astype(mx.bfloat16)
        _save_checkpoint(tmp_path, {"t.scales": scales})
        store = CheckpointExpertStore(tmp_path)
        got = store.fetch_expert("t.scales", 3)
        mx.eval(got)
        assert got.dtype == mx.bfloat16
        assert bool(mx.array_equal(got.view(mx.uint16), scales[3].view(mx.uint16)))

    def test_multi_shard(self, tmp_path):
        glu = _make_glu()
        t = _glu_tensors(glu, "layers.0.experts.switch_glu")
        names = sorted(t)
        mx.save_safetensors(
            str(tmp_path / "model-00001-of-00002.safetensors"),
            {k: t[k] for k in names[:3]},
        )
        mx.save_safetensors(
            str(tmp_path / "model-00002-of-00002.safetensors"),
            {k: t[k] for k in names[3:]},
        )
        store = CheckpointExpertStore(tmp_path)
        for k in names:
            assert store.has(k)
            assert bool(mx.array_equal(store.fetch_expert(k, 2), t[k][2]))

    def test_empty_dir_is_falsy(self, tmp_path):
        assert not CheckpointExpertStore(tmp_path)


class TestApplyAndForward:
    def _wrapped_model(self, tmp_path, n_layers=2, fraction=0.25):
        glus = [_make_glu(seed=i) for i in range(n_layers)]
        tensors = {}
        for i, g in enumerate(glus):
            tensors.update(_glu_tensors(g, f"layers.{i}.experts.switch_glu"))
        _save_checkpoint(tmp_path, tensors)
        model = _MiniMoE(glus)
        return model, glus

    def test_apply_wraps_all_covered_layers(self, tmp_path):
        model, _ = self._wrapped_model(tmp_path)
        n = apply_moe_expert_offload(model, tmp_path, resident_fraction=0.25)
        assert n == 2
        for layer in model.layers:
            assert isinstance(layer.experts.switch_glu, OffloadSwitchGLU)
            assert layer.experts.switch_glu.cache.capacity == 8  # 25% of 32

    def test_decode_bit_exact_at_partial_residency(self, tmp_path):
        model, _ = self._wrapped_model(tmp_path)
        cases = [
            (mx.random.normal((1, 1, D)), _ri(1, 1, K)),
            (mx.random.normal((4, 1, D)), _ri(4, 1, K)),
        ]
        refs = [model(x, i) for x, i in cases]
        mx.eval(*refs)
        assert apply_moe_expert_offload(model, tmp_path, 0.25) == 2
        for (x, i), ref in zip(cases, refs):
            got = model(x, i)
            mx.eval(got)
            assert bool(mx.array_equal(ref, got))
        stats = moe_offload_stats(model)
        assert stats["layers"] == 2 and stats["misses"] > 0

    def test_chunked_prefill_bit_exact_below_sort_threshold(self, tmp_path):
        # 27 tokens x k=2 = 54 indices: below the sort threshold, above the
        # 8-slot working set -> the chunking path runs and stays bit-exact.
        model, _ = self._wrapped_model(tmp_path)
        x, i = mx.random.normal((3, 9, D)), _ri(3, 9, K)
        ref = model(x, i)
        mx.eval(ref)
        apply_moe_expert_offload(model, tmp_path, 0.25)
        got = model(x, i)
        mx.eval(got)
        assert bool(mx.array_equal(ref, got))

    def test_batch_invariance(self, tmp_path):
        model, _ = self._wrapped_model(tmp_path, n_layers=1)
        apply_moe_expert_offload(model, tmp_path, 0.25)
        glu = model.layers[0].experts.switch_glu
        rows = [(mx.random.normal((1, 1, D)), _ri(1, 1, K)) for _ in range(3)]
        singles = [glu(x, i) for x, i in rows]
        batched = glu(
            mx.concatenate([r[0] for r in rows]), mx.concatenate([r[1] for r in rows])
        )
        mx.eval(*singles, batched)
        for j in range(3):
            assert bool(mx.array_equal(batched[j], singles[j][0]))

    def test_skips_uncovered_layer(self, tmp_path):
        glus = [_make_glu(seed=0), _make_glu(seed=1)]
        # checkpoint covers only layer 0
        _save_checkpoint(tmp_path, _glu_tensors(glus[0], "layers.0.experts.switch_glu"))
        model = _MiniMoE(glus)
        assert apply_moe_expert_offload(model, tmp_path, 0.25) == 1
        assert isinstance(model.layers[0].experts.switch_glu, OffloadSwitchGLU)
        assert type(model.layers[1].experts.switch_glu) is SwitchGLU

    def test_skips_non_quantized(self, tmp_path):
        mx.random.seed(9)
        glu = SwitchGLU(D, INTER, E)  # float — unsupported in v1
        _save_checkpoint(
            tmp_path,
            {
                f"layers.0.experts.switch_glu.{p}.weight": getattr(glu, p)["weight"]
                for p in ("gate_proj", "up_proj", "down_proj")
            },
        )
        assert apply_moe_expert_offload(_MiniMoE([glu]), tmp_path, 0.25) == 0

    def test_skips_unknown_dtype(self, tmp_path):
        """A checkpoint field in an unrecognized storage format must skip the
        layer at coverage time, not KeyError at the first cache miss."""
        import json as _json
        import struct as _struct

        glus = [_make_glu(seed=0)]
        _save_checkpoint(tmp_path, _glu_tensors(glus[0], "layers.0.experts.switch_glu"))
        # Rewrite one field's header dtype tag to something unsupported.
        # data_offsets are relative to the data section, so a resized header
        # keeps them valid.
        p = tmp_path / "model.safetensors"
        raw = p.read_bytes()
        n = _struct.unpack("<Q", raw[:8])[0]
        header = _json.loads(raw[8 : 8 + n])
        header["layers.0.experts.switch_glu.up_proj.scales"]["dtype"] = "F64"
        new_header = _json.dumps(header).encode()
        p.write_bytes(_struct.pack("<Q", len(new_header)) + new_header + raw[8 + n :])
        assert apply_moe_expert_offload(_MiniMoE(glus), tmp_path, 0.25) == 0

    def test_kill_switch(self, tmp_path, monkeypatch):
        model, _ = self._wrapped_model(tmp_path)
        monkeypatch.setenv("OMLX_MOE_EXPERT_OFFLOAD", "0")
        assert apply_moe_expert_offload(model, tmp_path, 0.25) == 0
        assert type(model.layers[0].experts.switch_glu) is SwitchGLU

    def test_idempotent_second_apply_is_noop(self, tmp_path):
        model, _ = self._wrapped_model(tmp_path)
        assert apply_moe_expert_offload(model, tmp_path, 0.25) == 2
        # OffloadSwitchGLU is not `type(...) is SwitchGLU`; nothing to rewrap
        assert apply_moe_expert_offload(model, tmp_path, 0.25) == 0


@pytest.mark.slow
class TestRealGeometry:
    """gemma-26B expert geometry: where the sorted kernel is presentation-
    invariant, so even the sorted prefill path is bit-exact."""

    E, D, INTER, K, GROUP = 128, 2816, 704, 8, 64

    @pytest.fixture(scope="class")
    def setup(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("real_geom")
        glu = _make_glu(seed=42, e=self.E, d=self.D, inter=self.INTER, group=self.GROUP)
        _save_checkpoint(tmp, _glu_tensors(glu, "layers.0.experts.switch_glu"))
        return tmp, glu

    def _fresh(self, setup, fraction):
        tmp, glu = setup
        model = _MiniMoE([glu])
        n = apply_moe_expert_offload(model, tmp, fraction)
        assert n == 1
        return model, model.layers[0].experts.switch_glu, glu

    def test_sorted_prefill_bit_exact_at_full_residency(self, setup):
        _, wrapped, glu = self._fresh(setup, 1.0)
        x = mx.random.normal((2, 40, self.D))
        i = _ri(2, 40, self.K, e=self.E)
        ref, got = glu(x, i), wrapped(x, i)
        mx.eval(ref, got)
        assert bool(mx.array_equal(ref, got))

    def test_quarter_residency_rounding_bounded(self, setup):
        _, wrapped, glu = self._fresh(setup, 0.25)
        x = mx.random.normal((2, 64, self.D))
        i = _ri(2, 64, self.K, e=self.E)
        ref, got = glu(x, i), wrapped(x, i)
        mx.eval(ref, got)
        assert float(mx.abs(ref - got).max()) < 2e-2

    def test_decode_bit_exact_at_quarter_residency(self, setup):
        _, wrapped, glu = self._fresh(setup, 0.25)
        x = mx.random.normal((8, 1, self.D))
        i = _ri(8, 1, self.K, e=self.E)
        ref, got = glu(x, i), wrapped(x, i)
        mx.eval(ref, got)
        assert bool(mx.array_equal(ref, got))
