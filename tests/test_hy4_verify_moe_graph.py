# SPDX-License-Identifier: Apache-2.0
"""Short-verify graph eligibility, live parameters and layer lifetime."""

import gc
import weakref

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from omlx.patches.deepseek_v4.switch_layers import (  # noqa: E402
    QuantizedSwitchLinear,
    SwitchGLU,
)
from omlx.patches.hy_v4 import language, layers, verify_dispatch  # noqa: E402


def fixture(bits):
    decoder = language.DecoderLayer.__new__(language.DecoderLayer)
    nn.Module.__init__(decoder)
    decoder._verify_moe_graph = None
    moe = language.MoE.__new__(language.MoE)
    nn.Module.__init__(moe)
    switch = SwitchGLU.__new__(SwitchGLU)
    nn.Module.__init__(switch)
    for name in ("up_proj", "gate_proj", "down_proj"):
        projection = QuantizedSwitchLinear.__new__(QuantizedSwitchLinear)
        nn.Module.__init__(projection)
        projection.mode = "affine"
        projection.bits = bits
        projection.group_size = 64
        switch[name] = projection
    moe.switch_mlp = switch
    moe.gain = mx.array(1, dtype=mx.bfloat16)
    decoder.mlp = moe
    return decoder.eval()


def test_short_verify_graph_is_bounded_by_decoder_lifetime(monkeypatch):
    monkeypatch.setattr(language.MoE, "__call__", lambda self, x: x * self.gain)
    monkeypatch.setattr(verify_dispatch, "eligible", lambda rows: 2 <= rows <= 6)

    with mx.stream(mx.cpu):
        decoder = fixture(4)
        child = weakref.ref(decoder.mlp)
        owner = weakref.ref(decoder)
        sample = mx.ones((1, 2, 2), dtype=mx.bfloat16)
        decoder._mlp_forward(sample[:, :1]).tolist()
        assert decoder._verify_moe_graph is None
        assert mx.array_equal(decoder._mlp_forward(sample), sample).item()
        assert decoder._verify_moe_graph is not None
        assert "_verify_moe_graph" not in decoder.state
        decoder.mlp.gain = mx.array(2, dtype=mx.bfloat16)
        assert mx.array_equal(decoder._mlp_forward(sample), sample * 2).item()
        del decoder
        gc.collect()
        assert owner() is None and child() is None

        other = fixture(3)
        assert mx.array_equal(other._mlp_forward(sample), sample).item()
        assert other._verify_moe_graph is None


def test_short_verify_tail_updates_children_and_unloads(monkeypatch):
    monkeypatch.setattr(language.MoE, "__call__", lambda self, x: x * self.gain)
    monkeypatch.setattr(
        layers.HyperConnectionPre,
        "_mix",
        lambda self, x, weights: (x[:, :, 0], mx.ones(x.shape[:-1])),
    )
    with mx.stream(mx.cpu):
        decoder = fixture(4)
        decoder._verify_tail_graph = None
        attention = layers.Attention.__new__(layers.Attention)
        nn.Module.__init__(attention)
        attention.o_proj = nn.Identity()
        decoder.self_attn = attention
        decoder.post_attention_layernorm = nn.Identity()
        connector = layers.HyperConnection.__new__(layers.HyperConnection)
        nn.Module.__init__(connector)
        pre = layers.HyperConnectionPre.__new__(layers.HyperConnectionPre)
        nn.Module.__init__(pre)
        pre.hc_fn = mx.zeros((8, 8))
        pre._fused = layers._FusedWeight()
        connector.hc_pre = pre
        decoder.hc_mlp_layer = connector

        values = mx.ones((1, 1, 2, 2), dtype=mx.bfloat16)
        gate = mx.ones((1, 2, 1, 2), dtype=mx.bfloat16)
        streams = mx.zeros((1, 2, 4, 2), dtype=mx.bfloat16)
        post = mx.ones((1, 2, 4))
        args = (values, gate, streams, post)
        assert mx.array_equal(decoder._verify_tail(*args), streams + 2).item()
        assert "_verify_tail_graph" not in decoder.state

        decoder.mlp.gain = mx.array(2, dtype=mx.bfloat16)
        assert mx.array_equal(decoder._verify_tail(*args), streams + 3).item()
        decoder.mlp = fixture(4).mlp
        decoder.mlp.gain = mx.array(3, dtype=mx.bfloat16)
        assert mx.array_equal(decoder._verify_tail(*args), streams + 4).item()
        owner, child = weakref.ref(decoder), weakref.ref(decoder.mlp)
        del decoder
        gc.collect()
        assert owner() is None and child() is None
