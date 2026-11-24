# SPDX-License-Identifier: Apache-2.0
"""Short-verify graph eligibility, live parameters and layer lifetime."""

import gc
import weakref

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from omlx.patches.deepseek_v4.switch_layers import QuantizedSwitchLinear, SwitchGLU
from omlx.patches.hy_v4 import language, verify_dispatch


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
