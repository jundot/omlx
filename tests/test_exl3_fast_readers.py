"""Sushi 1.0.5 readers must preserve codewords and native accumulation results."""

import mlx.core as mx
import numpy as np
import pytest

from omlx.quantization import exl3

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


@pytest.mark.parametrize("halfwords", range(32, 65, 2))
def test_funnel_codewords_match_independent_pairwise_reader(halfwords):
    rng = np.random.default_rng(halfwords)
    packed = rng.integers(0, 65536, halfwords, dtype=np.uint16)
    probe = mx.fast.metal_kernel(
        name="exl3_codeword_probe",
        input_names=["packed"],
        output_names=["codes"],
        header=exl3._COOP_HEADER,
        source="""
uint t = uint(thread_position_in_grid.x);
const device uint *words = (const device uint *)packed;
ulong bits = exl3_lane<uint(NHW)>(words, t / 8u);
codes[t] = uint(bits >> exl3_lane_sh(uint(NHW), t % 8u)) & 65535u;
""",
    )
    codes = probe(inputs=[mx.array(packed)], template=[("NHW", halfwords)],
                  grid=(256, 1, 1), threadgroup=(128, 1, 1),
                  output_shapes=[(256,)], output_dtypes=[mx.uint32])[0]
    mx.eval(codes)
    # Independent pairwise 32-bit funnel oracle (not the new lane reader).
    words = [int(packed[i]) | (int(packed[i + 1]) << 16)
             for i in range(0, halfwords, 2)]
    expected = []
    for t in range(128):
        a = ((t * 2 + 1) * halfwords) // 16
        b = ((t * 2 + 2) * halfwords) // 16
        i = (a + halfwords * 16 - 16) // 32
        j = (b + halfwords * 16 - 1) // 32
        shift = (j + 1) * 32 - (b + halfwords * 16)
        funnel = ((words[i % len(words)] << 32) | words[j % len(words)]) >> shift
        expected.extend([(funnel >> (b - a)) & 65535, funnel & 65535])
    np.testing.assert_array_equal(np.asarray(codes), expected)


@pytest.mark.parametrize("halfwords", range(32, 65, 2))
@pytest.mark.parametrize("rows", [3, 53])
def test_fast_projection_is_bit_identical_to_legacy(halfwords, rows, monkeypatch):
    rng = np.random.default_rng(halfwords + rows)
    packed = mx.array(rng.integers(0, 65536, (2, 8, 8, halfwords), dtype=np.uint16))
    scales = mx.array(rng.choice([-1.0, 1.0], (2, 128)).astype(np.float16))
    out_scales = mx.array(rng.uniform(0.01, 0.15, (2, 128)).astype(np.float16))
    layer = exl3.Exl3SwitchLinear(packed, scales, out_scales, exl3.Exl3Spec(halfwords, 15))
    x = mx.array(rng.normal(size=(rows, 1, 128)).astype(np.float16))
    ids = mx.array(rng.integers(0, 2, rows, dtype=np.uint32))
    monkeypatch.setattr(exl3, "_FAST_READERS", False)
    old = layer(x, ids)
    mx.eval(old)
    monkeypatch.setattr(exl3, "_FAST_READERS", True)
    new = layer(x, ids)
    mx.eval(new)
    np.testing.assert_array_equal(np.asarray(new), np.asarray(old))
