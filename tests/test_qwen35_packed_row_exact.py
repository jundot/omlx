# SPDX-License-Identifier: Apache-2.0
"""Row-exact verify through M5's packed 4-bit projections.

With row-exact verify armed, every multi-row ``PackedLinear`` call and every
grouped ``_linears`` dispatch must give each row the bits of the serial
one-row decode call of that same projection. Serial decode runs the SIMD
matvec on the projection's own width, and its K split depends on that width,
so a shared store must not be served with its combined width. Disarmed, the
multi-row paths are unchanged.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches import qwen35_packed_linear as packed
from omlx.patches import qwen35_verify_qmm


def _nax_available() -> bool:
    try:
        from omlx.custom_kernels.nax import is_nax_available

        return bool(is_nax_available())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _nax_available(), reason="needs the M5 tensor unit")


@pytest.fixture(autouse=True)
def disarm():
    qwen35_verify_qmm.set_verify_qmm_armed(False)
    yield
    qwen35_verify_qmm.set_verify_qmm_armed(False)


def _quantized_linear(n, k, bits, seed):
    mx.random.seed(seed)
    layer = nn.QuantizedLinear(k, n, bias=False, group_size=64, bits=bits)
    weight = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    layer.weight, layer.scales, layer.biases = mx.quantize(
        weight, group_size=64, bits=bits
    )
    return layer


def _serial(linear, x):
    return mx.concatenate([linear(x[:, r : r + 1]) for r in range(x.shape[1])], axis=1)


def _bit_equal(a, b):
    return mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item()


def _x(rows, k, seed):
    mx.random.seed(seed)
    return (mx.random.normal((1, rows, k)) * 0.5).astype(mx.bfloat16)


# (K, widths sharing one store). K = 5120 with 32 + 48 tiles: each projection
# alone splits K 20 ways, the 80-tile store only 10 ways. K = 6144 and K = 256
# (one K split, bf16 written directly) cover single projections.
STORES = [
    (5120, (4096, 6144)),
    (6144, (1024,)),
    (256, (512,)),
]


@pytest.mark.parametrize("k,widths", STORES)
@pytest.mark.parametrize("rows", [2, 4, 8, 12])
def test_armed_call_rows_equal_serial_decode(k, widths, rows):
    store = packed._pack(
        [_quantized_linear(n, k, 4, 10 + i) for i, n in enumerate(widths)]
    )
    x = _x(rows, k, 1)
    serial = [_serial(lin, x) for lin in store]
    qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=True)
    armed = [lin(x) for lin in store]
    for out, ref in zip(armed, serial):
        assert out.shape == ref.shape
        assert _bit_equal(out, ref)


@pytest.mark.parametrize("rows", [2, 5, 8])
def test_armed_grouped_store_rows_equal_serial_decode(rows):
    k = 5120
    qkv = _quantized_linear(4096, k, 4, 1)
    z = _quantized_linear(6144, k, 4, 2)
    b = _quantized_linear(48, k, 5, 3)
    a = _quantized_linear(48, k, 5, 4)
    qkv_p, z_p = packed._pack([qkv, z])
    assert qkv_p._store is z_p._store
    assert packed._ksplit(qkv_p._store.N // packed.TILE_N, k // 256) != packed._ksplit(
        z_p.output_dims // packed.TILE_N, k // 256
    )

    class Verifier:
        def _linears(self, linears, x):
            return tuple(linear(x) for linear in linears)

    packed.apply(Verifier)
    x = _x(rows, k, 2)
    serial = [_serial(qkv_p, x), _serial(z_p, x)]
    qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=True)
    tail = [b(x), a(x)]
    outputs = Verifier()._linears((qkv_p, z_p, b, a), x)
    qwen35_verify_qmm.set_verify_qmm_armed(False)
    assert _bit_equal(outputs[0], serial[0])
    assert _bit_equal(outputs[1], serial[1])
    # The non-packed tail keeps the original routing.
    assert _bit_equal(outputs[2], tail[0])
    assert _bit_equal(outputs[3], tail[1])


def test_disarmed_multi_row_paths_unchanged():
    k = 5120
    qkv_p, z_p = packed._pack(
        [_quantized_linear(4096, k, 4, 1), _quantized_linear(6144, k, 4, 2)]
    )
    x = _x(4, k, 3)
    x2 = x.reshape(-1, k)
    store = qkv_p._store

    class Verifier:
        def _linears(self, linears, x):
            return tuple(linear(x) for linear in linears)

    packed.apply(Verifier)
    grouped = Verifier()._linears((qkv_p, z_p), x)
    combined = packed.packed_matmul(x2, store.w, store.sc, store.bi, k, store.N)
    assert _bit_equal(grouped[0][0], combined[:, :4096])
    assert _bit_equal(grouped[1][0], combined[:, 4096:])
    own = packed.packed_matmul(
        x2, z_p.packed_weight, z_p.packed_scales, z_p.packed_biases, k, 6144
    )
    assert _bit_equal(z_p(x)[0], own)
