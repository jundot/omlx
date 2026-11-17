# SPDX-License-Identifier: Apache-2.0
"""q8q prefill attention (INT8 Q x INT8 K from the affine8 codes) against the
dequantize + fused SDPA route on the same cache."""

import mlx.core as mx
import pytest

import omlx.affine4 as a4
from tests.test_affine4 import require_tensor_ops


@pytest.mark.parametrize("keys,rows", [(4096, 1024), (6144, 2048)])
def test_q8q_matches_the_dequantize_route(monkeypatch, keys, rows):
    require_tensor_ops()
    monkeypatch.setattr(a4, "_Q8Q_MIN_KEYS", 1024)
    cache = a4.Affine8KVCache()
    for start in range(0, keys, 2048):
        n = min(2048, keys - start)
        cache.update_and_fetch(
            mx.random.normal((1, 4, n, 256)).astype(mx.bfloat16),
            mx.random.normal((1, 4, n, 256)).astype(mx.bfloat16),
        )
    mx.eval(cache.state)
    ks, vs = cache.state
    q = mx.random.normal((1, 24, rows, 256)).astype(mx.bfloat16)

    def run(on):
        monkeypatch.setattr(a4, "_Q8Q_ATTENTION", on)
        out = cache.attention(
            q, keys_state=ks, values_state=vs, scale=256**-0.5, mask="causal"
        )
        return out.astype(mx.float32)

    ref, out = run(False), run(True)
    diff = mx.abs(out - ref)
    assert out.shape == ref.shape
    assert mx.max(diff).item() < 0.02 * mx.max(mx.abs(ref)).item() + 1e-3
    assert mx.mean(diff).item() < 2e-3
