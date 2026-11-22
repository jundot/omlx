# SPDX-License-Identifier: Apache-2.0
"""Split-operand clamped SwiGLU on the sorted affine prefill path."""

import mlx.core as mx
import pytest

from omlx.patches import m5_gather_qmm_nax as kernel


@pytest.mark.skipif(
    not kernel.is_nax_available(), reason="requires NAX tensor operations"
)
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize(
    ("bits", "group_size"),
    [(2, 64), (3, 64), (4, 32), (4, 64), (8, 32), (8, 64), (8, 128)],
)
def test_split_affine_gate_up_matches_two_gathers(dtype, bits, group_size):
    w, scales, biases, _, x, idx = kernel._canary_problem(
        dtype, "affine", bits, group_size, 128, 256
    )
    gate = (w[:, :64], scales[:, :64], biases[:, :64])
    up = (w[:, 64:], scales[:, 64:], biases[:, 64:])
    mapping = mx.arange(len(idx) - 1, -1, -1, dtype=mx.uint32)
    expanded = x[mapping]
    stock = kernel._stock_gather_qmm()

    def project(operand):
        return stock(
            expanded,
            *operand,
            rhs_indices=idx,
            group_size=group_size,
            bits=bits,
            sorted_indices=True,
        )

    expected = kernel.reference_activation(project(up), project(gate), 10.0)
    fused = kernel.split_affine_swiglu(
        x, gate, up, idx, mapping, 10.0, bits=bits, group_size=group_size
    )
    assert fused is not None
    assert kernel._bits_equal(expected, fused)
    assert (
        kernel.split_affine_swiglu(
            x, gate, up, idx[:16], mapping[:16], 10.0, bits=bits, group_size=group_size
        )
        is None
    )


def test_split_affine_leaves_non_nax_hosts_on_the_existing_path(monkeypatch):
    monkeypatch.setattr(kernel, "is_nax_available", lambda: False)
    monkeypatch.setattr(
        kernel, "_launch", lambda *args, **kwargs: pytest.fail("launched")
    )
    operand = (mx.zeros((8, 64, 24), dtype=mx.uint32), mx.ones((8, 64, 4)), None)
    x, indices = mx.zeros((128, 1, 256)), mx.zeros((128,), dtype=mx.uint32)
    assert (
        kernel.split_affine_swiglu(
            x, operand, operand, indices, indices, 10.0, bits=3, group_size=64
        )
        is None
    )
