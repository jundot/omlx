# SPDX-License-Identifier: Apache-2.0
import mlx.core as mx
import pytest

from omlx.patches.inkling2.model import ExpertProj


@pytest.mark.parametrize("in_dims", [2304, 4608])
@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
def test_affine_expert_carriers_match_mlx_quantize(in_dims, bits):
    group_size = 128
    experts, out_dims = 2, 8
    packed, scales, biases = mx.quantize(
        mx.zeros((experts, out_dims, in_dims), dtype=mx.bfloat16),
        group_size=group_size,
        bits=bits,
        mode="affine",
    )
    proj = ExpertProj(
        experts,
        out_dims,
        in_dims,
        {"mode": "affine", "bits": bits, "group_size": group_size},
    )
    assert proj.weight.shape == packed.shape
    assert proj.scales.shape == scales.shape
    assert proj.biases.shape == biases.shape
