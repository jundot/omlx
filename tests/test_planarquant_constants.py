# SPDX-License-Identifier: Apache-2.0
"""Bit-exact parity with llama.cpp ggml-planar-quant.c and CUDA constants."""

from __future__ import annotations

import mlx.core as mx

from omlx.cache.planarquant.constants import (
    PLANAR_BITS,
    PLANAR_BLOCK_BYTES,
    PLANAR_CENTROIDS_3BIT,
    PLANAR_CUDA_COS_64,
    PLANAR_CUDA_SIN_64,
    PLANAR_C_REF_COS_64,
    PLANAR_C_REF_SIN_64,
    PLANAR_D,
    PLANAR_MID_3BIT,
    PLANAR_PAIRS,
    PLANAR_QS_SIZE,
    PLANAR_SIGNS_SIZE,
    centroids_mx,
    cos_sin_mx,
    midpoints_mx,
)


def test_planar_d_and_pairs():
    assert PLANAR_D == 128
    assert PLANAR_PAIRS == 64
    assert PLANAR_BITS == 3


def test_packed_sizes():
    assert PLANAR_QS_SIZE == 32  # D/4 = 128/4
    assert PLANAR_SIGNS_SIZE == 16  # D/8 = 128/8
    assert PLANAR_BLOCK_BYTES == 50  # 2 + 32 + 16


def test_centroid_bit_exact_parity():
    expected = (
        -0.1906850000, -0.1178320000, -0.0657170000, -0.0214600000,
        0.0214600000, 0.0657170000, 0.1178320000, 0.1906850000,
    )
    assert expected == PLANAR_CENTROIDS_3BIT
    assert len(PLANAR_CENTROIDS_3BIT) == 8


def test_midpoints_between_centroids():
    assert len(PLANAR_MID_3BIT) == 7
    # Each midpoint should be between adjacent centroids
    for i in range(7):
        assert PLANAR_CENTROIDS_3BIT[i] <= PLANAR_MID_3BIT[i] <= PLANAR_CENTROIDS_3BIT[i + 1]


def test_cuda_cos_endpoints():
    assert PLANAR_CUDA_COS_64[0] == -0.9095053397
    assert PLANAR_CUDA_COS_64[1] == 0.1535578452
    assert PLANAR_CUDA_COS_64[62] == 0.6100589016
    assert PLANAR_CUDA_COS_64[63] == 0.0350818915


def test_cuda_sin_endpoints():
    assert PLANAR_CUDA_SIN_64[0] == -0.4156922383
    assert PLANAR_CUDA_SIN_64[1] == 0.9881396603
    assert PLANAR_CUDA_SIN_64[62] == -0.7923560668
    assert PLANAR_CUDA_SIN_64[63] == -0.9993844410


def test_c_ref_cos_endpoints():
    assert PLANAR_C_REF_COS_64[0] == 0.7386546135
    assert PLANAR_C_REF_COS_64[63] == -0.4696439803


def test_c_ref_sin_endpoints():
    assert PLANAR_C_REF_SIN_64[0] == -0.6740840673
    assert PLANAR_C_REF_SIN_64[63] == 0.8828558922


def test_cos_sin_sum_of_squares_near_one():
    cos, sin = cos_sin_mx()
    sq = cos * cos + sin * sin
    max_err = float(mx.max(mx.abs(sq - 1.0)).item())
    assert max_err < 1e-6, f"cos^2+sin^2 drift: {max_err}"


def test_centroids_mx_roundtrip():
    arr = centroids_mx()
    assert arr.shape == (8,)
    assert arr.dtype == mx.float32


def test_midpoints_mx_roundtrip():
    arr = midpoints_mx()
    assert arr.shape == (7,)
    assert arr.dtype == mx.float32
