# SPDX-License-Identifier: Apache-2.0
"""PlanarQuant3 bit-exact constants.

Two rotation constant sets exist in the upstream llama.cpp fork:
  - C CPU reference: COS[]/SIN[] from ggml-planar-quant.c (LCG PRNG seed=42)
  - CUDA/GPU path:  PI_COS[]/PI_SIN[] from planar-iso-constants.cuh
    (generated from torch.manual_seed(42); torch.rand(64)*2π)

The upstream benchmarks use the CUDA constants. We default to the CUDA set
but keep the C reference tables for backward compat and testing.

Packed storage layout matches upstream block_planar3_0:
  norm:  fp16, 2 bytes
  qs:    D/4 bytes, 4 lower-2-bit indices per byte
  signs: D/8 bytes, 8 upper-1-bit signs per byte
  Total: 2 + D/4 + D/8 bytes per D=128 block = 50 bytes = 0.39 bytes/elem
"""

from __future__ import annotations

import mlx.core as mx

PLANAR_D: int = 128
PLANAR_PAIRS: int = PLANAR_D // 2
PLANAR_BITS: int = 3

# Packed block sizes matching upstream block_planar3_0
PLANAR_QS_SIZE: int = PLANAR_D // 4     # 32 bytes: 4 lower-2-bit indices per byte
PLANAR_SIGNS_SIZE: int = PLANAR_D // 8  # 16 bytes: 8 upper-1-bit signs per byte
PLANAR_BLOCK_BYTES: int = 2 + PLANAR_QS_SIZE + PLANAR_SIGNS_SIZE  # 50 bytes

# Lloyd-Max optimal centroids for N(0, 1/128) at 3 bits, symmetric around 0.
PLANAR_CENTROIDS_3BIT: tuple[float, ...] = (
    -0.1906850000,
    -0.1178320000,
    -0.0657170000,
    -0.0214600000,
    +0.0214600000,
    +0.0657170000,
    +0.1178320000,
    +0.1906850000,
)

# 3-bit midpoints for fast quantization (midpoints between adjacent centroids)
PLANAR_MID_3BIT: tuple[float, ...] = (
    -0.154259,
    -0.091775,
    -0.043589,
    0.0,
    0.043589,
    0.091775,
    0.154259,
)

# --- CUDA/GPU rotation constants (PI_COS/PI_SIN) ---
# From planar-iso-constants.cuh, generated from torch.manual_seed(42); torch.rand(64)*2π
# These are the constants used in the upstream CUDA benchmarks.
PLANAR_CUDA_COS_64: tuple[float, ...] = (
    -0.9095053397, +0.1535578452, -0.8537489227, -0.6827218011,
    -0.4249387949, +0.9864510046, +0.9906673944, +0.5752363372,
    -0.9866459035, +0.9878848090, -0.6215683804, -0.9835597698,
    +0.8777263755, -0.4624640047, +0.2843135922, -0.7739960698,
    +0.2385234222, +0.9121914932, -0.8815003943, -0.2639699512,
    -0.5517087300, -0.9035294557, -0.8520543188, -0.5600635985,
    -0.7667286376, -0.9877949369, -0.9781949787, -0.9953372831,
    -0.8622053901, -0.7382118186, +0.9136037642, -0.2558504503,
    -0.8541000475, -0.6159335408, +0.9861256679, -0.6758560284,
    +0.4249571682, -0.6219544719, +0.9130573430, -0.5948161096,
    +0.5759782996, +0.9729901203, +0.6535998325, +0.9222195491,
    -0.7668084044, +0.5116178563, -0.7848786574, +0.9902111051,
    +0.1997167840, +0.7173003220, -0.9999998006, -0.9557868691,
    +0.5594852693, -0.9980111824, +0.9782398557, -0.9150004329,
    -0.4084754305, +0.0071549185, +0.9558482753, -0.0971921648,
    -0.9469334002, +0.9999492419, +0.6100589016, +0.0350818915,
)

PLANAR_CUDA_SIN_64: tuple[float, ...] = (
    -0.4156922383, +0.9881396603, +0.5206849114, -0.7306784124,
    -0.9052220836, +0.1640561354, +0.1363015542, +0.8179872593,
    +0.1628798979, +0.1551889303, +0.7833599099, -0.1805828875,
    -0.4791621957, +0.8866380571, -0.9587313395, +0.6331904010,
    -0.9711367448, +0.4097641756, +0.4721832852, -0.9645309040,
    +0.8340368561, +0.4285259884, +0.5234533769, +0.8284496156,
    +0.6419713361, -0.1557599517, -0.2076886701, +0.0964556523,
    +0.5065588468, -0.6745689815, -0.4066056591, -0.9667163736,
    +0.5201087471, -0.7877981171, +0.1660005034, -0.7370336688,
    +0.9052134584, +0.7830534049, -0.4078312009, -0.8038618014,
    +0.8174649829, -0.2308467584, -0.7568403127, -0.3866666566,
    +0.6418760557, -0.8592131104, +0.6196494922, +0.1395778183,
    +0.9798536657, +0.6967641265, -0.0006314605, +0.2940603015,
    +0.8288402943, -0.0630371303, +0.2074771907, +0.4034528570,
    +0.9127693152, -0.9999744032, +0.2938606379, +0.9952656344,
    +0.3214298299, +0.0100754012, -0.7923560668, -0.9993844410,
)

# --- C CPU reference rotation constants (COS/SIN) ---
# From ggml-planar-quant.c planar_init_rotation(), generated from LCG seed=42
# Kept for backward compat; the CUDA set is the default.
PLANAR_C_REF_COS_64: tuple[float, ...] = (
    +0.7386546135, +0.8607548475, -0.7411674857, +0.9674890637,
    -0.7723053098, -0.8056974411, -0.0412844308, +0.2707833052,
    +0.9315500855, +0.6698185802, +0.9167487621, -0.8320636749,
    +0.6818146110, -0.9108457565, -0.0559285842, -0.9032276273,
    +0.7519487143, -0.8941103816, -0.1039871648, -0.6961420774,
    -0.1230370328, -0.9328963161, -0.2905603051, +0.4910068214,
    +0.7889407277, -0.1221836656, -0.6316579580, +0.3128163815,
    -0.9563610554, +0.9992509484, +0.9540294409, +0.8902468085,
    +0.7543080449, -0.8664138913, -0.5232898593, +0.3621287644,
    -0.8825117350, +0.8234673142, -0.9416025877, -0.5480425358,
    -0.6644080281, -0.6585279703, -0.2460795939, +0.9438471198,
    +0.2427810431, -0.1960992366, +0.2403578013, -0.8461306095,
    +0.0246123374, +0.3372744620, +0.9994974732, -0.3494733870,
    +0.7438930869, +0.8452339768, -0.6177822948, -0.2662552595,
    -0.5457068086, -0.9985070229, +0.7757105827, +0.6141811609,
    -0.9805000424, +0.5425475240, -0.5663578510, -0.4696439803,
)

PLANAR_C_REF_SIN_64: tuple[float, ...] = (
    -0.6740840673, -0.5090196729, +0.6713201404, -0.2529129684,
    +0.6352515221, -0.5923272967, +0.9991474152, -0.9626403451,
    -0.3636130989, +0.7425247431, -0.3994642496, -0.5546801090,
    -0.7315250039, -0.4127469361, -0.9984347820, +0.4291617870,
    -0.6592215896, -0.4478466809, +0.9945786595, -0.7179040313,
    +0.9924020767, +0.3601450622, +0.9568566680, -0.8711557388,
    +0.6144692898, +0.9925075173, +0.7752471566, +0.9498136044,
    -0.2921875417, +0.0386975110, -0.2997128963, +0.4554784000,
    -0.6565206647, -0.4993265271, +0.8521547318, -0.9321280718,
    -0.4702904224, -0.5673637390, -0.3367263079, +0.8364504576,
    -0.7473700047, +0.7525562644, -0.9692496061, -0.3303825557,
    -0.9700810909, +0.9805840850, -0.9706843495, -0.5329755545,
    -0.9996970892, +0.9414063692, +0.0316982083, +0.9369462729,
    +0.6682986617, -0.5343964100, -0.7863491774, -0.9639025331,
    -0.8379761577, +0.0546237342, -0.6310887933, +0.7891650796,
    -0.1965190321, +0.8400250673, -0.8241594434, +0.8828558922,
)

# Backward compat aliases (C reference set)
PLANAR_COS_64 = PLANAR_CUDA_COS_64
PLANAR_SIN_64 = PLANAR_CUDA_SIN_64

assert len(PLANAR_CENTROIDS_3BIT) == (1 << PLANAR_BITS)
assert len(PLANAR_CUDA_COS_64) == PLANAR_PAIRS
assert len(PLANAR_CUDA_SIN_64) == PLANAR_PAIRS
assert len(PLANAR_C_REF_COS_64) == PLANAR_PAIRS
assert len(PLANAR_C_REF_SIN_64) == PLANAR_PAIRS

_centroids_cached: mx.array | None = None
_midpoints_cached: mx.array | None = None
_cos_sin_cache: dict[int, tuple[mx.array, mx.array]] = {}


def centroids_mx() -> mx.array:
    global _centroids_cached
    if _centroids_cached is None:
        _centroids_cached = mx.array(PLANAR_CENTROIDS_3BIT, dtype=mx.float32)
    return _centroids_cached


def midpoints_mx() -> mx.array:
    global _midpoints_cached
    if _midpoints_cached is None:
        _midpoints_cached = mx.array(PLANAR_MID_3BIT, dtype=mx.float32)
    return _midpoints_cached


def _generate_rotations(n_pairs: int) -> tuple[list[float], list[float]]:
    import math
    import numpy as np
    rng = np.random.default_rng(seed=42)
    thetas = rng.uniform(0.0, 2.0 * math.pi, size=n_pairs)
    cos_vals = [float(math.cos(t)) for t in thetas]
    sin_vals = [float(math.sin(t)) for t in thetas]
    return cos_vals, sin_vals


def cos_sin_mx(n_pairs: int | None = None) -> tuple[mx.array, mx.array]:
    """Return cos/sin rotation tables (CUDA/GPU set by default)."""
    if n_pairs is None:
        n_pairs = PLANAR_PAIRS
    cached = _cos_sin_cache.get(n_pairs)
    if cached is not None:
        return cached
    if n_pairs == PLANAR_PAIRS:
        cos_arr = mx.array(PLANAR_CUDA_COS_64, dtype=mx.float32)
        sin_arr = mx.array(PLANAR_CUDA_SIN_64, dtype=mx.float32)
    else:
        cos_vals, sin_vals = _generate_rotations(n_pairs)
        cos_arr = mx.array(cos_vals, dtype=mx.float32)
        sin_arr = mx.array(sin_vals, dtype=mx.float32)
    _cos_sin_cache[n_pairs] = (cos_arr, sin_arr)
    return cos_arr, sin_arr
