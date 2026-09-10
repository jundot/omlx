"""Correctness tests for the oQ mixed-bit QxA8 prefill kernels.

Kernel correctness and model accuracy are separate concerns, and the kernel
tests come first.

  1. decoded Q4/Q5 codes are bit-exact against an independent bit unpacker,
     which is in turn tied to MLX's own dequantize semantics
  2. Stage A's Qa is bit-exact and Ra is exactly the sum of the codes
  3. the INT32 dot and the FP32 affine accumulation match an independent
     implementation

Everything that needs the native extension is skipped when it is absent, so
this file is meaningful on a default install: the reference implementations and
the classification logic are exercised regardless.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

GROUP_SIZE = 64


def _kernels():
    """The native module, or None when this build/host cannot run it."""
    try:
        from omlx.custom_kernels.qwen35_prefill import fast
    except Exception:
        return None
    if not fast.oq_a8_available():
        return None
    return fast


requires_kernels = pytest.mark.skipif(
    _kernels() is None,
    reason="oQ A8 NAX kernels are unavailable (no native build or no tensor units)",
)


# --------------------------------------------------------------------------
# Independent references
# --------------------------------------------------------------------------


def unpack_codes(packed: np.ndarray, bits: int, K: int) -> np.ndarray:
    """Unpack MLX's affine uint32 code stream, independently of the kernel.

    Codes run little-endian along the bit stream of each row, so a row of K
    codes occupies exactly K * bits / 32 words -- which is why GS64 lands on a
    word boundary for both Q4 (8 words) and Q5 (10 words).
    """
    n_rows = packed.shape[0]
    stream = np.unpackbits(
        packed.view(np.uint8).reshape(n_rows, -1), axis=1, bitorder="little"
    )
    starts = np.arange(K) * bits
    codes = np.zeros((n_rows, K), dtype=np.int64)
    for b in range(bits):
        codes |= stream[:, starts + b].astype(np.int64) << b
    return codes


def quantize_activations(x: np.ndarray, act_mode: int):
    """Reference Stage A.

    rint() is roundTiesToEven in both Metal and numpy, so the codes compare
    exactly rather than within a tolerance.
    """
    x = x.astype(np.float32)
    rows, K = x.shape
    groups = K // GROUP_SIZE

    if act_mode == 0:
        amax = np.abs(x).max(axis=1)
        scale = np.where(amax > 0, amax / 127.0, 0.0).astype(np.float32)
        inv = np.where(amax > 0, 127.0 / amax, 0.0).astype(np.float32)
        qa = np.clip(np.rint(x * inv[:, None]), -127, 127).astype(np.int8)
    else:
        blocks = x.reshape(rows, groups, GROUP_SIZE)
        amax = np.abs(blocks).max(axis=2)
        scale = np.where(amax > 0, amax / 127.0, 0.0).astype(np.float32)
        inv = np.where(amax > 0, 127.0 / amax, 0.0).astype(np.float32)
        qa = np.clip(np.rint(blocks * inv[:, :, None]), -127, 127)
        qa = qa.astype(np.int8).reshape(rows, K)

    ra = qa.reshape(rows, groups, GROUP_SIZE).sum(axis=2, dtype=np.int64)
    # |Ra| <= 64 * 127 = 8128 always fits INT16.
    assert np.abs(ra).max() <= GROUP_SIZE * 127
    return qa, scale, ra.astype(np.int16)


def affine_reference(qa, sa, ra, codes, sw, bw, act_mode) -> np.ndarray:
    """Reference for the group-accumulated affine GEMM.

    Deliberately written the long way -- integer dot per group, then scale and
    bias -- rather than by dequantizing the weights, so it is an independent
    check on the accumulator hierarchy and not a restatement of it.
    """
    rows, K = qa.shape
    groups = K // GROUP_SIZE
    n_out = codes.shape[0]

    qa_g = qa.reshape(rows, groups, GROUP_SIZE).astype(np.int64)
    qw_g = codes.reshape(n_out, groups, GROUP_SIZE).astype(np.int64)

    out = np.zeros((rows, n_out), dtype=np.float64)
    for g in range(groups):
        # D[m, n] = sum_k Qa * Qw, exact in integer arithmetic.
        d = qa_g[:, g, :] @ qw_g[:, g, :].T
        contribution = sw[None, :, g] * d + bw[None, :, g] * ra[:, g, None]
        scale = sa[:, None] if act_mode == 0 else sa[:, g, None]
        out += scale * contribution
    return out


def make_quantized(n_out: int, K: int, bits: int, dtype=mx.float16, seed: int = 0):
    rng = np.random.default_rng(seed)
    w = mx.array(rng.standard_normal((n_out, K)).astype(np.float32) * 0.05, dtype=dtype)
    packed, scales, biases = mx.quantize(
        w, group_size=GROUP_SIZE, bits=bits, mode="affine"
    )
    mx.eval(packed, scales, biases)
    return packed, scales, biases


# --------------------------------------------------------------------------
# 1. Decoder
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [4, 5])
def test_unpack_matches_mlx_dequantize(bits):
    """Tie the reference unpacker to MLX's affine semantics.

    If this holds, the decoded codes really are the codes MLX would have used,
    so the kernel comparisons below are anchored to the checkpoint format and
    not just to the test's own idea of it.
    """
    K = 256
    packed, scales, biases = make_quantized(64, K, bits)
    codes = unpack_codes(np.array(packed), bits, K)

    assert codes.min() >= 0
    assert codes.max() <= (1 << bits) - 1

    s = np.array(scales.astype(mx.float32))
    b = np.array(biases.astype(mx.float32))
    expected = np.repeat(s, GROUP_SIZE, axis=1) * codes + np.repeat(
        b, GROUP_SIZE, axis=1
    )
    got = np.array(
        mx.dequantize(
            packed, scales, biases, group_size=GROUP_SIZE, bits=bits, mode="affine"
        ).astype(mx.float32)
    )
    np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("bits", [4, 5])
def test_group_words(bits):
    """Q4 uses 8 uint32 per group and Q5 uses 10 -- both word-aligned."""
    K = 512
    packed, _, _ = make_quantized(64, K, bits)
    words_per_group = (GROUP_SIZE * bits) // 32
    assert packed.shape[1] == (K // GROUP_SIZE) * words_per_group


@requires_kernels
@pytest.mark.parametrize("bits", [4, 5])
def test_kernel_decode_is_bit_exact(bits):
    """Every decoded code matches the reference unpacker exactly."""
    fast = _kernels()
    K = 512
    packed, _, _ = make_quantized(96, K, bits, seed=bits)

    decoded = fast.qwen35_oq_a8_decode_weights(packed, bits, K // GROUP_SIZE)
    mx.eval(decoded)

    expected = unpack_codes(np.array(packed), bits, K)
    np.testing.assert_array_equal(np.array(decoded).astype(np.int64), expected)


# --------------------------------------------------------------------------
# 2. Stage A
# --------------------------------------------------------------------------


@requires_kernels
@pytest.mark.parametrize("act_mode", [0, 1])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_stage_a_matches_reference(act_mode, dtype):
    fast = _kernels()
    rows, K = 96, 256
    rng = np.random.default_rng(7)
    x_np = (rng.standard_normal((rows, K)) * 0.7).astype(np.float32)
    x = mx.array(x_np, dtype=dtype)

    qa, sa, ra = fast.qwen35_oq_a8_quantize(x, act_mode)
    mx.eval(qa, sa, ra)

    # Quantize from the same rounded input the kernel saw, so the comparison
    # isolates the kernel rather than the host-side cast.
    x_ref = np.array(x.astype(mx.float32))
    qa_ref, sa_ref, ra_ref = quantize_activations(x_ref, act_mode)

    np.testing.assert_array_equal(np.array(qa), qa_ref)
    np.testing.assert_array_equal(np.array(ra), ra_ref)
    np.testing.assert_allclose(
        np.array(sa).reshape(sa_ref.shape), sa_ref, rtol=1e-6, atol=1e-8
    )


@requires_kernels
@pytest.mark.parametrize("act_mode", [0, 1])
def test_stage_a_group_sum_is_exactly_the_code_sum(act_mode):
    """Ra must come from the rounded codes, not from x."""
    fast = _kernels()
    rows, K = 64, 320
    rng = np.random.default_rng(11)
    x = mx.array((rng.standard_normal((rows, K)) * 2.0).astype(np.float32), mx.float16)

    qa, _, ra = fast.qwen35_oq_a8_quantize(x, act_mode)
    mx.eval(qa, ra)

    codes = np.array(qa).astype(np.int64).reshape(rows, K // GROUP_SIZE, GROUP_SIZE)
    np.testing.assert_array_equal(np.array(ra).astype(np.int64), codes.sum(axis=2))


@requires_kernels
def test_stage_a_zero_row_is_not_nan():
    """An all-zero row has no representable scale; it must not produce NaN."""
    fast = _kernels()
    x = mx.zeros((64, 128), dtype=mx.float16)
    qa, sa, ra = fast.qwen35_oq_a8_quantize(x, 0)
    mx.eval(qa, sa, ra)

    assert not np.isnan(np.array(sa)).any()
    assert not np.array(qa).any()
    assert not np.array(ra).any()


# --------------------------------------------------------------------------
# 3. GEMM
# --------------------------------------------------------------------------


@requires_kernels
@pytest.mark.parametrize("bits", [4, 5])
@pytest.mark.parametrize("act_mode", [0, 1])
def test_qmm_matches_affine_reference(bits, act_mode):
    """FP32 affine accumulation against an independent implementation.

    The integer dot inside is exact, so any disagreement beyond FP32 rounding
    means the group accumulator or the correction is wrong.
    """
    fast = _kernels()
    M, K, N = 128, 256, 128
    packed, scales, biases = make_quantized(N, K, bits, seed=bits + 3)

    rng = np.random.default_rng(23)
    x = mx.array((rng.standard_normal((M, K)) * 0.5).astype(np.float32), mx.float16)

    qa, sa, ra = fast.qwen35_oq_a8_quantize(x, act_mode)
    got = fast.qwen35_oq_a8_qmm_t(
        qa, sa, ra, packed, scales, biases, bits, act_mode, 0
    )
    mx.eval(qa, sa, ra, got)

    expected = affine_reference(
        np.array(qa).astype(np.int64),
        np.array(sa).reshape(M, -1).squeeze() if act_mode == 0 else np.array(sa),
        np.array(ra).astype(np.int64),
        unpack_codes(np.array(packed), bits, K),
        np.array(scales.astype(mx.float32)),
        np.array(biases.astype(mx.float32)),
        act_mode,
    )

    got_np = np.array(got.astype(mx.float32))
    scale = max(np.abs(expected).max(), 1e-6)
    np.testing.assert_allclose(got_np, expected, rtol=2e-2, atol=3e-3 * scale)


@requires_kernels
@pytest.mark.parametrize("bits", [4, 5])
def test_qmm_tracks_the_unquantized_projection(bits):
    """Accuracy sanity: A8 changes numerics but must not change the answer.

    This is not the end-to-end accuracy gate -- that one runs on the model,
    not on a single GEMM -- but a kernel that is merely self-consistent and
    wrong would pass the reference test above and fail here.
    """
    fast = _kernels()
    M, K, N = 128, 512, 128
    packed, scales, biases = make_quantized(N, K, bits, seed=bits + 9)

    rng = np.random.default_rng(31)
    x = mx.array((rng.standard_normal((M, K)) * 0.5).astype(np.float32), mx.float16)

    got = fast.qwen35_oq_a8_linear(x, packed, scales, biases, bits, 0, 0)
    reference = mx.quantized_matmul(
        x,
        packed,
        scales,
        biases,
        transpose=True,
        group_size=GROUP_SIZE,
        bits=bits,
        mode="affine",
    )
    mx.eval(got, reference)

    got_np = np.array(got.astype(mx.float32))
    ref_np = np.array(reference.astype(mx.float32))
    rel = np.abs(got_np - ref_np).max() / max(np.abs(ref_np).max(), 1e-6)
    assert rel < 0.05, f"A8 projection drifted {rel:.4f} from the W{bits}A16 result"


@requires_kernels
@pytest.mark.parametrize("variant", [0, 2, 5, 6])
def test_qmm_variants_agree(variant):
    """Every tile must compute the same thing; only speed may differ."""
    fast = _kernels()
    M, K, N = 200, 256, 128
    packed, scales, biases = make_quantized(N, K, 4, seed=5)
    rng = np.random.default_rng(41)
    x = mx.array((rng.standard_normal((M, K)) * 0.5).astype(np.float32), mx.float16)

    qa, sa, ra = fast.qwen35_oq_a8_quantize(x, 0)
    base = fast.qwen35_oq_a8_qmm_t(qa, sa, ra, packed, scales, biases, 4, 0, 0)
    other = fast.qwen35_oq_a8_qmm_t(qa, sa, ra, packed, scales, biases, 4, 0, variant)
    mx.eval(base, other)

    np.testing.assert_allclose(
        np.array(base.astype(mx.float32)),
        np.array(other.astype(mx.float32)),
        rtol=1e-3,
        atol=1e-3,
    )


@requires_kernels
def test_qmm_handles_a_partial_row_tile():
    """M is the token count and need not tile; N is checked host-side."""
    fast = _kernels()
    M, K, N = 130, 128, 64  # 130 is not a multiple of any BM
    packed, scales, biases = make_quantized(N, K, 4, seed=13)
    rng = np.random.default_rng(53)
    x = mx.array((rng.standard_normal((M, K)) * 0.5).astype(np.float32), mx.float16)

    got = fast.qwen35_oq_a8_linear(x, packed, scales, biases, 4, 0, 0)
    mx.eval(got)
    assert got.shape == (M, N)
    assert not np.isnan(np.array(got.astype(mx.float32))).any()


@requires_kernels
def test_qmm_rejects_untiled_output_width():
    fast = _kernels()
    K, N = 128, 96  # 96 is not a multiple of BN=64
    packed, scales, biases = make_quantized(N, K, 4, seed=17)
    x = mx.zeros((64, K), dtype=mx.float16)
    qa, sa, ra = fast.qwen35_oq_a8_quantize(x, 0)
    with pytest.raises(ValueError):
        fast.qwen35_oq_a8_qmm_t(qa, sa, ra, packed, scales, biases, 4, 0, 0)


# --------------------------------------------------------------------------
# Q4 decode-free path
# --------------------------------------------------------------------------


def test_nibble_flip_maps_unsigned_codes_to_signed():
    """(q ^ 8) read as two's-complement 4-bit is exactly q - 8.

    This identity is the whole basis of the decode-free path: it lets the
    hardware's signed int4b_format consume oQ's unsigned affine codes, with the
    offset absorbed into the bias.
    """
    for q in range(16):
        flipped = q ^ 8
        signed = flipped - 16 if flipped >= 8 else flipped
        assert signed == q - 8, f"q={q}: {signed} != {q - 8}"


@requires_kernels
@pytest.mark.parametrize("act_mode", [0, 1])
def test_i4_matches_the_decoding_kernel(act_mode):
    """The decode-free Q4 path must agree with the decoder path exactly.

    Same inputs, same math, different route to the tensor units -- so this is
    the tightest available check on the bit flip and the folded bias.
    """
    fast = _kernels()
    M, K, N = 128, 512, 128
    packed, scales, biases = make_quantized(N, K, 4, seed=77)

    rng = np.random.default_rng(97)
    x = mx.array((rng.standard_normal((M, K)) * 0.5).astype(np.float32), mx.float16)

    qa, sa, ra = fast.qwen35_oq_a8_quantize(x, act_mode)
    decoded = fast.qwen35_oq_a8_qmm_t(
        qa, sa, ra, packed, scales, biases, 4, act_mode, 0
    )

    w4, s4, b4 = fast.qwen35_oq_a8_prepare_q4(packed, scales, biases)
    direct = fast.qwen35_oq_a8_i4_qmm_t(qa, sa, ra, w4, s4, b4, act_mode, 0)
    mx.eval(decoded, direct)

    a = np.array(decoded.astype(mx.float32))
    b = np.array(direct.astype(mx.float32))
    rel = np.abs(a - b).max() / max(np.abs(a).max(), 1e-6)
    assert rel < 2e-3, f"decode-free Q4 diverged from the decoder path by {rel:.5f}"


@requires_kernels
def test_i4_prepare_is_a_pure_relabelling():
    """The flip and bias fold must not change what the weights mean."""
    fast = _kernels()
    K, N = 256, 64
    packed, scales, biases = make_quantized(N, K, 4, seed=5)
    w4, s4, b4 = fast.qwen35_oq_a8_prepare_q4(packed, scales, biases)
    mx.eval(w4, s4, b4)

    codes = unpack_codes(np.array(packed), 4, K)
    flipped = unpack_codes(np.array(w4), 4, K)
    signed = np.where(flipped >= 8, flipped - 16, flipped)
    np.testing.assert_array_equal(signed, codes - 8)

    # Sw*q + Bw must equal Sw*(q - 8) + Bw'
    s = np.repeat(np.array(scales.astype(mx.float32)), GROUP_SIZE, axis=1)
    b = np.repeat(np.array(biases.astype(mx.float32)), GROUP_SIZE, axis=1)
    b_folded = np.repeat(np.array(b4.astype(mx.float32)), GROUP_SIZE, axis=1)
    np.testing.assert_allclose(
        s * codes + b, s * (codes - 8) + b_folded, rtol=1e-2, atol=1e-2
    )


@requires_kernels
@pytest.mark.parametrize("variant", [0, 1, 3])
def test_i4_variants_agree(variant):
    fast = _kernels()
    M, K, N = 192, 256, 128
    packed, scales, biases = make_quantized(N, K, 4, seed=8)
    w4, s4, b4 = fast.qwen35_oq_a8_prepare_q4(packed, scales, biases)
    rng = np.random.default_rng(19)
    x = mx.array((rng.standard_normal((M, K)) * 0.5).astype(np.float32), mx.float16)

    qa, sa, ra = fast.qwen35_oq_a8_quantize(x, 0)
    base = fast.qwen35_oq_a8_i4_qmm_t(qa, sa, ra, w4, s4, b4, 0, 0)
    other = fast.qwen35_oq_a8_i4_qmm_t(qa, sa, ra, w4, s4, b4, 0, variant)
    mx.eval(base, other)

    np.testing.assert_allclose(
        np.array(base.astype(mx.float32)),
        np.array(other.astype(mx.float32)),
        rtol=1e-3,
        atol=1e-3,
    )


@requires_kernels
def test_i4_tracks_the_unquantized_projection():
    fast = _kernels()
    M, K, N = 128, 512, 128
    packed, scales, biases = make_quantized(N, K, 4, seed=23)
    w4, s4, b4 = fast.qwen35_oq_a8_prepare_q4(packed, scales, biases)

    rng = np.random.default_rng(29)
    x = mx.array((rng.standard_normal((M, K)) * 0.5).astype(np.float32), mx.float16)

    got = fast.qwen35_oq_a8_i4_linear(x, w4, s4, b4, 0, 0)
    reference = mx.quantized_matmul(
        x, packed, scales, biases, transpose=True,
        group_size=GROUP_SIZE, bits=4, mode="affine",
    )
    mx.eval(got, reference)

    g = np.array(got.astype(mx.float32))
    r = np.array(reference.astype(mx.float32))
    rel = np.abs(g - r).max() / max(np.abs(r).max(), 1e-6)
    assert rel < 0.05, f"decode-free Q4 drifted {rel:.4f} from the W4A16 result"


@requires_kernels
def test_i4_handles_a_partial_row_tile():
    fast = _kernels()
    M, K, N = 130, 128, 64
    packed, scales, biases = make_quantized(N, K, 4, seed=31)
    w4, s4, b4 = fast.qwen35_oq_a8_prepare_q4(packed, scales, biases)
    rng = np.random.default_rng(37)
    x = mx.array((rng.standard_normal((M, K)) * 0.5).astype(np.float32), mx.float16)

    got = fast.qwen35_oq_a8_i4_linear(x, w4, s4, b4, 0, 0)
    mx.eval(got)
    assert got.shape == (M, N)
    assert not np.isnan(np.array(got.astype(mx.float32))).any()


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------


def _quantized_linear(in_dim, out_dim, bits, group_size=GROUP_SIZE):
    """A QuantizedLinear shaped like a real checkpoint layer.

    nn.QuantizedLinear defaults to float32 scales; checkpoints store them at
    the activation dtype. Without the cast the classifier correctly rejects the
    module and every dispatch assertion below would pass vacuously.
    """
    import mlx.nn as nn

    linear = nn.QuantizedLinear(
        in_dim, out_dim, bias=False, group_size=group_size, bits=bits
    )
    linear.set_dtype(mx.float16)
    return linear


@requires_kernels
def test_classification_is_frozen_and_memoized():
    """The forward path must never re-parse quantization config."""
    from omlx.patches import qwen35_oq_a8 as dispatch

    linear = _quantized_linear(256, 128, 4)
    first = dispatch.classify_linear(linear)
    second = dispatch.classify_linear(linear)
    assert first is second
    assert first is not None, "a checkpoint-shaped Q4 GS64 layer must classify"
    assert first.bits == 4
    assert first.group_size == 64
    assert first.kernel == "q4a8_g64"


@requires_kernels
def test_float32_scales_are_not_claimed():
    """Only checkpoint-dtype scales route here; float32 stays on MLX."""
    import mlx.nn as nn

    from omlx.patches import qwen35_oq_a8 as dispatch

    linear = nn.QuantizedLinear(256, 128, bias=False, group_size=64, bits=4)
    assert linear.scales.dtype == mx.float32
    assert dispatch.classify_linear(linear) is None


@requires_kernels
@pytest.mark.parametrize("variant", [800, 803, 806])
def test_dispatch_runs_and_tracks_the_original(monkeypatch, variant):
    """oq_a8_linear must actually route to the kernel, not fall back."""
    from omlx.patches import qwen35_oq_a8 as dispatch

    monkeypatch.setenv("OMLX_OQ_A8", "1")
    monkeypatch.setenv("OMLX_OQ_A8_VARIANT", str(variant))
    monkeypatch.setenv("OMLX_OQ_A8_MIN_TOKENS", "64")
    assert dispatch.enabled()

    linear = _quantized_linear(512, 128, 4)
    plan = dispatch.classify_linear(linear)
    assert plan is not None and plan.variant == variant

    rng = np.random.default_rng(3)
    x = mx.array((rng.standard_normal((128, 512)) * 0.5).astype(np.float32), mx.float16)
    got = dispatch.oq_a8_linear(linear, x)
    reference = linear(x)
    mx.eval(got, reference)

    assert hasattr(linear, "_omlx_oq_a8_prepared"), "operand transform not cached"

    g = np.array(got.astype(mx.float32))
    r = np.array(reference.astype(mx.float32))
    # A8 changes numerics, so this must differ from W4A16 -- but only a little.
    # An exact match would mean the dispatcher silently fell back.
    rel = np.abs(g - r).max() / max(np.abs(r).max(), 1e-6)
    assert 0.0 < rel < 0.05, f"dispatch did not reach the kernel (rel={rel})"


@pytest.mark.parametrize("bits", [2, 6, 8])
def test_unsupported_bit_widths_are_not_claimed(bits):
    """Only Q4 and Q5 are production paths here; the rest stay on MLX."""
    from omlx.patches import qwen35_oq_a8 as dispatch

    assert dispatch.classify_linear(_quantized_linear(256, 128, bits)) is None


def test_group_size_128_is_not_claimed():
    from omlx.patches import qwen35_oq_a8 as dispatch

    linear = _quantized_linear(256, 128, 4, group_size=128)
    assert dispatch.classify_linear(linear) is None


def test_disabled_without_opt_in(monkeypatch):
    """Turning this on changes inference numerics, so it must be explicit."""
    from omlx.patches import qwen35_oq_a8 as dispatch

    monkeypatch.delenv("OMLX_OQ_A8", raising=False)
    assert dispatch.enabled() is False


# --------------------------------------------------------------------------
# Step-transposed GEMM (variants >= 800)
# --------------------------------------------------------------------------


def _v8_act_from():
    """Fragment slot -> K within an affine group, derived from the schedule.

    Slot ``16c + 4t + j`` of the group carries ``k = 16c + 8*(t>>1) + 2j +
    (t&1)``: the assignment under which a Q4 step is one nibble parity of one
    word of the untouched checkpoint stream.
    """
    a_from = np.empty(GROUP_SIZE, dtype=np.int64)
    for c in range(4):
        for t in range(4):
            for j in range(4):
                a_from[16 * c + 4 * t + j] = 16 * c + 8 * (t >> 1) + 2 * j + (t & 1)
    return a_from


@requires_kernels
@pytest.mark.parametrize("act_mode", [0, 1])
def test_stage_a_v8_carries_the_schedules_k_order(act_mode):
    """Qa is the only operand that moves, and it moves by exactly this map."""
    fast = _kernels()
    M, K = 33, 256
    rng = np.random.default_rng(21)
    x = mx.array((rng.standard_normal((M, K)) * 0.5).astype(np.float32), mx.float16)
    mx.eval(x)

    qa, sa, ra = fast.qwen35_oq_a8_quantize(x, act_mode)
    qa8, sa8, ra8 = fast.qwen35_oq_a8_stage_a_v8(x, act_mode)
    mx.eval(qa, sa, ra, qa8, sa8, ra8)

    groups = K // GROUP_SIZE
    plain = np.array(qa).reshape(M, groups, GROUP_SIZE)
    np.testing.assert_array_equal(
        np.array(qa8).reshape(M, groups, GROUP_SIZE), plain[:, :, _v8_act_from()]
    )
    # Group membership is untouched, which is what keeps Ra and the affine
    # scales valid under the reorder.
    np.testing.assert_array_equal(
        np.sort(plain, axis=2),
        np.sort(np.array(qa8).reshape(M, groups, GROUP_SIZE), axis=2),
    )
    np.testing.assert_array_equal(np.array(ra8), np.array(ra).T)
    expected_sa = np.array(sa).T if act_mode else np.array(sa)
    np.testing.assert_array_equal(np.array(sa8), expected_sa)


@requires_kernels
def test_v8_reads_the_checkpoint_weight_stream_unchanged():
    """The whole point: no repacked weight array, so nothing is held twice."""
    from omlx.patches import qwen35_oq_a8 as dispatch

    linear = _quantized_linear(512, 128, 4)
    assert dispatch.classify_linear(linear) is not None
    weight, scales, biases = dispatch._prepared_weights(linear)
    assert weight is linear.weight, "the packed stream must be reused, not copied"
    np.testing.assert_array_equal(np.array(scales), np.array(linear.scales).T)
    np.testing.assert_array_equal(np.array(biases), np.array(linear.biases).T)


@requires_kernels
@pytest.mark.parametrize("bits", [4, 5])
@pytest.mark.parametrize("act_mode", [0, 1])
@pytest.mark.parametrize("M", [1, 17, 128, 513])
def test_v8_matches_w4a16_within_quantization_error(bits, act_mode, M):
    """End to end off the unmodified checkpoint layout."""
    fast = _kernels()
    K, N = 320, 128
    rng = np.random.default_rng(hash((bits, act_mode, M, 8)) % 2**31)
    w = mx.array(rng.standard_normal((N, K)).astype(np.float32) * 0.05, mx.float16)
    packed, scales, biases = mx.quantize(
        w, group_size=GROUP_SIZE, bits=bits, mode="affine"
    )
    x = mx.array((rng.standard_normal((M, K)) * 0.5).astype(np.float32), mx.float16)
    mx.eval(packed, scales, biases, x)

    sc_t = mx.contiguous(scales.T)
    bi_t = mx.contiguous(biases.T)
    qa, sa, ra = fast.qwen35_oq_a8_stage_a_v8(x, act_mode)
    mx.eval(sc_t, bi_t, qa, sa, ra)

    got = fast.qwen35_oq_a8_qmm_t(
        qa, sa, ra, packed, sc_t, bi_t, bits, act_mode, 800
    )
    ref = mx.quantized_matmul(
        x, packed, scales, biases, transpose=True,
        group_size=GROUP_SIZE, bits=bits, mode="affine",
    )
    mx.eval(got, ref)
    g = np.array(got.astype(mx.float32))
    r = np.array(ref.astype(mx.float32))
    assert np.isfinite(g).all()
    rel = np.abs(g - r).max() / max(np.abs(r).max(), 1e-6)
    assert 0.0 < rel < 0.06, f"rel={rel}"


@requires_kernels
@pytest.mark.parametrize("variant", [800, 801, 802, 803, 804, 805, 806])
@pytest.mark.parametrize("bits", [4, 5])
def test_v8_tiles_agree_with_each_other(variant, bits):
    """Every tile computes the same thing; only the simdgroup grid differs."""
    fast = _kernels()
    M, K, N = 96, 256, 128
    rng = np.random.default_rng(5)
    w = mx.array(rng.standard_normal((N, K)).astype(np.float32) * 0.05, mx.float16)
    packed, scales, biases = mx.quantize(
        w, group_size=GROUP_SIZE, bits=bits, mode="affine"
    )
    x = mx.array((rng.standard_normal((M, K)) * 0.5).astype(np.float32), mx.float16)
    mx.eval(packed, scales, biases, x)
    sc_t = mx.contiguous(scales.T)
    bi_t = mx.contiguous(biases.T)
    qa, sa, ra = fast.qwen35_oq_a8_stage_a_v8(x, 0)
    mx.eval(sc_t, bi_t, qa, sa, ra)

    base = fast.qwen35_oq_a8_qmm_t(qa, sa, ra, packed, sc_t, bi_t, bits, 0, 800)
    got = fast.qwen35_oq_a8_qmm_t(qa, sa, ra, packed, sc_t, bi_t, bits, 0, variant)
    mx.eval(base, got)
    np.testing.assert_allclose(
        np.array(got.astype(mx.float32)), np.array(base.astype(mx.float32)),
        rtol=0, atol=0,
    )


@requires_kernels
@pytest.mark.parametrize("act_mode", [0, 1])
def test_stage_a_keeps_the_batch_rank(act_mode):
    """The op derives the output shape from Qa, so its rank has to survive.

    A flattened [B*S, K] Qa yields a [B*S, N] result. At B == 1 that
    broadcasts against the residual and hides.
    """
    fast = _kernels()
    rng = np.random.default_rng(31)
    x = mx.array((rng.standard_normal((3, 128, 256)) * 0.5).astype(np.float32),
                 mx.float16)
    mx.eval(x)
    qa, _, _ = fast.qwen35_oq_a8_stage_a_v8(x, act_mode)
    mx.eval(qa)
    assert qa.shape == x.shape


@requires_kernels
@pytest.mark.parametrize("act_mode", [0, 1])
def test_stage_a_group_major_metadata_is_not_axis_reversed(act_mode):
    """Ra must come back [K/64, M], not [K/64, S, B].

    mx.transpose reverses every axis, so transposing a 3-D Ra in place gives
    the layout the kernel wants only when B == 1 and interleaves the
    sequences when it is not.
    """
    fast = _kernels()
    B, S, K = 3, 128, 256
    groups = K // GROUP_SIZE
    rng = np.random.default_rng(32)
    x = mx.array((rng.standard_normal((B, S, K)) * 0.5).astype(np.float32), mx.float16)
    mx.eval(x)

    _, sa_flat, ra_flat = fast.qwen35_oq_a8_quantize(x, act_mode)
    _, sa, ra = fast.qwen35_oq_a8_stage_a_v8(x, act_mode)
    mx.eval(sa_flat, ra_flat, sa, ra)

    expected = np.array(ra_flat).reshape(B * S, groups).T
    np.testing.assert_array_equal(np.array(ra), expected)
    if act_mode != 0:
        np.testing.assert_array_equal(
            np.array(sa), np.array(sa_flat).reshape(B * S, groups).T
        )


@requires_kernels
@pytest.mark.parametrize("B", [1, 2, 3])
def test_batched_prefill_keeps_sequences_independent(B):
    """End to end at B > 1: row b of the output depends only on row b of x."""
    fast = _kernels()
    S, K, N = 96, 256, 128
    rng = np.random.default_rng(33)
    w = mx.array(rng.standard_normal((N, K)).astype(np.float32) * 0.05, mx.float16)
    packed, scales, biases = mx.quantize(
        w, group_size=GROUP_SIZE, bits=4, mode="affine"
    )
    x = mx.array((rng.standard_normal((B, S, K)) * 0.5).astype(np.float32), mx.float16)
    mx.eval(packed, scales, biases, x)
    sc_t, bi_t = mx.contiguous(scales.T), mx.contiguous(biases.T)
    mx.eval(sc_t, bi_t)

    def run(inp):
        qa, sa, ra = fast.qwen35_oq_a8_stage_a_v8(inp, 0)
        mx.eval(qa, sa, ra)
        out = fast.qwen35_oq_a8_qmm_t(qa, sa, ra, packed, sc_t, bi_t, 4, 0, 806)
        mx.eval(out)
        return out

    got = run(x)
    assert got.shape == (B, S, N)
    ref = mx.quantized_matmul(x, packed, scales, biases, transpose=True,
                              group_size=GROUP_SIZE, bits=4, mode="affine")
    mx.eval(ref)
    g = np.array(got.astype(mx.float32))
    r = np.array(ref.astype(mx.float32))
    rel = np.abs(g - r).max() / max(np.abs(r).max(), 1e-6)
    assert rel < 0.06, f"rel={rel}"

    # Batching must not change any one sequence's result.
    for b in range(B):
        alone = np.array(run(x[b : b + 1]).astype(mx.float32))[0]
        np.testing.assert_array_equal(alone, g[b])


@requires_kernels
def test_short_sequences_stay_on_the_existing_path(monkeypatch):
    """Below the token floor the Stage-A pass costs more than the GEMM saves."""
    from omlx.patches import qwen35_oq_a8 as dispatch

    monkeypatch.setenv("OMLX_OQ_A8", "1")
    monkeypatch.setenv("OMLX_OQ_A8_MIN_TOKENS", "512")
    linear = _quantized_linear(512, 128, 4)
    rng = np.random.default_rng(9)
    x = mx.array((rng.standard_normal((64, 512)) * 0.5).astype(np.float32), mx.float16)
    got = dispatch.oq_a8_linear(linear, x)
    mx.eval(got)
    # Bit-identical to the unrouted call, i.e. it really did fall through.
    np.testing.assert_array_equal(
        np.array(got.astype(mx.float32)), np.array(linear(x).astype(mx.float32))
    )


def test_out_of_family_variants_are_rejected():
    """A variant outside the shipped family has to fail where it enters.

    The number arrives from an environment variable, so an out-of-range one
    otherwise reaches the op as a missing kernel name deep in dispatch.
    """
    from omlx.patches import qwen35_oq_a8 as dispatch

    for variant in (800, 803, 806):
        assert dispatch.check_variant(variant) == variant
    for variant in (0, 6, 206, 799, 807, -1):
        with pytest.raises(ValueError, match="not a shipped kernel"):
            dispatch.check_variant(variant)


@requires_kernels
def test_an_unusable_variant_leaves_the_projection_alone(monkeypatch):
    """Refusing to classify keeps the model on MLX rather than crashing it."""
    from omlx.patches import qwen35_oq_a8 as dispatch

    monkeypatch.setenv("OMLX_OQ_A8", "1")
    monkeypatch.setenv("OMLX_OQ_A8_VARIANT", "807")
    linear = _quantized_linear(512, 128, 4)
    assert dispatch.classify_linear(linear) is None

    rng = np.random.default_rng(4)
    x = mx.array((rng.standard_normal((128, 512)) * 0.5).astype(np.float32), mx.float16)
    got = dispatch.oq_a8_linear(linear, x)
    mx.eval(got)
    np.testing.assert_array_equal(
        np.array(got.astype(mx.float32)), np.array(linear(x).astype(mx.float32))
    )


def test_patch_is_idempotent_and_reports_state():
    """Installing twice must not stack wrappers on the MLP class."""
    from omlx.patches import qwen35_oq_a8 as dispatch

    if not dispatch._kernels_available():
        pytest.skip("oQ A8 kernels unavailable")
    first = dispatch.apply_qwen35_oq_a8_patch()
    second = dispatch.apply_qwen35_oq_a8_patch()
    assert first == second


@requires_kernels
def test_patched_mlp_routes_and_falls_back(monkeypatch):
    """The installed wrapper must route long prompts and pass short ones through."""
    import mlx.nn as nn

    from omlx.patches import qwen35_oq_a8 as dispatch

    monkeypatch.setenv("OMLX_OQ_A8", "1")
    monkeypatch.setenv("OMLX_OQ_A8_MIN_TOKENS", "128")

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = _quantized_linear(256, 512, 4)
            self.up_proj = _quantized_linear(256, 512, 4)
            self.down_proj = _quantized_linear(512, 256, 4)

        def __call__(self, x, *args, **kwargs):
            return self.down_proj(
                nn.silu(self.gate_proj(x)) * self.up_proj(x)
            )

    monkeypatch.setattr(dispatch, "_SWIGLU", lambda g, u: nn.silu(g) * u)

    mlp = MLP()
    orig = MLP.__call__
    MLP.__call__ = dispatch._make_patched_mlp(orig)

    rng = np.random.default_rng(21)
    long_x = mx.array(
        (rng.standard_normal((1, 256, 256)) * 0.4).astype(np.float32), mx.float16
    )
    short_x = long_x[:, :64, :]
    mx.eval(long_x, short_x)

    routed = mlp(long_x)
    plain = orig(mlp, long_x)
    mx.eval(routed, plain)
    rel = np.abs(
        np.array(routed.astype(mx.float32)) - np.array(plain.astype(mx.float32))
    ).max() / max(np.abs(np.array(plain.astype(mx.float32))).max(), 1e-6)
    assert 0.0 < rel < 0.08, f"long prompt did not route (rel={rel})"

    # Below the floor it must be the original computation, bit for bit.
    np.testing.assert_array_equal(
        np.array(mlp(short_x).astype(mx.float32)),
        np.array(orig(mlp, short_x).astype(mx.float32)),
    )

    # target_verify forwards are decode-shaped and must never route.
    np.testing.assert_array_equal(
        np.array(mlp(long_x, target_verify=True).astype(mx.float32)),
        np.array(plain.astype(mx.float32)),
    )


@requires_kernels
def test_patch_takes_its_configuration_from_the_caller(monkeypatch):
    """The engine passes the model's settings in, rather than via the process
    environment, so two engines cannot silently reconfigure each other."""
    import mlx.nn as nn

    from omlx.patches import qwen35_oq_a8 as dispatch

    monkeypatch.delenv("OMLX_OQ_A8", raising=False)
    monkeypatch.delenv("OMLX_OQ_A8_VARIANT", raising=False)
    monkeypatch.delenv("OMLX_OQ_A8_MIN_TOKENS", raising=False)

    assert dispatch._variant_for_bits(4) == dispatch._DEFAULT_VARIANT_Q4

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = _quantized_linear(512, 128, 4)

    model = Model()
    dispatch.apply_qwen35_oq_a8_patch(model, min_tokens=2048)
    config = dispatch._config_for(model.proj)
    assert config is not None and dispatch._min_tokens(config) == 2048

    # An untouched model keeps the default, i.e. the floor rides on the model
    # rather than on the process.
    assert dispatch._min_tokens(dispatch._ENV_CONFIG) == dispatch._MIN_TOKENS_DEFAULT

    # Q4 and Q5 are tuned independently; the tile is not a user setting.
    assert dispatch._variant_for_bits(4) == dispatch._DEFAULT_VARIANT_Q4
    assert dispatch._variant_for_bits(5) == dispatch._DEFAULT_VARIANT_Q5

    # The environment still wins, for benchmarking.
    monkeypatch.setenv("OMLX_OQ_A8_VARIANT", "801")
    assert dispatch._variant_for_bits(4) == 801
    monkeypatch.setenv("OMLX_OQ_A8_MIN_TOKENS", "64")
    assert dispatch._min_tokens(config) == 64


@requires_kernels
def test_turning_the_setting_off_actually_stops_routing(monkeypatch):
    """Enable, then reload with the setting off: the second model must run on
    MLX.

    The patch replaces Qwen3_5MLP.__call__ process-wide and never removes it,
    so a process-wide "enabled" flag would stay stuck on and the reload would
    keep routing at full speed. Opt-in lives on the model's own modules
    instead, and an untagged model falls straight through.
    """
    import mlx.nn as nn

    from omlx.patches import qwen35_oq_a8 as dispatch

    monkeypatch.delenv("OMLX_OQ_A8", raising=False)
    monkeypatch.setenv("OMLX_OQ_A8_MIN_TOKENS", "64")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = _quantized_linear(512, 128, 4)

    on = Model()
    dispatch.apply_qwen35_oq_a8_patch(on, min_tokens=64)

    # A second model loaded with the setting off is simply never tagged.
    off = Model()

    rng = np.random.default_rng(11)
    x = mx.array((rng.standard_normal((128, 512)) * 0.5).astype(np.float32), mx.float16)

    routed = dispatch.oq_a8_linear(on.proj, x)
    untouched = dispatch.oq_a8_linear(off.proj, x)
    reference_on = on.proj(x)
    reference_off = off.proj(x)
    mx.eval(routed, untouched, reference_on, reference_off)

    # The tagged model reached the kernel: close, but not bit-identical.
    rel = np.abs(
        np.array(routed.astype(mx.float32)) - np.array(reference_on.astype(mx.float32))
    ).max() / max(np.abs(np.array(reference_on.astype(mx.float32))).max(), 1e-6)
    assert 0.0 < rel < 0.05, f"tagged model did not reach the kernel (rel={rel})"

    # The untagged one is bit-identical to MLX, i.e. it never left that path.
    np.testing.assert_array_equal(
        np.array(untouched.astype(mx.float32)),
        np.array(reference_off.astype(mx.float32)),
    )
    assert dispatch._config_for(off.proj) is None


@requires_kernels
def test_two_resident_models_keep_their_own_settings(monkeypatch):
    """The wrapper is shared, so the floors must not be last-writer-wins."""
    import mlx.nn as nn

    from omlx.patches import qwen35_oq_a8 as dispatch

    monkeypatch.delenv("OMLX_OQ_A8", raising=False)
    monkeypatch.delenv("OMLX_OQ_A8_MIN_TOKENS", raising=False)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = _quantized_linear(512, 128, 4)

    short, long = Model(), Model()
    dispatch.apply_qwen35_oq_a8_patch(short, min_tokens=64)
    dispatch.apply_qwen35_oq_a8_patch(long, min_tokens=4096)

    assert dispatch._min_tokens(dispatch._config_for(short.proj)) == 64
    assert dispatch._min_tokens(dispatch._config_for(long.proj)) == 4096
