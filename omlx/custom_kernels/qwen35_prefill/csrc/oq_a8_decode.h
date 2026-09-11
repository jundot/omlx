// Register-resident decoders for the oQ mixed-bit (Q4/Q5, GS64, affine)
// checkpoint layout, shared by the Stage-A/reference kernels and the NAX
// INT8 GEMM.
//
// The checkpoint stores packed codes in uint32 words exactly as MLX's affine
// QuantizedLinear does, so a row of K weights occupies K * BITS / 32 words and
// oMLX's existing `weight.shape[1] * 32 == input_dim * bits` validation still
// holds. Nothing here writes decoded weights to device memory: the GEMM calls
// oq_decode4() straight into tensor-op fragment registers.
//
// GS64 makes the geometry unusually friendly:
//
//     Q4: 64 codes * 4 bits = 256 bits = 8 uint32 words
//     Q5: 64 codes * 5 bits = 320 bits = 10 uint32 words
//
// so every affine group starts and ends on a word boundary and no code stream
// crosses a group boundary. A group can therefore be decoded in isolation.

#pragma once

#include <metal_stdlib>

namespace omlx {
namespace oq_a8 {

constant constexpr int kGroupSize = 64;

// uint32 words backing one affine group of 64 codes: 8 for Q4, 10 for Q5.
constexpr int oq_group_words(int bits) {
  return (kGroupSize * bits) / 32;
}

// Code mask for a BITS-wide field: 0xf for Q4, 0x1f for Q5.
constexpr uint32_t oq_code_mask(int bits) {
  return (1u << bits) - 1u;
}

// Decode four consecutive codes starting at `c0` within one affine group.
//
// `c0` is always a multiple of 4 in the GEMM: the NAX 16x16 fragment hands
// each lane four adjacent K positions. Bit arithmetic is shifts and masks
// only -- no division or modulo in the hot decoder.
//
// A Q5 field can straddle a word boundary, so the two source words are read
// as a pair. `hi` is only stepped past `lo` when the four fields actually
// reach beyond the first word, which keeps the read inside the group and off
// the end of the weight buffer for the final group of the final row.
// Aligned four-code decode: the same result as oq_decode4() under the extra
// guarantee that `c0` is a multiple of 4, which every GEMM call site meets.
//
// oq_decode4() has to ask, per code, whether the field starts past the end of
// `lo` and whether it straddles the boundary. Neither question can be folded
// away at compile time because the shift comes from the lane's fragment
// column, so the general form costs two selects per code. With c0 % 4 == 0
// the four fields are one contiguous run of 4*BITS bits at a known alignment,
// so the run is extracted once and the codes fall out of it with a shift and
// a mask apiece.
//
//   Q4: 16 bits at a 16-bit boundary -- a single ushort load, nothing else.
//   Q5: 20 bits at a 4-bit boundary  -- two words joined, then one shift.
template <int BITS>
inline void oq_decode4_aligned(
    const device uint32_t* group_words,
    int c0,
    thread int8_t (&out)[4]) {
  uint32_t run;

  if (BITS == 4) {
    run = uint32_t(
        reinterpret_cast<const device ushort*>(group_words)[c0 >> 2]);
  } else {
    const int bit = c0 * BITS;
    const int word = bit >> 5;
    const int sh = bit & 31;
    const uint32_t lo = group_words[word];
    // The run is 4*BITS <= 20 bits wide and starts at `sh <= 28`, so it ends
    // by bit 48: at most one word past `lo`, and only when sh > 12. Reading
    // that word unconditionally would run off the end of the buffer on the
    // last group of the last row, so the step is predicated.
    const uint32_t hi = group_words[word + ((sh + 4 * BITS) > 32 ? 1 : 0)];
    // Shifting by 32 is undefined, so the high part is brought down in two
    // steps; for sh == 0 this leaves it entirely out of the way.
    run = (lo >> sh) | ((hi << (31 - sh)) << 1);
  }

  // extract_bits is a single instruction; a shift and a mask are two. Over a
  // 5120-deep K that difference is 64 instructions per lane per affine group.
#pragma clang loop unroll(full)
  for (int j = 0; j < 4; ++j) {
    out[j] = static_cast<int8_t>(metal::extract_bits(run, j * BITS, BITS));
  }
}

template <int BITS>
inline void oq_decode4(
    const device uint32_t* group_words,
    int c0,
    thread int8_t (&out)[4]) {
  constexpr uint32_t mask = oq_code_mask(BITS);

  const int bit = c0 * BITS;
  const int word = bit >> 5;
  const int sh = bit & 31;

  // Four fields span 4 * BITS bits; only reach for the next word when they
  // cross out of `lo`.
  const bool need_hi = (sh + 4 * BITS) > 32;
  const uint32_t lo = group_words[word];
  const uint32_t hi = group_words[word + (need_hi ? 1 : 0)];

#pragma clang loop unroll(full)
  for (int j = 0; j < 4; ++j) {
    const int b = sh + j * BITS;
    uint32_t v;
    if (b >= 32) {
      v = hi >> (b - 32);
    } else {
      v = lo >> b;
      if (b + BITS > 32) {
        // Metal leaves a shift by the full width undefined; b > 32 - BITS
        // here, so 32 - b is in [1, BITS) and always well defined.
        v |= hi << (32 - b);
      }
    }
    // Affine codes are unsigned (0..15 for Q4, 0..31 for Q5) and so land in
    // signed INT8 unchanged -- no sign correction is needed before the
    // INT8 x INT8 tensor op.
    out[j] = static_cast<int8_t>(v & mask);
  }
}

// Decode a whole affine group with compile-time shifts.
//
// The reference form, and the one the bit-exactness tests compare against
// MLX's dequantize. It is deliberately NOT used by the GEMM: materializing 64
// codes per lane would spill, and writing them to device memory would throw
// away oQ's memory advantage.
template <int BITS>
inline void oq_decode_group(
    const device uint32_t* group_words,
    thread int8_t (&out)[kGroupSize]) {
#pragma clang loop unroll(full)
  for (int c = 0; c < kGroupSize; c += 4) {
    int8_t quad[4];
    oq_decode4<BITS>(group_words, c, quad);
    out[c + 0] = quad[0];
    out[c + 1] = quad[1];
    out[c + 2] = quad[2];
    out[c + 3] = quad[3];
  }
}

// 32 codes is the natural Q5 decode block (5 uint32 -> 32 INT8): 32 * 5 = 160
// bits closes on a word boundary halfway through the group. Provided for the
// tests and for anyone hand-tuning the Q5 path.
inline void oq_decode_q5_32(
    const device uint32_t* half_group_words,
    thread int8_t (&out)[32]) {
#pragma clang loop unroll(full)
  for (int c = 0; c < 32; c += 4) {
    int8_t quad[4];
    oq_decode4<5>(half_group_words, c, quad);
    out[c + 0] = quad[0];
    out[c + 1] = quad[1];
    out[c + 2] = quad[2];
    out[c + 3] = quad[3];
  }
}

} // namespace oq_a8
} // namespace omlx
