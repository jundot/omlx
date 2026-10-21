// Copyright © 2023-2024 Apple Inc.

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];

using namespace metal;

#define MLX_MTL_CONST static constant constexpr const

MLX_MTL_CONST int SIMD_SIZE = 32;
MLX_MTL_CONST int QUAD_SIZE = 4;

template <int bits, int wsize = 8>
inline constexpr short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector(const device T* x, thread U* x_thread) {
  static_assert(
      bits == 1 || bits == 2 || bits == 3 || bits == 4 || bits == 5 ||
          bits == 6 || bits == 8,
      "Template undefined for bits not in {1, 2, 3, 4, 5, 6, 8}");

  U sum = 0;

  if (bits == 1) {
    // Pre-scale x by 1/2^k so qdot can use FMA (x_thread[k] * (wb & 2^k) = x[k] * bit_k).
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i]     = x[i];
      x_thread[i + 1] = x[i + 1] * U(0.5f);
      x_thread[i + 2] = x[i + 2] * U(0.25f);
      x_thread[i + 3] = x[i + 3] * U(0.125f);
      x_thread[i + 4] = x[i + 4] * U(0.0625f);
      x_thread[i + 5] = x[i + 5] * U(0.03125f);
      x_thread[i + 6] = x[i + 6] * U(0.015625f);
      x_thread[i + 7] = x[i + 7] * U(0.0078125f);
    }
  }

  else if (bits == 2) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i]     = x[i];
      x_thread[i + 1] = x[i + 1] * U(0.25f);
      x_thread[i + 2] = x[i + 2] * U(0.0625f);
      x_thread[i + 3] = x[i + 3] * U(0.015625f);
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }

  return sum;
}

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector_safe(const device T* x, thread U* x_thread, int N) {
  static_assert(
      bits == 1 || bits == 2 || bits == 3 || bits == 4 || bits == 5 ||
          bits == 6 || bits == 8,
      "Template undefined for bits not in {1, 2, 3, 4, 5, 6, 8}");

  U sum = 0;

  if (bits == 1) {
    for (int i = 0; i < N; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i]     = x[i];
      x_thread[i + 1] = x[i + 1] * U(0.5f);
      x_thread[i + 2] = x[i + 2] * U(0.25f);
      x_thread[i + 3] = x[i + 3] * U(0.125f);
      x_thread[i + 4] = x[i + 4] * U(0.0625f);
      x_thread[i + 5] = x[i + 5] * U(0.03125f);
      x_thread[i + 6] = x[i + 6] * U(0.015625f);
      x_thread[i + 7] = x[i + 7] * U(0.0078125f);
    }
  }

  else if (bits == 2) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i]     = x[i];
      x_thread[i + 1] = x[i + 1] * U(0.25f);
      x_thread[i + 2] = x[i + 2] * U(0.0625f);
      x_thread[i + 3] = x[i + 3] * U(0.015625f);
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < N; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];

      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < N; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }

  for (int i = N; i < values_per_thread; i++) {
    x_thread[i] = 0;
  }

  return sum;
}

template <typename U, int values_per_thread, int bits>
inline U qdot(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  static_assert(
      bits == 1 || bits == 2 || bits == 3 || bits == 4 || bits == 5 ||
          bits == 6 || bits == 8,
      "Template undefined for bits not in {1, 2, 3, 4, 5, 6, 8}");

  U accum = 0;

  if (bits == 1) {
    // One uint32 load per 32 elements; x_thread is pre-scaled by 1/2^k so
    // x_thread[k] * (wb & 2^k) = x[k] * bit_k — pure FMA, no select/bool.
    const device uint32_t* wp = (const device uint32_t*)w;
    for (int i = 0; i < (values_per_thread / 32); i++) {
      uint32_t wb32 = wp[i];
      for (int j = 0; j < 4; j++) {
        uint8_t wb = uint8_t(wb32 >> (8 * j));
        int base = 32 * i + 8 * j;
        accum += x_thread[base + 0] * U(wb & 0x01);
        accum += x_thread[base + 1] * U(wb & 0x02);
        accum += x_thread[base + 2] * U(wb & 0x04);
        accum += x_thread[base + 3] * U(wb & 0x08);
        accum += x_thread[base + 4] * U(wb & 0x10);
        accum += x_thread[base + 5] * U(wb & 0x20);
        accum += x_thread[base + 6] * U(wb & 0x40);
        accum += x_thread[base + 7] * U(wb & 0x80);
      }
    }
  }

  else if (bits == 2) {
    // One uint32 load covers 16 packed 2-bit values (4× fewer loads than byte-by-byte).
    // x_thread is pre-scaled by 1/4^k so x_thread[k] * (wb & shifted_mask_k) = x[k]*q_k.
    const device uint32_t* wp = (const device uint32_t*)w;
    for (int i = 0; i < (values_per_thread / 16); i++) {
      uint32_t wb32 = wp[i];
      for (int j = 0; j < 4; j++) {
        uint8_t wb = uint8_t(wb32 >> (8 * j));
        int base = 16 * i + 4 * j;
        accum += x_thread[base + 0] * U(wb & 0x03)
              +  x_thread[base + 1] * U(wb & 0x0c)
              +  x_thread[base + 2] * U(wb & 0x30)
              +  x_thread[base + 3] * U(wb & 0xc0);
      }
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

template <typename U, int values_per_thread, int bits>
inline U qdot_safe(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum,
    int N) {
  static_assert(
      bits == 1 || bits == 2 || bits == 3 || bits == 4 || bits == 5 ||
          bits == 6 || bits == 8,
      "Template undefined for bits not in {1, 2, 3, 4, 5, 6, 8}");

  U accum = 0;

  if (bits == 1) {
    for (int i = 0; i < (N / 8); i++) {
      uint8_t wb = w[i];
      accum += x_thread[8 * i + 0] * U(wb & 0x01);
      accum += x_thread[8 * i + 1] * U(wb & 0x02);
      accum += x_thread[8 * i + 2] * U(wb & 0x04);
      accum += x_thread[8 * i + 3] * U(wb & 0x08);
      accum += x_thread[8 * i + 4] * U(wb & 0x10);
      accum += x_thread[8 * i + 5] * U(wb & 0x20);
      accum += x_thread[8 * i + 6] * U(wb & 0x40);
      accum += x_thread[8 * i + 7] * U(wb & 0x80);
    }
  }

  else if (bits == 2) {
    // uint32 loads for 16-aligned chunks; byte loads for any remainder.
    const device uint32_t* wp = (const device uint32_t*)w;
    int i = 0;
    for (; i < (N / 16); i++) {
      uint32_t wb32 = wp[i];
      for (int j = 0; j < 4; j++) {
        uint8_t wb = uint8_t(wb32 >> (8 * j));
        int base = 16 * i + 4 * j;
        accum += x_thread[base + 0] * U(wb & 0x03)
              +  x_thread[base + 1] * U(wb & 0x0c)
              +  x_thread[base + 2] * U(wb & 0x30)
              +  x_thread[base + 3] * U(wb & 0xc0);
      }
    }
    for (int k = i * 4; k < (N / 4); k++) {
      accum += x_thread[4 * k + 0] * U(w[k] & 0x03)
            +  x_thread[4 * k + 1] * U(w[k] & 0x0c)
            +  x_thread[4 * k + 2] * U(w[k] & 0x30)
            +  x_thread[4 * k + 3] * U(w[k] & 0xc0);
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (N / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

template <typename U, int values_per_thread, int bits>
inline void
qouter(const thread uint8_t* w, U x, U scale, U bias, thread U* result) {
  static_assert(
      bits == 1 || bits == 2 || bits == 3 || bits == 4 || bits == 5 ||
          bits == 6 || bits == 8,
      "Template undefined for bits not in {1, 2, 3, 4, 5, 6, 8}");

  if (bits == 1) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      uint8_t wb = w[i];
      result[8 * i] += x * (select(U(0), scale, bool(wb & 0x01)) + bias);
      result[8 * i + 1] += x * (select(U(0), scale, bool(wb & 0x02)) + bias);
      result[8 * i + 2] += x * (select(U(0), scale, bool(wb & 0x04)) + bias);
      result[8 * i + 3] += x * (select(U(0), scale, bool(wb & 0x08)) + bias);
      result[8 * i + 4] += x * (select(U(0), scale, bool(wb & 0x10)) + bias);
      result[8 * i + 5] += x * (select(U(0), scale, bool(wb & 0x20)) + bias);
      result[8 * i + 6] += x * (select(U(0), scale, bool(wb & 0x40)) + bias);
      result[8 * i + 7] += x * (select(U(0), scale, bool(wb & 0x80)) + bias);
    }
  }

  else if (bits == 2) {
    U s[4] = {scale, scale / 4.0f, scale / 16.0f, scale / 64.0f};
    for (int i = 0; i < (values_per_thread / 4); i++) {
      result[4 * i] += x * (s[0] * (w[i] & 0x03) + bias);
      result[4 * i + 1] += x * (s[1] * (w[i] & 0x0c) + bias);
      result[4 * i + 2] += x * (s[2] * (w[i] & 0x30) + bias);
      result[4 * i + 3] += x * (s[3] * (w[i] & 0xc0) + bias);
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      uint8_t w0 = w[3 * i];
      uint8_t w1 = w[3 * i + 1];
      uint8_t w2 = w[3 * i + 2];

      result[8 * i] += x * ((w0 & 0x7) * scale + bias);
      result[8 * i + 1] += x * (((w0 & 0x38) >> 3) * scale + bias);
      result[8 * i + 2] +=
          x * ((((w0 & 0xc0) >> 6) + ((w1 & 0x1) << 2)) * scale + bias);
      result[8 * i + 3] += x * (((w1 & 0xe) >> 1) * scale + bias);
      result[8 * i + 4] += x * (((w1 & 0x70) >> 4) * scale + bias);
      result[8 * i + 5] +=
          x * ((((w1 & 0x80) >> 7) + ((w2 & 0x3) << 1)) * scale + bias);
      result[8 * i + 6] += x * (((w2 & 0x1c) >> 2) * scale + bias);
      result[8 * i + 7] += x * (((w2 & 0xe0) >> 5) * scale + bias);
    }
  }

  else if (bits == 4) {
    U s[2] = {scale, scale / 16.0f};
    for (int i = 0; i < (values_per_thread / 2); i++) {
      result[2 * i] += x * (s[0] * (w[i] & 0x0f) + bias);
      result[2 * i + 1] += x * (s[1] * (w[i] & 0xf0) + bias);
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      uint8_t w0 = w[5 * i];
      uint8_t w1 = w[5 * i + 1];
      uint8_t w2 = w[5 * i + 2];
      uint8_t w3 = w[5 * i + 3];
      uint8_t w4 = w[5 * i + 4];
      result[8 * i] += x * ((w0 & 0x1f) * scale + bias);
      result[8 * i + 1] +=
          x * ((((w0 & 0xe0) >> 5) + ((w1 & 0x3) << 3)) * scale + bias);
      result[8 * i + 2] += x * (((w1 & 0x7c) >> 2) * scale + bias);
      result[8 * i + 3] +=
          x * ((((w1 & 0x80) >> 7) + ((w2 & 0xf) << 1)) * scale + bias);
      result[8 * i + 4] +=
          x * ((((w2 & 0xf0) >> 4) + ((w3 & 0x1) << 4)) * scale + bias);
      result[8 * i + 5] += x * (((w3 & 0x3e) >> 1) * scale + bias);
      result[8 * i + 6] +=
          x * ((((w3 & 0xc0) >> 6) + ((w4 & 0x7) << 2)) * scale + bias);
      result[8 * i + 7] += x * (((w4 & 0xf8) >> 3) * scale + bias);
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      uint8_t w0 = w[3 * i];
      uint8_t w1 = w[3 * i + 1];
      uint8_t w2 = w[3 * i + 2];

      result[4 * i] += x * ((w0 & 0x3f) * scale + bias);
      result[4 * i + 1] +=
          x * ((((w0 >> 6) & 0x03) + ((w1 & 0x0f) << 2)) * scale + bias);
      result[4 * i + 2] +=
          x * ((((w1 >> 4) & 0x0f) + ((w2 & 0x03) << 4)) * scale + bias);
      result[4 * i + 3] += x * (((w2 >> 2) & 0x3f) * scale + bias);
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      result[i] += x * (scale * w[i] + bias);
    }
  }
}

// Decode one quantized block (scale * q + bias) into w_local. W (the output
// pointer type) serves the threadgroup block loader or a thread-local decode.
template <typename U, int N, int bits, typename W>
inline void dequantize(const device uint8_t* w, U scale, U bias, W w_local) {
  static_assert(
      bits == 1 || bits == 2 || bits == 3 || bits == 4 || bits == 5 ||
          bits == 6 || bits == 8,
      "Template undefined for bits not in {1, 2, 3, 4, 5, 6, 8}");

  if (bits == 1) {
    // 1-bit values are always 0 or 1 — each dequantized element is either
    // bias or (scale + bias). No multiply needed; precompute once per group.
    U scale_plus_bias = scale + bias;
    for (int i = 0; i < (N / 8); i++) {
      uint8_t wb = w[i];
      w_local[8 * i + 0] = select(bias, scale_plus_bias, bool(wb & 0x01));
      w_local[8 * i + 1] = select(bias, scale_plus_bias, bool(wb & 0x02));
      w_local[8 * i + 2] = select(bias, scale_plus_bias, bool(wb & 0x04));
      w_local[8 * i + 3] = select(bias, scale_plus_bias, bool(wb & 0x08));
      w_local[8 * i + 4] = select(bias, scale_plus_bias, bool(wb & 0x10));
      w_local[8 * i + 5] = select(bias, scale_plus_bias, bool(wb & 0x20));
      w_local[8 * i + 6] = select(bias, scale_plus_bias, bool(wb & 0x40));
      w_local[8 * i + 7] = select(bias, scale_plus_bias, bool(wb & 0x80));
    }
  }

  else if (bits == 2) {
    // 4-value mux: decoded[q] = scale*q + bias for q in {0,1,2,3}.
    // Use select() pairs (3 instructions per value, no dynamic array index which
    // would spill to device memory on Apple GPU).
    // Precompute all 4 possible decoded values once; compiler dead-code-eliminates
    // values when N is a compile-time constant.
    U d0 = bias;
    U d1 = scale + bias;
    U d2 = fma(U(2), scale, bias);
    U d3 = fma(U(3), scale, bias);
    for (int i = 0; i < (N / 4); i++) {
      uint8_t wb = w[i];
      // Decode each 2-bit field with two select()s: no FP multiply per element.
      auto mux2 = [&](uint8_t q) -> U {
        return select(select(d0, d1, bool(q & 1)), select(d2, d3, bool(q & 1)), bool(q & 2));
      };
      w_local[4 * i + 0] = mux2(wb & 0x03);
      w_local[4 * i + 1] = mux2((wb >> 2) & 0x03);
      w_local[4 * i + 2] = mux2((wb >> 4) & 0x03);
      w_local[4 * i + 3] = mux2((wb >> 6) & 0x03);
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (N / 8); i++) {
      w_local += 8 * i;
      w += 3 * i;

      w_local[0] = (w[0] & 0x7) * scale + bias;
      w_local[1] = ((w[0] & 0x38) >> 3) * scale + bias;
      w_local[2] = (((w[0] & 0xc0) >> 6) + ((w[1] & 0x1) << 2)) * scale + bias;
      w_local[3] = ((w[1] & 0xe) >> 1) * scale + bias;
      w_local[4] = ((w[1] & 0x70) >> 4) * scale + bias;
      w_local[5] = (((w[1] & 0x80) >> 7) + ((w[2] & 0x3) << 1)) * scale + bias;
      w_local[6] = ((w[2] & 0x1c) >> 2) * scale + bias;
      w_local[7] = ((w[2] & 0xe0) >> 5) * scale + bias;
    }
  }

  else if (bits == 4) {
    U s[2] = {scale, scale / static_cast<U>(16.0f)};
    for (int i = 0; i < (N / 2); i++) {
      w_local[2 * i] = s[0] * (w[i] & 0x0f) + bias;
      w_local[2 * i + 1] = s[1] * (w[i] & 0xf0) + bias;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (N / 8); i++) {
      w_local += 8 * i;
      w += 5 * i;

      w_local[0] = (w[0] & 0x1f) * scale + bias;
      w_local[1] = (((w[0] & 0xe0) >> 5) + ((w[1] & 0x3) << 3)) * scale + bias;
      w_local[2] = ((w[1] & 0x7c) >> 2) * scale + bias;
      w_local[3] = (((w[1] & 0x80) >> 7) + ((w[2] & 0xf) << 1)) * scale + bias;
      w_local[4] = (((w[2] & 0xf0) >> 4) + ((w[3] & 0x1) << 4)) * scale + bias;
      w_local[5] = ((w[3] & 0x3e) >> 1) * scale + bias;
      w_local[6] = (((w[3] & 0xc0) >> 6) + ((w[4] & 0x7) << 2)) * scale + bias;
      w_local[7] = ((w[4] & 0xf8) >> 3) * scale + bias;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (N / 4); i++) {
      w_local += 4 * i;
      w += 3 * i;
      w_local[0] = (w[0] & 0x3f) * scale + bias;
      w_local[1] = (((w[0] >> 6) & 0x03) + ((w[1] & 0x0f) << 2)) * scale + bias;
      w_local[2] = (((w[1] >> 4) & 0x0f) + ((w[2] & 0x03) << 4)) * scale + bias;
      w_local[3] = ((w[2] >> 2) & 0x3f) * scale + bias;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      w_local[i] = scale * w[i] + bias;
    }
  }
}

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short group_size,
    short bits>
struct QuantizedBlockLoader {
  static_assert(
      BCOLS <= group_size,
      "The group size should be larger than the columns");
  static_assert(
      group_size % BCOLS == 0,
      "The group size should be divisible by the columns");
  static_assert(
      bits == 1 || bits == 2 || bits == 3 || bits == 4 || bits == 5 ||
          bits == 6 || bits == 8,
      "Template undefined for bits not in {1, 2, 3, 4, 5, 6, 8}");

  MLX_MTL_CONST short pack_factor = get_pack_factor<bits, 8>();
  MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short group_steps = group_size / BCOLS;

  const int src_ld;
  const int tile_stride;
  short group_step_cnt;
  const int group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  const device T* scales;
  const device T* biases;

  QuantizedBlockLoader(
      const device uint8_t* src_,
      const device T* scales_,
      const device T* biases_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(
            reduction_dim ? BCOLS_PACKED * bytes_per_pack
                          : BROWS * src_ld * bytes_per_pack / pack_factor),
        group_step_cnt(0),
        group_stride(BROWS * src_ld / group_size),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        scales(scales_ + bi * src_ld / group_size),
        biases(biases_ + bi * src_ld / group_size) {}

  void load_unsafe() const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    T scale = *scales;
    T bias = *biases;
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    if (reduction_dim == 1 && bi >= src_tile_dim.x) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    if (reduction_dim == 0 && bi >= src_tile_dim.y) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    T scale = *scales;
    T bias = *biases;
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          (device uint8_t*)(src + i * bytes_per_pack),
          scale,
          bias,
          dst + i * pack_factor);
    }
  }

  void next() {
    src += tile_stride;
    if (reduction_dim == 1) {
      if (group_steps > 1) {
        group_step_cnt++;
        if (group_step_cnt == group_steps) {
          group_step_cnt = 0;
          scales++;
          biases++;
        }
      } else {
        scales++;
        biases++;
      }
    } else {
      scales += group_stride;
      biases += group_stride;
    }
  }
};

template <typename T, int group_size, int bits, int D>
METAL_FUNC void qmv_quad_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint quad_gid [[quadgroup_index_in_threadgroup]],
    uint quad_lid [[thread_index_in_quadgroup]]) {
  constexpr int quads_per_simd = SIMD_SIZE / QUAD_SIZE;
  constexpr int pack_factor = 32 / bits;
  constexpr int values_per_thread = D / QUAD_SIZE;
  constexpr int packs_per_thread = values_per_thread / pack_factor;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  constexpr int results_per_quadgroup = 8;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_quadgroup] = {0};

  // Adjust positions
  const int in_vec_size_w = in_vec_size / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * quads_per_simd * results_per_quadgroup + quad_gid;

  w += out_row * in_vec_size_w + quad_lid * packs_per_thread;
  scales += out_row * in_vec_size_g + quad_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + quad_lid / scale_step_per_thread;
  x += tid.x * in_vec_size + quad_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;

  U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

  for (int row = 0; row < results_per_quadgroup; row++) {
    auto wl = (const device uint8_t*)(w + row * in_vec_size_w * quads_per_simd);
    const device T* sl = scales + row * in_vec_size_g * quads_per_simd;
    const device T* bl = biases + row * in_vec_size_g * quads_per_simd;

    U s = sl[0];
    U b = bl[0];
    if (row * quads_per_simd + out_row < out_vec_size) {
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }
  }

  for (int row = 0; row < results_per_quadgroup; row++) {
    result[row] = quad_sum(result[row]);
    if (quad_lid == 0 && row * quads_per_simd + out_row < out_vec_size) {
      y[row * quads_per_simd] = static_cast<T>(result[row]);
    }
  }
}

template <typename T, int group_size, int bits, bool symmetric = false>
METAL_FUNC void qmv_fast_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int packs_per_thread = bits <= 2 ? 1 : 2;  // 1-bit: 1 pack (vpt=32) for occupancy
  constexpr int num_simdgroups = 4;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  // Adjust positions
  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  if constexpr (!symmetric) {
    biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  }
  x += tid.x * in_vec_size + simd_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;

  const int aligned_end = (in_vec_size / block_size) * block_size;

  for (int k = 0; k < aligned_end; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = scales + row * in_vec_size_g;

      U s = sl[0];
      U b;
      if constexpr (symmetric) {
        // I-B: bias = -scale * ratio; no DRAM load needed.
        // bits=1: bias = -scale/2 (q in {0,1}, dequant = scale*(q-0.5))
        // bits=2: bias = -scale   (q in {0,1,2}, dequant = scale*(q-1))
        b = -s * U(bits == 1 ? 0.5f : 1.0f);
      } else {
        const device T* bl = biases + row * in_vec_size_g;
        b = bl[0];
      }
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    if constexpr (!symmetric) {
      biases += block_size / group_size;
    }
    x += block_size;
  }

  if (aligned_end < in_vec_size) {
    bool in_bounds = (aligned_end + simd_lid * values_per_thread) < in_vec_size;
    U sum = 0;
    if (in_bounds) {
      sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);
    } else {
      for (int i = 0; i < values_per_thread; i++)
        x_thread[i] = 0;
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = scales + row * in_vec_size_g;

      U s = in_bounds ? (U)sl[0] : (U)0;
      U b;
      if constexpr (symmetric) {
        b = -s * U(bits == 1 ? 0.5f : 1.0f);
      } else {
        const device T* bl = biases + row * in_vec_size_g;
        b = in_bounds ? (U)bl[0] : (U)0;
      }
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[row] = static_cast<T>(result[row]);
    }
  }
}

template <typename T, int group_size, int bits>
METAL_FUNC void qmv_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int num_simdgroups = 4;
  constexpr int results_per_simdgroup = 4;
  constexpr int packs_per_thread = 1;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();

  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  // Adjust positions
  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);

  if (out_row >= out_vec_size) {
    return;
  }

  // In this case we need to properly guard all our reads because there isn't
  // even 1 tile in the matrix
  if (out_vec_size < (num_simdgroups * results_per_simdgroup)) {
    ws +=
        out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    x += tid.x * in_vec_size + simd_lid * values_per_thread;
    y += tid.x * out_vec_size + out_row;

    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
      U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

      for (int row = 0;
           row < results_per_simdgroup && out_row + row < out_vec_size;
           row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] +=
            qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
      }

      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      biases += block_size / group_size;
      x += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
        0,
        values_per_thread);
    if (remaining > 0) {
      U sum = load_vector_safe<T, U, values_per_thread, bits>(
          x, x_thread, remaining);

      for (int row = 0;
           row < results_per_simdgroup && out_row + row < out_vec_size;
           row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] += qdot_safe<U, values_per_thread, bits>(
            wl, x_thread, s, b, sum, remaining);
      }
    }

    for (int row = 0;
         row < results_per_simdgroup && out_row + row < out_vec_size;
         row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T>(result[row]);
      }
    }
  }

  // In this case the last tile is moved back to redo some output values
  else {
    ws += used_out_row * in_vec_size_w +
        simd_lid * packs_per_thread * bytes_per_pack;
    scales += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    x += tid.x * in_vec_size + simd_lid * values_per_thread;
    y += tid.x * out_vec_size + used_out_row;

    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
      U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] +=
            qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
      }

      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      biases += block_size / group_size;
      x += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
        0,
        values_per_thread);
    if (remaining > 0) {
      U sum = load_vector_safe<T, U, values_per_thread, bits>(
          x, x_thread, remaining);

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] += qdot_safe<U, values_per_thread, bits>(
            wl, x_thread, s, b, sum, remaining);
      }
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T>(result[row]);
      }
    }
  }
}

// Affine analog of fp_qmv_wide. Weights carry a scale and bias per group, so
// each group is decoded in 8-value sub-chunks (scale * q + bias, registers
// bounded for any group_size) and reused across the vecs_per_tg vectors.
template <typename T, int group_size, int bits, int vecs_per_tg, int k_lanes, bool symmetric = false>
METAL_FUNC void qmv_wide_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int num_simdgroups = 4;
  constexpr int results_per_simdgroup = SIMD_SIZE / k_lanes;
  constexpr int sub = 8; // values per sub-chunk (== bits bytes, byte-aligned)

  typedef float U;

  const short k_lane = simd_lid % k_lanes;
  const short sg_row = simd_lid / k_lanes;

  const int out_row = tid.y * (results_per_simdgroup * num_simdgroups) +
      results_per_simdgroup * simd_gid + sg_row;
  const int vec0 = tid.x * vecs_per_tg;

  const int row = min(out_row, out_vec_size - 1);

  const int in_vec_size_w = in_vec_size * bits / 8; // bytes per weight row
  const int in_vec_size_g = in_vec_size / group_size;
  const device uint8_t* wrow = (const device uint8_t*)w + row * in_vec_size_w;
  const device T* srow = scales + row * in_vec_size_g;
  // brow not declared when symmetric=true; DCE'd by compiler when false.
  const device T* brow = symmetric ? nullptr : biases + row * in_vec_size_g;

  const device T* xv[vecs_per_tg];
  for (int v = 0; v < vecs_per_tg; v++) {
    xv[v] = x + min(vec0 + v, M - 1) * in_vec_size;
  }

  U result[vecs_per_tg] = {0};

  // Each lane reduces a strided subset of the row's groups: decode the group in
  // 8-value sub-chunks and reuse each chunk across the streamed vectors.
  for (int g = k_lane; g < in_vec_size_g; g += k_lanes) {
    U scale = srow[g];
    // I-B: symmetric layers have bias = -scale * ratio; no DRAM load.
    // bits=1: ratio=0.5 (bias=-scale/2), bits=2: ratio=1.0 (bias=-scale)
    U bias;
    if constexpr (symmetric) {
      bias = -scale * U(bits == 1 ? 0.5f : 1.0f);
    } else {
      bias = brow[g];
    }
    // Precompute once per group; compiler dead-code-eliminates unused vars.
    U spb  = scale + bias;          // bits==1: lut[1]; bits==2: lut[1]
    U lut2 = fma(U(2), scale, bias); // bits==2: lut[2]; eliminated otherwise
    U lut3 = fma(U(3), scale, bias); // bits==2: lut[3]; eliminated otherwise
#pragma unroll
    for (int sc = 0; sc < group_size / sub; sc++) {
      const int k0 = g * group_size + sc * sub;
      const device uint8_t* wc = wrow + k0 * bits / 8;
      U w_dq[sub];
      if constexpr (bits == 1) {
        // sub=8 elements packed in 1 byte; spb hoisted to group scope.
        uint8_t wb = wc[0];
        w_dq[0] = select(bias, spb, bool(wb & 0x01));
        w_dq[1] = select(bias, spb, bool(wb & 0x02));
        w_dq[2] = select(bias, spb, bool(wb & 0x04));
        w_dq[3] = select(bias, spb, bool(wb & 0x08));
        w_dq[4] = select(bias, spb, bool(wb & 0x10));
        w_dq[5] = select(bias, spb, bool(wb & 0x20));
        w_dq[6] = select(bias, spb, bool(wb & 0x40));
        w_dq[7] = select(bias, spb, bool(wb & 0x80));
      } else if constexpr (bits == 2) {
        // sub=8 values from 2 bytes. Hoisted d0..d3 (computed once per group)
        // replace per-element scale*q+bias with 2 select() ops per value.
        // select() → single hardware instruction; no dynamic array index (avoids
        // register spilling on Apple GPU).
        auto mux2 = [&](uint8_t q) -> U {
          return select(select(bias, spb, bool(q & 1)),
                        select(lut2, lut3, bool(q & 1)), bool(q & 2));
        };
        uint8_t wb0 = wc[0], wb1 = wc[1];
        w_dq[0] = mux2(wb0 & 0x03);
        w_dq[1] = mux2((wb0 >> 2) & 0x03);
        w_dq[2] = mux2((wb0 >> 4) & 0x03);
        w_dq[3] = mux2((wb0 >> 6) & 0x03);
        w_dq[4] = mux2(wb1 & 0x03);
        w_dq[5] = mux2((wb1 >> 2) & 0x03);
        w_dq[6] = mux2((wb1 >> 4) & 0x03);
        w_dq[7] = mux2((wb1 >> 6) & 0x03);
      } else {
        dequantize<U, sub, bits>(wc, scale, bias, w_dq);
      }
#pragma unroll
      for (int v = 0; v < vecs_per_tg; v++) {
        const device T* xc = xv[v] + k0;
        U acc = 0;
#pragma unroll
        for (int i = 0; i < sub; i++) {
          acc += static_cast<U>(xc[i]) * w_dq[i];
        }
        result[v] += acc;
      }
    }
  }

  // Reduce each vector's partial over its k_lanes with a shuffle ladder:
  // simd_sum would mix the results_per_simdgroup rows a simdgroup spans.
  for (int v = 0; v < vecs_per_tg; v++) {
    if constexpr (k_lanes >= 32) {
      result[v] += simd_shuffle_down(result[v], 16);
    }
    if constexpr (k_lanes >= 16) {
      result[v] += simd_shuffle_down(result[v], 8);
    }
    if constexpr (k_lanes >= 8) {
      result[v] += simd_shuffle_down(result[v], 4);
    }
    if constexpr (k_lanes >= 4) {
      result[v] += simd_shuffle_down(result[v], 2);
    }
    if constexpr (k_lanes >= 2) {
      result[v] += simd_shuffle_down(result[v], 1);
    }
  }

  if (k_lane == 0 && out_row < out_vec_size) {
    for (int v = 0; v < vecs_per_tg; v++) {
      if (vec0 + v < M) {
        y[(vec0 + v) * out_vec_size + out_row] = static_cast<T>(result[v]);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Realization 2.3 (I-K): Granlund-Montgomery magic divmod-3, exact for v ∈ [0,255].
// ⌊v/3⌋ = (v·171) >> 9  — no integer divide instruction, pure multiply + shift.
// Then v mod 3 = v − 3q = v − (q<<1) − q.
METAL_FUNC uint t5_div3(uint v)  { return (v * 171u) >> 9u; }

// Parallel trit extraction helpers (I-L): all operate on the ORIGINAL byte v,
// so ALL four calls can be issued simultaneously with no serial data dependency.
// Exact for v ∈ [0, 242] (max valid t5 byte = 3^5 − 1 = 242).
METAL_FUNC uint t5_div9(uint v)  { return (v * 228u) >> 11u; }  // ⌊v/9⌋
METAL_FUNC uint t5_div27(uint v) { return (v * 76u)  >> 11u; }  // ⌊v/27⌋
// ⌊v/81⌋ ∈ {0,1,2}: two independent comparisons, no multiply needed.
// Precondition: v ∈ [0, 242] (max valid t5 byte = 3^5 − 1 = 242).
// v=243 cannot appear in a validly-encoded t5 stream, so v/81 ≤ 2 always.
METAL_FUNC uint t5_div81(uint v) { return uint(v >= 81u) + uint(v >= 162u); }

// T5_TO_B4: radix-conversion LUT — maps each t5 byte (0..242) to its 5 trits
// repacked as 2-bit fields in a uint.  Field k occupies bits [2k+1 : 2k]:
//   bits[1:0] = t0,  bits[3:2] = t1,  bits[5:4] = t2,
//   bits[7:6] = t3,  bits[9:8] = t4,  tₖ ∈ {0,1,2}.
//
// This enables the same pre-scaling trick as 2-bit affine (I-L):
//   x_pre[k] = x[k] × 4^{−k}
//   x_pre[k] × (T5_TO_B4[v] & (3 << 2k))  =  x[k] × tₖ   (no division)
// The −1 offset is absorbed via x_sum: result = s × (Σ x[k]tₖ − Σ x[k]).
// 256 entries × 4 bytes = 1 KB; permanently L1-resident during GEMV.
constant constexpr uint T5_TO_B4[256] = {
    0x000, 0x001, 0x002, 0x004, 0x005, 0x006, 0x008, 0x009,
    0x00A, 0x010, 0x011, 0x012, 0x014, 0x015, 0x016, 0x018,
    0x019, 0x01A, 0x020, 0x021, 0x022, 0x024, 0x025, 0x026,
    0x028, 0x029, 0x02A, 0x040, 0x041, 0x042, 0x044, 0x045,
    0x046, 0x048, 0x049, 0x04A, 0x050, 0x051, 0x052, 0x054,
    0x055, 0x056, 0x058, 0x059, 0x05A, 0x060, 0x061, 0x062,
    0x064, 0x065, 0x066, 0x068, 0x069, 0x06A, 0x080, 0x081,
    0x082, 0x084, 0x085, 0x086, 0x088, 0x089, 0x08A, 0x090,
    0x091, 0x092, 0x094, 0x095, 0x096, 0x098, 0x099, 0x09A,
    0x0A0, 0x0A1, 0x0A2, 0x0A4, 0x0A5, 0x0A6, 0x0A8, 0x0A9,
    0x0AA, 0x100, 0x101, 0x102, 0x104, 0x105, 0x106, 0x108,
    0x109, 0x10A, 0x110, 0x111, 0x112, 0x114, 0x115, 0x116,
    0x118, 0x119, 0x11A, 0x120, 0x121, 0x122, 0x124, 0x125,
    0x126, 0x128, 0x129, 0x12A, 0x140, 0x141, 0x142, 0x144,
    0x145, 0x146, 0x148, 0x149, 0x14A, 0x150, 0x151, 0x152,
    0x154, 0x155, 0x156, 0x158, 0x159, 0x15A, 0x160, 0x161,
    0x162, 0x164, 0x165, 0x166, 0x168, 0x169, 0x16A, 0x180,
    0x181, 0x182, 0x184, 0x185, 0x186, 0x188, 0x189, 0x18A,
    0x190, 0x191, 0x192, 0x194, 0x195, 0x196, 0x198, 0x199,
    0x19A, 0x1A0, 0x1A1, 0x1A2, 0x1A4, 0x1A5, 0x1A6, 0x1A8,
    0x1A9, 0x1AA, 0x200, 0x201, 0x202, 0x204, 0x205, 0x206,
    0x208, 0x209, 0x20A, 0x210, 0x211, 0x212, 0x214, 0x215,
    0x216, 0x218, 0x219, 0x21A, 0x220, 0x221, 0x222, 0x224,
    0x225, 0x226, 0x228, 0x229, 0x22A, 0x240, 0x241, 0x242,
    0x244, 0x245, 0x246, 0x248, 0x249, 0x24A, 0x250, 0x251,
    0x252, 0x254, 0x255, 0x256, 0x258, 0x259, 0x25A, 0x260,
    0x261, 0x262, 0x264, 0x265, 0x266, 0x268, 0x269, 0x26A,
    0x280, 0x281, 0x282, 0x284, 0x285, 0x286, 0x288, 0x289,
    0x28A, 0x290, 0x291, 0x292, 0x294, 0x295, 0x296, 0x298,
    0x299, 0x29A, 0x2A0, 0x2A1, 0x2A2, 0x2A4, 0x2A5, 0x2A6,
    0x2A8, 0x2A9, 0x2AA, 0x000, 0x000, 0x000, 0x000, 0x000,
    0x000, 0x000, 0x000, 0x000, 0x000, 0x000, 0x000, 0x000,
};
// ---------------------------------------------------------------------------
// qmv_fast_t5_impl — optimized (O1-O5)
//
// [O1] packed_uchar4 weight loads (alignment 1, 7 loads vs 26 per row-group)
// [O2] packed_half4 activation loads (2 loads vs 5 per 5-trit byte)
// [O3] 20-trit chunk structure (4 bytes x 5 trits, fully unrolled)
// [O4] USE_SIGMA: skip x_sum + pre-scale muls (sigma precomputed elsewhere)
// [O5] row clamping instead of duplicated boundary loop
// ---------------------------------------------------------------------------
template <typename T, int group_size, bool USE_SIGMA = false>
METAL_FUNC void qmv_fast_t5_impl(
    const device uint8_t* w,
    const device T* scales,
    const device T* x,            // pre-scaled by 4^{-(j%5)} when USE_SIGMA
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const device float* sigma,    // [n_groups] group sums; used iff USE_SIGMA
    threadgroup const uint* lut,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  constexpr int bytes_per_group = (group_size + 4) / 5;
  constexpr int full_bytes = group_size / 5;         // 25 for gs=128
  constexpr int chunk_bytes = full_bytes & ~3;       // 24: packed_uchar4 part
  constexpr int tail_full   = full_bytes - chunk_bytes; // 1 full byte in tail
  constexpr int rem_trits   = group_size % 5;        // 3 for gs=128
  constexpr int num_simdgroups = 4;
  constexpr int results_per_simdgroup = 4;

  typedef float U;
  constexpr U PS0 = 1.0f, PS1 = 0.25f, PS2 = 0.0625f,
              PS3 = 0.015625f, PS4 = 0.00390625f;

  thread U result[results_per_simdgroup] = {0};
  const int n_groups = in_vec_size / group_size;
  const int out_row  = tid.y * (num_simdgroups * results_per_simdgroup) +
                       simd_gid * results_per_simdgroup;
  const int ng_bpg   = n_groups * bytes_per_group;

  int rows[results_per_simdgroup];
  bool row_ok[results_per_simdgroup];
  for (int r = 0; r < results_per_simdgroup; r++) {
    row_ok[r] = (out_row + r) < out_vec_size;
    rows[r]   = row_ok[r] ? (out_row + r) : 0;
  }

  const device T* x_batch = x + tid.x * in_vec_size;

  for (int g = simd_lid; g < n_groups; g += SIMD_SIZE) {
    const device T* xg = x_batch + g * group_size;
    const device uint8_t* wg0 = w + rows[0]*ng_bpg + g*bytes_per_group;
    const device uint8_t* wg1 = w + rows[1]*ng_bpg + g*bytes_per_group;
    const device uint8_t* wg2 = w + rows[2]*ng_bpg + g*bytes_per_group;
    const device uint8_t* wg3 = w + rows[3]*ng_bpg + g*bytes_per_group;
    const U s0 = U(scales[rows[0]*n_groups + g]);
    const U s1 = U(scales[rows[1]*n_groups + g]);
    const U s2 = U(scales[rows[2]*n_groups + g]);
    const U s3 = U(scales[rows[3]*n_groups + g]);

    U a0 = 0, a1 = 0, a2 = 0, a3 = 0, x_sum = 0;

#pragma clang loop unroll(full)
    for (int c = 0; c < chunk_bytes / 4; c++) {
      const packed_uchar4 q0 = *((const device packed_uchar4*)(wg0 + c*4));
      const packed_uchar4 q1 = *((const device packed_uchar4*)(wg1 + c*4));
      const packed_uchar4 q2 = *((const device packed_uchar4*)(wg2 + c*4));
      const packed_uchar4 q3 = *((const device packed_uchar4*)(wg3 + c*4));
#pragma clang loop unroll(full)
      for (int bb = 0; bb < 4; bb++) {
        const uint p0 = lut[q0[bb]], p1 = lut[q1[bb]];
        const uint p2 = lut[q2[bb]], p3 = lut[q3[bb]];
        const int base = (c*4 + bb) * 5;
        const packed_half4 xv4 = *((const device packed_half4*)(xg + base));
        const U xv0 = U(xv4[0]), xv1 = U(xv4[1]), xv2 = U(xv4[2]), xv3 = U(xv4[3]);
        const U xv4s = U(xg[base + 4]);
        U xp1, xp2, xp3, xp4;
        if constexpr (USE_SIGMA) { xp1=xv1; xp2=xv2; xp3=xv3; xp4=xv4s; }
        else { x_sum += xv0+xv1+xv2+xv3+xv4s; xp1=xv1*PS1; xp2=xv2*PS2; xp3=xv3*PS3; xp4=xv4s*PS4; }
        a0+=xv0*U(p0&0x003u)+xp1*U(p0&0x00Cu)+xp2*U(p0&0x030u)+xp3*U(p0&0x0C0u)+xp4*U(p0&0x300u);
        a1+=xv0*U(p1&0x003u)+xp1*U(p1&0x00Cu)+xp2*U(p1&0x030u)+xp3*U(p1&0x0C0u)+xp4*U(p1&0x300u);
        a2+=xv0*U(p2&0x003u)+xp1*U(p2&0x00Cu)+xp2*U(p2&0x030u)+xp3*U(p2&0x0C0u)+xp4*U(p2&0x300u);
        a3+=xv0*U(p3&0x003u)+xp1*U(p3&0x00Cu)+xp2*U(p3&0x030u)+xp3*U(p3&0x0C0u)+xp4*U(p3&0x300u);
      }
    }
    if constexpr (tail_full > 0 || rem_trits > 0) {
      ushort t0, t1, t2, t3;
      if constexpr (tail_full > 0) {
        t0 = *((const device ushort*)(wg0 + chunk_bytes));
        t1 = *((const device ushort*)(wg1 + chunk_bytes));
        t2 = *((const device ushort*)(wg2 + chunk_bytes));
        t3 = *((const device ushort*)(wg3 + chunk_bytes));
      } else {
        t0 = ushort(wg0[chunk_bytes]) << 8; t1 = ushort(wg1[chunk_bytes]) << 8;
        t2 = ushort(wg2[chunk_bytes]) << 8; t3 = ushort(wg3[chunk_bytes]) << 8;
      }
      if constexpr (tail_full > 0) {
        const uint p0=lut[t0&0xFF], p1=lut[t1&0xFF], p2=lut[t2&0xFF], p3=lut[t3&0xFF];
        const int base = chunk_bytes * 5;
        const packed_half4 xv4 = *((const device packed_half4*)(xg + base));
        const U xv0=U(xv4[0]), xv1=U(xv4[1]), xv2=U(xv4[2]), xv3=U(xv4[3]), xv4s=U(xg[base+4]);
        U xp1,xp2,xp3,xp4;
        if constexpr (USE_SIGMA) { xp1=xv1;xp2=xv2;xp3=xv3;xp4=xv4s; }
        else { x_sum+=xv0+xv1+xv2+xv3+xv4s; xp1=xv1*PS1;xp2=xv2*PS2;xp3=xv3*PS3;xp4=xv4s*PS4; }
        a0+=xv0*U(p0&0x003u)+xp1*U(p0&0x00Cu)+xp2*U(p0&0x030u)+xp3*U(p0&0x0C0u)+xp4*U(p0&0x300u);
        a1+=xv0*U(p1&0x003u)+xp1*U(p1&0x00Cu)+xp2*U(p1&0x030u)+xp3*U(p1&0x0C0u)+xp4*U(p1&0x300u);
        a2+=xv0*U(p2&0x003u)+xp1*U(p2&0x00Cu)+xp2*U(p2&0x030u)+xp3*U(p2&0x0C0u)+xp4*U(p2&0x300u);
        a3+=xv0*U(p3&0x003u)+xp1*U(p3&0x00Cu)+xp2*U(p3&0x030u)+xp3*U(p3&0x0C0u)+xp4*U(p3&0x300u);
      }
      if constexpr (rem_trits > 0) {
        const uint p0=lut[t0>>8], p1=lut[t1>>8], p2=lut[t2>>8], p3=lut[t3>>8];
        const int base = full_bytes * 5;
#pragma clang loop unroll(full)
        for (int k = 0; k < rem_trits; k++) {
          const U xv=U(xg[base+k]); U xp;
          if constexpr (USE_SIGMA) { xp=xv; }
          else { x_sum+=xv; xp=xv*(k==0?PS0:k==1?PS1:k==2?PS2:k==3?PS3:PS4); }
          const uint m=0x3u<<(2*k);
          a0+=xp*U(p0&m); a1+=xp*U(p1&m); a2+=xp*U(p2&m); a3+=xp*U(p3&m);
        }
      }
    }
    const U xs = USE_SIGMA ? U(sigma[g]) : x_sum;
    result[0] += s0*(a0-xs); result[1] += s1*(a1-xs);
    result[2] += s2*(a2-xs); result[3] += s3*(a3-xs);
  }
  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0 && row_ok[row])
      y[tid.x * out_vec_size + out_row + row] = T(result[row]);
  }
}


// qmv_wide_t5: small-batch decode (M = vecs_per_tg = 2..5).
//
// Amortises the weight stream across all M vectors (Identity I-C):
// decode each t5 byte once, multiply with all M activations.
// k_lanes threads stride over groups, results_per_simdgroup=SIMD_SIZE/k_lanes.
template <typename T, int group_size, int vecs_per_tg, int k_lanes>
METAL_FUNC void qmv_wide_t5_impl(
    const device uint8_t* w,
    const device T* scales,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& M,
    threadgroup const uint* lut,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  constexpr int bytes_per_group = (group_size + 4) / 5;
  constexpr int full_bytes = group_size / 5;
  constexpr int rem_trits  = group_size % 5;
  constexpr int num_simdgroups = 4;
  constexpr int results_per_simdgroup = SIMD_SIZE / k_lanes;

  typedef float U;
  const short k_lane = simd_lid % k_lanes;
  const short sg_row = simd_lid / k_lanes;

  const int out_row = tid.y * (results_per_simdgroup * num_simdgroups) +
                      results_per_simdgroup * simd_gid + sg_row;
  const int vec0 = tid.x * vecs_per_tg;
  const int n_groups = in_vec_size / group_size;

  const int row = min(out_row, out_vec_size - 1);
  const device uint8_t* wrow = w + row * (n_groups * bytes_per_group);
  const device T* srow = scales + row * n_groups;

  const device T* xv[vecs_per_tg];
  for (int v = 0; v < vecs_per_tg; v++) {
    xv[v] = x + min(vec0 + v, M - 1) * in_vec_size;
  }

  U result[vecs_per_tg] = {0};

  for (int g = k_lane; g < n_groups; g += k_lanes) {
    const device uint8_t* wg = wrow + g * bytes_per_group;
    const U s = U(srow[g]);
    const int k0 = g * group_size;

    // Full bytes: T5_TO_B4 LUT replaces serial divmod chain; all 5 dq values
    // are independent (no v=q reassignment), hiding decode latency.
    for (int b = 0; b < full_bytes; b++) {
      const uint p = T5_TO_B4[wg[b]];
      const int base = k0 + b * 5;
      const U dq0 = s * (U(p & 0x003u)        - 1.0f);
      const U dq1 = s * (U((p >>  2u) & 0x3u) - 1.0f);
      const U dq2 = s * (U((p >>  4u) & 0x3u) - 1.0f);
      const U dq3 = s * (U((p >>  6u) & 0x3u) - 1.0f);
      const U dq4 = s * (U((p >>  8u) & 0x3u) - 1.0f);
#pragma unroll
      for (int vi = 0; vi < vecs_per_tg; vi++) {
        result[vi] += U(xv[vi][base + 0]) * dq0
                    + U(xv[vi][base + 1]) * dq1
                    + U(xv[vi][base + 2]) * dq2
                    + U(xv[vi][base + 3]) * dq3
                    + U(xv[vi][base + 4]) * dq4;
      }
    }
    // Last partial byte via T5_TO_B4 LUT.
    if constexpr (rem_trits > 0) {
      const uint p = T5_TO_B4[wg[full_bytes]];
      const int base = k0 + full_bytes * 5;
      U dq0 = U(0), dq1 = U(0), dq2 = U(0), dq3 = U(0);
      if constexpr (rem_trits >= 1) { dq0 = s * (U(p & 0x003u) - 1.0f); }
      if constexpr (rem_trits >= 2) { dq1 = s * (U((p >> 2u) & 0x3u) - 1.0f); }
      if constexpr (rem_trits >= 3) { dq2 = s * (U((p >> 4u) & 0x3u) - 1.0f); }
      if constexpr (rem_trits >= 4) { dq3 = s * (U((p >> 6u) & 0x3u) - 1.0f); }
#pragma unroll
      for (int vi = 0; vi < vecs_per_tg; vi++) {
        if constexpr (rem_trits >= 1) result[vi] += U(xv[vi][base + 0]) * dq0;
        if constexpr (rem_trits >= 2) result[vi] += U(xv[vi][base + 1]) * dq1;
        if constexpr (rem_trits >= 3) result[vi] += U(xv[vi][base + 2]) * dq2;
        if constexpr (rem_trits >= 4) result[vi] += U(xv[vi][base + 3]) * dq3;
      }
    }
  }

  // k_lane shuffle reduction (same ladder as qmv_wide_impl).
  for (int v = 0; v < vecs_per_tg; v++) {
    if constexpr (k_lanes >= 32) result[v] += simd_shuffle_down(result[v], 16);
    if constexpr (k_lanes >= 16) result[v] += simd_shuffle_down(result[v], 8);
    if constexpr (k_lanes >= 8)  result[v] += simd_shuffle_down(result[v], 4);
    if constexpr (k_lanes >= 4)  result[v] += simd_shuffle_down(result[v], 2);
    if constexpr (k_lanes >= 2)  result[v] += simd_shuffle_down(result[v], 1);
  }

  if (k_lane == 0 && out_row < out_vec_size) {
    for (int v = 0; v < vecs_per_tg; v++) {
      if (vec0 + v < M) {
        y[(vec0 + v) * out_vec_size + out_row] = T(result[v]);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// qmm_t5_impl: simdgroup MMA GEMM for t5 ternary weights (Identity I-M).
//
// Computes out[M, N] = x[M, K] @ t5_decode(w[N, K]).T  (float32 accumulate).
//
// Architecture (adapted from MetalTile mt_qmm_mma_int2):
//   BM=BN=32, BK=group_size (one complete t5 group per K-block iteration)
//   128 threads = 4 SG × 32 lanes; 2×2 SG tile layout (sm=sg/2, sn=sg%2)
//   xs[BM][BK+4] and ws[BN][BK+4] threadgroup tiles
//   BK/8 simdgroup_multiply_accumulate calls per K-block
//
// Thread (lane_in_tg = sg_id*32 + lane):
//   X loading :  row = lane_in_tg/4, k_sub = lane_in_tg%4 → loads BK/4 x-elems
//   W dequant :  row = lane_in_tg/4, sub  = lane_in_tg%4 → decodes BK/4 trits
//                (sub 0..3 each decode a contiguous BK/4 K-positions from t5 bytes)
//
// Grid: (ceil(N/32), ceil(M/32), B)  Threadgroup: (32, 4, 1)
// ---------------------------------------------------------------------------
template <typename T, int group_size>
METAL_FUNC void qmm_t5_impl(
    const device uint8_t* w,    // (N, n_groups * bpg) t5 bytes
    const device T* scales,     // (N, n_groups)
    const device T* x,          // (M, K)
    device T* out,              // (M, N)
    const constant int& M_c,
    const constant int& N_c,
    const constant int& K_c,
    threadgroup T* xs,          // BM*(group_size+4) — declared by caller
    threadgroup T* ws,          // BN*(group_size+4) — declared by caller
    uint2 tgid,                  // (n_tile, m_tile)
    uint  lane,                  // thread_index_in_simdgroup  (0..31)
    uint  sg_id)                 // simdgroup_index_in_threadgroup (0..3)
{
  constexpr int bpg    = (group_size + 4) / 5;   // bytes per group
  constexpr int xs_ld  = group_size + 4;          // padded stride for xs
  constexpr int ws_ld  = group_size + 4;          // padded stride for ws
  constexpr int BM = 32, BN = 32;
  (void)BM; (void)BN;

  // Float32 accumulators: 2×2 SG tile → 4 frags per SG (each covers 8×8 out elements)
  simdgroup_matrix<float, 8, 8> c00, c01, c10, c11;
  c00.thread_elements()[0] = 0.f;  c00.thread_elements()[1] = 0.f;
  c01.thread_elements()[0] = 0.f;  c01.thread_elements()[1] = 0.f;
  c10.thread_elements()[0] = 0.f;  c10.thread_elements()[1] = 0.f;
  c11.thread_elements()[0] = 0.f;  c11.thread_elements()[1] = 0.f;

  const uint sm = sg_id >> 1u;       // SG M-index (0..1)
  const uint sn = sg_id & 1u;        // SG N-index (0..1)
  const uint lane_in_tg = sg_id * 32u + lane;

  // Fragment indices within an 8×8 sub-tile (same derivation as mt_qmm_mma_int2)
  const uint qid = lane >> 2u;
  const uint fm  = (qid & 4u) + ((lane >> 1u) & 3u);    // frag M offset in [0,7]
  const uint fn0 = ((qid & 2u) << 1u) + ((lane & 1u) << 1u);  // frag N offset
  const uint fn1 = fn0 + 1u;

  const uint m_base = tgid.y * (uint)BM;
  const uint n_base = tgid.x * (uint)BN;

  const int M = M_c, N = N_c, K = K_c;
  const int n_groups = K / group_size;

  // Per-block tile-loading assignment
  const uint tl_row     = lane_in_tg >> 2u;    // tile row 0..31 (both X and W)
  const uint tl_sub     = lane_in_tg & 3u;     // K-quarter 0..3
  const uint k_sub_size = (uint)(group_size >> 2);  // BK/4 per sub (32 for gs=128)

  // Precomputed threadgroup-memory read indices for MMA (invariant per K-block)
  const uint xs_m0 = (sm * 16u + fm) * xs_ld;
  const uint xs_m1 = (sm * 16u + 8u + fm) * xs_ld;
  const uint ws_n00 = (sn * 16u       + fn0) * ws_ld;
  const uint ws_n01 = (sn * 16u       + fn1) * ws_ld;
  const uint ws_n10 = (sn * 16u + 8u  + fn0) * ws_ld;
  const uint ws_n11 = (sn * 16u + 8u  + fn1) * ws_ld;

  for (int g = 0; g < n_groups; g++) {
    // ---- Load X tile into xs ----
    {
      const uint m_row = m_base + tl_row;
      const bool m_ok  = m_row < (uint)M;
      const uint k_off = (uint)(g * group_size) + tl_sub * k_sub_size;
      const device T* xp = x + min(m_row, (uint)(M - 1)) * K + k_off;
      const uint xs_dst  = tl_row * xs_ld + tl_sub * k_sub_size;
      for (uint i = 0; i < k_sub_size; i++) {
        xs[xs_dst + i] = m_ok ? xp[i] : T(0);
      }
    }

    // ---- Dequant W tile into ws ----
    {
      const uint n_row  = n_base + tl_row;
      const bool n_ok   = n_row < (uint)N;
      const float sc    = n_ok ? float(scales[n_row * n_groups + g]) : 0.f;
      const uint w_base = n_row * (uint)(n_groups * bpg) + (uint)(g * bpg);
      const uint ks     = tl_sub * k_sub_size;   // K-start within group (0,32,64,96)
      const uint ke     = ks + k_sub_size;        // K-end within group
      const uint b_lo   = ks / 5u;               // first byte index (inclusive)
      const uint b_hi   = (ke - 1u) / 5u;        // last byte index  (inclusive)

      for (uint b = b_lo; b <= b_hi; b++) {
        // Read t5 byte; pad with 1 (neutral trit=1→dq=0) if out of bounds.
        // T5_TO_B4 LUT replaces 4 divmod chains: trit k = (p >> 2k) & 3.
        const uint bval = (n_ok && b < (uint)bpg) ? (uint)w[w_base + b] : 1u;
        const uint p  = T5_TO_B4[bval];
        const uint bk = b * 5u;  // K-index of trit 0 of this byte
        // Write only the trits whose K-position falls in [ks, ke)
        if (bk     >= ks && bk     < ke) ws[tl_row * ws_ld + bk    ] = T(sc * (float( p        & 0x3u) - 1.f));
        if (bk+1u  >= ks && bk+1u  < ke) ws[tl_row * ws_ld + bk+1u ] = T(sc * (float((p >> 2u) & 0x3u) - 1.f));
        if (bk+2u  >= ks && bk+2u  < ke) ws[tl_row * ws_ld + bk+2u ] = T(sc * (float((p >> 4u) & 0x3u) - 1.f));
        if (bk+3u  >= ks && bk+3u  < ke) ws[tl_row * ws_ld + bk+3u ] = T(sc * (float((p >> 6u) & 0x3u) - 1.f));
        if (bk+4u  >= ks && bk+4u  < ke) ws[tl_row * ws_ld + bk+4u ] = T(sc * (float((p >> 8u) & 0x3u) - 1.f));
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- BK/8 simdgroup MMA steps ----
    {
      simdgroup_matrix<T, 8, 8> a0, a1, b0, b1;
      for (uint ki = 0; ki < (uint)(group_size >> 3); ki++) {
        const uint ko = ki * 8u;
        a0.thread_elements()[0] = xs[xs_m0 + fn0 + ko];
        a0.thread_elements()[1] = xs[xs_m0 + fn1 + ko];
        a1.thread_elements()[0] = xs[xs_m1 + fn0 + ko];
        a1.thread_elements()[1] = xs[xs_m1 + fn1 + ko];
        b0.thread_elements()[0] = ws[ws_n00 + fm + ko];
        b0.thread_elements()[1] = ws[ws_n01 + fm + ko];
        b1.thread_elements()[0] = ws[ws_n10 + fm + ko];
        b1.thread_elements()[1] = ws[ws_n11 + fm + ko];
        simdgroup_barrier(mem_flags::mem_none);
        simdgroup_multiply_accumulate(c00, a0, b0, c00);
        simdgroup_multiply_accumulate(c01, a0, b1, c01);
        simdgroup_multiply_accumulate(c10, a1, b0, c10);
        simdgroup_multiply_accumulate(c11, a1, b1, c11);
        simdgroup_barrier(mem_flags::mem_none);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // ---- Write output ----
  const uint om0  = m_base + sm * 16u + fm;
  const uint om1  = m_base + sm * 16u + 8u + fm;
  const uint on0  = n_base + sn * 16u + fn0;
  const uint on1  = n_base + sn * 16u + fn1;
  const uint on80 = n_base + sn * 16u + 8u + fn0;
  const uint on81 = n_base + sn * 16u + 8u + fn1;

  if (om0 < (uint)M) {
    if (on0  < (uint)N) out[om0 * N + on0 ] = T(c00.thread_elements()[0]);
    if (on1  < (uint)N) out[om0 * N + on1 ] = T(c00.thread_elements()[1]);
    if (on80 < (uint)N) out[om0 * N + on80] = T(c01.thread_elements()[0]);
    if (on81 < (uint)N) out[om0 * N + on81] = T(c01.thread_elements()[1]);
  }
  if (om1 < (uint)M) {
    if (on0  < (uint)N) out[om1 * N + on0 ] = T(c10.thread_elements()[0]);
    if (on1  < (uint)N) out[om1 * N + on1 ] = T(c10.thread_elements()[1]);
    if (on80 < (uint)N) out[om1 * N + on80] = T(c11.thread_elements()[0]);
    if (on81 < (uint)N) out[om1 * N + on81] = T(c11.thread_elements()[1]);
  }
}

template <typename T, const int group_size, const int bits>
METAL_FUNC void qvm_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const int in_vec_size,
    const int out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  constexpr int num_simdgroups = 2;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  constexpr int tn = 32 / pack_factor;
  constexpr int block_size = SIMD_SIZE;

  using W_T =
      typename ConditionalType<power_of_2_bits, uint32_t, uint8_t>::type;
  const device W_T* ws = (const device W_T*)w;

  typedef float U;
  typedef struct {
    W_T wi[tn * bytes_per_pack];
  } vec_w;

  thread vec_w w_local;
  thread U result[tn * pack_factor] = {0};
  thread U scale = 1;
  thread U bias = 0;
  thread U x_local = 0;

  // Adjust positions
  const int out_vec_size_w = out_vec_size * bytes_per_pack / pack_factor;
  const int out_vec_size_g = out_vec_size / group_size;
  int out_col = pack_factor * tn * (tid.y * num_simdgroups + simd_gid);
  ws += out_col * bytes_per_pack / pack_factor + simd_lid * out_vec_size_w;
  scales += out_col / group_size + simd_lid * out_vec_size_g;
  biases += out_col / group_size + simd_lid * out_vec_size_g;
  x += tid.x * in_vec_size + simd_lid;
  y += tid.x * out_vec_size + out_col;

  if (out_col >= out_vec_size) {
    return;
  }

  // Loop over in_vec in blocks of block_size
  int remaining = in_vec_size % block_size;
  if (remaining == 0) {
    for (int i = 0; i < in_vec_size; i += block_size) {
      x_local = *x;
      scale = *scales;
      bias = *biases;
      w_local = *((device vec_w*)ws);
      qouter<U, tn * pack_factor, bits>(
          (thread uint8_t*)&w_local, x_local, scale, bias, result);

      x += block_size;
      scales += block_size * out_vec_size_g;
      biases += block_size * out_vec_size_g;
      ws += block_size * out_vec_size_w;
    }
  } else {
    for (int i = block_size; i < in_vec_size; i += block_size) {
      x_local = *x;
      scale = *scales;
      bias = *biases;
      w_local = *((device vec_w*)ws);

      qouter<U, tn * pack_factor, bits>(
          (thread uint8_t*)&w_local, x_local, scale, bias, result);

      x += block_size;
      scales += block_size * out_vec_size_g;
      biases += block_size * out_vec_size_g;
      ws += block_size * out_vec_size_w;
    }
    if (static_cast<int>(simd_lid) < remaining) {
      x_local = *x;
      scale = *scales;
      bias = *biases;
      w_local = *((device vec_w*)ws);
    } else {
      x_local = 0;
      scale = 0;
      bias = 0;
    }
    qouter<U, tn * pack_factor, bits>(
        (thread uint8_t*)&w_local, x_local, scale, bias, result);
  }

// Accumulate in the simdgroup
#pragma clang loop unroll(full)
  for (int k = 0; k < tn * pack_factor; k++) {
    result[k] = simd_sum(result[k]);
  }

  // Store the result
  if (simd_lid == 0) {
#pragma clang loop unroll(full)
    for (int k = 0; k < tn * pack_factor; k++) {
      y[k] = static_cast<T>(result[k]);
    }
  }
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
METAL_FUNC void qmm_t_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& K_eff,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  // Instantiate the appropriate BlockMMA and Loader
  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      1,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  // Set the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;

  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  y += y_row * static_cast<int64_t>(N) + y_col;

  // Make the x loader and mma operation
  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (num_els < BM) {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  } else {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);

        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  }

  // Store results to device memory
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM || num_outs < BN) {
    mma_op.store_result_safe(y, N, short2(num_outs, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}

template <
    typename T,
    const int group_size,
    const int bits,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
METAL_FUNC void qmm_n_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  // Instantiate the appropriate BlockMMA and Loader
  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, false, BK_padded, BN_padded>;
  using loader_x_t = mlx::steel::
      BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE, 1, 4>;
  using loader_w_t = QuantizedBlockLoader<
      T,
      BK,
      BN,
      BN_padded,
      0,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  auto wl = (const device uint8_t*)w;

  // Set the block
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  x += y_row * static_cast<int64_t>(K);
  wl += y_col * bytes_per_pack / pack_factor;
  scales += y_col / group_size;
  biases += y_col / group_size;
  y += y_row * static_cast<int64_t>(N) + y_col;

  // Make the x loader and mma operation
  const short num_els = min(BM, M - y_row);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, N, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (num_els < BM) {
    if ((K % BK) != 0) {
      const int k_blocks = K / BK;
      for (int k = 0; k < k_blocks; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
      const short num_k = K - k_blocks * BK;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(num_k, num_els));
      loader_w.load_safe(short2(BN, num_k));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
    } else {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  } else {
    if ((K % BK) != 0) {
      const int k_blocks = K / BK;
      for (int k = 0; k < k_blocks; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
      const short num_k = K - k_blocks * BK;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(num_k, BM));
      loader_w.load_safe(short2(BN, num_k));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
    } else {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  }

  // Store results to device memory
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM) {
    mma_op.store_result_safe(y, N, short2(BN, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}

template <typename T>
METAL_FUNC void adjust_matrix_offsets(
    const device T*& x,
    const device uint32_t*& w,
    const device T*& scales,
    const device T*& biases,
    device T*& y,
    int output_stride,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    const constant int64_t* b_strides,
    uint3 tid [[threadgroup_position_in_grid]]) {
  // Set the input/output matrices
  uint32_t x_idx = tid.z;
  uint32_t w_idx = tid.z;
  if (x_batch_ndims == 1) {
    x += x_idx * x_strides[0];
  } else {
    x += elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims);
  }
  if (w_batch_ndims == 1) {
    w += w_idx * w_strides[0];
    scales += w_idx * s_strides[0];
    biases += w_idx * b_strides[0];
  } else {
    ulong3 idx = elem_to_loc_broadcast(
        w_idx, w_shape, w_strides, s_strides, b_strides, w_batch_ndims);
    w += idx.x;
    scales += idx.y;
    biases += idx.z;
  }
  y += tid.z * output_stride;
}

template <typename T>
METAL_FUNC void adjust_matrix_offsets(
    const device T*& x,
    const device uint32_t*& w,
    const device T*& scales,
    const device T*& biases,
    const device uint32_t* lhs_indices,
    const device uint32_t* rhs_indices,
    device T*& y,
    int output_stride,
    const constant int& batch_ndims,
    const constant int* batch_shape,
    const constant int64_t* lhs_strides,
    const constant int64_t* rhs_strides,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    const constant int64_t* b_strides,
    uint3 tid [[threadgroup_position_in_grid]]) {
  // Set the input/output matrices
  uint32_t x_idx;
  uint32_t w_idx;
  if (batch_ndims == 1) {
    x_idx = lhs_indices[tid.z * lhs_strides[0]];
    w_idx = rhs_indices[tid.z * rhs_strides[0]];
  } else {
    ulong2 idx = elem_to_loc_broadcast(
        tid.z, batch_shape, lhs_strides, rhs_strides, batch_ndims);
    x_idx = lhs_indices[idx.x];
    w_idx = rhs_indices[idx.y];
  }
  if (x_batch_ndims == 1) {
    x += x_idx * x_strides[0];
  } else {
    x += elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims);
  }
  if (w_batch_ndims == 1) {
    w += w_idx * w_strides[0];
    scales += w_idx * s_strides[0];
    biases += w_idx * b_strides[0];
  } else {
    ulong3 idx = elem_to_loc_broadcast(
        w_idx, w_shape, w_strides, s_strides, b_strides, w_batch_ndims);
    w += idx.x;
    scales += idx.y;
    biases += idx.z;
  }
  y += tid.z * output_stride;
}

template <typename T, int group_size, int bits, int D, bool batched>
[[kernel]] void affine_qmv_quad(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& x_batch_ndims [[buffer(7)]],
    const constant int* x_shape [[buffer(8)]],
    const constant int64_t* x_strides [[buffer(9)]],
    const constant int& w_batch_ndims [[buffer(10)]],
    const constant int* w_shape [[buffer(11)]],
    const constant int64_t* w_strides [[buffer(12)]],
    const constant int64_t* s_strides [[buffer(13)]],
    const constant int64_t* b_strides [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint quad_gid [[quadgroup_index_in_threadgroup]],
    uint quad_lid [[thread_index_in_quadgroup]]) {
  if (batched) {
    int M = x_shape[x_batch_ndims];
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        out_vec_size * M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }
  qmv_quad_impl<T, group_size, bits, D>(
      w,
      scales,
      biases,
      x,
      y,
      in_vec_size,
      out_vec_size,
      tid,
      quad_gid,
      quad_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void affine_qmv_fast(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& x_batch_ndims [[buffer(7)]],
    const constant int* x_shape [[buffer(8)]],
    const constant int64_t* x_strides [[buffer(9)]],
    const constant int& w_batch_ndims [[buffer(10)]],
    const constant int* w_shape [[buffer(11)]],
    const constant int64_t* w_strides [[buffer(12)]],
    const constant int64_t* s_strides [[buffer(13)]],
    const constant int64_t* b_strides [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (batched) {
    int M = x_shape[x_batch_ndims];
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        out_vec_size * M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }
  qmv_fast_impl<T, group_size, bits>(
      w,
      scales,
      biases,
      x,
      y,
      in_vec_size,
      out_vec_size,
      tid,
      simd_gid,
      simd_lid);
}

// Symmetric variant: skips biases DRAM load; computes bias = -scale*ratio (I-B).
template <typename T, int group_size, int bits, bool batched>
[[kernel]] void affine_qmv_fast_sym(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],  // bound but not read
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& x_batch_ndims [[buffer(7)]],
    const constant int* x_shape [[buffer(8)]],
    const constant int64_t* x_strides [[buffer(9)]],
    const constant int& w_batch_ndims [[buffer(10)]],
    const constant int* w_shape [[buffer(11)]],
    const constant int64_t* w_strides [[buffer(12)]],
    const constant int64_t* s_strides [[buffer(13)]],
    const constant int64_t* b_strides [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (batched) {
    int M = x_shape[x_batch_ndims];
    adjust_matrix_offsets<T>(
        x, w, scales, biases, y,
        out_vec_size * M,
        x_batch_ndims, x_shape, x_strides,
        w_batch_ndims, w_shape, w_strides,
        s_strides, b_strides, tid);
  }
  qmv_fast_impl<T, group_size, bits, /*symmetric=*/true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, const int group_size, const int bits, bool batched>
[[kernel]] void affine_qmv(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& x_batch_ndims [[buffer(7)]],
    const constant int* x_shape [[buffer(8)]],
    const constant int64_t* x_strides [[buffer(9)]],
    const constant int& w_batch_ndims [[buffer(10)]],
    const constant int* w_shape [[buffer(11)]],
    const constant int64_t* w_strides [[buffer(12)]],
    const constant int64_t* s_strides [[buffer(13)]],
    const constant int64_t* b_strides [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (batched) {
    int M = x_shape[x_batch_ndims];
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        out_vec_size * M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }
  qmv_impl<T, group_size, bits>(
      w,
      scales,
      biases,
      x,
      y,
      in_vec_size,
      out_vec_size,
      tid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
    int group_size,
    int bits,
    int vecs_per_tg,
    int k_lanes,
    bool batched>
[[kernel]] void affine_qmv_wide(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    const constant int64_t* b_strides,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (batched) {
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        out_vec_size * M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }
  qmv_wide_impl<T, group_size, bits, vecs_per_tg, k_lanes>(
      w,
      scales,
      biases,
      x,
      y,
      in_vec_size,
      out_vec_size,
      M,
      tid,
      simd_gid,
      simd_lid);
}

// Symmetric wide variant: skips biases DRAM load (I-B).
template <typename T, int group_size, int bits, int vecs_per_tg, int k_lanes, bool batched>
[[kernel]] void affine_qmv_wide_sym(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,  // bound but not read
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    const constant int64_t* b_strides,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (batched) {
    adjust_matrix_offsets<T>(
        x, w, scales, biases, y,
        out_vec_size * M,
        x_batch_ndims, x_shape, x_strides,
        w_batch_ndims, w_shape, w_strides,
        s_strides, b_strides, tid);
  }
  qmv_wide_impl<T, group_size, bits, vecs_per_tg, k_lanes, /*symmetric=*/true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size, M, tid, simd_gid, simd_lid);
}

template <typename T, const int group_size, const int bits, bool batched>
[[kernel]] void affine_qvm(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& x_batch_ndims [[buffer(7)]],
    const constant int* x_shape [[buffer(8)]],
    const constant int64_t* x_strides [[buffer(9)]],
    const constant int& w_batch_ndims [[buffer(10)]],
    const constant int* w_shape [[buffer(11)]],
    const constant int64_t* w_strides [[buffer(12)]],
    const constant int64_t* s_strides [[buffer(13)]],
    const constant int64_t* b_strides [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (batched) {
    int M = x_shape[x_batch_ndims];
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        out_vec_size * M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }
  qvm_impl<T, group_size, bits>(
      w,
      scales,
      biases,
      x,
      y,
      in_vec_size,
      out_vec_size,
      tid,
      simd_gid,
      simd_lid);
}

template <typename T, const int group_size, const int bits, int split_k = 32>
[[kernel]] void affine_qvm_split_k(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& x_batch_ndims [[buffer(7)]],
    const constant int* x_shape [[buffer(8)]],
    const constant int64_t* x_strides [[buffer(9)]],
    const constant int& w_batch_ndims [[buffer(10)]],
    const constant int* w_shape [[buffer(11)]],
    const constant int64_t* w_strides [[buffer(12)]],
    const constant int64_t* s_strides [[buffer(13)]],
    const constant int64_t* b_strides [[buffer(14)]],
    const constant int& final_block_size [[buffer(15)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  int M = x_shape[x_batch_ndims];
  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      y,
      out_vec_size * M,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);

  // When (in_vec_size % split_k != 0) the final block needs to be smaller
  int in_vec_size_adj =
      tid.z % split_k == split_k - 1 ? final_block_size : in_vec_size;

  qvm_impl<T, group_size, bits>(
      w,
      scales,
      biases,
      x,
      y,
      in_vec_size_adj,
      out_vec_size,
      tid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const bool batched,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
[[kernel]] void affine_qmm_t(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    const constant int& x_batch_ndims [[buffer(8)]],
    const constant int* x_shape [[buffer(9)]],
    const constant int64_t* x_strides [[buffer(10)]],
    const constant int& w_batch_ndims [[buffer(11)]],
    const constant int* w_shape [[buffer(12)]],
    const constant int64_t* w_strides [[buffer(13)]],
    const constant int64_t* s_strides [[buffer(14)]],
    const constant int64_t* b_strides [[buffer(15)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  if (batched) {
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }
  qmm_t_impl<T, group_size, bits, aligned_N, BM, BK, BN>(
      w,
      scales,
      biases,
      x,
      y,
      Xs,
      Ws,
      K,
      N,
      M,
      K,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
[[kernel]] void affine_qmm_t_splitk(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    const constant int& k_partition_size [[buffer(8)]],
    const constant int& split_k_partition_stride [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  const int k_start = tid.z * k_partition_size;
  x += k_start;

  auto wl = (const device uint8_t*)w;
  wl += k_start * bytes_per_pack / pack_factor;
  scales += k_start / group_size;
  biases += k_start / group_size;
  y += tid.z * static_cast<int64_t>(split_k_partition_stride);

  qmm_t_impl<T, group_size, bits, aligned_N, BM, BK, BN>(
      (const device uint32_t*)wl,
      scales,
      biases,
      x,
      y,
      Xs,
      Ws,
      K,
      N,
      M,
      k_partition_size,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool batched,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
[[kernel]] void affine_qmm_n(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    const constant int& x_batch_ndims [[buffer(8)]],
    const constant int* x_shape [[buffer(9)]],
    const constant int64_t* x_strides [[buffer(10)]],
    const constant int& w_batch_ndims [[buffer(11)]],
    const constant int* w_shape [[buffer(12)]],
    const constant int64_t* w_strides [[buffer(13)]],
    const constant int64_t* s_strides [[buffer(14)]],
    const constant int64_t* b_strides [[buffer(15)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];

  if (batched) {
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }

  qmm_n_impl<T, group_size, bits, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits>
[[kernel]] void affine_gather_qmv_fast(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& in_vec_size [[buffer(7)]],
    const constant int& out_vec_size [[buffer(8)]],
    const constant int& x_batch_ndims [[buffer(9)]],
    const constant int* x_shape [[buffer(10)]],
    const constant int64_t* x_strides [[buffer(11)]],
    const constant int& w_batch_ndims [[buffer(12)]],
    const constant int* w_shape [[buffer(13)]],
    const constant int64_t* w_strides [[buffer(14)]],
    const constant int64_t* s_strides [[buffer(15)]],
    const constant int64_t* b_strides [[buffer(16)]],
    const constant int& batch_ndims [[buffer(17)]],
    const constant int* batch_shape [[buffer(18)]],
    const constant int64_t* lhs_strides [[buffer(19)]],
    const constant int64_t* rhs_strides [[buffer(20)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  int M = x_shape[x_batch_ndims];
  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      y,
      out_vec_size * M,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);
  qmv_fast_impl<T, group_size, bits>(
      w,
      scales,
      biases,
      x,
      y,
      in_vec_size,
      out_vec_size,
      tid,
      simd_gid,
      simd_lid);
}

template <typename T, int group_size, int bits>
[[kernel]] void affine_gather_qmv(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& in_vec_size [[buffer(7)]],
    const constant int& out_vec_size [[buffer(8)]],
    const constant int& x_batch_ndims [[buffer(9)]],
    const constant int* x_shape [[buffer(10)]],
    const constant int64_t* x_strides [[buffer(11)]],
    const constant int& w_batch_ndims [[buffer(12)]],
    const constant int* w_shape [[buffer(13)]],
    const constant int64_t* w_strides [[buffer(14)]],
    const constant int64_t* s_strides [[buffer(15)]],
    const constant int64_t* b_strides [[buffer(16)]],
    const constant int& batch_ndims [[buffer(17)]],
    const constant int* batch_shape [[buffer(18)]],
    const constant int64_t* lhs_strides [[buffer(19)]],
    const constant int64_t* rhs_strides [[buffer(20)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  int M = x_shape[x_batch_ndims];
  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      y,
      out_vec_size * M,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);
  qmv_impl<T, group_size, bits>(
      w,
      scales,
      biases,
      x,
      y,
      in_vec_size,
      out_vec_size,
      tid,
      simd_gid,
      simd_lid);
}

template <typename T, int group_size, int bits>
[[kernel]] void affine_gather_qvm(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& in_vec_size [[buffer(7)]],
    const constant int& out_vec_size [[buffer(8)]],
    const constant int& x_batch_ndims [[buffer(9)]],
    const constant int* x_shape [[buffer(10)]],
    const constant int64_t* x_strides [[buffer(11)]],
    const constant int& w_batch_ndims [[buffer(12)]],
    const constant int* w_shape [[buffer(13)]],
    const constant int64_t* w_strides [[buffer(14)]],
    const constant int64_t* s_strides [[buffer(15)]],
    const constant int64_t* b_strides [[buffer(16)]],
    const constant int& batch_ndims [[buffer(17)]],
    const constant int* batch_shape [[buffer(18)]],
    const constant int64_t* lhs_strides [[buffer(19)]],
    const constant int64_t* rhs_strides [[buffer(20)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  int M = x_shape[x_batch_ndims];
  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      y,
      out_vec_size * M,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);
  qvm_impl<T, group_size, bits>(
      w,
      scales,
      biases,
      x,
      y,
      in_vec_size,
      out_vec_size,
      tid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
[[kernel]] void affine_gather_qmm_t(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant int& M [[buffer(9)]],
    const constant int& x_batch_ndims [[buffer(10)]],
    const constant int* x_shape [[buffer(11)]],
    const constant int64_t* x_strides [[buffer(12)]],
    const constant int& w_batch_ndims [[buffer(13)]],
    const constant int* w_shape [[buffer(14)]],
    const constant int64_t* w_strides [[buffer(15)]],
    const constant int64_t* s_strides [[buffer(16)]],
    const constant int64_t* b_strides [[buffer(17)]],
    const constant int& batch_ndims [[buffer(18)]],
    const constant int* batch_shape [[buffer(19)]],
    const constant int64_t* lhs_strides [[buffer(20)]],
    const constant int64_t* rhs_strides [[buffer(21)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      y,
      M * N,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);
  qmm_t_impl<T, group_size, bits, aligned_N, BM, BK, BN>(
      w,
      scales,
      biases,
      x,
      y,
      Xs,
      Ws,
      K,
      N,
      M,
      K,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
[[kernel]] void affine_gather_qmm_n(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant int& M [[buffer(9)]],
    const constant int& x_batch_ndims [[buffer(10)]],
    const constant int* x_shape [[buffer(11)]],
    const constant int64_t* x_strides [[buffer(12)]],
    const constant int& w_batch_ndims [[buffer(13)]],
    const constant int* w_shape [[buffer(14)]],
    const constant int64_t* w_strides [[buffer(15)]],
    const constant int64_t* s_strides [[buffer(16)]],
    const constant int64_t* b_strides [[buffer(17)]],
    const constant int& batch_ndims [[buffer(18)]],
    const constant int* batch_shape [[buffer(19)]],
    const constant int64_t* lhs_strides [[buffer(20)]],
    const constant int64_t* rhs_strides [[buffer(21)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];

  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      y,
      M * N,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);
  qmm_n_impl<T, group_size, bits, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    int group_size,
    int bits,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose>
[[kernel]] void affine_gather_qmm_rhs(
    const device T* x [[buffer(0)]],
    const device uint32_t* w [[buffer(1)]],
    const device T* scales [[buffer(2)]],
    const device T* biases [[buffer(3)]],
    const device uint32_t* indices [[buffer(4)]],
    device T* y [[buffer(5)]],
    const constant int& M [[buffer(6)]],
    const constant int& N [[buffer(7)]],
    const constant int& K [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  using mma_t = mlx::steel::BlockMMA<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      false,
      transpose,
      BK_padded,
      transpose ? BK_padded : BN_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T,
      transpose ? BN : BK,
      transpose ? BK : BN,
      transpose ? BK_padded : BN_padded,
      transpose,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[transpose ? BN * BK_padded : BK * BN_padded];

  // Compute the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int N_w = N * bytes_per_pack / pack_factor;
  const int N_g = N / group_size;
  const int K_it = K / BK;
  const size_t stride_w = transpose ? N * K_w : K * N_w;
  const size_t stride_s = transpose ? N * K_g : K * N_g;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  const size_t y_row_long = size_t(y_row);
  const size_t y_col_long = size_t(y_col);

  // Prepare threadgroup bounds
  const short tgp_bm = align_M ? BM : short(min(BM, M - y_row));
  const short tgp_bn = align_N ? BN : short(min(BN, N - y_col));

  // Calculate the final tiles in the case that K is not aligned
  const int k_remain = K - K_it * BK;
  const short2 tile_x = short2(k_remain, tgp_bm);
  const short2 tile_w =
      transpose ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);

  // Move x and output to the correct block
  auto wl = (const device uint8_t*)w;
  x += y_row_long * K;
  y += y_row_long * N + y_col_long;
  wl += transpose ? y_col_long * K_w : y_col * bytes_per_pack / pack_factor;
  scales += transpose ? y_col_long * K_g : y_col / group_size;
  biases += transpose ? y_col_long * K_g : y_col / group_size;

  // Do as many matmuls as necessary
  uint32_t index;
  short offset;
  uint32_t index_next = indices[y_row];
  short offset_next = 0;
  int n = 0;
  while (n < tgp_bm) {
    n++;
    offset = offset_next;
    index = index_next;
    offset_next = tgp_bm;
    for (; n < tgp_bm; n++) {
      if (indices[y_row + n] != index) {
        offset_next = n;
        index_next = indices[y_row + n];
        break;
      }
    }
    threadgroup_barrier(mem_flags::mem_none);

    // Prepare threadgroup mma operation
    thread mma_t mma_op(simd_group_id, simd_lane_id);

    // Prepare threadgroup loading operations
    thread loader_x_t loader_x(x, K, Xs, simd_group_id, simd_lane_id);
    thread loader_w_t loader_w(
        wl + index * stride_w,
        scales + index * stride_s,
        biases + index * stride_s,
        transpose ? K : N,
        Ws,
        simd_group_id,
        simd_lane_id);

    // Matrices are all aligned check nothing
    if (align_M && align_N) {
      gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
      if (!align_K) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
      }

      // Store results to device memory
      if (offset_next - offset == BM) {
        mma_op.store_result(y, N);
      } else {
        mma_op.store_result_slice(
            y, N, short2(0, offset), short2(BN, offset_next));
      }
    } else {
      // Tile aligned so check outside of the hot loop
      if ((align_M || tgp_bm == BM) && (align_N || tgp_bn == BN)) {
        gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          gemm_loop_finalize(
              Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
        }

        // Store results to device memory
        if (offset_next - offset == BM) {
          mma_op.store_result(y, N);
        } else {
          mma_op.store_result_slice(
              y, N, short2(0, offset), short2(BN, offset_next));
        }
      }

      // Tile partially aligned check rows
      else if (align_N || tgp_bn == BN) {
        gemm_loop_unaligned<false, true, transpose>(
            Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          gemm_loop_finalize(
              Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
        }
        mma_op.store_result_slice(
            y, N, short2(0, offset), short2(BN, offset_next));
      }

      // Tile partially aligned check cols
      else if (align_M || tgp_bm == BM) {
        gemm_loop_unaligned<true, false, transpose>(
            Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          gemm_loop_finalize(
              Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
        }
        mma_op.store_result_slice(
            y, N, short2(0, offset), short2(tgp_bn, offset_next));
      }

      // Nothing aligned so check both rows and cols
      else {
        gemm_loop_unaligned<false, false, transpose>(
            Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          gemm_loop_finalize(
              Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
        }
        mma_op.store_result_slice(
            y, N, short2(0, offset), short2(tgp_bn, offset_next));
      }
    }
  }
}

template <typename T, const int group_size, const int bits>
[[kernel]] void affine_quantize(
    const device T* w [[buffer(0)]],
    device uint8_t* out [[buffer(1)]],
    device T* scales [[buffer(2)]],
    device T* biases [[buffer(3)]],
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  constexpr float eps = 1e-7;
  constexpr int simd_size = 32;
  constexpr float n_bins = (1 << bits) - 1;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int values_per_reduce = group_size / simd_size;
  constexpr int writes_per_reduce = pack_factor / values_per_reduce;
  constexpr int writes_per_pack =
      writes_per_reduce > 1 ? 1 : values_per_reduce / pack_factor;
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;

  static_assert(
      group_size % simd_size == 0,
      "Group size must be divisible by simd size.");

  size_t offset = index.x + grid_dim.x * size_t(index.y);
  size_t in_index = offset * values_per_reduce;
  size_t out_index = power_of_2_bits
      ? offset * writes_per_pack
      : offset * bytes_per_pack / writes_per_reduce;

  float w_thread[values_per_reduce];
  float w_min = Limits<T>::max;
  float w_max = 0;

#pragma clang loop unroll(full)
  for (int i = 0; i < values_per_reduce; i++) {
    float val = w[in_index + i];
    w_thread[i] = val;
    w_min = min(w_min, val);
    w_max = max(w_max, val);
  }

  w_min = simd_min(w_min);
  w_max = simd_max(w_max);

  float scale;
  float bias;

  if (bits == 1) {
    // Affine 1-bit: bit 0 -> w_min, bit 1 -> w_max
    scale = max(w_max - w_min, eps);
    bias = w_min;
  } else {
    scale = max((w_max - w_min) / n_bins, eps);
    bool side = abs(w_min) > abs(w_max);
    scale = side ? scale : -scale;
    float edge = side ? w_min : w_max;
    float q0 = round(edge / scale);
    bool at_zero = q0 == 0.0f;
    scale = at_zero ? scale : edge / q0;
    bias = at_zero ? 0 : edge;
  }

  // Write out the scales and biases
  size_t gindex = in_index / group_size;
  if (in_index % group_size == 0) {
    scales[gindex] = static_cast<T>(scale);
    biases[gindex] = static_cast<T>(bias);
  }

  using OutType = metal::conditional_t<bits == 5, uint64_t, uint32_t>;
  OutType output = 0;

#pragma clang loop unroll(full)
  for (int i = 0; i < values_per_reduce; i++) {
    uint8_t val = min(round((w_thread[i] - bias) / scale), n_bins);
    if (bits == 8) {
      output = val;
    } else {
      output |= val << (bits * (i % pack_factor));
    }

    if (pack_factor < values_per_reduce && i % pack_factor == pack_factor - 1) {
      out[out_index + i / pack_factor] = output;
      output = 0;
    } else {
#pragma clang loop unroll(full)
      for (int j = 1; j < writes_per_reduce; j++) {
        uint8_t sval = simd_shuffle_down(val, j);
        output |= static_cast<OutType>(sval)
            << (bits * (j * values_per_reduce + i));
      }
    }
  }
  if (bits == 3 || bits == 6) {
    if (in_index % pack_factor == 0 && out_index % bytes_per_pack == 0) {
      out[out_index] = output & 0xff;
      out[out_index + 1] = (output & 0xff00) >> 8;
      out[out_index + 2] = (output & 0xff0000) >> 16;
    }
  } else if (bits == 5) {
    if (in_index % pack_factor == 0 && out_index % bytes_per_pack == 0) {
      out[out_index] = output & 0xff;
      out[out_index + 1] = (output & 0xff00) >> 8;
      out[out_index + 2] = (output & 0xff0000) >> 16;
      out[out_index + 3] = (output & 0xff000000) >> 24;
      out[out_index + 4] = (output & 0xff00000000) >> 32;
    }
  } else {
    if (writes_per_reduce > 0 && out_index % writes_per_reduce == 0) {
      out[out_index / writes_per_reduce] = output;
    }
  }
}

template <typename T, const int group_size, const int bits>
[[kernel]] void affine_dequantize(
    const device uint8_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    device T* out [[buffer(3)]],
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  size_t offset = index.x + grid_dim.x * size_t(index.y);
  size_t oindex = offset * pack_factor;
  size_t gindex = oindex / group_size;
  T scale = scales[gindex];
  T bias = biases[gindex];

  out += oindex;

  if (bits == 3) {
    w += offset * bytes_per_pack;
    out[0] = (w[0] & 0x7) * scale + bias;
    out[1] = ((w[0] & 0x38) >> 3) * scale + bias;
    out[2] = (((w[0] & 0xc0) >> 6) + ((w[1] & 0x1) << 2)) * scale + bias;
    out[3] = ((w[1] & 0xe) >> 1) * scale + bias;
    out[4] = ((w[1] & 0x70) >> 4) * scale + bias;
    out[5] = (((w[1] & 0x80) >> 7) + ((w[2] & 0x3) << 1)) * scale + bias;
    out[6] = ((w[2] & 0x1c) >> 2) * scale + bias;
    out[7] = ((w[2] & 0xe0) >> 5) * scale + bias;
  } else if (bits == 5) {
    w += offset * bytes_per_pack;
    out[0] = (w[0] & 0x1f) * scale + bias;
    out[1] = (((w[0] & 0xe0) >> 5) + ((w[1] & 0x3) << 3)) * scale + bias;
    out[2] = ((w[1] & 0x7c) >> 2) * scale + bias;
    out[3] = (((w[1] & 0x80) >> 7) + ((w[2] & 0xf) << 1)) * scale + bias;
    out[4] = (((w[2] & 0xf0) >> 4) + ((w[3] & 0x1) << 4)) * scale + bias;
    out[5] = ((w[3] & 0x3e) >> 1) * scale + bias;
    out[6] = (((w[3] & 0xc0) >> 6) + ((w[4] & 0x7) << 2)) * scale + bias;
    out[7] = ((w[4] & 0xf8) >> 3) * scale + bias;
  } else if (bits == 6) {
    w += offset * bytes_per_pack;
    out[0] = (w[0] & 0x3f) * scale + bias;
    out[1] = (((w[0] >> 6) & 0x03) + ((w[1] & 0x0f) << 2)) * scale + bias;
    out[2] = (((w[1] >> 4) & 0x0f) + ((w[2] & 0x03) << 4)) * scale + bias;
    out[3] = ((w[2] >> 2) & 0x3f) * scale + bias;
  } else {
    uint val = w[offset];
#pragma clang loop unroll(full)
    for (int i = 0; i < pack_factor; i++) {
      uint8_t d;
      if (bits == 1) {
        d = (val >> i) & 0x01;
      } else if (bits == 2) {
        d = (val >> (bits * i)) & 0x03;
      } else if (bits == 4) {
        d = (val >> (bits * i)) & 0x0f;
      } else if (bits == 8) {
        d = val;
      }
      out[i] = scale * d + bias;
    }
  }
}
