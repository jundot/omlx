// SPDX-License-Identifier: Apache-2.0
// Row-wise rotation of a batch KV bank along its token axis.
// The array is viewed as [rows, capacity, inner].
// Row r's first `length` positions rotate right by shifts[r]; later positions keep their values.

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

constant constexpr int kRollBlock = 256;
constant constexpr int kRollTile = 16;

// Save the shift positions before each block, since the in-place pass overwrites them.
// Block 0 takes the wrapped tail of the row.
template <typename T>
[[kernel]] void kv_roll_halo(
    const device T* x [[buffer(0)]],
    const device int* shifts [[buffer(1)]],
    device T* halo [[buffer(2)]],
    const constant int& capacity [[buffer(3)]],
    const constant int& length [[buffer(4)]],
    const constant int& inner [[buffer(5)]],
    const constant int& max_shift [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]) {
  const int t = int(gid.x) / inner;
  const int i = int(gid.x) % inner;
  const int block = int(gid.y);
  const int row = int(gid.z);
  const int shift = shifts[row];
  if (t >= shift) {
    return;
  }
  int src = block * kRollBlock - shift + t;
  if (src < 0) {
    src += length;
  }
  const int blocks = (length + kRollBlock - 1) / kRollBlock;
  halo[((size_t(row) * blocks + block) * max_shift + t) * inner + i] =
      x[(size_t(row) * capacity + src) * inner + i];
}

// One threadgroup owns one block of one row for a tile of inner columns.
// It reads every source value before it writes any of them.
template <typename T>
[[kernel]] void kv_roll_apply(
    device T* x [[buffer(0)]],
    const device int* shifts [[buffer(1)]],
    const device T* halo [[buffer(2)]],
    const constant int& capacity [[buffer(3)]],
    const constant int& length [[buffer(4)]],
    const constant int& inner [[buffer(5)]],
    const constant int& max_shift [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]],
    uint3 threads3 [[threads_per_threadgroup]]) {
  const int lid = int(lid3.x);
  const int threads = int(threads3.x);
  threadgroup T stage[kRollBlock * kRollTile];
  const int row = int(group.z);
  const int shift = shifts[row];
  if (shift == 0) {
    return;
  }
  const int block = int(group.y);
  const int start = block * kRollBlock;
  const int count = min(kRollBlock, length - start);
  const int i0 = int(group.x) * kRollTile;
  const int tile = min(kRollTile, inner - i0);
  const int blocks = (length + kRollBlock - 1) / kRollBlock;
  const size_t row_base = size_t(row) * capacity;
  const size_t halo_base = (size_t(row) * blocks + block) * max_shift;
  const int total = count * tile;

  for (int e = lid; e < total; e += threads) {
    const int j = e / tile;
    const int c = i0 + e % tile;
    const int src = start + j - shift;
    stage[e] = src >= start
        ? x[(row_base + src) * inner + c]
        : halo[(halo_base + src - (start - shift)) * inner + c];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int e = lid; e < total; e += threads) {
    const int j = e / tile;
    const int c = i0 + e % tile;
    x[(row_base + start + j) * inner + c] = stage[e];
  }
}

// Out-of-place rotation for inputs that cannot be updated in place.
template <typename T>
[[kernel]] void kv_roll_copy(
    const device T* x [[buffer(0)]],
    const device int* shifts [[buffer(1)]],
    device T* out [[buffer(2)]],
    const constant int& capacity [[buffer(3)]],
    const constant int& length [[buffer(4)]],
    const constant int& inner [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]) {
  const int i = int(gid.x);
  const int pos = int(gid.y);
  const int row = int(gid.z);
  int src = pos;
  if (pos < length) {
    src = pos - shifts[row];
    if (src < 0) {
      src += length;
    }
  }
  const size_t row_base = size_t(row) * capacity;
  out[(row_base + pos) * inner + i] = x[(row_base + src) * inner + i];
}

#define instantiate_kv_roll(tname, type)                         \
  instantiate_kernel("kv_roll_halo_" #tname, kv_roll_halo, type)   \
  instantiate_kernel("kv_roll_apply_" #tname, kv_roll_apply, type) \
  instantiate_kernel("kv_roll_copy_" #tname, kv_roll_copy, type)

instantiate_kv_roll(b1, uint8_t)
instantiate_kv_roll(b2, uint16_t)
instantiate_kv_roll(b4, uint32_t)
