#include "mlx/backend/metal/kernels/utils.h"

[[kernel]] void qwen35_ane_commit_guard(uint gid [[thread_position_in_grid]]) {
  (void)gid;
}

template <typename T>
[[kernel]] void qwen35_ane_pack_input(const device T *x [[buffer(0)]],
                                      device float16_t *planar [[buffer(1)]],
                                      constant int &M [[buffer(2)]],
                                      constant int &K [[buffer(3)]],
                                      uint2 gid [[thread_position_in_grid]]) {
  const uint k = gid.x;
  const uint m = gid.y;
  if (m < static_cast<uint>(M) && k < static_cast<uint>(K)) {
    planar[k * M + m] = static_cast<float16_t>(x[m * K + k]);
  }
}

template <typename T>
[[kernel]] void qwen35_ane_pack_input_dual(
    const device T *x [[buffer(0)]], device float16_t *planar0 [[buffer(1)]],
    device float16_t *planar1 [[buffer(2)]],
    constant int &M [[buffer(3)]], constant int &K [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint k = gid.x;
  const uint m = gid.y;
  if (m < static_cast<uint>(M) && k < static_cast<uint>(K)) {
    const float16_t value = static_cast<float16_t>(x[m * K + k]);
    planar0[k * M + m] = value;
    planar1[k * M + m] = value;
  }
}

template <typename T>
[[kernel]] void qwen35_cpu_pack_input(
    const device T *x [[buffer(0)]], device float16_t *rows [[buffer(1)]],
    constant int &M [[buffer(2)]], constant int &K [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint k = gid.x;
  const uint m = gid.y;
  if (m < static_cast<uint>(M) && k < static_cast<uint>(K)) {
    rows[m * K + k] = static_cast<float16_t>(x[m * K + k]);
  }
}

template <typename T>
[[kernel]] void qwen35_ane_pack_input_dual_cpu(
    const device T *x [[buffer(0)]], device float16_t *planar0 [[buffer(1)]],
    device float16_t *planar1 [[buffer(2)]],
    device float16_t *cpu_rows [[buffer(3)]],
    constant int &M [[buffer(4)]], constant int &K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint k = gid.x;
  const uint m = gid.y;
  if (m < static_cast<uint>(M) && k < static_cast<uint>(K)) {
    const float16_t value = static_cast<float16_t>(x[m * K + k]);
    planar0[k * M + m] = value;
    planar1[k * M + m] = value;
    cpu_rows[m * K + k] = value;
  }
}

template <typename T>
[[kernel]] void qwen35_ane_pack_input_cpu(
    const device T *x [[buffer(0)]], device float16_t *planar [[buffer(1)]],
    device float16_t *cpu_rows [[buffer(2)]],
    constant int &M [[buffer(3)]], constant int &K [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint k = gid.x;
  const uint m = gid.y;
  if (m < static_cast<uint>(M) && k < static_cast<uint>(K)) {
    const float16_t value = static_cast<float16_t>(x[m * K + k]);
    planar[k * M + m] = value;
    cpu_rows[m * K + k] = value;
  }
}

template <typename T>
[[kernel]] void qwen35_cpu_merge_output(
    const device float16_t *cpu_rows [[buffer(0)]],
    const device T *gpu_rows [[buffer(1)]], device T *output [[buffer(2)]],
    constant int &M [[buffer(3)]], constant int &cpu_n [[buffer(4)]],
    constant int &gpu_n [[buffer(5)]], uint2 gid [[thread_position_in_grid]]) {
  const uint n = gid.x;
  const uint m = gid.y;
  const uint total_n = static_cast<uint>(cpu_n + gpu_n);
  if (m >= static_cast<uint>(M) || n >= total_n) {
    return;
  }
  output[m * total_n + n] = n < static_cast<uint>(cpu_n)
      ? static_cast<T>(cpu_rows[m * cpu_n + n])
      : gpu_rows[m * gpu_n + n - cpu_n];
}

template <typename T>
[[kernel]] void qwen35_ane_merge_output(
    const device float16_t *ane_planar [[buffer(0)]],
    const device T *gpu_rows [[buffer(1)]], device T *output [[buffer(2)]],
    constant int &M [[buffer(3)]], constant int &ane_n [[buffer(4)]],
    constant int &gpu_n [[buffer(5)]], uint2 gid [[thread_position_in_grid]]) {
  const uint n = gid.x;
  const uint m = gid.y;
  const uint total_n = static_cast<uint>(ane_n + gpu_n);
  if (m >= static_cast<uint>(M) || n >= total_n) {
    return;
  }
  if (n < static_cast<uint>(ane_n)) {
    output[m * total_n + n] = static_cast<T>(ane_planar[n * M + m]);
  } else {
    output[m * total_n + n] = gpu_rows[m * gpu_n + n - ane_n];
  }
}

template <typename T>
[[kernel]] void qwen35_ane_merge_swiglu_output(
    const device float16_t *ane_planar [[buffer(0)]],
    const device T *gpu_rows [[buffer(1)]], device T *activation [[buffer(2)]],
    constant int &M [[buffer(3)]], constant int &ane_hidden [[buffer(4)]],
    constant int &gpu_hidden [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint n = gid.x;
  const uint m = gid.y;
  const uint total_hidden = static_cast<uint>(ane_hidden + gpu_hidden);
  if (m >= static_cast<uint>(M) || n >= total_hidden) {
    return;
  }

  float gate;
  float up;
  if (n < static_cast<uint>(ane_hidden)) {
    gate = static_cast<float>(ane_planar[n * M + m]);
    up = static_cast<float>(
        ane_planar[(static_cast<uint>(ane_hidden) + n) * M + m]);
  } else {
    const uint suffix = n - static_cast<uint>(ane_hidden);
    const uint base = m * static_cast<uint>(2 * gpu_hidden);
    gate = static_cast<float>(gpu_rows[base + suffix]);
    up = static_cast<float>(
        gpu_rows[base + static_cast<uint>(gpu_hidden) + suffix]);
  }
  activation[m * total_hidden + n] =
      static_cast<T>(gate * up / (1.0f + exp(-gate)));
}

template <typename T>
[[kernel]] void qwen35_ane_merge_dual_output(
    const device float16_t *ane0_planar [[buffer(0)]],
    const device float16_t *ane1_planar [[buffer(1)]],
    const device T *gpu_rows [[buffer(2)]], device T *output [[buffer(3)]],
    constant int &M [[buffer(4)]], constant int &ane0_n [[buffer(5)]],
    constant int &ane1_n [[buffer(6)]], constant int &gpu_n [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint n = gid.x;
  const uint m = gid.y;
  const uint ane_n = static_cast<uint>(ane0_n + ane1_n);
  const uint total_n = ane_n + static_cast<uint>(gpu_n);
  if (m >= static_cast<uint>(M) || n >= total_n) {
    return;
  }
  if (n < static_cast<uint>(ane0_n)) {
    output[m * total_n + n] = static_cast<T>(ane0_planar[n * M + m]);
  } else if (n < ane_n) {
    const uint local = n - static_cast<uint>(ane0_n);
    output[m * total_n + n] = static_cast<T>(ane1_planar[local * M + m]);
  } else {
    output[m * total_n + n] = gpu_rows[m * gpu_n + n - ane_n];
  }
}

template <typename T>
[[kernel]] void qwen35_ane_merge_dual_swiglu_output(
    const device float16_t *ane0_planar [[buffer(0)]],
    const device float16_t *ane1_planar [[buffer(1)]],
    const device T *gpu_rows [[buffer(2)]],
    device T *activation [[buffer(3)]], constant int &M [[buffer(4)]],
    constant int &ane0_hidden [[buffer(5)]],
    constant int &ane1_hidden [[buffer(6)]],
    constant int &gpu_hidden [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint n = gid.x;
  const uint m = gid.y;
  const uint ane_hidden = static_cast<uint>(ane0_hidden + ane1_hidden);
  const uint total_hidden = ane_hidden + static_cast<uint>(gpu_hidden);
  if (m >= static_cast<uint>(M) || n >= total_hidden) {
    return;
  }

  float gate;
  float up;
  if (n < static_cast<uint>(ane0_hidden)) {
    gate = static_cast<float>(ane0_planar[n * M + m]);
    up = static_cast<float>(
        ane0_planar[(static_cast<uint>(ane0_hidden) + n) * M + m]);
  } else if (n < ane_hidden) {
    const uint local = n - static_cast<uint>(ane0_hidden);
    gate = static_cast<float>(ane1_planar[local * M + m]);
    up = static_cast<float>(
        ane1_planar[(static_cast<uint>(ane1_hidden) + local) * M + m]);
  } else {
    const uint suffix = n - ane_hidden;
    const uint base = m * static_cast<uint>(2 * gpu_hidden);
    gate = static_cast<float>(gpu_rows[base + suffix]);
    up = static_cast<float>(
        gpu_rows[base + static_cast<uint>(gpu_hidden) + suffix]);
  }
  activation[m * total_hidden + n] =
      static_cast<T>(gate * up / (1.0f + exp(-gate)));
}

template <typename T>
[[kernel]] void qwen35_ane_merge_dual_cpu_swiglu_output(
    const device float16_t *ane0_planar [[buffer(0)]],
    const device float16_t *ane1_planar [[buffer(1)]],
    const device float16_t *cpu_rows [[buffer(2)]],
    const device T *gpu_rows [[buffer(3)]],
    device T *activation [[buffer(4)]], constant int &M [[buffer(5)]],
    constant int &ane0_hidden [[buffer(6)]],
    constant int &ane1_hidden [[buffer(7)]],
    constant int &cpu_hidden [[buffer(8)]],
    constant int &gpu_hidden [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint n = gid.x;
  const uint m = gid.y;
  const uint ane_hidden = static_cast<uint>(ane0_hidden + ane1_hidden);
  const uint cpu_end = ane_hidden + static_cast<uint>(cpu_hidden);
  const uint total_hidden = cpu_end + static_cast<uint>(gpu_hidden);
  if (m >= static_cast<uint>(M) || n >= total_hidden) {
    return;
  }

  float gate;
  float up;
  if (n < static_cast<uint>(ane0_hidden)) {
    gate = static_cast<float>(ane0_planar[n * M + m]);
    up = static_cast<float>(
        ane0_planar[(static_cast<uint>(ane0_hidden) + n) * M + m]);
  } else if (n < ane_hidden) {
    const uint local = n - static_cast<uint>(ane0_hidden);
    gate = static_cast<float>(ane1_planar[local * M + m]);
    up = static_cast<float>(
        ane1_planar[(static_cast<uint>(ane1_hidden) + local) * M + m]);
  } else if (n < cpu_end) {
    const uint local = n - ane_hidden;
    const uint base = m * static_cast<uint>(2 * cpu_hidden);
    gate = static_cast<float>(cpu_rows[base + local]);
    up = static_cast<float>(
        cpu_rows[base + static_cast<uint>(cpu_hidden) + local]);
  } else {
    const uint local = n - cpu_end;
    const uint base = m * static_cast<uint>(2 * gpu_hidden);
    gate = static_cast<float>(gpu_rows[base + local]);
    up = static_cast<float>(
        gpu_rows[base + static_cast<uint>(gpu_hidden) + local]);
  }
  activation[m * total_hidden + n] =
      static_cast<T>(gate * up / (1.0f + exp(-gate)));
}

template <typename T>
[[kernel]] void qwen35_ane_merge_dual_cpu_output(
    const device float16_t *ane0_planar [[buffer(0)]],
    const device float16_t *ane1_planar [[buffer(1)]],
    const device float16_t *cpu_rows [[buffer(2)]],
    const device T *gpu_rows [[buffer(3)]],
    device T *output [[buffer(4)]], constant int &M [[buffer(5)]],
    constant int &ane0_n [[buffer(6)]],
    constant int &ane1_n [[buffer(7)]],
    constant int &cpu_n [[buffer(8)]],
    constant int &gpu_n [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint n = gid.x;
  const uint m = gid.y;
  const uint total_n = static_cast<uint>(ane0_n + ane1_n + cpu_n + gpu_n);
  if (m >= static_cast<uint>(M) || n >= total_n) {
    return;
  }
  if (n < static_cast<uint>(ane0_n)) {
    output[m * total_n + n] = static_cast<T>(ane0_planar[n * M + m]);
  } else if (n < static_cast<uint>(ane0_n + ane1_n)) {
    const uint local = n - static_cast<uint>(ane0_n);
    output[m * total_n + n] = static_cast<T>(ane1_planar[local * M + m]);
  } else if (n < static_cast<uint>(ane0_n + ane1_n + cpu_n)) {
    const uint local = n - static_cast<uint>(ane0_n + ane1_n);
    output[m * total_n + n] = static_cast<T>(cpu_rows[m * cpu_n + local]);
  } else {
    const uint local = n - static_cast<uint>(ane0_n + ane1_n + cpu_n);
    output[m * total_n + n] = gpu_rows[m * gpu_n + local];
  }
}

template <typename T>
[[kernel]] void qwen35_ane_merge_cpu_swiglu_output(
    const device float16_t *ane_planar [[buffer(0)]],
    const device float16_t *cpu_rows [[buffer(1)]],
    const device T *gpu_rows [[buffer(2)]],
    device T *activation [[buffer(3)]], constant int &M [[buffer(4)]],
    constant int &ane_hidden [[buffer(5)]],
    constant int &cpu_hidden [[buffer(6)]],
    constant int &gpu_hidden [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint n = gid.x;
  const uint m = gid.y;
  const uint cpu_end = static_cast<uint>(ane_hidden + cpu_hidden);
  const uint total_hidden = cpu_end + static_cast<uint>(gpu_hidden);
  if (m >= static_cast<uint>(M) || n >= total_hidden) {
    return;
  }

  float gate;
  float up;
  if (n < static_cast<uint>(ane_hidden)) {
    gate = static_cast<float>(ane_planar[n * M + m]);
    up = static_cast<float>(
        ane_planar[(static_cast<uint>(ane_hidden) + n) * M + m]);
  } else if (n < cpu_end) {
    const uint local = n - static_cast<uint>(ane_hidden);
    const uint base = m * static_cast<uint>(2 * cpu_hidden);
    gate = static_cast<float>(cpu_rows[base + local]);
    up = static_cast<float>(
        cpu_rows[base + static_cast<uint>(cpu_hidden) + local]);
  } else {
    const uint local = n - cpu_end;
    const uint base = m * static_cast<uint>(2 * gpu_hidden);
    gate = static_cast<float>(gpu_rows[base + local]);
    up = static_cast<float>(
        gpu_rows[base + static_cast<uint>(gpu_hidden) + local]);
  }
  activation[m * total_hidden + n] =
      static_cast<T>(gate * up / (1.0f + exp(-gate)));
}

template <typename T>
[[kernel]] void qwen35_ane_merge_cpu_output(
    const device float16_t *ane_planar [[buffer(0)]],
    const device float16_t *cpu_rows [[buffer(1)]],
    const device T *gpu_rows [[buffer(2)]], device T *output [[buffer(3)]],
    constant int &M [[buffer(4)]], constant int &ane_n [[buffer(5)]],
    constant int &cpu_n [[buffer(6)]], constant int &gpu_n [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint n = gid.x;
  const uint m = gid.y;
  const uint cpu_end = static_cast<uint>(ane_n + cpu_n);
  const uint total_n = cpu_end + static_cast<uint>(gpu_n);
  if (m >= static_cast<uint>(M) || n >= total_n) {
    return;
  }
  if (n < static_cast<uint>(ane_n)) {
    output[m * total_n + n] = static_cast<T>(ane_planar[n * M + m]);
  } else if (n < cpu_end) {
    const uint local = n - static_cast<uint>(ane_n);
    output[m * total_n + n] = static_cast<T>(cpu_rows[m * cpu_n + local]);
  } else {
    const uint local = n - cpu_end;
    output[m * total_n + n] = gpu_rows[m * gpu_n + local];
  }
}

template <typename T>
[[kernel]] void qwen35_ane_swiglu_suffix(
    const device T *gate_up [[buffer(0)]], device T *activation [[buffer(1)]],
    constant int &M [[buffer(2)]], constant int &N [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint n = gid.x;
  const uint m = gid.y;
  if (m >= static_cast<uint>(M) || n >= static_cast<uint>(N)) {
    return;
  }
  const uint base = m * 2 * N;
  const float gate = static_cast<float>(gate_up[base + n]);
  const float up = static_cast<float>(gate_up[base + N + n]);
  activation[m * N + n] = static_cast<T>(gate * up / (1.0f + exp(-gate)));
}

template <typename T>
[[kernel]] void qwen35_ane_sum_output(
    const device float16_t *ane_planar [[buffer(0)]],
    const device T *gpu_rows [[buffer(1)]], device T *output [[buffer(2)]],
    constant int &M [[buffer(3)]], constant int &N [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint n = gid.x;
  const uint m = gid.y;
  if (m < static_cast<uint>(M) && n < static_cast<uint>(N)) {
    const float total = static_cast<float>(ane_planar[n * M + m]) +
                        static_cast<float>(gpu_rows[m * N + n]);
    output[m * N + n] = static_cast<T>(total);
  }
}

template <typename T>
[[kernel]] void qwen35_ane_sum_dual_output(
    const device float16_t *ane0_planar [[buffer(0)]],
    const device float16_t *ane1_planar [[buffer(1)]],
    const device T *gpu_rows [[buffer(2)]], device T *output [[buffer(3)]],
    constant int &M [[buffer(4)]], constant int &N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint n = gid.x;
  const uint m = gid.y;
  if (m < static_cast<uint>(M) && n < static_cast<uint>(N)) {
    const float total = static_cast<float>(ane0_planar[n * M + m]) +
                        static_cast<float>(ane1_planar[n * M + m]) +
                        static_cast<float>(gpu_rows[m * N + n]);
    output[m * N + n] = static_cast<T>(total);
  }
}

template <typename T>
[[kernel]] void qwen35_ane_sum_dual_cpu_output(
    const device float16_t *ane0_planar [[buffer(0)]],
    const device float16_t *ane1_planar [[buffer(1)]],
    const device float16_t *cpu_rows [[buffer(2)]],
    const device T *gpu_rows [[buffer(3)]], device T *output [[buffer(4)]],
    constant int &M [[buffer(5)]], constant int &N [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint n = gid.x;
  const uint m = gid.y;
  if (m < static_cast<uint>(M) && n < static_cast<uint>(N)) {
    const float total = static_cast<float>(ane0_planar[n * M + m]) +
                        static_cast<float>(ane1_planar[n * M + m]) +
                        cpu_rows[m * N + n] +
                        static_cast<float>(gpu_rows[m * N + n]);
    output[m * N + n] = static_cast<T>(total);
  }
}

#define instantiate_ane_io(type)                                               \
  instantiate_kernel("qwen35_ane_pack_input_" #type, qwen35_ane_pack_input,    \
                     type);                                                    \
  instantiate_kernel("qwen35_ane_pack_input_dual_" #type,                     \
                     qwen35_ane_pack_input_dual, type);                        \
  instantiate_kernel("qwen35_cpu_pack_input_" #type, qwen35_cpu_pack_input,   \
                     type);                                                    \
  instantiate_kernel("qwen35_ane_pack_input_dual_cpu_" #type,                \
                     qwen35_ane_pack_input_dual_cpu, type);                   \
  instantiate_kernel("qwen35_ane_pack_input_cpu_" #type,                     \
                     qwen35_ane_pack_input_cpu, type);                        \
  instantiate_kernel("qwen35_cpu_merge_output_" #type,                        \
                     qwen35_cpu_merge_output, type);                           \
  instantiate_kernel("qwen35_ane_merge_output_" #type,                         \
                     qwen35_ane_merge_output, type);                            \
  instantiate_kernel("qwen35_ane_merge_swiglu_output_" #type,                  \
                     qwen35_ane_merge_swiglu_output, type);                     \
  instantiate_kernel("qwen35_ane_merge_dual_output_" #type,                   \
                     qwen35_ane_merge_dual_output, type);                      \
  instantiate_kernel("qwen35_ane_merge_dual_swiglu_output_" #type,            \
                     qwen35_ane_merge_dual_swiglu_output, type);               \
  instantiate_kernel("qwen35_ane_merge_dual_cpu_swiglu_output_" #type,        \
                     qwen35_ane_merge_dual_cpu_swiglu_output, type);           \
  instantiate_kernel("qwen35_ane_merge_dual_cpu_output_" #type,               \
                     qwen35_ane_merge_dual_cpu_output, type);                  \
  instantiate_kernel("qwen35_ane_merge_cpu_swiglu_output_" #type,             \
                     qwen35_ane_merge_cpu_swiglu_output, type);                \
  instantiate_kernel("qwen35_ane_merge_cpu_output_" #type,                    \
                     qwen35_ane_merge_cpu_output, type);                       \
  instantiate_kernel("qwen35_ane_swiglu_suffix_" #type,                        \
                     qwen35_ane_swiglu_suffix, type);                           \
  instantiate_kernel("qwen35_ane_sum_output_" #type,                           \
                     qwen35_ane_sum_output, type);                              \
  instantiate_kernel("qwen35_ane_sum_dual_output_" #type,                      \
                     qwen35_ane_sum_dual_output, type);                        \
  instantiate_kernel("qwen35_ane_sum_dual_cpu_output_" #type,                 \
                     qwen35_ane_sum_dual_cpu_output, type)

instantiate_ane_io(float16_t);
instantiate_ane_io(bfloat16_t);
