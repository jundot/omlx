// Copyright © 2026 IndenScale

#include <metal_simdgroup>

using namespace metal;

template <typename T, int D, int G, int HPT>
[[kernel]] void paged_sdpa_vector_2pass_1_gqa(
    const device T* queries [[buffer(0)]],
    const device T* key_pages [[buffer(1)]],
    const device T* value_pages [[buffer(2)]],
    const device uint* block_tables [[buffer(3)]],
    const device uint* lengths [[buffer(4)]],
    device T* partials [[buffer(5)]],
    device float* sums [[buffer(6)]],
    device float* maxs [[buffer(7)]],
    const constant int& page_size [[buffer(8)]],
    const constant int& max_blocks [[buffer(9)]],
    const constant int& num_kv_heads [[buffer(10)]],
    const constant float& scale [[buffer(11)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int per_thread = D / BD;
  constexpr int NT = G / HPT;
  typedef float U;

  const int kv_head = tid.x;
  const int batch = tid.y;
  const int block = tid.z;
  const int blocks = tpg.z;
  const int group = tidtg.y;
  const int context_length = lengths[batch];
  const int context_chunk = (context_length + blocks - 1) / blocks;
  const int chunk_start = block * context_chunk;
  const int chunk_end = min(context_length, chunk_start + context_chunk);
  const int sub_chunk = (context_chunk + HPT - 1) / HPT;
  const int chunk_index = group / NT;
  const int head_start = (group % NT) * HPT;
  const int token_start = chunk_start + chunk_index * sub_chunk;
  const int token_end = min(chunk_end, token_start + sub_chunk);
  const int num_query_heads = num_kv_heads * G;
  const int base_head = batch * num_query_heads + kv_head * G;

  U q[HPT][per_thread];
  U accum[HPT][per_thread];
  U maximum[HPT];
  U normalizer[HPT];
  for (int head = 0; head < HPT; ++head) {
    const device T* query =
        queries + (base_head + head_start + head) * D + simd_lid * per_thread;
    for (int dim = 0; dim < per_thread; ++dim) {
      q[head][dim] = static_cast<U>(query[dim]) * scale;
      accum[head][dim] = 0;
    }
    maximum[head] = Limits<U>::finite_min;
    normalizer[head] = 0;
  }

  uint logical_page = uint(-1);
  size_t token_base = 0;
  for (int token = token_start; token < token_end; ++token) {
    const uint next_logical_page = token / page_size;
    if (next_logical_page != logical_page) {
      const uint physical_page =
          block_tables[batch * max_blocks + next_logical_page];
      const uint page_offset = token % page_size;
      token_base =
          ((physical_page * num_kv_heads + kv_head) * page_size + page_offset) *
          D;
      logical_page = next_logical_page;
    } else {
      token_base += D;
    }

    U key[per_thread];
    U value[per_thread];
    for (int dim = 0; dim < per_thread; ++dim) {
      key[dim] =
          static_cast<U>(key_pages[token_base + simd_lid * per_thread + dim]);
      value[dim] =
          static_cast<U>(value_pages[token_base + simd_lid * per_thread + dim]);
    }

    for (int head = 0; head < HPT; ++head) {
      U score = 0;
      for (int dim = 0; dim < per_thread; ++dim) {
        score += q[head][dim] * key[dim];
      }
      score = simd_sum(score);
      if (score > maximum[head]) {
        const U old_weight = fast::exp(maximum[head] - score);
        normalizer[head] = normalizer[head] * old_weight + 1;
        for (int dim = 0; dim < per_thread; ++dim) {
          accum[head][dim] = accum[head][dim] * old_weight + value[dim];
        }
        maximum[head] = score;
      } else {
        const U new_weight = fast::exp(score - maximum[head]);
        normalizer[head] += new_weight;
        for (int dim = 0; dim < per_thread; ++dim) {
          accum[head][dim] += value[dim] * new_weight;
        }
      }
    }
  }

  threadgroup U partial_accum[G * HPT * D];
  threadgroup U partial_sums[G * HPT];
  threadgroup U partial_maxs[G * HPT];
  for (int head = 0; head < HPT; ++head) {
    const int slot = (head_start + head) * HPT + chunk_index;
    for (int dim = 0; dim < per_thread; ++dim) {
      partial_accum[slot * D + simd_lid * per_thread + dim] = accum[head][dim];
    }
    if (simd_lid == 0) {
      partial_sums[slot] = normalizer[head];
      partial_maxs[slot] = maximum[head];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  U global_max = Limits<U>::finite_min;
  for (int part = 0; part < HPT; ++part) {
    global_max = max(global_max, partial_maxs[group * HPT + part]);
  }
  U denominator = 0;
  U output[per_thread] = {0};
  for (int part = 0; part < HPT; ++part) {
    const U weight = fast::exp(partial_maxs[group * HPT + part] - global_max);
    denominator += partial_sums[group * HPT + part] * weight;
    for (int dim = 0; dim < per_thread; ++dim) {
      output[dim] += weight *
          partial_accum[(group * HPT + part) * D + simd_lid * per_thread + dim];
    }
  }

  const int output_head = base_head + group;
  const size_t output_offset =
      (output_head * blocks + block) * D + simd_lid * per_thread;
  for (int dim = 0; dim < per_thread; ++dim) {
    partials[output_offset + dim] = static_cast<T>(output[dim]);
  }
  if (simd_lid == 0) {
    sums[output_head * blocks + block] = denominator;
    maxs[output_head * blocks + block] = global_max;
  }
}
