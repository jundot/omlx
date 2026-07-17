// Copyright © 2024 Apple Inc.

#pragma once

///////////////////////////////////////////////////////////////////////////////
// Attn param classes
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

struct AttnParams {
  int B; ///< Batch Size
  int H; ///< Heads
  int D; ///< Head Dim

  int qL; ///< Query Sequence Length
  int kL; ///< Key Sequence Length

  int gqa_factor; ///< Group Query factor
  float scale; ///< Attention scale

  int NQ; ///< Number of query blocks
  int NK; ///< Number of key/value blocks

  int NQ_aligned; ///< Number of full query blocks
  int NK_aligned; ///< Number of full key/value blocks

  int qL_rem; ///< Remainder in last query block
  int kL_rem; ///< Remainder in last key/value block
  int qL_off; ///< Offset in query sequence start

  int64_t Q_strides[3]; ///< Query  strides (B, H, L, D = 1)
  int64_t K_strides[3]; ///< Key    strides (B, H, L, D = 1)
  int64_t V_strides[3]; ///< Value  strides (B, H, L, D = 1)
  int64_t O_strides[3]; ///< Output strides (B, H, L, D = 1)
};

struct AttnMaskParams {
  int64_t M_strides[3]; ///< Mask  strides (B, H, qL, kL = 1)
};

struct AttnBlockMaskParams {
  int64_t BM_strides[3]; ///< Block mask strides (B, H, qBlock, kBlock = 1)
};

struct AttnBlockTokenMaskParams {
  int64_t BTM_strides[3]; ///< Token mask strides (B, H, qL, kBlock = 1)
};

struct AttnBlockIndexParams {
  int64_t BI_strides[3]; ///< Block index strides (B, H, qBlock, budget = 1)
  int budget; ///< Number of selected K blocks per query block
};

struct AttnSplitQKParams {
  int B; ///< Batch Size
  int H; ///< Heads
  int D; ///< Head Dim

  int qL; ///< Query Sequence Length
  int kL; ///< Key Sequence Length

  float scale; ///< Attention scale

  int NQ; ///< Number of query blocks
  int NK; ///< Number of key/value blocks

  int NQ_aligned; ///< Number of full query blocks
  int NK_aligned; ///< Number of full key/value blocks

  int qL_rem; ///< Remainder in last query block
  int kL_rem; ///< Remainder in last key/value block
  int qL_off; ///< Offset in query sequence start

  int64_t Q_nope_strides[3]; ///< Query no-PE strides (B, H, L, D = 1)
  int64_t Q_pe_strides[3]; ///< Query PE strides (B, H, L, D = 1)
  int64_t K_nope_strides[3]; ///< Key no-PE strides (B, H, L, D = 1)
  int64_t K_pe_strides[3]; ///< Key PE strides (B, 1, L, D = 1)
  int64_t V_strides[3]; ///< Value strides (B, H, L, D = 1)
  int64_t O_strides[3]; ///< Output strides (B, H, L, D = 1)
};

struct GlmDsaSparseMlaParams {
  int B; ///< Batch Size
  int H; ///< Query heads
  int qL; ///< Query sequence length
  int kL; ///< Key/value sequence length
  int topk; ///< Number of selected tokens per query
  int topk_valid_prefix; ///< Whether valid causal top-k entries precede invalids
  int causal_prefix_indices; ///< Whether early causal rows use implicit 0..q indices
  int has_topk_length; ///< Whether a per-query valid top-k prefix length exists
  int causal_prefix_rows; ///< Number of leading query rows omitted from top-k

  float scale; ///< Attention scale
  int qL_off; ///< Offset in query sequence start

  int64_t Q_latent_strides[3]; ///< Query latent strides (B, H, L, D = 1)
  int64_t Q_pe_strides[3]; ///< Query PE strides (B, H, L, D = 1)
  int64_t KV_latent_strides[3]; ///< KV latent strides (B, 1, L, D = 1)
  int64_t K_pe_strides[3]; ///< Key PE strides (B, 1, L, D = 1)
  int64_t Topk_strides[3]; ///< Top-k strides (B, 1, L, topk = 1)
  int64_t TopkLength_strides[2]; ///< Top-k length strides (B, L)
  int64_t O_strides[3]; ///< Output strides (B, H, L, D = 1)
};

struct GlmDsaSparseMlaBlockTableParams {
  int B; ///< Batch size
  int H; ///< Query heads
  int qL; ///< Query sequence length
  int kL; ///< Key/value sequence length
  int table_size; ///< Number of block-table entries per query
  int k_block_size; ///< Number of tokens represented by one table bitset

  float scale; ///< Attention scale
  int qL_off; ///< Offset in query sequence start

  int64_t Q_latent_strides[3]; ///< Query latent strides (B, H, L, D = 1)
  int64_t Q_pe_strides[3]; ///< Query PE strides (B, H, L, D = 1)
  int64_t KV_latent_strides[3]; ///< KV latent strides (B, 1, L, D = 1)
  int64_t K_pe_strides[3]; ///< Key PE strides (B, 1, L, D = 1)
  int64_t BlockTable_strides[4]; ///< Block table strides (B, L, slot, pair)
  int64_t O_strides[3]; ///< Output strides (B, H, L, D = 1)
};

struct GlmDsaSparseMlaQBlockParams {
  int B; ///< Batch size
  int H; ///< Query heads
  int qL; ///< Query sequence length
  int kL; ///< Key/value sequence length
  int q_block_size; ///< Number of query rows represented by one union row
  int capacity; ///< Union-token capacity per query block

  float scale; ///< Attention scale
  int qL_off; ///< Offset in query sequence start

  int64_t Q_latent_strides[3]; ///< Query latent strides (B, H, L, D = 1)
  int64_t Q_pe_strides[3]; ///< Query PE strides (B, H, L, D = 1)
  int64_t KV_latent_strides[3]; ///< KV latent strides (B, 1, L, D = 1)
  int64_t K_pe_strides[3]; ///< Key PE strides (B, 1, L, D = 1)
  int64_t Union_strides[3]; ///< Union token strides (B, qBlock, capacity)
  int64_t RowBits_strides[3]; ///< Row-bit strides (B, qBlock, capacity)
  int64_t Length_strides[2]; ///< Union length strides (B, qBlock)
  int64_t O_strides[3]; ///< Output strides (B, H, L, D = 1)
};

struct DeepseekV4SparseAttentionParams {
  int B; ///< Batch size
  int H; ///< Query heads
  int qL; ///< Query sequence length
  int localL; ///< Local rotating KV length
  int pooledL; ///< Number of pooled KV tokens
  int topk; ///< Number of selected pooled tokens per query
  int local_window; ///< Sliding-window size for local KV
  int compress_ratio; ///< Pooled compression ratio
  int q_offset; ///< Absolute query offset before this chunk

  float scale; ///< Attention scale

  int64_t Q_strides[3]; ///< Query strides (B, H, L, D = 1)
  int64_t Local_strides[3]; ///< Local KV strides (B, 1, L, D = 1)
  int64_t Pooled_strides[2]; ///< Pooled KV strides (B, L, D = 1)
  int64_t Topk_strides[3]; ///< Top-k strides (B, 1, L, topk = 1)
  int64_t O_strides[3]; ///< Output strides (B, H, L, D = 1)
};

struct AttnChunkReduceParams {
  int C; ///< Number of key chunks
  int H; ///< Heads
  int qL; ///< Query sequence length
  int D; ///< Head dim

  int64_t o_chunk_stride; ///< Elements per chunk in the partial O slab
  int64_t lse_chunk_stride; ///< Elements per chunk in the logsumexp slab
  int64_t O_strides[3]; ///< Output strides (B, H, L, D = 1)
};

} // namespace steel
} // namespace mlx
