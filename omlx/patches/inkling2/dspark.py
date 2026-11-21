# SPDX-License-Identifier: Apache-2.0
"""Bundled DSpark drafter for Inkling 2 (``mtp.safetensors``).

A small block drafter. Each layer mixes the target's context taps (the
partial block sums entering the layer visits ``target_layer_ids[:-1]`` and the
final-norm output) with its own softmax weights and projects them to context
K/V. Looped targets count every pass through a core layer as a visit. The
block attends to that context and to itself in both directions, and the mask
positions propose through the target LM head plus an RNN Markov head.

Local layers (``local_layer_ids``, e.g. ``[0, 2]``) attend to the newest
``sliding_window_size - 1`` context positions in a bounded ring, while global
layers (layer ``1``) attend to the full committed context history.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from . import INKLING2_MODEL_TYPES

logger = logging.getLogger(__name__)

DRAFT_FILENAME = "mtp.safetensors"


def bundled_inkling2_dspark_path(model_path: str | Path) -> Path | None:
    """``<model>/mtp.safetensors`` when the checkpoint declares a DSpark draft."""
    path = Path(model_path)
    weights = path / DRAFT_FILENAME
    try:
        config = json.loads((path / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    mtp = config.get("mtp_config") or {}
    if config.get("model_type") not in INKLING2_MODEL_TYPES:
        return None
    if not mtp or not weights.is_file():
        return None
    if mtp.get("markov_head_type") != "rnn_hidden":
        return None
    return weights


@dataclass
class Inkling2DSparkConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    local_layer_ids: list
    sliding_window: int
    block_size: int
    markov_rank: int
    target_layer_ids: list
    vocab_size: int
    unpadded_vocab_size: int
    final_logit_softcapping: float
    mask_token_id: int = 0
    sample_from_anchor: bool = False

    @classmethod
    def from_target_config(cls, config: dict) -> Inkling2DSparkConfig:
        text = config["text_config"]
        mtp = config["mtp_config"]
        if mtp.get("q_bias") or mtp.get("o_bias"):
            raise ValueError("Inkling 2 DSpark does not support projection biases")
        return cls(
            hidden_size=int(mtp["hidden_size"]),
            num_hidden_layers=int(mtp["num_hidden_layers"]),
            num_attention_heads=int(mtp["num_attention_heads"]),
            num_key_value_heads=int(mtp["num_key_value_heads"]),
            head_dim=int(mtp["head_dim"]),
            intermediate_size=int(mtp["intermediate_size"]),
            rms_norm_eps=float(mtp.get("rms_norm_eps", 1e-6)),
            rope_theta=float(mtp.get("rope_theta", 1e6)),
            local_layer_ids=list(mtp.get("local_layer_ids") or []),
            sliding_window=int(mtp["sliding_window_size"]),
            block_size=int(mtp["block_size"]),
            markov_rank=int(mtp["markov_rank"]),
            target_layer_ids=[int(i) for i in mtp["target_layer_ids"]],
            vocab_size=int(text["vocab_size"]),
            unpadded_vocab_size=int(
                text.get("unpadded_vocab_size") or text["vocab_size"]
            ),
            final_logit_softcapping=float(text.get("final_logit_softcapping") or 0),
        )


_MARKOV_Q8_ARGMAX_P1_SOURCE = """
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tg = threadgroup_position_in_grid.x;
    const uint b = threadgroup_position_in_grid.y;
    const uint num_tg = threadgroups_per_grid.x;

    constexpr uint PER = R / 32;
    const device bfloat* lp = latent + (size_t)b * R + lane * PER;
    float lv[PER];
    float lsum = 0.0f;
    for (uint i = 0; i < PER; ++i) {
        lv[i] = float(lp[i]);
        lsum += lv[i];
    }
    const uint gi = (lane * PER) / GS;
    constexpr uint K_W = R / 4;
    constexpr uint K_GS = R / GS;

    const uint chunk = (V_UNPAD + num_tg - 1) / num_tg;
    const uint v_begin = tg * chunk;
    const uint v_end = min((uint)V_UNPAD, v_begin + chunk);
    const device bfloat* blp = base_logits + (size_t)b * V_FULL;

    float best_v = -INFINITY;
    uint best_i = 0;

    for (uint v = v_begin + sg; v < v_end; v += 8) {
        float sc = float(scales[(size_t)v * K_GS + gi]);
        float bi = float(biases[(size_t)v * K_GS + gi]);
        float d0 = 0.0f;
        const device uint* wp = w_q + (size_t)v * K_W + lane * (PER / 4);
        for (uint w = 0; w < PER / 4; ++w) {
            uint pk = wp[w];
            d0 += float((pk >> 0) & 0xFFu) * lv[w * 4 + 0]
                + float((pk >> 8) & 0xFFu) * lv[w * 4 + 1]
                + float((pk >> 16) & 0xFFu) * lv[w * 4 + 2]
                + float((pk >> 24) & 0xFFu) * lv[w * 4 + 3];
        }
        float dot = simd_sum(d0 * sc + lsum * bi);
        if (lane == 0) {
            float score = float(blp[v]) + float(fold_bias[v]) + dot;
            if (score > best_v || (score == best_v && v < best_i)) {
                best_v = score;
                best_i = v;
            }
        }
    }
    threadgroup float tg_v[8];
    threadgroup uint tg_i[8];
    if (lane == 0) {
        tg_v[sg] = best_v;
        tg_i[sg] = best_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        float bv = tg_v[0];
        uint bi_idx = tg_i[0];
        for (int i = 1; i < 8; ++i) {
            if (tg_v[i] > bv || (tg_v[i] == bv && tg_i[i] < bi_idx)) {
                bv = tg_v[i];
                bi_idx = tg_i[i];
            }
        }
        part_v[(size_t)b * num_tg + tg] = bv;
        part_i[(size_t)b * num_tg + tg] = bi_idx;
    }
"""

_MARKOV_ARGMAX_P2_SOURCE = """
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_position_in_threadgroup.x;
    const uint b = threadgroup_position_in_grid.x;

    float bv = -INFINITY;
    uint bi = 0;
    const device float* pv = part_v + (size_t)b * NTG;
    const device uint* pi = part_i + (size_t)b * NTG;
    for (uint i = tid; i < NTG; i += 128) {
        float v = pv[i];
        uint idx = pi[i];
        if (v > bv || (v == bv && idx < bi)) {
            bv = v;
            bi = idx;
        }
    }
    float sm = simd_max(bv);
    uint win_i = simd_min((bv == sm) ? bi : 0xFFFFFFFFu);
    threadgroup float red_v[4];
    threadgroup uint red_i[4];
    if (lane == 0) {
        red_v[sg] = sm;
        red_i[sg] = win_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        float fv = red_v[0];
        uint fi = red_i[0];
        for (int i = 1; i < 4; ++i) {
            if (red_v[i] > fv || (red_v[i] == fv && red_i[i] < fi)) {
                fv = red_v[i];
                fi = red_i[i];
            }
        }
        out_tok[b] = fi;
    }
"""

_markov_q8_p1_kernel = None
_markov_argmax_p2_kernel = None
_MARKOV_FOLD_GROUP_SIZE = 64
_MARKOV_FOLD_BITS = 8
_MARKOV_ARGMAX_NTG = 256

# ---------------------------------------------------------------------------
# Basalt-class 2-bit screened exact DSpark head + calibrated conf[i].
#
# Stage 0 (once at fold()): pack 2-bit symmetric copies (4 levels {-1.5, -0.5,
#   +0.5, +1.5} * scale, group_size=32, Basalt's UNSHRINK variance correction)
#   of both target lm_head and folded Markov matrix in tile-transposed row
#   order: screen row r = tile*32 + lane maps to vocab token
#   v = lane*T_tiles + tile, spreading contiguous punctuation/common tokens
#   across 32 different tiles.
# Stage 1A (once per block): dot 2-bit head_sc against all M = 1..7 draft
#   hidden states in one pass (~144 MB vs 492 MB Q8 lm_head).
# Stage 1B (per Markov step): dot 2-bit fold_sc against the step latent (~16 MB
#   vs 55 MB Q8 fold), add approx_head + fold_bp, apply softcap, emit per-tile
#   softmax partials (tile_max, tile_sum_exp) and top-4 candidate screen rows.
# Stage 2 (per Markov step): collect top <= 128 candidates within DELTA of the
#   global approx max, rescore ONLY those candidates using exact Q8/BF16 head
#   + exact Q8 fold + fold_bias + softcap, pick the exact argmax (lowest vocab
#   id on tie), and compute its calibrated softmax probability conf[i] by
#   swapping rescored rows' approx terms for exact terms in the tile sum.
# ---------------------------------------------------------------------------

_SCREEN_K_PER = 2
_SCREEN_K_MAX_CAND = 64
_SCREEN_DELTA = 6.0

_DSPARK_LUT2_HEADER = """
constant half2 DS_LUT2[16] = {
    half2(-1.5h,-1.5h), half2(-0.5h,-1.5h), half2( 0.5h,-1.5h), half2( 1.5h,-1.5h),
    half2(-1.5h,-0.5h), half2(-0.5h,-0.5h), half2( 0.5h,-0.5h), half2( 1.5h,-0.5h),
    half2(-1.5h, 0.5h), half2(-0.5h, 0.5h), half2( 0.5h, 0.5h), half2( 1.5h, 0.5h),
    half2(-1.5h, 1.5h), half2(-0.5h, 1.5h), half2( 0.5h, 1.5h), half2( 1.5h, 1.5h)
};
"""

_DSPARK_SCREEN_PACK_SOURCE = """
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tg = threadgroup_position_in_grid.x;
    const uint g = threadgroup_position_in_grid.y;
    const uint tile = tg * 4 + sg;
    if (tile >= T_TILES) return;

    const uint v = lane * T_TILES + tile;
    const size_t out_idx = ((size_t)tile * N_GROUPS + g) * 32 + lane;
    if (v >= V_UNPAD) {
        out_wq[out_idx * 2 + 0] = 0u;
        out_wq[out_idx * 2 + 1] = 0u;
        out_sc[out_idx] = half(0.0f);
        return;
    }

    float w[32];
    if (IS_Q8) {
        const uint k0 = g * 32;
        const uint gi = k0 / SRC_GS;
        const float sc = float(w_scales[(size_t)v * (N_IN / SRC_GS) + gi]);
        const float bi = float(w_biases[(size_t)v * (N_IN / SRC_GS) + gi]);
        for (uint j = 0; j < 8; ++j) {
            uint pk = w_q[(size_t)v * (N_IN / 4) + (k0 / 4) + j];
            w[j * 4 + 0] = float((pk >> 0) & 0xFFu) * sc + bi;
            w[j * 4 + 1] = float((pk >> 8) & 0xFFu) * sc + bi;
            w[j * 4 + 2] = float((pk >> 16) & 0xFFu) * sc + bi;
            w[j * 4 + 3] = float((pk >> 24) & 0xFFu) * sc + bi;
        }
    } else {
        const size_t base_w = (size_t)v * N_IN + g * 32;
        for (uint j = 0; j < 32; ++j) {
            w[j] = float(w_dense[base_w + j]);
        }
    }

    float amax = 1e-9f;
    for (uint j = 0; j < 32; ++j) {
        amax = max(amax, metal::fabs(w[j]));
    }
    float best_s = amax / 1.5f;
    float best_e = INFINITY;
    for (int t = 0; t < 6; ++t) {
        float s = (amax / 1.5f) * (1.0f - 0.1f * float(t));
        float sh = float(half(s));
        float inv_sh = (sh > 0.0f) ? (1.0f / sh) : 0.0f;
        float e = 0.0f;
        for (uint j = 0; j < 32; ++j) {
            int lv = clamp(int(metal::floor(w[j] * inv_sh + 2.0f)), 0, 3);
            float r = (float(lv) - 1.5f) * sh - w[j];
            e += r * r;
        }
        if (e < best_e) {
            best_e = e;
            best_s = s;
        }
    }

    half hs = half(best_s);
    float sh = float(hs);
    float inv_sh = (sh > 0.0f) ? (1.0f / sh) : 0.0f;

    uint word0 = 0u;
    uint word1 = 0u;
    float ww = 0.0f;
    float wr = 0.0f;
    for (uint k = 0; k < 8; ++k) {
        int lv0 = clamp(int(metal::floor(w[2 * k] * inv_sh + 2.0f)), 0, 3);
        int lv1 = clamp(int(metal::floor(w[2 * k + 1] * inv_sh + 2.0f)), 0, 3);
        word0 |= (uint(lv0) | (uint(lv1) << 2)) << (4 * k);
        ww += w[2 * k] * w[2 * k] + w[2 * k + 1] * w[2 * k + 1];
        wr += w[2 * k] * ((float(lv0) - 1.5f) * sh)
            + w[2 * k + 1] * ((float(lv1) - 1.5f) * sh);

        int lv2 = clamp(int(metal::floor(w[16 + 2 * k] * inv_sh + 2.0f)), 0, 3);
        int lv3 = clamp(int(metal::floor(w[16 + 2 * k + 1] * inv_sh + 2.0f)), 0, 3);
        word1 |= (uint(lv2) | (uint(lv3) << 2)) << (4 * k);
        ww += w[16 + 2 * k] * w[16 + 2 * k] + w[16 + 2 * k + 1] * w[16 + 2 * k + 1];
        wr += w[16 + 2 * k] * ((float(lv2) - 1.5f) * sh)
            + w[16 + 2 * k + 1] * ((float(lv3) - 1.5f) * sh);
    }
    if (wr > 0.0f) {
        hs = half(sh * metal::sqrt(ww / wr));
    }
    out_wq[out_idx * 2 + 0] = word0;
    out_wq[out_idx * 2 + 1] = word1;
    out_sc[out_idx] = hs;
"""

_DSPARK_MARKOV_STEP_SOURCE = """
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint r = threadgroup_position_in_grid.x * 4 + sg;
    const uint b = threadgroup_position_in_grid.y;
    if (r >= R) return;

    const uint prev_tok = prev_tokens[b];
    const device bfloat* st_b = state_in + (size_t)b * R;
    const device bfloat* w1_b = markov_w1 + (size_t)prev_tok * R;

    const device bfloat* wg = w_sp + (size_t)r * (2 * R);
    const device bfloat* wc = w_sp + (size_t)(r + R) * (2 * R);
    const device bfloat* wl = w_sp + (size_t)(r + 2 * R) * (2 * R);

    float dg = 0.0f;
    float dc = 0.0f;
    float dl = 0.0f;
    for (uint k = lane; k < R; k += 32) {
        float sv = float(st_b[k]);
        float tv = float(w1_b[k]);
        dg += float(wg[k]) * sv + float(wg[R + k]) * tv;
        dc += float(wc[k]) * sv + float(wc[R + k]) * tv;
        dl += float(wl[k]) * sv + float(wl[R + k]) * tv;
    }
    dg = simd_sum(dg);
    dc = simd_sum(dc);
    dl = simd_sum(dl);

    if (lane == 0) {
        const size_t base = (size_t)b * (3 * R);
        const float raw_g = float(ctx_step[base + r]) + dg;
        const float raw_c = float(ctx_step[base + R + r]) + dc;
        const float raw_l = float(ctx_step[base + 2 * R + r]) + dl;
        const float gate_f = 1.0f / (1.0f + metal::exp(metal::clamp(-raw_g, -80.0f, 80.0f)));
        const bfloat gate = bfloat(gate_f);
        const bfloat cand = bfloat(metal::tanh(metal::clamp(raw_c, -20.0f, 20.0f)));
        const bfloat st = st_b[r];
        const bfloat kept = bfloat(float(gate) * float(st));
        const bfloat blend = bfloat((1.0f - float(gate)) * float(cand));
        state_out[(size_t)b * R + r] = bfloat(float(kept) + float(blend));
        const bfloat lat = bfloat(metal::tanh(metal::clamp(raw_l, -20.0f, 20.0f)));
        lat_bf16[(size_t)b * R + r] = lat;
        lat_f16[(size_t)b * R + r] = half(float(lat));
    }
"""

_DSPARK_HEAD_BASE_SOURCE = """
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tile = threadgroup_position_in_grid.x;
    const uint b = threadgroup_position_in_grid.y;

    constexpr uint GP = (N_GROUPS + 3) / 4;
    const uint g_begin = sg * GP;
    const uint g_end = min((uint)N_GROUPS, g_begin + GP);

    const device half2* hb = (const device half2*)(h_f16 + (size_t)b * M * N_IN);
    const device uint2* wq = (const device uint2*)head_wq;

    float acc[M];
    for (uint m = 0; m < M; ++m) acc[m] = 0.0f;

    const size_t t_base = (size_t)tile * N_GROUPS;
    for (uint g = g_begin; g < g_end; ++g) {
        const size_t idx = (t_base + g) * 32 + lane;
        const uint2 pk = wq[idx];
        const float s = float(head_sc[idx]);
        const uint h2_base = g * 16;

        half2 w0 = DS_LUT2[pk.x & 0xFu];
        half2 w1 = DS_LUT2[(pk.x >> 4) & 0xFu];
        half2 w2 = DS_LUT2[(pk.x >> 8) & 0xFu];
        half2 w3 = DS_LUT2[(pk.x >> 12) & 0xFu];
        half2 w4 = DS_LUT2[(pk.x >> 16) & 0xFu];
        half2 w5 = DS_LUT2[(pk.x >> 20) & 0xFu];
        half2 w6 = DS_LUT2[(pk.x >> 24) & 0xFu];
        half2 w7 = DS_LUT2[(pk.x >> 28) & 0xFu];

        float psum[M];
        for (uint m = 0; m < M; ++m) {
            const device half2* xm = hb + (size_t)m * (N_IN / 2) + h2_base;
            half2 gacc = w0 * xm[0];
            gacc = fma(w1, xm[1], gacc);
            gacc = fma(w2, xm[2], gacc);
            gacc = fma(w3, xm[3], gacc);
            gacc = fma(w4, xm[4], gacc);
            gacc = fma(w5, xm[5], gacc);
            gacc = fma(w6, xm[6], gacc);
            gacc = fma(w7, xm[7], gacc);
            psum[m] = float(gacc.x) + float(gacc.y);
        }

        w0 = DS_LUT2[pk.y & 0xFu];
        w1 = DS_LUT2[(pk.y >> 4) & 0xFu];
        w2 = DS_LUT2[(pk.y >> 8) & 0xFu];
        w3 = DS_LUT2[(pk.y >> 12) & 0xFu];
        w4 = DS_LUT2[(pk.y >> 16) & 0xFu];
        w5 = DS_LUT2[(pk.y >> 20) & 0xFu];
        w6 = DS_LUT2[(pk.y >> 24) & 0xFu];
        w7 = DS_LUT2[(pk.y >> 28) & 0xFu];

        for (uint m = 0; m < M; ++m) {
            const device half2* xm = hb + (size_t)m * (N_IN / 2) + h2_base + 8;
            half2 gacc = w0 * xm[0];
            gacc = fma(w1, xm[1], gacc);
            gacc = fma(w2, xm[2], gacc);
            gacc = fma(w3, xm[3], gacc);
            gacc = fma(w4, xm[4], gacc);
            gacc = fma(w5, xm[5], gacc);
            gacc = fma(w6, xm[6], gacc);
            gacc = fma(w7, xm[7], gacc);
            acc[m] += s * (psum[m] + float(gacc.x) + float(gacc.y));
        }
    }

    threadgroup float red[4][32];
    const uint sr = tile * 32 + lane;
    for (uint m = 0; m < M; ++m) {
        red[sg][lane] = acc[m];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float tot = red[0][lane] + red[1][lane] + red[2][lane] + red[3][lane];
            approx_out[((size_t)b * M + m) * (T_TILES * 32) + sr] = half(tot);
        }
        if (m + 1 < M) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
"""

_DSPARK_FOLD_SCREEN_SOURCE = """
    constexpr float LOG2E = 1.4426950408889634f;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tile = threadgroup_position_in_grid.x * 4 + sg;
    const uint b = threadgroup_position_in_grid.y;
    if (tile >= T_TILES) return;

    const uint sr = tile * 32 + lane;
    const uint v_id = lane * T_TILES + tile;
    const device half2* lb = (const device half2*)(lat_f16 + (size_t)b * R);
    const device uint2* wq = (const device uint2*)fold_wq;

    float score = -INFINITY;
    if (v_id < V_UNPAD) {
        float acc = 0.0f;
        const size_t t_base = (size_t)tile * R_GROUPS;
        for (uint g = 0; g < R_GROUPS; ++g) {
            const size_t idx = (t_base + g) * 32 + lane;
            const uint2 pk = wq[idx];
            const float s = float(fold_sc[idx]);
            const device half2* xm = lb + g * 16;
            half2 gacc = DS_LUT2[pk.x & 0xFu] * xm[0];
            gacc = fma(DS_LUT2[(pk.x >> 4) & 0xFu], xm[1], gacc);
            gacc = fma(DS_LUT2[(pk.x >> 8) & 0xFu], xm[2], gacc);
            gacc = fma(DS_LUT2[(pk.x >> 12) & 0xFu], xm[3], gacc);
            gacc = fma(DS_LUT2[(pk.x >> 16) & 0xFu], xm[4], gacc);
            gacc = fma(DS_LUT2[(pk.x >> 20) & 0xFu], xm[5], gacc);
            gacc = fma(DS_LUT2[(pk.x >> 24) & 0xFu], xm[6], gacc);
            gacc = fma(DS_LUT2[(pk.x >> 28) & 0xFu], xm[7], gacc);
            gacc = fma(DS_LUT2[pk.y & 0xFu], xm[8], gacc);
            gacc = fma(DS_LUT2[(pk.y >> 4) & 0xFu], xm[9], gacc);
            gacc = fma(DS_LUT2[(pk.y >> 8) & 0xFu], xm[10], gacc);
            gacc = fma(DS_LUT2[(pk.y >> 12) & 0xFu], xm[11], gacc);
            gacc = fma(DS_LUT2[(pk.y >> 16) & 0xFu], xm[12], gacc);
            gacc = fma(DS_LUT2[(pk.y >> 20) & 0xFu], xm[13], gacc);
            gacc = fma(DS_LUT2[(pk.y >> 24) & 0xFu], xm[14], gacc);
            gacc = fma(DS_LUT2[(pk.y >> 28) & 0xFu], xm[15], gacc);
            acc += s * (float(gacc.x) + float(gacc.y));
        }
        float raw = float(approx_base[(size_t)b * (T_TILES * 32) + sr])
                  + float(fold_bp[sr]) + acc;
        float cap = cap_arr[0];
        if (cap > 0.0f) {
            score = cap * fast::tanh(metal::clamp(raw / cap, -20.0f, 20.0f));
        } else {
            score = raw;
        }
    }

    float tmax = simd_max(score);
    float e = (score > -INFINITY) ? fast::exp2((score - tmax) * LOG2E) : 0.0f;
    float tsum = simd_sum(e);
    if (lane == 0) {
        const size_t p_idx = ((size_t)b * T_TILES + tile) * 2;
        part_out[p_idx + 0] = tmax;
        part_out[p_idx + 1] = tsum;
    }

    float cur = score;
    const size_t c_base = ((size_t)b * T_TILES + tile) * K_PER;
    for (uint k = 0; k < K_PER; ++k) {
        float bv = simd_max(cur);
        uint win_lane = simd_min((cur == bv) ? lane : 32u);
        if (lane == 0) {
            cand_v[c_base + k] = bv;
            cand_r[c_base + k] = tile * 32 + (win_lane < 32u ? win_lane : 0u);
        }
        if (lane == win_lane) {
            cur = -INFINITY;
        }
    }
"""

_DSPARK_HEAD_PICK_SOURCE = """
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint b = threadgroup_position_in_grid.x;

    const device float* cv = cand_v + (size_t)b * N_CAND_IN;
    const device uint* cr = cand_r + (size_t)b * N_CAND_IN;

    // 1. Global approximate maximum across all tile candidates.
    float local_m = -INFINITY;
    for (uint i = tid; i < N_CAND_IN; i += 256) {
        local_m = max(local_m, cv[i]);
    }
    local_m = simd_max(local_m);
    threadgroup float tg_red[8];
    if (lane == 0) tg_red[sg] = local_m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gmax = tg_red[0];
    for (uint i = 1; i < 8; ++i) gmax = max(gmax, tg_red[i]);

    if (!(gmax > -INFINITY)) {
        if (tid == 0) {
            out_tok[b] = 0u;
            out_conf[b] = 1.0f;
        }
        return;
    }

    // 2. Collect up to K_MAX candidates within DELTA of gmax.
    threadgroup atomic_uint tg_cnt;
    threadgroup uint tg_vocab[K_MAX];
    threadgroup float tg_lp[K_MAX];
    threadgroup float tg_ex[K_MAX];
    if (tid == 0) atomic_store_explicit(&tg_cnt, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float cut0 = gmax - DELTA_VAL;
    for (uint i = tid; i < N_CAND_IN; i += 256) {
        float sc = cv[i];
        if (sc >= cut0) {
            uint pos = atomic_fetch_add_explicit(&tg_cnt, 1u, memory_order_relaxed);
            if (pos < K_MAX) {
                uint sr = cr[i];
                uint v = (sr & 31u) * T_TILES + (sr >> 5);
                tg_vocab[pos] = v;
                tg_lp[pos] = sc;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint total = atomic_load_explicit(&tg_cnt, memory_order_relaxed);
    if (total > K_MAX) {
        // Binary search exact threshold in [cut0, gmax] to retain top K_MAX.
        float lo = cut0;
        float hi = gmax;
        threadgroup uint tg_bs[8];
        for (int it = 0; it < 14; ++it) {
            float mid = 0.5f * (lo + hi);
            uint c = 0u;
            for (uint i = tid; i < N_CAND_IN; i += 256) {
                if (cv[i] >= mid) c += 1u;
            }
            c = simd_sum(c);
            if (lane == 0) tg_bs[sg] = c;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint tot_c = 0u;
            for (uint j = 0; j < 8; ++j) tot_c += tg_bs[j];
            if (tot_c > K_MAX) {
                lo = mid;
            } else {
                hi = mid;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) atomic_store_explicit(&tg_cnt, 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < N_CAND_IN; i += 256) {
            float sc = cv[i];
            if (sc > hi) {
                uint pos = atomic_fetch_add_explicit(&tg_cnt, 1u, memory_order_relaxed);
                if (pos < K_MAX) {
                    uint sr = cr[i];
                    tg_vocab[pos] = (sr & 31u) * T_TILES + (sr >> 5);
                    tg_lp[pos] = sc;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < N_CAND_IN; i += 256) {
            float sc = cv[i];
            if (sc >= lo && sc <= hi) {
                uint pos = atomic_fetch_add_explicit(&tg_cnt, 1u, memory_order_relaxed);
                if (pos < K_MAX) {
                    uint sr = cr[i];
                    tg_vocab[pos] = (sr & 31u) * T_TILES + (sr >> 5);
                    tg_lp[pos] = sc;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        total = min((uint)K_MAX, atomic_load_explicit(&tg_cnt, memory_order_relaxed));
    }
    const uint n_cand = total;

    // 3. Stage exact h_step and lat_step into threadgroup memory for rescoring.
    threadgroup float tg_h[H];
    threadgroup float tg_lat[R];
    const device bfloat* hp = h_step + (size_t)b * H;
    const device bfloat* lp = lat_step + (size_t)b * R;
    for (uint i = tid; i < H; i += 256) tg_h[i] = float(hp[i]);
    for (uint i = tid; i < R; i += 256) tg_lat[i] = float(lp[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4. Exact Q8/BF16 rescore of the <= K_MAX candidates (8 simdgroups in parallel).
    constexpr uint H_KW = H / 4;
    constexpr uint H_KGS = H / HEAD_GS;
    constexpr uint R_KW = R / 4;
    constexpr uint R_KGS = R / FOLD_GS;
    const float cap = cap_arr[0];

    for (uint c = sg; c < n_cand; c += 8) {
        const uint v = tg_vocab[c];
        float hdot = 0.0f;
        if (HEAD_Q8) {
            for (uint k4 = lane; k4 < H_KW; k4 += 32) {
                uint k0 = k4 * 4;
                uint gi = k0 / HEAD_GS;
                float sc = float(head_s[(size_t)v * H_KGS + gi]);
                float bi = float(head_b[(size_t)v * H_KGS + gi]);
                uint pk = head_wq[(size_t)v * H_KW + k4];
                hdot += (float((pk >> 0) & 0xFFu) * sc + bi) * tg_h[k0 + 0]
                      + (float((pk >> 8) & 0xFFu) * sc + bi) * tg_h[k0 + 1]
                      + (float((pk >> 16) & 0xFFu) * sc + bi) * tg_h[k0 + 2]
                      + (float((pk >> 24) & 0xFFu) * sc + bi) * tg_h[k0 + 3];
            }
        } else {
            const size_t base_w = (size_t)v * H;
            for (uint k = lane; k < H; k += 32) {
                hdot += float(head_dense[base_w + k]) * tg_h[k];
            }
        }
        hdot = simd_sum(hdot);

        float fdot = 0.0f;
        for (uint k4 = lane; k4 < R_KW; k4 += 32) {
            uint k0 = k4 * 4;
            uint gi = k0 / FOLD_GS;
            float sc = float(fold_s[(size_t)v * R_KGS + gi]);
            float bi = float(fold_b[(size_t)v * R_KGS + gi]);
            uint pk = fold_wq[(size_t)v * R_KW + k4];
            fdot += (float((pk >> 0) & 0xFFu) * sc + bi) * tg_lat[k0 + 0]
                  + (float((pk >> 8) & 0xFFu) * sc + bi) * tg_lat[k0 + 1]
                  + (float((pk >> 16) & 0xFFu) * sc + bi) * tg_lat[k0 + 2]
                  + (float((pk >> 24) & 0xFFu) * sc + bi) * tg_lat[k0 + 3];
        }
        fdot = simd_sum(fdot);

        if (lane == 0) {
            float raw = hdot + fdot + float(fold_bias[v]);
            tg_ex[c] = (cap > 0.0f)
                ? (cap * metal::tanh(metal::clamp(raw / cap, -20.0f, 20.0f)))
                : raw;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 5. Pick exact argmax (lowest vocab_id on tie) and calibrated softmax conf[i].
    threadgroup uint tg_best_i;
    threadgroup float tg_best_v;
    if (sg == 0) {
        float bv = -INFINITY;
        uint bi = 0xFFFFFFFFu;
        for (uint c = lane; c < n_cand; c += 32) {
            float ev = tg_ex[c];
            uint vi = tg_vocab[c];
            if (ev > bv || (ev == bv && vi < bi)) {
                bv = ev;
                bi = vi;
            }
        }
        float top = simd_max(bv);
        uint win = simd_min((bv == top) ? bi : 0xFFFFFFFFu);
        if (lane == 0) {
            tg_best_v = top;
            tg_best_i = (win < 0xFFFFFFFFu) ? win : 0u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr float LOG2E = 1.4426950408889634f;
    const float bv = tg_best_v;
    const uint bi = tg_best_i;
    const device float* pp = part_in + (size_t)b * T_TILES * 2;
    float sum = 0.0f;
    for (uint t = tid; t < T_TILES; t += 256) {
        float pm = pp[t * 2 + 0];
        float ps = pp[t * 2 + 1];
        if (ps > 0.0f) {
            sum += ps * fast::exp2((pm - bv) * LOG2E);
        }
    }
    for (uint c = tid; c < n_cand; c += 256) {
        sum += fast::exp2((tg_ex[c] - bv) * LOG2E) - fast::exp2((tg_lp[c] - bv) * LOG2E);
    }
    sum = simd_sum(sum);
    if (lane == 0) tg_red[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total_sum = 0.0f;
        for (uint i = 0; i < 8; ++i) total_sum += tg_red[i];
        total_sum = max(total_sum, 1.0f);
        out_tok[b] = bi;
        out_conf[b] = 1.0f / total_sum;
    }
"""

_dspark_screen_pack_kernel = None
_dspark_head_base_kernel = None
_dspark_fold_screen_kernel = None
_dspark_head_pick_kernel = None
_dspark_markov_step_kernel = None
_DUMMY_U32 = None
_DUMMY_BF16 = None


def _dummy_u32() -> mx.array:
    global _DUMMY_U32
    if _DUMMY_U32 is None:
        _DUMMY_U32 = mx.zeros((4,), dtype=mx.uint32)
    return _DUMMY_U32


def _dummy_bf16() -> mx.array:
    global _DUMMY_BF16
    if _DUMMY_BF16 is None:
        _DUMMY_BF16 = mx.zeros((4,), dtype=mx.bfloat16)
    return _DUMMY_BF16


def _dspark_screen_pack(
    *,
    v_unpad: int,
    n_in: int,
    w_q: mx.array | None = None,
    w_scales: mx.array | None = None,
    w_biases: mx.array | None = None,
    src_gs: int = 64,
    w_dense: mx.array | None = None,
) -> tuple[mx.array, mx.array, int]:
    """Pack a Q8-affine or BF16 matrix into Basalt's 2-bit tile-transposed screen copy."""
    global _dspark_screen_pack_kernel
    if _dspark_screen_pack_kernel is None:
        _dspark_screen_pack_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_dspark_screen_pack",
            input_names=["w_q", "w_scales", "w_biases", "w_dense"],
            output_names=["out_wq", "out_sc"],
            source=_DSPARK_SCREEN_PACK_SOURCE,
        )
    t_tiles = (v_unpad + 31) // 32
    n_groups = n_in // 32
    is_q8 = 1 if w_q is not None else 0
    wt = w_dense.dtype if w_dense is not None else mx.bfloat16
    ntg_x = (t_tiles + 3) // 4
    out_wq, out_sc = _dspark_screen_pack_kernel(
        inputs=[
            w_q if is_q8 else _dummy_u32(),
            w_scales if is_q8 else _dummy_bf16(),
            w_biases if is_q8 else _dummy_bf16(),
            w_dense if not is_q8 else _dummy_bf16(),
        ],
        template=[
            ("WT", wt),
            ("T_TILES", t_tiles),
            ("N_GROUPS", n_groups),
            ("N_IN", n_in),
            ("V_UNPAD", v_unpad),
            ("SRC_GS", src_gs),
            ("IS_Q8", is_q8),
        ],
        grid=(ntg_x * 128, n_groups, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(t_tiles, n_groups, 32, 2), (t_tiles, n_groups, 32)],
        output_dtypes=[mx.uint32, mx.float16],
    )
    return out_wq, out_sc, t_tiles


def _dspark_screen_order(by_token: mx.array, v_unpad: int, t_tiles: int) -> mx.array:
    """Permute a [V] vector into tile-transposed screen row order [32 * T_tiles] fp16."""
    n_screen = t_tiles * 32
    sr = mx.arange(n_screen, dtype=mx.uint32)
    v = (sr & mx.array(31, dtype=mx.uint32)) * mx.array(t_tiles, dtype=mx.uint32) + (
        sr >> mx.array(5, dtype=mx.uint32)
    )
    valid = v < v_unpad
    v_safe = mx.where(valid, v, mx.array(0, dtype=mx.uint32))
    gathered = by_token[v_safe].astype(mx.float16)
    return mx.where(valid, gathered, mx.array(0.0, dtype=mx.float16))


def _dspark_head_base(
    h_bf16: mx.array, head_wq: mx.array, head_sc: mx.array, t_tiles: int
) -> mx.array:
    """Stage 1A: 2-bit screened head dot product across all M block rows at once."""
    global _dspark_head_base_kernel
    if _dspark_head_base_kernel is None:
        _dspark_head_base_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_dspark_head_base",
            input_names=["h_f16", "head_wq", "head_sc"],
            output_names=["approx_out"],
            source=_DSPARK_HEAD_BASE_SOURCE,
            header=_DSPARK_LUT2_HEADER,
        )
    b, m, n_in = h_bf16.shape
    h_f16 = mx.contiguous(h_bf16.astype(mx.float16))
    return _dspark_head_base_kernel(
        inputs=[h_f16, head_wq, head_sc],
        template=[
            ("T_TILES", t_tiles),
            ("N_GROUPS", n_in // 32),
            ("N_IN", n_in),
            ("M", m),
        ],
        grid=(t_tiles * 128, b, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(b, m, t_tiles * 32)],
        output_dtypes=[mx.float16],
    )[0]


def _dspark_markov_step_fused(
    state_in: mx.array,
    prev_tokens: mx.array,
    ctx_step: mx.array,
    w_sp: mx.array,
    markov_w1: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Fused Markov RNN recurrence: returns ``(state_out, lat_bf16, lat_f16)``."""
    global _dspark_markov_step_kernel
    if _dspark_markov_step_kernel is None:
        _dspark_markov_step_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_dspark_markov_step",
            input_names=["state_in", "prev_tokens", "ctx_step", "w_sp", "markov_w1"],
            output_names=["state_out", "lat_bf16", "lat_f16"],
            source=_DSPARK_MARKOV_STEP_SOURCE,
        )
    b, r = state_in.shape
    ntg_x = (r + 3) // 4
    return _dspark_markov_step_kernel(
        inputs=[
            state_in,
            prev_tokens.reshape(-1).astype(mx.uint32),
            ctx_step,
            w_sp,
            markov_w1,
        ],
        template=[("R", r)],
        grid=(ntg_x * 128, b, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(b, r), (b, r), (b, r)],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.float16],
    )


def _dspark_screened_pick(
    *,
    h_step: mx.array,
    lat_step: mx.array,
    lat_f16: mx.array | None = None,
    approx_base: mx.array,
    fold_sc_wq: mx.array,
    fold_sc_sc: mx.array,
    fold_bp: mx.array,
    head_wq: mx.array | None,
    head_s: mx.array | None,
    head_b: mx.array | None,
    head_dense: mx.array | None,
    head_gs: int,
    fold_wq: mx.array,
    fold_s: mx.array,
    fold_b: mx.array,
    fold_bias: mx.array,
    v_unpad: int,
    t_tiles: int,
    cap_arr: mx.array,
) -> tuple[mx.array, mx.array]:
    """Stage 1B + Stage 2: 2-bit fold screen -> exact Q8/BF16 rescore -> (tok, conf)."""
    global _dspark_fold_screen_kernel, _dspark_head_pick_kernel
    if _dspark_fold_screen_kernel is None:
        _dspark_fold_screen_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_dspark_fold_screen",
            input_names=[
                "lat_f16",
                "fold_wq",
                "fold_sc",
                "fold_bp",
                "approx_base",
                "cap_arr",
            ],
            output_names=["part_out", "cand_v", "cand_r"],
            source=_DSPARK_FOLD_SCREEN_SOURCE,
            header=_DSPARK_LUT2_HEADER,
        )
        _dspark_head_pick_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_dspark_head_pick",
            input_names=[
                "cand_v",
                "cand_r",
                "part_in",
                "h_step",
                "lat_step",
                "head_wq",
                "head_s",
                "head_b",
                "head_dense",
                "fold_wq",
                "fold_s",
                "fold_b",
                "fold_bias",
                "cap_arr",
            ],
            output_names=["out_tok", "out_conf"],
            source=_DSPARK_HEAD_PICK_SOURCE,
        )
    b, r = lat_step.shape
    h_dim = h_step.shape[-1]
    if lat_f16 is None:
        lat_f16 = mx.contiguous(lat_step.astype(mx.float16))
    ntg_x = (t_tiles + 3) // 4
    part, cand_v, cand_r = _dspark_fold_screen_kernel(
        inputs=[lat_f16, fold_sc_wq, fold_sc_sc, fold_bp, approx_base, cap_arr],
        template=[
            ("T_TILES", t_tiles),
            ("R", r),
            ("R_GROUPS", r // 32),
            ("V_UNPAD", v_unpad),
            ("K_PER", _SCREEN_K_PER),
        ],
        grid=(ntg_x * 128, b, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[
            (b, t_tiles, 2),
            (b, t_tiles * _SCREEN_K_PER),
            (b, t_tiles * _SCREEN_K_PER),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.uint32],
    )
    head_is_q8 = 1 if head_wq is not None else 0
    wt = head_dense.dtype if head_dense is not None else mx.bfloat16
    tok, conf = _dspark_head_pick_kernel(
        inputs=[
            cand_v,
            cand_r,
            part,
            h_step if h_step.dtype == mx.bfloat16 else h_step.astype(mx.bfloat16),
            lat_step if lat_step.dtype == mx.bfloat16 else lat_step.astype(mx.bfloat16),
            head_wq if head_is_q8 else _dummy_u32(),
            head_s if head_is_q8 else _dummy_bf16(),
            head_b if head_is_q8 else _dummy_bf16(),
            head_dense if not head_is_q8 else _dummy_bf16(),
            fold_wq,
            fold_s,
            fold_b,
            fold_bias,
            cap_arr,
        ],
        template=[
            ("WT", wt),
            ("T_TILES", t_tiles),
            ("N_CAND_IN", t_tiles * _SCREEN_K_PER),
            ("K_MAX", _SCREEN_K_MAX_CAND),
            ("DELTA_VAL", int(_SCREEN_DELTA)),
            ("H", h_dim),
            ("R", r),
            ("HEAD_GS", head_gs),
            ("FOLD_GS", _MARKOV_FOLD_GROUP_SIZE),
            ("HEAD_Q8", head_is_q8),
        ],
        grid=(b * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(b,), (b,)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return tok, conf


def _markov_q8_greedy_argmax(
    latent: mx.array,
    w_q: mx.array,
    scales: mx.array,
    biases: mx.array,
    fold_bias: mx.array,
    base_logits: mx.array,
    v_unpad: int,
) -> mx.array:
    """Fused Q8 affine Markov projection + base_logits add + greedy argmax."""
    global _markov_q8_p1_kernel, _markov_argmax_p2_kernel
    if _markov_q8_p1_kernel is None:
        _markov_q8_p1_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_markov_q8_p1",
            input_names=[
                "latent",
                "w_q",
                "scales",
                "biases",
                "fold_bias",
                "base_logits",
            ],
            output_names=["part_v", "part_i"],
            source=_MARKOV_Q8_ARGMAX_P1_SOURCE,
        )
        _markov_argmax_p2_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_markov_argmax_p2",
            input_names=["part_v", "part_i"],
            output_names=["out_tok"],
            source=_MARKOV_ARGMAX_P2_SOURCE,
        )
    b, r = latent.shape
    v_full = base_logits.shape[-1]
    ntg = _MARKOV_ARGMAX_NTG
    pv, pi = _markov_q8_p1_kernel(
        inputs=[latent, w_q, scales, biases, fold_bias, base_logits],
        template=[
            ("R", r),
            ("GS", _MARKOV_FOLD_GROUP_SIZE),
            ("V_UNPAD", v_unpad),
            ("V_FULL", v_full),
        ],
        grid=(ntg * 256, b, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(b, ntg), (b, ntg)],
        output_dtypes=[mx.float32, mx.uint32],
    )
    return _markov_argmax_p2_kernel(
        inputs=[pv, pi],
        template=[("NTG", ntg)],
        grid=(b * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(b,)],
        output_dtypes=[mx.uint32],
    )[0]


@mx.compile
def mix_rmsnorm(context: mx.array, alpha: mx.array, weight: mx.array, eps: float):
    """Softmax-weighted sum of the context taps, RMS-normalized.

    ``context`` is ``[B, n, T * H]`` (taps concatenated in feature order).
    """
    b, n, width = context.shape
    taps = alpha.shape[0]
    x = context.reshape(b, n, taps, width // taps).astype(mx.float32)
    mixed = mx.sum(x * alpha.astype(mx.float32)[:, None], axis=2)
    return mx.fast.rms_norm(mixed, weight.astype(mx.float32), eps).astype(context.dtype)


class DraftAttention(nn.Module):
    def __init__(self, config: Inkling2DSparkConfig):
        super().__init__()
        h = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        dq = self.n_heads * self.head_dim
        dkv = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(h, dq, bias=False)
        self.k_proj = nn.Linear(h, dkv, bias=False)
        self.v_proj = nn.Linear(h, dkv, bias=False)
        self.o_proj = nn.Linear(dq, h, bias=False)
        self.k_proj_ctx = nn.Linear(h, dkv, bias=False)
        self.v_proj_ctx = nn.Linear(h, dkv, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm_ctx = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self._eps = config.rms_norm_eps
        # Bound by the draft model after loading: this layer's tap mixture and
        # its dflare_norm weight.
        self._alpha = None
        self._mix_weight = None

    def project_context_kv(self, context: mx.array):
        mixed = mix_rmsnorm(context, self._alpha, self._mix_weight, self._eps)
        return self.k_proj_ctx(mixed), self.v_proj_ctx(mixed)


class DraftMLP(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DraftLayer(nn.Module):
    def __init__(self, config: Inkling2DSparkConfig):
        super().__init__()
        h = config.hidden_size
        self.self_attn = DraftAttention(config)
        self.mlp = DraftMLP(h, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(h, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(h, eps=config.rms_norm_eps)
        self.dflare_norm = nn.RMSNorm(h, eps=config.rms_norm_eps)


class Inkling2MarkovHead(nn.Module):
    """RNN Markov head: per-step bias over the vocabulary.

    The bias is ``lm_head(hidden_out(latent))``; ``lm_head @ hidden_out`` is
    folded once at bind time so a step reads a ``[V, rank]`` matrix instead of
    the full LM head.
    """

    def __init__(self, config: Inkling2DSparkConfig):
        super().__init__()
        r, h = config.markov_rank, config.hidden_size
        self.rank = r
        self.markov_w1 = nn.Embedding(config.vocab_size, r)
        self.rnn_proj = nn.Linear(2 * r + 2 * h, 3 * r, bias=True)
        self.hidden_out = nn.Linear(r, h, bias=True)
        self.dflare_norm = nn.RMSNorm(h, eps=config.rms_norm_eps)
        self._eps = config.rms_norm_eps
        self._softcap = config.final_logit_softcapping
        self._unpadded_vocab_size = config.unpadded_vocab_size
        self._vocab_bias = None
        if config.unpadded_vocab_size < config.vocab_size:
            self._vocab_bias = mx.where(
                mx.arange(config.vocab_size) < config.unpadded_vocab_size,
                0.0,
                -mx.inf,
            ).astype(mx.float32)
        self._fold_weight = None
        self._fold_wq = None
        self._fold_scales = None
        self._fold_biases = None
        self._fold_bias = None
        self._head_wq = None
        self._head_scales = None
        self._head_biases = None
        self._head_dense = None
        self._head_gs = 64
        self._head_sc_wq = None
        self._head_sc_sc = None
        self._fold_sc_wq = None
        self._fold_sc_sc = None
        self._fold_bp = None
        self._screen_tiles = 0
        self._cap_arr = mx.array([self._softcap], dtype=mx.float32)
        self._rnn_w_sp = None
        self._rnn_w_h = None
        self._rnn_w_f = None
        self._rnn_ctx = None
        self._alpha = None
        self._f_rnn: dict[Any, mx.array] = {}
        self._hidden = None
        self._rows_f = None
        self._state = None
        self._step = 0

    @property
    def screen_enabled(self) -> bool:
        return self._head_sc_wq is not None and self._fold_sc_wq is not None

    def fold(self, lm_head: nn.Module) -> None:
        """Fold the target LM head into the latent projection and build 2-bit screen copies.

        At release vocab/rank sizes the folded ``[V, rank]`` table is quantized
        to 8-bit affine (matching the 8-bit target LM head) to halve per-step
        DRAM bandwidth while preserving greedy draft agreement, and 2-bit
        tile-transposed screen copies of ``lm_head`` and ``fold`` are packed
        for two-stage exact screened argmax + calibrated ``conf[i]``.
        """
        is_q8_affine = (
            isinstance(lm_head, nn.QuantizedLinear)
            and getattr(lm_head, "mode", "affine") == "affine"
            and int(getattr(lm_head, "bits", 0)) == 8
            and lm_head.get("biases") is not None
        )
        if isinstance(lm_head, nn.QuantizedLinear):
            weight = mx.dequantize(
                lm_head.weight,
                lm_head.scales,
                lm_head.get("biases"),
                lm_head.group_size,
                lm_head.bits,
                mode=getattr(lm_head, "mode", "affine"),
            )
        else:
            weight = lm_head.weight
        proj = self.hidden_out.weight.astype(mx.float32)
        bias = self.hidden_out.bias.astype(mx.float32)
        rows, biases = [], []
        for start in range(0, weight.shape[0], 16384):
            head = weight[start : start + 16384].astype(mx.float32)
            rows.append((head @ proj).astype(mx.bfloat16))
            biases.append((head @ bias).astype(mx.bfloat16))
            mx.eval(rows[-1], biases[-1])
        folded = mx.concatenate(rows, axis=0)
        self._fold_bias = mx.concatenate(biases, axis=0)
        self._head_wq = self._head_scales = self._head_biases = self._head_dense = None
        self._head_sc_wq = self._head_sc_sc = None
        self._fold_sc_wq = self._fold_sc_sc = self._fold_bp = None
        self._screen_tiles = 0
        if (
            folded.shape[0] >= 1024
            and self.rank % 128 == 0
            and self.rank >= _MARKOV_FOLD_GROUP_SIZE
        ):
            self._fold_wq, self._fold_scales, self._fold_biases = mx.quantize(
                folded,
                group_size=_MARKOV_FOLD_GROUP_SIZE,
                bits=_MARKOV_FOLD_BITS,
                mode="affine",
            )
            self._fold_weight = None
            eval_list = [
                self._fold_wq,
                self._fold_scales,
                self._fold_biases,
                self._fold_bias,
            ]
            h_dim = int(weight.shape[1])
            can_screen = (
                mx.default_device() == mx.gpu
                and h_dim % 64 == 0
                and self.rank % 64 == 0
                and (
                    is_q8_affine
                    or (
                        not isinstance(lm_head, nn.QuantizedLinear)
                        and lm_head.weight.dtype in (mx.bfloat16, mx.float16, mx.float32)
                    )
                )
            )
            if can_screen:
                if is_q8_affine:
                    self._head_wq = lm_head.weight
                    self._head_scales = lm_head.scales
                    self._head_biases = lm_head.biases
                    self._head_gs = int(lm_head.group_size)
                    self._head_sc_wq, self._head_sc_sc, t_tiles = _dspark_screen_pack(
                        v_unpad=self._unpadded_vocab_size,
                        n_in=h_dim,
                        w_q=self._head_wq,
                        w_scales=self._head_scales,
                        w_biases=self._head_biases,
                        src_gs=self._head_gs,
                    )
                else:
                    self._head_dense = lm_head.weight
                    self._head_gs = 64
                    self._head_sc_wq, self._head_sc_sc, t_tiles = _dspark_screen_pack(
                        v_unpad=self._unpadded_vocab_size,
                        n_in=h_dim,
                        w_dense=self._head_dense,
                    )
                self._fold_sc_wq, self._fold_sc_sc, _ = _dspark_screen_pack(
                    v_unpad=self._unpadded_vocab_size,
                    n_in=self.rank,
                    w_q=self._fold_wq,
                    w_scales=self._fold_scales,
                    w_biases=self._fold_biases,
                    src_gs=_MARKOV_FOLD_GROUP_SIZE,
                )
                self._fold_bp = _dspark_screen_order(
                    self._fold_bias, self._unpadded_vocab_size, t_tiles
                )
                self._screen_tiles = t_tiles
                # Split rnn_proj once (Basalt dspark_split_rnn) so ctx_f + ctx_h
                # run once per block and each step reads only the 0.78 MB w_sp.
                rw = self.rnn_proj.weight.astype(mx.bfloat16)
                r2 = 2 * self.rank
                self._rnn_w_sp = mx.contiguous(rw[:, :r2])
                self._rnn_w_h = mx.contiguous(rw[:, r2 : r2 + h_dim])
                self._rnn_w_f = mx.contiguous(rw[:, r2 + h_dim :])
                eval_list.extend(
                    [
                        self._head_sc_wq,
                        self._head_sc_sc,
                        self._fold_sc_wq,
                        self._fold_sc_sc,
                        self._fold_bp,
                        self._cap_arr,
                        self._rnn_w_sp,
                        self._rnn_w_h,
                        self._rnn_w_f,
                    ]
                )
            mx.eval(eval_list)
        else:
            self._fold_weight = folded
            self._fold_wq = self._fold_scales = self._fold_biases = None
            mx.eval(self._fold_weight, self._fold_bias)

    def screen_base(self, draft_hidden: mx.array) -> mx.array:
        """Stage 1A: approximate head logits in screen order for all block steps."""
        if not self.screen_enabled:
            raise RuntimeError("2-bit screened head is not initialized")
        return _dspark_head_base(
            draft_hidden, self._head_sc_wq, self._head_sc_sc, self._screen_tiles
        )

    def _is_screened_base(self, base_logits: mx.array | None) -> bool:
        return (
            self.screen_enabled
            and base_logits is not None
            and base_logits.dtype == mx.float16
            and base_logits.shape[-1] == self._screen_tiles * 32
        )

    def apply_step_greedy_with_conf(
        self, base_logits: mx.array, prev_tokens: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Greedy Markov step returning ``(token_ids [B] uint32, conf [B] float32)``.

        When ``base_logits`` is the fp16 screen-order output of ``screen_base``,
        runs Stage 1B (2-bit fold screen) + Stage 2 (exact Q8/BF16 rescore of
        top candidates + calibrated softmax probability ``conf``). Otherwise
        computes exact logits and softmax probability directly.
        """
        if (
            self._is_screened_base(base_logits)
            and mx.default_device() == mx.gpu
            and self._hidden is not None
        ):
            step_idx = self._step
            h_step = self._hidden[:, step_idx]
            latent, lat_f16 = self._step_latent_pair(prev_tokens)
            return _dspark_screened_pick(
                h_step=h_step,
                lat_step=latent,
                lat_f16=lat_f16,
                approx_base=base_logits,
                fold_sc_wq=self._fold_sc_wq,
                fold_sc_sc=self._fold_sc_sc,
                fold_bp=self._fold_bp,
                head_wq=self._head_wq,
                head_s=self._head_scales,
                head_b=self._head_biases,
                head_dense=self._head_dense,
                head_gs=self._head_gs,
                fold_wq=self._fold_wq,
                fold_s=self._fold_scales,
                fold_b=self._fold_biases,
                fold_bias=self._fold_bias,
                v_unpad=self._unpadded_vocab_size,
                t_tiles=self._screen_tiles,
                cap_arr=self._cap_arr,
            )
        logits = self.apply_step_logits(base_logits, prev_tokens)
        tok = mx.argmax(logits, axis=-1).astype(mx.uint32)
        m = mx.max(logits, axis=-1, keepdims=True)
        conf = mx.exp(mx.zeros_like(m[..., 0])) / mx.sum(mx.exp(logits - m), axis=-1)
        return tok, conf.astype(mx.float32)

    def prepare_block(self, draft_hidden, context, lengths, uids) -> None:
        """Reset the RNN for a new block; refresh each row's committed taps."""
        rows = []
        for index, (uid, n) in enumerate(zip(uids, lengths)):
            if n:
                last = context[index : index + 1, n - 1 : n]
                self._f_rnn[uid] = mix_rmsnorm(
                    last, self._alpha, self.dflare_norm.weight, self._eps
                )[0]
            rows.append(self._f_rnn[uid])
        self._rows_f = mx.concatenate(rows, axis=0)
        self._hidden = draft_hidden
        self._state = mx.zeros((len(uids), self.rank), dtype=draft_hidden.dtype)
        self._step = 0
        if (
            self._rnn_w_sp is not None
            and mx.default_device() == mx.gpu
            and draft_hidden.dtype == mx.bfloat16
        ):
            # Precompute ctx_h + ctx_f + bias once for the entire block!
            ctx_f = (
                self._rows_f.astype(mx.bfloat16) @ self._rnn_w_f.T
                + self.rnn_proj.bias.astype(mx.bfloat16)
            )
            ctx_h = draft_hidden @ self._rnn_w_h.T
            self._rnn_ctx = (ctx_h.astype(mx.float32) + ctx_f[:, None, :].astype(mx.float32))
        else:
            self._rnn_ctx = None

    def forget(self, uids) -> None:
        for uid in uids:
            self._f_rnn.pop(uid, None)

    def _step_latent_pair(self, prev_tokens: mx.array) -> tuple[mx.array, mx.array | None]:
        """Advance RNN state; uses fused Metal step when split weights are prepared."""
        if (
            self._rnn_w_sp is not None
            and mx.default_device() == mx.gpu
            and self._state.dtype == mx.bfloat16
            and self.markov_w1.weight.dtype == mx.bfloat16
        ):
            if self._rnn_ctx is None:
                ctx_f = (
                    self._rows_f.astype(mx.bfloat16) @ self._rnn_w_f.T
                    + self.rnn_proj.bias.astype(mx.bfloat16)
                )
                ctx_h = self._hidden.astype(mx.bfloat16) @ self._rnn_w_h.T
                self._rnn_ctx = (
                    ctx_h.astype(mx.float32) + ctx_f[:, None, :].astype(mx.float32)
                )
            ctx_step = self._rnn_ctx[:, self._step]
            self._state, lat_bf16, lat_f16 = _dspark_markov_step_fused(
                self._state,
                prev_tokens,
                ctx_step,
                self._rnn_w_sp,
                self.markov_w1.weight,
            )
            self._step += 1
            return lat_bf16, lat_f16
        return self._step_latent(prev_tokens), None

    def _step_latent(self, prev_tokens: mx.array) -> mx.array:
        state = self._state
        dtype = state.dtype
        inputs = mx.concatenate(
            [
                state,
                self.markov_w1(prev_tokens.reshape(-1)).astype(dtype),
                self._hidden[:, self._step].astype(dtype),
                self._rows_f.astype(dtype),
            ],
            axis=-1,
        )
        proj = self.rnn_proj(inputs).astype(mx.float32)
        r = self.rank
        gate = mx.sigmoid(proj[:, :r]).astype(dtype)
        kept = (gate * state).astype(dtype)
        blend = ((1.0 - gate) * mx.tanh(proj[:, r : 2 * r]).astype(dtype)).astype(dtype)
        self._state = kept + blend
        self._step += 1
        return mx.tanh(proj[:, 2 * r :]).astype(dtype)

    def apply_step_greedy(
        self, base_logits: mx.array, prev_tokens: mx.array
    ) -> mx.array:
        """Greedy Markov step: 2-bit screened exact argmax or fused Q8 argmax."""
        if (
            self._is_screened_base(base_logits)
            and mx.default_device() == mx.gpu
            and self._hidden is not None
        ):
            return self.apply_step_greedy_with_conf(base_logits, prev_tokens)[0]
        if (
            self._fold_wq is not None
            and mx.default_device() == mx.gpu
            and base_logits.dtype == mx.bfloat16
            and self._state.dtype == mx.bfloat16
            and self.rank % 128 == 0
        ):
            latent = self._step_latent(prev_tokens)
            return _markov_q8_greedy_argmax(
                latent,
                self._fold_wq,
                self._fold_scales,
                self._fold_biases,
                self._fold_bias,
                base_logits,
                self._unpadded_vocab_size,
            )
        return mx.argmax(self.apply_step_logits(base_logits, prev_tokens), axis=-1)

    def apply_step_logits(self, base_logits: mx.array, prev_tokens: mx.array):
        latent = self._step_latent(prev_tokens)
        if self._fold_wq is not None:
            bias = (
                mx.quantized_matmul(
                    latent,
                    self._fold_wq,
                    self._fold_scales,
                    self._fold_biases,
                    transpose=True,
                    group_size=_MARKOV_FOLD_GROUP_SIZE,
                    bits=_MARKOV_FOLD_BITS,
                    mode="affine",
                )
                + self._fold_bias
            )
        else:
            bias = latent @ self._fold_weight.T + self._fold_bias
        logits = base_logits.astype(mx.float32) + bias.astype(mx.float32)
        if self._softcap:
            logits = mx.tanh(logits / self._softcap) * self._softcap
        if self._vocab_bias is not None:
            logits = logits + self._vocab_bias
        return logits


class Inkling2DSparkModel(nn.Module):
    # The context rows are the raw target taps; each layer projects its own mix.
    raw_context = True

    def __init__(self, config: Inkling2DSparkConfig):
        super().__init__()
        self.config = config
        self.layers = [DraftLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mask_embed = mx.zeros((config.hidden_size,))
        self.fuse_weight = mx.zeros(
            (config.num_hidden_layers + 1, len(config.target_layer_ids))
        )
        self.markov_head = Inkling2MarkovHead(config)
        self.rope = nn.RoPE(config.head_dim, traditional=False, base=config.rope_theta)
        self.context_dtype = mx.bfloat16
        self._embed_fn = None
        self._lm_head_fn = None

    def bind(self, target: nn.Module) -> Inkling2DSparkModel:
        """Share the target's normalized embedding and LM head."""
        text = target.model
        self._embed_fn = lambda tokens: text.embed_norm(text.embed(tokens))
        lm_head = target.lm_head
        self._lm_head_fn = lambda hidden: lm_head(hidden)
        alpha = mx.softmax(self.fuse_weight.astype(mx.float32), axis=-1).astype(
            mx.bfloat16
        )
        for index, layer in enumerate(self.layers):
            layer.self_attn._alpha = alpha[index]
            layer.self_attn._mix_weight = layer.dflare_norm.weight
        self.markov_head._alpha = alpha[-1]
        self.markov_head.fold(lm_head)
        # Generation runs on another thread; nothing lazy may outlive the load.
        lazy = [layer.self_attn._alpha for layer in self.layers]
        lazy.append(self.markov_head._alpha)
        if self.markov_head._vocab_bias is not None:
            lazy.append(self.markov_head._vocab_bias)
        mx.eval(lazy)
        return self

    def _embed_input_tokens(self, tokens: mx.array) -> mx.array:
        return self._embed_fn(tokens)

    def _logits(self, hidden: mx.array) -> mx.array:
        return self._lm_head_fn(hidden)


# Draft layer projections run as 8-bit affine: every cycle reads them all, and
# the target verifies every proposal, so only proposal quality is affected.
_DRAFT_QUANT = {"group_size": 64, "bits": 8}


def load_inkling2_dspark_model(model_path: str | Path) -> Inkling2DSparkModel:
    path = Path(model_path)
    config = Inkling2DSparkConfig.from_target_config(
        json.loads((path / "config.json").read_text())
    )
    model = Inkling2DSparkModel(config)
    weights = mx.load(str(path / DRAFT_FILENAME))
    model.load_weights(list(weights.items()), strict=True)
    nn.quantize(
        model,
        **_DRAFT_QUANT,
        class_predicate=lambda p, m: isinstance(m, nn.Linear)
        and p.startswith("layers."),
    )
    mx.eval(model.parameters())
    return model


_GLOBAL_KV_STEP = 256
_GLOBAL_PROJ_CHUNK = 1024


def _make_inkling2_dspark_drafter_cls():
    from ...speculative.dflash import DFlashDrafter, selection, variants
    from ...speculative.dflash.drafter import _Cohort, _Predraft

    class Inkling2DSparkDrafter(DFlashDrafter):
        """DSpark drafter with per-layer local/global context and 2-bit screened head.

        - Local layers (``config.local_layer_ids``, e.g. ``[0, 2]``) keep the
          bounded ``sliding_window - 1`` context ring.
        - Global layers (e.g. layer ``1``) maintain a growing full-context K/V
          cache covering positions ``0 .. total - 1`` (matching Basalt / SGLang).
        - Greedy proposals run through the 2-bit screened exact head when
          available, avoiding the 492 MB Q8 ``lm_head`` projection and exposing
          per-position calibrated confidence ``state.draft_conf``.
        """

        def __init__(self, model: Inkling2DSparkModel, **kwargs):
            super().__init__(model, **kwargs)
            n_layers = len(model.layers)
            self._local_layers = {
                int(i) for i in (getattr(model.config, "local_layer_ids", None) or ())
            }
            self._global_layers = {
                i for i in range(n_layers) if i not in self._local_layers
            }
            # Per-row global K/V caches: uid -> {layer_idx: (k, v, length)}
            # where k, v have shape (n_kv, cap, head_dim) and rows [0..length)
            # hold positions [start .. start + length) (start = row_gstart[uid]).
            self._global_rows: dict[Any, dict[int, list]] = {}
            self._global_start: dict[Any, int] = {}
            self._predraft_global: dict[int, tuple[mx.array, mx.array, list[int]]] | None = None
            self._predraft_conf: list | None = None
            self._last_conf: dict[Any, list[mx.array]] = {}

        def _bound_pending(self, row) -> None:
            if self._global_layers:
                # Global layers attend to all committed tokens; keep pending
                # until projected into the per-layer caches on draft().
                return
            super()._bound_pending(row)

        def release(self, uids) -> None:
            uids = list(uids)
            self._predraft_global = None
            self._predraft_conf = None
            for uid in uids:
                self._global_rows.pop(uid, None)
                self._global_start.pop(uid, None)
                self._last_conf.pop(uid, None)
            super().release(uids)

        def reset_rows(self) -> None:
            self._predraft_global = None
            self._predraft_conf = None
            self._global_rows.clear()
            self._global_start.clear()
            self._last_conf.clear()
            super().reset_rows()

        def discard_predraft(self) -> None:
            self._predraft_global = None
            self._predraft_conf = None
            super().discard_predraft()

        def adopt_predraft(self, state, count: int) -> None:
            pg = self._predraft_global
            pc = self._predraft_conf
            self._predraft_global = None
            self._predraft_conf = None
            super().adopt_predraft(state, count)
            if pg is not None:
                uid = state.uid
                gmap = self._global_rows.setdefault(uid, {})
                for layer_idx, (gk, gv, lengths) in pg.items():
                    gmap[layer_idx] = [gk[0], gv[0], lengths[0]]
            if pc is not None:
                n_keep = self._proposal_count(state)
                state.draft_conf = pc[0][:n_keep]
                self._last_conf[state.uid] = state.draft_conf

        def _commit_proposals(self, state, tokens, accept_lps) -> None:
            super()._commit_proposals(state, tokens, accept_lps)
            confs = self._last_conf.get(state.uid)
            if confs is not None:
                count = int(state.drafts.shape[0])
                state.draft_conf = confs[:count]

        def _append_global_kv(
            self,
            uid: Any,
            layer_idx: int,
            k_new: mx.array,
            v_new: mx.array,
            start_pos: int,
            n_write: int,
            commit: bool,
        ) -> tuple[mx.array, mx.array, int]:
            """Append ``(n_kv, n_write, D)`` into ``uid``'s global K/V buffer.

            Returns ``(k_buf, v_buf, valid_len)`` where valid entries occupy
            ``[0 .. valid_len)`` starting at ``self._global_start[uid]``.
            """
            gmap = self._global_rows.setdefault(uid, {})
            entry = gmap.get(layer_idx)
            if uid not in self._global_start:
                self._global_start[uid] = start_pos
            g_start = self._global_start[uid]
            offset = max(0, start_pos - g_start)
            end = offset + n_write
            n_kv, _, head_dim = k_new.shape
            if entry is None:
                cap = max(_GLOBAL_KV_STEP, ((end + _GLOBAL_KV_STEP - 1) // _GLOBAL_KV_STEP) * _GLOBAL_KV_STEP)
                k_buf = mx.zeros((n_kv, cap, head_dim), dtype=k_new.dtype)
                v_buf = mx.zeros((n_kv, cap, head_dim), dtype=v_new.dtype)
                cur_len = 0
            else:
                k_buf, v_buf, cur_len = entry
                if not commit:
                    k_buf = mx.stop_gradient(k_buf)
                    v_buf = mx.stop_gradient(v_buf)
                cap = int(k_buf.shape[1])
                if end > cap:
                    new_cap = ((end + _GLOBAL_KV_STEP - 1) // _GLOBAL_KV_STEP) * _GLOBAL_KV_STEP
                    nk = mx.zeros((n_kv, new_cap, head_dim), dtype=k_buf.dtype)
                    nv = mx.zeros((n_kv, new_cap, head_dim), dtype=v_buf.dtype)
                    if cur_len > 0:
                        nk[:, :cur_len, :] = k_buf[:, :cur_len, :]
                        nv[:, :cur_len, :] = v_buf[:, :cur_len, :]
                    k_buf, v_buf = nk, nv
            if n_write > 0:
                k_buf[:, offset:end, :] = k_new[:, :n_write, :]
                v_buf[:, offset:end, :] = v_new[:, :n_write, :]
            new_len = max(cur_len, end)
            if commit:
                gmap[layer_idx] = [k_buf, v_buf, new_len]
            return k_buf, v_buf, new_len

        def _project_ctx_kv_chunked(
            self, attn, ctx_row: mx.array, base_pos: int, model
        ) -> tuple[mx.array, mx.array]:
            """Project ``(1, n, T*H)`` context row to RoPE'd ``(n_kv, n, D)`` K/V."""
            n = int(ctx_row.shape[1])
            if n == 0:
                return (
                    mx.zeros((attn.n_kv_heads, 0, attn.head_dim), dtype=ctx_row.dtype),
                    mx.zeros((attn.n_kv_heads, 0, attn.head_dim), dtype=ctx_row.dtype),
                )
            ctx_k_norm = getattr(attn, "k_norm_ctx", attn.k_norm)
            ks, vs = [], []
            for s0 in range(0, n, _GLOBAL_PROJ_CHUNK):
                s1 = min(n, s0 + _GLOBAL_PROJ_CHUNK)
                chunk = ctx_row[:, s0:s1]
                ck, cv = attn.project_context_kv(chunk)
                ck = ctx_k_norm(
                    ck.reshape(1, s1 - s0, attn.n_kv_heads, attn.head_dim)
                ).transpose(0, 2, 1, 3)
                cv = cv.reshape(
                    1, s1 - s0, attn.n_kv_heads, attn.head_dim
                ).transpose(0, 2, 1, 3)
                if self.value_scale is not None:
                    cv = cv * self.value_scale
                ck = model.rope(ck, offset=base_pos + s0)
                ks.append(ck[0])
                vs.append(cv[0])
            if len(ks) == 1:
                return ks[0], vs[0]
            return mx.concatenate(ks, axis=1), mx.concatenate(vs, axis=1)

        def _draft_batched(self, rows, counts=None, commit=True):
            model = self.model
            uids = [state.uid for state, *_ in rows]
            slots = self.ring_slots
            capacity = self.capacity
            batch = len(rows)
            block = self.block_size
            dtype = getattr(model, "context_dtype", mx.bfloat16)

            # Full un-truncated context per row (for global layers) and
            # bounded local segments (at most ``slots`` newest rows for local rings).
            full_contexts, full_bases, full_lengths = [], [], []
            local_segments, local_bases, local_lengths = [], [], []
            for _state, row, context, _anchor, _sampler in rows:
                prior = row.restored if commit else None
                if commit:
                    row.restored = None
                full_ctx = (
                    mx.concatenate([prior.astype(context.dtype), context], axis=1)
                    if prior is not None
                    else context
                )
                n_full = int(full_ctx.shape[1])
                full_contexts.append(full_ctx)
                full_bases.append(row.fed)
                full_lengths.append(n_full)

                skip_local = max(0, n_full - slots)
                loc = full_ctx[:, skip_local:] if skip_local else full_ctx
                local_segments.append(loc)
                local_bases.append(row.fed + skip_local)
                local_lengths.append(int(loc.shape[1]))

            # Padded local context batch for local layers.
            loc_width = max(1, max(local_lengths))
            h_ctx_local = mx.concatenate(
                [
                    (
                        seg
                        if seg.shape[1] == loc_width
                        else mx.concatenate(
                            [
                                seg,
                                mx.zeros(
                                    (1, loc_width - seg.shape[1], seg.shape[2]),
                                    dtype=seg.dtype,
                                ),
                            ],
                            axis=1,
                        )
                    )
                    for seg in local_segments
                ],
                axis=0,
            )

            anchors = mx.concatenate([anchor for _, _, _, anchor, _ in rows]).astype(
                mx.int32
            )
            markov = getattr(model, "markov_head", None)
            anchor_slot = markov is not None and bool(
                getattr(model.config, "sample_from_anchor", False)
            )
            if anchor_slot:
                block -= 1
            masks = mx.full(
                (batch, block - 1), int(model.config.mask_token_id), dtype=mx.int32
            )
            inputs = mx.concatenate([anchors[:, None], masks], axis=1)
            h = model._embed_input_tokens(inputs)
            if self.mask_embedding is not None:
                fill = self.mask_embedding.astype(h.dtype)
                h = mx.concatenate(
                    [h[:, :1], mx.broadcast_to(fill, (batch, block - 1, h.shape[-1]))],
                    axis=1,
                )

            cohort = self._assemble_cohort(uids, dtype)
            local_base_arr = mx.array(local_bases, dtype=mx.int32)
            full_base_arr = mx.array(full_bases, dtype=mx.int32)
            if counts is None:
                totals = mx.array(
                    [b + n for b, n in zip(full_bases, full_lengths)], dtype=mx.int32
                )
            else:
                totals = full_base_arr + mx.concatenate(counts).astype(mx.int32)
            query_offsets = totals

            write_slots = [
                mx.array([(b + j) % capacity for j in range(n)], dtype=mx.int32)
                for b, n in zip(local_bases, local_lengths)
            ]
            pos = mx.stop_gradient(cohort.pos)
            for index, (b, n, slots_b) in enumerate(
                zip(local_bases, local_lengths, write_slots)
            ):
                if not n:
                    continue
                pos[index, slots_b] = mx.arange(b, b + n, dtype=mx.int32)
            ring_valid = (
                (pos >= 0)
                & (pos >= (totals - slots)[:, None])
                & (pos < totals[:, None])
            )
            local_mask = mx.concatenate(
                [ring_valid, mx.ones((batch, block), dtype=mx.bool_)], axis=1
            )[:, None, None, :]

            new_keys, new_values = [], []
            predraft_global: dict[int, tuple[mx.array, mx.array, list[int]]] = {}
            for layer_index, layer in enumerate(model.layers):
                attn = layer.self_attn
                residual = h
                x = layer.input_layernorm(h)

                queries = attn.q_proj(x)
                prop_keys = attn.k_proj(x)
                prop_values = attn.v_proj(x)
                queries = attn.q_norm(
                    queries.reshape(batch, block, attn.n_heads, -1)
                ).transpose(0, 2, 1, 3)
                prop_keys = attn.k_norm(
                    prop_keys.reshape(batch, block, attn.n_kv_heads, -1)
                ).transpose(0, 2, 1, 3)
                prop_values = prop_values.reshape(
                    batch, block, attn.n_kv_heads, -1
                ).transpose(0, 2, 1, 3)
                if self.value_scale is not None:
                    prop_values = prop_values * self.value_scale
                queries = model.rope(queries, offset=query_offsets)
                prop_keys = model.rope(prop_keys, offset=query_offsets)

                if layer_index in self._global_layers:
                    # Global layer: project ALL new context tokens and append to
                    # unbounded growing per-row K/V buffer.
                    row_ks, row_vs, row_lens, row_starts = [], [], [], []
                    for index, (uid, fctx, fb, fn) in enumerate(
                        zip(uids, full_contexts, full_bases, full_lengths)
                    ):
                        ck_row, cv_row = self._project_ctx_kv_chunked(
                            attn, fctx, fb, model
                        )
                        k_buf, v_buf, vlen = self._append_global_kv(
                            uid, layer_index, ck_row, cv_row, fb, fn, commit
                        )
                        row_ks.append(k_buf)
                        row_vs.append(v_buf)
                        row_lens.append(vlen)
                        row_starts.append(self._global_start[uid])

                    max_len = max(row_lens) if row_lens else 0
                    if batch == 1:
                        ctx_k = row_ks[0][None, :, :max_len, :]
                        ctx_v = row_vs[0][None, :, :max_len, :]
                        if not commit:
                            predraft_global[layer_index] = (
                                row_ks[0][None],
                                row_vs[0][None],
                                row_lens,
                            )
                    else:
                        padded_k, padded_v = [], []
                        for kb, vb, vl in zip(row_ks, row_vs, row_lens):
                            ks_i = kb[:, :vl, :]
                            vs_i = vb[:, :vl, :]
                            if vl < max_len:
                                pad = mx.zeros(
                                    (kb.shape[0], max_len - vl, kb.shape[2]),
                                    dtype=kb.dtype,
                                )
                                ks_i = mx.concatenate([ks_i, pad], axis=1)
                                vs_i = mx.concatenate([vs_i, pad], axis=1)
                            padded_k.append(ks_i)
                            padded_v.append(vs_i)
                        ctx_k = mx.stack(padded_k, axis=0)
                        ctx_v = mx.stack(padded_v, axis=0)
                        if not commit:
                            predraft_global[layer_index] = (ctx_k, ctx_v, row_lens)

                    if batch == 1 and counts is None:
                        layer_mask = None
                    else:
                        g_pos = (
                            mx.array(row_starts, dtype=mx.int32)[:, None]
                            + mx.arange(max_len, dtype=mx.int32)[None, :]
                        )
                        g_valid = (
                            (
                                mx.arange(max_len, dtype=mx.int32)[None, :]
                                < mx.array(row_lens, dtype=mx.int32)[:, None]
                            )
                            & (g_pos < totals[:, None])
                        )
                        layer_mask = mx.concatenate(
                            [g_valid, mx.ones((batch, block), dtype=mx.bool_)], axis=1
                        )[:, None, None, :]
                    keys = mx.concatenate([ctx_k, prop_keys], axis=2)
                    values = mx.concatenate([ctx_v, prop_values], axis=2)
                    # Keep cohort ring slot untouched for global layers.
                    new_keys.append(cohort.keys[layer_index])
                    new_values.append(cohort.values[layer_index])
                else:
                    # Local sliding-window layer: update bounded ring buffer.
                    ctx_keys, ctx_values = attn.project_context_kv(h_ctx_local)
                    ctx_k_norm = getattr(attn, "k_norm_ctx", attn.k_norm)
                    ctx_keys = ctx_k_norm(
                        ctx_keys.reshape(
                            batch, loc_width, attn.n_kv_heads, attn.head_dim
                        )
                    ).transpose(0, 2, 1, 3)
                    ctx_values = ctx_values.reshape(
                        batch, loc_width, attn.n_kv_heads, attn.head_dim
                    ).transpose(0, 2, 1, 3)
                    if self.value_scale is not None:
                        ctx_values = ctx_values * self.value_scale
                    ctx_keys = model.rope(ctx_keys, offset=local_base_arr)

                    ring_keys = mx.stop_gradient(cohort.keys[layer_index])
                    ring_values = mx.stop_gradient(cohort.values[layer_index])
                    for index, (n, slots_b) in enumerate(
                        zip(local_lengths, write_slots)
                    ):
                        if not n:
                            continue
                        ring_keys[index, :, slots_b, :] = ctx_keys[index, :, :n, :]
                        ring_values[index, :, slots_b, :] = ctx_values[index, :, :n, :]
                    new_keys.append(ring_keys)
                    new_values.append(ring_values)
                    keys = mx.concatenate([ring_keys, prop_keys], axis=2)
                    values = mx.concatenate([ring_values, prop_values], axis=2)
                    layer_mask = local_mask

                attended = mx.fast.scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    scale=attn.scale,
                    mask=layer_mask,
                    sinks=getattr(attn, "attention_sink_bias", None),
                )
                attended = attended.transpose(0, 2, 1, 3).reshape(batch, block, -1)
                attended = attn.o_proj(variants.gate_output(attn, x, attended))
                h = residual + attended

                residual = h
                x = layer.post_attention_layernorm(h)
                x = variants.gated_mlp(layer.mlp, x)
                h = residual + x

            if commit:
                cohort.keys = new_keys
                cohort.values = new_values
                cohort.pos = pos
                for index, ((_state, row, *_), fb, fn, ln) in enumerate(
                    zip(rows, full_bases, full_lengths, local_lengths)
                ):
                    row.fed = fb + fn
                    self._remember(row, h_ctx_local[index : index + 1, :ln])

            draft_hidden = model.norm(h) if anchor_slot else model.norm(h)[:, 1:]
            if commit and markov is not None:
                max_proposals = max(
                    (self._proposal_count(state) for state, *_ in rows), default=0
                )
                if 0 < max_proposals < draft_hidden.shape[1]:
                    draft_hidden = draft_hidden[:, :max_proposals]

            samplers = [sampler for *_, sampler in rows]
            greedy = all(sampler is None for sampler in samplers)
            if markov is not None:
                markov.prepare_block(draft_hidden, h_ctx_local, local_lengths, uids)
                if greedy and markov.screen_enabled and mx.default_device() == mx.gpu:
                    # Target 1: 2-bit screened exact head + calibrated conf[i]!
                    # Skips the 492 MB Q8 lm_head projection completely.
                    approx_base = markov.screen_base(draft_hidden)
                    prev = anchors.reshape(-1)
                    tokens_steps, conf_steps = [], []
                    for step in range(approx_base.shape[1]):
                        prev, conf = markov.apply_step_greedy_with_conf(
                            approx_base[:, step], prev
                        )
                        tokens_steps.append(prev)
                        conf_steps.append(conf)
                    proposals = [
                        ([tok[index : index + 1] for tok in tokens_steps], [])
                        for index in range(batch)
                    ]
                    row_confs = [
                        [c[index : index + 1] for c in conf_steps]
                        for index in range(batch)
                    ]
                    if commit:
                        for uid, rc in zip(uids, row_confs):
                            self._last_conf[uid] = rc
                    else:
                        self._predraft_conf = row_confs
                else:
                    logits = model._logits(draft_hidden)
                    proposals = selection.select_markov(markov, logits, anchors, samplers)
            else:
                logits = model._logits(draft_hidden)
                proposals = selection.sample_positions(logits, samplers)

            if not commit:
                self._predraft = _Predraft(
                    uids=tuple(uids),
                    keys=new_keys,
                    values=new_values,
                    pos=pos,
                    bases=full_bases,
                    unfed=[],
                    proposals=proposals,
                    contexts=[
                        h_ctx_local[i : i + 1, :n] for i, n in enumerate(local_lengths)
                    ],
                )
                self._predraft_global = predraft_global if predraft_global else None
            return proposals

    return Inkling2DSparkDrafter


def wrap_inkling2_dspark(
    model: Inkling2DSparkModel,
    source_path: str,
    depth: int | None = None,
    fixed: bool = False,
):
    """Wrap a bound draft for the batched verify path.

    The trained block is the anchor plus ``block_size - 1`` masks, and only
    the masks propose, so the drafter block equals ``block_size``.
    """
    drafter_cls = _make_inkling2_dspark_drafter_cls()
    drafter = drafter_cls(
        model,
        block_size=model.config.block_size,
        source_path=source_path,
        adaptive=not fixed,
        mask_embedding=model.mask_embed,
    )
    if depth:
        drafter.depth = max(1, min(drafter.depth, int(depth)))
    return drafter


def load_inkling2_dspark_drafter(
    model_path: str | Path,
    target_model: Any,
    depth: int | None = None,
    fixed: bool = False,
):
    """Load ``mtp.safetensors`` and wrap it for the batched verify path."""
    model = load_inkling2_dspark_model(model_path).bind(target_model)
    drafter = wrap_inkling2_dspark(model, str(model_path), depth, fixed)
    logger.info(
        "Inkling 2 DSpark drafter loaded: path=%s block=%d depth=%d adaptive=%s "
        "target_layers=%s",
        model_path,
        drafter.block_size,
        drafter.depth,
        drafter.adaptive,
        drafter.target_layer_ids,
    )
    return drafter
