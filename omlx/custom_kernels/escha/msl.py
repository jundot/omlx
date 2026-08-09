"""EXL3 trellis decode+GEMM Metal kernel (MSL).

Port of the kernel from dusterbloom/higgs (MIT, Copyright (c) 2026 Jonathan
Reyes; exllamav3 is the upstream reference for EXL3). Edited only to remove
Rust string-wrapper noise; the MSL template parameters K (bits/weight), TK and
TN (tile grid) are substituted textually by the caller.

Reference (bit-exact) decode of the same codes lives in ``fast.ref_trellis``.
"""
MIT = True  # provenance marker for packaging review

MSL = """
constexpr int NT = 128;         // threads in the threadgroup
constexpr int BM = 32;          // rows in a block
constexpr int TNB = 8;          // tile columns in a block
constexpr int BN = 16 * TNB;    // output columns in a block
constexpr int RM = 4;           // rows one thread owns
constexpr int RN = 8;           // columns one thread owns
constexpr int CG = BN / RN;     // column groups, BM / RM * CG must equal NT
constexpr int WORDS = 8 * K;    // 32-bit words in one packed tile
constexpr int IN = TK * 16;
constexpr int OUT = TN * 16;
constexpr int XP = BM + 1;      // the pad keeps the staged rows off one bank
constexpr int PAIRS = TNB * 128;    // code pairs in one tile row of a block
constexpr uint RMASK = (1u << RM) - 1u;

// The block must divide over the threads, and a thread must keep the same
// code pair on every tile it decodes. The second holds when NT is a whole
// number of the 128 pairs of one tile.
static_assert(BM / RM * CG == NT, "the threads must cover the output block");
static_assert(NT % 128 == 0, "a thread must own one code pair of every tile");
static_assert(PAIRS % NT == 0, "the code pairs must divide over the threads");
static_assert(BM * 16 % NT == 0, "the activation slab must divide over the threads");

threadgroup float x_sh[16 * XP];
threadgroup float w_sh[16 * BN];
threadgroup uint e_sh[BM];

uint tid = thread_index_in_threadgroup;
uint rows = cb[4];
uint row0 = threadgroup_position_in_grid.y * uint(BM);
uint col0 = threadgroup_position_in_grid.x * uint(BN);
uint nrow = min(uint(BM), rows - row0);

if (tid < nrow) {
    e_sh[tid] = eids[row0 + tid];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// The thread owns RM rows and RN columns of the block. CG threads share one
// row group, so they read the same activation values as a broadcast and one
// contiguous span of the weight block between them.
uint rbase = (tid / uint(CG)) * uint(RM);
uint cbase = (tid % uint(CG)) * uint(RN);

float acc[RM][RN];
for (uint i = 0u; i < uint(RM); ++i) {
    for (uint j = 0u; j < uint(RN); ++j) {
        acc[i][j] = 0.0f;
    }
}

// The decode address math of this thread. A thread owns the same code pair
// of every tile it touches, so the bit offsets and the two element slots
// hold for the whole walk and leave the loops below.
//
// The pair index is tid % 128 and the tile column starts at tid / 128, since
// the block holds TNB * 128 pairs and the threadgroup holds 256 threads.
uint p = tid & 127u;
uint tb0 = tid >> 7;

// The bit offsets copy unpack_tile. The wrap term 256 * K comes before the
// term -16. Thus the unsigned value stays 0 or more.
uint b0 = 2u * p * uint(K) + uint(K) + 256u * uint(K) - 16u;
uint b2 = b0 + uint(K) + 16u;
uint i0 = (b0 / 32u) % uint(WORDS);
uint i1w = (b2 - 1u) / 32u;
uint s1 = (i1w + 1u) * 32u - b2;
uint i1 = i1w % uint(WORDS);

// The closed form of tile_perm gives the slot of the two elements inside a
// tile. The tile column adds tb * 16 to the column at use.
uint woff[2];
for (uint e = 0u; e < 2u; ++e) {
    uint i = 2u * p + e;
    uint g = i >> 3;
    uint ii = i & 3u;
    uint r = (g % 4u) * 2u + (ii & 1u) + 8u * (ii >> 1);
    uint c = g / 4u + 8u * ((i >> 2) & 1u);
    woff[e] = r * uint(BN) + c;
}

// The activation slot of this thread. It is fixed for the same reason.
uint xm = tid >> 4;
uint xk = tid & 15u;
uint tn0 = col0 / 16u;

// The expert walk. Every thread reads the same e_sh and derives the same
// mask, so the pass count is uniform across the threadgroup.
uint valid = (nrow >= 32u) ? 0xFFFFFFFFu : ((1u << nrow) - 1u);
uint done = 0u;

while (done != valid) {
    uint pending = valid & ~done;
    uint cur = 0u;
    uint live = 0u;
    bool found = false;
    for (uint m = 0u; m < nrow; ++m) {
        if (((pending >> m) & 1u) == 0u) {
            continue;
        }
        if (!found) {
            cur = e_sh[m];
            found = true;
        }
        if (e_sh[m] == cur) {
            live |= 1u << m;
        }
    }
    done |= live;

    // Whether this thread holds a row of the pass. A sorted block splits
    // between experts at one row, so on each extra pass most row groups are
    // empty and skip the product below.
    bool active = ((live >> rbase) & RMASK) != 0u;

    // The packed tiles of this expert. A tile starts on a multiple of
    // 16 * K shorts, which is a multiple of four bytes, so the 32-bit view
    // of the words stays aligned.
    const device uint* ecode = (const device uint*)(
        code + ulong(cur) * ulong(TK) * ulong(TN) * ulong(16 * K));

    for (uint tk = 0u; tk < uint(TK); ++tk) {
        // The barrier closes the read of the previous step.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Stage the activation slab. A row outside the expert stages zero.
        for (uint q = 0u; q < uint(BM) * 16u / uint(NT); ++q) {
            uint m = xm + q * (uint(NT) / 16u);
            float v = 0.0f;
            if (((live >> m) & 1u) != 0u) {
                v = xh[(row0 + m) * uint(IN) + tk * 16u + xk];
            }
            x_sh[xk * uint(XP) + m] = v;
        }

        // Decode the TNB tiles of this tile row. The bit math and the
        // element order copy the tile decode kernel.
        for (uint q = 0u; q < uint(PAIRS) / uint(NT); ++q) {
            uint tb = tb0 + q * (uint(NT) / 128u);
            uint tn = tn0 + tb;

            uint w1 = 0u;
            if (tn < uint(TN)) {
                const device uint* tile =
                    ecode + (tk * uint(TN) + tn) * uint(WORDS);

                // The 64-bit funnel makes the shift safe when s1 is 0.
                ulong bits = (ulong(tile[i0]) << 32) | ulong(tile[i1]);
                w1 = uint(bits >> s1);

                // The codebook hash. The half cast repeats the f16 round
                // of the CPU decode.
                uint h0 = ((w1 >> uint(K)) & 0xFFFFu) * cb[0] + cb[1];
                uint h1 = (w1 & 0xFFFFu) * cb[0] + cb[1];
                half2 v0 = as_type<half2>((h0 & cb[2]) ^ cb[3]);
                half2 v1 = as_type<half2>((h1 & cb[2]) ^ cb[3]);
                w_sh[woff[0] + tb * 16u] =
                    float(half(float(v0.x) + float(v0.y)));
                w_sh[woff[1] + tb * 16u] =
                    float(half(float(v1.x) + float(v1.y)));
            } else {
                w_sh[woff[0] + tb * 16u] = 0.0f;
                w_sh[woff[1] + tb * 16u] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            for (uint kk = 0u; kk < 16u; ++kk) {
                float a[RM];
                float b[RN];
                for (uint i = 0u; i < uint(RM); ++i) {
                    a[i] = x_sh[kk * uint(XP) + rbase + i];
                }
                for (uint j = 0u; j < uint(RN); ++j) {
                    b[j] = w_sh[kk * uint(BN) + cbase + j];
                }
                for (uint i = 0u; i < uint(RM); ++i) {
                    for (uint j = 0u; j < uint(RN); ++j) {
                        acc[i][j] = fma(a[i], b[j], acc[i][j]);
                    }
                }
            }
        }
    }
}

for (uint i = 0u; i < uint(RM); ++i) {
    uint m = rbase + i;
    if (m >= nrow) {
        break;
    }
    for (uint j = 0u; j < uint(RN); ++j) {
        uint n = col0 + cbase + j;
        if (n < uint(OUT)) {
            dst[(row0 + m) * uint(OUT) + n] = acc[i][j];
        }
    }
}

"""
