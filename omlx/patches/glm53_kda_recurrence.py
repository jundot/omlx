"""KDA recurrence kernels for GLM-5.3 prefill.

GLM-5.3's linear-attention layers run the vector-gated delta rule (Kimi
delta attention): per token and head, with a per-channel forget gate g,

    S = S * g                 (decay, broadcast over value rows)
    p = S . k                 (128-wide dot per value row)
    S = S + k * (v - p) * beta
    y = S . q

The stock prefill evaluates it with ``gated_delta_kernel``: one simdgroup
per value row, g materialized as an fp32 [T, H, 128] tensor by
``compute_g_safe``. Both kernels here run the same recurrence with the gate
exp(lb * sigmoid(exp(A_log) * (a + dt_bias))) evaluated while staging, with
``compute_g_safe``'s expression tree and MLX's Sigmoid functor (bit-identical
gate), so the fp32 gate tensor never round-trips through memory. The fp32
state stays in registers: a value row is 8 lanes x 16 channels, and per
16-channel segment the decay, a float4 k.S / q.S accumulation reduced as
(x + y) + (z + w), then the pairing tree (s, s^4) -> (s, s^2) -> (s, s^1)
over the 8 segments. That is ``gated_delta_kernel``'s per-step math with a
different fp32 summation order of the two dots, and the same order in both
kernels, so they are bit-identical to each other.

* Per-core (default on NAX hosts): the H*128 value rows are split into one contiguous
  range per GPU core (~102 rows on an 80-core M5 Ultra), so every core runs
  one threadgroup with the same amount of work; a range may cross a head
  boundary, and its threadgroup then stages k, q, gate and beta for both
  heads. One value row per 8 lanes (26 simdgroups per threadgroup) spreads a
  core's rows over its SIMD units in steps of 4 rows. Full 12-token blocks
  are unrolled, and the q.S partials of four steps are reduced together (a
  reduce-scatter with the same pairing tree, so bit-identical) after the next
  step's decay and k.S FMAs.
* Blocked (``RecurrenceConfig``): 64 value rows per threadgroup, two rows per
  8 lanes, two threadgroups per head, 16-token blocks. The default on hosts
  without NAX (per-core measured ~1.5x slower on an 80-core M3 Ultra), and
  the fallback when the GPU core count is unknown, a core would get more
  than 128 rows, or the dtype is not 16-bit.
"""

from __future__ import annotations

import functools
import re
import subprocess
from typing import NamedTuple, Optional, Tuple

import mlx.core as mx

from omlx.custom_kernels.nax import is_nax_available


class RecurrenceConfig(NamedTuple):
    """Launch shape of the blocked recurrence (every variant is exact)."""

    tb: int = 16  # tokens staged per threadgroup block
    db: int = 64  # value rows per threadgroup
    rows: int = 2  # value rows per thread (1 or 2)
    prefetch: bool = True  # load the next block into registers meanwhile

    @property
    def threads(self) -> int:
        return self.db // self.rows * 8


DEFAULT_CONFIG = RecurrenceConfig()

_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""


def _reduce_all(rows: int) -> str:
    """Sum part[rows] over the 8 lanes of a row group into P[rows] on every lane."""
    if rows == 1:
        return """
            float P[1] = {part[0]};
            P[0] += simd_shuffle_xor(P[0], 4);
            P[0] += simd_shuffle_xor(P[0], 2);
            P[0] += simd_shuffle_xor(P[0], 1);"""
    # Reduce-scatter (each half of the lanes finishes one row), then swap.
    return """
            const bool h2 = (seg & 4) != 0;
            float keep = h2 ? part[1] : part[0];
            keep += simd_shuffle_xor(h2 ? part[0] : part[1], 4);
            keep += simd_shuffle_xor(keep, 2);
            keep += simd_shuffle_xor(keep, 1);
            const float other = simd_shuffle_xor(keep, 4);
            float P[2] = {h2 ? other : keep, h2 ? keep : other};"""


def _reduce_one(rows: int) -> str:
    """Sum part[rows] over the row lanes; `writer` lanes hold row `own` in `keep`."""
    if rows == 1:
        return """
            float keep = part[0];
            keep += simd_shuffle_down(keep, 4);
            keep += simd_shuffle_down(keep, 2);
            keep += simd_shuffle_down(keep, 1);
            const int own = 0;
            const bool writer = seg == 0;"""
    return """
            const bool h2 = (seg & 4) != 0;
            float keep = h2 ? part[1] : part[0];
            keep += simd_shuffle_xor(h2 ? part[0] : part[1], 4);
            keep += simd_shuffle_xor(keep, 2);
            keep += simd_shuffle_xor(keep, 1);
            const int own = h2 ? 1 : 0;
            const bool writer = (seg & 3) == 0;"""


# compute_g_safe: exp(lb * sigmoid(exp(A_log) * (a + dt_bias))) with MLX's
# Sigmoid functor, rounded like the compiled reference -> bit-identical gate.
_GATE_EXPR = """{
                const float x = decay * (static_cast<float>(SRC) + dtb[d]);
                const float e = 1 / (1 + metal::precise::exp(metal::abs(x)));
                const float sig = (x < 0) ? e : 1 - e;
                g_s[r][d] = metal::precise::exp(lb * sig);
            }"""


def _source(cfg: RecurrenceConfig) -> str:
    if cfg.rows not in (1, 2) or cfg.db % (cfg.rows * 4) or cfg.threads > 1024:
        raise ValueError(f"unsupported recurrence config {cfg}")
    if cfg.prefetch:
        prologue = """
    constexpr int NQK = (TB * Dk + NT - 1) / NT;
    constexpr int NV = (TB * DB + NT - 1) / NT;
    InT pk[NQK];
    InT pq[NQK];
    InT pa[NQK];
    InT pv[NV];
    InT pb = InT(0);
#define KDA_FETCH(T0N) { \\
        const int ttn = min(TB, T - (T0N)); \\
        for (int j = 0; j < NQK; ++j) { \\
            const int p = tid + j * NT; \\
            if (p < ttn * Dk) { \\
                const int r = p / Dk, d = p % Dk; \\
                const size_t off = (size_t)((T0N) + r) * qk_row + d; \\
                pk[j] = k_base[off]; \\
                pq[j] = q_base[off]; \\
                pa[j] = a_base[off]; \\
            } \\
        } \\
        for (int j = 0; j < NV; ++j) { \\
            const int p = tid + j * NT; \\
            if (p < ttn * DB) { \\
                pv[j] = v_base[(size_t)((T0N) + p / DB) * v_row + p % DB]; \\
            } \\
        } \\
        if (tid < ttn) pb = beta_base[(size_t)((T0N) + tid) * H]; \\
    }
    KDA_FETCH(0)"""
        staging = f"""
        for (int j = 0; j < NQK; ++j) {{
            const int p = tid + j * NT;
            if (p < tt * Dk) {{
                const int r = p / Dk, d = p % Dk;
                k_s[r][d] = pk[j];
                q_s[r][d] = pq[j];
                {_GATE_EXPR.replace("SRC", "pa[j]")}
            }}
        }}
        for (int j = 0; j < NV; ++j) {{
            const int p = tid + j * NT;
            if (p < tt * DB) {{
                v_s[p / DB][p % DB] = pv[j];
            }}
        }}
        if (tid < tt) b_s[tid] = static_cast<float>(pb);"""
        after_barrier = "        if (t0 + TB < T) KDA_FETCH(t0 + TB)"
    else:
        prologue = ""
        staging = f"""
        for (int p = tid; p < tt * Dk; p += NT) {{
            const int r = p / Dk, d = p % Dk;
            const size_t off = (size_t)(t0 + r) * qk_row + d;
            k_s[r][d] = k_base[off];
            q_s[r][d] = q_base[off];
            {_GATE_EXPR.replace("SRC", "a_base[off]")}
        }}
        for (int p = tid; p < tt * DB; p += NT) {{
            v_s[p / DB][p % DB] = v_base[(size_t)(t0 + p / DB) * v_row + p % DB];
        }}
        for (int p = tid; p < tt; p += NT) {{
            b_s[p] = static_cast<float>(beta_base[(size_t)(t0 + p) * H]);
        }}"""
        after_barrier = ""

    return f"""
    constexpr int TB = {cfg.tb};
    constexpr int DB = {cfg.db};
    constexpr int R = {cfg.rows};
    constexpr int NT = {cfg.threads};
    const int tid = thread_position_in_threadgroup.x;
    const int blk = threadgroup_position_in_grid.x;
    const int h = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;
    const int dv0 = blk * DB;
    // thread -> (group of R value rows, 16-channel segment); the 8 segment
    // lanes of a row group are adjacent in one simdgroup.
    const int rg = tid / 8;
    const int seg = tid % 8;
    const int d0 = seg * 16;
    threadgroup InT k_s[TB][Dk + 8];
    threadgroup InT q_s[TB][Dk + 8];
    threadgroup InT v_s[TB][DB + 8];
    threadgroup float b_s[TB];
    threadgroup float g_s[TB][Dk + 4];

    const size_t qk_row = (size_t)H * Dk;
    const size_t v_row = (size_t)H * Dv;
    const device InT* k_base = k + ((size_t)b * T * H + h) * Dk;
    const device InT* q_base = q + ((size_t)b * T * H + h) * Dk;
    const device InT* a_base = a + ((size_t)b * T * H + h) * Dk;
    const device InT* v_base = v + ((size_t)b * T * H + h) * Dv + dv0;
    auto beta_base = beta + (size_t)b * T * H + h;
    const device float* dtb = dt_bias + (size_t)h * Dk;
    const float decay = metal::precise::exp(A_log[h]);
    const float lb = lower_bound[0];

    float4 st[R][4];
    for (int r = 0; r < R; ++r) {{
        const device float4* S_in = (const device float4*)(
            state_in + (((size_t)b * H + h) * Dv + dv0 + rg * R + r) * Dk + d0);
        for (int i = 0; i < 4; ++i) st[r][i] = S_in[i];
    }}
    device InT* y_base = y + ((size_t)b * T * H + h) * Dv + dv0 + rg * R;{prologue}

    for (int t0 = 0; t0 < T; t0 += TB) {{
        const int tt = min(TB, T - t0);{staging}
        threadgroup_barrier(mem_flags::mem_threadgroup);
{after_barrier}
        for (int t = 0; t < tt; ++t) {{
            const threadgroup float4* g4 = (const threadgroup float4*)&g_s[t][d0];
            const float bt = b_s[t];
            const threadgroup vec<InT, 4>* k4 = (const threadgroup vec<InT, 4>*)&k_s[t][d0];
            const threadgroup vec<InT, 4>* q4 = (const threadgroup vec<InT, 4>*)&q_s[t][d0];
            float4 kf[4];
            for (int i = 0; i < 4; ++i) kf[i] = float4(k4[i]);
            float part[R];
            // decay the state rows in place, then k.S
            for (int r = 0; r < R; ++r) {{
                float4 p4 = 0.0f;
                for (int i = 0; i < 4; ++i) {{
                    st[r][i] = st[r][i] * g4[i];
                    p4 += st[r][i] * kf[i];
                }}
                part[r] = (p4.x + p4.y) + (p4.z + p4.w);
            }}
            {{{_reduce_all(cfg.rows)}
                // delta rule update, then q.S
                for (int r = 0; r < R; ++r) {{
                    const float delta =
                        (static_cast<float>(v_s[t][rg * R + r]) - P[r]) * bt;
                    float4 o4 = 0.0f;
                    for (int i = 0; i < 4; ++i) {{
                        st[r][i] = st[r][i] + kf[i] * delta;
                        o4 += st[r][i] * float4(q4[i]);
                    }}
                    part[r] = (o4.x + o4.y) + (o4.z + o4.w);
                }}
            }}
            {{{_reduce_one(cfg.rows)}
                if (writer) {{
                    y_base[(size_t)(t0 + t) * v_row + own] = static_cast<InT>(keep);
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    for (int r = 0; r < R; ++r) {{
        device float4* S_out = (device float4*)(
            state_out + (((size_t)b * H + h) * Dv + dv0 + rg * R + r) * Dk + d0);
        for (int i = 0; i < 4; ++i) S_out[i] = st[r][i];
    }}
"""


_KERNELS: dict = {}


def _kernel(cfg: RecurrenceConfig):
    kernel = _KERNELS.get(cfg)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=(
                f"omlx_glm53_kda_recurrence_tb{cfg.tb}_db{cfg.db}_r{cfg.rows}"
                f"{'_pf' if cfg.prefetch else ''}"
            ),
            input_names=[
                "q", "k", "v", "a", "beta", "A_log", "dt_bias", "lower_bound",
                "state_in", "T",
            ],
            output_names=["y", "state_out"],
            source=_source(cfg),
            header=_HEADER,
        )
        _KERNELS[cfg] = kernel
    return kernel


def _blocked(q, k, v, a, beta, a_log, dt_bias, lower_bound, state, cfg):
    B, T, H, Dk = q.shape
    Dv = v.shape[-1]
    dtype = q.dtype
    return _kernel(cfg)(
        inputs=[
            q,
            k,
            v,
            a.astype(dtype),
            beta.astype(dtype),
            a_log.reshape(H).astype(mx.float32),
            dt_bias.reshape(H, Dk).astype(mx.float32),
            mx.array([lower_bound], dtype=mx.float32),
            state.astype(mx.float32),
            T,
        ],
        template=[("InT", dtype), ("Dk", Dk), ("Dv", Dv), ("H", H)],
        grid=(cfg.threads * (Dv // cfg.db), H, B),
        threadgroup=(cfg.threads, 1, 1),
        output_shapes=[(B, T, H, Dv), (B, H, Dv, Dk)],
        output_dtypes=[dtype, mx.float32],
    )


# ---------------------------------------------------------------------------
# Per-core kernel: one threadgroup per GPU core over a contiguous range of the
# H*128 value rows (see the module docstring). Row ranges are even-aligned and
# at most 128 rows, so a threadgroup covers the tail of one head and/or the
# start of the next; k/q/gate/beta are staged in two head slots.
# ---------------------------------------------------------------------------

_PC_TB = 12  # tokens per staged block (full blocks are unrolled)
_PC_MAX_ROWS = 128  # rows per threadgroup: 8 lanes each, and at most 2 heads


class PerCoreConfig(NamedTuple):
    """Launch shape of the per-core recurrence (every variant is exact)."""

    threadgroups: Optional[int] = None  # default: one per GPU core
    tb: int = _PC_TB


@functools.lru_cache(maxsize=1)
def gpu_core_count() -> Optional[int]:
    """GPU core count from the IORegistry (AGXAccelerator gpu-core-count)."""
    try:
        out = subprocess.run(
            ["/usr/sbin/ioreg", "-r", "-c", "AGXAccelerator", "-d", "1"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except Exception:
        return None
    match = re.search(r'"gpu-core-count"\s*=\s*(\d+)', out)
    return int(match.group(1)) if match else None


def _percore_threadgroups(rows: int, cfg: PerCoreConfig) -> Optional[int]:
    ntg = cfg.threadgroups
    if ntg is None:
        # One threadgroup per core: measured to land exactly one per core,
        # whereas 1.25x-2x the core count does not spread evenly. GPUs whose
        # cores would need more than 128 rows each keep the blocked kernel.
        ntg = gpu_core_count()
        if not ntg or rows > ntg * _PC_MAX_ROWS:
            return None
    ntg = min(ntg, rows // 2)
    if ntg < 1 or 2 * -(-(rows // 2) // ntg) > _PC_MAX_ROWS:
        return None
    return ntg


def _pc_gate(src: str, d: str, s: str, dst: str) -> str:
    return f"""{{
                const float x = decay[{s}] * (static_cast<float>({src}) + dtb[{s}][{d}]);
                const float e = 1 / (1 + metal::precise::exp(metal::abs(x)));
                const float sig = (x < 0) ? e : 1 - e;
                {dst} = metal::precise::exp(lb * sig);
            }}"""


def _pc_reduce_y(t: str) -> str:
    # q.S partial of one step -> y on lane 0 (tree 4, 2, 1)
    return f"""
            {{
                float yv = o;
                yv += simd_shuffle_xor(yv, 4);
                yv += simd_shuffle_xor(yv, 2);
                yv += simd_shuffle_xor(yv, 1);
                if (ln == 0 && active) y_row[(size_t)(t0 + {t}) * v_row] = static_cast<InT>(yv);
            }}"""


def _pc_reduce_y4(s0: int) -> str:
    # q.S partials of steps s0..s0+3 (yq0..yq3) -> a reduce-scatter over the
    # 8 lanes with the tree of _pc_reduce_y (lane ^4, ^2, ^1), leaving step
    # s0 + 2*b2 + b1 on the lanes (b2, b1, 0): bit-identical to per-step sums.
    return f"""
            {{
                const float ya = (qb2 ? yq2 : yq0) + simd_shuffle_xor(qb2 ? yq0 : yq2, 4);
                const float yb = (qb2 ? yq3 : yq1) + simd_shuffle_xor(qb2 ? yq1 : yq3, 4);
                float yv = (qb1 ? yb : ya) + simd_shuffle_xor(qb1 ? ya : yb, 2);
                yv += simd_shuffle_xor(yv, 1);
                if ((ln & 1) == 0 && active)
                    y_row[(size_t)(t0 + {s0} + ysel) * v_row] = static_cast<InT>(yv);
            }}"""


def _pc_step(t, reduce4_prev: Optional[int], keep: Optional[int]) -> str:
    """One recurrence step at block-local index t (int, or "t" in the tail loop).

    reduce4_prev: reduce the four pending q.S partials of steps
    reduce4_prev..+3 after this step's decay and k.S FMAs. keep: store this
    step's partial in yq<keep> (None: reduce it right away).
    """
    code = f"""
        {{ // step {t}
            const threadgroup float4* g4 = (const threadgroup float4*)&g_s[slot][{t}][d0];
            const threadgroup vec<InT, 4>* k4 = (const threadgroup vec<InT, 4>*)&k_s[slot][{t}][d0];
            const threadgroup vec<InT, 4>* q4 = (const threadgroup vec<InT, 4>*)&q_s[slot][{t}][d0];
            const float bt = b_s[slot][{t}];
            const float vt = static_cast<float>(v_s[{t}][rg]);
            float4 kf[4];
            for (int i = 0; i < 4; ++i) kf[i] = float4(k4[i]);
            // decay the state row in place, then k.S
            float4 p4 = 0.0f;
            for (int i = 0; i < 4; ++i) {{
                st[i] = st[i] * g4[i];
                p4 += st[i] * kf[i];
            }}
            float p = (p4.x + p4.y) + (p4.z + p4.w);"""
    if reduce4_prev is not None:
        code += _pc_reduce_y4(reduce4_prev)
    code += """
            p += simd_shuffle_xor(p, 4);
            p += simd_shuffle_xor(p, 2);
            p += simd_shuffle_xor(p, 1);
            // delta rule update, then q.S
            const float delta = (vt - p) * bt;
            float4 o4 = 0.0f;
            for (int i = 0; i < 4; ++i) {
                st[i] = st[i] + kf[i] * delta;
                o4 += st[i] * float4(q4[i]);
            }
            const float o = (o4.x + o4.y) + (o4.z + o4.w);"""
    if keep is None:
        code += _pc_reduce_y(t)
    else:
        code += f"\n            yq{keep} = o;"
    return code + "\n        }\n"


def _percore_source(tb: int, max_rows: int) -> str:
    n4 = tb // 4 * 4
    full = ""
    for t in range(tb):
        if t < n4:
            full += _pc_step(t, t - 4 if t % 4 == 0 and t > 0 else None, t % 4)
        else:
            if t == n4 and n4 > 0:
                full += _pc_reduce_y4(n4 - 4)
            full += _pc_step(t, None, None)
    if n4 == tb:
        full += _pc_reduce_y4(tb - 4)
    tail = "for (int t = 0; t < tt; ++t) " + _pc_step("t", None, None)
    gates = f"""
        for (int j = 0; j < NQK; ++j) {{
            const int p = tid + j * NT;
            const int s = min(p / (TB * (Dk / 4)), 1);
            const int c4 = (p % (TB * (Dk / 4))) % (Dk / 4);
            for (int c = 0; c < 4; ++c) {_pc_gate("pa[j][c]", "4 * c4 + c", "s", "gv[j][c]")}
        }}"""
    return f"""
    constexpr int TB = {tb};
    constexpr int MR = {max_rows};
    constexpr int NT = MR * 8;
    constexpr int TOTAL = H * Dv;
    const int tid = thread_position_in_threadgroup.x;
    const int tg = threadgroup_position_in_grid.x;
    const int b = threadgroup_position_in_grid.z;
    // even-aligned row range [r0, r1) of the flattened (head, value row) axis
    const int r0 = 2 * ((tg * (TOTAL / 2)) / NTG);
    const int r1 = 2 * (((tg + 1) * (TOTAL / 2)) / NTG);
    const int nrows = r1 - r0;
    const int hA = r0 / Dv;
    const int nh = (r1 - 1) / Dv - hA + 1;  // heads covered: 1 or 2
    const int rg = tid / 8;  // value row within the range
    const int ln = tid % 8;  // 16-channel segment
    const int d0 = ln * 16;
    const bool active = r0 + rg < r1;
    const int slot = active ? ((r0 + rg) / Dv - hA) : 0;
    const bool qb2 = (ln & 4) != 0, qb1 = (ln & 2) != 0;
    const int ysel = (qb2 ? 2 : 0) + (qb1 ? 1 : 0);

    threadgroup InT k_s[2][TB][Dk + 8];
    threadgroup InT q_s[2][TB][Dk + 8];
    threadgroup float g_s[2][TB][Dk + 4];
    threadgroup InT v_s[TB][MR];
    threadgroup float b_s[2][TB];

    const size_t qk_row = (size_t)H * Dk;
    const size_t v_row = (size_t)H * Dv;
    const device InT* k_base = k + ((size_t)b * T * H + hA) * Dk;
    const device InT* q_base = q + ((size_t)b * T * H + hA) * Dk;
    const device InT* a_base = a + ((size_t)b * T * H + hA) * Dk;
    const device InT* v_base = v + (size_t)b * T * H * Dv + r0;
    auto beta_base = beta + (size_t)b * T * H + hA;
    device InT* y_row = y + (size_t)b * T * H * Dv + r0 + rg;
    const int hB = min(hA + 1, H - 1);
    const device float* dtb[2] = {{dt_bias + (size_t)hA * Dk, dt_bias + (size_t)hB * Dk}};
    const float decay[2] = {{metal::precise::exp(A_log[hA]), metal::precise::exp(A_log[hB])}};
    const float lb = lower_bound[0];

    float4 st[4];
    {{
        const device float4* S_in = (const device float4*)(
            state_in + ((size_t)b * H * Dv + (active ? r0 + rg : r0)) * Dk + d0);
        for (int i = 0; i < 4; ++i) st[i] = S_in[i];
    }}

    // register prefetch of one block for both head slots
    constexpr int NQK = (2 * TB * (Dk / 4) + NT - 1) / NT;
    constexpr int NV = (TB * (MR / 2) + NT - 1) / NT;
    vec<InT, 4> pk[NQK], pq[NQK], pa[NQK];
    float4 gv[NQK];
    for (int j = 0; j < NQK; ++j) {{
        pk[j] = vec<InT, 4>(0);
        pq[j] = vec<InT, 4>(0);
        pa[j] = vec<InT, 4>(0);
    }}
    vec<InT, 2> pv[NV];
    InT pb = InT(0);
#define KDA_FETCH(T0N) {{ \\
        const int ttn = min(TB, T - (T0N)); \\
        for (int j = 0; j < NQK; ++j) {{ \\
            const int p = tid + j * NT; \\
            const int s = p / (TB * (Dk / 4)), rem = p % (TB * (Dk / 4)); \\
            const int r = rem / (Dk / 4), c4 = rem % (Dk / 4); \\
            if (s < nh && r < ttn) {{ \\
                const size_t off = (size_t)((T0N) + r) * qk_row + (size_t)s * Dk + 4 * c4; \\
                pk[j] = *(const device vec<InT, 4>*)(k_base + off); \\
                pq[j] = *(const device vec<InT, 4>*)(q_base + off); \\
                pa[j] = *(const device vec<InT, 4>*)(a_base + off); \\
            }} \\
        }} \\
        for (int j = 0; j < NV; ++j) {{ \\
            const int p = tid + j * NT; \\
            const int r = p / (nrows / 2), c2 = p % (nrows / 2); \\
            if (r < ttn) \\
                pv[j] = *(const device vec<InT, 2>*)(v_base + (size_t)((T0N) + r) * v_row + 2 * c2); \\
        }} \\
        if (tid < nh * TB && tid % TB < ttn) \\
            pb = beta_base[(size_t)((T0N) + tid % TB) * H + tid / TB]; \\
    }}
#define KDA_STAGE(TTN) {{ \\
        for (int j = 0; j < NQK; ++j) {{ \\
            const int p = tid + j * NT; \\
            const int s = p / (TB * (Dk / 4)), rem = p % (TB * (Dk / 4)); \\
            const int r = rem / (Dk / 4), c4 = rem % (Dk / 4); \\
            if (s < nh && r < (TTN)) {{ \\
                *(threadgroup vec<InT, 4>*)(&k_s[s][r][4 * c4]) = pk[j]; \\
                *(threadgroup vec<InT, 4>*)(&q_s[s][r][4 * c4]) = pq[j]; \\
                *(threadgroup float4*)(&g_s[s][r][4 * c4]) = gv[j]; \\
            }} \\
        }} \\
        for (int j = 0; j < NV; ++j) {{ \\
            const int p = tid + j * NT; \\
            const int r = p / (nrows / 2), c2 = p % (nrows / 2); \\
            if (r < (TTN)) *(threadgroup vec<InT, 2>*)(&v_s[r][2 * c2]) = pv[j]; \\
        }} \\
        if (tid < nh * TB && tid % TB < (TTN)) b_s[tid / TB][tid % TB] = static_cast<float>(pb); \\
    }}

    KDA_FETCH(0)
{gates}
    KDA_STAGE(min(TB, T))
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float yq0 = 0.0f, yq1 = 0.0f, yq2 = 0.0f, yq3 = 0.0f;
    for (int t0 = 0; t0 < T; t0 += TB) {{
        const int tt = min(TB, T - t0);
        const bool more = t0 + TB < T;
        if (more) KDA_FETCH(t0 + TB)
        if (active) {{
            if (tt == TB) {{
{full}
            }} else {{
{tail}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (more) {{
{gates}
            KDA_STAGE(min(TB, T - t0 - TB))
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
    }}
#undef KDA_FETCH
#undef KDA_STAGE

    if (active) {{
        device float4* S_out = (device float4*)(
            state_out + ((size_t)b * H * Dv + r0 + rg) * Dk + d0);
        for (int i = 0; i < 4; ++i) S_out[i] = st[i];
    }}
"""


_PC_KERNELS: dict = {}


def _percore_kernel(tb: int, max_rows: int):
    key = (tb, max_rows)
    kernel = _PC_KERNELS.get(key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"omlx_glm53_kda_recurrence_percore_tb{tb}_r{max_rows}",
            input_names=[
                "q", "k", "v", "a", "beta", "A_log", "dt_bias", "lower_bound",
                "state_in", "T",
            ],
            output_names=["y", "state_out"],
            source=_percore_source(tb, max_rows),
            header=_HEADER,
        )
        _PC_KERNELS[key] = kernel
    return kernel


@functools.lru_cache(maxsize=None)
def _percore_launchable(tb: int, max_rows: int, dtype, Dk: int, Dv: int, H: int, ntg: int) -> bool:
    """Whether this GPU runs the per-core kernel with ``max_rows * 8`` threads.

    A pipeline's thread limit depends on the GPU and the kernel's register
    use (some GPUs allow only 640-768 threads here), and MLX only checks it
    when the kernel is evaluated, so probe one tiny launch per variant.
    """
    try:
        z = mx.zeros((1, 1, H, Dk), dtype=dtype)
        y, s = _percore_kernel(tb, max_rows)(
            inputs=[
                z,
                z,
                mx.zeros((1, 1, H, Dv), dtype=dtype),
                z,
                mx.zeros((1, 1, H), dtype=dtype),
                mx.zeros((H,), dtype=mx.float32),
                mx.zeros((H, Dk), dtype=mx.float32),
                mx.array([-5.0], dtype=mx.float32),
                mx.zeros((1, H, Dv, Dk), dtype=mx.float32),
                1,
            ],
            template=[("InT", dtype), ("Dk", Dk), ("Dv", Dv), ("H", H), ("NTG", ntg)],
            grid=(max_rows * 8 * ntg, 1, 1),
            threadgroup=(max_rows * 8, 1, 1),
            output_shapes=[(1, 1, H, Dv), (1, H, Dv, Dk)],
            output_dtypes=[dtype, mx.float32],
        )
        mx.eval(y, s)
        return True
    except Exception:  # noqa: BLE001 - any launch failure keeps the blocked kernel
        return False


def _percore(q, k, v, a, beta, a_log, dt_bias, lower_bound, state, cfg):
    """Per-core kernel launch, or None when the shape/device is not covered."""
    B, T, H, Dk = q.shape
    Dv = v.shape[-1]
    if (
        q.dtype not in (mx.bfloat16, mx.float16)
        or k.dtype != q.dtype
        or v.dtype != q.dtype
        or T < 1
        or cfg.tb < 1
    ):
        return None
    ntg = _percore_threadgroups(H * Dv, cfg)
    if ntg is None:
        return None
    max_rows = 2 * -(-(H * Dv // 2) // ntg)
    # k/q (2 bytes) + gate (fp32) for two head slots, v and beta per block
    tg_bytes = cfg.tb * (2 * 2 * (Dk + 8) * 2 + 2 * (Dk + 4) * 4 + max_rows * 2 + 2 * 4)
    if tg_bytes > 32768:
        return None
    dtype = q.dtype
    if not _percore_launchable(cfg.tb, max_rows, dtype, Dk, Dv, H, ntg):
        return None
    return _percore_kernel(cfg.tb, max_rows)(
        inputs=[
            q,
            k,
            v,
            a.astype(dtype),
            beta.astype(dtype),
            a_log.reshape(H).astype(mx.float32),
            dt_bias.reshape(H, Dk).astype(mx.float32),
            mx.array([lower_bound], dtype=mx.float32),
            state.astype(mx.float32),
            T,
        ],
        template=[("InT", dtype), ("Dk", Dk), ("Dv", Dv), ("H", H), ("NTG", ntg)],
        grid=(max_rows * 8 * ntg, 1, B),
        threadgroup=(max_rows * 8, 1, 1),
        output_shapes=[(B, T, H, Dv), (B, H, Dv, Dk)],
        output_dtypes=[dtype, mx.float32],
    )


def kda_recurrence(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    beta: mx.array,
    a_log: mx.array,
    dt_bias: mx.array,
    lower_bound: float,
    state: mx.array,
    config=None,
) -> Tuple[mx.array, mx.array]:
    """Vector-gated delta rule over a prompt chunk (safe-gate variant).

    q, k, a: [B, T, H, 128]; v: [B, T, H, 128]; beta: [B, T, H] (sigmoid
    already applied); a_log: [H] fp32; dt_bias: [H * 128] fp32; state:
    [B, H, 128, 128] fp32. Returns y [B, T, H, 128] (q.dtype) and the fp32
    state, like ``gated_delta_update(..., lower_bound=lower_bound)``.

    ``config``: None (per-core on NAX hosts when covered, else blocked), a
    ``PerCoreConfig`` or a ``RecurrenceConfig`` (blocked kernel).
    """
    Dk, Dv = q.shape[-1], v.shape[-1]
    if Dk != 128 or Dv != 128:
        raise ValueError("kda_recurrence needs 128-wide heads")
    if isinstance(config, RecurrenceConfig):
        return _blocked(q, k, v, a, beta, a_log, dt_bias, lower_bound, state, config)
    if config is None and not _PERCORE_DEFAULT:
        return _blocked(q, k, v, a, beta, a_log, dt_bias, lower_bound, state, DEFAULT_CONFIG)
    out = _percore(
        q, k, v, a, beta, a_log, dt_bias, lower_bound, state, config or PerCoreConfig()
    )
    if out is None:
        out = _blocked(q, k, v, a, beta, a_log, dt_bias, lower_bound, state, DEFAULT_CONFIG)
    return out


_PERCORE_DEFAULT = is_nax_available()
if _PERCORE_DEFAULT:
    # Resolve the core count (~20 ms ioreg call) at import, i.e. at model
    # load, rather than inside the first prefill.
    gpu_core_count()
