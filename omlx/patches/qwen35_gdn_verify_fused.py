# SPDX-License-Identifier: Apache-2.0
#
# The recurrence keeps the thread layout of mlx-vlm's gated_delta kernel
# (MIT, https://github.com/Blaizzy/mlx-vlm).
"""Fused Qwen3.5 GDN verify recurrence with deferred state commit.

The stock speculative verify runs, per GDN layer, the gate/beta ops, a
recurrence kernel that also stores the state after every draft position,
the gated RMSNorm and its SiLU gate. On the 27B the stored states alone are
about 1 GB per verify.

This module runs the gates, both state passes and the recurrence in one
launch per layer, then the gated norm in a second one, and stores no
per-position state. The cache records the block's k/v/a/b rows instead. A commit keeps
``m`` positions by assigning a lazy replay of those rows to the state slot.
The next verify launch detects that pending replay and applies it in
registers before its own block, so the replay costs no extra launch or state
read. Any other reader of the slot evaluates the lazy replay normally.

The arithmetic matches the stock chain bit for bit: the per-row reduction
order of ``gated_delta_update``, the bf16 rounding points of ``compute_g``
and ``sigmoid``, MLX ``rms_norm`` and ``_precise_swiglu``. Compiled MLX graphs
use precise transcendentals while custom kernels default to fast math, so the
helpers call ``metal::precise`` explicitly.
"""

from __future__ import annotations

import logging

import mlx.core as mx

from . import qwen35_verify_qmm

logger = logging.getLogger(__name__)

_KERNELS: dict = {}
_PATCHED = False
_NORM_CLASS: type | None = None

_HELPERS = """
template <typename InT>
inline float gdn_decay(InT a, InT dt, float neg_a) {
    InT s = a + dt;
    InT zero = InT(0);
    InT hi = s > zero ? s : zero;
    InT lo = s > zero ? zero : s;
    InT sp;
    if (metal::isnan(s)) {
        sp = s;
    } else if (lo == -metal::numeric_limits<InT>::infinity()
               || hi == metal::numeric_limits<InT>::infinity()) {
        sp = hi;
    } else {
        InT e = static_cast<InT>(metal::precise::exp(static_cast<float>(lo - hi)));
        float xp1 = 1.0f + static_cast<float>(e);
        InT l1p;
        if (xp1 == metal::numeric_limits<float>::max()) {
            l1p = metal::numeric_limits<InT>::max();
        } else if (xp1 == 1.0f) {
            l1p = e;
        } else {
            l1p = static_cast<InT>(e * (metal::precise::log(xp1) / (xp1 - 1.0f)));
        }
        sp = hi + l1p;
    }
    return metal::precise::exp(neg_a * static_cast<float>(sp));
}

template <typename InT>
inline float gdn_beta(InT x) {
    InT ax = static_cast<InT>(metal::abs(static_cast<float>(x)));
    InT e = static_cast<InT>(metal::precise::exp(static_cast<float>(ax)));
    auto y = 1 / (1 + e);
    InT r = (x < 0) ? y : 1 - y;
    return static_cast<float>(r);
}
"""

_PROLOGUE = """
    constexpr int NK = Dk / 32;
    uint lane = thread_position_in_threadgroup.x;
    uint n = thread_position_in_grid.z;
    uint b_idx = n / Hv;
    uint hv = n % Hv;
    uint hk = hv / (Hv / Hk);
    uint dv = thread_position_in_grid.y;
    float st[NK];
    {
        auto s = state_in + (n * Dv + dv) * Dk + lane * NK;
        for (int i = 0; i < NK; ++i)
            st[i] = static_cast<float>(s[i]);
    }
    float neg_a = -metal::precise::exp(static_cast<float>(A_log[hv]));
    InT dtb = dt_bias[hv];
"""

_REPLAY = """
    {
        int keep = keep_rows[b_idx];
        float lane_g = 0.0f, lane_b = 0.0f;
        if (int(lane) < keep) {
            int gi = (b_idx * P + lane) * Hv + hv;
            lane_g = gdn_decay(pa[gi], dtb, neg_a);
            lane_b = gdn_beta(pb[gi]);
        }
        for (int t = 0; t < keep; ++t) {
            float gt = simd_shuffle(lane_g, ushort(t));
            float bt = simd_shuffle(lane_b, ushort(t));
            auto kp = pk + ((b_idx * P + t) * Hk + hk) * Dk + lane * NK;
            float kv = 0.0f;
            for (int i = 0; i < NK; ++i) {
                st[i] = st[i] * gt;
                kv += st[i] * kp[i];
            }
            kv = simd_sum(kv);
            float delta = (pv[((b_idx * P + t) * Hv + hv) * Dv + dv] - kv) * bt;
            for (int i = 0; i < NK; ++i)
                st[i] = st[i] + kp[i] * delta;
        }
        auto o = state_out + (n * Dv + dv) * Dk + lane * NK;
        for (int i = 0; i < NK; ++i)
            o[i] = st[i];
    }
"""

_MAIN = """
    float lane_g = 0.0f, lane_b = 0.0f;
    if (int(lane) < T) {
        int gi = (b_idx * T + lane) * Hv + hv;
        lane_g = gdn_decay(a[gi], dtb, neg_a);
        lane_b = gdn_beta(b[gi]);
    }
    for (int t = 0; t < T; ++t) {
        float gt = simd_shuffle(lane_g, ushort(t));
        float bt = simd_shuffle(lane_b, ushort(t));
        auto kp = k + ((b_idx * T + t) * Hk + hk) * Dk + lane * NK;
        auto qp = q + ((b_idx * T + t) * Hk + hk) * Dk + lane * NK;
        float kv = 0.0f;
        for (int i = 0; i < NK; ++i) {
            st[i] = st[i] * gt;
            kv += st[i] * kp[i];
        }
        kv = simd_sum(kv);
        float delta = (v[((b_idx * T + t) * Hv + hv) * Dv + dv] - kv) * bt;
        float acc = 0.0f;
        for (int i = 0; i < NK; ++i) {
            st[i] = st[i] + kp[i] * delta;
            acc += st[i] * qp[i];
        }
        acc = simd_sum(acc);
        if (lane == 0)
            y[((b_idx * T + t) * Hv + hv) * Dv + dv] = static_cast<InT>(acc);
    }
"""

# One simdgroup per (row, head) of 128 values: MLX rms_norm then the
# fp32 SiLU gate of ``_precise_swiglu``.
_NORM_GATE = """
    uint lane = thread_position_in_threadgroup.x;
    uint row = thread_position_in_grid.y;
    auto yp = y + row * 128 + lane * 4;
    float x[4];
    float ss = 0.0f;
    for (int i = 0; i < 4; ++i) {
        x[i] = yp[i];
        ss += x[i] * x[i];
    }
    ss = simd_sum(ss);
    float inv = metal::precise::rsqrt(ss / 128.0f + EPS);
    auto zp = z + row * 128 + lane * 4;
    auto op = out + row * 128 + lane * 4;
    float part = 0.0f;
    for (int i = 0; i < 4; ++i) {
        InT normed = norm_w[lane * 4 + i] * static_cast<InT>(x[i] * inv);
        float g = static_cast<float>(zp[i]);
        float sy = 1 / (1 + metal::precise::exp(metal::abs(g)));
        float sig = (g < 0) ? sy : 1 - sy;
        InT o = static_cast<InT>((g * sig) * static_cast<float>(normed));
        op[i] = o;
        part += float(o);
    }
    // Per-64 sums of the output feed the out projection.
    for (int off = 1; off < 16; off <<= 1)
        part += simd_shuffle_xor(part, ushort(off));
    if ((lane & 15) == 0)
        xs[row * 2 + lane / 16] = part;
"""


def _kernel(main: bool, replay: bool):
    key = (main, replay)
    kernel = _KERNELS.get(key)
    if kernel is not None:
        return kernel
    inputs = ["state_in", "A_log", "dt_bias"]
    outputs = []
    body = _PROLOGUE
    if replay:
        inputs += ["pk", "pv", "pa", "pb", "keep_rows"]
        outputs.append("state_out")
        body += _REPLAY
    if main:
        inputs += ["q", "k", "v", "a", "b"]
        outputs.insert(0, "y")
        body += _MAIN
    name = "omlx_gdn_verify" + ("_main" if main else "") + ("_replay" if replay else "")
    kernel = mx.fast.metal_kernel(
        name=name,
        input_names=inputs,
        output_names=outputs,
        source=body,
        header=_HELPERS,
    )
    _KERNELS[key] = kernel
    return kernel


def _norm_gate_kernel(eps: float):
    key = ("norm_gate", float(eps))
    kernel = _KERNELS.get(key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name="omlx_gdn_norm_gate_eps" + f"{eps:.0e}".replace("-", "m"),
            input_names=["y", "z", "norm_w"],
            output_names=["out", "xs"],
            source=_NORM_GATE.replace("EPS", f"{float(eps)!r}f"),
        )
        _KERNELS[key] = kernel
    return kernel


def _geometry(layer, batch):
    return dict(
        Hk=layer.num_k_heads,
        Hv=layer.num_v_heads,
        Dk=layer.head_k_dim,
        Dv=layer.head_v_dim,
        grid=(32, layer.head_v_dim, batch * layer.num_v_heads),
    )


def fused_eligible(layer, q, cache, length) -> bool:
    """One block covering the whole speculative window, stock norm, bf16."""
    transaction = getattr(cache, "_speculation", None)
    state = cache[1]
    return (
        _PATCHED
        and transaction is not None
        and transaction["length"] == length
        and 1 not in transaction["records"]
        and isinstance(layer.norm, _NORM_CLASS)
        and q.dtype == mx.bfloat16
        and layer.head_k_dim == 128
        and layer.head_v_dim == 128
        and layer.num_v_heads % layer.num_k_heads == 0
        and layer.A_log.dtype == mx.bfloat16
        and layer.dt_bias.dtype == mx.bfloat16
        and layer.norm.weight.dtype == mx.bfloat16
        and (state is None or state.dtype == mx.float32)
        and length <= 32
    )


def replay_state(layer, base, rows, keep):
    """State after the first ``keep[b]`` recorded rows of each batch row."""
    pk, pv, pa, pb = rows
    geo = _geometry(layer, base.shape[0])
    (state,) = _kernel(False, True)(
        inputs=[base, layer.A_log, layer.dt_bias, pk, pv, pa, pb, keep],
        template=[
            ("InT", mx.bfloat16),
            ("Hk", geo["Hk"]),
            ("Hv", geo["Hv"]),
            ("Dk", geo["Dk"]),
            ("Dv", geo["Dv"]),
            ("P", pk.shape[1]),
        ],
        grid=geo["grid"],
        threadgroup=(32, 4, 1),
        output_shapes=[base.shape],
        output_dtypes=[mx.float32],
    )
    return state


def verify_block(layer, cache, q, k, v, a, b, z):
    """Gated RMSNorm output of one verify block; records the block for commit.

    Returns ``(B, T, Hv * Dv)`` with its per-64 sums registered for the out
    projection. ``cache[1]`` becomes the lazy state after
    the whole block, and the speculative record keeps the block rows.
    """
    batch, length = q.shape[:2]
    geo = _geometry(layer, batch)
    state = cache[1]
    if state is None:
        state = mx.zeros((batch, geo["Hv"], geo["Dv"], geo["Dk"]), dtype=mx.float32)
    pending = getattr(cache, "_omlx_gdn_pending", None)
    replay = pending is not None and pending[0] is state
    a = a.reshape(batch, length, geo["Hv"])
    b = b.reshape(batch, length, geo["Hv"])
    z = z.reshape(batch, length, geo["Hv"], geo["Dv"])
    inputs = [state, layer.A_log, layer.dt_bias]
    template = [
        ("InT", mx.bfloat16),
        ("Hk", geo["Hk"]),
        ("Hv", geo["Hv"]),
        ("Dk", geo["Dk"]),
        ("Dv", geo["Dv"]),
        ("T", length),
    ]
    output_shapes = [(batch, length, geo["Hv"], geo["Dv"])]
    output_dtypes = [mx.bfloat16]
    if replay:
        _, base, rows, keep = pending
        inputs = [base, layer.A_log, layer.dt_bias, *rows, keep]
        template.append(("P", rows[0].shape[1]))
        output_shapes.append(base.shape)
        output_dtypes.append(mx.float32)
    inputs += [q, k, v, a, b]
    outs = _kernel(True, replay)(
        inputs=inputs,
        template=template,
        grid=geo["grid"],
        threadgroup=(32, 4, 1),
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
    )
    rows_total = batch * length * geo["Hv"]
    width = geo["Hv"] * geo["Dv"]
    out, sums = _norm_gate_kernel(layer.norm.eps)(
        inputs=[outs[0], z, layer.norm.weight],
        template=[("InT", mx.bfloat16)],
        grid=(32, rows_total, 1),
        threadgroup=(32, 8, 1),
        output_shapes=[(batch, length, width), (batch * length, width // 64)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )
    qwen35_verify_qmm.register_group_sums(out, sums)
    start = outs[1] if replay else state
    rows = (k, v, a, b)
    record_block(cache, 1, layer, start, rows, length)
    return out


def record_block(cache, index, layer, start, rows, length):
    """Record a replayable block; the slot holds its lazy final state."""
    full = mx.full((start.shape[0],), length, dtype=mx.int32)
    final = replay_state(layer, start, rows, full)
    transaction = cache._speculation
    transaction["records"][int(index)] = ("replay", start, rows, int(length), layer)
    cache[index] = final
    cache._omlx_gdn_pending = (final, start, rows, full)


def record_window(cache, index, previous, rows, width):
    """Record a causal window as its previous state plus the block rows.

    The commit slices the kept window from ``rows`` when it lies inside the
    block, so no concatenation runs per verify.
    """
    transaction = cache._speculation
    transaction["records"][int(index)] = ("window_pair", previous, rows, int(width))


def _commit_window(cache, index, record, lengths):
    _, previous, rows, width = record
    if len(set(lengths)) == 1:
        keep = lengths[0]
        if keep == 0:
            state = previous
        elif keep >= width:
            state = rows[:, keep - width : keep]
        else:
            state = mx.concatenate([previous[:, keep:], rows[:, :keep]], axis=1)
    else:
        source = mx.concatenate([previous, rows], axis=1)
        positions = mx.array(lengths, dtype=mx.int32)[:, None] + mx.arange(width)
        positions = positions.reshape(*positions.shape, *([1] * (source.ndim - 2)))
        state = mx.take_along_axis(source, positions, axis=1)
    cache[index] = state


def _commit_replay(cache, index, record, lengths):
    _, start, rows, _length, layer = record
    if all(value == 0 for value in lengths):
        cache[index] = start
        return
    keep = mx.array(lengths, dtype=mx.int32)
    state = replay_state(layer, start, rows, keep)
    cache[index] = state
    cache._omlx_gdn_pending = (state, start, rows, keep)


def apply_arrays_cache_replay_patch() -> bool:
    """Teach ArraysCache transactions the ``replay`` and ``window_pair`` kinds."""
    global _PATCHED, _NORM_CLASS
    if _PATCHED:
        return True
    from mlx_vlm.models.cache import ArraysCache
    from mlx_vlm.models.qwen3_5.language import Qwen3_5RMSNormGated

    orig_recorded = ArraysCache._recorded_length
    orig_commit = ArraysCache.commit_speculation

    def _recorded_length(self, index):
        record = self._speculation["records"].get(index)
        if record is not None and record[0] == "replay":
            return record[3]
        if record is not None and record[0] == "window_pair":
            return record[2].shape[1]
        return orig_recorded(self, index)

    def commit_speculation(self, lengths, generation=None):
        transaction = self._speculation
        custom = {}
        if transaction is not None:
            custom = {
                index: record
                for index, record in transaction["records"].items()
                if record[0] in ("replay", "window_pair")
            }
        if not custom:
            return orig_commit(self, lengths, generation)
        _, resolved = self.validate_speculation(lengths, generation)
        # The stock commit then sees these slots as untouched.
        for index in custom:
            del transaction["records"][index]
            self.cache[index] = transaction["initial_state"][index]
        orig_commit(self, lengths, generation)
        for index, record in custom.items():
            if record[0] == "replay":
                _commit_replay(self, index, record, resolved)
            else:
                _commit_window(self, index, record, resolved)

    ArraysCache._recorded_length = _recorded_length
    ArraysCache.commit_speculation = commit_speculation
    _NORM_CLASS = Qwen3_5RMSNormGated
    _PATCHED = True
    logger.info("GDN verify replay commit patch applied")
    return True
