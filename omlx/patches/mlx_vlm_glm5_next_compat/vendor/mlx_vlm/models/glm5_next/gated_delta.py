from functools import partial
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@partial(mx.compile, shapeless=True)
def compute_g(A_log, a, dt_bias):
    return mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias))


@partial(mx.compile, shapeless=True)
def compute_g_safe(A_log, a, dt_bias, lower_bound):
    return mx.exp(
        lower_bound * mx.sigmoid(mx.exp(A_log.astype(mx.float32)) * (a + dt_bias))
    )


def _make_gated_delta_kernel(has_mask=False, vectorized=False):
    if not mx.metal.is_available():
        return None
    mask_source = "mask[b_idx * T + t]" if has_mask else "true"

    # Configure g indexing based on whether gating is vectorized
    if vectorized:
        g_comment = "// g: [B, T, Hv, Dk]"
        g_setup = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
        g_hoist = """
            for (int i = 0; i < n_per_t; ++i) {
              gt[i] = static_cast<float>(g_[n_per_t * dk_idx + i]);
            }"""
        g_decay = "state[r][i] * gt[i]"
        g_advance = "g_ += Hv * Dk;"
    else:
        g_comment = "// g: [B, T, Hv]"
        g_setup = "auto g_ = g + b_idx * T * Hv;"
        g_hoist = "float gg = static_cast<float>(g_[hv_idx]);"
        g_decay = "state[r][i] * gg"
        g_advance = "g_ += Hv;"

    # One thread owns R consecutive value rows. GLM's vector gate is shared by
    # those rows, as are q, k, and beta, so row blocking removes redundant
    # loads while preserving the per-row arithmetic and reduction order.
    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv0 = thread_position_in_grid.y * R;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv0) * Dk + n_per_t * dk_idx;
        auto o_state = state_out + (n * Dv + dv0) * Dk + n_per_t * dk_idx;

        float state[R][n_per_t];
        for (int r = 0; r < R; ++r) {{
          for (int i = 0; i < n_per_t; ++i) {{
            state[r][i] = static_cast<float>(i_state[r * Dk + i]);
          }}
        }}

        {g_comment}
        {g_setup}
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {{
          if ({mask_source}) {{
            // q, k, vector-g, and beta are identical for all R value rows.
            float kk[n_per_t];
            float qq[n_per_t];
            {"float gt[n_per_t];" if vectorized else ""}
            for (int i = 0; i < n_per_t; ++i) {{
              kk[i] = static_cast<float>(k_[n_per_t * dk_idx + i]);
              qq[i] = static_cast<float>(q_[n_per_t * dk_idx + i]);
            }}
            {g_hoist}
            float bb = static_cast<float>(beta_[hv_idx]);

            for (int r = 0; r < R; ++r) {{
              float kv_mem = 0.0f;
              for (int i = 0; i < n_per_t; ++i) {{
                state[r][i] = {g_decay};
                kv_mem += state[r][i] * kk[i];
              }}
              kv_mem = simd_sum(kv_mem);

              auto delta = (static_cast<float>(v_[dv0 + r]) - kv_mem) * bb;

              float out = 0.0f;
              for (int i = 0; i < n_per_t; ++i) {{
                state[r][i] = state[r][i] + kk[i] * delta;
                out += state[r][i] * qq[i];
              }}
              out = simd_sum(out);
              if (thread_index_in_simdgroup == 0) {{
                y[dv0 + r] = static_cast<InT>(out);
              }}
            }}
          }} else {{
            for (int r = 0; r < R; ++r) {{
              y[dv0 + r] = static_cast<InT>(0);
            }}
          }}
          // Increment data pointers to next time step
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          {g_advance}
          beta_ += Hv;
        }}
        for (int r = 0; r < R; ++r) {{
          for (int i = 0; i < n_per_t; ++i) {{
            o_state[r * Dk + i] = static_cast<StT>(state[r][i]);
          }}
        }}
    """
    inputs = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if has_mask:
        inputs.append("mask")

    suffix = ""
    if vectorized:
        suffix += "_vec"
    if has_mask:
        suffix += "_mask"

    return mx.fast.metal_kernel(
        name=f"gated_delta_step{suffix}",
        input_names=inputs,
        output_names=["y", "state_out"],
        source=source,
    )


_gated_delta_kernel = _make_gated_delta_kernel(has_mask=False, vectorized=False)
_gated_delta_kernel_masked = _make_gated_delta_kernel(has_mask=True, vectorized=False)
_gated_delta_kernel_vec = _make_gated_delta_kernel(has_mask=False, vectorized=True)
_gated_delta_kernel_vec_masked = _make_gated_delta_kernel(
    has_mask=True, vectorized=True
)


@mx.compile
def _gated_delta_step_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Ops-based reference implementation for a single recurrent step.

    Shapes:
      - q, k: [B, H, Dk]
      - v: [B, H, Dv]
      - g: [B, H] or [B, H, Dk]
      - beta: [B, H]
      - state: [B, H, Dv, Dk]
    Returns:
      - y: [B, H, Dv]
      - new_state: [B, H, Dv, Dk]
    """

    # Decay
    old_state = state
    if g.ndim == 2:
        decay = g[..., None, None]
    elif g.ndim == 3:
        decay = g[..., None, :]
    else:
        raise ValueError(f"Unsupported gating shape {g.shape}")
    state = state * decay
    kv_mem = (state * k[..., None, :]).sum(axis=-1)  # [B, H, Dv]
    delta = (v - kv_mem) * beta[..., None]  # [B, H, Dv]
    state = state + k[..., None, :] * delta[..., None]
    # Output projection along key dim with q
    y = (state * q[..., None, :]).sum(axis=-1)  # [B, H, Dv]

    if mask is not None:
        state_mask = mx.expand_dims(mask, axis=(1, 2, 3))
        output_mask = mx.expand_dims(mask, axis=(1, 2))
        state = mx.where(state_mask, state, old_state)
        y = mx.where(output_mask, y, 0)
    return y.astype(q.dtype), state


def _gated_delta_kernel_rows(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
    *,
    rows_per_thread: int,
    threadgroup_y: int,
) -> Tuple[mx.array, mx.array]:
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    input_type = q.dtype
    state_type = state.dtype
    if g.ndim == 4:
        kernel = _gated_delta_kernel_vec
        inputs = [q, k, v, g, beta, state, T]
        if mask is not None:
            kernel = _gated_delta_kernel_vec_masked
            inputs.append(mask)
    else:
        kernel = _gated_delta_kernel
        inputs = [q, k, v, g, beta, state, T]
        if mask is not None:
            kernel = _gated_delta_kernel_masked
            inputs.append(mask)

    if rows_per_thread < 1 or Dv % rows_per_thread != 0:
        raise ValueError(
            f"rows_per_thread={rows_per_thread} must divide value width {Dv}"
        )

    return kernel(
        inputs=inputs,
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("R", rows_per_thread),
        ],
        grid=(32, Dv // rows_per_thread, B * Hv),
        threadgroup=(32, threadgroup_y, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[input_type, state_type],
    )


def gated_delta_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    Dv = v.shape[-1]
    rows_per_thread = 4 if Dv % 4 == 0 else (2 if Dv % 2 == 0 else 1)
    threadgroup_y = 2 if (Dv // rows_per_thread) % 2 == 0 else 1
    return _gated_delta_kernel_rows(
        q,
        k,
        v,
        g,
        beta,
        state,
        mask,
        rows_per_thread=rows_per_thread,
        threadgroup_y=threadgroup_y,
    )


def gated_delta_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Ops-based reference implementation for prompt prefill (sequential loop).
    Supports both scalar and vectorized gating.

    Shapes:
      - q, k: [B, T, Hk, Dk]
      - v: [B, T, Hv, Dv]
      - g: [B, T, Hv] (scalar) or [B, T, Hv, Dk] (vectorized)
      - beta: [B, T, Hv]
      - state: [B, Hv, Dv, Dk]
    Returns:
      - y: [B, T, Hv, Dv]
      - state: [B, Hv, Dv, Dk]
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    if (repeat_factor := Hv // Hk) > 1:
        q = mx.repeat(q, repeat_factor, -2)
        k = mx.repeat(k, repeat_factor, -2)

    ys = []
    for t in range(T):
        y, state = _gated_delta_step_ops(
            q[:, t],
            k[:, t],
            v[:, t],
            g[:, t],
            beta[:, t],
            state,
            None if mask is None else mask[:, t],
        )
        ys.append(y)
    y = mx.stack(ys, axis=1)
    return y, state


def gated_delta_update(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
    use_kernel: bool = True,
    lower_bound: Optional[float] = None,
) -> Tuple[mx.array, mx.array]:
    beta = mx.sigmoid(b)
    if lower_bound is None:
        g = compute_g(A_log, a, dt_bias)
    else:
        g = compute_g_safe(A_log, a, dt_bias, lower_bound)
    if state is None:
        B, _, Hk, Dk = q.shape
        Hv, Dv = v.shape[-2:]
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    if (
        not use_kernel
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or k.shape[-1] < 32
        or k.shape[-1] % 32 != 0
    ):
        return gated_delta_ops(q, k, v, g, beta, state, mask)
    return gated_delta_kernel(q, k, v, g, beta, state, mask)
