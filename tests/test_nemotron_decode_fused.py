"""Tests for the Nemotron-H fused decode-step kernels.

Reference is the stock mlx_lm NemotronHMamba2Mixer decode path (sequential
per-position calls — exactly how decode and the MTP chain verify run it)
with identical weights. bfloat16 output checks are anchored to an fp32
ground truth: the fused path must track it as closely as the stock bf16
path does, which separates real defects from accumulated-rounding noise.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
nh = pytest.importorskip("mlx_lm.models.nemotron_h")

from mlx_lm.models.cache import ArraysCache  # noqa: E402

from omlx.patches.nemotron_decode_fused import (  # noqa: E402
    apply_nemotron_decode_fused_patch,
    mamba2_decode_step,
    supported,
)

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="Metal is not available"
)

# The apply function patches the Mixer class in place, so later tests see it.
# Capture the pre-fused __call__ now: reference paths must never route
# through the kernels they are validating. (If the mlx_lm_mtp patches were
# applied by an earlier test file, this is their wrap — which is exactly the
# stock decode behavior for plain calls.)
_STOCK_CALL = nh.NemotronHMamba2Mixer.__call__
_STOCK_ATTN_CALL = nh.NemotronHAttention.__call__

# Small but kernel-eligible dims: Dh 64, Ds 128, norm group 512 (=2048/4).
DIMS = dict(
    mamba_num_heads=32,
    hidden_size=1024,
    ssm_state_size=128,
    conv_kernel=4,
    n_groups=4,
    mamba_head_dim=64,
    time_step_limit=(0.0, float("inf")),
    layer_norm_epsilon=1e-5,
    use_conv_bias=True,
    mamba_proj_bias=False,
)


def make_mixer(dtype, seed=17, **overrides):
    args = SimpleNamespace(**{**DIMS, **overrides})
    mx.random.seed(seed)
    mixer = nh.NemotronHMamba2Mixer(args)
    scale = 0.05
    mixer.in_proj.weight = mx.random.normal(mixer.in_proj.weight.shape) * scale
    mixer.out_proj.weight = mx.random.normal(mixer.out_proj.weight.shape) * scale
    mixer.conv1d.weight = mx.random.normal(mixer.conv1d.weight.shape) * 0.3
    mixer.conv1d.bias = mx.random.normal(mixer.conv1d.bias.shape) * 0.1
    mixer.norm.weight = mx.random.normal(mixer.norm.weight.shape) * 0.2 + 1.0
    mixer.dt_bias = mx.random.normal((args.mamba_num_heads,)) * 0.5
    mixer.D = mx.random.normal((args.mamba_num_heads,)) * 0.5 + 1.0
    mixer.set_dtype(dtype)
    mixer.A_log = mx.log(mx.arange(1, args.mamba_num_heads + 1, dtype=mx.float32) * 0.5)
    mixer.dt_bias = mixer.dt_bias.astype(mx.float32)
    return mixer, args


def stock_decode(mixer, hs, cache):
    outs = []
    for s in range(hs.shape[1]):
        outs.append(_STOCK_CALL(mixer, hs[:, s : s + 1, :], None, cache))
    return mx.concatenate(outs, axis=1)


def fused_decode(mixer, args, hs, cache, capture=False):
    proj = mixer.in_proj(hs)
    n = hs.shape[0]
    if cache[0] is None:
        cache[0] = mx.zeros((n, 3, mixer.conv_dim), dtype=hs.dtype)
        cache[1] = mx.zeros(
            (n, mixer.num_heads, mixer.head_dim, mixer.ssm_state_size),
            dtype=mx.float32,
        )
    y, conv_out, ssm_out, cap_s, cap_c = mamba2_decode_step(
        proj,
        cache[0],
        cache[1],
        mixer.conv1d.weight,
        mixer.conv1d.bias,
        mixer.A_log,
        mixer.dt_bias,
        mixer.D.astype(hs.dtype),
        mixer.norm.weight,
        num_heads=mixer.num_heads,
        head_dim=mixer.head_dim,
        state_size=mixer.ssm_state_size,
        n_groups=mixer.n_groups,
        eps=1e-5,
        group_size=mixer.norm.group_size,
        time_step_limit=args.time_step_limit,
        capture=capture,
    )
    cache[0], cache[1] = conv_out, ssm_out
    cache.advance(hs.shape[1])
    return mixer.out_proj(y), cap_s, cap_c


def rel(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return (mx.abs(a - b).max() / (mx.abs(b).max() + 1e-8)).item()


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("S", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("warm", [False, True])
def test_matches_stock_decode(dtype, S, warm):
    assert supported(32, 64, 128, 4, 4, 512)
    mixer, args = make_mixer(dtype)
    mixer32, args32 = make_mixer(mx.float32)
    mx.random.seed(23 + S)
    hs32 = mx.random.normal((1, S, 1024)) * 0.5
    hs = hs32.astype(dtype)

    c_ref, c_fus, c_32 = (ArraysCache(size=2) for _ in range(3))
    if warm:
        warm32 = mx.random.normal((1, 6, 1024)) * 0.5
        _ = stock_decode(mixer, warm32.astype(dtype), c_ref)
        _ = stock_decode(mixer, warm32.astype(dtype), c_fus)
        _ = stock_decode(mixer32, warm32, c_32)
        mx.eval(c_ref[1], c_fus[1], c_32[1])

    ref = stock_decode(mixer, hs, c_ref)
    truth = stock_decode(mixer32, hs32, c_32)
    fus, _, _ = fused_decode(mixer, args, hs, c_fus)
    mx.eval(ref, truth, fus)

    e_stock = rel(ref, truth)
    e_fused = rel(fus, truth)
    assert e_fused < max(1.5 * e_stock, 5e-3), (e_fused, e_stock)
    tol = 2e-2 if dtype == mx.bfloat16 else 6e-3
    assert rel(c_fus[0], c_ref[0]) < tol
    assert rel(c_fus[1], c_ref[1]) < tol


def test_per_position_capture_matches_prefix_replay():
    dtype = mx.bfloat16
    S = 4
    mixer, args = make_mixer(dtype)
    mx.random.seed(31)
    hs = (mx.random.normal((1, S, 1024)) * 0.5).astype(dtype)
    warm = (mx.random.normal((1, 6, 1024)) * 0.5).astype(dtype)

    c_fus = ArraysCache(size=2)
    _ = stock_decode(mixer, warm, c_fus)
    _, cap_s, cap_c = fused_decode(mixer, args, hs, c_fus, capture=True)

    for keep in (1, 2, 3, 4):
        c_replay = ArraysCache(size=2)
        _ = stock_decode(mixer, warm, c_replay)
        _ = stock_decode(mixer, hs[:, :keep], c_replay)
        mx.eval(c_replay[1])
        assert rel(cap_s[keep - 1].reshape(c_replay[1].shape), c_replay[1]) < 2e-2
        assert rel(cap_c[keep - 1], c_replay[0]) < 2e-2


def test_mixer_wrap_verify_window_and_fall_through():
    from omlx.patches.mlx_lm_mtp import nemotron_h_chain, nemotron_h_model

    assert nemotron_h_model.apply()
    assert nemotron_h_chain.apply()
    assert apply_nemotron_decode_fused_patch()
    assert apply_nemotron_decode_fused_patch()  # idempotent

    dtype = mx.bfloat16
    mixer, args = make_mixer(dtype)
    mx.random.seed(41)
    warm = (mx.random.normal((1, 6, 1024)) * 0.5).astype(dtype)
    hs = (mx.random.normal((1, 4, 1024)) * 0.5).astype(dtype)

    cache = ArraysCache(size=2)
    for s in range(6):
        _ = mixer(warm[:, s : s + 1, :], None, cache)
    pre_conv, pre_ssm = cache[0], cache[1]

    out = mixer(hs, None, cache, n_confirmed=2)
    mx.eval(out, cache[0], cache[1])
    pos = cache._mtp_pos_states
    assert pos is not None and len(pos) == 4
    assert cache.rollback_state[0] is pre_conv
    assert cache.rollback_state[1] is pre_ssm
    assert cache._mtp_draft_stash is hs
    for conv_m, ssm_m in pos:
        assert conv_m.shape == pre_conv.shape
        assert ssm_m.shape == pre_ssm.shape
        assert ssm_m.dtype == pre_ssm.dtype

    # rollback to pos[0], continue decoding — must match a stock replay
    mixer_ref, _ = make_mixer(dtype)
    c_ref = ArraysCache(size=2)
    for s in range(6):
        _ = mixer_ref(warm[:, s : s + 1, :], None, c_ref)
    _ = mixer_ref(hs[:, :1], None, c_ref)
    cache[0], cache[1] = pos[0]
    nxt = (mx.random.normal((1, 1, 1024)) * 0.5).astype(dtype)
    o_fus = mixer(nxt, None, cache)
    o_ref = mixer_ref(nxt, None, c_ref)
    mx.eval(o_fus, o_ref)
    assert rel(o_fus, o_ref) < 2e-2

    # large window falls through to the chain's sequential path
    c_big = ArraysCache(size=2)
    for s in range(6):
        _ = mixer(warm[:, s : s + 1, :], None, c_big)
    hs_big = (mx.random.normal((1, 12, 1024)) * 0.5).astype(dtype)
    o_big = mixer(hs_big, None, c_big, n_confirmed=3)
    mx.eval(o_big, c_big[0], c_big[1])
    assert c_big._mtp_pos_states is not None and len(c_big._mtp_pos_states) == 12


def test_unsupported_dims_fall_through():
    # Dh=16 is outside the kernel's shape set; the wrap must route to stock.
    assert not supported(4, 16, 32, 2, 4, 32)
    assert apply_nemotron_decode_fused_patch()
    dtype = mx.bfloat16
    mixer, args = make_mixer(
        dtype,
        mamba_num_heads=4,
        mamba_head_dim=16,
        ssm_state_size=32,
        n_groups=2,
        hidden_size=1024,
    )
    cache = ArraysCache(size=2)
    hs = (mx.random.normal((1, 1, 1024)) * 0.5).astype(dtype)
    out = mixer(hs, None, cache)
    mx.eval(out, cache[0], cache[1])
    assert out.shape == (1, 1, 1024)


def test_attention_verify_per_position_sdpa():
    """3 <= S <= 8 attention routes to per-position SDPA (exact causal
    semantics via KV slicing) and matches the stock fused path; S=2 and
    S=12 fall through byte-identically."""
    from mlx_lm.models.cache import KVCache

    from omlx.patches.nemotron_decode_fused import (
        apply_nemotron_attention_verify_patch,
    )

    stock_call = _STOCK_ATTN_CALL
    assert apply_nemotron_attention_verify_patch()
    assert apply_nemotron_attention_verify_patch()  # idempotent

    args = SimpleNamespace(
        hidden_size=1024,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=128,
        attention_bias=False,
    )
    mx.random.seed(21)
    attn = nh.NemotronHAttention(args)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        lin = getattr(attn, name)
        lin.weight = mx.random.normal(lin.weight.shape) * 0.05
    attn.set_dtype(mx.bfloat16)

    def warm_kv(ctx, seed):
        mx.random.seed(seed)
        c = KVCache()
        k = (mx.random.normal((1, 2, ctx, 128)) * 0.3).astype(mx.bfloat16)
        v = (mx.random.normal((1, 2, ctx, 128)) * 0.3).astype(mx.bfloat16)
        c.update_and_fetch(k, v)
        mx.eval(c.keys, c.values)
        return c

    for S, exact in ((2, True), (3, False), (4, False), (8, False), (12, True)):
        mx.random.seed(50 + S)
        x = (mx.random.normal((1, S, 1024)) * 0.1).astype(mx.bfloat16)
        c_ref = warm_kv(512, seed=99)
        c_new = warm_kv(512, seed=99)
        ref = stock_call(attn, x, "causal", c_ref)
        out = attn(x, "causal", c_new)
        mx.eval(ref, out)
        err = rel(out, ref)
        assert err < (1e-9 if exact else 1e-4), (S, err)
        assert (
            rel(c_new.keys[:, :, : c_new.offset], c_ref.keys[:, :, : c_ref.offset])
            == 0.0
        )
