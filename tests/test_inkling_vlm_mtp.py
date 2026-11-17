# SPDX-License-Identifier: Apache-2.0
"""Inkling Lightning MTP runtime tests (uniform-window multi-block cycle)."""

from __future__ import annotations

import numpy as np
import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


@pytest.fixture()
def strict_math_device():
    """Use deterministic CPU reductions for chunk-shape parity assertions."""
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


@pytest.fixture(scope="module")
def runtime():
    from omlx.patches.mlx_lm_mtp import set_mtp_active, set_mtp_depth
    from omlx.patches.mlx_vlm_mtp import inkling_vlm_runtime

    assert inkling_vlm_runtime.apply()
    set_mtp_active(True)
    set_mtp_depth(4)
    yield inkling_vlm_runtime
    set_mtp_active(False)


def _mtp_language_model():
    import importlib

    from tests.test_mlx_vlm_inkling_compat import _tiny_text_config

    language = importlib.import_module("mlx_vlm.models.inkling.language")
    config = _tiny_text_config()
    config.mtp_num_hidden_layers = 3
    config.mtp_local_layer_ids = [0, 2]
    mx.random.seed(21)
    model = language.LanguageModel(config)
    mx.eval(model.parameters())
    return model


def _hidden_table(n, hidden=32, seed=7):
    mx.random.seed(seed)
    return mx.random.normal((1, n, hidden)) * 0.1


def _run_cycles(model, cache, toks, table, script):
    """Drive the generator contract: per cycle fold m+1 committed pairs,
    then depth-1 chain calls. ``script`` entries are (m_accepted, depth)
    where the m accepted drafts of a cycle equal the next cycle's first m
    committed tokens (fed to the chain as matching drafts)."""
    f = 0
    prev_depth = None
    for i, (m, depth) in enumerate(script):
        n = m + 1
        assert prev_depth is None or m <= prev_depth, "script accepts > drafts"
        assert f + n <= len(toks)
        hid = table[:, f : f + n]
        ids = mx.array([toks[f : f + n]], dtype=mx.uint32)
        model.mtp_begin_cycle(cache, depth)
        model.mtp_forward(hid, ids, cache, return_hidden=True, logits_keep=1)
        f += n
        next_m = script[i + 1][0] if i + 1 < len(script) else 0
        for j in range(1, depth):
            # Chain call j feeds draft j-1. Drafts the next cycle accepts
            # must equal the tokens it then commits; the rest are wrong.
            true_idx = f + j - 1
            if j - 1 < next_m and true_idx < len(toks):
                tok = toks[true_idx]
            else:
                tok = (toks[true_idx] + 63) % 128 if true_idx < len(toks) else 1
            draft = mx.array([[tok]], dtype=mx.uint32)
            model.mtp_forward(cache.win_hid[:, -1:], draft, cache, return_hidden=True)
        prev_depth = depth
    return f


def _reference_blocks(model, toks, table, f, depth):
    """One-shot chained fold over the full committed history: pass j
    covers slots [0, f-1-j] with the token stream shifted by j."""
    from mlx_lm.models.cache import ArraysCache, CacheList, KVCache

    caches = [CacheList(KVCache(), ArraysCache(4)) for _ in model.mtp.blocks]
    win_hid = table[:, :f]
    for j in range(depth):
        cols = f - j
        ids = mx.array([toks[j:f]], dtype=mx.uint32)
        win_hid = model._mtp_run_block(j, caches[j], win_hid[:, :cols], ids)
    return caches


def test_config_plumb_and_attach(runtime):
    import importlib

    inkling_pkg = importlib.import_module("mlx_vlm.models.inkling")
    config = inkling_pkg.ModelConfig.from_dict(
        {
            "model_type": "inkling_mm_model",
            "text_config": {"hidden_size": 32, "num_hidden_layers": 2},
            "mtp_config": {
                "num_nextn_predict_layers": 8,
                "local_layer_ids": [0, 2, 4, 5, 6, 7],
                "chain_hidden_post_norm": False,
            },
        }
    )
    assert config.text_config.mtp_num_hidden_layers == 8
    assert config.text_config.mtp_local_layer_ids == [0, 2, 4, 5, 6, 7]

    model = _mtp_language_model()
    assert hasattr(model, "mtp")
    assert len(model.mtp.blocks) == 3
    assert model._omlx_mtp_decode_enabled
    assert model._omlx_mtp_chain
    assert model._omlx_mtp_head_prenorm
    # No per-cycle clone and no row-wise batch path: provisional rows live
    # on the persistent caches and the next fold trims them.
    assert model._omlx_mtp_head_clone is False
    assert model._omlx_mtp_rowwise_unsupported is True
    assert model._omlx_mtp_depth == 3  # clamped to the shipped block count
    assert all(
        hasattr(block.transformer_block.self_attn, "qkvr_proj")
        for block in model.mtp.blocks
    )


def test_cycle_routing_uses_block_j(runtime):
    """Chain call j must run block j (vLLM draft-j <- block-j mapping),
    with provisional rows appended to the persistent caches."""
    model = _mtp_language_model()
    cache = model.make_mtp_cache()
    assert len(cache) == 3

    hid = _hidden_table(1)
    model.mtp_begin_cycle(cache, 3)
    logits, _ = model.mtp_forward(
        hid, mx.array([[5]], dtype=mx.uint32), cache, return_hidden=True, logits_keep=1
    )
    assert logits.shape == (1, 1, 128)
    # Fold = pass 0 on block 0 only.
    assert [cl[0].offset for cl in cache] == [1, 0, 0]

    model.mtp_forward(
        cache.win_hid[:, -1:], mx.array([[7]], dtype=mx.uint32), cache, return_hidden=True
    )
    assert [cl[0].offset for cl in cache] == [1, 1, 0], "chain 1 must run block 1"
    model.mtp_forward(
        cache.win_hid[:, -1:], mx.array([[9]], dtype=mx.uint32), cache, return_hidden=True
    )
    assert [cl[0].offset for cl in cache] == [1, 1, 1], "chain 2 must run block 2"
    assert cache.frontier == 1

    # Next fold trims every provisional row back to the uniform window
    # start and refolds block 0 over it; deep blocks refold during their
    # own chain passes.
    model.mtp_begin_cycle(cache, 3)
    model.mtp_forward(
        hid, mx.array([[6]], dtype=mx.uint32), cache, return_hidden=True, logits_keep=1
    )
    assert cache.frontier == 2
    assert [cl[0].offset for cl in cache] == [2, 0, 0]


def test_committed_prefix_matches_oneshot_oracle(runtime):
    """After a mixed accept/reject cycle script, every block's committed
    rows must equal a from-scratch chained fold over the full history."""
    model = _mtp_language_model()
    cache = model.make_mtp_cache()
    toks = [(i * 13 + 3) % 128 for i in range(30)]
    table = _hidden_table(30)
    script = [(0, 3), (2, 3), (1, 3), (2, 3), (0, 2), (1, 3), (2, 3), (0, 3)]
    f = _run_cycles(model, cache, toks, table, script)

    ref = _reference_blocks(model, toks, table, f, 3)
    for j in range(3):
        valid = f - 1 - j
        live_k, live_v = cache[j][0].keys_and_values()
        ref_k, ref_v = ref[j][0].keys_and_values()
        assert live_k.shape[2] >= valid and ref_k.shape[2] >= valid
        dk = mx.max(mx.abs(live_k[:, :, :valid] - ref_k[:, :, :valid])).item()
        dv = mx.max(mx.abs(live_v[:, :, :valid] - ref_v[:, :, :valid])).item()
        assert dk < 1e-4 and dv < 1e-4, f"block {j} diverged: k={dk} v={dv}"


def test_full_rejection_gap_rewrite(runtime):
    """Consecutive all-reject cycles exercise the trim + conv-rewind path
    every fold; committed rows must still match the one-shot oracle."""
    model = _mtp_language_model()
    cache = model.make_mtp_cache()
    toks = [(i * 7 + 11) % 128 for i in range(16)]
    table = _hidden_table(16, seed=9)
    script = [(0, 3)] * 8
    f = _run_cycles(model, cache, toks, table, script)
    assert f == 8

    ref = _reference_blocks(model, toks, table, f, 3)
    for j in range(3):
        valid = f - 1 - j
        live_k, _ = cache[j][0].keys_and_values()
        ref_k, _ = ref[j][0].keys_and_values()
        dk = mx.max(mx.abs(live_k[:, :, :valid] - ref_k[:, :, :valid])).item()
        assert dk < 1e-4, f"block {j} diverged after gap rewrites: {dk}"


def test_variable_depth_lag_heals(runtime):
    """Dropping to depth 1 leaves deep blocks lagging; raising the depth
    again must refold them from the ring back to the oracle state."""
    model = _mtp_language_model()
    cache = model.make_mtp_cache()
    toks = [(i * 5 + 2) % 128 for i in range(24)]
    table = _hidden_table(24, seed=11)
    script = [(0, 3), (1, 3), (0, 1), (0, 1), (0, 1), (0, 3), (1, 3)]
    f = _run_cycles(model, cache, toks, table, script)

    ref = _reference_blocks(model, toks, table, f, 3)
    for j in range(3):
        valid = f - 1 - j
        live_k, _ = cache[j][0].keys_and_values()
        ref_k, _ = ref[j][0].keys_and_values()
        dk = mx.max(mx.abs(live_k[:, :, :valid] - ref_k[:, :, :valid])).item()
        assert dk < 1e-4, f"block {j} did not heal after depth dip: {dk}"


def test_verify_rollback_matches_sequential_decode(runtime, strict_math_device):
    """Rolling back a rejected verify chunk must leave the backbone cache
    equivalent to having decoded only the accepted tokens one by one."""
    model = _mtp_language_model()
    prompt = mx.array([[5, 17, 42, 91, 12, 63]])
    step_tokens = [7, 33, 54, 76]  # verify chunk; accept first 3, reject last

    ref_cache = model.make_cache()
    model(prompt, cache=ref_cache)
    for tok in step_tokens[:3]:
        model(mx.array([[tok]]), cache=ref_cache)

    cache = model.make_cache()
    model(prompt, cache=cache)
    verify = mx.array([step_tokens])
    out = model(verify, cache=cache, return_hidden=True)
    assert isinstance(out.gdn_states, dict)
    assert out.gdn_states["verify_len"] == 4

    accepted = model.rollback_speculative_cache(
        cache, out.gdn_states, accepted=2, block_size=4
    )
    assert accepted == 2
    for layer_cache, ref_layer in zip(cache, ref_cache):
        assert layer_cache[0].offset == ref_layer[0].offset
        for slot in range(4):
            got = layer_cache[1][slot]
            want = ref_layer[1][slot]
            assert got is not None and want is not None
            diff = mx.max(mx.abs(got - want)).item()
            assert diff < 1e-4, f"conv slot {slot} diverged after rollback: {diff}"

    ref_out = model(mx.array([[100]]), cache=ref_cache)
    test_out = model(mx.array([[100]]), cache=cache)
    mx.eval(ref_out.logits, test_out.logits)
    diff = mx.max(mx.abs(test_out.logits - ref_out.logits)).item()
    assert diff < 1e-3, f"post-rollback logits diverged: {diff}"


def test_sanitize_hook_maps_mtp_keys(runtime):
    import importlib

    inkling_mod = importlib.import_module("mlx_vlm.models.inkling.inkling")
    model = inkling_mod.Model.__new__(inkling_mod.Model)

    hidden, inter = 8, 4
    w13 = mx.arange(2 * inter * hidden, dtype=mx.float32).reshape(2 * inter, hidden)
    weights = {
        "model.mtp.layers.0.input_proj.weight": mx.zeros((hidden, 2 * hidden)),
        "model.mtp.layers.0.embed_norm.weight": mx.ones((hidden,)),
        "model.mtp.layers.0.transformer_block.attn.wq_du.weight": mx.zeros(
            (hidden, hidden)
        ),
        "model.mtp.layers.0.transformer_block.attn.wk_dv.weight": mx.zeros(
            (hidden, hidden)
        ),
        "model.mtp.layers.0.transformer_block.attn.wv_dv.weight": mx.zeros(
            (hidden, hidden)
        ),
        "model.mtp.layers.0.transformer_block.attn.wr_du.weight": mx.zeros(
            (hidden, hidden)
        ),
        "model.mtp.layers.0.transformer_block.attn.k_sconv.weight": mx.zeros(
            (hidden, 1, 4)
        ),
        "model.mtp.layers.0.transformer_block.mlp.w13_dn.weight": w13,
        "model.llm.embed.weight": mx.zeros((16, hidden)),
    }
    out = inkling_mod.Model.sanitize(model, weights)
    base = "language_model.mtp.blocks.0."
    assert base + "input_proj.weight" in out
    assert base + "embed_norm.weight" in out
    assert base + "transformer_block.self_attn.qkvr_proj.weight" in out
    assert out[base + "transformer_block.self_attn.k_sconv.conv.weight"].shape == (
        hidden,
        4,
        1,
    )
    gate = out[base + "transformer_block.mlp.gate_proj.weight"]
    ref = w13.reshape(inter, 2, hidden)
    assert mx.array_equal(gate, ref[:, 0, :])
    assert "language_model.model.embed_tokens.weight" in out


def test_prompt_priming_capture_and_take(runtime):
    """Chunked prefill captures the pair window (no head forwards); at
    activation mtp_take_primed folds every block with the lag invariant
    end_j = F - j."""
    from omlx.patches.mlx_lm_mtp import prompt_priming

    model = _mtp_language_model()
    cache = model.make_cache()
    ids = mx.array([[3, 9, 4, 7, 1, 8, 2, 6, 12, 15, 22, 30]])
    model(ids[:, :6], cache=cache)
    model(ids[:, 6:], cache=cache)

    ctx = getattr(model, prompt_priming._CTX_ATTR, None)
    assert ctx is not None, "priming context was not captured"
    assert ctx.total == 11  # P-1 pairs captured, head caches untouched
    assert ctx.pending_hidden is not None

    # The activation forward (return_hidden=True) must NOT capture — the
    # v1 primed=0 bug: it broke the ctx offset chain before take_primed.
    out = model(mx.array([[41]]), cache=cache, return_hidden=True)
    assert out is not None
    ctx2 = getattr(model, prompt_priming._CTX_ATTR, None)
    assert ctx2 is ctx and ctx.total == 11

    primed = prompt_priming.take_primed(model, cache, mx.array([41]))
    assert primed is not None, "activation seam rejected the capture"
    head_cache, hist = primed
    assert hist == 12  # 11 prompt pairs + seam pair
    assert head_cache.frontier == 12
    assert head_cache.base == [0] * len(head_cache)
    for j in range(model._omlx_mtp_depth):
        assert head_cache[j][0].offset == 12 - j, f"block {j} lag broken"

    # The primed cache must drive a normal cycle.
    model.mtp_begin_cycle(head_cache, 3)
    logits, _ = model.mtp_forward(
        _hidden_table(1), mx.array([[9]], dtype=mx.uint32), head_cache,
        return_hidden=True, logits_keep=1,
    )
    assert logits.shape == (1, 1, 128)
    assert head_cache.frontier == 13


def test_prompt_priming_window_slides(runtime, monkeypatch):
    """Prompts longer than the priming window slide chunks out instead of
    invalidating the context."""
    from omlx.patches.mlx_lm_mtp import prompt_priming

    monkeypatch.setattr(runtime, "_PRIME_WINDOW", 8)
    model = _mtp_language_model()
    cache = model.make_cache()
    ids = mx.array([[(i * 3 + 1) % 128 for i in range(18)]])
    model(ids[:, :6], cache=cache)
    model(ids[:, 6:12], cache=cache)
    model(ids[:, 12:], cache=cache)

    ctx = getattr(model, prompt_priming._CTX_ATTR, None)
    assert ctx is not None
    assert ctx.total >= 8, "window slide dropped below the priming window"

    model(mx.array([[50]]), cache=cache, return_hidden=True)  # activation seam
    primed = prompt_priming.take_primed(model, cache, mx.array([50]))
    assert primed is not None
    head_cache, hist = primed
    w_eff = 8
    assert hist == ctx.total + 1
    assert all(b == hist - w_eff for b in head_cache.base)
    for j in range(model._omlx_mtp_depth):
        assert head_cache[j][0].offset == w_eff - j


def test_keepalive_refolds_lagging_blocks(runtime):
    """A shallow cruise lets deep blocks lag; once the lag crosses the
    threshold the next fold refolds every reachable block from the ring
    (no clamp, honest deep probes afterwards)."""
    from omlx.patches.mlx_vlm_mtp.inkling_vlm_runtime import _KEEPALIVE_LAG

    model = _mtp_language_model()
    cache = model.make_mtp_cache()
    assert cache.active_max == 3
    table = _hidden_table(64, seed=13)
    fired_at = None
    for i in range(_KEEPALIVE_LAG + 6):
        model.mtp_begin_cycle(cache, 1)
        model.mtp_forward(
            table[:, i : i + 1],
            mx.array([[(i * 3 + 1) % 128]], dtype=mx.uint32),
            cache,
            return_hidden=True,
            logits_keep=1,
        )
        if cache.fold_keepalive and fired_at is None:
            fired_at = i
            cache.fold_keepalive = False
    assert fired_at is not None, "keepalive never fired"
    f = cache.frontier
    for j in range(3):
        lag = (f - 1 - j) - cache.valid_rows[j]
        assert lag < _KEEPALIVE_LAG, f"block {j} lag {lag} not bounded"
    assert not cache.clamp_logged

    # A deep cycle right after must run without any clamp and land all
    # blocks aligned at the new frontier.
    model.mtp_begin_cycle(cache, 3)
    model.mtp_forward(
        table[:, 40:41], mx.array([[9]], dtype=mx.uint32), cache,
        return_hidden=True, logits_keep=1,
    )
    for j in range(1, 3):
        model.mtp_forward(
            cache.win_hid[:, -1:], mx.array([[5]], dtype=mx.uint32), cache,
            return_hidden=True,
        )
    assert not cache.clamp_logged
    assert [cache.base[j] + cache[j][0].offset for j in range(3)] == [
        cache.frontier
    ] * 3


def test_keepalive_resets_unreachable_block(runtime):
    """A block whose committed rows sit below the ring window restarts
    fresh at the window start in its own base frame."""
    model = _mtp_language_model()
    cache = model.make_mtp_cache()
    cache.active_max = 1  # suppress keepalive during the cruise
    table = _hidden_table(96, seed=17)
    for i in range(80):
        model.mtp_begin_cycle(cache, 1)
        model.mtp_forward(
            table[:, i : i + 1],
            mx.array([[(i * 5 + 2) % 128]], dtype=mx.uint32),
            cache,
            return_hidden=True,
            logits_keep=1,
        )
    assert cache.frontier == 80
    cache.active_max = 3
    model.mtp_begin_cycle(cache, 1)
    model.mtp_forward(
        table[:, 80:81], mx.array([[7]], dtype=mx.uint32), cache,
        return_hidden=True, logits_keep=1,
    )
    assert cache.fold_keepalive
    # Blocks 1 and 2 could not reach back past the ring: fresh base at the
    # window start, rows covering [w0, F_prev - j).
    for j in (1, 2):
        assert cache.base[j] > 0, f"block {j} was not reset"
        assert cache.base[j] + cache[j][0].offset == 80 - j
        assert cache.valid_rows[j] >= 80 - 2 - j


def test_controller_observe_time_sample_gate():
    from omlx.patches.mlx_lm_mtp.batch_generator import _DepthController

    c = _DepthController(4)
    c._warmup = []
    c.observe(2, 1, 40.0)
    t_before = dict(c.t)
    c.observe(2, 1, 400.0, time_sample=False)
    assert c.t == t_before, "keepalive cycle time leaked into t_est"
    c.observe(2, 1, 40.0)
    assert c.t != t_before or c.t[2] == t_before[2]


_TINY_VISION = dict(
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    native_patch_size=4,
    adapter_merge_size=2,
    posemb_grid_size=4,
    projection_hidden_dim=64,
    mlp_ratio=2.0,
)


def _inkling2_tiny(formats=("mxfp8",) * 6 + ("nvfp4",) * 6, vision=None, **dims):
    from mlx.utils import tree_flatten

    from omlx.patches.inkling2 import model as inkling2

    layers = len(formats)
    text = dict(
        hidden_size=128,
        num_hidden_layers=layers,
        vocab_size=320,
        unpadded_vocab_size=300,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=32,
        sliding_window_size=8,
        rel_extent=8,
        d_rel=8,
        local_layer_ids=[i for i in range(layers) if (i + 1) % 6],
        n_routed_experts=8,
        num_experts_per_tok=2,
        intermediate_size=64,
        expert_formats=list(formats),
    )
    text.update(dims)
    model = inkling2.Model(
        inkling2.ModelArgs(model_type="inkling2", text_config=text, vision_config=vision)
    )
    mx.random.seed(0)
    weights = []
    for name, value in tree_flatten(model.parameters()):
        if value.dtype == mx.uint32:
            # Small FP8/FP4 codes only: random words would hit E4M3 NaNs.
            raw = mx.random.randint(0, 0x70, (*value.shape[:-1], value.shape[-1] * 4))
            weights.append((name, raw.astype(mx.uint8).view(mx.uint32)))
        elif value.dtype == mx.uint8:
            weights.append((name, mx.full(value.shape, 120, dtype=mx.uint8)))
        elif name.endswith("global_scale"):
            weights.append((name, value))
        else:
            noise = mx.random.normal(value.shape) * 0.05
            weights.append((name, noise.astype(value.dtype)))
    model.load_weights(weights)
    return model


def test_inkling2_incremental_and_chunked_match_full_forward():
    model = _inkling2_tiny()
    ids = mx.random.randint(0, 300, (1, 26))
    full = model(ids)[..., :300]
    cache = model.make_cache()
    steps = [model(ids[:, :18], cache=cache)[:, -1]]
    for t in range(18, 26):
        steps.append(model(ids[:, t : t + 1], cache=cache)[:, -1])
    incremental = mx.stack(steps[1:], axis=1)[..., :300]
    assert mx.abs(full[:, 18:] - incremental).max().item() < 1e-3
    chunked = model.make_cache()
    model(ids[:, :11], cache=chunked)
    tail = model(ids[:, 11:], cache=chunked)[..., :300]
    assert mx.abs(full[:, 11:] - tail).max().item() < 1e-3


def test_inkling2_verify_rollback_matches_sequential_decode():
    model = _inkling2_tiny()
    ids = mx.random.randint(0, 300, (1, 30))
    reference = model.make_cache()
    model(ids[:, :20], cache=reference)
    for t in range(20, 24):
        model(ids[:, t : t + 1], cache=reference)
    expected = model(ids[:, 24:25], cache=reference)
    cache = model.make_cache()
    model(ids[:, :20], cache=cache)
    out = model(
        ids[:, 20:28], cache=cache, return_hidden=True, capture_layer_ids=[5, 12]
    )
    assert [h.shape for h in out.hidden_states] == [(1, 8, 128)] * 3
    # Three of seven drafts accepted: the confirmed token plus three stay.
    assert model.mtp_partial_rollback(cache, 3, 7)
    got = model(ids[:, 24:25], cache=cache)
    assert cache[0][0].offset == reference[0][0].offset == 25
    assert mx.abs(got[..., :300] - expected[..., :300]).max().item() < 1e-4


def test_inkling2_fused_kernels_match_reference():
    from omlx.patches.inkling2 import kernels

    hidden = 256
    blocks = [mx.fast.rms_norm(mx.random.normal((2, 3, hidden)), None, 1e-6)] * 4
    partial = mx.fast.rms_norm(mx.random.normal((2, 3, hidden)), None, 1e-6)
    query = mx.random.normal((3, hidden)) * 0.1
    src = mx.stack([*blocks, partial], axis=-2)
    weights = mx.softmax((src @ query.T) * hidden**-0.5, axis=-2, precise=True)
    expected = weights.swapaxes(-1, -2) @ src
    got = kernels.attn_res_mix(blocks, partial, query, hidden**-0.5)
    assert mx.abs(got - expected).max().item() < 1e-5

    logits = mx.random.normal((5, 64)) * 2
    bias = mx.random.normal((64,)) * 0.05
    inds, w = kernels.route_topk(logits, bias, 4, 8.0)
    scores = mx.sigmoid(logits)
    ref = mx.argpartition(-(scores + bias), kth=3, axis=-1)[..., :4]
    assert mx.array_equal(
        mx.sort(inds.astype(mx.int32), axis=-1), mx.sort(ref.astype(mx.int32), axis=-1)
    )
    picked = mx.take_along_axis(scores, inds.astype(mx.int32), axis=-1)
    assert mx.allclose(w, picked / picked.sum(-1, keepdims=True) * 8.0, atol=1e-5)


def test_inkling2_model_is_freed_without_cyclic_gc():
    import gc
    import weakref

    model = _inkling2_tiny()
    ids = mx.random.randint(0, 300, (1, 4))
    mx.eval(model(ids, cache=model.make_cache()))
    ref = weakref.ref(model.model.layers[3])
    gc.disable()
    try:
        del model
        assert ref() is None
    finally:
        gc.enable()


@pytest.mark.parametrize(
    "batch, heads, kv_heads, length, keys, window, extent, left_padding",
    [
        (1, 4, 2, 64, 64, 16, 16, None),
        (1, 4, 2, 37, 300, 16, 16, None),
        (1, 6, 2, 100, 300, 0, 32, None),
        (2, 4, 1, 70, 90, 0, 32, [0, 13]),
    ],
)
def test_inkling2_flash_attention_matches_masked_sdpa(
    batch, heads, kv_heads, length, keys, window, extent, left_padding
):
    from omlx.patches.inkling2 import kernels

    if mx.default_device() != mx.gpu:
        pytest.skip("Metal kernel")
    dim = 128
    q = (mx.random.normal((batch, heads, length, dim)) * 0.3).astype(mx.bfloat16)
    # K/V as a strided slice of a larger cache buffer.
    shape = (batch, kv_heads, keys + 32, dim)
    kbuf = (mx.random.normal(shape) * 0.3).astype(mx.bfloat16)
    vbuf = mx.random.normal(shape).astype(mx.bfloat16)
    k, v = kbuf[:, :, :keys], vbuf[:, :, :keys]
    rel = mx.random.normal((batch, length, heads, extent)).astype(mx.bfloat16)
    sink = (mx.random.normal((batch, length, heads)) * 2).astype(mx.bfloat16)
    lp = mx.array(left_padding, dtype=mx.int32) if left_padding else None
    args = (q, k, v, rel, sink)
    expected = kernels._masked_attention(*args, dim**-0.5, window, lp)
    got = kernels.flash_banded_attention(
        *args, scale=dim**-0.5, window=window, left_padding=lp
    )
    assert got.shape == expected.shape
    diff = mx.abs(got.astype(mx.float32) - expected.astype(mx.float32)).max()
    assert diff.item() < 1e-2


def test_inkling2_masked_attention_holds_one_mask_block(monkeypatch):
    # NAX hosts run full-attention prefill through the masked path; at 64K
    # keys the masks of all query blocks once took 12 GiB together.
    from omlx.patches.inkling2 import kernels

    if mx.default_device() != mx.gpu:
        pytest.skip("Metal peak memory")
    heads, length, keys, dim = 4, 256, 4096, 32
    monkeypatch.setattr(kernels, "_MASK_BUDGET", 1 << 16)
    q = mx.random.normal((1, heads, length, dim)).astype(mx.bfloat16)
    k = mx.random.normal((1, 2, keys, dim)).astype(mx.bfloat16)
    v = mx.random.normal((1, 2, keys, dim)).astype(mx.bfloat16)
    rel = mx.random.normal((1, length, heads, 32)).astype(mx.bfloat16)
    sink = mx.random.normal((1, length, heads)).astype(mx.bfloat16)
    mx.eval(q, k, v, rel, sink)
    block = heads * 16 * (keys + 1) * 2  # one 16-row bf16 mask
    mx.reset_peak_memory()
    base = mx.get_active_memory()
    mx.eval(kernels._masked_attention(q, k, v, rel, sink, dim**-0.5, 0, None))
    # One block's mask, scores and the padded K/V take about 4 blocks; the 16
    # masks held together took over 30.
    assert mx.get_peak_memory() - base < 10 * block


def test_inkling2_image_processor_pads_normalizes_and_orders_blocks():
    from PIL import Image

    from omlx.patches.inkling2.processing import Inkling2ImageProcessor

    pixels = np.arange(50 * 90 * 3, dtype=np.uint8).reshape(50, 90, 3)
    out = Inkling2ImageProcessor().preprocess([Image.fromarray(pixels)])
    # 50 x 90 pads to 2 x 3 blocks of 40 px.
    assert out["image_grid_thw"].tolist() == [[1, 2, 3]]
    blocks = out["pixel_values"]
    assert blocks.shape == (6, 40, 40, 3)
    normalized = (pixels.astype(np.float32) / 255 - 0.5) / 0.5
    np.testing.assert_allclose(blocks[1, :, :], normalized[:40, 40:80], atol=1e-6)
    pad = (-1.0 / 255 - 0.5) / 0.5
    np.testing.assert_allclose(blocks[5, 10:, :], pad, atol=1e-6)
    np.testing.assert_allclose(blocks[2, :, 10:], pad, atol=1e-6)


def test_inkling2_position_table_matches_bilinear_resize():
    from omlx.patches.inkling2.vision import _bilinear_taps, _grid_coords

    side, rows, cols = 4, 6, 10
    table = np.random.default_rng(0).normal(size=(side, side)).astype(np.float32)

    def resize(size, i):
        # F.interpolate(bilinear, align_corners=False): clamped half-pixel source.
        src = max((i + 0.5) * side / size - 0.5, 0.0)
        low = min(int(np.floor(src)), side - 1)
        high = min(low + 1, side - 1)
        return low, high, src - low

    row, col = _grid_coords(rows, cols, 2)
    h_taps, h_w = _bilinear_taps(row, rows, side)
    w_taps, w_w = _bilinear_taps(col, cols, side)
    got = np.einsum("pa,pb,pab->p", h_w, w_w, table[h_taps[:, :, None], w_taps[:, None, :]])
    for p, (r, c) in enumerate(zip(row, col)):
        r0, r1, fr = resize(rows, r)
        c0, c1, fc = resize(cols, c)
        top = table[r0, c0] * (1 - fc) + table[r0, c1] * fc
        bottom = table[r1, c0] * (1 - fc) + table[r1, c1] * fc
        assert abs(got[p] - (top * (1 - fr) + bottom * fr)) < 1e-5


def test_inkling2_vision_encodes_images_independently_and_splices_tokens():
    model = _inkling2_tiny(formats=("mxfp8",) * 6, vision=_TINY_VISION)
    first = mx.random.normal((2 * 3, 8, 8, 3))
    second = mx.random.normal((1 * 2, 8, 8, 3))
    together = model.encode_image(
        mx.concatenate([first, second]), [[1, 2, 3], [1, 1, 2]]
    )
    assert together.shape == (8, 128)
    alone = model.encode_image(second, [[1, 1, 2]])
    assert mx.abs(together[6:] - alone).max().item() < 1e-4

    image = model.args.image_token_id
    ids = mx.array([[5, image, image, 9]])
    embeds = model.embed_inputs(ids, second, [[1, 1, 2]])
    text = model.model.embed_norm(model.model.embed(ids))
    assert mx.allclose(embeds[0, 0], text[0, 0])
    assert mx.allclose(embeds[0, 1:3], alone.astype(embeds.dtype))
    with pytest.raises(ValueError):
        model.embed_inputs(mx.array([[image, 1]]), second, [[1, 1, 2]])
