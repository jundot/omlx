# SPDX-License-Identifier: Apache-2.0
"""Inkling Lightning MTP runtime tests (uniform-window multi-block cycle)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from mlx.utils import tree_flatten

try:
    import mlx.core as mx
    import mlx.nn as nn

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from omlx.patches.inkling2 import fused, kernels, loopcd
from omlx.patches.inkling2 import model as inkling2_model
from omlx.patches.moe_expert_offload import (
    apply_moe_expert_offload,
    materialize_offload_state,
    moe_offload_stats,
)
from omlx.patches.qwen35_verify_qmm import (
    apply_verify_qmm_patch,
    set_verify_qmm_armed,
)

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


@pytest.fixture()
def strict_math_device():
    """Use deterministic CPU reductions for chunk-shape parity assertions."""
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.disable_compile()
    try:
        yield
    finally:
        mx.enable_compile()
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

_TINY_PLAIN = dict(
    formats=("mxfp8",) * 3 + ("nvfp4",) * 3,
    n_layers_per_attn_res_block=3,
    local_layer_ids=[0, 1, 3, 4],
)
_TINY_LOOP = dict(
    formats=("mxfp8",) * 9,
    n_layers_per_attn_res_block=3,
    local_layer_ids=[0, 1, 3, 4, 6, 7],
    loop_config=dict(core_start=3, core_end=6, num_passes=3, gate_rank=64),
)
_TINY_LOOPCD = dict(_TINY_LOOP, skip_thinking=True)
_FIRST_CONTROL, _OPEN_THINKING, _END_BLOCK = 290, 291, 299


@pytest.fixture(autouse=True)
def tiny_control_tokens(monkeypatch):
    monkeypatch.setattr(loopcd, "FIRST_CONTROL_TOKEN", _FIRST_CONTROL)
    monkeypatch.setattr(loopcd, "THINKING_TOKEN", _OPEN_THINKING)


def _inkling2_tiny(
    formats=("mxfp8",) * 6 + ("nvfp4",) * 6, vision=None, skip_thinking=None, **dims
):
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
            if name.endswith(("loop_down", "loop_gain_forget")):
                # Checkpoints ship the controllers as BF16 values.
                noise = noise.astype(mx.bfloat16)
            weights.append((name, noise.astype(value.dtype)))
    model.load_weights(weights)
    if skip_thinking is not None:
        assert model.enable_loopcd(skip_thinking)
    return model


@pytest.mark.parametrize("dims", [_TINY_PLAIN, _TINY_LOOP], ids=["plain", "looped"])
def test_inkling2_incremental_and_chunked_match_full_forward(dims):
    model = _inkling2_tiny(**dims)
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


def _unguided_and_guided_logits(ids):
    one_pass = dict(_TINY_LOOP["loop_config"], num_passes=1)
    first = _inkling2_tiny(**{**_TINY_LOOP, "loop_config": one_pass})(ids)[..., :300]
    final = _inkling2_tiny(**_TINY_LOOP)(ids)[..., :300]
    return final, final + 0.5 * (final - first)


def test_inkling2_loopcd_contrasts_with_the_one_pass_model():
    ids = mx.random.randint(0, 300, (2, 26))
    _, expected = _unguided_and_guided_logits(ids)
    model = _inkling2_tiny(**_TINY_LOOP, skip_thinking=False)
    out = model(ids, capture_layer_ids=[15], return_hidden=True)
    assert mx.abs(out.logits[..., :300] - expected).max().item() < 1e-4
    assert (out.logits[..., 300:] == -mx.inf).all()
    assert mx.array_equal(out.hidden_states[0], out.hidden_states[1])


@pytest.mark.parametrize("change", [dict(num_passes=1), dict(core_end=9)])
def test_inkling2_loopcd_needs_a_second_pass_and_a_coda(change):
    loop = dict(_TINY_LOOP["loop_config"], **change)
    model = _inkling2_tiny(**{**_TINY_LOOP, "loop_config": loop})
    assert not model.enable_loopcd(False)


def test_inkling2_loopcd_strength_follows_the_top_two_margin():
    tie, clear_winner = [3.0, 3.0, 0.0], [4.0, 1.0, 0.0]
    first = mx.array([[1.0, 2.0, 0.0], [2.0, 2.0, 0.0]])
    got = loopcd.guide(mx.array([tie, clear_winner]), first, 1e3, None, None)
    expected = mx.array([[4.0, 3.5, 0.0], [4.1104, 0.9448, 0.0]])
    assert mx.allclose(got, expected, atol=1e-3)


def test_inkling2_loopcd_skip_thinking_keeps_control_probabilities(monkeypatch):
    monkeypatch.setattr(loopcd, "FIRST_CONTROL_TOKEN", 2)
    final, first = mx.array([[[3.0, 2.0, 2.5]]]), mx.array([[[0.0, 2.0, 1.0]]])
    guided = loopcd.guide(final, first, 1e3, None, mx.array([[False]]))
    p, base = mx.softmax(guided[0, 0]), mx.softmax(final[0, 0])
    assert p[2].item() == pytest.approx(base[2].item(), abs=1e-4)
    assert p[0].item() > base[0].item() + 0.05


def test_inkling2_loopcd_can_skip_thinking_blocks():
    ids = mx.random.randint(0, _FIRST_CONTROL, (1, 32))
    ids[0, 20], ids[0, 25] = _OPEN_THINKING, _END_BLOCK
    final, guided = _unguided_and_guided_logits(ids)
    model = _inkling2_tiny(**_TINY_LOOPCD)
    full = model(ids)[..., :300]
    assert mx.abs(full[0, 20:25] - final[0, 20:25]).max().item() < 1e-4
    for t in (19, 25, 31):
        text, want = (mx.softmax(x[0, t, :_FIRST_CONTROL]) for x in (full, guided))
        assert mx.abs(text - want).max().item() < 1e-6

    cache = model.make_cache()
    model(ids[:, :18], cache=cache)
    model(ids[:, 18:26], cache=cache)
    assert model.mtp_partial_rollback(cache, 3, 7)
    for t in range(22, 28):
        step = model(ids[:, t : t + 1], cache=cache)[0, 0, :300]
        assert mx.abs(step - full[0, t]).max().item() < 1e-3


def test_inkling2_full_attention_visits_share_one_log_scaling(monkeypatch):
    # Past a 4-token log floor the queries scale with position. A forward
    # computes that once for every full-attention visit; each visit computing
    # its own from its cache must give the same bits.
    from omlx.patches.inkling2.model import Inkling2Layer

    model = _inkling2_tiny(log_scaling_n_floor=4, **_TINY_LOOP)
    ids = mx.random.randint(0, 300, (1, 30))

    def run():
        cache = model.make_cache()
        outs = [model(ids[:, :20], cache=cache), model(ids[:, 20:28], cache=cache)]
        return outs + [model(ids[:, t : t + 1], cache=cache) for t in (28, 29)]

    shared = run()
    own = Inkling2Layer.__call__
    monkeypatch.setattr(
        Inkling2Layer, "__call__", lambda self, *a, tau=None, **kw: own(self, *a, **kw)
    )
    for a, b in zip(shared, run()):
        assert mx.array_equal(a, b)


def test_inkling2_loop_controller_ignores_prefill_chunking():
    from omlx.patches.inkling2.model import _mix_messages

    # Release shapes: the 2H = 4608 -> r = 64 down projection is where a plain
    # matmul picks a row-count dependent kernel.
    hidden, rank, rows = 2304, 64, 2048
    for n in (1, 3):
        x = mx.random.normal((n, 1, rows, hidden))
        msg = mx.random.normal((n, 1, rows, hidden))
        down = mx.random.normal((n, 2, hidden, rank)) * 0.02
        gates = mx.random.normal((n, rank, 2 * hidden)) * 0.1
        full = _mix_messages(x, msg, down, gates)
        part = _mix_messages(x[:, :, :1000], msg[:, :, :1000], down, gates)
        assert mx.array_equal(part, full[:, :, :1000])


def test_inkling2_bf16_loop_kernels_match_the_fp32_path():
    import mlx.nn as nn

    from omlx.patches.inkling2.model import _bf16_loop_weights, _mix_messages_normed

    if mx.default_device() != mx.gpu:
        pytest.skip("Metal kernels")
    hidden, rank = 2304, 64
    mx.random.seed(5)
    layer = nn.Module()
    layer.loop_down = mx.random.normal((4, 2, hidden, rank)) * 0.05
    layer.loop_gain_forget = mx.random.normal((4, rank, 2 * hidden)) * 0.05
    # Weights that BF16 cannot hold exactly keep the FP32 path.
    assert _bf16_loop_weights(layer) is None
    for name in ("loop_down", "loop_gain_forget"):
        value = getattr(layer, name).astype(mx.bfloat16).astype(mx.float32)
        setattr(layer, name, value)
    packed = _bf16_loop_weights(layer)
    messages = mx.random.normal((2, 4, 4, hidden))
    # Full and sliding attention inputs, then the MLP input.
    for m0, n, qx, bf16 in (
        (0, 3, 3, packed[0]),
        (0, 3, 1, packed[0]),
        (3, 1, 1, packed[1]),
    ):
        args = (
            mx.random.normal((2, 4, qx, hidden)),
            messages,
            m0,
            n,
            layer.loop_down[m0 : m0 + n],
            layer.loop_gain_forget[m0 : m0 + n],
            mx.ones((qx, hidden)),
            1e-6,
            mx.bfloat16,
        )
        want = _mix_messages_normed(*args, want_f32=True)[1]
        got = _mix_messages_normed(*args, want_f32=True, bf16_weights=bf16)[1]
        # The paths sum in different orders, so low can round one BF16 ULP apart.
        assert mx.allclose(got, want, rtol=1e-2, atol=1e-2)


def test_inkling2_loop_model_type_uses_inkling2(tmp_path):
    from mlx_lm.utils import _get_classes

    from omlx.patches.inkling2 import apply_inkling2_patch
    from omlx.patches.inkling2 import model as inkling2
    from omlx.patches.inkling2.dspark import bundled_inkling2_dspark_path
    from omlx.utils.model_loading import normalize_inkling2_modelopt_quant

    assert apply_inkling2_patch()
    assert _get_classes({"model_type": "inkling2_loop"})[0] is inkling2.Model
    release = {
        "model_type": "inkling2_loop",
        "quantization_config": {"quant_method": "modelopt"},
    }
    assert "quantization_config" not in normalize_inkling2_modelopt_quant(release)
    mtp = {"markov_head_type": "rnn_hidden"}
    config = {"model_type": "inkling2_loop", "mtp_config": mtp}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "mtp.safetensors").touch()
    assert bundled_inkling2_dspark_path(tmp_path) is not None


# 15 drafts: the widest window the model declares (a verbatim context copy).
@pytest.mark.parametrize(
    "dims, drafts", [(_TINY_PLAIN, 7), (_TINY_LOOP, 15)], ids=["plain", "looped"]
)
def test_inkling2_verify_rollback_matches_sequential_decode(dims, drafts):
    model = _inkling2_tiny(**dims)
    assert drafts + 1 <= model._omlx_mtp_max_verify_rows
    ids = mx.random.randint(0, 300, (1, 40))
    reference = model.make_cache()
    model(ids[:, :20], cache=reference)
    for t in range(20, 24):
        model(ids[:, t : t + 1], cache=reference)
    expected = model(ids[:, 24:25], cache=reference)
    cache = model.make_cache()
    model(ids[:, :20], cache=cache)
    out = model(
        ids[:, 20 : 21 + drafts],
        cache=cache,
        return_hidden=True,
        capture_layer_ids=[5, model.model.num_visits],
    )
    assert [h.shape for h in out.hidden_states] == [(1, drafts + 1, 128)] * 3
    # Three drafts accepted: the confirmed token plus three stay.
    assert model.mtp_partial_rollback(cache, 3, drafts)
    got = model(ids[:, 24:25], cache=cache)
    full = next(c for c in cache if hasattr(c, "offset"))
    assert full.offset == next(c for c in reference if hasattr(c, "offset")).offset == 25
    assert mx.abs(got[..., :300] - expected[..., :300]).max().item() < 1e-4


def _bf16(model):
    from mlx.utils import tree_flatten, tree_unflatten

    cast = model.cast_predicate
    model.update(
        tree_unflatten(
            [
                (k, v.astype(mx.bfloat16) if v.dtype == mx.float32 and cast(k) else v)
                for k, v in tree_flatten(model.parameters())
            ]
        )
    )
    return model


def _run_inkling2_steps(model, ids, steps):
    """Prefill chunks, decode steps and verify blocks with a rollback."""
    cache, outs, pos = model.make_cache(), [], 0
    for step in steps:
        if isinstance(step, int):
            outs.append(model(ids[:, pos : pos + step], cache=cache)[:, -1])
            pos += step
        else:
            block, accepted = step
            out = model(
                ids[:, pos : pos + block],
                cache=cache,
                return_hidden=True,
                capture_layer_ids=[5],
            )
            assert model.mtp_partial_rollback(cache, accepted, block - 1)
            outs.append(out.logits[:, : accepted + 1])
            pos += accepted + 1
        mx.eval(outs[-1], [c.state for c in cache])
    return outs, cache


def test_inkling2_sliding_state_matches_a_cache_of_every_key(monkeypatch):
    # BF16 with chunks of 32 tokens or more runs the flash prefill kernel,
    # whose tiles must not move when old keys are dropped.
    from omlx.patches.inkling2 import cache as windows

    window = 48
    model = _bf16(
        _inkling2_tiny(sliding_window_size=window, rel_extent=window, **_TINY_LOOP)
    )
    ids = mx.random.randint(0, 300, (1, 520))
    steps = [200, 77, 33, 1, 1]
    steps += [(8, 3), 1, (8, 7), (6, 0), 1] * 2
    steps += [64, 1, (8, 2)]
    got, cache = _run_inkling2_steps(model, ids, steps)

    # The reference state never drops a key; attention still masks the window.
    adopt = windows.ArraysCache.adopt.__func__
    monkeypatch.setattr(
        windows.ArraysCache,
        "adopt",
        classmethod(lambda cls, entry, window=None: adopt(cls, entry, 4096)),
    )
    expected, reference = _run_inkling2_steps(model, ids, steps)
    for a, b in zip(got, expected):
        assert mx.array_equal(a, b)
    held = cache[0][windows.KEYS].shape[2]
    assert window - 1 <= held < window + 16
    assert reference[0][windows.KEYS].shape[2] == 410


def test_inkling2_rows_with_different_windows_decode_together():
    from omlx.patches.inkling2 import cache as windows

    model = _inkling2_tiny(sliding_window_size=16, rel_extent=16, **_TINY_LOOPCD)
    ids = mx.random.randint(0, 300, (2, 60))
    lengths = (41, 9)
    alone, expected = [], []
    for row, n in enumerate(lengths):
        cache = model.make_cache()
        model(ids[row : row + 1, :n], cache=cache)
        alone.append(cache)
        reference = model.make_cache()
        model(ids[row : row + 1, :n], cache=reference)
        steps = [
            model(ids[row : row + 1, n + t : n + t + 1], cache=reference)[:, -1]
            for t in range(4)
        ]
        expected.append(mx.stack(steps, axis=1))
    # Join the way the batch generator does: KV rows left-padded, states
    # extended in place.
    joined = []
    for a, b in zip(*alone):
        if hasattr(a, "offset"):
            joined.append(type(a).merge([a, b]))
        else:
            a.extend(b)
            joined.append(a)
    pads = joined[0][windows.PADDING].tolist()
    assert pads[0] == 0 and pads[1] > 0
    tokens = mx.stack(
        [ids[row, n : n + 4] for row, n in enumerate(lengths)], axis=0
    )
    steps = [model(tokens[:, t : t + 1], cache=joined)[:, -1] for t in range(4)]
    got = mx.stack(steps, axis=1)
    for row in range(2):
        diff = mx.abs(got[row, :, :300] - expected[row][0, :, :300]).max().item()
        assert diff < 1e-3
    # A row leaves the batch without the other row's padding.
    short = [c.extract(1) for c in joined]
    assert short[0][windows.KEYS].shape[2] == lengths[1] + 4
    last = ids[1:2, lengths[1] + 4 : lengths[1] + 5]
    reference = model.make_cache()
    model(ids[1:2, : lengths[1] + 4], cache=reference)
    diff = mx.abs(model(last, cache=short) - model(last, cache=reference))
    assert diff[..., :300].max().item() < 1e-3


def test_inkling2_batch_verify_rolls_each_row_back_to_its_own_count():
    model = _inkling2_tiny(sliding_window_size=16, rel_extent=16, **_TINY_LOOPCD)
    ids = mx.random.randint(0, 300, (2, 70))
    lengths, rounds, block = (41, 9), ([1, 5], [6, 0]), 7
    ends = list(lengths)
    for accepted in rounds:
        ends = [end + a + 1 for end, a in zip(ends, accepted)]
    alone, expected = [], []
    for row, (n, end) in enumerate(zip(lengths, ends)):
        cache = model.make_cache()
        model(ids[row : row + 1, :n], cache=cache)
        alone.append(cache)
        reference = model.make_cache()
        model(ids[row : row + 1, :end], cache=reference)
        expected.append(model(ids[row : row + 1, end : end + 1], cache=reference))
    joined = []
    for a, b in zip(*alone):
        if hasattr(a, "offset"):
            joined.append(type(a).merge([a, b]))
        else:
            a.extend(b)
            joined.append(a)
    starts = list(lengths)
    for accepted in rounds:
        tokens = mx.stack([ids[row, s : s + block] for row, s in enumerate(starts)])
        out = model(tokens, cache=joined, return_hidden=True, capture_layer_ids=[5])
        assert out.gdn_states == []
        model.rollback_speculative_cache(joined, out.gdn_states, accepted, block)
        starts = [s + a + 1 for s, a in zip(starts, accepted)]
    nxt = mx.stack([ids[row, end : end + 1] for row, end in enumerate(ends)])
    short = [c.extract(1) for c in joined]
    got = mx.concatenate([model(nxt, cache=joined), model(nxt[1:], cache=short)])
    for row, want in enumerate([*expected, expected[1]]):
        diff = mx.abs(got[row, :, :300] - want[0, :, :300]).max().item()
        assert diff < 1e-3


def test_inkling2_restored_state_decodes_and_joins_like_the_live_one():
    # The prefix cache rebuilds a sliding state as a plain ArraysCache inside
    # oMLX's size wrapper, without the growth buffers.
    from mlx_lm.models.cache import ArraysCache, KVCache

    from omlx.cache.paged_ssd_cache import (
        _extract_tensor_bytes,
        _restore_tensor_from_bytes,
    )
    from omlx.cache.type_handlers import SizedArraysCache

    model = _inkling2_tiny(sliding_window_size=16, rel_extent=16, **_TINY_LOOPCD)
    ids = mx.random.randint(0, 300, (1, 50))
    live = model.make_cache()
    model(ids[:, :40], cache=live)
    restored = []
    for entry in live:
        if isinstance(entry, KVCache):
            copy = KVCache()
            copy.state = entry.state
        else:
            plain = ArraysCache(len(entry.cache))
            plain.cache = list(entry.cache)
            copy = SizedArraysCache(plain, 40)
        restored.append(copy)
    restored = model.restore_cache(restored)
    expected = [model(ids[:, 40:48], cache=live, return_hidden=True).logits]
    assert model.mtp_partial_rollback(live, 2, 7)
    # The conv states are now views of the verify windows; the SSD cache
    # writes them through the buffer protocol.
    for entry in live:
        for state in [] if isinstance(entry, KVCache) else entry.cache:
            if state is not None:
                raw = _extract_tensor_bytes(state)
                assert mx.array_equal(_restore_tensor_from_bytes(*raw), state)
    got = [model(ids[:, 40:48], cache=restored, return_hidden=True).logits]
    assert model.mtp_partial_rollback(restored, 2, 7)
    for t in range(43, 46):
        expected.append(model(ids[:, t : t + 1], cache=live))
        got.append(model(ids[:, t : t + 1], cache=restored))
    for a, b in zip(got, expected):
        assert mx.array_equal(a, b)
    # Another row joins the restored one.
    other = model.make_cache()
    model(ids[:, :7], cache=other)
    for a, b in zip(restored, other):
        if not isinstance(a, KVCache):
            a.extend(b)
            assert a.batch_size == 2


@pytest.mark.parametrize(
    "dims, width, proj",
    [
        (_TINY_PLAIN, 8, "dense"),
        (_TINY_PLAIN, 16, "quantized"),
        (_TINY_LOOPCD, 16, "dense"),
        (_TINY_LOOPCD, 8, "quantized"),
    ],
    ids=["plain-8-dense", "plain-16-quantized", "loopcd-16-dense", "loopcd-8-quantized"],
)
def test_inkling2_row_exact_verify_rows_match_serial_decode(dims, width, proj):
    """With quantized projections, or BF16 dense ones as in the release, each
    verify row has the bits of the serial decode step for its token, also when
    an unflagged forward of the verify width traced the compiled layers first."""
    import mlx.nn as nn

    from omlx.patches.qwen35_verify_qmm import (
        apply_verify_qmm_patch,
        set_verify_qmm_armed,
    )

    if mx.default_device() != mx.gpu:
        pytest.skip("row-exact projections run on the GPU")
    apply_verify_qmm_patch()
    # Eight experts a token, so an 8-row block routes the 64 pairs that the
    # gather would otherwise sort.
    model = _inkling2_tiny(num_experts_per_tok=8, **dims)
    if proj == "quantized":
        nn.quantize(
            model,
            group_size=64,
            bits=8,
            class_predicate=lambda _, m: isinstance(m, nn.Linear),
        )
    else:
        model = _bf16(model)
    assert model._omlx_mtp_row_exact_verify
    ids = mx.random.randint(0, 300, (1, 40 + width))
    model(ids[:, :width], cache=model.make_cache())

    def prefilled():
        cache = model.make_cache()
        model(ids[:, :40], cache=cache)
        # As the server does after each prefill chunk.
        mx.eval([c.state for c in cache])
        return cache

    cache = prefilled()
    serial = [
        model(ids[:, t : t + 1], cache=cache)[0, -1] for t in range(40, 40 + width)
    ]
    cache = prefilled()
    set_verify_qmm_armed(True, row_exact=True)
    try:
        block = model(ids[:, 40 : 40 + width], cache=cache)[0]
    finally:
        set_verify_qmm_armed(False)
    for row, expected in enumerate(serial):
        assert mx.array_equal(block[row], expected)
    if "loop_config" in dims:
        # The looped rows ran the BF16 controller kernels.
        assert model.model.layers[3].__dict__["_omlx_loop_bf16"][2] is not None


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


def _assert_bits(fn, reference, args, **kw):
    """``fn`` matches ``reference`` bit for bit: eager, as prefill chunks ran
    it, and in BF16/FP16 also compiled at decode and verify lengths, as those
    rows ran it. A compiled MLX graph is not a reference elsewhere: in FP32,
    and at prefill lengths on some GPUs (M2), it can round differently from
    the eager ops."""
    from omlx.patches.inkling2.model import _COMPILED_MAX_TOKENS

    got = fn(*args, **kw)
    expected = [reference(*args, **kw)]
    low_precision = kw.get("dtype", args[0].dtype) != mx.float32
    if low_precision and args[0].shape[1] <= _COMPILED_MAX_TOKENS:
        expected.append(mx.compile(lambda *a: reference(*a, **kw))(*args))
    for ref in expected:
        for a, b in zip(got, ref):
            assert a.dtype == b.dtype
            assert mx.array_equal(a, b)


# FP32 (the tiny models' dtype) has no rounding that could hide a fused
# multiply-add in a kernel.
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_inkling2_conv_kernels_keep_the_reference_bits(dtype):
    from omlx.patches.inkling2 import kernels

    if mx.default_device() != mx.gpu:
        pytest.skip("Metal kernels")
    dtype = getattr(mx, dtype)
    mx.random.seed(4)
    batch, channels, k, heads = 2, 256, 4, 8
    w = (mx.random.normal((channels, k)) * 0.5).astype(dtype)
    static_w = (mx.random.normal((k, channels)) * 0.5).astype(dtype)
    for length in (5, 70):  # a verify block and a prefill chunk
        # Projection outputs are column slices; a restored state may be a view.
        out = mx.random.normal((batch, length, 2 * channels + heads * k)) * 4
        out = out.astype(dtype)
        x, h = out[..., :channels], out[..., channels : 2 * channels]
        dyn = out[..., 2 * channels :]
        state = (mx.random.normal((batch, 8, channels)) * 4)[:, 2:5]
        residual = mx.random.normal((batch, length, channels)) * 3
        wide = mx.random.normal((batch, length, channels)) * 4
        for args, kw in (
            ((x, w, state, 0.4), {}),
            ((x, w, state, 0.4), dict(out_dtype=mx.float32)),
            ((x, w, state, 0.4), dict(residual=h, out_dtype=mx.float32)),
            ((wide, w, state, 0.4), dict(residual=residual, dtype=dtype)),
        ):
            _assert_bits(kernels.short_conv, kernels.short_conv_reference, args, **kw)
        args = (x, dyn, static_w, state, heads, 0.3)
        for kw in ({}, dict(out_dtype=mx.float32)):
            _assert_bits(
                kernels.dynamic_conv, kernels.dynamic_conv_reference, args, **kw
            )


def test_inkling2_row_exact_matmul_keeps_one_row_products():
    # Release shapes of the router and loop controller matmuls, where a plain
    # multi-row matmul takes another kernel than a single row.
    from omlx.patches.inkling2.model import _row_matmul
    from omlx.patches.qwen35_verify_qmm import set_verify_qmm_armed

    if mx.default_device() != mx.gpu:
        pytest.skip("GPU matmul kernels")
    hidden = 2304
    cases = [
        (mx.random.normal((8, hidden)), mx.random.normal((128, hidden)).T),
        (mx.random.normal((6, 8, hidden)), mx.random.normal((6, hidden, 64)) * 0.02),
        (mx.random.normal((3, 8, 64)), mx.random.normal((3, 64, 2 * hidden)) * 0.1),
    ]
    set_verify_qmm_armed(True, row_exact=True)
    try:
        for x, w in cases:
            rows = [x[..., r : r + 1, :] @ w for r in range(x.shape[-2])]
            assert mx.array_equal(_row_matmul(x, w), mx.concatenate(rows, axis=-2))
    finally:
        set_verify_qmm_armed(False)


def test_inkling2_fp32_norm_weights_stay_out_of_the_parameters():
    from mlx.utils import tree_flatten

    from omlx.patches.inkling2.model import _norm
    from omlx.utils.model_loading import cast_bf16_params_to_fp16

    model = _bf16(_inkling2_tiny(formats=("mxfp8",) * 2, local_layer_ids=[0]))

    def params():
        return [(k, v.shape) for k, v in tree_flatten(model.parameters())]

    before = params()
    ids = mx.random.randint(0, 300, (1, 20))
    x = mx.random.normal((1, 3, 128))
    norm = model.model.layers[1].mlp_norm
    for _ in range(2):
        mx.eval(model(ids))
        # Never saved, loaded or quantized with the weights.
        assert params() == before
        assert mx.array_equal(_norm(norm, x), norm(x))
        low = x.astype(norm.weight.dtype)
        got = _norm(norm, low)
        assert got.dtype == low.dtype and mx.array_equal(got, norm(low))
        # The float16 activation recast replaces every BF16 weight.
        cast_bf16_params_to_fp16(model)


def test_inkling2_compiled_tails_keep_the_eager_bits():
    from omlx.patches.inkling2 import model as inkling2

    logits = (mx.random.normal((1, 4, 320)) * 40).astype(mx.bfloat16)
    bias = mx.where(mx.arange(320) < 300, 0.0, -mx.inf)
    for b in (None, bias):
        compiled = inkling2._logit_tail_compiled(logits, 30.0, b)
        assert mx.array_equal(compiled, inkling2._logit_tail(logits, 30.0, b))
    # The NVFP4 tensor scale, applied inside the activation as on the matmul
    # output before.
    y = (mx.random.normal((6, 2, 1, 128)) * 3).astype(mx.bfloat16)
    s = mx.random.uniform(shape=(6, 2, 1, 1)).astype(mx.bfloat16)
    caps = mx.array(4.0), mx.array(25.0)
    scaled = y * s
    swiglu = inkling2._softcapped_swiglu
    expected = swiglu(scaled[..., 0::2], scaled[..., 1::2], *caps)
    got = swiglu(y[..., 0::2], y[..., 1::2], *caps, s)
    assert mx.array_equal(got, expected)


def test_inkling2_fused_layer_kernels_keep_the_mlx_bits(monkeypatch):
    """The kernels that fold a layer's small ops together (projection
    epilogue, attention-residual norms, loop controllers, expert sum) give the
    MLX ops' bits on eager prefill chunks and compiled decode rows."""
    if mx.default_device() != mx.gpu:
        pytest.skip("fused kernels are GPU-only")
    from omlx.patches.inkling2 import kernels

    # The BF16 controller kernels keep row-exactness only, not these bits.
    monkeypatch.setattr(kernels, "loop_bf16_supported", lambda *a: False)
    # 128-wide heads: the projection epilogue covers that width only.
    wide = dict(
        head_dim=128,
        swa_head_dim=128,
        num_attention_heads=2,
        swa_num_attention_heads=2,
        num_key_value_heads=1,
        swa_num_key_value_heads=1,
    )
    ids = mx.random.randint(0, 300, (1, 20))

    def run(model):
        from mlx.utils import tree_flatten

        # BF16 activations, with the FP32 tensors the loader keeps.
        keep = model.cast_predicate
        cast = [
            (k, v.astype(mx.bfloat16))
            for k, v in tree_flatten(model.parameters())
            if v.dtype == mx.float32 and keep(k)
        ]
        model.load_weights(cast, strict=False)
        cache = model.make_cache()
        outs = [model(ids[:, :17], cache=cache)]
        outs += [model(ids[:, t : t + 1], cache=cache) for t in range(17, 20)]
        mx.eval(outs)
        return outs

    names = ("qkv_post", "attn_res_fused", "moe_short_conv", "loop_message_outputs")
    used = dict.fromkeys(names, 0)
    for name in names:
        original = getattr(kernels, name)

        def spy(*args, _original=original, _name=name, **kwargs):
            out = _original(*args, **kwargs)
            used[_name] += out is not None
            return out

        monkeypatch.setattr(kernels, name, spy)
    fused = run(_inkling2_tiny(**wide, **_TINY_LOOP))
    assert all(used.values())

    for name in ("qkv_post", "attn_res_fused", "moe_short_conv"):
        monkeypatch.setattr(kernels, name, lambda *a, **k: None)
    monkeypatch.setattr(kernels, "loop_messages_supported", lambda *a: False)
    # A fresh model: compiled layer stages keep the graph of their first trace.
    reference = run(_inkling2_tiny(**wide, **_TINY_LOOP))
    for got, want in zip(fused, reference):
        assert mx.array_equal(got, want)


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


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
def test_inkling2_affine_expert_carrier_matches_mlx_packing(bits):
    from omlx.patches.inkling2.model import ExpertProj

    # 64 columns hide a wrong 3/5/6-bit width; the real 4608 does not.
    packed, _, _ = mx.quantize(mx.zeros((2, 8, 4608)), group_size=128, bits=bits)
    proj = ExpertProj(2, 8, 4608, {"mode": "affine", "bits": bits, "group_size": 128})
    assert proj.weight.shape == packed.shape


def _nvfp4_experts(experts=4, hidden=128, inter=64):
    """NVFP4 routed experts whose unscaled matmul outputs pass float16's
    range while the scaled outputs stay small, as in Inkling 2 Mini."""
    from omlx.patches.inkling2 import model as inkling2

    args = inkling2.TextArgs(
        hidden_size=hidden,
        intermediate_size=inter,
        n_routed_experts=experts,
        num_experts_per_tok=2,
    )
    spec = inkling2.expert_quant_spec("nvfp4")
    mod = inkling2.Inkling2Experts(args, spec, spec)
    rng = np.random.default_rng(0)
    for proj, out_dims, in_dims in (
        (mod.w13, 2 * inter, hidden),
        (mod.w2, hidden, inter),
    ):
        codes, scales, amaxes = [], [], []
        for _ in range(experts):
            w = mx.array(
                (0.1 + 0.02 * rng.standard_normal((out_dims, in_dims))).astype(
                    np.float32
                )
            )
            amax = mx.abs(w).max()
            q, s = mx.quantize(w, mode="nvfp4", global_scale=amax)
            codes.append(q)
            scales.append(s)
            amaxes.append(amax)
        proj.weight = mx.stack(codes)
        proj.scales = mx.stack(scales)
        proj.global_scale = mx.stack(amaxes).astype(mx.float32)
    return mod


def test_inkling2_nvfp4_experts_stay_finite_in_float16():
    if mx.default_device() != mx.gpu:
        pytest.skip("gather_qmm global_scale runs on the GPU")
    mod = _nvfp4_experts()
    tokens = 64  # >= 64 routed rows takes the sorted gather path too
    x = mx.full((tokens, 128), 3.0) + mx.random.normal(
        (tokens, 128), key=mx.random.key(1)
    )
    inds = mx.random.randint(0, 4, (tokens, 2), key=mx.random.key(2)).astype(mx.uint32)
    weights = mx.full((tokens, 2), 0.5)

    expected = mod(x.astype(mx.float32), inds, weights)
    got = mod(x.astype(mx.float16), inds, weights)
    mx.eval(expected, got)

    # Unscaled, the first projection is ~128 * 3 * 2688 ~ 1e6.
    assert mx.abs(expected).max().item() < 65504
    assert mx.isfinite(got).all().item()
    rel = mx.abs(got.astype(mx.float32) - expected).max() / mx.abs(expected).max()
    assert rel.item() < 1e-2


_OFFLOAD_DIMS = dict(n_routed_experts=32, num_experts_per_tok=8)


def _inkling2_offloaded(tmp_path, dims):
    reference, model = (_inkling2_tiny(**dims, **_OFFLOAD_DIMS) for _ in range(2))
    for m in (reference, model):
        for layer in m.model.layers:
            for proj in (layer.mlp.experts.w13, layer.mlp.experts.w2):
                if "global_scale" in proj:
                    proj.global_scale = proj.global_scale * mx.linspace(0.5, 2, 32)
    mx.eval(model.parameters())
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(reference.parameters()))
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "inkling2_loop", "text_config": _OFFLOAD_DIMS})
    )
    assert apply_moe_expert_offload(model, tmp_path, 0.25) == len(model.model.layers)
    materialize_offload_state(model)
    return reference, model


@pytest.mark.parametrize("dims", [{}, _TINY_LOOP], ids=["mxfp8+nvfp4", "looped"])
def test_inkling2_offloaded_logits_match_resident(tmp_path, dims):
    reference, model = _inkling2_offloaded(tmp_path, dims)
    ids = mx.random.randint(0, 300, (1, 24))
    outs = []
    for m in (reference, model):
        cache = m.make_cache()
        logits = [m(ids[:, t : t + 1], cache=cache) for t in range(24)]
        outs.append(mx.concatenate(logits, axis=1))
    mx.eval(outs)
    assert mx.array_equal(outs[0][..., :300], outs[1][..., :300]).item()
    assert moe_offload_stats(model)["misses"] > 8 * len(model.model.layers)


def test_inkling2_offload_over_capacity_reads_only_missing_experts(tmp_path):
    reference, model = _inkling2_offloaded(tmp_path, {})
    experts = model.model.layers[-1].mlp.experts
    x = mx.random.normal((64, 128)).astype(mx.bfloat16)
    inds = mx.arange(64 * 8).reshape(64, 8) % 32
    w = mx.ones((64, 8))
    expected = reference.model.layers[-1].mlp.experts(x, inds, w)
    got = inkling2_model.moe_sum(*experts.unreduced(x, inds, w))
    assert (mx.abs(got - expected).max() / mx.abs(expected).max()).item() < 1e-2
    before = experts.cache.misses
    mx.eval(experts.unreduced(x, inds, w))
    assert experts.cache.misses - before == 32 - experts.cache.capacity


def test_inkling2_offloaded_row_exact_verify_matches_resident(tmp_path):
    reference, model = _inkling2_offloaded(tmp_path, {})
    ids = mx.random.randint(0, 300, (1, 19))
    blocks = []
    for m in (reference, model):
        cache = m.make_cache()
        for t in range(16):
            m(ids[:, t : t + 1], cache=cache)
        set_verify_qmm_armed(True, row_exact=True)
        try:
            blocks.append(m(ids[:, 16:], cache=cache))
        finally:
            set_verify_qmm_armed(False)
    mx.eval(blocks)
    assert mx.array_equal(blocks[0][..., :300], blocks[1][..., :300]).item()


@pytest.mark.parametrize(
    "batch, heads, kv_heads, length, keys, window, extent, left_padding, dtype",
    [
        (1, 4, 2, 64, 64, 16, 16, None, "bfloat16"),
        (1, 4, 2, 37, 300, 16, 16, None, "bfloat16"),
        (1, 4, 2, 37, 300, 16, 16, None, "float16"),
        (1, 6, 2, 100, 300, 0, 32, None, "bfloat16"),
        (2, 4, 1, 70, 90, 0, 32, [0, 13], "bfloat16"),
    ],
)
def test_inkling2_flash_attention_matches_masked_sdpa(
    batch, heads, kv_heads, length, keys, window, extent, left_padding, dtype
):
    from omlx.patches.inkling2 import kernels

    dtype = getattr(mx, dtype)

    if mx.default_device() != mx.gpu:
        pytest.skip("Metal kernel")
    dim = 128
    q = (mx.random.normal((batch, heads, length, dim)) * 0.3).astype(dtype)
    # K/V as a strided slice of a larger cache buffer.
    shape = (batch, kv_heads, keys + 32, dim)
    kbuf = (mx.random.normal(shape) * 0.3).astype(dtype)
    vbuf = mx.random.normal(shape).astype(dtype)
    k, v = kbuf[:, :, :keys], vbuf[:, :, :keys]
    rel = mx.random.normal((batch, length, heads, extent)).astype(dtype)
    sink = (mx.random.normal((batch, length, heads)) * 2).astype(dtype)
    lp = mx.array(left_padding, dtype=mx.int32) if left_padding else None
    args = (q, k, v, rel, sink)
    expected = kernels._masked_attention(*args, dim**-0.5, window, lp)
    got = kernels.flash_banded_attention(
        *args, scale=dim**-0.5, window=window, left_padding=lp
    )
    assert got.shape == expected.shape
    diff = mx.abs(got.astype(mx.float32) - expected.astype(mx.float32)).max()
    assert diff.item() < 1e-2


def _nax_banded_or_skip():
    from omlx.custom_kernels.nax import is_nax_available
    from omlx.utils import nax_attention

    if mx.default_device() != mx.gpu or not is_nax_available():
        pytest.skip("NAX GPU")
    return nax_attention


@pytest.mark.parametrize(
    "batch, heads, kv_heads, length, keys, extent, left_padding",
    [
        (1, 6, 2, 64, 64, 16, None),  # no cached keys, aligned tiles
        (1, 6, 2, 33, 300, 32, None),  # ragged query and key tails
        (1, 6, 2, 9, 12301, 48, None),  # shortest block, three key-range passes
        (1, 4, 4, 100, 100, 256, None),  # band longer than the keys
        (3, 6, 2, 40, 300, 48, [0, 31, 260]),  # left-padded batch
    ],
)
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_nax_banded_attention_matches_fp32(
    batch, heads, kv_heads, length, keys, extent, left_padding, dtype
):
    nax_attention = _nax_banded_or_skip()
    q, k, v, rel, sink = _decode_attention_case(
        batch, heads, kv_heads, length, keys, extent, getattr(mx, dtype), dim=128
    )
    # Peaked, so a wrong band, sink or padding moves the result past the bound.
    rel, sink = rel * 6, sink * 4
    lp = mx.array(left_padding, dtype=mx.int32) if left_padding else None
    scale = 128**-0.5
    got = nax_attention.nax_banded_sink_attention(
        q, k, v, rel, sink, scale=scale, left_padding=lp
    )
    ref = nax_attention._banded_reference(q, k, v, rel, sink, scale, lp)
    assert mx.abs(got.astype(mx.float32) - ref).max().item() < 2e-2


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_nax_banded_attention_passes_repeat_one_dispatch(dtype):
    # Key-range passes hand the fp32 row state on: same bits as one dispatch,
    # also past the passes queued between evaluations.
    nax_attention = _nax_banded_or_skip()
    q, k, v, rel, sink = _decode_attention_case(
        2, 6, 2, 77, 1000, 48, getattr(mx, dtype), 128
    )
    lp = mx.array([0, 100], dtype=mx.int32)
    scale = 128**-0.5
    one = nax_attention._run_banded(q, k, v, rel, sink, scale, 0, lp)
    for pass_keys in (32, 96, 512):
        many = nax_attention._run_banded(q, k, v, rel, sink, scale, pass_keys, lp)
        assert mx.array_equal(one, many).item()


def test_nax_banded_attention_rows_do_not_depend_on_the_block():
    # A row's output depends on its own keys only, so prefill chunk
    # boundaries do not change it.
    nax_attention = _nax_banded_or_skip()
    q, k, v, rel, sink = _decode_attention_case(1, 6, 2, 96, 700, 48, mx.bfloat16, 128)
    scale = 128**-0.5
    whole = nax_attention.nax_banded_sink_attention(q, k, v, rel, sink, scale=scale)
    head = nax_attention.nax_banded_sink_attention(
        q[:, :, :64],
        k[:, :, :-32],
        v[:, :, :-32],
        rel[:, :64],
        sink[:, :64],
        scale=scale,
    )
    tail = nax_attention.nax_banded_sink_attention(
        q[:, :, 64:], k, v, rel[:, 64:], sink[:, 64:], scale=scale
    )
    assert mx.array_equal(whole, mx.concatenate([head, tail], axis=2)).item()


def test_inkling2_full_attention_prefill_takes_nax_banded(monkeypatch):
    from omlx.patches.inkling2 import kernels

    nax_attention = _nax_banded_or_skip()
    scale = 128**-0.5
    calls = []
    banded = nax_attention.nax_banded_sink_attention

    def counting(*a, **kw):
        out = banded(*a, **kw)
        calls.append(out is not None)
        return out

    monkeypatch.setattr(nax_attention, "nax_banded_sink_attention", counting)
    lp = mx.array([0, 13], dtype=mx.int32)
    for batch, length, padding, window, taken in (
        (1, 70, None, 0, True),
        (2, 70, lp, 0, True),
        (1, 9, None, 0, True),
        (1, 8, None, 0, False),
        (1, 70, None, 32, True),
    ):
        args = _decode_attention_case(batch, 6, 2, length, 300, 32, mx.bfloat16, 128)
        calls.clear()
        got = kernels.banded_sink_attention(
            *args, scale=scale, window=window, left_padding=padding
        )
        assert any(calls) == taken
        if taken:
            ref = kernels._masked_attention(*args, scale, window, padding)
            diff = mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max()
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


@pytest.mark.parametrize(
    "quant",
    [
        None,
        {"group_size": 64, "bits": 8},
        {"group_size": 64, "bits": 4},
        {"group_size": 32, "bits": 8, "mode": "mxfp8"},
        {"group_size": 16, "bits": 4, "mode": "nvfp4"},
    ],
    ids=["bf16", "affine8", "affine4", "mxfp8", "nvfp4"],
)
def test_inkling2_loop_projection_slices_match_the_stacked_call(
    strict_math_device, quant
):
    """Later loop passes project q, k and v through their own rows of
    ``in_proj``; a wrapped projection keeps the stacked call."""
    import mlx.nn as nn

    from omlx.patches.inkling2.model import Inkling2Attention, TextArgs

    args = TextArgs(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=32,
        sliding_window_size=8,
        rel_extent=8,
        d_rel=8,
        local_layer_ids=[0],
    )
    attn = Inkling2Attention(args, layer_idx=0)
    mx.random.seed(11)
    attn.update(
        {
            name: (mx.random.normal(getattr(attn, name).shape) * 0.1 + offset).astype(
                mx.bfloat16
            )
            for name, offset in (
                ("k_conv", 0.0),
                ("v_conv", 0.0),
                ("q_scale", 1.0),
                ("r_scale", 1.0),
                ("rel_proj", 0.0),
            )
        }
    )
    attn.in_proj.weight = (mx.random.normal(attn.in_proj.weight.shape) * 0.05).astype(
        mx.bfloat16
    )
    if quant is not None:
        attn.in_proj = nn.QuantizedLinear.from_linear(attn.in_proj, **quant)

    class Wrapped(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def __call__(self, x):
            return self.inner(x)

    sliced, stacked = attn.in_proj, Wrapped(attn.in_proj)
    for batch, length in ((1, 1), (2, 5), (1, 20)):
        xq, xk, xv = (
            (mx.random.normal((batch, length, 128)) * 0.2).astype(mx.bfloat16)
            for _ in range(3)
        )
        state = mx.random.normal((batch, args.sconv_kernel_size - 1, attn.dkv)) * 0.1
        attn.in_proj = sliced
        got = attn.project(xq, xk, xv, state, state, None)
        attn.in_proj = stacked
        expected = attn.project(xq, xk, xv, state, state, None)
        for g, e in zip(got, expected):
            assert mx.abs(g.astype(mx.float32) - e.astype(mx.float32)).max() < 1e-5


@pytest.mark.parametrize(
    "kwargs, effort",
    [
        ({"enable_thinking": False}, "none"),
        ({"reasoning_effort": "off", "enable_thinking": False}, "none"),
        ({"enable_thinking": False, "reasoning_effort": "low"}, "low"),
        ({"enable_thinking": True}, None),
    ],
)
def test_inkling2_thinking_switch_reaches_the_template_effort(kwargs, effort):
    from omlx.engine.vlm import _apply_template_thinking_mode

    _apply_template_thinking_mode("inkling2_loop", kwargs)
    assert kwargs.get("reasoning_effort") == effort


@pytest.mark.parametrize("engine", ["text", "vlm"])
def test_inkling2_keeps_dense_kv_when_kv_quantization_is_on(mock_tokenizer, engine):
    """Its attention reads dense keys: a packed cache fails on the first forward."""
    from types import SimpleNamespace

    from omlx.models.vlm import VLMModelAdapter
    from omlx.scheduler import Scheduler

    model = _inkling2_tiny()
    if engine == "vlm":
        model = VLMModelAdapter(SimpleNamespace(language_model=model))
    scheduler = Scheduler(model=model, tokenizer=mock_tokenizer)
    scheduler._turboquant_kv_bits = 8.0
    scheduler._turboquant_kv_scheme = "affine8"
    assert not scheduler._turboquant_eligible(model.make_cache())


def test_inkling2_varloop_single_pass_and_dspark_visit_guard(tmp_path):
    from omlx.patches.inkling2.dspark import bundled_inkling2_dspark_path

    # TML-d24-varloop-research supports num_passes=1..5 and text_config.loop_num_passes.
    model = _inkling2_tiny(
        formats=("mxfp8",) * 18,
        loop_config=dict(core_start=6, core_end=12, num_passes=3, gate_rank=8, is_varloop=True),
        loop_num_passes=1,
    )
    assert model.model.num_visits == 18
    assert hasattr(model.layers[6], "loop_down")
    ids = mx.random.randint(0, 300, (1, 10))
    full = model(ids)[..., :300]
    cache = model.make_cache()
    model(ids[:, :7], cache=cache)
    step = model(ids[:, 7:8], cache=cache)[:, -1, :300]
    assert mx.abs(full[:, 7] - step).max().item() < 1e-3

    # DSpark's target_layer_ids=[5, 23, 41, 48] are calibrated for 48 visits (3 passes);
    # bundled_inkling2_dspark_path must skip the drafter when loop_num_passes != 3.
    (tmp_path / "mtp.safetensors").touch()
    base_text = {
        "num_hidden_layers": 24,
        "n_layers_per_attn_res_block": 6,
        "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64, "is_varloop": True},
    }
    mtp = {"markov_head_type": "rnn_hidden", "target_layer_ids": [5, 23, 41, 48]}
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "inkling2_loop", "text_config": base_text, "mtp_config": mtp})
    )
    assert bundled_inkling2_dspark_path(tmp_path) is not None

    for passes in (1, 2, 5):
        override_text = {**base_text, "loop_num_passes": passes}
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "inkling2_loop", "text_config": override_text, "mtp_config": mtp})
        )
        assert bundled_inkling2_dspark_path(tmp_path) is None


def _decode_attention_case(batch, heads, kv_heads, length, keys, extent, dtype, dim=64):
    mx.random.seed(3)
    q = (mx.random.normal((batch, heads, length, dim)) * 0.3).astype(dtype)
    # K/V as a slice of a larger cache buffer, as the caches hand them over.
    shape = (batch, kv_heads, keys + 40, dim)
    k = (mx.random.normal(shape) * 0.3).astype(dtype)[:, :, 7 : 7 + keys]
    v = mx.random.normal(shape).astype(dtype)[:, :, 7 : 7 + keys]
    rel = mx.random.normal((batch, length, heads, extent)).astype(dtype)
    sink = (mx.random.normal((batch, length, heads)) * 2).astype(dtype)
    return q, k, v, rel, sink


def _small_decode_chunks(monkeypatch, kernels):
    """Several chunks at test sizes."""
    monkeypatch.setattr(kernels, "_DECODE_CHUNK", 64)
    monkeypatch.setattr(kernels, "_DECODE_SLIDING_CHUNK", 16)


@pytest.mark.parametrize(
    "heads, kv_heads, length, keys, window, left_padding, dtype",
    [
        (6, 2, 1, 1, 0, None, "bfloat16"),
        (6, 2, 1, 300, 0, [0, 113], "bfloat16"),
        (6, 2, 1, 300, 0, [0, 113], "float16"),
        (6, 2, 1, 90, 32, [0, 70], "bfloat16"),
        (6, 2, 4, 300, 0, [41, 0], "bfloat16"),
        (6, 2, 5, 300, 32, None, "bfloat16"),
        (12, 1, 2, 300, 0, None, "bfloat16"),
    ],
)
def test_inkling2_flash_decode_matches_masked_sdpa(
    heads, kv_heads, length, keys, window, left_padding, dtype, monkeypatch
):
    """Batched decode and verify blocks; 300 keys are five chunks of 64, a
    window of 32 is two chunks of 16, and 12 heads on one KV head fill two
    simdgroups."""
    from omlx.patches.inkling2 import kernels

    if mx.default_device() != mx.gpu:
        pytest.skip("Metal kernel")
    _small_decode_chunks(monkeypatch, kernels)
    args = _decode_attention_case(
        2, heads, kv_heads, length, keys, 32, getattr(mx, dtype)
    )
    lp = mx.array(left_padding, dtype=mx.int32) if left_padding else None
    scale = args[0].shape[-1] ** -0.5
    expected = kernels._masked_attention(*args, scale, window, lp)
    got = kernels.gqa_flash_decode_attention(
        *args, scale=scale, window=window, left_padding=lp
    )
    routed = kernels.banded_sink_attention(
        *args, scale=scale, window=window, left_padding=lp
    )
    assert mx.array_equal(routed, got)
    assert got.shape == expected.shape and got.dtype == expected.dtype
    diff = mx.abs(got.astype(mx.float32) - expected.astype(mx.float32)).max()
    assert diff.item() < 1e-2
    # Keys strided in the head dimension take the scalar staging loop.
    q, k, v, rel, sink = args
    wide = mx.repeat(k, 2, axis=-1)[..., ::2], mx.repeat(v, 2, axis=-1)[..., ::2]
    strided = kernels.gqa_flash_decode_attention(
        q, *wide, rel, sink, scale=scale, window=window, left_padding=lp
    )
    assert mx.array_equal(strided, got)


@pytest.mark.parametrize("window", [0, 32])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_inkling2_flash_decode_rows_have_the_bits_of_one_row_calls(
    window, dtype, monkeypatch
):
    """Each row of a left-padded batch of verify blocks is the decode step of
    its token: one query over that row's own keys, alone."""
    from omlx.patches.inkling2 import kernels

    if mx.default_device() != mx.gpu:
        pytest.skip("Metal kernel")
    _small_decode_chunks(monkeypatch, kernels)
    length, keys, pads = 8, 300, [0, 113]
    q, k, v, rel, sink = _decode_attention_case(
        2, 6, 2, length, keys, 32, getattr(mx, dtype)
    )
    scale = q.shape[-1] ** -0.5
    block = kernels.banded_sink_attention(
        q,
        k,
        v,
        rel,
        sink,
        scale=scale,
        window=window,
        left_padding=mx.array(pads, dtype=mx.int32),
    )
    for row, pad in enumerate(pads):
        for pos in range(length):
            end = keys - length + 1 + pos
            step = kernels.gqa_flash_decode_attention(
                q[row : row + 1, :, pos : pos + 1],
                k[row : row + 1, :, pad:end],
                v[row : row + 1, :, pad:end],
                rel[row : row + 1, pos : pos + 1],
                sink[row : row + 1, pos : pos + 1],
                scale=scale,
                window=window,
            )
            assert mx.array_equal(block[row : row + 1, :, pos : pos + 1], step)



@pytest.mark.parametrize("length", [1, 4])
@pytest.mark.parametrize("window, pad", [(0, 0), (0, 7), (32, 0), (32, 280)])
def test_inkling2_left_padded_row_takes_the_decode_kernel(
    length, window, pad, monkeypatch
):
    """A row left alone in a batch cache keeps its left_padding. Its decode
    steps and verify blocks take the decode kernel, not the masked path."""
    from omlx.patches.inkling2 import kernels

    if mx.default_device() != mx.gpu:
        pytest.skip("Metal kernel")
    _small_decode_chunks(monkeypatch, kernels)
    args = _decode_attention_case(1, 6, 2, length, 300, 32, mx.bfloat16)
    lp = mx.array([pad], dtype=mx.int32)
    scale = args[0].shape[-1] ** -0.5
    expected = kernels._masked_attention(*args, scale, window, lp)

    def no_masked(*a, **kw):
        raise AssertionError("a left-padded row took the masked path")

    monkeypatch.setattr(kernels, "_masked_attention", no_masked)
    got = kernels.banded_sink_attention(
        *args, scale=scale, window=window, left_padding=lp
    )
    assert mx.array_equal(
        got,
        kernels.gqa_flash_decode_attention(
            *args, scale=scale, window=window, left_padding=lp
        ),
    )
    diff = mx.abs(got.astype(mx.float32) - expected.astype(mx.float32)).max()
    assert diff.item() < 1e-2

def test_inkling2_flash_decode_guards_wide_heads_and_unaligned_keys():
    """A head too wide for the kernel's threadgroup memory falls back instead
    of failing to load, and K/V views whose rows or base are off the 16-byte
    grid of the vector load have the bits of their contiguous copies."""
    from omlx.patches.inkling2 import kernels

    if mx.default_device() != mx.gpu:
        pytest.skip("Metal kernel")
    dt = mx.bfloat16
    wide = _decode_attention_case(2, 2, 1, 1, 40, 32, dt, dim=512)
    assert not kernels._use_flash_decode(wide[0], wide[1])
    scale = 512**-0.5
    assert mx.array_equal(
        kernels.banded_sink_attention(*wide, scale=scale),
        kernels._masked_attention(*wide, scale, 0, None),
    )

    q, _, _, rel, sink = _decode_attention_case(1, 6, 2, 1, 300, 32, dt)
    scale = q.shape[-1] ** -0.5
    # (buffer width, first row, first column): unaligned rows, unaligned base.
    for width, row, col in ((71, 8, 0), (72, 7, 1)):
        shape = (1, 2, 344, width)
        view = (slice(None), slice(None), slice(row, row + 300), slice(col, col + 64))
        k = (mx.random.normal(shape) * 0.3).astype(dt)[view]
        v = mx.random.normal(shape).astype(dt)[view]
        got = kernels.gqa_flash_decode_attention(
            q, k, v, rel, sink, scale=scale, window=0
        )
        packed = kernels.gqa_flash_decode_attention(
            q, mx.contiguous(k), mx.contiguous(v), rel, sink, scale=scale, window=0
        )
        assert mx.array_equal(got, packed)
        expected = kernels._masked_attention(q, k, v, rel, sink, scale, 0, None)
        diff = mx.abs(got.astype(mx.float32) - expected.astype(mx.float32))
        assert diff.max().item() < 1e-2


def test_inkling2_fits_threadgroup_caches_and_only_swallows_size_errors():
    from omlx.patches.inkling2 import kernels

    calls = []

    def rejected():
        calls.append(1)
        raise ValueError(
            "Thread group size (576) is greater than  the maximum allowed "
            "threads per threadgroup (448)."
        )

    key = ("test-variant", 1)
    kernels._launch_ok.pop(key, None)
    try:
        assert kernels._fits_threadgroup(key, rejected) is False
        assert kernels._fits_threadgroup(key, rejected) is False
        assert calls == [1], "a variant is probed once"
    finally:
        kernels._launch_ok.pop(key, None)

    def other_error():
        raise ValueError("shape mismatch")

    with pytest.raises(ValueError, match="shape mismatch"):
        kernels._fits_threadgroup(("test-variant", 2), other_error)
    assert ("test-variant", 2) not in kernels._launch_ok


def test_inkling2_attn_res_fused_falls_back_when_threadgroup_rejected(monkeypatch):
    """M1-class GPUs cap some fused variants below their 576-thread groups
    (hidden 2304); those calls must return None so the MLX ops run instead."""
    from omlx.patches.inkling2 import kernels

    if mx.default_device() != mx.gpu:
        pytest.skip("Metal kernel")
    hidden = 2304
    blocks = [mx.random.normal((1, 2, hidden)) for _ in range(4)]
    args = (
        blocks,
        mx.random.normal((1, 2, hidden)),
        mx.random.normal((3, hidden)),
        1.0,
    )
    kwargs = dict(
        unit_eps=1e-6,
        norm_weight=mx.ones((hidden,)),
        norm_eps=1e-6,
        out_dtype=mx.float16,
        want_partial=True,
    )
    monkeypatch.setattr(kernels, "_fits_threadgroup", lambda key, launch: False)
    assert kernels.attn_res_fused(*args, **kwargs) is None
    monkeypatch.setattr(kernels, "_fits_threadgroup", lambda key, launch: True)
    monkeypatch.setattr(kernels, "_launch_ok", {})
    out = kernels.attn_res_fused(*args, **kwargs)
    assert out is not None


def test_inkling2_dspark_fold_dequantizes_the_head_in_chunks():
    # Two 16384-row chunks of a quantized head give the folded bits of the
    # whole-matrix dequantize.
    import mlx.nn as nn

    from omlx.patches.inkling2.dspark import Inkling2DSparkConfig, Inkling2MarkovHead

    vocab = 16896
    lm_head = nn.QuantizedLinear.from_linear(
        nn.Linear(64, vocab, bias=False), group_size=64, bits=8
    )
    head = Inkling2MarkovHead(
        Inkling2DSparkConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=32,
            intermediate_size=128,
            rms_norm_eps=1e-6,
            rope_theta=1e6,
            local_layer_ids=[],
            sliding_window=64,
            block_size=4,
            markov_rank=32,
            target_layer_ids=[1],
            vocab_size=vocab,
            unpadded_vocab_size=vocab,
            final_logit_softcapping=30.0,
        )
    )
    head._alpha = mx.array([1.0])
    head.fold(lm_head)
    weight = mx.dequantize(lm_head.weight, lm_head.scales, lm_head.biases, 64, 8)
    proj = head.hidden_out.weight.astype(mx.float32)
    bias = head.hidden_out.bias.astype(mx.float32)
    for start, end in ((0, 16384), (16384, vocab)):
        rows = weight[start:end].astype(mx.float32)
        expected = (rows @ proj).astype(mx.bfloat16)
        assert mx.array_equal(head._fold_weight[start:end], expected)
        expected = (rows @ bias).astype(mx.bfloat16)
        assert mx.array_equal(head._fold_bias[start:end], expected)


def _attention_inputs(length, keys, hq=8, hkv=2, d=128, r=16):
    mx.random.seed(length)
    q = (mx.random.normal((1, hq, length, d)) * 3).astype(mx.bfloat16)
    k = mx.random.normal((1, hkv, keys, d)).astype(mx.bfloat16)
    v = mx.random.normal((1, hkv, keys, d)).astype(mx.bfloat16)
    rel = (mx.random.normal((1, length, hq, r)) * 4).astype(mx.bfloat16)
    sink = (mx.random.normal((1, length, hq)) * 4 + 6).astype(mx.bfloat16)
    return q, k, v, rel, sink, d**-0.5


@pytest.mark.parametrize("hq", [8, 12])
@pytest.mark.parametrize("padding", [0, 700])
@pytest.mark.parametrize("length", [1, 5, 16])
def test_inkling2_long_attention_matches_reference(length, padding, hq):
    if mx.default_device() != mx.gpu:
        pytest.skip("the long-context attention kernel runs on the GPU")
    q, k, v, rel, sink, scale = _attention_inputs(length, 2600, hq=hq)
    lp = mx.array([padding]) if padding else None
    got = fused.long_decode_attention(q, k, v, rel, sink, scale=scale, left_padding=lp)
    expected = kernels._masked_attention(q, k, v, rel, sink, scale, 0, lp)
    diff = mx.abs(got.astype(mx.float32) - expected.astype(mx.float32))
    assert diff.max().item() < 2e-2
    s = k.shape[2]
    for t in range(length):
        end = s - length + t + 1
        alone = fused.long_decode_attention(
            q[:, :, t : t + 1],
            k[:, :, :end],
            v[:, :, :end],
            rel[:, t : t + 1],
            sink[:, t : t + 1],
            scale=scale,
            left_padding=lp,
        )
        assert mx.array_equal(got[:, :, t : t + 1], alone)


def test_inkling2_long_context_verify_rows_match_serial_decode(monkeypatch):
    if mx.default_device() != mx.gpu:
        pytest.skip("row-exact projections run on the GPU")
    monkeypatch.setattr(inkling2_model, "_LONG_ATTENTION_MIN_KEYS", 4096)
    calls = []
    kernel = fused.long_decode_attention
    monkeypatch.setattr(
        fused,
        "long_decode_attention",
        lambda *a, **kw: calls.append(a[0].shape[2]) or kernel(*a, **kw),
    )
    apply_verify_qmm_patch()
    model = _bf16(_inkling2_tiny(head_dim=128, swa_head_dim=128, **_TINY_LOOP))
    nn.quantize(
        model,
        group_size=64,
        bits=8,
        class_predicate=lambda _, m: isinstance(m, nn.Linear),
    )
    prefix, width = 4092, 16
    ids = mx.random.randint(0, 300, (1, prefix + width))

    def prefilled():
        cache = model.make_cache()
        for s in range(0, prefix, 2048):
            model(ids[:, s : min(prefix, s + 2048)], cache=cache)
            mx.eval([c.state for c in cache])
        return cache

    cache = prefilled()
    serial = [
        model(ids[:, t : t + 1], cache=cache)[0, -1]
        for t in range(prefix, prefix + width)
    ]
    cache = prefilled()
    calls.clear()
    set_verify_qmm_armed(True, row_exact=True)
    try:
        block = model(ids[:, prefix : prefix + width], cache=cache)[0]
    finally:
        set_verify_qmm_armed(False)
    assert calls
    for row, expected in enumerate(serial):
        assert mx.array_equal(block[row], expected)
