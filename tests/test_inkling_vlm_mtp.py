# SPDX-License-Identifier: Apache-2.0
"""Inkling Lightning MTP runtime tests (uniform-window multi-block cycle)."""

from __future__ import annotations

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
        live_k, live_v = cache[j][0].state
        ref_k, ref_v = ref[j][0].state
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
        live_k, _ = cache[j][0].state
        ref_k, _ = ref[j][0].state
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
        live_k, _ = cache[j][0].state
        ref_k, _ = ref[j][0].state
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

    monkeypatch.setenv("OMLX_INKLING_MTP_PRIME_WINDOW", "8")
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
