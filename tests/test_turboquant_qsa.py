# SPDX-License-Identifier: Apache-2.0
"""Phase 1: TurboQuant-QSA hybrid cache for qwen4_exp.

TurboQuantQSAKVCache stores attention K/V as TurboQuant packed states
(mlx_vlm MSE codecs, quantized on write) while the QSA indexer sidecar
(raw keys, positions, pooled block bank) stays dense — the mlx-serve
aux_state pattern. Contract under test:

* conversion from a populated QSAKVCache preserves offset/indexer state
  and quantizes K/V within codec tolerance;
* the gathered decode arm accepts the hybrid and stays close to the
  dense-cache attention output (selection is indexer-side; only the
  gathered rows are dequantized);
* sub-budget prefill runs through the mask/SDPA arm on packed states;
* state round-trips as a 4-tuple (TQ key state, TQ value state,
  index_keys, index_position_ids) for the store/restore paths;
* trim keeps KV/indexer alignment (the gathered arms fail closed on
  misalignment);
* to_batch/merge keep rows packed in BatchTurboQuantQSAKVCache;
* the scheduler converts QSA layers when TurboQuant is enabled and
  honors the OMLX_TQ_QSA=0 kill switch;
* nbytes reports packed KV plus the dense indexer sidecar.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

compat.apply_mlx_vlm_qwen4_exp_compat_patch()
from mlx_vlm.models.cache import ArraysCache  # noqa: E402
from mlx_vlm.models.qwen4_exp import TextConfig  # noqa: E402
from mlx_vlm.models.qwen4_exp.language import (  # noqa: E402
    QSAKVCache,
    Qwen4ExpAttention,
    TurboQuantQSAKVCache,
)
from mlx_vlm.turboquant import TurboQuantMSEState  # noqa: E402

pytestmark = pytest.mark.turboquant


@pytest.fixture(scope="module", autouse=True)
def _tq_attention_patch():
    """Route SDPA through the TQ-aware dispatcher, as the engine does."""
    from omlx.patches.turboquant_attention import (
        apply_turboquant_attention_patch,
    )

    apply_turboquant_attention_patch()


KV_HEADS = 1
HEAD_DIM = 64
IDX_DIM = 8
BITS = 4


def _rng(seed: int = 0):
    return np.random.default_rng(seed)


def _dense_qsa(tokens: int, seed: int = 0) -> tuple[QSAKVCache, mx.array, mx.array]:
    """Populated dense QSA cache with aligned indexer state."""
    cache = QSAKVCache()
    rng = _rng(seed)
    k = mx.array(
        rng.standard_normal((1, KV_HEADS, tokens, HEAD_DIM)).astype(np.float32)
    )
    v = mx.array(
        rng.standard_normal((1, KV_HEADS, tokens, HEAD_DIM)).astype(np.float32)
    )
    cache.update_and_fetch(k, v)
    idx = mx.array(rng.standard_normal((1, tokens, IDX_DIM)).astype(np.float32))
    positions = mx.arange(tokens, dtype=mx.int32)[None, :]
    cache.update_indexer(idx, positions)
    return cache, k, v


def _cosine(a: mx.array, b: mx.array) -> float:
    a = a.reshape(-1).astype(mx.float32)
    b = b.reshape(-1).astype(mx.float32)
    return float(mx.sum(a * b) / (mx.linalg.norm(a) * mx.linalg.norm(b)))


def _text_config():
    """Tiny qwen4_exp config (mirrors tests/test_qwen4_yarn_rope.py).

    indexer_budget=8 / compress_ratio=2 => block_topk=4, so the gathered
    decode arm engages past ~9 cached tokens.
    """
    return TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=3,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=16,
        moe_intermediate_size=16,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=1,
        max_position_embeddings=4096,
        hc_count=2,
        hc_lowrank=8,
        head_dim=128,
        layer_types=["linear_attention", "full_attention"],
        ple_layer_ids=[1],
        ple_embed_dim=32,
        ple_conv_kernel_size=3,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=2,
        eos_token_id=1,
        rope_parameters={
            "type": "default",
            "rope_theta": 10_000_000,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.5,
        },
    )


# ---------------------------------------------------------------------------
# Conversion + storage contract
# ---------------------------------------------------------------------------


def test_from_qsa_cache_quantizes_kv_and_preserves_indexer():
    cache, k, v = _dense_qsa(24)
    hybrid = TurboQuantQSAKVCache.from_qsa_cache(cache, bits=BITS)

    assert isinstance(hybrid.keys, TurboQuantMSEState)
    assert isinstance(hybrid.values, TurboQuantMSEState)
    assert hybrid.offset == cache.offset == 24
    # Indexer sidecar carried over verbatim (dense, aligned).
    assert mx.array_equal(hybrid.index_keys, cache.index_keys)
    assert mx.array_equal(hybrid.index_position_ids, cache.index_position_ids)
    # Quantized K/V reconstruct within 4-bit codec tolerance.
    dk, dv = hybrid.dequantize()
    assert _cosine(dk, k) > 0.98
    assert _cosine(dv, v) > 0.98
    # And actually compress.
    assert hybrid.nbytes < cache.nbytes * 0.5


def test_full_state_round_trips_as_four_tuple():
    cache, _, _ = _dense_qsa(16)
    hybrid = TurboQuantQSAKVCache.from_qsa_cache(cache, bits=BITS)
    # Runtime `state` stays the TQ 2-tuple the inherited machinery expects.
    assert isinstance(hybrid.state, tuple) and len(hybrid.state) == 2
    state = hybrid.full_state
    assert isinstance(state, tuple) and len(state) == 4
    ks, vs, idx, pos = state
    assert isinstance(ks, TurboQuantMSEState) and isinstance(vs, TurboQuantMSEState)

    fresh = TurboQuantQSAKVCache(bits=BITS)
    fresh.full_state = state
    assert fresh.offset == hybrid.offset
    dk_a, dv_a = hybrid.dequantize()
    dk_b, dv_b = fresh.dequantize()
    assert mx.allclose(dk_a, dk_b).item() and mx.allclose(dv_a, dv_b).item()
    assert mx.array_equal(fresh.index_keys, idx)
    assert mx.array_equal(fresh.index_position_ids, pos)


def test_trim_keeps_kv_and_indexer_aligned():
    cache, _, _ = _dense_qsa(24)
    hybrid = TurboQuantQSAKVCache.from_qsa_cache(cache, bits=BITS)
    hybrid.trim(4)
    assert hybrid.offset == 20
    assert hybrid.index_keys.shape[1] == 20
    assert hybrid.index_position_ids.shape[-1] == 20
    dk, _ = hybrid.dequantize()
    assert dk.shape[2] == 20


def test_nbytes_counts_packed_kv_plus_indexer():
    cache, _, _ = _dense_qsa(24)
    hybrid = TurboQuantQSAKVCache.from_qsa_cache(cache, bits=BITS)
    assert hybrid.nbytes >= hybrid.indexer_nbytes
    assert hybrid.nbytes == pytest.approx(
        hybrid.nbytes - hybrid.indexer_nbytes + hybrid.indexer_nbytes
    )
    # Packed KV alone must be smaller than the dense KV alone.
    assert hybrid.nbytes - hybrid.indexer_nbytes < cache.nbytes


# ---------------------------------------------------------------------------
# Attention-level behavior (tiny model)
# ---------------------------------------------------------------------------


def _prefill(attn, cache, tokens: int, seed: int):
    rng = _rng(seed)
    x = mx.array(rng.standard_normal((1, tokens, 32)).astype(np.float32))
    positions = mx.arange(tokens, dtype=mx.int32)[None, :]
    return attn(x, cache=cache, position_ids=positions)


_ISOLATED_CHILD_ENV = "OMLX_TQ_QSA_ISOLATED_CHILD"


def test_gathered_decode_matches_dense_within_codec_tolerance():
    """Run the dense-vs-hybrid comparison in a pristine subprocess.

    Module-forward numeric comparisons are corrupted by lazy MLX state
    that earlier tests in the same interpreter still hold: pending graphs
    get submitted mid-comparison, and a start-of-test drain cannot
    retract them (~25% flake in full-suite order, cosine 0.18-0.58
    varying per run; passes standalone and file-granular). A fresh
    process makes the failure class impossible by construction — no
    prior MLX state exists in it.
    """
    if os.environ.get(_ISOLATED_CHILD_ENV):
        pytest.skip("executed via the wrapper's isolated child process")
    child = "tests/test_turboquant_qsa.py::test_gathered_decode_isolated_child"
    env = {**os.environ, _ISOLATED_CHILD_ENV: "1"}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            child,
            "-q",
            "--tb=short",
            "-p",
            "no:cacheprovider",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    # "1 passed" pins that the child actually ran and passed: a silent
    # deselect/skip (marker changes, collection drift) must not read as
    # success.
    assert proc.returncode == 0 and "1 passed" in proc.stdout, (
        "isolated dense-vs-hybrid comparison failed:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


def test_gathered_decode_isolated_child():
    if not os.environ.get(_ISOLATED_CHILD_ENV):
        pytest.skip("runs only inside the wrapper's fresh subprocess")
    mx.random.seed(11)  # pin module init; the codec-tolerance band assumes
    # stable weights across the dense/hybrid comparison AND across runs.
    cfg = _text_config()
    attn = Qwen4ExpAttention(cfg)
    tokens = 24  # > indexer_budget: gathered decode arm engages

    dense_cache = QSAKVCache()
    _prefill(attn, dense_cache, tokens, seed=1)
    hybrid_cache = TurboQuantQSAKVCache.from_qsa_cache(dense_cache, bits=BITS)

    rng = _rng(7)
    x = mx.array(rng.standard_normal((1, 1, 32)).astype(np.float32))
    out_dense = attn(x, cache=dense_cache)
    # Evaluate before building the second chain: constructing a fresh lazy
    # forward over the shared module while an earlier graph is still pending
    # can corrupt numerics (pre-existing vendored-model hazard, reproducible
    # dense-vs-dense; production evaluates every step before the next).
    mx.eval(out_dense)
    out_hybrid = attn(x, cache=hybrid_cache)

    assert out_hybrid.shape == out_dense.shape
    assert mx.isfinite(out_hybrid).all().item()
    # 4-bit KV error passes through softmax-weighted V: cosine stays high,
    # relative L2 stays codec-scale. A broken gather/sidecar shows up as
    # garbage (cosine ~0) or a crash.
    assert _cosine(out_hybrid, out_dense) > 0.95
    rel = float(
        mx.linalg.norm(out_hybrid - out_dense)
        / mx.maximum(mx.linalg.norm(out_dense), 1e-6)
    )
    assert rel < 0.15, f"relative L2 {rel:.3f} exceeds codec-scale drift"
    # The hybrid stayed packed through the decode step and grew by one.
    assert isinstance(hybrid_cache.keys, TurboQuantMSEState)
    assert hybrid_cache.offset == tokens + 1
    assert hybrid_cache.index_keys.shape[1] == tokens + 1


def test_sub_budget_prefill_runs_on_packed_state():
    mx.random.seed(12)  # pin module init (see sibling test)
    cfg = _text_config()
    attn = Qwen4ExpAttention(cfg)
    hybrid = TurboQuantQSAKVCache(bits=BITS)
    out = _prefill(attn, hybrid, 6, seed=3)  # below indexer_budget=8: mask arm
    assert out.shape[0] == 1 and out.shape[1] == 6
    assert mx.isfinite(out).all().item()
    assert isinstance(hybrid.keys, TurboQuantMSEState)
    assert hybrid.offset == 6
    assert hybrid.index_keys.shape[1] == 6


# ---------------------------------------------------------------------------
# Scheduler wiring
# ---------------------------------------------------------------------------


class _SchedulerStub:
    _turboquant_kv_bits = float(BITS)
    _turboquant_skip_last = True

    def _model_uses_mla(self):
        return False

    def _model_uses_attention_sinks(self):
        return False


def _make_prompt_cache():
    return [QSAKVCache(), ArraysCache(size=2), QSAKVCache(), QSAKVCache()]


def test_scheduler_eligibility_accepts_qsa_layers():
    from omlx.scheduler import Scheduler

    stub = _SchedulerStub()
    assert Scheduler._turboquant_eligible(stub, _make_prompt_cache()) is True


def test_scheduler_kill_switch_disables_qsa_conversion(monkeypatch):
    from omlx.scheduler import Scheduler

    monkeypatch.setenv("OMLX_TQ_QSA", "0")
    stub = _SchedulerStub()
    assert Scheduler._turboquant_eligible(stub, _make_prompt_cache()) is False


def test_scheduler_converts_qsa_layers_and_skips_last():
    from omlx.scheduler import Scheduler

    stub = _SchedulerStub()
    caches = _make_prompt_cache()
    Scheduler._apply_turboquant_kv_convert(stub, caches)
    # skip_last: the final QSA layer stays dense fp16.
    assert isinstance(caches[0], TurboQuantQSAKVCache)
    assert isinstance(caches[1], ArraysCache)
    assert isinstance(caches[2], TurboQuantQSAKVCache)
    assert type(caches[3]) is QSAKVCache


def test_qwen4_exp_kv_pricing_models_the_hybrid():
    """The guard must see packed-KV pricing once QSA layers convert."""
    from omlx.memory_monitor import estimate_qwen4_exp_kv_bytes_per_token

    cfg = _text_config()  # kv_heads=1, head_dim=128, indexer_head_dim=8
    caches = _make_prompt_cache()  # 3 QSA layers + 1 ArraysCache
    aux = 8 * 2.0 + 24  # raw index key + 3 int64 MRoPE coords
    dense_kv = 2 * 1 * 128 * 2.0

    dense = estimate_qwen4_exp_kv_bytes_per_token(cfg, caches, 2.0)
    assert dense == 3 * (dense_kv + aux)

    tq = estimate_qwen4_exp_kv_bytes_per_token(
        cfg, caches, 2.0, tq_bits=4.0, tq_skip_last=True
    )
    hybrid_kv = 2 * 1 * (128 * 4.0 / 8.0 + 2.0)  # packed + fp16 norm
    # skip-last: one dense QSA layer, two hybrids.
    assert tq == (dense_kv + aux) + 2 * (hybrid_kv + aux)
    assert tq < dense * 0.6

    # No skip-last: all three layers convert.
    tq_all = estimate_qwen4_exp_kv_bytes_per_token(
        cfg, caches, 2.0, tq_bits=4.0, tq_skip_last=False
    )
    assert tq_all == 3 * (hybrid_kv + aux)


# ---------------------------------------------------------------------------
# Phase 2: serialization (store/restore) contracts
# ---------------------------------------------------------------------------


def _packed_hybrid(tokens: int = 8, seed: int = 0):
    """(ks, vs, index_keys, index_positions) 4-tuple via the real codecs."""
    from mlx_vlm.turboquant import _build_codec

    rng = _rng(seed)
    k = mx.array(
        rng.standard_normal((1, KV_HEADS, tokens, HEAD_DIM)).astype(np.float32)
    )
    v = mx.array(
        rng.standard_normal((1, KV_HEADS, tokens, HEAD_DIM)).astype(np.float32)
    )
    ks = _build_codec(k, BITS, mode="mse", seed=0).quantize(k)
    vs = _build_codec(v, BITS, mode="mse", seed=1).quantize(v)
    ik = mx.array(rng.standard_normal((1, tokens, IDX_DIM)).astype(np.float32))
    ip = mx.arange(tokens, dtype=mx.int32)[None, :]
    return ks, vs, ik, ip


def test_slice_concat_full_state_round_trip():
    """The store/restore primitives: slice per block, concat back, full_state."""
    from omlx.turboquant_kv import _concat_state_token_axis, _slice_state_range

    cache, _, _ = _dense_qsa(16)
    hybrid = TurboQuantQSAKVCache.from_qsa_cache(cache, bits=BITS)
    ks, vs, ik, ip = hybrid.full_state
    parts = [
        (
            _slice_state_range(ks, a, b),
            _slice_state_range(vs, a, b),
            ik[:, a:b],
            ip[..., a:b],
        )
        for a, b in ((0, 8), (8, 16))
    ]
    fresh = TurboQuantQSAKVCache(bits=BITS)
    fresh.full_state = (
        _concat_state_token_axis([p[0] for p in parts]),
        _concat_state_token_axis([p[1] for p in parts]),
        mx.concatenate([p[2] for p in parts], axis=1),
        mx.concatenate([p[3] for p in parts], axis=-1),
    )
    assert fresh.offset == 16
    d0 = hybrid.dequantize()
    d1 = fresh.dequantize()
    assert mx.allclose(d0[0], d1[0], atol=1e-3, rtol=1e-3).item()
    assert mx.array_equal(fresh.index_keys, ik)
    assert mx.array_equal(fresh.index_position_ids, ip)


def test_save_block_round_trips_hybrid_payload(tmp_path):
    import json as _json
    import time as _time

    from omlx.cache.paged_ssd_cache import PagedSSDCacheManager

    ks, vs, ik, ip = _packed_hybrid(8)
    mgr = PagedSSDCacheManager(
        cache_dir=tmp_path / "ssd",
        max_size_bytes=1 << 30,
        expected_model_name="m",
        expected_num_layers=1,
        expected_block_size=8,
    )
    try:
        mgr.set_expected_layer_signature(
            ["TurboQuantQSAKVCache"], turboquant_kv_bits=4.0
        )
        saved = mgr.save_block(
            block_hash=b"5a" * 16,
            cache_data=[("__turboquant_qsa_v1__", (ks, vs, ik, ip))],
            token_count=8,
            model_name="m",
            layer_cache_types=["TurboQuantQSAKVCache"],
            layer_meta_states=[(8, 4.0, 0)],
        )
        assert saved is True
        path = mgr._get_file_path(b"5a" * 16)
        for _ in range(50):
            if path.exists():
                break
            _time.sleep(0.1)
        assert path.exists(), "background writer never produced the file"
        arrays, metadata = mx.load(str(path), return_metadata=True)
        sig = _json.loads(metadata["cache_signature"])
        assert sig["turboquant_kv_bits"] == 4.0
        assert metadata.get("layer_0_turboquant_qsa_v1") == "1"
        data = mgr._reconstruct_cache_data(
            arrays, metadata, 1, ["TurboQuantQSAKVCache"]
        )
        assert data is not None and len(data) == 1
        tag, payload = data[0]
        assert tag == "__turboquant_qsa_v1__"
        rks, rvs, rik, rip = payload
        assert mx.array_equal(rks.norms, ks.norms)
        assert mx.array_equal(rks.indices, ks.indices)
        assert mx.array_equal(rvs.norms, vs.norms)
        assert mx.array_equal(rik, ik)
        assert mx.array_equal(rip, ip)
    finally:
        mgr.close()


def _scheduler_method_stub(**extra):
    """_SchedulerStub carrying the real Scheduler helper methods as attrs."""
    import omlx.scheduler as sched

    attrs = {
        "_cache_layer_is_reconstructible": (
            sched.Scheduler._cache_layer_is_reconstructible
        ),
        "_prefix_reuse_supports_cache_class": (
            sched.Scheduler._prefix_reuse_supports_cache_class
        ),
        "_turboquant_eligible": sched.Scheduler._turboquant_eligible,
        "_EXTRA_RECONSTRUCTIBLE_CACHE_TYPES": (
            sched.Scheduler._EXTRA_RECONSTRUCTIBLE_CACHE_TYPES
        ),
    }
    attrs.update(extra)
    return type("_SchedStub", (_SchedulerStub,), attrs)()


def test_signature_inference_names_hybrid_layers(monkeypatch):
    from types import SimpleNamespace

    import omlx.scheduler as sched

    caches = _make_prompt_cache()
    monkeypatch.setattr(sched, "make_prompt_cache", lambda model: list(caches))
    stub = _scheduler_method_stub()
    stub.model = object()
    stub.config = SimpleNamespace(model_name="m")
    types, bits, _sub = sched.Scheduler._infer_live_layer_cache_types(stub)
    assert types == [
        "TurboQuantQSAKVCache",
        "ArraysCache",
        "TurboQuantQSAKVCache",
        "QSAKVCache",
    ]
    assert bits == 4.0


def test_store_gate_allows_hybrid_conversion(monkeypatch):
    """Phase 2: hybrid chains are reconstructible — storage re-enabled."""
    import omlx.scheduler as sched

    caches = _make_prompt_cache()
    monkeypatch.setattr(sched, "make_prompt_cache", lambda model: list(caches))
    stub = _scheduler_method_stub()
    stub.model = object()
    assert sched.Scheduler._model_has_unreconstructible_cache(stub) is False


def test_prefix_reuse_supports_hybrid_name():
    stub = _scheduler_method_stub()
    assert stub._prefix_reuse_supports_cache_class("TurboQuantQSAKVCache") is True


def test_layer_scanner_detects_qsa_payload():
    from omlx.cache.prefix_cache import BlockAwarePrefixCache

    qsa_blocks = [[("__turboquant_qsa_v1__", (None, None, None, None))]]
    tq_blocks = [[("__turboquant_v2__", (None, None))]]
    assert BlockAwarePrefixCache._layer_has_tq_qsa_payload(qsa_blocks, 0) is True
    assert BlockAwarePrefixCache._layer_has_tq_qsa_payload(tq_blocks, 0) is False


def test_qsa_index_positions_concat_promotes_mixed_ranks():
    """A dedup'd chain mixing position ranks must still reconstruct.

    Blocks capture ``index_position_ids`` at whatever rank the live cache
    held: plain text stores ``(B, T)`` while the mRoPE-shaped capture
    stores the same positions replicated across the three channels as
    ``(3, B, T)``. Block dedup chains blocks captured under different
    ranks, and concatenating those raised a rank mismatch that rejected
    the whole prefix hit — so every later request in the lineage
    re-prefilled from scratch. Promotion replicates across channels,
    exactly what ``update_indexer`` does at runtime, and is lossless
    because text positions are channel-equal.
    """
    from omlx.turboquant_kv import _concat_qsa_index_positions

    text_block = mx.arange(4, dtype=mx.int32).reshape(1, 4)
    mrope_block = mx.broadcast_to(
        mx.arange(4, 8, dtype=mx.int32).reshape(1, 1, 4), (3, 1, 4)
    )
    expected = mx.arange(8, dtype=mx.int32).reshape(1, 8)

    cases = [
        ([text_block, mrope_block], [0, 1, 2, 3, 4, 5, 6, 7]),
        ([mrope_block, text_block], [4, 5, 6, 7, 0, 1, 2, 3]),
    ]
    for parts, tokens in cases:
        out = _concat_qsa_index_positions(parts)
        assert out.ndim == 3 and out.shape == (3, 1, 8)
        for channel in range(3):
            assert out[channel].tolist() == [tokens]

    # A homogeneous rank-2 chain stays rank-2: no gratuitous promotion.
    plain = _concat_qsa_index_positions([text_block, text_block])
    assert plain.ndim == 2 and plain.shape == (1, 8)
    assert plain.tolist() == [[0, 1, 2, 3, 0, 1, 2, 3]]


# ---------------------------------------------------------------------------
# Phase 2b: gathered prefill arm on the hybrid cache
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [4, 6])
def test_gathered_prefill_kernel_parity_within_codec_tolerance(bits):
    """TQ prefill kernel == dense kernel on identical selection inputs.

    Kernel-level with synthetic arrays and identity indexer norm/rope: no
    module forwards, so the test is immune to the pre-existing lazy-mutation
    hazard that corrupts back-to-back module forwards in one process (see
    the wiring test below). Selection is shared by construction — the packed
    rows must dequantize into the same SDPA the dense arm runs.
    """
    from mlx_vlm.models.qwen4_exp.qsa_fast import (
        contiguous_causal_gathered_qsa,
        contiguous_causal_gathered_qsa_tq,
        pool_completed_index_keys,
    )

    rng = _rng(5)
    heads, dim, idx_dim = 2, HEAD_DIM, IDX_DIM
    ratio, budget = 2, 8
    # max_blocks=16 > block_budget=4: sparse selection engages
    n_tokens, n_queries = 32, 8
    keys = mx.array(
        rng.standard_normal((1, KV_HEADS, n_tokens, dim)).astype(np.float32)
    )
    values = mx.array(
        rng.standard_normal((1, KV_HEADS, n_tokens, dim)).astype(np.float32)
    )
    queries = mx.array(
        rng.standard_normal((1, heads, n_queries, dim)).astype(np.float32)
    )
    index_queries = mx.array(
        rng.standard_normal((1, n_queries, 2, idx_dim)).astype(np.float32)
    )
    ik = mx.array(rng.standard_normal((1, n_tokens, idx_dim)).astype(np.float32))
    ip = mx.arange(n_tokens, dtype=mx.int32)[None, :]

    def ident_norm(x):
        return x

    def ident_rope(x, positions):
        return x

    pooled = pool_completed_index_keys(
        ik,
        ip,
        compress_ratio=ratio,
        index_key_norm=ident_norm,
        apply_index_rope=ident_rope,
    )

    dense_cache = QSAKVCache()
    dense_cache.state = (keys, values, ik, ip)
    hybrid = TurboQuantQSAKVCache.from_qsa_cache(dense_cache, bits=bits)

    kwargs = {
        "num_query_heads": heads,
        "num_key_value_heads": KV_HEADS,
        "head_dim": dim,
        "indexer_head_dim": idx_dim,
        "compress_ratio": ratio,
        "token_budget": budget,
        "index_key_norm": ident_norm,
        "apply_index_rope": ident_rope,
        "pooled_index_keys": pooled,
    }
    out_dense = contiguous_causal_gathered_qsa(
        queries, keys, values, index_queries, ik, ip, **kwargs
    )
    out_tq = contiguous_causal_gathered_qsa_tq(
        queries, hybrid, index_queries, ik, ip, **kwargs
    )

    assert out_tq.shape == out_dense.shape
    assert mx.isfinite(out_tq).all().item()
    assert _cosine(out_dense, out_tq) > 0.95
    rel = float(
        mx.linalg.norm(out_tq - out_dense) / mx.maximum(mx.linalg.norm(out_dense), 1e-6)
    )
    assert rel < 0.15, f"relative L2 {rel:.3f} exceeds codec-scale drift"


@pytest.mark.parametrize("bits", [4, 6])
def test_gathered_decode_kernel_parity_within_codec_tolerance(bits):
    """TQ decode arm == dense decode arm on identical selection inputs.

    Kernel-level with synthetic arrays and identity indexer norm/rope: no
    module forwards, so back-to-back parametrized runs stay immune to the
    lazy-mutation hazard that corrupts repeated module-forward sequences
    in one process (which is why the module-level decode test above may
    run exactly one numeric comparison per process).
    """
    from mlx_vlm.models.qwen4_exp.qsa_fast import (
        contiguous_causal_gathered_qsa_decode,
        contiguous_causal_gathered_qsa_decode_tq,
        pool_completed_index_keys,
    )

    rng = _rng(31)
    heads, dim, idx_dim = 2, HEAD_DIM, IDX_DIM
    ratio, budget = 2, 8
    # max_blocks=16 > block_budget=4: sparse selection engages
    n_tokens = 32
    keys = mx.array(
        rng.standard_normal((1, KV_HEADS, n_tokens, dim)).astype(np.float32)
    )
    values = mx.array(
        rng.standard_normal((1, KV_HEADS, n_tokens, dim)).astype(np.float32)
    )
    queries = mx.array(rng.standard_normal((1, heads, 1, dim)).astype(np.float32))
    index_queries = mx.array(rng.standard_normal((1, 1, 2, idx_dim)).astype(np.float32))
    ik = mx.array(rng.standard_normal((1, n_tokens, idx_dim)).astype(np.float32))
    ip = mx.arange(n_tokens, dtype=mx.int32)[None, :]

    def ident_norm(x):
        return x

    def ident_rope(x, positions):
        return x

    pooled = pool_completed_index_keys(
        ik,
        ip,
        compress_ratio=ratio,
        index_key_norm=ident_norm,
        apply_index_rope=ident_rope,
    )

    dense_cache = QSAKVCache()
    dense_cache.state = (keys, values, ik, ip)
    hybrid = TurboQuantQSAKVCache.from_qsa_cache(dense_cache, bits=bits)

    kwargs = {
        "num_query_heads": heads,
        "num_key_value_heads": KV_HEADS,
        "head_dim": dim,
        "indexer_head_dim": idx_dim,
        "compress_ratio": ratio,
        "token_budget": budget,
    }
    out_dense = contiguous_causal_gathered_qsa_decode(
        queries, keys, values, index_queries, pooled, **kwargs
    )
    out_tq = contiguous_causal_gathered_qsa_decode_tq(
        queries, hybrid, index_queries, pooled, **kwargs
    )

    assert out_tq.shape == out_dense.shape
    assert mx.isfinite(out_tq).all().item()
    assert _cosine(out_dense, out_tq) > 0.95
    rel = float(
        mx.linalg.norm(out_tq - out_dense) / mx.maximum(mx.linalg.norm(out_dense), 1e-6)
    )
    # The decode gather attends only budget + ratio - 1 rows (~11 here), so
    # 4-bit codec noise averages out less than in the prefill parity test
    # (measured rel-L2 0.164 at 4 bits, well under it at 6).
    assert rel < 0.25, f"relative L2 {rel:.3f} exceeds codec-scale drift"


# ---------------------------------------------------------------------------
# Native packed-row kernel: per-width numeric parity
# ---------------------------------------------------------------------------

NATIVE_Q_HEADS = 24
NATIVE_KV_HEADS = 2
NATIVE_DIM = 256
NATIVE_TOKENS = 512
NATIVE_ROWS = 32  # must be >= _native_main_min_rows() (default 24)

# Both arms consume the identical selection; the dense arm sees the
# dequantized rows, so any residual divergence is in-kernel rounding only
# (fp16/bf16 staging, norm fold points, log2-domain softmax) — no codec
# noise. Tolerances calibrated against the 4-bit kernel's measured rel-L2.
_NATIVE_TOL = {
    mx.float16: dict(cos=0.999, rel=2e-2),
    mx.bfloat16: dict(cos=0.999, rel=5e-2),
}


def _native_qsa_symbols_available() -> bool:
    from omlx.custom_kernels.glm_moe_dsa import fast

    try:
        return bool(
            fast.is_native_available()
            and fast.has_symbol("qwen4_qsa_sparse_gqa_attention_tq")
            and fast.has_symbol("qwen4_qsa_sparse_gqa_attention")
        )
    except Exception:
        return False


requires_native_qsa = pytest.mark.skipif(
    not _native_qsa_symbols_available(),
    reason="native QSA kernel extension is not built",
)


def _native_parity_case(bits: float, dtype: mx.Dtype):
    """TQ native arm vs dense native arm on an identical selection."""
    from mlx_vlm.models.qwen4_exp import qsa_fast as qf

    rng = _rng(2400 + int(bits * 2))
    n_tokens, n_rows = NATIVE_TOKENS, NATIVE_ROWS
    q_offset = n_tokens - n_rows
    keys = mx.array(
        rng.standard_normal((1, NATIVE_KV_HEADS, n_tokens, NATIVE_DIM)).astype(
            np.float32
        )
    )
    values = mx.array(
        rng.standard_normal((1, NATIVE_KV_HEADS, n_tokens, NATIVE_DIM)).astype(
            np.float32
        )
    )
    queries = mx.array(
        rng.standard_normal((1, NATIVE_Q_HEADS, n_rows, NATIVE_DIM)).astype(np.float32)
    ).astype(dtype)

    hybrid = TurboQuantQSAKVCache(bits=bits)
    hybrid.update_and_fetch(keys, values)

    # Chronological block IDs; blocks beyond the causal horizon expand to
    # candidates >= kL and are masked identically in both kernels.
    selected = mx.broadcast_to(
        mx.arange(512, dtype=mx.uint32).reshape(1, 1, 512), (1, n_rows, 512)
    )

    dk, dv = hybrid.dequantize()
    ref = qf._native_sparse_gqa_attention(
        queries, dk.astype(dtype), dv.astype(dtype), selected, q_offset=q_offset
    )
    out = qf._native_sparse_gqa_attention_tq(
        queries, hybrid, selected, q_offset=q_offset
    )
    assert ref is not None, "dense native reference arm unavailable"
    assert out is not None, f"native TQ arm rejected bits={bits}"

    a = out.astype(mx.float32)
    b = ref.astype(mx.float32)
    tol = _NATIVE_TOL[dtype]
    cos = _cosine(a, b)
    rel = float(mx.linalg.norm(a - b) / mx.maximum(mx.linalg.norm(b), 1e-6))
    assert cos > tol["cos"], f"bits={bits} cosine {cos:.6f} below {tol['cos']}"
    assert rel < tol["rel"], f"bits={bits} rel L2 {rel:.4f} exceeds {tol['rel']}"


@requires_native_qsa
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("bits", [2, 2.5, 3, 3.5, 4, 6, 8])
def test_native_tq_kernel_matches_dense_per_width(bits, dtype):
    _native_parity_case(bits, dtype)


@requires_native_qsa
@pytest.mark.parametrize("bits", [5, 7])
def test_native_tq_arm_fails_closed_on_uninstantiated_widths(bits):
    """Valid TurboQuant widths with no native instantiation must not dispatch."""
    from mlx_vlm.models.qwen4_exp import qsa_fast as qf

    rng = _rng(77)
    n_tokens, n_rows = 256, NATIVE_ROWS
    keys = mx.array(
        rng.standard_normal((1, NATIVE_KV_HEADS, n_tokens, NATIVE_DIM)).astype(
            np.float32
        )
    )
    values = mx.array(
        rng.standard_normal((1, NATIVE_KV_HEADS, n_tokens, NATIVE_DIM)).astype(
            np.float32
        )
    )
    queries = mx.array(
        rng.standard_normal((1, NATIVE_Q_HEADS, n_rows, NATIVE_DIM)).astype(np.float32)
    ).astype(mx.float16)
    hybrid = TurboQuantQSAKVCache(bits=bits)
    hybrid.update_and_fetch(keys, values)
    selected = mx.broadcast_to(
        mx.arange(512, dtype=mx.uint32).reshape(1, 1, 512), (1, n_rows, 512)
    )
    out = qf._native_sparse_gqa_attention_tq(
        queries, hybrid, selected, q_offset=n_tokens - n_rows
    )
    assert out is None


def test_native_qsa_tq_available_is_bits_aware():
    from mlx_vlm.models.qwen4_exp.qsa_fast import native_qsa_tq_available

    symbol_only = native_qsa_tq_available()
    for bits in (2, 2.5, 3, 3.5, 4, 6, 8):
        assert native_qsa_tq_available(bits) == symbol_only, bits
    for bits in (1, 4.5, 5, 6.5, 7, 16):
        assert native_qsa_tq_available(bits) is False, bits


def test_gathered_prefill_eligible_and_wired_on_hybrid(monkeypatch):
    """Hybrid prefill gate opens only inside the TQ crossover band.

    Below the floor the TQ-patched mask SDPA beats the portable gathered
    arm (measured crossover ≈16k on M5); above it the gate must open and
    a suffix prefill through the module must run the gathered TQ arm
    without crashing, staying packed through the append. Numerics are
    covered by the kernel parity test.
    """
    monkeypatch.setenv("OMLX_QWEN4_GATHERED_MIN_QUERY", "4")
    monkeypatch.delenv("OMLX_QWEN4_TQ_PREFILL_MIN_CONTEXT", raising=False)
    mx.random.seed(13)  # pin module init for stable weights across runs
    cfg = _text_config()
    attn = Qwen4ExpAttention(cfg)
    dense_cache = QSAKVCache()
    _prefill(attn, dense_cache, 24, seed=1)  # dense prefix past the budget
    hybrid_cache = TurboQuantQSAKVCache.from_qsa_cache(dense_cache, bits=BITS)

    rng = _rng(9)
    x = mx.array(rng.standard_normal((1, 8, 32)).astype(np.float32))
    positions = mx.arange(24, 32, dtype=mx.int32)[None, :]

    # Cost crossover: the dense gate opens at 32 tokens; the hybrid stays
    # on the mask arm until the default floor (16k).
    assert attn._gathered_text_prefill_eligible(
        x, None, dense_cache, positions, None, False
    )
    assert not attn._gathered_text_prefill_eligible(
        x, None, hybrid_cache, positions, None, False
    )

    monkeypatch.setenv("OMLX_QWEN4_TQ_PREFILL_MIN_CONTEXT", "16")
    assert attn._gathered_text_prefill_eligible(
        x, None, hybrid_cache, positions, None, False
    )

    # Above the band the tiled mask route takes over (750k ladder stall);
    # a 0 ceiling disables the cap. Pin the native-symbol probe so these
    # assertions hold on boxes with and without the built extension.
    import mlx_vlm.models.qwen4_exp.language as lang

    monkeypatch.setattr(lang, "native_qsa_tq_available", lambda *args: False)
    monkeypatch.setenv("OMLX_QWEN4_TQ_PREFILL_MAX_CONTEXT", "16")
    assert not attn._gathered_text_prefill_eligible(
        x, None, hybrid_cache, positions, None, False
    )

    # The native packed-row kernel lifts the ceiling (dense-class rate at
    # any context); boxes without the symbol keep the measured band.
    monkeypatch.setattr(lang, "native_qsa_tq_available", lambda *args: True)
    assert attn._gathered_text_prefill_eligible(
        x, None, hybrid_cache, positions, None, False
    )
    monkeypatch.setattr(lang, "native_qsa_tq_available", lambda *args: False)
    monkeypatch.setenv("OMLX_QWEN4_TQ_PREFILL_MAX_CONTEXT", "0")
    assert attn._gathered_text_prefill_eligible(
        x, None, hybrid_cache, positions, None, False
    )

    out = attn(x, cache=hybrid_cache, position_ids=positions)

    assert out.shape[0] == 1 and out.shape[1] == 8
    assert mx.isfinite(out).all().item()
    # The suffix rows appended while the cache stayed packed.
    assert isinstance(hybrid_cache.keys, TurboQuantMSEState)
    assert hybrid_cache.offset == 32
    assert hybrid_cache.index_keys.shape[1] == 32


# ---------------------------------------------------------------------------
# Phase 3: incremental prefill conversion (bounded dense transient)
# ---------------------------------------------------------------------------


def test_prefill_convert_interval_env(monkeypatch):
    import omlx.scheduler as sched

    monkeypatch.delenv("OMLX_TQ_PREFILL_CONVERT_INTERVAL", raising=False)
    assert sched._tq_prefill_convert_interval() == 0
    monkeypatch.setenv("OMLX_TQ_PREFILL_CONVERT_INTERVAL", "65536")
    assert sched._tq_prefill_convert_interval() == 65536
    monkeypatch.setenv("OMLX_TQ_PREFILL_CONVERT_INTERVAL", "0")
    assert sched._tq_prefill_convert_interval() == 0
    monkeypatch.setenv("OMLX_TQ_PREFILL_CONVERT_INTERVAL", "garbage")
    assert sched._tq_prefill_convert_interval() == 0


def test_incremental_convert_idempotent_and_sidecar_safe():
    """The mid-prefill trigger relies on convert being re-entrant.

    A second pass over already-hybrid layers must be a no-op (identical
    objects, sidecar untouched), and the first pass must preserve sidecar
    content and alignment — the gathered TQ arm keeps selecting from it
    while later chunks append packed rows.
    """
    import omlx.scheduler as sched

    stub = _scheduler_method_stub()
    caches = _make_prompt_cache()  # [QSA, Arrays, QSA, QSA(skip-last)]
    dense0 = caches[0]
    rng = _rng(3)
    dense0.update_and_fetch(
        mx.array(rng.standard_normal((1, KV_HEADS, 6, HEAD_DIM)).astype(np.float32)),
        mx.array(rng.standard_normal((1, KV_HEADS, 6, HEAD_DIM)).astype(np.float32)),
    )
    ik = mx.array(rng.standard_normal((1, 6, IDX_DIM)).astype(np.float32))
    dense0.update_indexer(ik, mx.arange(6, dtype=mx.int32)[None, :])
    ik_before = dense0.index_keys

    sched.Scheduler._apply_turboquant_kv_convert(stub, caches)
    assert type(caches[0]).__name__ == "TurboQuantQSAKVCache"
    assert type(caches[2]).__name__ == "TurboQuantQSAKVCache"
    assert type(caches[3]).__name__ == "QSAKVCache"  # skip-last stays dense
    assert mx.array_equal(caches[0].index_keys, ik_before).item()
    assert caches[0].offset == 6

    snapshot = list(caches)
    sched.Scheduler._apply_turboquant_kv_convert(stub, caches)
    assert all(a is b for a, b in zip(snapshot, caches))
    assert mx.array_equal(caches[0].index_keys, ik_before).item()


def test_incremental_trigger_folds_at_interval(monkeypatch):
    """Threshold bookkeeping: fold when due, advance, then no-op."""
    import omlx.scheduler as sched

    monkeypatch.setenv("OMLX_TQ_PREFILL_CONVERT_INTERVAL", "131072")
    stub = _scheduler_method_stub(
        _apply_turboquant_kv_convert=(sched.Scheduler._apply_turboquant_kv_convert),
        _reserve_qsa_index_capacity=sched.Scheduler._reserve_qsa_index_capacity,
        _tq_pressure_fold_due=sched.Scheduler._tq_pressure_fold_due,
        _tq_convertible_qsa_layers=sched.Scheduler._tq_convertible_qsa_layers,
    )
    caches = _make_prompt_cache()

    next_at = sched.Scheduler._tq_incremental_convert(stub, caches, 100, 131072, 999)
    assert next_at == 131072
    assert type(caches[0]).__name__ == "QSAKVCache"  # not due yet

    next_at = sched.Scheduler._tq_incremental_convert(
        stub, caches, 131072, next_at, 999
    )
    assert next_at == 262144
    assert type(caches[0]).__name__ == "TurboQuantQSAKVCache"

    next_at = sched.Scheduler._tq_incremental_convert(
        stub, caches, 262144, next_at, 999
    )
    assert next_at == 393216  # crossing still advances; convert is a no-op


def test_incremental_trigger_disabled(monkeypatch):
    import omlx.scheduler as sched

    monkeypatch.setenv("OMLX_TQ_PREFILL_CONVERT_INTERVAL", "0")
    stub = _scheduler_method_stub(
        _apply_turboquant_kv_convert=(sched.Scheduler._apply_turboquant_kv_convert),
        _tq_pressure_fold_due=sched.Scheduler._tq_pressure_fold_due,
        _tq_convertible_qsa_layers=sched.Scheduler._tq_convertible_qsa_layers,
    )
    caches = _make_prompt_cache()
    next_at = sched.Scheduler._tq_incremental_convert(stub, caches, 999999, 0, 999)
    assert next_at == 0
    assert type(caches[0]).__name__ == "QSAKVCache"


def test_fold_margin_env(monkeypatch):
    import omlx.scheduler as sched

    monkeypatch.delenv("OMLX_TQ_FOLD_MARGIN_GB", raising=False)
    assert sched._tq_fold_margin_bytes() == 4 * 1024**3
    monkeypatch.setenv("OMLX_TQ_FOLD_MARGIN_GB", "2.5")
    assert sched._tq_fold_margin_bytes() == int(2.5 * 1024**3)
    monkeypatch.setenv("OMLX_TQ_FOLD_MARGIN_GB", "junk")
    assert sched._tq_fold_margin_bytes() == 4 * 1024**3


def test_pressure_fold_fires_only_on_projected_breach():
    """Pressure fold: dense end-of-prefill projection vs guard headroom.

    The established contract keeps prefill fp16-exact; the fold is an
    emergency valve that must stay shut until the projection (usage +
    remaining × dense bytes/token) would breach hard-minus-margin even
    after pool reclaim, and must ignore non-QSA cache lists entirely.
    """
    import omlx.scheduler as sched

    stub = _scheduler_method_stub(
        _tq_pressure_fold_due=sched.Scheduler._tq_pressure_fold_due,
        _tq_convertible_qsa_layers=sched.Scheduler._tq_convertible_qsa_layers,
    )
    gib = 1024**3
    stub._memory_hard_limit_bytes = 100 * gib
    stub._tq_dense_kv_bytes_per_token = 28_000
    stub._current_usage_bytes = lambda **kw: 90 * gib
    stub._reclaim_prefill_headroom = lambda: 90 * gib
    caches = _make_prompt_cache()

    # 100k remaining × 28KB ≈ 2.6GiB → 92.6GB < 96GB (hard - 4GB margin)
    assert stub._tq_pressure_fold_due(caches, 100_000) is False
    # 400k remaining ≈ 10.4GiB → 100.4GB > 96GB → fold
    assert stub._tq_pressure_fold_due(caches, 400_000) is True
    # A reclaim that frees enough headroom keeps the contract intact.
    stub._reclaim_prefill_headroom = lambda: 80 * gib
    assert stub._tq_pressure_fold_due(caches, 400_000) is False
    # Non-QSA cache lists never fold (standard TQ models keep the
    # convert-at-end contract unconditionally).
    stub._reclaim_prefill_headroom = lambda: 90 * gib
    assert stub._tq_pressure_fold_due([ArraysCache(size=2)], 400_000) is False


def test_pressure_fold_prices_remaining_at_packed_rate_after_conversion():
    """After the first fold the projection must switch to the packed rate.

    Skip-last leaves one dense QSAKVCache behind, so the family gate
    stays open post-conversion; pricing the remaining tokens at the dense
    rate keeps the fold "due" on every subsequent chunk — the v11 750k
    run logged the same breach every ~0.65s for the rest of prefill and
    re-ran the (idempotent) conversion each time.
    """
    import omlx.scheduler as sched

    stub = _scheduler_method_stub(
        _tq_pressure_fold_due=sched.Scheduler._tq_pressure_fold_due,
        _tq_convertible_qsa_layers=sched.Scheduler._tq_convertible_qsa_layers,
    )
    gib = 1024**3
    stub._memory_hard_limit_bytes = 100 * gib
    stub._tq_dense_kv_bytes_per_token = 28_000
    stub._tq_packed_kv_bytes_per_token = 14_000
    stub._current_usage_bytes = lambda **kw: 90 * gib
    stub._reclaim_prefill_headroom = lambda: 90 * gib
    caches = _make_prompt_cache()
    # 400k remaining: dense projection ≈ 100.4GB > 96GB guard → due.
    assert stub._tq_pressure_fold_due(caches, 400_000) is True
    # Post-conversion (leading layers packed; skip-last keeps dense QSA
    # layers): packed projection ≈ 95.2GB < 96GB → not due.
    converted = [TurboQuantQSAKVCache(bits=BITS), *caches[1:]]
    assert stub._tq_pressure_fold_due(converted, 400_000) is False
    # The skip-last dense remainder is NOT convertible: with everything
    # else packed, the due-check must stop claiming the fold (the 1M run
    # re-reclaimed and re-ran the no-op conversion every ~0.6s chunk).
    stub._current_usage_bytes = lambda **kw: 100 * gib
    stub._reclaim_prefill_headroom = lambda: 100 * gib
    folded = [TurboQuantQSAKVCache(bits=BITS) for _ in range(3)]
    folded.append(QSAKVCache())
    assert stub._tq_pressure_fold_due(folded, 400_000) is False
    # A convertible dense layer with the same projection still folds.
    convertible = [TurboQuantQSAKVCache(bits=BITS), QSAKVCache(), QSAKVCache()]
    assert stub._tq_pressure_fold_due(convertible, 400_000) is True


def test_materialize_collects_packed_and_sidecar_arrays():
    """The post-fold materialization must reach every lazy backing array.

    Packed K/V are NamedTuples and the sidecar lives on extra attributes;
    a collector that only takes bare .keys/.values arrays silently skips
    them and the #3305-class fence hang returns.
    """
    from omlx.scheduler import _collect_cache_storage_arrays

    cache, _, _ = _dense_qsa(8)
    hybrid = TurboQuantQSAKVCache.from_qsa_cache(cache, bits=BITS)
    arrays = _collect_cache_storage_arrays(hybrid)
    # k/v packed states (norms + indices each) plus sidecar keys/positions.
    assert len(arrays) >= 6
    assert all(isinstance(a, mx.array) for a in arrays)
    # The sidecar is collected via the property views; evaling a view
    # materializes its backing buffer, so content coverage is the contract.
    assert any(mx.array_equal(a, hybrid.index_keys).item() for a in arrays)
    assert any(mx.array_equal(a, hybrid.index_position_ids).item() for a in arrays)


# ---------------------------------------------------------------------------
# Phase 2c: batched hybrid (BatchTurboQuantQSAKVCache)
# ---------------------------------------------------------------------------


def _hybrid_rows(lengths=(3, 5)):
    """Packed hybrid singletons with aligned sidecars (distinct content)."""
    rows = []
    for i, tokens in enumerate(lengths):
        cache, _, _ = _dense_qsa(tokens, seed=i)
        rows.append(TurboQuantQSAKVCache.from_qsa_cache(cache, bits=BITS))
    return rows


def test_batch_merge_extract_round_trips_packed_rows():
    from mlx_vlm.models.qwen4_exp.language import BatchTurboQuantQSAKVCache

    from omlx.turboquant_kv import BatchTurboQuantKVCache

    (row,) = _hybrid_rows((5,))
    batch = BatchTurboQuantQSAKVCache.merge([row])

    assert isinstance(batch, BatchTurboQuantQSAKVCache)
    assert isinstance(batch.kv_cache, BatchTurboQuantKVCache)
    assert batch.index_offset == 5
    assert batch.index_keys.shape[1] == 5

    out = batch.extract(0)
    assert isinstance(out, TurboQuantQSAKVCache)
    assert out.offset == 5
    # Rows stay packed through the round trip — no dense materialization.
    assert isinstance(out.keys, TurboQuantMSEState)
    # Sidecar rows survive exactly.
    assert mx.array_equal(out.index_keys, row.index_keys).item()
    assert mx.array_equal(out.index_position_ids, row.index_position_ids).item()
    # Packed KV content survives at codec scale (same codecs, same rows).
    k, v = out.dequantize()
    k0, v0 = row.dequantize()
    assert _cosine(k, k0) > 0.99
    assert _cosine(v, v0) > 0.99


def test_to_batch_keeps_hybrid_packed():
    from mlx_vlm.models.qwen4_exp.language import BatchTurboQuantQSAKVCache

    (row,) = _hybrid_rows((4,))
    batch = row.to_batch([0])
    assert isinstance(batch, BatchTurboQuantQSAKVCache)
    # Still packed — the phase-1 dequantizing fallback is gone.
    assert isinstance(batch.kv_cache.keys, TurboQuantMSEState)
    assert batch.index_offset == 4
    assert mx.array_equal(batch.index_keys, row.index_keys).item()
    # The patched SDPA unwraps proxies via `_cache` to reach the TQ batch.
    assert batch._cache is batch.kv_cache

    (row2,) = _hybrid_rows((4,))
    padded = row2.to_batch([3])
    assert padded.index_keys.shape[1] == 7
    assert padded.index_offset == 7
    assert padded.kv_cache._phys_end == 7
    assert int(padded.kv_cache.offset) == 4  # logical tokens
    out = padded.extract(0)
    assert out.offset == 4
    assert mx.array_equal(out.index_keys, row2.index_keys).item()


def test_scheduler_batch_join_keeps_hybrid_packed():
    """The scheduler's singleton->batch flow must not dequantize the hybrid."""
    import importlib

    from mlx_vlm.models.qwen4_exp.language import BatchTurboQuantQSAKVCache

    import omlx.scheduler  # noqa: F401  (installs BatchGenerator cache patches)

    (row,) = _hybrid_rows((3,))

    class Model:
        layers = (object(),)

        def make_cache(self):
            return [row]

    generate = importlib.import_module("mlx_lm.generate")
    caches = [
        omlx.scheduler._to_batched_cache_layer(c)
        for c in generate._merge_caches([Model().make_cache()])
    ]

    assert len(caches) == 1
    assert isinstance(caches[0], BatchTurboQuantQSAKVCache)
    assert isinstance(caches[0].kv_cache.keys, TurboQuantMSEState)
    assert caches[0].index_offset == 3


def test_batch_multirow_merge_falls_back_dense(monkeypatch):
    """Multi-row joins dequantize (interim); the flag restores packed."""
    from mlx_vlm.models.qwen4_exp.language import (
        BatchQSAKVCache,
        BatchTurboQuantQSAKVCache,
    )

    monkeypatch.delenv("OMLX_TQ_QSA_BATCH_ROWS", raising=False)
    row_a, row_b = _hybrid_rows((3, 5))
    batch = BatchTurboQuantQSAKVCache.merge([row_a, row_b])
    assert isinstance(batch, BatchQSAKVCache)
    assert not isinstance(batch, BatchTurboQuantQSAKVCache)
    assert batch.index_offset == 5
    assert batch.index_keys.shape[0] == 2

    monkeypatch.setenv("OMLX_TQ_QSA_BATCH_ROWS", "1")
    row_a, row_b = _hybrid_rows((3, 5))
    packed = BatchTurboQuantQSAKVCache.merge([row_a, row_b])
    assert isinstance(packed, BatchTurboQuantQSAKVCache)
    out_a = packed.extract(0)
    out_b = packed.extract(1)
    assert out_a.offset == 3 and out_b.offset == 5
    assert isinstance(out_a.keys, TurboQuantMSEState)
    # Row a sits behind its 2-column pad; the sidecar slice strips it.
    assert mx.array_equal(out_a.index_keys, row_a.index_keys).item()
    assert mx.array_equal(out_b.index_keys, row_b.index_keys).item()
    ka, _ = out_a.dequantize()
    ka0, _ = row_a.dequantize()
    assert _cosine(ka, ka0) > 0.99


def test_batch_finalize_rolls_packed_like_dense():
    """MTP rollback finalize must match the dense parent exactly."""

    (row,) = _hybrid_rows((6,))
    dense_row = row._to_dense_singleton()

    packed_batch = row.to_batch([0])
    dense_batch = dense_row.to_batch([0])
    for batch in (packed_batch, dense_batch):
        batch.kv_cache.prepare(right_padding=[2])
        batch.finalize()

    assert packed_batch.kv_cache._right_padding is None
    assert int(packed_batch.kv_cache.left_padding[0]) == 2
    p_out = packed_batch.extract(0)
    d_out = dense_batch.extract(0)
    assert p_out.offset == d_out.offset
    pk, pv = p_out.dequantize()
    assert pk.shape == d_out.keys.shape
    assert mx.allclose(pk, d_out.keys.astype(pk.dtype), atol=2e-2, rtol=2e-2).item()
    assert mx.allclose(pv, d_out.values.astype(pv.dtype), atol=2e-2, rtol=2e-2).item()
    assert mx.array_equal(p_out.index_keys, d_out.index_keys).item()


def test_batch_trim_keeps_sidecar_aligned():

    (row,) = _hybrid_rows((6,))
    batch = row.to_batch([0])
    assert batch.trim(2) == 2
    assert batch.index_offset == 4
    out = batch.extract(0)
    assert out.offset == 4
    assert mx.array_equal(out.index_keys, row.index_keys[:, :4]).item()


def test_batch_trim_rewinds_packed_rows_like_dense(monkeypatch):
    """B>1 MTP rollback trim must match the dense parent, scalar in/out.

    _trim_append_caches rewinds a uniform-acceptance verify block with a
    scalar trim(n). The packed B>1 batch keeps per-row array offsets plus
    the shared _phys_end cursor, and the inherited singleton trim dies on
    min(array, int) — the late-join crash that kept concurrent TQ-QSA
    requests serialized at admission.
    """
    from mlx_vlm.models.qwen4_exp.language import (
        BatchQSAKVCache,
        BatchTurboQuantQSAKVCache,
    )

    from omlx.turboquant_kv import BatchTurboQuantKVCache

    monkeypatch.setenv("OMLX_TQ_QSA_BATCH_ROWS", "1")
    row_a, row_b = _hybrid_rows((4, 6))
    dense_a = row_a._to_dense_singleton()
    dense_b = row_b._to_dense_singleton()
    packed = BatchTurboQuantQSAKVCache.merge([row_a, row_b])
    dense = BatchQSAKVCache.merge([dense_a, dense_b])
    assert isinstance(packed.kv_cache, BatchTurboQuantKVCache)

    rng = _rng(17)

    def _step():
        k = mx.array(rng.standard_normal((2, KV_HEADS, 1, HEAD_DIM)).astype(np.float32))
        v = mx.array(rng.standard_normal((2, KV_HEADS, 1, HEAD_DIM)).astype(np.float32))
        return k, v

    k1, v1 = _step()
    for batch in (packed, dense):
        batch.kv_cache.update_and_fetch(k1, v1)
    assert packed.kv_cache._phys_end == dense.kv_cache._idx == 7

    # The rollback contract: scalar in, scalar out, uniform rewind.
    assert packed.trim(1) == 1
    assert dense.trim(1) == 1
    assert mx.array_equal(packed.kv_cache.offset, dense.kv_cache.offset).item()
    assert packed.kv_cache._phys_end == dense.kv_cache._idx == 6
    assert packed.index_offset == dense.index_offset == 5

    # The next append lands at the rewound end: the rolled-back column is
    # overwritten and the retained prefix dequantizes at codec scale.
    k2, v2 = _step()
    for batch in (packed, dense):
        batch.kv_cache.update_and_fetch(k2, v2)
    assert packed.kv_cache._phys_end == dense.kv_cache._idx == 7
    for idx in (0, 1):
        p = packed.extract(idx)
        d = dense.extract(idx)
        assert p.offset == d.offset
        kp, vp = p.dequantize()
        assert _cosine(kp, d.keys) > 0.99
        assert _cosine(vp, d.values) > 0.99


def test_batch_extend_falls_back_dense(monkeypatch):
    from omlx.turboquant_kv import BatchTurboQuantKVCache

    monkeypatch.delenv("OMLX_TQ_QSA_BATCH_ROWS", raising=False)
    (r1,) = _hybrid_rows((3,))
    (r2,) = _hybrid_rows((5,))
    b1 = r1.to_batch([0])
    b2 = r2.to_batch([0])
    b1.extend(b2)
    assert not isinstance(b1.kv_cache, BatchTurboQuantKVCache)
    assert b1.index_keys.shape[0] == 2
    assert b1.index_offset == 5


def test_batch_extend_joins_live_packed_batch(monkeypatch):
    """A late prefill joining a live packed batch must not scalar-convert
    the per-row offsets.

    BatchQSAKVCache.extend's alignment check calls kv_cache.size(); the
    inherited singleton size() returns self.offset, which is a per-row
    array at B>1 — the ValueError that killed all four streams when a
    fourth concurrent session extended the running generation batch.
    """
    from mlx_vlm.models.qwen4_exp.language import BatchTurboQuantQSAKVCache

    from omlx.turboquant_kv import BatchTurboQuantKVCache

    monkeypatch.setenv("OMLX_TQ_QSA_BATCH_ROWS", "1")
    row_a, row_b = _hybrid_rows((3, 5))
    live = BatchTurboQuantQSAKVCache.merge([row_a, row_b])
    (row_c,) = _hybrid_rows((4,))
    joiner = row_c.to_batch([0])

    live.extend(joiner)

    assert isinstance(live.kv_cache, BatchTurboQuantKVCache)
    outs = [live.extract(i) for i in range(3)]
    assert [o.offset for o in outs] == [3, 5, 4]
    kc, _ = outs[2].dequantize()
    kc0, _ = row_c.dequantize()
    assert _cosine(kc, kc0) > 0.99
    assert mx.array_equal(outs[2].index_keys, row_c.index_keys).item()


def test_tq_qsa_serialized_decode_predicate(monkeypatch):
    import omlx.scheduler as sched

    monkeypatch.delenv("OMLX_TQ_QSA_BATCH_ROWS", raising=False)
    stub = _scheduler_method_stub()
    stub.model = object()
    caches = _make_prompt_cache()
    monkeypatch.setattr(sched, "make_prompt_cache", lambda model: list(caches))
    assert sched.Scheduler._tq_qsa_serialized_decode(stub) is True
    # The debug flag restores concurrent admissions...
    monkeypatch.setenv("OMLX_TQ_QSA_BATCH_ROWS", "1")
    assert sched.Scheduler._tq_qsa_serialized_decode(stub) is False
    # ...and non-TQ engines never serialize.
    stub2 = _scheduler_method_stub()
    stub2._turboquant_kv_bits = None
    stub2.model = object()
    assert sched.Scheduler._tq_qsa_serialized_decode(stub2) is False


def test_decode_key_chunk_capped_on_hybrid_caches():
    """Masked decode over the hybrid must stay chunked.

    The inherited 1<<30 default runs the whole context as one chunk and
    its weighted-sum value expansion allocates an (B, heads, 1, T, D)
    fp32 intermediate — ~19 GB at 767k tokens (the v9 ladder's
    decode-start abort three seconds after a flat packed insert).
    """
    from mlx_vlm.models.qwen4_exp.language import BatchTurboQuantQSAKVCache

    assert TurboQuantQSAKVCache.decode_key_chunk_size == 32768
    (row,) = _hybrid_rows((4,))
    batch = row.to_batch([0])
    assert batch.kv_cache.decode_key_chunk_size == 32768
    merged = BatchTurboQuantQSAKVCache.merge([row])
    assert merged.kv_cache.decode_key_chunk_size == 32768


# ---------------------------------------------------------------------------
# B=1 packed-batch gathered arms (decode + Lightning MTP verify)
# ---------------------------------------------------------------------------


def _boom_dequantize(*args, **kwargs):
    raise AssertionError("gathered dispatch must never dequantize the whole cache")


def _b1_batch(attn, tokens: int = 24, seed: int = 1):
    """Dense reference, hybrid singleton, and its pad-free B=1 batch."""
    dense = QSAKVCache()
    mx.eval(_prefill(attn, dense, tokens, seed=seed))
    hybrid = TurboQuantQSAKVCache.from_qsa_cache(dense, bits=BITS)
    # Materialize the packed states before any further module forward:
    # pending lazy chains over the shared attn corrupt numerics (the
    # documented dense-vs-dense hazard; production evaluates every step).
    mx.eval(
        hybrid.keys.norms,
        hybrid.keys.indices,
        hybrid.values.norms,
        hybrid.values.indices,
        hybrid.index_keys,
        hybrid.index_position_ids,
    )
    return dense, hybrid, hybrid.to_batch([0])


def test_gathered_decode_eligibility_on_b1_packed_batch(monkeypatch):
    """Pad-free B=1 packed batches join the singleton gathered decode arm.

    Left-padded rows (physical/logical divergence) and multi-row batches
    must stay on the official masked path.
    """
    mx.random.seed(21)  # pin module init (see sibling tests)
    attn = Qwen4ExpAttention(_text_config())
    _, hybrid, batch = _b1_batch(attn)
    x = mx.array(_rng(7).standard_normal((1, 1, 32)).astype(np.float32))

    assert attn._gathered_text_decode_eligible(x, None, batch, None, None, False)

    padded = hybrid.to_batch([3])
    assert not attn._gathered_text_decode_eligible(x, None, padded, None, None, False)

    dense2 = QSAKVCache()
    _prefill(attn, dense2, 24, seed=5)  # same provenance as _b1_batch's row
    other = TurboQuantQSAKVCache.from_qsa_cache(dense2, bits=BITS)
    monkeypatch.setenv("OMLX_TQ_QSA_BATCH_ROWS", "1")
    from mlx_vlm.models.qwen4_exp.language import BatchTurboQuantQSAKVCache

    merged = BatchTurboQuantQSAKVCache.merge([hybrid, other])
    assert not attn._gathered_text_decode_eligible(x, None, merged, None, None, False)


def test_gathered_batch_arms_match_dense_cache_level():
    """Batch/singleton gathered arms == dense arm on identical selections.

    Cache-level with synthetic arrays and identity indexer norm/rope: no
    module forwards, so the test is immune to the pre-existing lazy-mutation
    hazard (see the kernel parity test's docstring). Covers the decode arm
    and the multi-row Lightning-MTP verify arm on the B=1 packed batch,
    the incremental batch pooled-block bank, and pins whole-cache
    dequantize off — the 750k ladder's +17.7GB decode-start abort ran
    through verify falling to the official path's dequantize branch.
    """
    from mlx_vlm.models.qwen4_exp.qsa_fast import (
        contiguous_causal_gathered_qsa,
        contiguous_causal_gathered_qsa_decode,
        contiguous_causal_gathered_qsa_decode_tq,
        contiguous_causal_gathered_qsa_tq,
    )

    rng = _rng(17)
    heads, dim, idx_dim = 2, HEAD_DIM, IDX_DIM
    ratio, budget = 2, 8
    n_tokens = 32  # max_blocks=16 > block_budget=4: sparse selection engages
    keys = mx.array(
        rng.standard_normal((1, KV_HEADS, n_tokens, dim)).astype(np.float32)
    )
    values = mx.array(
        rng.standard_normal((1, KV_HEADS, n_tokens, dim)).astype(np.float32)
    )
    ik = mx.array(rng.standard_normal((1, n_tokens, idx_dim)).astype(np.float32))
    ip = mx.arange(n_tokens, dtype=mx.int32)[None, :]

    def ident_norm(x):
        return x

    def ident_rope(x, positions):
        return x

    dense_cache = QSAKVCache()
    dense_cache.state = (keys, values, ik, ip)
    hybrid = TurboQuantQSAKVCache.from_qsa_cache(dense_cache, bits=BITS)
    batch = hybrid.to_batch([0])
    batch.kv_cache.dequantize = _boom_dequantize
    hybrid.dequantize = _boom_dequantize

    kwargs = {
        "num_query_heads": heads,
        "num_key_value_heads": KV_HEADS,
        "head_dim": dim,
        "indexer_head_dim": idx_dim,
        "compress_ratio": ratio,
        "token_budget": budget,
    }
    pooled_dense = dense_cache.pooled_indexer_keys(ratio, ident_norm, ident_rope)
    pooled_batch = batch.pooled_indexer_keys(ratio, ident_norm, ident_rope)
    assert pooled_batch.shape == pooled_dense.shape
    assert _cosine(pooled_batch, pooled_dense) > 0.9999
    # Second call serves the incremental bank unchanged.
    assert (
        _cosine(batch.pooled_indexer_keys(ratio, ident_norm, ident_rope), pooled_dense)
        > 0.9999
    )

    # --- decode arm (q_len=1) ---
    queries = mx.array(rng.standard_normal((1, heads, 1, dim)).astype(np.float32))
    index_queries = mx.array(rng.standard_normal((1, 1, 2, idx_dim)).astype(np.float32))
    out_dense = contiguous_causal_gathered_qsa_decode(
        queries, keys, values, index_queries, pooled_dense, **kwargs
    )
    out_batch = contiguous_causal_gathered_qsa_decode_tq(
        queries, batch, index_queries, pooled_batch, **kwargs
    )
    mx.eval(out_dense, out_batch)
    assert out_batch.shape == out_dense.shape
    assert mx.isfinite(out_batch).all().item()
    assert _cosine(out_batch, out_dense) > 0.95

    # --- verify arm (multi-row, Lightning MTP q_len=4) ---
    queries4 = mx.array(rng.standard_normal((1, heads, 4, dim)).astype(np.float32))
    index_queries4 = mx.array(
        rng.standard_normal((1, 4, 2, idx_dim)).astype(np.float32)
    )
    prefill_kwargs = {
        **kwargs,
        "index_key_norm": ident_norm,
        "apply_index_rope": ident_rope,
    }
    out_dense4 = contiguous_causal_gathered_qsa(
        queries4,
        keys,
        values,
        index_queries4,
        ik,
        ip,
        pooled_index_keys=pooled_dense,
        **prefill_kwargs,
    )
    out_batch4 = contiguous_causal_gathered_qsa_tq(
        queries4,
        batch,
        index_queries4,
        ik,
        ip,
        pooled_index_keys=pooled_batch,
        **prefill_kwargs,
    )
    mx.eval(out_dense4, out_batch4)
    assert out_batch4.shape == out_dense4.shape
    assert mx.isfinite(out_batch4).all().item()
    assert _cosine(out_batch4, out_dense4) > 0.95

    out_single4 = contiguous_causal_gathered_qsa_tq(
        queries4,
        hybrid,
        index_queries4,
        ik,
        ip,
        pooled_index_keys=dense_cache.pooled_indexer_keys(
            ratio, ident_norm, ident_rope
        ),
        **prefill_kwargs,
    )
    mx.eval(out_single4)
    assert _cosine(out_single4, out_dense4) > 0.95


def test_gathered_batch_dispatch_wiring(monkeypatch):
    """Module dispatch routes B=1 packed batches into the gathered arms.

    Routing-only: the pre-existing module-forward hazard makes cross-run
    cosine comparisons flaky in-process, so numerics live in the cache-level
    parity test above. The pinned dequantize makes a routing regression —
    a fall-through to the official path — an explosion instead of a silent
    whole-cache dequantize per verify cycle.
    """
    import mlx_vlm.models.qwen4_exp.language as lang

    mx.random.seed(24)
    attn = Qwen4ExpAttention(_text_config())
    _, hybrid, batch = _b1_batch(attn, tokens=24, seed=1)

    decode_calls, verify_calls = [], []
    orig_decode = lang.contiguous_causal_gathered_qsa_decode_tq
    orig_verify = lang.contiguous_causal_gathered_qsa_tq

    def decode_spy(*args, **kwargs):
        decode_calls.append(args[1])
        return orig_decode(*args, **kwargs)

    def verify_spy(*args, **kwargs):
        verify_calls.append(args[1])
        return orig_verify(*args, **kwargs)

    monkeypatch.setattr(lang, "contiguous_causal_gathered_qsa_decode_tq", decode_spy)
    monkeypatch.setattr(lang, "contiguous_causal_gathered_qsa_tq", verify_spy)

    batch.kv_cache.dequantize = _boom_dequantize
    hybrid.dequantize = _boom_dequantize

    x1 = mx.array(_rng(7).standard_normal((1, 1, 32)).astype(np.float32))
    out_decode = attn(x1, cache=batch)
    mx.eval(out_decode)
    assert decode_calls == [batch]
    assert mx.isfinite(out_decode).all().item()
    assert isinstance(batch.kv_cache.keys, TurboQuantMSEState)
    assert batch.offset == 25
    assert batch.index_keys.shape[1] == 25

    x4 = mx.array(_rng(9).standard_normal((1, 4, 32)).astype(np.float32))
    positions = mx.arange(25, 29, dtype=mx.int32)[None, :]
    out_verify = attn(x4, cache=batch, position_ids=positions, target_verify=True)
    mx.eval(out_verify)
    assert verify_calls == [batch]
    assert mx.isfinite(out_verify).all().item()
    assert batch.offset == 29
    assert batch.index_keys.shape[1] == 29

    out_single = attn(x4, cache=hybrid, position_ids=positions, target_verify=True)
    mx.eval(out_single)
    assert verify_calls == [batch, hybrid]
    assert mx.isfinite(out_single).all().item()
    assert hybrid.offset == 28
    assert isinstance(hybrid.keys, TurboQuantMSEState)
