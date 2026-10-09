# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the GLM-5.3 DFlash2 target adapter."""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("dflash_mlx")

import mlx.nn as nn  # noqa: E402


def _fake_target(*, hidden_size=4096, vocab_size=154880, layers=45):
    inner = SimpleNamespace(
        layers=[SimpleNamespace() for _ in range(layers)],
        embed_tokens=object(),
        fa_idx=3,
        ssm_idx=0,
    )
    language_model = SimpleNamespace(
        args=SimpleNamespace(
            model_type="glm5_next_text",
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            mhc=True,
            hc_mult=4,
        ),
        model=inner,
    )
    return SimpleNamespace(model_type="glm5_next", language_model=language_model)


def _fake_draft(*, hidden_size=4096, vocab_size=154880, layers=45):
    args = SimpleNamespace(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_target_layers=layers,
    )
    return SimpleNamespace(
        args=args,
        is_dflash2=True,
        target_layer_ids=[5, 14, 24, 33, 42],
    )


def _draft_meta(architecture="DFlash2DraftModel"):
    return {"config": {"architectures": [architecture]}}


# ---------------------------------------------------------------------------
# Gate, pairing, capture contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_type", ["glm5_next", "glm5_next_text"])
def test_glm5_target_gate_accepts_both_config_spellings(tmp_path, model_type):
    from omlx.engine.dflash import is_dflash_compatible
    from omlx.patches.dflash_glm5 import is_glm5_dflash_target

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": model_type}), encoding="utf-8"
    )
    assert is_dflash_compatible(tmp_path) == (True, "")
    assert is_glm5_dflash_target(tmp_path) is True


def test_is_glm5_dflash_target_rejects_other_and_missing_configs(tmp_path):
    from omlx.patches.dflash_glm5 import is_glm5_dflash_target

    assert is_glm5_dflash_target(tmp_path) is False
    assert is_glm5_dflash_target(None) is False
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5"}), encoding="utf-8"
    )
    assert is_glm5_dflash_target(tmp_path) is False


def test_dflash2_pair_validation_accepts_published_geometry():
    from omlx.patches.dflash_glm5 import validate_glm5_dflash_pair

    validate_glm5_dflash_pair(_fake_target(), _fake_draft(), _draft_meta())


def test_dflash2_pair_validation_ignores_non_glm_targets():
    from omlx.patches.dflash_glm5 import validate_glm5_dflash_pair

    target = SimpleNamespace(model_type="qwen3_5")
    validate_glm5_dflash_pair(target, _fake_draft(), _draft_meta("DFlashDraftModel"))


@pytest.mark.parametrize(
    ("draft", "match"),
    [
        (_fake_draft(hidden_size=2048), "hidden_size mismatch"),
        (_fake_draft(vocab_size=32000), "vocab_size mismatch"),
        (_fake_draft(layers=44), "num_target_layers mismatch"),
    ],
)
def test_dflash2_pair_validation_rejects_geometry_mismatch(draft, match):
    from omlx.patches.dflash_glm5 import validate_glm5_dflash_pair

    with pytest.raises(ValueError, match=match):
        validate_glm5_dflash_pair(_fake_target(), draft, _draft_meta())


def test_dflash2_pair_validation_rejects_non_dflash2():
    from omlx.patches.dflash_glm5 import validate_glm5_dflash_pair

    with pytest.raises(ValueError, match="DFlash2DraftModel"):
        validate_glm5_dflash_pair(
            _fake_target(), _fake_draft(), _draft_meta("DFlashDraftModel")
        )


def test_mhc_capture_contract_is_the_stream_mean():
    from omlx.patches.dflash_glm5 import _contract_mhc_hidden

    hidden = mx.arange(48, dtype=mx.float32).reshape(1, 2, 4, 6)
    actual = _contract_mhc_hidden(hidden)
    expected = hidden.mean(axis=2)
    mx.eval(actual, expected)
    assert actual.shape == (1, 2, 6)
    assert mx.array_equal(actual, expected).item()
    flat = mx.zeros((1, 2, 6))
    assert _contract_mhc_hidden(flat) is flat
    with pytest.raises(ValueError, match="rank"):
        _contract_mhc_hidden(mx.zeros((2, 6)))


def test_hidden_extraction_maps_target_layer_k_to_capture_k_plus_one():
    from omlx.patches.dflash_glm5 import Glm5NextTargetOps

    captured = {
        6: mx.full((1, 2, 3), 6),
        15: mx.full((1, 2, 3), 15),
        25: mx.full((1, 2, 3), 25),
        34: mx.full((1, 2, 3), 34),
        43: mx.full((1, 2, 3), 43),
    }
    feature = Glm5NextTargetOps().extract_context_feature(captured, [5, 14, 24, 33, 42])
    mx.eval(feature)
    assert feature.shape == (1, 2, 15)
    assert feature[0, 0].tolist() == [6] * 3 + [15] * 3 + [25] * 3 + [34] * 3 + [43] * 3


def test_capabilities_fail_closed_for_unproven_paths():
    from omlx.patches.dflash_glm5 import Glm5NextTargetOps

    caps = Glm5NextTargetOps().capabilities_for(_fake_target())
    assert caps.supports_dflash is True
    assert caps.supports_recurrent_rollback is True
    assert caps.supports_kv_trim is True
    assert caps.supports_prefix_snapshot is False
    assert caps.supports_verify_linear is False
    assert caps.supports_tree_verify is False
    assert Glm5NextTargetOps().supports_tree_cache([]) is False
    with pytest.raises(NotImplementedError):
        Glm5NextTargetOps().verify_tree_block(
            target_model=None, tree_inputs=None, target_cache=[]
        )


def test_backend_registry_resolves_glm_and_not_qwen():
    from dflash_mlx.engine.target_ops import resolve_target_ops

    from omlx.patches.dflash_glm5 import (
        Glm5NextTargetOps,
        install_dflash_glm5_backend,
    )
    from omlx.patches.dflash_glm5 import _BACKEND_PATH
    from dflash_mlx.engine import target_ops

    install_dflash_glm5_backend()
    assert install_dflash_glm5_backend() is False
    assert target_ops.TARGET_BACKENDS.count(_BACKEND_PATH) == 1
    resolved = resolve_target_ops(_fake_target())
    assert isinstance(resolved, Glm5NextTargetOps)
    assert resolved.family(_fake_target()) == "glm5_next_kda_dsa"


# ---------------------------------------------------------------------------
# Forward capture (single chunk and chunked cold prefill)
# ---------------------------------------------------------------------------


class _RecordingLayer:
    def __init__(self, is_linear: bool, calls: list):
        self.is_linear = is_linear
        self._calls = calls

    def __call__(self, h, mask=None, cache=None, defer=False):
        self._calls.append((self.is_linear, int(h.shape[1]), mask is None))
        return h + 1.0


def _capture_target(hidden=8, vocab=12, hc_mult=2, calls=None):
    calls = [] if calls is None else calls
    embedding = nn.Embedding(vocab, hidden)
    layers = [
        _RecordingLayer(True, calls),
        _RecordingLayer(False, calls),
        _RecordingLayer(True, calls),
    ]
    inner = SimpleNamespace(
        layers=layers,
        embed_tokens=embedding,
        norm=lambda x: x * 0.5,
        fa_idx=1,
        ssm_idx=0,
        hc_mult=hc_mult,
    )
    wrapper = SimpleNamespace(
        model=inner,
        args=SimpleNamespace(tie_word_embeddings=False, model_type="glm5_next_text"),
        lm_head=nn.Linear(hidden, vocab, bias=False),
    )
    return SimpleNamespace(model_type="glm5_next", language_model=wrapper), calls


def _eval_captured(captured):
    values = list(captured.values()) if isinstance(captured, dict) else list(captured)
    mx.eval(*values)


def test_forward_capture_contracts_streams_and_keeps_last_logits():
    from omlx.patches.dflash_glm5 import Glm5NextTargetOps

    target, calls = _capture_target()
    ops = Glm5NextTargetOps()
    ids = mx.array([[1, 2, 3, 4, 5]], dtype=mx.uint32)
    logits, captured = ops.forward_with_hidden_capture(
        target,
        input_ids=ids,
        cache=[None, None, None],
        capture_layer_ids={0, 2},
        logits_last_only=True,
    )
    mx.eval(logits)
    _eval_captured(captured)
    assert logits.shape == (1, 1, 12)
    assert sorted(captured) == [-1, 0, 2]
    assert captured[0].shape == (1, 5, 8)
    assert captured[2].shape == (1, 5, 8)
    assert captured[-1].shape == (1, 5, 8)
    embedded = target.language_model.model.embed_tokens(ids)
    mx.eval(embedded)
    # Layer k output is the (stream-mean) embedding plus k residual steps.
    assert mx.allclose(captured[0], embedded).item()
    assert mx.allclose(captured[2], embedded + 2.0).item()
    assert calls == [(True, 5, True), (False, 5, False), (True, 5, True)]


def test_forward_capture_rejects_cache_length_mismatch():
    from omlx.patches.dflash_glm5 import Glm5NextTargetOps

    target, _ = _capture_target()
    with pytest.raises(ValueError, match="cache/layer count"):
        Glm5NextTargetOps().forward_with_hidden_capture(
            target,
            input_ids=mx.array([[1]], dtype=mx.uint32),
            cache=[None],
        )


@pytest.mark.parametrize("logits_last_only", [True, False])
def test_chunked_prefill_matches_single_forward(logits_last_only):
    from omlx.patches.dflash_glm5 import Glm5NextTargetOps

    target, calls = _capture_target()
    ids = mx.array([[list(range(1, 12))]], dtype=mx.uint32).reshape(1, 11)

    whole = Glm5NextTargetOps()
    whole.prefill_chunk_size = 64
    logits_ref, captured_ref = whole.forward_with_hidden_capture(
        target,
        input_ids=ids,
        cache=[None, None, None],
        capture_layer_ids={1, 3},
        logits_last_only=logits_last_only,
    )
    mx.eval(logits_ref)
    _eval_captured(captured_ref)
    calls.clear()

    chunked = Glm5NextTargetOps()
    chunked.prefill_chunk_size = 4
    logits, captured = chunked.forward_with_hidden_capture(
        target,
        input_ids=ids,
        cache=[None, None, None],
        capture_layer_ids={1, 3},
        logits_last_only=logits_last_only,
    )
    mx.eval(logits)
    _eval_captured(captured)

    widths = [width for _, width, _ in calls]
    assert widths == [4, 4, 4, 4, 4, 4, 3, 3, 3]
    assert logits.shape == logits_ref.shape
    assert mx.allclose(logits, logits_ref).item()
    assert sorted(captured) == sorted(captured_ref)
    for key in (1, 3):
        assert captured[key].shape == (1, 11, 8)
        assert mx.allclose(captured[key], captured_ref[key]).item()
    if logits_last_only:
        # The final-position hidden entry only carries the last chunk.
        assert captured[-1].shape == (1, 3, 8)


def test_chunked_prefill_capture_all_lists_every_layer():
    from omlx.patches.dflash_glm5 import Glm5NextTargetOps

    target, _ = _capture_target()
    ids = mx.arange(1, 10, dtype=mx.uint32).reshape(1, 9)
    ops = Glm5NextTargetOps()
    ops.prefill_chunk_size = 4
    logits, captured = ops.forward_with_hidden_capture(
        target, input_ids=ids, cache=[None, None, None], logits_last_only=True
    )
    mx.eval(logits)
    _eval_captured(captured)
    assert isinstance(captured, list)
    assert len(captured) == 4
    assert all(entry.shape == (1, 9, 8) for entry in captured)
    assert logits.shape == (1, 1, 12)


# ---------------------------------------------------------------------------
# Cache construction and rollback
# ---------------------------------------------------------------------------


def test_make_cache_replaces_linear_layer_caches_and_fails_closed():
    from dflash_mlx.recurrent_rollback_cache import RecurrentRollbackCache
    from mlx_lm.models.cache import ArraysCache, CacheList, KVCache

    from omlx.patches.dflash_glm5 import Glm5NextTargetOps
    from omlx.patches.glm53_kda_prework import glm53_kda_prefill_eligible

    layers = [
        SimpleNamespace(is_linear=True, self_attn=SimpleNamespace(conv_kernel_size=4)),
        SimpleNamespace(is_linear=False),
        SimpleNamespace(is_linear=True, self_attn=SimpleNamespace(conv_kernel_size=4)),
    ]
    inner = SimpleNamespace(layers=layers)
    made = [ArraysCache(size=2), CacheList(KVCache(), object()), ArraysCache(size=2)]
    wrapper = SimpleNamespace(model=inner, make_cache=lambda: list(made))
    target = SimpleNamespace(model_type="glm5_next", language_model=wrapper)

    ops = Glm5NextTargetOps()
    caches = ops.make_cache(target, enable_speculative_linear_cache=True)
    assert isinstance(caches[0], RecurrentRollbackCache)
    assert caches[0].conv_kernel_size == 4
    assert caches[1] is made[1]
    assert isinstance(caches[2], RecurrentRollbackCache)

    # Cold prefill takes the fused KDA path; armed verify windows do not.
    module = SimpleNamespace(
        conv_kernel_size=4, head_dim=128, num_heads=2, qkv_dim=256, conv_dim=768
    )
    inputs = mx.zeros((1, 64, 8), dtype=mx.bfloat16)
    assert glm53_kda_prefill_eligible(module, inputs, None, caches[0])
    caches[0].arm_rollback(prefix_len=0)
    assert not glm53_kda_prefill_eligible(module, inputs, None, caches[0])
    caches[0].clear_transients()
    assert glm53_kda_prefill_eligible(module, inputs, None, caches[0])

    with pytest.raises(ValueError, match="recurrent rollback"):
        ops.make_cache(target, enable_speculative_linear_cache=False)
    with pytest.raises(ValueError, match="KV quantization"):
        ops.make_cache(
            target, enable_speculative_linear_cache=True, quantize_kv_cache=True
        )
    with pytest.raises(ValueError, match="target_fa_window"):
        ops.make_cache(
            target, enable_speculative_linear_cache=True, target_fa_window=128
        )


def test_composite_dsa_cache_rollback_uses_kv_offset_and_checks_trim():
    from omlx.patches.dflash_glm5 import Glm5NextTargetOps

    class Composite:
        def __init__(self):
            self.kv = SimpleNamespace(offset=18)
            self.trimmed = 0

        def __getitem__(self, index):
            assert index == 0
            return self.kv

        def trim(self, count):
            self.trimmed += count
            self.kv.offset -= count
            return count

    cache = Composite()
    elapsed = Glm5NextTargetOps().restore_after_acceptance(
        [cache], target_len=15, acceptance_length=1, drafted_tokens=7
    )
    assert elapsed > 0
    assert cache.trimmed == 3
    assert cache.kv.offset == 15


def test_composite_dsa_cache_rollback_fails_closed_on_partial_trim():
    from omlx.patches.dflash_glm5 import Glm5NextTargetOps

    class Composite:
        def __getitem__(self, index):
            return SimpleNamespace(offset=18)

        def trim(self, count):
            return count - 1

    with pytest.raises(RuntimeError, match="rollback failed"):
        Glm5NextTargetOps().restore_after_acceptance(
            [Composite()], target_len=15, acceptance_length=1, drafted_tokens=7
        )


def _pooling_cache_list():
    from mlx_lm.models.cache import CacheList, KVCache

    from omlx.patches.deepseek_v4 import apply_pooling_cache_support

    apply_pooling_cache_support()
    from mlx_lm.models.cache import PoolingCache

    return CacheList(KVCache(), PoolingCache(4))


def _append_tokens(cache, tokens, *, offset):
    kv = mx.arange(offset * 3, (offset + tokens) * 3, dtype=mx.float32).reshape(
        1, tokens, 3
    )
    gate = mx.ones((1, tokens, 1), dtype=mx.float32)
    ready_kv, _ready_gate, _ = cache[1].accumulate_windows(kv, gate, offset)
    pooled = ready_kv[:, ::4] if ready_kv.shape[1] else ready_kv
    cache[1].update_and_fetch(pooled)
    keys = kv[:, None]
    values = mx.zeros((1, 1, tokens, 0), dtype=mx.float32)
    cache[0].update_and_fetch(keys, values)


def _assert_same_cache_list(actual, reference):
    """Compare the logical (offset-sliced) KV rows and the pooling state.

    Only the pooling fields GLM's indexer reads are compared: the pooled
    rows and the remainder buffer. ``prev_win_kv/gate`` is DeepSeek-V4
    overlap-compressor carry that ``trim`` repopulates and GLM never reads.
    """
    actual_kv = actual[0].keys_and_values()
    reference_kv = reference[0].keys_and_values()
    # state = (buf_kv[:remainder], buf_gate[:remainder], pooled, prev_kv, prev_gate)
    actual_pool = actual[1].state[:3]
    reference_pool = reference[1].state[:3]
    mx.eval(
        *actual_kv,
        *reference_kv,
        *[v for v in actual_pool if v is not None],
        *[v for v in reference_pool if v is not None],
    )
    assert actual[0].offset == reference[0].offset
    assert actual[1].remainder == reference[1].remainder
    for lhs, rhs in zip(actual_kv, reference_kv, strict=True):
        assert lhs.shape == rhs.shape
        assert mx.array_equal(lhs, rhs).item()
    for lhs, rhs in zip(actual_pool, reference_pool, strict=True):
        if lhs is None or rhs is None:
            assert lhs is rhs
        else:
            assert lhs.shape == rhs.shape
            assert mx.array_equal(lhs, rhs).item()


def test_actual_cache_list_rollback_crosses_pooling_boundary_exactly():
    from omlx.patches.dflash_glm5 import Glm5NextTargetOps

    actual = _pooling_cache_list()
    _append_tokens(actual, 3, offset=0)
    _append_tokens(actual, 4, offset=3)
    Glm5NextTargetOps().restore_after_acceptance(
        [actual], target_len=4, acceptance_length=0, drafted_tokens=3
    )

    reference = _pooling_cache_list()
    _append_tokens(reference, 3, offset=0)
    _append_tokens(reference, 1, offset=3)
    _assert_same_cache_list(actual, reference)


@pytest.mark.parametrize("accepted", [0, 1, 5, 9, 14])
def test_sixteen_token_verify_block_rolls_back_across_windows(accepted):
    """A 16-token DFlash block must be trimmable to any accepted prefix."""
    from omlx.patches.deepseek_v4.cache_extras import POOLING_UNDO_MAX_TOKENS
    from omlx.patches.dflash_glm5 import Glm5NextTargetOps

    assert POOLING_UNDO_MAX_TOKENS >= 16
    prefix = 3
    actual = _pooling_cache_list()
    _append_tokens(actual, prefix, offset=0)
    _append_tokens(actual, 16, offset=prefix)
    assert actual[1]._undo is not None
    target_len = prefix + 1 + accepted
    Glm5NextTargetOps().restore_after_acceptance(
        [actual], target_len=target_len, acceptance_length=accepted, drafted_tokens=15
    )
    assert actual[1]._undo is None

    reference = _pooling_cache_list()
    _append_tokens(reference, prefix, offset=0)
    _append_tokens(reference, 1 + accepted, offset=prefix)
    _assert_same_cache_list(actual, reference)


def test_verify_block_refuses_blocks_wider_than_the_pooling_undo_bound():
    from omlx.patches.deepseek_v4.cache_extras import POOLING_UNDO_MAX_TOKENS
    from omlx.patches.dflash_glm5 import Glm5NextTargetOps

    too_wide = mx.zeros((1, POOLING_UNDO_MAX_TOKENS + 1), dtype=mx.uint32)
    with pytest.raises(ValueError, match="dflash_block_size"):
        Glm5NextTargetOps().verify_block(
            target_model=object(), verify_ids=too_wide, target_cache=[]
        )
    with pytest.raises(ValueError, match="at least one token"):
        Glm5NextTargetOps().verify_block(
            target_model=object(),
            verify_ids=mx.zeros((1, 0), dtype=mx.uint32),
            target_cache=[],
        )


def test_verify_block_scopes_pooling_undo_gate_even_on_error(monkeypatch):
    from omlx.patches.dflash_glm5 import Glm5NextTargetOps
    from omlx.patches.mlx_lm_mtp import cache_rollback

    ops = Glm5NextTargetOps()

    def fail_while_armed(*args, **kwargs):
        assert cache_rollback._is_undo_armed() is True
        raise RuntimeError("verify failed")

    monkeypatch.setattr(ops, "forward_with_hidden_capture", fail_while_armed)
    with pytest.raises(RuntimeError, match="verify failed"):
        ops.verify_block(
            target_model=object(),
            verify_ids=mx.zeros((1, 2), dtype=mx.int32),
            target_cache=[],
        )
    assert cache_rollback._is_undo_armed() is False


def test_fully_accepted_cycle_clears_pooling_undo_without_changing_state():
    from omlx.patches.dflash_glm5 import Glm5NextTargetOps
    from omlx.patches.mlx_lm_mtp import cache_rollback

    cache = _pooling_cache_list()
    keys = mx.ones((1, 1, 1, 3), dtype=mx.float32)
    values = mx.zeros((1, 1, 1, 0), dtype=mx.float32)
    cache[0].update_and_fetch(keys, values)
    cache_rollback.set_undo_armed(True)
    try:
        kv = mx.ones((1, 1, 3), dtype=mx.float32)
        gate = mx.ones((1, 1, 1), dtype=mx.float32)
        cache[1].accumulate_windows(kv, gate, 0)
    finally:
        cache_rollback.set_undo_armed(False)
    kv_before = cache[0].keys_and_values()
    pooled_before = [v for v in cache[1].state if v is not None]
    mx.eval(*kv_before, *pooled_before)
    assert cache[1]._undo is not None

    Glm5NextTargetOps().restore_after_acceptance(
        [cache], target_len=1, acceptance_length=0, drafted_tokens=0
    )
    assert cache[1]._undo is None
    assert cache[1]._undo_chain is False
    assert cache[0].offset == 1
    for before, after in zip(kv_before, cache[0].keys_and_values(), strict=True):
        assert mx.array_equal(before, after).item()
    pooled_after = [v for v in cache[1].state if v is not None]
    for before, after in zip(pooled_before, pooled_after, strict=True):
        assert mx.array_equal(before, after).item()


def test_recurrent_rollback_requires_retained_verify_state():
    from dflash_mlx.recurrent_rollback_cache import RecurrentRollbackCache

    from omlx.patches.dflash_glm5 import Glm5NextTargetOps

    cache = RecurrentRollbackCache(size=2, conv_kernel_size=4)
    cache.arm_rollback(prefix_len=0)
    with pytest.raises(RuntimeError, match="rollback state is missing"):
        Glm5NextTargetOps().restore_after_acceptance(
            [cache], target_len=1, acceptance_length=0, drafted_tokens=3
        )
    assert cache._armed is False


def test_shared_lifecycle_restore_removes_glm_hook_without_qwen_backups():
    from omlx.patches.dflash_glm5 import _install_glm5_recurrent_hook
    from omlx.patches.dflash_lifecycle import restore_dflash_class_patches

    class Attention:
        def __call__(self, inputs, mask=None, cache=None):
            return inputs

    original = Attention.__call__
    _install_glm5_recurrent_hook(Attention())
    _install_glm5_recurrent_hook(Attention())
    assert Attention.__call__ is not original
    assert getattr(Attention.__call__, "_omlx_dflash_glm5", False)
    restore_dflash_class_patches()
    assert Attention.__call__ is original


def test_recurrent_verify_hook_matches_target_and_replays_accepted_prefix():
    """Replaying the accepted prefix natively must reproduce serial state."""
    from mlx_lm.models.cache import ArraysCache

    from omlx.patches.mlx_vlm_glm5_next_compat import (
        apply_mlx_vlm_glm5_next_compat_patch,
    )

    apply_mlx_vlm_glm5_next_compat_patch()
    from dflash_mlx.recurrent_rollback_cache import RecurrentRollbackCache
    from mlx_vlm.models.glm5_next.language import Glm5NextLinearAttention

    from omlx.patches.dflash_glm5 import (
        Glm5NextTargetOps,
        _install_glm5_recurrent_hook,
        restore_glm5_dflash_class_patches,
    )

    config = SimpleNamespace(
        hidden_size=64,
        linear_num_heads=2,
        linear_head_dim=32,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
        linear_lower_bound=-5.0,
    )
    mx.random.seed(7)
    attention = Glm5NextLinearAttention(config)
    prefix = mx.random.normal((1, 3, 64)).astype(mx.bfloat16)
    verify = mx.random.normal((1, 4, 64)).astype(mx.bfloat16)

    baseline_cache = ArraysCache(size=2)
    attention(prefix, cache=baseline_cache)
    expected = attention(verify, cache=baseline_cache)
    mx.eval(expected, *[v for v in baseline_cache.cache if v is not None])

    rollback_cache = RecurrentRollbackCache(size=2, conv_kernel_size=4)
    attention(prefix, cache=rollback_cache)
    mx.eval(*[v for v in rollback_cache.cache if v is not None])
    rollback_cache.arm_rollback(prefix_len=3)
    _install_glm5_recurrent_hook(attention)
    try:
        # Fully accepted block: state is kept, transients are cleared.
        full_cache = RecurrentRollbackCache(size=2, conv_kernel_size=4)
        attention(prefix, cache=full_cache)
        full_cache.arm_rollback(prefix_len=3)
        attention(verify, cache=full_cache)
        full_state = list(full_cache.cache)
        Glm5NextTargetOps().restore_after_acceptance(
            [full_cache], target_len=7, acceptance_length=3, drafted_tokens=3
        )
        mx.eval(
            *[v for v in full_cache.cache if v is not None],
            *[v for v in full_state if v is not None],
        )
        for retained, expected_retained in zip(
            full_cache.cache, full_state, strict=True
        ):
            assert mx.array_equal(retained, expected_retained).item()
        assert not hasattr(full_cache, "_omlx_glm5_verify")

        # The hook must not change the target output.
        actual = attention(verify, cache=rollback_cache)
        mx.eval(actual, *[v for v in rollback_cache.cache if v is not None])
        assert mx.array_equal(actual, expected).item()
        assert rollback_cache._omlx_glm5_verify is not None

        # acceptance_length=1 commits the target-owned first token plus one
        # accepted draft token; compare with a serial two-token run.
        Glm5NextTargetOps().restore_after_acceptance(
            [rollback_cache], target_len=5, acceptance_length=1, drafted_tokens=3
        )
        reference_cache = ArraysCache(size=2)
        attention(prefix, cache=reference_cache)
        attention(verify[:, :2], cache=reference_cache)
        mx.eval(
            *[v for v in rollback_cache.cache if v is not None],
            *[v for v in reference_cache.cache if v is not None],
        )
        for replayed, reference in zip(
            rollback_cache.cache, reference_cache.cache, strict=True
        ):
            assert mx.array_equal(replayed, reference).item()
        assert not hasattr(rollback_cache, "_omlx_glm5_verify")

        # A second rejection from the replayed state catches cumulative
        # drift that a one-cycle snapshot test would miss.
        verify_two = mx.random.normal((1, 4, 64)).astype(mx.bfloat16)
        baseline_two = ArraysCache(size=2)
        attention(prefix, cache=baseline_two)
        attention(verify[:, :2], cache=baseline_two)
        expected_two = attention(verify_two, cache=baseline_two)
        rollback_cache.arm_rollback(prefix_len=5)
        actual_two = attention(verify_two, cache=rollback_cache)
        mx.eval(actual_two, expected_two)
        assert mx.array_equal(actual_two, expected_two).item()
        Glm5NextTargetOps().restore_after_acceptance(
            [rollback_cache], target_len=6, acceptance_length=0, drafted_tokens=3
        )

        reference_two = ArraysCache(size=2)
        attention(prefix, cache=reference_two)
        attention(verify[:, :2], cache=reference_two)
        attention(verify_two[:, :1], cache=reference_two)
        mx.eval(
            *[v for v in rollback_cache.cache if v is not None],
            *[v for v in reference_two.cache if v is not None],
        )
        for replayed, reference in zip(
            rollback_cache.cache, reference_two.cache, strict=True
        ):
            assert mx.array_equal(replayed, reference).item()
    finally:
        restore_glm5_dflash_class_patches()


# ---------------------------------------------------------------------------
# Target loader
# ---------------------------------------------------------------------------


def _write_glm_config(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "glm5_next"}), encoding="utf-8"
    )


def _silence_processor_patches(monkeypatch):
    from omlx.engine import vlm as vlm_engine

    monkeypatch.setattr(vlm_engine, "_patch_video_processor_bug", lambda: None)
    monkeypatch.setattr(vlm_engine, "_patch_torch_free_image_processor", lambda: None)
    monkeypatch.setattr(vlm_engine, "apply_pixtral_torch_free_patch", lambda: False)


def test_glm_target_loader_prefers_omlx_custom_vlm_loader(tmp_path, monkeypatch):
    import mlx_vlm.utils as vlm_utils

    from omlx.patches import dflash_glm5
    from omlx.patches.dflash_glm5 import (
        install_dflash_glm5_backend,
        load_glm5_target_bundle,
    )
    from omlx.utils import model_loading

    _write_glm_config(tmp_path)
    _silence_processor_patches(monkeypatch)
    target = _fake_target()
    processor = SimpleNamespace(tokenizer=object())
    seen = []

    def custom_loader(model_ref, *, is_vlm):
        seen.append(("load", model_ref, is_vlm))
        return target, processor

    monkeypatch.setattr(model_loading, "maybe_load_custom_quantization", custom_loader)
    monkeypatch.setattr(
        model_loading,
        "materialize_lazy_state",
        lambda model: seen.append(("materialize", model)),
    )
    monkeypatch.setattr(
        dflash_glm5.Glm5NextTargetOps,
        "install_speculative_hooks",
        lambda self, model: seen.append(("hooks", model)),
    )
    monkeypatch.setattr(
        vlm_utils,
        "load",
        lambda *args, **kwargs: pytest.fail("plain mlx-vlm loader must not run"),
    )
    install_dflash_glm5_backend()
    bundle = load_glm5_target_bundle(tmp_path)
    assert bundle.model is target
    assert bundle.tokenizer is processor.tokenizer
    assert bundle.meta["config"] == {"model_type": "glm5_next"}
    assert bundle.meta["verify_linear_enabled"] is False
    assert bundle.target_ops.backend_name == "glm5_next"
    assert seen == [
        ("load", str(tmp_path), True),
        ("materialize", target),
        ("hooks", target),
    ]


def test_glm_target_loader_forces_eager_mlx_vlm_fallback(tmp_path, monkeypatch):
    import mlx_vlm.utils as vlm_utils

    from omlx.patches import dflash_glm5
    from omlx.patches.dflash_glm5 import load_glm5_target_bundle
    from omlx.utils import model_loading

    _write_glm_config(tmp_path)
    _silence_processor_patches(monkeypatch)
    target = _fake_target()
    processor = SimpleNamespace(tokenizer=object())
    seen = {}

    def vlm_loader(model_ref, **kwargs):
        seen.update(model_ref=model_ref, **kwargs)
        return target, processor

    monkeypatch.setattr(
        model_loading, "maybe_load_custom_quantization", lambda *a, **k: None
    )
    monkeypatch.setattr(model_loading, "materialize_lazy_state", lambda _model: None)
    monkeypatch.setattr(vlm_utils, "load", vlm_loader)
    monkeypatch.setattr(
        dflash_glm5.Glm5NextTargetOps,
        "install_speculative_hooks",
        lambda self, model: None,
    )

    bundle = load_glm5_target_bundle(tmp_path, lazy=True, trust_remote_code=True)
    assert bundle.model is target
    assert seen == {
        "model_ref": str(tmp_path),
        "lazy": False,
        "strict": True,
        "trust_remote_code": True,
    }


@pytest.mark.parametrize("use_custom_loader", [True, False])
def test_glm_target_loader_scopes_prequant_sanitize_around_both_loaders(
    tmp_path, monkeypatch, use_custom_loader
):
    import mlx_vlm.utils as vlm_utils

    from omlx.engine import vlm as vlm_engine
    from omlx.patches import dflash_glm5
    from omlx.patches.dflash_glm5 import (
        install_dflash_glm5_backend,
        load_glm5_target_bundle,
    )
    from omlx.utils import model_loading

    _write_glm_config(tmp_path)
    _silence_processor_patches(monkeypatch)
    target = _fake_target()
    processor = SimpleNamespace(tokenizer=object())
    events = []

    @contextmanager
    def sanitize_scope(model_dir):
        assert model_dir == tmp_path
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    def custom_loader(*_args, **_kwargs):
        events.append("custom")
        return (target, processor) if use_custom_loader else None

    def fallback_loader(*_args, **_kwargs):
        events.append("fallback")
        return target, processor

    monkeypatch.setattr(vlm_engine, "_force_qwen4_exp_sanitize_on_load", sanitize_scope)
    monkeypatch.setattr(model_loading, "maybe_load_custom_quantization", custom_loader)
    monkeypatch.setattr(
        model_loading,
        "materialize_lazy_state",
        lambda _model: events.append("materialize"),
    )
    monkeypatch.setattr(vlm_utils, "load", fallback_loader)
    monkeypatch.setattr(
        dflash_glm5.Glm5NextTargetOps,
        "install_speculative_hooks",
        lambda self, model: events.append("hooks"),
    )
    install_dflash_glm5_backend()

    load_glm5_target_bundle(tmp_path)
    load_events = ["custom"] if use_custom_loader else ["custom", "fallback"]
    assert events == ["enter", *load_events, "exit", "materialize", "hooks"]


def test_glm_target_loader_restores_prequant_sanitize_after_load_error(
    tmp_path, monkeypatch
):
    from omlx.engine import vlm as vlm_engine
    from omlx.patches.dflash_glm5 import load_glm5_target_bundle
    from omlx.utils import model_loading

    _write_glm_config(tmp_path)
    _silence_processor_patches(monkeypatch)
    events = []

    @contextmanager
    def sanitize_scope(_model_dir):
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    def fail_load(*_args, **_kwargs):
        events.append("load")
        raise RuntimeError("load failed")

    monkeypatch.setattr(vlm_engine, "_force_qwen4_exp_sanitize_on_load", sanitize_scope)
    monkeypatch.setattr(model_loading, "maybe_load_custom_quantization", fail_load)
    monkeypatch.setattr(
        model_loading,
        "materialize_lazy_state",
        lambda _model: pytest.fail("materialize must not run after load failure"),
    )

    with pytest.raises(RuntimeError, match="load failed"):
        load_glm5_target_bundle(tmp_path)
    assert events == ["enter", "load", "exit"]


def test_glm_target_loader_fails_before_hooks_if_materialization_fails(
    tmp_path, monkeypatch
):
    from omlx.patches import dflash_glm5
    from omlx.patches.dflash_glm5 import (
        install_dflash_glm5_backend,
        load_glm5_target_bundle,
    )
    from omlx.utils import model_loading

    _write_glm_config(tmp_path)
    _silence_processor_patches(monkeypatch)
    target = _fake_target()
    processor = SimpleNamespace(tokenizer=object())
    hooks = []

    def fail_materialize(_model):
        raise RuntimeError("materialize failed")

    monkeypatch.setattr(
        model_loading,
        "maybe_load_custom_quantization",
        lambda *_args, **_kwargs: (target, processor),
    )
    monkeypatch.setattr(model_loading, "materialize_lazy_state", fail_materialize)
    monkeypatch.setattr(
        dflash_glm5.Glm5NextTargetOps,
        "install_speculative_hooks",
        lambda self, model: hooks.append(model),
    )

    install_dflash_glm5_backend()
    with pytest.raises(RuntimeError, match="materialize failed"):
        load_glm5_target_bundle(tmp_path)
    assert hooks == []


def test_glm_target_loader_rejects_kv_quantization_and_foreign_configs(tmp_path):
    from omlx.patches.dflash_glm5 import load_glm5_target_bundle

    _write_glm_config(tmp_path)
    with pytest.raises(ValueError, match="KV quantization"):
        load_glm5_target_bundle(tmp_path, quantize_kv_cache=True)
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not a GLM-5.3 checkpoint"):
        load_glm5_target_bundle(tmp_path)


def test_glm_adapter_prefill_chunk_follows_scheduler_floor(monkeypatch):
    """DFlash GLM-5.3 prefill uses the batched scheduler's floor when wider."""
    import types

    import omlx.engine.dflash as dflash_engine
    from omlx.engine.dflash import _adapter_prefill_chunk

    glm = types.SimpleNamespace(backend_name="glm5_next")
    other = types.SimpleNamespace(backend_name="qwen3")
    monkeypatch.setattr(dflash_engine, "_glm5_next_prefill_floor", lambda: 4096)
    assert _adapter_prefill_chunk(glm, 2048) == 4096
    assert _adapter_prefill_chunk(glm, 8192) == 8192
    assert _adapter_prefill_chunk(other, 2048) == 2048
    monkeypatch.setattr(dflash_engine, "_glm5_next_prefill_floor", lambda: 0)
    assert _adapter_prefill_chunk(glm, 2048) == 2048


@pytest.mark.parametrize(
    "native,memory_gb,nax,nax_mla,expected",
    [
        (False, 512, False, False, 0),
        (True, 32, False, False, 0),
        (True, 512, False, False, 4096),
        (True, 256, True, True, 4096),
        (True, 256, True, False, 0),
    ],
)
def test_glm5_next_prefill_floor(
    monkeypatch, native, memory_gb, nax, nax_mla, expected
):
    from omlx import settings
    from omlx.custom_kernels import nax as nax_mod
    from omlx.custom_kernels.glm_moe_dsa import fast
    from omlx.patches.glm_moe_dsa import sparse_mla_nax
    from omlx.scheduler import _glm5_next_prefill_floor

    monkeypatch.setattr(fast, "is_native_available", lambda: native)
    monkeypatch.setattr(fast, "has_symbol", lambda name: native)
    monkeypatch.setattr(settings, "get_system_memory", lambda: memory_gb * 1024**3)
    monkeypatch.setattr(nax_mod, "is_nax_available", lambda: nax)
    monkeypatch.setattr(sparse_mla_nax, "nax_sparse_mla_available", lambda: nax_mla)
    assert _glm5_next_prefill_floor() == expected
