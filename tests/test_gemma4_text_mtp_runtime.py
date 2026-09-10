# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.patches.mlx_lm_mtp.gemma4_text_model.

Covers assistant-config retention through ``ModelArgs.from_dict``, head
attach gating, the two outputs the head consumes from a backbone forward
(per-layer-type K/V sink and the pre-norm hidden), head dtype alignment,
``sanitize``'s strip-or-keep branch, and depth-k partial rollback across
gemma4's mixed sliding/full cache classes.

Tiny configs throughout — real modules, no checkpoint.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

pytest.importorskip("mlx_lm.models.gemma4_text")

from omlx.patches import mlx_lm_mtp as lm_mtp  # noqa: E402
from omlx.patches.mlx_lm_mtp import gemma4_text_model  # noqa: E402

TINY_TEXT_CONFIG = {
    "model_type": "gemma4_text",
    "hidden_size": 24,
    "num_hidden_layers": 4,
    "intermediate_size": 32,
    "num_attention_heads": 2,
    "head_dim": 8,
    "global_head_dim": 8,
    "num_key_value_heads": 1,
    "num_global_key_value_heads": 1,
    "num_kv_shared_layers": 0,
    "vocab_size": 64,
    "sliding_window": 8,
    "sliding_window_pattern": 2,
    "attention_k_eq_v": True,
    "hidden_size_per_layer_input": 0,
    "use_double_wide_mlp": False,
    "tie_word_embeddings": True,
}

TINY_ASSISTANT_CONFIG = {
    "model_type": "gemma4_assistant",
    "backbone_hidden_size": 24,
    "tie_word_embeddings": True,
    "use_ordered_embeddings": False,
    "block_size": 4,
    "text_config": {
        "model_type": "gemma4_text",
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "intermediate_size": 32,
        "num_attention_heads": 2,
        "head_dim": 8,
        "global_head_dim": 8,
        "num_key_value_heads": 1,
        "num_global_key_value_heads": 1,
        "num_kv_shared_layers": 0,
        "vocab_size": 64,
        "sliding_window": 8,
        "sliding_window_pattern": 2,
        "attention_k_eq_v": True,
        "hidden_size_per_layer_input": 0,
        "use_double_wide_mlp": False,
    },
}


@pytest.fixture(autouse=True)
def _applied_patch():
    if not gemma4_text_model.apply():
        pytest.skip("mlx-lm gemma4 not importable")
    depth = lm_mtp.get_mtp_depth()
    lm_mtp.set_mtp_active(False)
    yield
    lm_mtp.set_mtp_active(False)
    lm_mtp.set_mtp_depth(depth)


def _args(extra: dict | None = None):
    from mlx_lm.models.gemma4_text import ModelArgs

    params = dict(TINY_TEXT_CONFIG)
    if extra:
        params.update(extra)
    return ModelArgs.from_dict(params)


def _inner(extra: dict | None = None):
    from mlx_lm.models.gemma4_text import Model

    return Model(_args(extra))


def _outer(extra: dict | None = None):
    from mlx_lm.models.gemma4 import Model, ModelArgs

    text = dict(TINY_TEXT_CONFIG)
    if extra:
        text.update(extra)
    return Model(
        ModelArgs.from_dict(
            {"model_type": "gemma4", "text_config": text, "vocab_size": 64}
        )
    )


def _with_assistant():
    return {"mtp_assistant_config": TINY_ASSISTANT_CONFIG}


# ---------------------------------------------------------------------------
# Patch application and config retention
# ---------------------------------------------------------------------------


def test_apply_is_idempotent():
    assert gemma4_text_model.apply()
    assert gemma4_text_model.apply()


def test_model_args_retains_assistant_config():
    # BaseModelArgs.from_dict keeps only dataclass fields, so the merged
    # assistant config needs carrying past it or the head cannot be sized.
    assert _args(_with_assistant()).mtp_assistant_config == TINY_ASSISTANT_CONFIG


def test_model_args_without_assistant_config_has_no_attribute():
    assert getattr(_args(), "mtp_assistant_config", None) is None


# ---------------------------------------------------------------------------
# Head attach gating
# ---------------------------------------------------------------------------


def test_no_attach_without_assistant_config():
    lm_mtp.set_mtp_active(True)
    model = _inner()
    assert getattr(model, "mtp", None) is None
    assert model._omlx_mtp_decode_enabled is False


def test_no_attach_when_mtp_inactive():
    # A merged checkpoint served with mtp_enabled=False keeps the stock
    # text path, head weights and all.
    model = _inner(_with_assistant())
    assert getattr(model, "mtp", None) is None
    assert model._omlx_mtp_decode_enabled is False


def test_attach_and_chain_flags_when_active():
    pytest.importorskip("mlx_vlm.speculative.drafters.gemma4_assistant")
    lm_mtp.set_mtp_active(True)
    lm_mtp.set_mtp_depth(3)
    model = _inner(_with_assistant())
    assert model.mtp is not None
    assert model._omlx_mtp_decode_enabled is True
    assert model._omlx_mtp_chain is True
    assert model._omlx_mtp_depth == 3
    # The head keeps no state of its own.
    assert model.make_mtp_cache() == []


# ---------------------------------------------------------------------------
# What a backbone forward has to hand the head
# ---------------------------------------------------------------------------


def test_plain_forward_is_unchanged_shape():
    model = _inner()
    out = model(mx.array([[1, 2, 3]]), cache=model.make_cache())
    assert out.shape == (1, 3, 64)


def test_forward_populates_kv_sink_per_layer_type():
    model = _inner()
    sink: dict = {}
    model(mx.array([[1, 2, 3]]), cache=model.make_cache(), shared_kv_sink=sink)
    # sliding_window_pattern=2 over 4 layers gives both types.
    assert sorted(sink) == ["full_attention", "sliding_attention"]
    for keys, values in sink.values():
        assert keys.shape == values.shape == (1, 1, 3, 8)


def test_return_hidden_yields_pre_norm_hidden():
    # batch_generator re-applies the trunk norm for any model that does not
    # set _omlx_mtp_head_hidden_normed, so capturing post-norm would
    # double-norm the head's input and show up only as weak acceptance.
    model = _inner()
    logits, hidden = model(
        mx.array([[1, 2, 3]]), cache=model.make_cache(), return_hidden=True
    )
    assert logits.shape == (1, 3, 64)
    assert hidden.shape == (1, 3, 24)
    assert not mx.allclose(hidden, model.model.norm(hidden)).item()


def test_kv_sink_is_cleared_between_forwards():
    model = _inner()
    ids = mx.array([[1, 2, 3]])
    model(ids, cache=model.make_cache(), shared_kv_sink={})
    # A forward that asks for neither must not write into a stale sink.
    from omlx.patches.mlx_lm_mtp.gemma4_text_model import _active

    assert getattr(_active, "sink", None) is None


class _LayerError(RuntimeError):
    """A deliberate mid-forward failure, so the test owns the exception."""


def test_trunk_norm_is_restored_when_the_forward_raises(monkeypatch):
    # The recorder stands in for self.norm for the duration of one forward.
    # If an exception could leave it installed, every later forward would
    # return the recorder's output and leak the previous request's hidden.
    from mlx_lm.models.gemma4_text import DecoderLayer

    def explode(self, *args, **kwargs):
        raise _LayerError("layer failed mid-forward")

    model = _inner()
    norm = model.model.norm
    monkeypatch.setattr(DecoderLayer, "__call__", explode)

    with pytest.raises(_LayerError):
        model(mx.array([[1, 2, 3]]), cache=model.make_cache(), return_hidden=True)

    assert model.model.norm is norm
    from omlx.patches.mlx_lm_mtp.gemma4_text_model import _active

    assert getattr(_active, "sink", None) is None


# ---------------------------------------------------------------------------
# Outer gemma4.Model wrapper
# ---------------------------------------------------------------------------


def test_outer_forward_passes_through_without_mtp_kwargs():
    model = _outer()
    out = model(mx.array([[1, 2, 3]]), cache=model.make_cache())
    assert out.shape == (1, 3, 64)


def test_outer_forward_tolerates_falsy_mtp_kwargs():
    # The engine passes return_hidden=False on ordinary steps; the stock
    # gemma4.Model signature would reject the kwarg outright.
    model = _outer()
    out = model(
        mx.array([[1, 2, 3]]),
        cache=model.make_cache(),
        return_hidden=False,
        n_confirmed=0,
    )
    assert out.shape == (1, 3, 64)


def test_outer_forward_reaches_the_inner_hidden_hook():
    model = _outer()
    logits, hidden = model(
        mx.array([[1, 2, 3]]), cache=model.make_cache(), return_hidden=True
    )
    assert logits.shape == (1, 3, 64)
    assert hidden.shape == (1, 3, 24)


# ---------------------------------------------------------------------------
# sanitize: strip the head when it has nowhere to bind, keep it when attached
# ---------------------------------------------------------------------------


def _head_and_backbone_weights():
    return {
        "model.language_model.mtp.pre_projection.weight": mx.zeros(
            (4, 4), dtype=mx.bfloat16
        ),
        "model.language_model.model.embed_tokens.weight": mx.zeros(
            (64, 24), dtype=mx.float16
        ),
    }


def test_sanitize_strips_the_head_when_it_is_not_attached():
    # The pre-existing behavior for every plain gemma4 text quant: no
    # binding site, so the merged head weights would be unexpected keys.
    model = _outer()
    assert getattr(model.language_model, "mtp", None) is None
    out = model.sanitize(_head_and_backbone_weights())
    assert not [k for k in out if ".mtp." in k]


def test_sanitize_keeps_the_head_when_it_is_attached():
    pytest.importorskip("mlx_vlm.speculative.drafters.gemma4_assistant")
    lm_mtp.set_mtp_active(True)
    model = _outer(_with_assistant())
    assert model.language_model.mtp is not None
    out = model.sanitize(_head_and_backbone_weights())
    assert [k for k in out if ".mtp." in k]


# ---------------------------------------------------------------------------
# Head dtype alignment
# ---------------------------------------------------------------------------


def _align(weights):
    return gemma4_text_model._align_head_dtype(weights)


def test_align_casts_a_bfloat16_head_to_a_float16_backbone():
    # The head ships bfloat16; against float16 activations MLX promotes to
    # float32 and the head runs at twice the memory traffic it needs.
    out = _align(
        {
            "model.embed_tokens.weight": mx.zeros((4, 4), dtype=mx.float16),
            "language_model.mtp.pre_projection.weight": mx.zeros(
                (4, 4), dtype=mx.bfloat16
            ),
        }
    )
    assert out["language_model.mtp.pre_projection.weight"].dtype == mx.float16


def test_align_casts_a_float16_head_to_a_bfloat16_backbone():
    out = _align(
        {
            "model.embed_tokens.weight": mx.zeros((4, 4), dtype=mx.bfloat16),
            "language_model.mtp.pre_projection.weight": mx.zeros(
                (4, 4), dtype=mx.float16
            ),
        }
    )
    assert out["language_model.mtp.pre_projection.weight"].dtype == mx.bfloat16


def test_align_leaves_non_float_head_tensors_alone():
    # Quantized weights and their packed scales/biases must survive intact.
    weights = {
        "model.embed_tokens.weight": mx.zeros((4, 4), dtype=mx.float16),
        "language_model.mtp.q_proj.weight": mx.zeros((4, 4), dtype=mx.uint32),
        "language_model.mtp.q_proj.scales": mx.zeros((4, 4), dtype=mx.bfloat16),
        "language_model.mtp.layer_idx": mx.zeros((1,), dtype=mx.int32),
    }
    out = _align(weights)
    assert out["language_model.mtp.q_proj.weight"].dtype == mx.uint32
    assert out["language_model.mtp.layer_idx"].dtype == mx.int32
    assert out["language_model.mtp.q_proj.scales"].dtype == mx.float16


def test_align_is_a_noop_when_the_backbone_has_no_float_tensor():
    weights = {
        "model.embed_tokens.weight": mx.zeros((4, 4), dtype=mx.uint32),
        "language_model.mtp.pre_projection.weight": mx.zeros(
            (4, 4), dtype=mx.bfloat16
        ),
    }
    out = _align(weights)
    assert out["language_model.mtp.pre_projection.weight"].dtype == mx.bfloat16


def test_align_targets_the_dtype_most_of_the_backbone_uses():
    # A quantized checkpoint is mostly packed uint32 with float scales; one
    # unrepresentative tensor early in dict order must not choose for the
    # whole model.
    weights = {
        "model.layers.0.self_attn.q_proj.scales": mx.zeros((4, 4), dtype=mx.bfloat16),
        "model.embed_tokens.weight": mx.zeros((4, 4), dtype=mx.float16),
        "model.norm.weight": mx.zeros((4,), dtype=mx.float16),
        "model.layers.0.input_layernorm.weight": mx.zeros((4,), dtype=mx.float16),
        "language_model.mtp.pre_projection.weight": mx.zeros(
            (4, 4), dtype=mx.bfloat16
        ),
    }
    out = _align(weights)
    assert out["language_model.mtp.pre_projection.weight"].dtype == mx.float16


def test_align_does_not_read_the_head_when_picking_the_target():
    # A head tensor appearing first must not become the target dtype.
    out = _align(
        {
            "language_model.mtp.pre_projection.weight": mx.zeros(
                (4, 4), dtype=mx.bfloat16
            ),
            "model.embed_tokens.weight": mx.zeros((4, 4), dtype=mx.float16),
        }
    )
    assert out["language_model.mtp.pre_projection.weight"].dtype == mx.float16


# ---------------------------------------------------------------------------
# Query position
# ---------------------------------------------------------------------------


def _position(entries):
    from omlx.patches.mlx_lm_gemma4_assistant import query_position

    return query_position(SimpleNamespace(_omlx_mtp_cache_ref=entries))


def test_query_position_prefers_the_rotating_absolute_offset():
    # BatchRotatingKVCache._offset is the committed length; _idx is a ring
    # index and offset may be a per-row array.
    assert _position([SimpleNamespace(_offset=5, _idx=99, offset=mx.array([5]))]) == 5


def test_query_position_falls_back_to_a_plain_int_offset():
    assert _position([SimpleNamespace(offset=6)]) == 6


def test_query_position_falls_back_to_the_batch_ring_index():
    assert _position([SimpleNamespace(offset=mx.array([4]), _idx=4)]) == 4


def test_query_position_raises_when_no_cache_carries_one():
    with pytest.raises(RuntimeError, match="no cache offset"):
        _position([SimpleNamespace()])


# ---------------------------------------------------------------------------
# mtp_forward preconditions
# ---------------------------------------------------------------------------


def test_mtp_forward_requires_a_prior_return_hidden_forward():
    pytest.importorskip("mlx_vlm.speculative.drafters.gemma4_assistant")
    lm_mtp.set_mtp_active(True)
    model = _inner(_with_assistant())
    model._omlx_mtp_shared_kv = None
    with pytest.raises(RuntimeError, match="shared K/V stash"):
        model.mtp_forward(mx.zeros((1, 1, 24)), mx.zeros((1, 1), dtype=mx.uint32), [])


def _stubbed_head_model(captured, committed):
    """Model with a stub drafter, a K/V stash and a cache at ``committed``.

    ``captured`` is how many positions the verify forward put in the stash;
    ``committed`` is where the cache sits once rollback has trimmed it.
    """
    from unittest.mock import MagicMock

    pytest.importorskip("mlx_vlm.speculative.drafters.gemma4_assistant")
    lm_mtp.set_mtp_active(True)
    model = _inner(_with_assistant())

    drafter = MagicMock()
    drafter._input_embed = lambda ids: mx.zeros((1, 1, 24), dtype=mx.float32)
    drafter._input_embed_scale = 1.0
    drafter.return_value = (
        mx.zeros((1, 1, 24), dtype=mx.float32),
        mx.zeros((1, 1, 64), dtype=mx.float32),
    )
    model.mtp = drafter
    model._omlx_mtp_shared_kv = {
        "full_attention": (
            mx.zeros((1, 1, captured, 8)),
            mx.zeros((1, 1, captured, 8)),
        ),
        "sliding_attention": (
            mx.zeros((1, 1, captured, 8)),
            mx.zeros((1, 1, captured, 8)),
        ),
    }
    model._omlx_mtp_kv_offset = captured
    model._omlx_mtp_cache_ref = [SimpleNamespace(offset=committed)]
    return model, drafter


def _draft(model):
    return model.mtp_forward(
        mx.zeros((1, 3, 24)), mx.zeros((1, 3), dtype=mx.uint32), []
    )


class TestRejectedTailSlicing:
    """The stash is taken at the verify forward, so after a rejection it still
    carries the rejected rows. Rollback trims the backbone cache first, which
    is what makes the stash longer than the committed length — the difference
    is exactly what has to come off before the head drafts again. Getting this
    wrong lets the head attend over K/V for tokens that were thrown away, and
    shows up only as a weak acceptance rate."""

    def test_the_rejected_tail_is_sliced_off_every_layer_type(self):
        # Depth-3 verify captured 4 positions; 2 drafts rejected leaves 2.
        model, drafter = _stubbed_head_model(captured=4, committed=2)
        _draft(model)
        _, shared_kv, _ = drafter.call_args.args
        for keys, values in shared_kv.values():
            assert keys.shape[-2] == 2
            assert values.shape[-2] == 2

    def test_a_full_accept_leaves_the_stash_alone(self):
        model, drafter = _stubbed_head_model(captured=4, committed=4)
        stash = model._omlx_mtp_shared_kv
        _draft(model)
        _, shared_kv, _ = drafter.call_args.args
        assert shared_kv is stash
        for keys, _values in shared_kv.values():
            assert keys.shape[-2] == 4

    def test_the_head_is_told_the_committed_length(self):
        model, drafter = _stubbed_head_model(captured=4, committed=2)
        _draft(model)
        assert drafter._kv_valid_len == 2

    def test_the_query_position_is_the_last_committed_slot(self):
        model, drafter = _stubbed_head_model(captured=4, committed=2)
        _draft(model)
        _, _, position_ids = drafter.call_args.args
        assert position_ids.tolist() == [[1]]

    def test_only_the_last_hidden_and_token_are_consumed(self):
        model, drafter = _stubbed_head_model(captured=4, committed=4)
        _draft(model)
        inputs_embeds = drafter.call_args.args[0]
        # Fused input is [token embedding (24) | backbone hidden (24)].
        assert inputs_embeds.shape == (1, 1, 48)

    def test_a_stale_bind_is_refreshed_before_drafting(self):
        # nn.quantize() swaps embed_tokens after the __init__-time bind; a
        # stale bind drafts through a random-init embedding.
        model, drafter = _stubbed_head_model(captured=2, committed=2)
        _draft(model)
        drafter.bind.assert_called_once_with(model)


# ---------------------------------------------------------------------------
# Depth-k partial rollback
# ---------------------------------------------------------------------------


class _FakeTrimmable:
    def __init__(self, trimmable=True):
        self._trimmable = trimmable
        self.trimmed = 0

    def is_trimmable(self):
        return self._trimmable

    def trim(self, n):
        self.trimmed += n
        return n


def _rollback(caches, accepted, num_drafts, layer_count=None, previous_kvs=None):
    from mlx_lm.models.gemma4_text import Model

    gemma4_text_model.apply()
    if layer_count is None:
        layer_count = len(caches)
    inner = SimpleNamespace(layers=[object()] * layer_count)
    if previous_kvs is not None:
        inner.previous_kvs = previous_kvs
    return Model.mtp_partial_rollback(
        SimpleNamespace(model=inner), caches, accepted, num_drafts
    )


def test_rollback_is_a_noop_when_every_draft_was_accepted():
    caches = [_FakeTrimmable(), _FakeTrimmable()]
    assert _rollback(caches, 3, 3) is True
    assert all(c.trimmed == 0 for c in caches)


def test_rollback_trims_the_rejected_tail_from_every_layer():
    caches = [_FakeTrimmable(), _FakeTrimmable()]
    assert _rollback(caches, 1, 3) is True
    assert all(c.trimmed == 2 for c in caches)


def test_rollback_refuses_rather_than_trimming_some_layers():
    # A partial trim desynchronizes per-layer KV lengths and is
    # unrecoverable; the caller falls back to a standard step instead.
    good_a, bad, good_b = _FakeTrimmable(), _FakeTrimmable(False), _FakeTrimmable()
    assert _rollback([good_a, bad, good_b], 0, 2) is False
    assert good_a.trimmed == 0
    assert good_b.trimmed == 0


def test_rollback_skips_none_cache_slots():
    live = _FakeTrimmable()
    assert _rollback([None, live, None], 0, 1) is True
    assert live.trimmed == 1


def test_rollback_refuses_an_empty_cache():
    assert _rollback([None, None], 0, 1) is False


def test_rollback_refuses_a_cache_of_the_wrong_length():
    # A stale cache list would otherwise trim whichever entries happen to be
    # present and leave the rest a position ahead.
    caches = [_FakeTrimmable(), _FakeTrimmable()]
    assert _rollback(caches, 0, 1, layer_count=3) is False
    assert all(c.trimmed == 0 for c in caches)


def test_rollback_accepts_a_kv_shared_cache_shorter_than_the_layer_count():
    # E2B shares the last 20 of 35 layers' K/V and E4B the last 18 of 42, so
    # make_cache returns one entry per distinct owner, not per layer. Counting
    # per layer refused every rollback on those checkpoints, and refused it
    # silently: the caller rebuilds the cache and takes a standard step, so
    # MTP stays switched on and simply stops paying for itself.
    caches = [_FakeTrimmable(), _FakeTrimmable()]
    assert _rollback(caches, 1, 3, layer_count=4, previous_kvs=[0, 1, 0, 1]) is True
    assert all(c.trimmed == 2 for c in caches)


def test_rollback_still_refuses_a_stale_cache_under_kv_sharing():
    # The distinct-owner count is the invariant; "shorter than the layer
    # count" on its own is not enough to accept.
    caches = [_FakeTrimmable()]
    assert _rollback(caches, 0, 1, layer_count=4, previous_kvs=[0, 1, 0, 1]) is False
    assert caches[0].trimmed == 0


def test_rollback_matches_the_backbones_own_cache_under_kv_sharing():
    # End to end against a real backbone with sharing configured, driven by
    # the cache its own make_cache builds -- the shape no local checkpoint
    # has, since 12B/26B/31B all set num_kv_shared_layers = 0 and reduce KV
    # with within-layer attention_k_eq_v instead.
    gemma4_text_model.apply()
    model = _inner({"num_kv_shared_layers": 2})
    caches = model.make_cache()
    assert len(caches) < len(model.model.layers)
    fakes = [_FakeTrimmable() for _ in caches]
    assert model.mtp_partial_rollback(fakes, 1, 3) is True
    assert all(c.trimmed == 2 for c in fakes)


class TestPartialRollbackAcrossCacheClasses:
    """gemma4 interleaves sliding and full attention, so one verify block is
    rolled back across two cache classes at once. Both must land on the same
    committed length as a reference cache that only ever saw the confirmed
    token plus the accepted drafts — otherwise the layers desynchronize by a
    position and every later forward is quietly wrong."""

    @staticmethod
    def _fill(cache, n, dim=4):
        for i in range(n):
            k = mx.full((1, 1, 1, dim), float(i))
            cache.update_and_fetch(k, k)

    @pytest.mark.parametrize("accepted", [0, 1, 2, 3])
    def test_mixed_cache_partial_accept_matches_a_reference(self, accepted):
        from mlx_lm.models.cache import KVCache, RotatingKVCache

        from omlx.patches.mlx_lm_mtp import cache_rollback

        cache_rollback.apply()
        num_drafts = 3
        verify_steps = num_drafts + 1

        def build():
            # max_size=8 with 12 tokens written: the ring has wrapped, so
            # stock trim is impossible and the armed undo log is load-bearing.
            caches = [RotatingKVCache(max_size=8), KVCache()]
            for c in caches:
                self._fill(c, 12)
            return caches

        caches = build()
        reference = build()

        verify = mx.broadcast_to(
            mx.arange(100, 100 + verify_steps, dtype=mx.float32).reshape(
                1, 1, verify_steps, 1
            ),
            (1, 1, verify_steps, 4),
        )
        cache_rollback.set_undo_armed(True)
        try:
            for c in caches:
                c.update_and_fetch(verify, verify)
        finally:
            cache_rollback.set_undo_armed(False)

        assert _rollback(caches, accepted, num_drafts) is True

        # The reference only ever saw the confirmed token plus the accepted
        # drafts: 1 + accepted of the verify block's verify_steps positions.
        keep = 1 + accepted
        for c in reference:
            c.update_and_fetch(verify[..., :keep, :], verify[..., :keep, :])

        nxt = mx.full((1, 1, 1, 4), 300.0)
        for actual, wanted in zip(caches, reference):
            ak, av = actual.update_and_fetch(nxt, nxt)
            rk, rv = wanted.update_and_fetch(nxt, nxt)
            mx.eval(ak, av, rk, rv)
            assert mx.array_equal(ak, rk).item()
            assert mx.array_equal(av, rv).item()
            assert actual.offset == wanted.offset

    def test_every_layer_lands_on_the_same_committed_length(self):
        from mlx_lm.models.cache import KVCache, RotatingKVCache

        from omlx.patches.mlx_lm_mtp import cache_rollback

        cache_rollback.apply()
        caches = [RotatingKVCache(max_size=8), KVCache(), RotatingKVCache(max_size=8)]
        for c in caches:
            self._fill(c, 12)
        verify = mx.zeros((1, 1, 4, 4))
        cache_rollback.set_undo_armed(True)
        try:
            for c in caches:
                c.update_and_fetch(verify, verify)
        finally:
            cache_rollback.set_undo_armed(False)

        assert _rollback(caches, 1, 3) is True
        assert len({c.offset for c in caches}) == 1
