"""CED scheduler contracts, short suffixes, and DSpark context continuity."""

import mlx.core as mx
import numpy as np
import pytest
from test_deepseek_v41 import load_reference_weights, tiny_ced

from omlx.cache.paged_ssd_cache import (
    PagedSSDCacheManager,
    numerics_revision_for_model,
)
from omlx.models.vlm import VLMModelAdapter
from omlx.patches.deepseek_v41.cache import DeepseekV41Cache
from omlx.patches.deepseek_v41.language import Attention
from omlx.patches.deepseek_v41.model import Model


def make_model(mtp=False, targets=(3, 4, 5), **config):
    kwargs = {}
    if mtp:
        kwargs = dict(
            preserve_mtp=True,
            n_mtp_layers=3,
            dspark_block_size=3,
            dspark_noise_token_id=2,
            dspark_target_layer_ids=targets,
            dspark_n_routed_experts=2,
            dspark_n_activated_experts=1,
            dspark_markov_rank=32,
            compress_ratios=(0, 2, 2, 1, 1, 1, 0, 0, 0),
        )
    model = Model(tiny_ced(**kwargs, **config))
    load_reference_weights(model.language_model)
    model.language_model.configure_mtp(mtp, 3)
    return model


@pytest.mark.parametrize("suffix", [1, 2, 3, 4])
def test_short_suffix_preserves_positions_and_window(suffix, monkeypatch):
    model = make_model().language_model
    prefix = mx.array([[5, 9, 3, 12, 20, 7]])
    cache, baseline = model.make_cache(), model.make_cache()
    model._omlx_prefill(prefix, cache=cache)
    model._omlx_prefill(prefix, cache=baseline)
    ids = mx.array([[21 + i for i in range(suffix)]])
    expected = model(ids, cache=baseline)
    seen = []
    original = Attention.__call__

    def observe(self, x, cache, shared, start, **kwargs):
        seen.append((self._layer, start, x.shape[1]))
        return original(self, x, cache, shared, start, **kwargs)

    monkeypatch.setattr(Attention, "__call__", observe)
    actual = model._omlx_prefill(ids, cache=cache)
    np.testing.assert_array_equal(actual, expected)
    assert seen == [(i, 6, suffix) for i in range(6)]
    for a, b in zip(cache, baseline):
        assert a.size() == b.size() == 6 + suffix
        assert a[1].shape[1] == 4
        for x, y in zip(a.cache, b.cache):
            np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("follow", [7, 2])
def test_skipped_decoder_tail_matches_running_it(follow):
    """A cache-only chunk announced to be followed by another stops at the
    midpoint. A longer next chunk replaces the decoder windows; a shorter one
    than announced replays them. Later outputs and caches match running the
    decoder."""
    model = make_model().language_model
    first = mx.array([[5, 9, 3, 12, 20, 7, 4]])
    nxt = mx.array([[8, 2, 11, 6, 13, 10, 1][:follow]])
    skipped, baseline = model.make_cache(), model.make_cache()
    model._omlx_prefill(first, cache=skipped, next_prefill_tokens=7)
    model._omlx_prefill(first, cache=baseline)
    assert skipped[3].has_pending() and not baseline[3].has_pending()
    model._omlx_prefill(nxt, cache=skipped)
    model._omlx_prefill(nxt, cache=baseline)
    assert not skipped[3].has_pending()
    token = mx.array([[30]])
    np.testing.assert_array_equal(
        model(token, cache=skipped), model(token, cache=baseline)
    )
    for a, b in zip(skipped, baseline):
        assert len(a.cache) == len(b.cache) == 7
        for x, y in zip(a.cache, b.cache):
            np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("follow", [7, 2])
def test_skipped_decoder_tail_keeps_dspark_priming(follow):
    """With every DSpark target after the midpoint, a skipped chunk captures
    nothing and a replay captures its window, so the draft context matches."""
    model = make_model(mtp=True, targets=(4, 5)).language_model
    first = mx.array([[5, 9, 3, 12, 20, 7, 4]])
    nxt = mx.array([[8, 2, 11, 6, 13, 10, 1][:follow]])
    runs = []
    for announced in (7, 0):
        cache = model.make_cache()
        model._omlx_prefill(first, cache=cache, next_prefill_tokens=announced)
        assert cache[3].has_pending() == bool(announced)
        model._omlx_prefill(nxt, cache=cache)
        context = model._omlx_mtp_prime_ctx
        offset = context.expected_target_offset
        keys = [np.asarray(c.keys.astype(mx.float32)) for c in context.caches]
        logits = np.asarray(model(mx.array([[30]]), cache=cache))
        runs.append((offset, keys, logits))
    (offset, keys, logits), (expected_offset, expected_keys, expected) = runs
    assert offset == expected_offset == 7 + follow
    for a, b in zip(keys, expected_keys):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(logits, expected)


def test_dspark_drafts_keep_the_a16_path():
    model = make_model(mtp=True, a8_prefill=True).language_model
    assert model.layers[1].ffn._a8_min_tokens == 128
    assert model.mtp[0].ffn._a8_min_tokens is None


def test_pending_replay_survives_single_row_batches():
    """A one-row batch keeps the replay slots through a write; a write to a
    mixed batch would drop them, so it is refused."""
    pending = DeepseekV41Cache(1)
    pending.cache = [mx.array([4], mx.int32)] + [None] * 6
    pending.set_pending(
        (
            mx.zeros((1, 2, 4, 8)),
            mx.zeros((1, 2, 4)),
            mx.zeros((1, 2, 3), mx.int32),
            mx.zeros((1, 2, 0), mx.int32),
        )
    )
    batch = DeepseekV41Cache.merge([pending])
    batch[0] = mx.array([4], mx.int32)
    assert batch.has_pending() and len(batch.pending()) == 4
    plain = DeepseekV41Cache(1)
    plain.cache = [mx.array([3], mx.int32)] + [None] * 6
    mixed = DeepseekV41Cache.merge([pending, plain])
    with pytest.raises(ValueError, match="cannot be batched"):
        mixed[0] = mx.array([4, 3], mx.int32)


def test_scoring_and_explicit_capture_are_full_depth():
    model = make_model(mtp=True)
    adapter = VLMModelAdapter(model)
    ids = mx.array([[3 + i % 20 for i in range(13)]])
    baseline = adapter(ids, cache=adapter.make_cache())
    model.config.ced_prefill = False
    expected = adapter(ids, cache=adapter.make_cache())
    np.testing.assert_array_equal(baseline, expected)
    model.config.ced_prefill = True
    tail = adapter._omlx_prefill(ids, cache=adapter.make_cache())
    assert tail.shape == (1, 4, 64)
    logits, hidden = model.language_model(
        ids, cache=adapter.make_cache(), return_hidden=True
    )
    assert logits.shape[1] == hidden.shape[1] == 13
    cache = adapter.make_cache()
    with pytest.raises(ValueError, match="full hidden/verify"):
        adapter._omlx_prefill(ids, cache=cache, return_hidden=True)
    assert all(c.size() == 0 for c in cache)


def test_dspark_ring_and_rollback_after_ced_prefill():
    model = make_model(mtp=True).language_model
    cache = model.make_cache()
    for start, count in [(0, 13), (13, 9), (22, 2)]:
        model._omlx_prefill(mx.array([[3 + i % 20 for i in range(count)]]), cache=cache)
        context = model._omlx_mtp_prime_ctx
        assert context.expected_target_offset == start + count
        assert all(
            c.offset == start + count and c.keys.shape[2] == 4 for c in context.caches
        )
    reference = model.make_cache()
    for a, b in zip(cache, reference):
        b.cache = list(a.cache)
    ids = mx.array([[21, 22, 23]])
    logits, hidden = model(ids, cache=cache, return_hidden=True, n_confirmed=1)
    assert logits.shape[1] == hidden.shape[1] == 3
    assert model.mtp_partial_rollback(cache, 1, 2)
    model(ids[:, :2], cache=reference, return_hidden=True)
    for a, b in zip(cache, reference):
        for x, y in zip(a.cache, b.cache):
            np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(
        model(mx.array([[25]]), cache=cache), model(mx.array([[25]]), cache=reference)
    )


def test_ced_and_full_prefill_blocks_share_one_ssd_budget(tmp_path):
    full = numerics_revision_for_model(
        VLMModelAdapter(Model(tiny_ced(ced_prefill=False)))
    )
    ced = numerics_revision_for_model(VLMModelAdapter(Model(tiny_ced())))
    assert full is None
    assert ced == "ced-prefill-1"

    data = [(mx.zeros((1, 2, 32, 8)), mx.zeros((1, 2, 32, 8)))]

    def open_manager(numerics, max_size):
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path,
            max_size_bytes=max_size,
            expected_model_name="v41",
            expected_num_layers=1,
            expected_block_size=32,
        )
        manager.set_expected_layer_signature(["KVCache"], numerics=numerics)
        manager.invalidate_stale_layer_signature()
        return manager

    def save(manager, block_hash):
        assert manager.save_block(
            block_hash, data, 32, model_name="v41", layer_cache_types=["KVCache"]
        )

    manager = open_manager(ced, 1 << 30)
    save(manager, b"\x01" * 32)
    manager.close()
    (ced_file,) = tmp_path.glob("*/*.safetensors")
    block_size = ced_file.stat().st_size

    # Full prefill must not hit the CED block, but still counts and evicts it.
    manager = open_manager(full, block_size * 3 // 2)
    assert not manager.has_block(b"\x01" * 32)
    assert manager._incompatible_index.total_size == block_size
    save(manager, b"\x02" * 32)
    manager.close()
    assert not ced_file.exists()

    manager = open_manager(ced, 1 << 30)
    assert not manager.has_block(b"\x02" * 32)
    manager.close()
