"""CED scheduler contracts, short suffixes, and DSpark context continuity."""

import mlx.core as mx
import numpy as np
import pytest
from test_deepseek_v41 import load_reference_weights, tiny_ced

from omlx.models.vlm import VLMModelAdapter
from omlx.patches.deepseek_v41.language import Attention
from omlx.patches.deepseek_v41.model import Model


def make_model(mtp=False):
    kwargs = {}
    if mtp:
        kwargs = dict(
            preserve_mtp=True,
            n_mtp_layers=3,
            dspark_block_size=3,
            dspark_noise_token_id=2,
            dspark_target_layer_ids=(3, 4, 5),
            dspark_n_routed_experts=2,
            dspark_n_activated_experts=1,
            dspark_markov_rank=32,
            compress_ratios=(0, 2, 2, 1, 1, 1, 0, 0, 0),
        )
    model = Model(tiny_ced(**kwargs))
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
        for x, y in zip(a.state, b.state):
            np.testing.assert_array_equal(x, y)


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
        for x, y in zip(a.state, b.state):
            np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(
        model(mx.array([[25]]), cache=cache), model(mx.array([[25]]), cache=reference)
    )


@pytest.mark.asyncio
async def test_clear_ssd_removes_both_modes_when_model_is_unloaded(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from omlx.admin import routes

    files = []
    for root in (tmp_path, tmp_path / "deepseek_v41_ced_v1"):
        folder = root / "a"
        folder.mkdir(parents=True)
        file = folder / "abc.safetensors"
        file.write_bytes(b"cache")
        files.append(file)
    monkeypatch.setattr(
        routes,
        "_get_engine_pool",
        lambda: SimpleNamespace(get_status=lambda: {"models": []}, _entries={}),
    )
    monkeypatch.setattr(
        routes,
        "_get_global_settings",
        lambda: SimpleNamespace(
            base_path=tmp_path,
            cache=SimpleNamespace(get_ssd_cache_dir=lambda _: tmp_path),
        ),
    )
    monkeypatch.setattr(
        routes, "_clear_cold_remote_cluster_cache_roots", lambda _: (0, 0)
    )
    settings = SimpleNamespace(
        base_path=tmp_path,
        cache=SimpleNamespace(
            get_ssd_cache_dir=lambda _: tmp_path,
            get_ssd_cache_max_size_bytes=lambda _: 1024,
        ),
    )
    stats = routes._build_runtime_cache_observability(settings)
    assert stats["total_num_files"] == 2
    assert stats["total_size_bytes"] == 10
    result = await routes.clear_ssd_cache(is_admin=True)
    assert result["total_deleted"] == 2
    assert not any(file.exists() for file in files)
