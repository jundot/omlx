"""QSA reservation integration tests for restored prefixes and model output."""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from test_qwen4_qsa_reserved_capacity import language

from omlx.scheduler import Scheduler


def test_restored_index_reserves_full_horizon_on_first_growth():
    cache = language.QSAKVCache()
    cache.index_step = 64
    cache.state = (None, None, mx.ones((1, 1088, 8)), mx.arange(1088)[None])
    cache.reserve_index_capacity(2000)
    cache.update_indexer(mx.ones((1, 1, 8)), mx.array([[1088]]))
    mx.eval(cache.index_keys)
    assert cache._index_keys.shape[1] == 2048
    assert mx.all(cache.index_keys == 1).item()


@pytest.mark.parametrize("snapshots_enabled", [False, True])
def test_chunked_reservation_includes_restored_prefix(snapshots_enabled):
    cache = language.QSAKVCache()
    cache.state = (
        mx.zeros((1, 1, 128, 4)),
        mx.zeros((1, 1, 128, 4)),
        mx.ones((1, 128, 8)),
        mx.arange(128)[None],
    )
    ns = SimpleNamespace(
        model=object(),
        config=SimpleNamespace(paged_cache_block_size=64),
        block_aware_cache=object() if snapshots_enabled else None,
        _stream=mx.default_stream(mx.gpu),
    )
    state = Scheduler._begin_prefill(
        ns,
        SimpleNamespace(cached_tokens=128, request_id="reservation"),
        [1] * 129,
        [cache, SimpleNamespace(offset=128)],
    )
    ns._prefill_step_size_for_progress = lambda *a: 64
    ns._reserve_qsa_index_capacity = (
        lambda caches, tokens: Scheduler._reserve_qsa_index_capacity(ns, caches, tokens)
    )

    def stop(*args, **kwargs):
        raise RuntimeError("reservation complete")

    ns._adaptive_chunk_size = stop
    with pytest.raises(RuntimeError, match="reservation complete"):
        Scheduler._step_prefill_chunk(ns, state)
    assert cache._index_reserved_tokens >= 256


def test_reserved_qsa_prefill_and_decode_match_unreserved_model():
    from mlx_vlm.models.qwen4_exp.language import LanguageModel
    from test_mlx_vlm_qwen4_exp_compat import _tiny_config

    mx.random.seed(3455)
    config = _tiny_config()
    model = LanguageModel(config.text_config, config)
    ordinary, reserved = model.make_cache(), model.make_cache()
    for cache in ordinary + reserved:
        if hasattr(cache, "index_step"):
            cache.index_step = 8
    Scheduler._reserve_qsa_index_capacity(object(), reserved, 36)
    for start in range(0, 36, 6):
        tokens = (mx.arange(start, start + 6) % 60 + 1)[None]
        expected = model(tokens, cache=ordinary).logits
        actual = model(tokens, cache=reserved).logits
        mx.eval(expected, actual)
        assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item()
    for _ in range(5):
        token = mx.argmax(expected[:, -1], axis=-1)[:, None]
        assert mx.array_equal(token, mx.argmax(actual[:, -1], axis=-1)[:, None]).item()
        expected = model(token, cache=ordinary).logits
        actual = model(token, cache=reserved).logits
        mx.eval(expected, actual)
        assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item()
