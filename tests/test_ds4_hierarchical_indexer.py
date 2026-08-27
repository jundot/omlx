"""Exactness and derived-cache tests for the certified DS4 index screen."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.custom_kernels.glm_moe_dsa import fast
from omlx.patches.deepseek_v4 import hierarchical_indexer as hi


pytestmark = pytest.mark.skipif(
    not (
        fast.is_native_available()
        and fast._EXT_MMA_SCORE
        and fast.has_symbol("dsa_topk_indices")
    ),
    reason="native rank-48 and top-k kernels are unavailable",
)


def _fixture(rows=16, pooled=2048, latent=16):
    mx.random.seed(20260825)
    basis_seed = mx.random.normal((128, latent)).astype(mx.float32)
    basis, _ = mx.linalg.qr(basis_seed, stream=mx.cpu)
    keys = (
        mx.random.normal((1, pooled, latent)).astype(mx.float32) @ basis.T
    ).astype(mx.bfloat16)
    # Queries share the same known subspace so the certified path succeeds;
    # random full-rank queries are still allowed to fail closed in production.
    q_rows = keys[0, :rows].astype(mx.float32)
    q = mx.broadcast_to(q_rows[None, None], (1, 64, rows, 128)).astype(
        mx.bfloat16
    )
    weights = mx.full((1, rows, 64), 0.05, dtype=mx.bfloat16)
    mx.eval(keys, q, weights)
    return q, keys, weights


def test_disabled_hierarchical_index_is_a_noop(monkeypatch):
    monkeypatch.setattr(hi, "_ENABLED", False)
    q, keys, weights = _fixture()
    assert (
        hi.hierarchical_topk(
            q,
            keys,
            weights,
            SimpleNamespace(),
            query_offset=8192,
            topk=512,
            ratio=4,
            kernels=fast,
        )
        is None
    )


@pytest.mark.parametrize("native_upper", (False, True))
def test_certified_hierarchical_indices_match_full_scan(monkeypatch, native_upper):
    monkeypatch.setattr(hi, "_ENABLED", True)
    monkeypatch.setattr(hi, "_NATIVE_UPPER_ENABLED", native_upper)
    monkeypatch.setattr(hi, "_NATIVE_UPPER_FAILED", False)
    monkeypatch.setattr(hi, "_MIN_POOL", 1024)
    monkeypatch.setattr(hi, "_REFRESH_POOL", 4096)
    # Keep all but one key so this unit test targets the mapping/tie/certificate
    # contract; real-fixture pruning and rate are covered by the physical gate.
    monkeypatch.setattr(hi, "_CANDIDATE_FRACTION", 0.99)
    q, keys, weights = _fixture()
    offset = 8192
    expected_scores = fast.dsa_indexer_scores_mma(
        q,
        keys[:, None],
        weights,
        mask_ratio=4,
        mask_q_offset=offset,
    )
    expected = mx.sort(
        fast.dsa_topk_indices(expected_scores, 512)[:, 0], axis=-1
    )
    actual = hi.hierarchical_topk(
        q,
        keys,
        weights,
        SimpleNamespace(),
        query_offset=offset,
        topk=512,
        ratio=4,
        kernels=fast,
    )
    assert actual is not None
    mx.eval(expected, actual)
    assert mx.array_equal(expected, actual).item()


def test_derived_key_projection_extends_without_basis_refresh(monkeypatch):
    monkeypatch.setattr(hi, "_REFRESH_POOL", 4096)
    _, keys, _ = _fixture(pooled=1024)
    cache = SimpleNamespace()
    first = hi._state_for_cache(cache, keys[:, :768])
    second = hi._state_for_cache(cache, keys)
    assert first.basis is second.basis
    assert second.basis_pool_length == 768
    assert second.projected_pool_length == 1024
    assert second.key_projection.shape == (1024, 48)


@pytest.mark.parametrize("rows,pooled", ((16, 513), (32, 2048)))
def test_native_group_upper_is_outward_from_python_bound(
    monkeypatch, rows, pooled
):
    monkeypatch.setattr(hi, "_NATIVE_UPPER_ENABLED", True)
    monkeypatch.setattr(hi, "_NATIVE_UPPER_FAILED", False)
    _, keys, _ = _fixture(rows=rows, pooled=pooled)
    state = hi._state_for_cache(SimpleNamespace(), keys)
    mx.random.seed(20260826 + rows)
    approximate = mx.random.uniform(-2, 2, (rows, pooled)).astype(mx.bfloat16)
    residual = mx.random.uniform(0, 0.2, (rows,)).astype(mx.float32)
    error = mx.random.uniform(0, 0.02, (rows,)).astype(mx.float32)
    norm = mx.random.uniform(0, 3, (rows,)).astype(mx.float32)

    actual = hi._native_group_upper(
        approximate,
        residual,
        error,
        norm,
        state,
    )
    assert actual is not None
    approximate_f = approximate.astype(mx.float32)
    error_bound = (
        residual[:, None] * state.key_orthogonal_residual[None]
        + error[:, None] * state.key_coordinate_norm[None]
        + (norm + error)[:, None] * state.key_coordinate_error[None]
    )
    reference = mx.max(
        (
            approximate_f
            + error_bound
            + hi._NUMERIC_ABS_GUARD
            + hi._NUMERIC_REL_GUARD * mx.abs(approximate_f)
        ).reshape(rows // hi._GROUP_ROWS, hi._GROUP_ROWS, pooled),
        axis=1,
    )
    mx.eval(actual, reference)

    delta = actual - reference
    assert bool(mx.all(delta >= 0).item())
    assert float(mx.max(delta).item()) < 5e-4


def test_certificate_miss_backoff_retries_on_exponential_refresh_boundaries(
    monkeypatch,
):
    monkeypatch.setattr(hi, "_REFRESH_POOL", 2048)
    cache = SimpleNamespace()

    first = hi._record_certificate_miss(cache, 16000)
    assert first == hi._MissBackoff(18048, 16000, 1)
    assert hi._miss_backoff_active(cache, 18047)
    assert not hi._miss_backoff_active(cache, 18048)

    second = hi._record_certificate_miss(cache, 18048)
    assert second == hi._MissBackoff(22144, 18048, 2)
    third = hi._record_certificate_miss(cache, 22144)
    assert third == hi._MissBackoff(30336, 22144, 3)
    fourth = hi._record_certificate_miss(cache, 30336)
    assert fourth == hi._MissBackoff(46720, 30336, 4)
    fifth = hi._record_certificate_miss(cache, 46720)
    assert fifth == hi._MissBackoff(63104, 46720, 5)

    hi._clear_certificate_miss(cache)
    assert not hi._miss_backoff_active(cache, 46721)


def test_certificate_miss_backoff_clears_when_cache_shrinks():
    cache = SimpleNamespace()
    setattr(cache, hi._MISS_BACKOFF_ATTR, hi._MissBackoff(20000, 18000, 2))

    assert not hi._miss_backoff_active(cache, 4096)
    assert not hasattr(cache, hi._MISS_BACKOFF_ATTR)


def test_hierarchy_backoff_skips_before_state_or_gpu_work(monkeypatch):
    monkeypatch.setattr(hi, "_ENABLED", True)
    monkeypatch.setattr(hi, "_MIN_POOL", 1024)
    q, keys, weights = _fixture(rows=16, pooled=2048)
    cache = SimpleNamespace()
    setattr(cache, hi._MISS_BACKOFF_ATTR, hi._MissBackoff(4096, 2048, 1))
    monkeypatch.setattr(
        hi,
        "_state_for_cache",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("backoff must skip before derived-state work")
        ),
    )

    assert (
        hi.hierarchical_topk(
            q,
            keys,
            weights,
            cache,
            query_offset=8192,
            topk=512,
            ratio=4,
            kernels=fast,
        )
        is None
    )
