"""Tests for shared DeepSeek V4 native-indexer dispatch state."""

from omlx.patches.deepseek_v4 import indexer_dispatch


def _shape_eligible(**overrides):
    values = {
        "query_tokens": 1817,
        "pooled_tokens": 86_982,
        "n_heads": 64,
        "head_dim": 128,
        "index_topk": 512,
        "dtype_supported": True,
    }
    values.update(overrides)
    return indexer_dispatch.native_indexer_shape_eligible(**values)


def test_unaligned_query_and_pool_lengths_are_shape_eligible():
    assert _shape_eligible()


def test_dispatch_policy_and_unsupported_contracts_are_rejected():
    # The raw tail-safe kernel supports M=1, but model dispatch deliberately
    # keeps single-token decode on the existing row-wise fp32 path.
    assert not _shape_eligible(query_tokens=1)
    assert not _shape_eligible(pooled_tokens=512)
    assert not _shape_eligible(n_heads=16)
    assert not _shape_eligible(head_dim=64)
    assert not _shape_eligible(index_topk=256)
    assert not _shape_eligible(dtype_supported=False)


def test_eligibility_checks_runtime_availability(monkeypatch):
    monkeypatch.setattr(indexer_dispatch, "native_indexer_available", lambda: True)
    assert indexer_dispatch.native_indexer_eligible(
        query_tokens=1817,
        pooled_tokens=86_982,
        n_heads=64,
        head_dim=128,
        index_topk=512,
        dtype_supported=True,
    )
    monkeypatch.setattr(indexer_dispatch, "native_indexer_available", lambda: False)
    assert not indexer_dispatch.native_indexer_eligible(
        query_tokens=1817,
        pooled_tokens=86_982,
        n_heads=64,
        head_dim=128,
        index_topk=512,
        dtype_supported=True,
    )


def test_runtime_failure_disables_native_state(monkeypatch):
    monkeypatch.setattr(indexer_dispatch, "_NATIVE_INDEXER_DISABLED", False)
    indexer_dispatch.disable_native_indexer()
    assert indexer_dispatch.native_indexer_disabled()
    assert not indexer_dispatch.native_indexer_available()
