# SPDX-License-Identifier: Apache-2.0
"""Split-K numerical checks and controlled per-row cache continuity on NAX."""

import copy
import importlib
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.patches.mlx_lm_mtp import batch_generator as bg
from omlx.patches.mlx_vlm_glm5_next_compat import apply_mlx_vlm_glm5_next_compat_patch
from omlx.patches.mlx_vlm_mtp import glm5_next_vlm_runtime
from omlx.patches.mlx_vlm_mtp.glm5_next_batch_rollback import rollback_rows
from tests.glm5_split_k_reference import (
    FusedProjectionReference,
    MLXBits,
    Phase,
    ProjectionReference,
)
from tests.glm5_split_k_trace import KernelTrace
from tests.test_glm5_next_mtp import TINY_TEXT_CONFIG

apply_mlx_vlm_glm5_next_compat_patch()
from mlx_vlm.models.cache import ArraysCache, CacheList  # noqa: E402
from mlx_vlm.models.glm5_next import language  # noqa: E402

BF16 = mx.bfloat16


@pytest.fixture
def split_k():
    # Import inside the fixture so every CPU regression is collected on main.
    return importlib.import_module("mlx_vlm.models.glm5_next.sparse_mla_decode")


def _gpu_nax():
    return mx.metal.is_available() and language.nax_sparse_mla_available()


def _inputs():
    return (
        mx.zeros((1, 16, 1, 128), mx.float16),
        mx.zeros((1, 1, 8, 128), mx.float16),
        mx.array([[0, 1]], mx.int32),
    )


@pytest.mark.parametrize(
    "case",
    ["heads", "latent", "dtype", "rank", "positions", "pool", "window_dtype", "empty"],
)
def test_rejects_unsupported_scalar_before_kernel(split_k, monkeypatch, case):
    monkeypatch.setattr(split_k, "_get", lambda: pytest.fail("kernel was requested"))
    q, kv, windows = _inputs()
    before, plen = 7, 2
    if case == "heads":
        q = q[:, :15]
    elif case == "latent":
        q, kv = q[..., :124], kv[..., :124]
    elif case == "dtype":
        q = q.astype(mx.float32)
    elif case == "rank":
        windows = windows[None]
    elif case == "positions":
        before = 8
    elif case == "pool":
        plen = 3
    elif case == "window_dtype":
        windows = windows.astype(mx.float32)
    else:
        windows = windows[:, :0]
    with pytest.raises(ValueError):
        split_k.sparse_mla_decode(q, kv, windows, before, plen, 0.1)


@pytest.mark.parametrize(
    "case", ["extent", "rows", "row_dtype", "batch", "window_rows"]
)
def test_rejects_unsupported_batch_before_kernel(split_k, monkeypatch, case):
    monkeypatch.setattr(
        split_k, "_get_batched", lambda: pytest.fail("kernel was requested")
    )
    q, kv, windows = _inputs()
    windows = windows[None]
    rows = mx.array([[2, 7, 0]], mx.int32)
    extent = 8
    if case == "extent":
        extent = 9
    elif case == "rows":
        rows = rows[:, :2]
    elif case == "row_dtype":
        rows = rows.astype(mx.float32)
    elif case == "batch":
        kv = mx.concatenate([kv, kv])
    else:
        windows = mx.concatenate([windows, windows], axis=1)
    with pytest.raises(ValueError):
        split_k.sparse_mla_decode_batched(q, kv, extent, windows, rows, 0.1)


def test_batched_dispatch_preserves_backing_stride_and_key_extent(split_k, monkeypatch):
    seen = {}
    q = mx.zeros((2, 16, 3, 128), mx.float16)
    kv = mx.zeros((2, 1, 256, 128), mx.float16)
    windows = mx.zeros((2, 3, 512), mx.int32)
    rows = mx.array([[2, 5, 4], [3, 9, 0]], mx.int32)

    def blocks(**kwargs):
        seen.update(kwargs)
        return [
            mx.zeros(shape, dtype)
            for shape, dtype in zip(kwargs["output_shapes"], kwargs["output_dtypes"])
        ]

    def merge(**kwargs):
        return [mx.zeros(kwargs["output_shapes"][0], mx.float16)]

    monkeypatch.setattr(split_k, "_get_batched", lambda: (blocks, merge))
    out = split_k.sparse_mla_decode_batched(q, kv, 12, windows, rows, 0.1)
    assert seen["inputs"][1] is kv
    dims = seen["inputs"][3].tolist()
    assert dims[1] == 12 and dims[6] == 256 * 128
    assert (
        dims[5]
        == (512 * 4 + 3 + split_k._SPLIT_K_KEY_TILE_ROWS - 1)
        // split_k._SPLIT_K_KEY_TILE_ROWS
    )
    assert out.shape == (2, 16, 3, 128)


def test_attention_returns_before_indexer_retry(split_k, monkeypatch):
    calls = []
    q = mx.zeros((1, 16, 1, 64), mx.float16)
    kv = mx.zeros((1, 1, 8, 128), mx.float16)
    windows = mx.zeros((1, 512), mx.int32)
    owner = SimpleNamespace(
        embed_q=lambda value: mx.zeros((1, 16, 1, 128), mx.float16),
        unembed_out=lambda value: value,
        scale=0.1,
        _finish=lambda value, _: value,
    )
    cache = [SimpleNamespace(keys=kv), SimpleNamespace(pooled=None)]

    def kernel(*args, **kwargs):
        calls.append(args)
        return mx.zeros((1, 16, 1, 128), mx.float16)

    monkeypatch.setattr(language, "sparse_mla_decode", kernel)
    got = language.Glm5NextSparseAttention._decode_fused(
        owner, q, kv, cache, windows, 7, 2
    )
    assert len(calls) == 1 and calls[0][1] is kv
    assert got.shape == (1, 1, 16 * 128)


def _spy(monkeypatch, owner, name, counter):
    original = getattr(owner, name)

    def wrapped(*args, **kwargs):
        counter.append(name)
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, wrapped)


@pytest.mark.skipif(not _gpu_nax(), reason="NAX GPU required")
def test_batched_decode_kernel_matches_reference_with_left_padding(split_k):
    """Window expansion, per-row validity, tail and left padding in-kernel."""
    mx.random.seed(11)
    batch_size, query_heads, block_width, latent_dim, kp = 3, 16, 3, 128, 4
    before = [41, 18, 77]
    lpad = [36, 59, 0]
    kv_capacity = max(b + block_width + left_pad for b, left_pad in zip(before, lpad))
    assert all(
        b + block_width + left_pad == kv_capacity for b, left_pad in zip(before, lpad)
    )
    plen = [(b + block_width) // kp for b in before]
    selected_windows = 16
    q = (
        mx.random.normal((batch_size, query_heads, block_width, latent_dim)) * 0.05
    ).astype(BF16)
    kv = mx.random.normal((batch_size, 1, kv_capacity + 40, latent_dim)).astype(
        BF16
    )  # spare capacity
    # Distinct candidates (top-k never repeats a window), including invalid
    # ids: negative, past the row's pool, and not yet complete for the row.
    span = max(plen) + 5
    order = mx.argsort(
        mx.random.uniform(shape=(batch_size, block_width, span)), axis=-1
    )
    windows = (order[..., :selected_windows] - 2).astype(mx.int32)
    rows = mx.array(
        [[p, b, left_pad] for p, b, left_pad in zip(plen, before, lpad)], dtype=mx.int32
    )
    got = split_k.sparse_mla_decode_batched(q, kv, kv_capacity, windows, rows, 0.1)
    got = got.astype(mx.float32)
    kvf = kv.astype(mx.float32)
    for b in range(batch_size):
        for s in range(block_width):
            qpos = before[b] + s
            toks = set()
            for w in windows[b, s].tolist():
                if 0 <= w < plen[b] and (w + 1) * kp - 1 <= qpos:
                    toks.update(range(w * kp, w * kp + kp))
            cnt = (qpos + 1) % kp
            toks.update(range(qpos + 1 - cnt, qpos + 1))
            idx = mx.array(sorted(t + lpad[b] for t in toks))
            keys = kvf[b, 0][idx]
            sc = (q[b, :, s].astype(mx.float32) @ keys.T) * 0.1
            ref = mx.softmax(sc, axis=-1) @ keys
            error = mx.abs(got[b, :, s] - ref)
            # Match test_nax_sparse_mla_matches_fp32_reference_like_native_kernel.
            assert mx.all(error <= mx.abs(ref) * 2.0**-8 + 1e-3).item(), (b, s)
            assert mx.all(mx.isfinite(got[b, :, s])).item()


def _host():
    assert glm5_next_vlm_runtime.apply()
    from mlx_vlm.models.glm5_next.config import TextConfig

    values = copy.deepcopy(TINY_TEXT_CONFIG)
    values["index_topk"] = 2048
    values["index_n_heads"] = 32
    values["index_head_dim"] = 128
    values["num_hidden_layers"] = 8
    values["num_nextn_predict_layers"] = 0
    values["layer_types"] = values["layer_types"][:8]
    values["mlp_layer_types"] = values["mlp_layer_types"][:8]
    values["linear_attn_config"]["kda_layers"] = [0, 1, 2, 4, 5, 6]
    values["linear_attn_config"]["full_attn_layers"] = [3, 7]
    # Fused decode needs 16k heads and a 128-multiple latent width.
    values["num_attention_heads"] = 16
    values["num_key_value_heads"] = 16
    values["kv_lora_rank"] = 128
    host = language.LanguageModel(TextConfig.from_dict(values))
    host.set_dtype(BF16)
    return host


def _prefill_rows(host, lengths):
    rows = []
    for n in lengths:
        cache = host.make_cache()
        tokens = mx.random.randint(0, 512, (1, n)).astype(mx.uint32)
        mx.eval(host(tokens, cache=cache).logits)
        rows.append(cache)
    return rows


def _assert_pool_rows(batch_cache, scalar_rows):
    for layer_number, layer in enumerate(batch_cache):
        if not isinstance(layer, CacheList):
            continue
        pooled = layer[1]
        for row_number, reference in enumerate(scalar_rows):
            scalar = reference[layer_number][1]
            assert pooled.ratio == scalar.ratio
            assert language.Glm5NextIndexer._processed(pooled)[
                row_number
            ] == language.Glm5NextIndexer._processed(scalar)
            assert language.Glm5NextIndexer._pool_lengths(pooled)[
                row_number
            ] == language.Glm5NextIndexer._pool_lengths(scalar)
            assert pooled.remainder[row_number] == scalar.remainder


def _close(a, b, what):
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    err = mx.abs(a - b).max().item()
    scale = mx.abs(b).max().item()
    # The project-main tiny-model batched reference uses this tolerance.
    assert mx.allclose(a, b, atol=3e-4, rtol=3e-4).item(), (what, err, scale)
    return err


@pytest.mark.skipif(not _gpu_nax(), reason="NAX GPU required")
@pytest.mark.parametrize("lengths", [[2051, 2057], [2057]])
def test_tiny_model_batched_decode_matches_rows_alone(split_k, monkeypatch, lengths):
    mx.random.seed(4242)
    monkeypatch.setattr(language, "_DSA_DECODE_FUSED", True)
    host = _host()
    # Keep each row in the native 512-window selector's supported regime.
    size = len(lengths)
    rows = _prefill_rows(host, lengths)
    cache = bg._merge_row_caches([copy.deepcopy(r) for r in rows])
    calls = []
    _spy(monkeypatch, language, "sparse_mla_decode_batched", calls)
    for step, width in enumerate([1, 1, 1, 3, 1, 4, 1, 1]):
        tokens = mx.random.randint(0, 512, (size, width)).astype(mx.uint32)
        batched = host(tokens, cache=cache).logits
        singles = [host(tokens[r : r + 1], cache=rows[r]).logits for r in range(size)]
        mx.eval(batched, singles)
        for r in range(size):
            _close(batched[r : r + 1], singles[r], (step, r))
            assert mx.array_equal(
                mx.argmax(batched[r], axis=-1), mx.argmax(singles[r][0], axis=-1)
            ).item(), (step, r)
        _assert_pool_rows(cache, rows)
    # Two sparse layers per step, all of them on the batched fused kernel.
    assert len(calls) == 2 * 8


class ContinuityMismatchError(AssertionError):
    def __init__(self, kind, context):
        super().__init__(f"{kind}: {context}")
        self.kind, self.context = kind, context


def _continuity_close(actual, reference, context):
    # Exact equality is a diagnostic gate; batch/scalar kernel variants and widths may differ.
    assert actual.shape == reference.shape and actual.dtype == reference.dtype
    assert mx.all(mx.isfinite(actual)).item() and mx.all(mx.isfinite(reference)).item()
    assert actual.dtype in (mx.bfloat16, mx.float32)
    bits = mx.uint16 if actual.dtype == mx.bfloat16 else mx.uint32
    numeric = mx.allclose(actual, reference, atol=0, rtol=0).item()
    raw_equal = mx.array_equal(actual.view(bits), reference.view(bits)).item()
    if not (numeric and raw_equal):
        error = ContinuityMismatchError("logits", context)
        a, b = (
            actual.view(bits).reshape(-1).tolist(),
            reference.view(bits).reshape(-1).tolist(),
        )
        first = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
        trace = getattr(language, "_glm40_test_trace", None)
        if trace is not None:
            trace.failures.append(
                {
                    "kind": "logits",
                    "context": context,
                    "dtype": str(actual.dtype),
                    "shape": list(actual.shape),
                    "first_flat_index": first,
                    "actual_bits": a[first],
                    "reference_bits": b[first],
                }
            )
        error.args = (
            f"logits: {context}; dtype={actual.dtype}; shape={actual.shape}; "
            f"first_flat_index={first}; actual_bits={a[first]}; reference_bits={b[first]}",
        )
        raise error


def _assert_sparse_kv_rows(cache, rows, cycle, operation):
    _assert_pool_rows(cache, rows)
    for number, layer in enumerate(cache):
        if not isinstance(layer, CacheList):
            continue
        for row, scalar in enumerate(rows):
            actual, expected = layer[0].extract(row), scalar[number][0]
            if actual.offset != expected.offset:
                raise ContinuityMismatchError("KV extent", (cycle, operation, row))
            a = actual.keys[:, :, : actual.offset]
            b = expected.keys[:, :, : expected.offset]
            assert a.shape == b.shape and a.dtype == b.dtype
            equal = mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item()
            trace = getattr(language, "_glm40_test_trace", None)
            if trace is not None:
                trace.state_checks.append(
                    {
                        "cycle": cycle,
                        "operation": operation,
                        "layer": number,
                        "row": row,
                        "kind": "KV values",
                        "shape": list(a.shape),
                        "dtype": str(a.dtype),
                        "raw_bits_equal": equal,
                    }
                )
            if not equal:
                first = mx.argmax(
                    (a.view(mx.uint16) != b.view(mx.uint16))
                    .astype(mx.int32)
                    .reshape(-1)
                ).item()
                remaining, coordinate = first, []
                for size in reversed(a.shape):
                    remaining, index = divmod(remaining, size)
                    coordinate.append(index)
                if trace is not None:
                    trace.state_checks[-1].update(
                        first_flat_index=first,
                        first_coordinate=list(reversed(coordinate)),
                        actual_bits=a.view(mx.uint16).reshape(-1)[first].item(),
                        reference_bits=b.view(mx.uint16).reshape(-1)[first].item(),
                    )
                raise ContinuityMismatchError("KV values", (cycle, operation, row))


def _pool_live_buffer(pool, name):
    assert name in ("buf_kv", "buf_gate")
    remaining, value = pool.remainder, getattr(pool, name)
    assert type(remaining) is int and 0 <= remaining < pool.ratio
    if value is None:
        assert remaining == 0, "missing live pooling buffer"
        return None
    assert value.ndim == 3 and value.shape[0] == 1
    assert remaining <= value.shape[1] <= pool.ratio
    assert value.dtype in (mx.bfloat16, mx.float32)
    if remaining == 0:
        assert (
            pool.pooled is not None
        ), "empty buffer layout requires pooled dtype/width anchor"
    if pool.pooled is not None:
        assert (
            value.shape[2] == pool.pooled.shape[2] and value.dtype == pool.pooled.dtype
        )
    # PoolingCache.state (cache_extras.py:281) and BatchPoolingCache.extract (:1121)
    # expose no live buffer at remainder=0, even if an allocated backing buffer remains.
    logical = value[:, :remaining]
    assert logical.shape == (1, remaining, value.shape[2])
    return logical


def _normalize_zero_pool_views(a, b):
    for value in (a, b):
        if value is not None:
            assert value.ndim == 3 and value.shape[0] == 1 and value.shape[1] == 0
            assert value.shape[2] > 0 and value.dtype in (mx.bfloat16, mx.float32)
    if a is not None and b is not None:
        assert a.shape == b.shape and a.dtype == b.dtype
    return None, None


def _assert_glm_nonoverlap(host):
    for layer in host.model.layers:
        if isinstance(layer.self_attn, language.Glm5NextSparseAttention):
            indexer = layer.self_attn.indexer
            assert type(indexer) is language.Glm5NextIndexer
            # GLM language.py:849-860 pools each current window without overlap.
            # It has no overlap flag; an explicitly enabled flag must fail this test.
            assert getattr(indexer, "overlap", False) is False, "overlap changed"
    return False


def _assert_cache_rows(cache, rows, cycle, operation, *, overlap=True):
    # State continuity compares exact stored bits, not different-shaped logits reductions.
    _assert_sparse_kv_rows(cache, rows, cycle, operation)
    for number, layer in enumerate(cache):
        for row, scalar in enumerate(rows):
            actual = layer.extract(row)
            expected = scalar[number]
            if isinstance(layer, CacheList):
                actual, expected = actual[1], expected[1]
                assert type(overlap) is bool
                names = ("pooled", "buf_kv", "buf_gate")
                # GLM language.py:1010-1011/1043-1044/1097-1101 never consumes carry;
                # DeepSeek deepseek_v4_model.py:1016-1027 consumes it when overlap=True.
                if overlap:
                    names += ("prev_win_kv", "prev_win_gate")
                tensors = [
                    (name, getattr(actual, name), getattr(expected, name))
                    for name in names
                ]
            else:
                assert isinstance(layer, ArraysCache), "unsupported continuity cache"
                tensors = [
                    (str(i), a, b)
                    for i, (a, b) in enumerate(zip(actual.state, expected.state))
                ]
                assert len(actual.state) == len(expected.state)
            for name, a, b in tensors:
                if isinstance(layer, CacheList) and name in ("buf_kv", "buf_gate"):
                    a = _pool_live_buffer(actual, name)
                    b = _pool_live_buffer(expected, name)
                    if actual.remainder == expected.remainder == 0:
                        a, b = _normalize_zero_pool_views(a, b)
                if a is None or b is None:
                    assert a is b, (cycle, operation, number, row, name)
                    continue
                assert a.shape == b.shape and a.dtype == b.dtype
                bits = mx.uint16 if a.dtype == mx.bfloat16 else mx.uint32
                assert a.dtype in (mx.bfloat16, mx.float32)
                equal = mx.array_equal(a.view(bits), b.view(bits)).item()
                trace = getattr(language, "_glm40_test_trace", None)
                if trace is not None:
                    trace.state_checks.append(
                        {
                            "cycle": cycle,
                            "operation": operation,
                            "layer": number,
                            "row": row,
                            "kind": "cache state",
                            "field": name,
                            "shape": list(a.shape),
                            "dtype": str(a.dtype),
                            "raw_bits_equal": equal,
                        }
                    )
                if not equal:
                    if trace is not None:
                        a_words, b_words = a.view(bits).reshape(-1), b.view(
                            bits
                        ).reshape(-1)
                        first = int(mx.argmax(a_words != b_words).item())
                        coordinate, remaining = [], first
                        for dim in reversed(a.shape):
                            coordinate.append(remaining % dim)
                            remaining //= dim
                        trace.state_checks[-1].update(
                            first_flat_index=first,
                            first_coordinate=list(reversed(coordinate)),
                            actual_bits=a_words[first].item(),
                            reference_bits=b_words[first].item(),
                        )
                    raise ContinuityMismatchError(
                        "cache state", (cycle, operation, row)
                    )


def _damage_row_mapping_kv(cache, sparse_layer):
    """Persist a wrong row address in the last accepted KV slot after rollback."""
    kv, pool = cache[sparse_layer][0], cache[sparse_layer][1]
    processed = language.Glm5NextIndexer._processed(pool)
    padding = kv.left_padding.tolist()
    row, source_row = 1, 0
    target = padding[row] + processed[row] - 1
    source = padding[source_row] + processed[source_row] - 1
    original = kv.keys[row : row + 1, :, target : target + 1]
    replacement = kv.keys[source_row : source_row + 1, :, source : source + 1]
    assert 0 <= target < kv.keys.shape[2] and 0 <= source < kv.keys.shape[2]
    assert not mx.array_equal(
        original.view(mx.uint16), replacement.view(mx.uint16)
    ).item()
    row_mask = mx.arange(kv.keys.shape[0])[:, None, None, None] == row
    position = mx.arange(kv.keys.shape[2])[None, None, :, None] == target
    kv.keys = mx.where(row_mask & position, replacement, kv.keys)
    return {
        "layer": sparse_layer,
        "row": row,
        "source_row": source_row,
        "physical_target": target,
        "logical_target": processed[row] - 1,
        "physical_source": source,
        "kind": "KV values",
    }


def _damage_accepted_kv(cache, sparse_layer, prefix, rejected):
    row = 1
    kv, pool = cache[sparse_layer][0], cache[sparse_layer][1]
    processed = language.Glm5NextIndexer._processed(pool)[row]
    padding = kv.left_padding.tolist()[row]
    retained = padding + processed - 1
    source = retained + 1 if rejected else retained - prefix
    assert 0 <= source < kv.keys.shape[2]
    original = kv.keys[row : row + 1, :, retained : retained + 1]
    replacement = kv.keys[row : row + 1, :, source : source + 1]
    assert not mx.array_equal(original, replacement).item(), "fault did not change KV"
    row_mask = mx.arange(kv.keys.shape[0])[:, None, None, None] == row
    position = mx.arange(kv.keys.shape[2])[None, None, :, None] == retained
    kv.keys = mx.where(row_mask & position, replacement, kv.keys)


def _inject_state_fault(cache, layer, prefix, fault):
    kv, pool = cache[layer][0], cache[layer][1]
    before = mx.array(kv.keys)
    processed = language.Glm5NextIndexer._processed(pool)
    padding = kv.left_padding.tolist()
    target = padding[1] + processed[1] - 1
    if fault == "indexing":
        mutation = _damage_row_mapping_kv(cache, layer)
    else:
        _damage_accepted_kv(cache, layer, prefix, fault == "rollback")
        mutation = {
            "layer": layer,
            "row": 1,
            "source_row": 1,
            "physical_target": target,
            "logical_target": processed[1] - 1,
            "physical_source": target + 1 if fault == "rollback" else target - prefix,
            "kind": "KV values",
        }
    unchanged_row = mx.array_equal(
        before[0].view(mx.uint16), kv.keys[0].view(mx.uint16)
    ).item()
    unchanged_prefix = mx.array_equal(
        before[1, :, :target].view(mx.uint16), kv.keys[1, :, :target].view(mx.uint16)
    ).item()
    unchanged_tail = mx.array_equal(
        before[1, :, target + 1 :].view(mx.uint16),
        kv.keys[1, :, target + 1 :].view(mx.uint16),
    ).item()
    changed = (before.view(mx.uint16) != kv.keys.view(mx.uint16)).astype(mx.int32)
    assert (
        unchanged_row
        and unchanged_prefix
        and unchanged_tail
        and mx.sum(changed).item() > 0
    )
    mutation.update(
        fault=fault,
        cycle=1,
        operation="verify",
        accepted_prefix=prefix,
        row0_bits_unchanged=unchanged_row,
        prior_accepted_bits_unchanged=unchanged_prefix,
        rejected_tail_bits_unchanged=unchanged_tail,
        changed_word_count=mx.sum(changed).item(),
        target_bits_before=before[1, :, target].view(mx.uint16).reshape(-1).tolist(),
        target_bits_after=kv.keys[1, :, target].view(mx.uint16).reshape(-1).tolist(),
    )
    return mutation


def _inject_conv_history_fault(cache):
    history = cache[2][0]
    before = history.view(mx.uint16)
    mask = mx.zeros(before.shape, mx.uint16)
    mask[1, -1, 0] = 1
    cache[2][0] = (before ^ mask).view(mx.bfloat16)
    after = cache[2][0].view(mx.uint16)
    assert mx.array_equal(before[0], after[0]).item()
    assert mx.array_equal(before[1, :-1], after[1, :-1]).item()
    assert mx.sum(before != after).item() == 1
    return {
        "fault": "conv_history",
        "cycle": 1,
        "operation": "verify",
        "layer": 2,
        "row": 1,
        "field": "0",
        "kind": "cache state",
        "coordinate": [0, 2, 0],
        "changed_word_count": 1,
        "row0_bits_unchanged": True,
        "prior_history_bits_unchanged": True,
        "target_bits_before": before[1, -1, 0].item(),
        "target_bits_after": after[1, -1, 0].item(),
    }


def _verify_rollback_continuity(monkeypatch, depth, fault=None, fault_events=None):
    # This tests cache state continuity, not logits equality across reduction paths.
    # Same-input Linear replay can differ between batch/scalar BF16 GEMM shapes.
    # Align only reference projections so reduction differences do not obscure
    # retained-state errors; every accepted input word must still match.
    mx.random.seed(977 + depth)
    monkeypatch.setattr(language, "_DSA_DECODE_FUSED", True)
    host = _host()
    assert _assert_glm_nonoverlap(host) is False
    lengths = [2051, 2057]
    size = len(lengths)
    rows = _prefill_rows(host, lengths)
    cache = bg._merge_row_caches([copy.deepcopy(r) for r in rows])
    targets = [
        (layer.self_attn.kv_a_proj_with_mqa, number)
        for number, layer in enumerate(host.model.layers)
        if isinstance(layer.self_attn, language.Glm5NextSparseAttention)
    ]
    width = depth + 1
    plan = [
        Phase(
            cycle,
            operation,
            width if operation == "verify" else 1,
            tuple(
                (row + cycle) % width + 1 if operation == "verify" else 1
                for row in range(size)
            ),
        )
        for cycle in range(3)
        for operation in ("verify", "decode0", "decode1")
    ]
    reference = ProjectionReference(MLXBits(mx), targets, plan)
    kda_targets = [
        (layer.self_attn, number)
        for number, layer in enumerate(host.model.layers)
        if isinstance(layer.self_attn, language.Glm5NextLinearAttention)
    ]
    assert all(owner.fuse_in for owner, _ in kda_targets)
    kda_reference = FusedProjectionReference(MLXBits(mx), kda_targets, plan)
    split_k = importlib.import_module("mlx_vlm.models.glm5_next.sparse_mla_decode")
    trace = KernelTrace(mx, reference)
    trace.kda_reference = kda_reference
    trace.install(monkeypatch, split_k)
    monkeypatch.setattr(language, "_glm40_test_trace", trace, raising=False)
    layer_ids = {
        id(host.model.layers[number].self_attn): number for _, number in targets
    }
    original_attention = language.Glm5NextSparseAttention.__call__
    original_kda = language.Glm5NextLinearAttention.__call__
    original_fused = language.Glm5NextLinearAttention._fused_in_proj
    kda_layers = {id(owner): number for owner, number in kda_targets}
    original_linear = language.linear_forward
    original_batched = language.sparse_mla_decode_batched
    original_single = language.sparse_mla_decode
    kernel_calls = {"batch": 0, "single": 0}
    faults = [] if fault_events is None else fault_events
    trace.fault_events = faults

    def attention(owner, *args, **kwargs):
        if id(owner) not in layer_ids:
            return original_attention(owner, *args, **kwargs)
        with reference.scope(layer_ids[id(owner)]):
            return original_attention(owner, *args, **kwargs)

    def linear(owner, value):
        return reference.project(owner, value, original_linear)

    def kda_attention(owner, *args, **kwargs):
        with kda_reference.scope(kda_layers[id(owner)]):
            return original_kda(owner, *args, **kwargs)

    def fused_projection(owner, value, split=True):
        assert split is True, "unexpected KDA projection variant"
        # BF16 GEMM shapes B2x8/B1x3 can differ despite identical input words.
        # The reference shares projection output to isolate conv history state.
        # Input/prefix/one-use guards apply; the actual batch object is unchanged.
        return kda_reference.project_fused(owner, value, original_fused)

    def batch_kernel(q, kv, *args, **kwargs):
        kernel_calls["batch"] += 1
        return original_batched(q, kv, *args, **kwargs)

    def single(*args, **kwargs):
        kernel_calls["single"] += 1
        return original_single(*args, **kwargs)

    monkeypatch.setattr(language.Glm5NextSparseAttention, "__call__", attention)
    monkeypatch.setattr(language, "linear_forward", linear)
    monkeypatch.setattr(language.Glm5NextLinearAttention, "__call__", kda_attention)
    monkeypatch.setattr(
        language.Glm5NextLinearAttention, "_fused_in_proj", fused_projection
    )
    monkeypatch.setattr(language, "sparse_mla_decode_batched", batch_kernel)
    monkeypatch.setattr(language, "sparse_mla_decode", single)
    for cycle in range(3):
        block = mx.random.randint(0, 512, (size, width)).astype(mx.uint32)
        phase = plan[cycle * 3]
        reference.begin(phase, block)
        kda_reference.begin(phase, block)
        before = kernel_calls["batch"]
        out = host(block, cache=cache, return_hidden=True)
        mx.eval(out.logits)
        reference.seal()
        kda_reference.seal()
        assert kernel_calls["batch"] - before == len(
            targets
        ), "batch split-K did not execute"
        accepted = [n - 1 for n in phase.prefixes]
        rollback_rows(language, cache, out.gdn_states, accepted, width)
        if fault in ("cache", "rollback", "indexing") and cycle == 1:
            assert not faults, "KV fault reused"
            trace.fault_mutations.append(
                _inject_state_fault(cache, targets[-1][1], phase.prefixes[1], fault)
            )
            faults.append((fault, 1, "verify", targets[-1][1], 1))
        elif fault == "conv_history" and cycle == 1:
            assert not faults, "conv history fault reused"
            trace.fault_mutations.append(_inject_conv_history_fault(cache))
            faults.append((fault, 1, "verify", 2, 1))
        verify_refs = []
        for r, a in enumerate(accepted):
            tokens = block[r : r + 1, : a + 1]
            reference.begin_single(phase, r, tokens)
            kda_reference.begin_single(phase, r, tokens)
            before = kernel_calls["single"]
            ref = host(tokens, cache=rows[r]).logits
            reference.finish_single()
            kda_reference.finish_single()
            mx.eval(ref)
            assert kernel_calls["single"] - before == len(
                targets
            ), "reference split-K did not execute"
            verify_refs.append(ref)
        reference.finish()
        kda_reference.finish()
        _assert_cache_rows(
            cache, rows, cycle, "verify", overlap=_assert_glm_nonoverlap(host)
        )
        for step in range(2):
            tokens = mx.random.randint(0, 512, (size, 1)).astype(mx.uint32)
            phase = plan[cycle * 3 + step + 1]
            reference.begin(phase, tokens)
            kda_reference.begin(phase, tokens)
            before = kernel_calls["batch"]
            batched = host(tokens, cache=cache).logits
            mx.eval(batched)
            reference.seal()
            kda_reference.seal()
            assert kernel_calls["batch"] - before == len(
                targets
            ), "batch split-K did not execute"
            singles = []
            for r in range(size):
                value = tokens[r : r + 1]
                reference.begin_single(phase, r, value)
                kda_reference.begin_single(phase, r, value)
                before = kernel_calls["single"]
                singles.append(host(value, cache=rows[r]).logits)
                reference.finish_single()
                kda_reference.finish_single()
                assert kernel_calls["single"] - before == len(
                    targets
                ), "reference split-K did not execute"
            reference.finish()
            kda_reference.finish()
            mx.eval(batched, singles)
            _assert_cache_rows(
                cache,
                rows,
                cycle,
                phase.operation,
                overlap=_assert_glm_nonoverlap(host),
            )
    reference.complete()
    kda_reference.complete()
    assert kda_reference.captures == len(plan) * len(kda_targets)
    assert kda_reference.consumptions == size * kda_reference.captures
    assert reference.captures == len(plan) * len(targets)
    assert reference.consumptions == size * reference.captures


@pytest.mark.skipif(not _gpu_nax(), reason="NAX GPU required")
@pytest.mark.parametrize("depth", [1, 3, 7])
def test_batched_verify_rollback_preserves_rows_with_shared_split_k(
    split_k, monkeypatch, depth
):
    error = None
    try:
        _verify_rollback_continuity(monkeypatch, depth)
    except BaseException as exc:
        error = exc
        raise
    finally:
        trace = getattr(language, "_glm40_test_trace", None)
        if trace is not None:
            trace.finish(error)


@pytest.mark.skipif(not _gpu_nax(), reason="NAX GPU required")
@pytest.mark.parametrize("fault", ["cache", "indexing", "rollback"])
def test_shared_split_k_continuity_detects_state_faults(split_k, monkeypatch, fault):
    events = []
    error = None
    try:
        with pytest.raises(ContinuityMismatchError) as caught:
            _verify_rollback_continuity(monkeypatch, 7, fault, events)
        trace = getattr(language, "_glm40_test_trace", None)
        if trace is not None:
            trace.fault_observation = {
                "kind": caught.value.kind,
                "context": caught.value.context,
            }
        assert events == [
            (fault, 1, "verify", 7, 1)
        ], "fault injection did not execute once"
        assert caught.value.context == (1, "verify", 1)
        assert caught.value.kind == "KV values"
    except BaseException as exc:
        error = exc
        raise
    finally:
        trace = getattr(language, "_glm40_test_trace", None)
        if trace is not None:
            trace.fault_events = list(events)
            trace.finish(error)


@pytest.mark.skipif(not _gpu_nax(), reason="NAX GPU required")
def test_shared_split_k_continuity_detects_conv_history_fault(split_k, monkeypatch):
    events, error = [], None
    try:
        with pytest.raises(ContinuityMismatchError) as caught:
            _verify_rollback_continuity(monkeypatch, 7, "conv_history", events)
        trace = language._glm40_test_trace
        trace.fault_observation = {
            "kind": caught.value.kind,
            "context": caught.value.context,
        }
        assert caught.value.kind == "cache state" and caught.value.context == (
            1,
            "verify",
            1,
        )
        assert events == [("conv_history", 1, "verify", 2, 1)]
        failed = [r for r in trace.state_checks if not r["raw_bits_equal"]]
        assert len(failed) == 1
        assert (failed[0]["layer"], failed[0]["row"], failed[0]["field"]) == (2, 1, "0")
        assert failed[0]["first_coordinate"] == [0, 2, 0]
        assert failed[0]["actual_bits"] ^ failed[0]["reference_bits"] == 1
    except BaseException as exc:
        error = exc
        raise
    finally:
        trace = getattr(language, "_glm40_test_trace", None)
        if trace is not None:
            trace.fault_events = list(events)
            trace.finish(error)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
def test_continuity_zero_gate_checks_raw_bits(dtype):
    value = mx.array([1.0, 0.0], dtype=dtype)
    _continuity_close(value, value, (0, "verify", 0))
    negative_zero = mx.array([1.0, -0.0], dtype=dtype)
    with pytest.raises(ContinuityMismatchError, match="reference_bits"):
        _continuity_close(value, negative_zero, (0, "verify", 0))


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
def test_trace_array_handle_preserves_call_time_bits(dtype):
    value = mx.array([1.0, 2.0], dtype=dtype)
    snapshot = mx.array(value)
    assert snapshot is not value
    value[0] = 9.0
    _continuity_close(snapshot, mx.array([1.0, 2.0], dtype=dtype), (0, "verify", 0))


@pytest.mark.parametrize("fault", ["cache", "indexing", "rollback"])
def test_fault_control_rejects_failure_before_injection(monkeypatch, fault):
    def premature_failure(*args):
        raise ContinuityMismatchError("logits", (0, "verify", 0))

    monkeypatch.setattr(
        __import__(__name__, fromlist=["_"]),
        "_verify_rollback_continuity",
        premature_failure,
    )
    with pytest.raises(AssertionError, match="fault injection did not execute once"):
        test_shared_split_k_continuity_detects_state_faults(None, monkeypatch, fault)


@pytest.mark.skipif(not _gpu_nax(), reason="NAX GPU required")
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("width", [1, 8])
def test_scalar_attention_output_rounding_bound(split_k, dtype, width):
    mx.random.seed(width)
    before = 516
    length = before + width
    q = (mx.random.normal((1, 16, width, 128)) * 0.05).astype(dtype)
    kv = mx.random.normal((1, 1, length, 128)).astype(dtype)
    windows = mx.broadcast_to(mx.arange(128, dtype=mx.int32)[None], (width, 128))
    got = split_k.sparse_mla_decode(q, kv, windows, before, length // 4, 0.1).astype(
        mx.float32
    )
    for row in range(width):
        position = before + row
        tokens = list(range(512)) + list(
            range(position + 1 - (position + 1) % 4, position + 1)
        )
        keys = kv[0, 0, mx.array(tokens, mx.int32)].astype(mx.float32)
        ref = (
            mx.softmax((q[0, :, row].astype(mx.float32) @ keys.T) * 0.1, axis=-1) @ keys
        )
        rounding = 2.0 ** (-8 if dtype == mx.bfloat16 else -11)
        assert mx.all(
            mx.abs(got[0, :, row] - ref) <= mx.abs(ref) * rounding + 1e-3
        ).item()
        assert mx.all(mx.isfinite(got)).item()


@pytest.mark.parametrize("batched", [False, True])
def test_fp32_kv_uses_gathered_attention_without_second_indexer_update(
    split_k, monkeypatch, batched
):
    from tests import test_glm5_small_query_indexer as source

    small = importlib.import_module("mlx_vlm.models.glm5_next.small_indexer")
    source.cpu_paths.__wrapped__(monkeypatch, language, small)
    monkeypatch.setattr(language, "_DSA_DECODE_FUSED", True)
    attn = source.attention.__wrapped__(language)
    batch = 2 if batched else 1
    cache = source._caches(attn.indexer, [2053] * batch, batched=batched)
    cache[0].keys = cache[0].keys.astype(mx.float32)
    count = source._count_updates(monkeypatch, cache[1])
    monkeypatch.setattr(language, "sparse_mla_decode", source._forbidden)
    monkeypatch.setattr(language, "sparse_mla_decode_batched", source._forbidden)
    monkeypatch.setattr(
        language.Glm5NextSparseAttention,
        "_gathered_attention",
        lambda self, q, kv, idx: mx.zeros(
            (batch, 2, self.num_heads * self.v_head_dim), mx.float16
        ),
    )
    x, _ = source._inputs(batch, 2)
    attn._forward(x, None, cache)
    assert count == ["accumulate_windows", "update_and_fetch"]


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("rank", [2, 3, 4])
def test_boolean_mask_is_carried_to_kernel(split_k, monkeypatch, batch, rank):
    q = mx.zeros((batch, 16, 1, 128), mx.float16)
    kv = mx.zeros((batch, 1, 8, 128), mx.float16)
    windows = mx.zeros((batch, 1, 512), mx.int32)
    rows = mx.array([[2, 7, 0]] * batch, mx.int32)
    mask = mx.array([[True, False] * 4])
    if rank >= 3:
        mask = mx.broadcast_to(mask, (batch, 1, 8))
    if rank == 4:
        mask = mask[:, None]
    captured = {}

    def blocks(**kwargs):
        captured.update(kwargs)
        return [
            mx.zeros(shape, dtype)
            for shape, dtype in zip(kwargs["output_shapes"], kwargs["output_dtypes"])
        ]

    def merge(**kwargs):
        return [mx.zeros(kwargs["output_shapes"][0], mx.float16)]

    monkeypatch.setattr(split_k, "_get_batched", lambda: (blocks, merge))
    split_k.sparse_mla_decode_batched(q, kv, 8, windows, rows, 0.1, mask=mask)
    assert ("HAS_MASK", True) in captured["template"]
    assert captured["inputs"][-1].tolist() == [True, False] * (4 * batch)


@pytest.mark.skipif(not _gpu_nax(), reason="NAX GPU required")
@pytest.mark.parametrize("batched", [False, True])
def test_boolean_mask_changes_real_attention(split_k, batched):
    mx.random.seed(113)
    batch = 2 if batched else 1
    q = (mx.random.normal((batch, 16, 1, 128)) * 0.05).astype(BF16)
    kv = mx.random.normal((batch, 1, 516, 128)).astype(BF16)
    windows = mx.broadcast_to(mx.arange(128, dtype=mx.int32), (batch, 1, 128))
    mask = (mx.arange(516) % 2 == 0)[None, None]
    if batched:
        rows = mx.array([[129, 515, 0]] * batch, mx.int32)
        got = split_k.sparse_mla_decode_batched(
            q, kv, 516, windows, rows, 0.1, mask=mask
        )
    else:
        got = split_k.sparse_mla_decode(q, kv, windows[0], 515, 129, 0.1, mask=mask)
    for row in range(batch):
        keys = kv[row, 0, mx.arange(0, 512, 2)].astype(mx.float32)
        ref = (
            mx.softmax((q[row, :, 0].astype(mx.float32) @ keys.T) * 0.1, axis=-1) @ keys
        )
        assert mx.all(
            mx.abs(got[row, :, 0].astype(mx.float32) - ref)
            <= mx.abs(ref) * 2.0**-8 + 1e-3
        ).item()
