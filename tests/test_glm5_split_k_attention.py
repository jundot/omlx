# SPDX-License-Identifier: Apache-2.0
"""Split-K contracts on CPU; numerical and state parity on NAX Metal."""

import copy
import importlib
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.patches.mlx_lm_mtp import batch_generator as bg
from omlx.patches.mlx_vlm_glm5_next_compat import apply_mlx_vlm_glm5_next_compat_patch
from omlx.patches.mlx_vlm_mtp import glm5_next_vlm_runtime
from omlx.patches.mlx_vlm_mtp.glm5_next_batch_rollback import rollback_rows
from tests.test_glm5_next_mtp import TINY_TEXT_CONFIG

apply_mlx_vlm_glm5_next_compat_patch()
from mlx_vlm.models.cache import CacheList  # noqa: E402
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


@pytest.mark.skipif(not _gpu_nax(), reason="NAX GPU required")
@pytest.mark.parametrize("depth", [1, 3, 7])
def test_tiny_model_batched_verify_rollback_matches_rows_alone(
    split_k, monkeypatch, depth
):
    mx.random.seed(977 + depth)
    monkeypatch.setattr(language, "_DSA_DECODE_FUSED", True)
    host = _host()
    lengths = [2051, 2057]
    size = len(lengths)
    rows = _prefill_rows(host, lengths)
    cache = bg._merge_row_caches([copy.deepcopy(r) for r in rows])
    calls = []
    _spy(monkeypatch, language, "sparse_mla_decode_batched", calls)
    for cycle in range(3):
        block = mx.random.randint(0, 512, (size, depth + 1)).astype(mx.uint32)
        out = host(block, cache=cache, return_hidden=True)
        mx.eval(out.logits)
        accepted = [(r + cycle) % (depth + 1) for r in range(size)]
        rollback_rows(language, cache, out.gdn_states, accepted, depth + 1)
        for r, a in enumerate(accepted):
            ref = host(block[r : r + 1, : a + 1], cache=rows[r]).logits
            mx.eval(ref)
            _close(out.logits[r : r + 1, : a + 1], ref, (cycle, r, "verify"))
        _assert_pool_rows(cache, rows)
        for step in range(2):
            tokens = mx.random.randint(0, 512, (size, 1)).astype(mx.uint32)
            batched = host(tokens, cache=cache).logits
            singles = [
                host(tokens[r : r + 1], cache=rows[r]).logits for r in range(size)
            ]
            mx.eval(batched, singles)
            for r in range(size):
                _close(batched[r : r + 1], singles[r], (cycle, step, r))
            _assert_pool_rows(cache, rows)
    assert calls, "batched fused decode never engaged"


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
