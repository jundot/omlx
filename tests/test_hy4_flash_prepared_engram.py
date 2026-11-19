# SPDX-License-Identifier: Apache-2.0
"""All-layer Engram fixtures only. Never open a real checkpoint or full model."""

import hashlib
import json
import os
from dataclasses import replace
from threading import Event, Thread, get_ident
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
import test_hy4_flash_model as graph_fixture
from mlx.utils import tree_flatten
from test_hy4_flash_engram import _LOOKUP, _TOKENS, _args
from test_hy4_flash_engram import artifact as _pinned_artifact

from omlx.patches.hy_v4_flash import prepared_engram as pe
from omlx.patches.hy_v4_flash.engram import EngramLayer
from omlx.patches.hy_v4_flash.model import DecoderLayer

artifact = _pinned_artifact  # reuse the existing real-Darwin-I/O pytest fixture
BLOCKS = (1, 11, 21, 31)


class _Tables:
    """Tiny finite BF16, deterministic and block-distinguishable."""

    def __init__(self, blocks, rows=72, width=5):
        self.reads = []
        self.bits = {}
        for b in blocks:
            x = mx.array(
                np.random.default_rng(b).normal(size=(rows, width)).astype("f")
            )
            self.bits[b] = np.array(x.astype(mx.bfloat16).view(mx.uint16))

    def gather(self, b, ids):
        self.reads.append((b, ids.copy()))
        return mx.array(self.bits[b][ids]).view(mx.bfloat16)


def _layers(tables):
    args = _args()
    layers = []
    for b in BLOCKS:
        rows, width = (
            tables.shape_for_block(b) if hasattr(tables, "shape_for_block") else (72, 5)
        )
        args.engram_layer_head_vocab_sizes[str(b + 1)] = [[13, 17], [19, rows - 49]]
        args.engram_n_embed_per_ngram = width * args.engram_n_head_per_ngram
        layer = EngramLayer(args, b)
        layer.load_weights(
            [("lookup_table", mx.array(_LOOKUP, mx.int32))], strict=False
        )
        layer.bind_tables(tables)
        layers.append(layer)
    return SimpleNamespace(
        layers=[SimpleNamespace(engram_layer=layer) for layer in layers]
    ), layers


@pytest.mark.parametrize("fp8", [True, False])
def test_four_layers_one_input_read_no_sync_or_hash_in_decoder(monkeypatch, fp8):
    tables = _Tables(BLOCKS)
    model, layers = _layers(tables)
    tokens = mx.array(_TOKENS)
    streams = mx.array(np.random.default_rng(34).normal(size=(2, 3, 4, 8)).astype("f"))
    for layer in layers:
        layer.fp8_embedding = fp8
    expected = [np.array(layer(streams, tokens)) for layer in layers]
    expected_rows, tables.reads = tables.reads, []
    with pe.install_prepared_engram(model) as prep:
        reads, compressions = [], []
        asarray, compress = np.asarray, pe._compress

        def spy_array(x, *args, **kw):
            if isinstance(x, mx.array):
                reads.append(x)
            return asarray(x, *args, **kw)

        def spy_compress(*args):
            compressions.append(1)
            return compress(*args)

        with monkeypatch.context() as m:
            m.setattr(np, "asarray", spy_array)
            m.setattr(pe, "_compress", spy_compress)
            m.setattr(
                EngramLayer, "hash_ngrams", lambda *a: pytest.fail("in-decoder hash")
            )
            with prep.batch(tokens) as batch:
                assert len(reads) == 1 and reads[0] is tokens
                assert len(compressions) == 1  # four independently loaded equal maps
                assert [b for b, _ in tables.reads] == list(BLOCKS)
                got = [layer(streams, batch) for layer in layers]
                mx.eval(*got)
                assert len(reads) == 1  # no readback for any intermediate layer
                with pytest.raises(RuntimeError, match="no current"):
                    batch.take(layers[0])
        with pytest.raises(RuntimeError, match="no current"):
            batch.take(layers[-1])  # no cross-forward row cache
        for actual, want in zip(got, expected):
            np.testing.assert_array_equal(np.array(actual), want)
        for (b, ids), (b0, ids0) in zip(tables.reads, expected_rows):
            assert b == b0
            np.testing.assert_array_equal(ids, ids0)
        assert prep.stats()["input_host_reads"] == 1
    assert not hasattr(model, "_engram_preparer")


def test_replaced_dissimilar_maps_unsigned_oov_and_pad(monkeypatch):
    tables = _Tables(BLOCKS)
    model, layers = _layers(tables)
    unsigned = _TOKENS.astype(np.uint64)
    unsigned[1, 1, 0] = 2**64 - 1  # hash as PAD, but don't mask this position
    tokens = mx.array(unsigned, dtype=mx.uint64)
    with pe.install_prepared_engram(model) as prep:
        layers[1].load_weights(
            [("lookup_table", mx.arange(9, dtype=mx.int32))], strict=False
        )
        rows = [layer.hash_ngrams(unsigned) + layer._offsets for layer in layers]
        compress = pe._compress
        groups = []

        def spy(*args):
            groups.append(1)
            return compress(*args)

        monkeypatch.setattr(pe, "_compress", spy)
        with prep.batch(tokens) as batch:
            assert len(groups) == 2
            for layer, ids in zip(layers, rows):
                x = batch.take(layer)
                assert x.dtype == mx.bfloat16
                np.testing.assert_array_equal(
                    np.asarray(x.view(mx.uint16)), tables.bits[layer.block][ids]
                )
                assert batch.tokens is tokens


def test_refusals_before_read_and_reversible_install(monkeypatch):
    tables = _Tables(BLOCKS)
    model, layers = _layers(tables)
    tokens = mx.array(_TOKENS)
    size = len(layers) * 2 * 3 * 4 * 5 * 2
    with (
        pytest.raises(ValueError, match="max_bytes"),
        pe.install_prepared_engram(model, max_bytes=pe.MAX_PREPARED_BYTES + 1),
    ):
        pytest.fail("exceeded hard bound")
    with pe.install_prepared_engram(model, max_bytes=size - 1) as prep:
        with pytest.raises(ValueError, match="chunk"), prep.batch(tokens):
            pass
        assert prep.stats()["input_host_reads"] == 0 and not tables.reads
        with (
            pytest.raises(RuntimeError, match="already installed"),
            pe.install_prepared_engram(model),
        ):
            pass
    with pytest.raises(RuntimeError, match="closed"), prep.batch(tokens):
        pass
    model._engram_preparer = None
    with (
        pytest.raises(ValueError, match="decoder stopped"),
        pe.install_prepared_engram(model, max_bytes=size) as prep,
    ):
        for inp, error in [
            (tokens.astype(mx.float32), "integer"),
            (tokens[..., :2], "integer"),
            (-tokens, "negative"),
        ]:
            with pytest.raises(ValueError, match=error), prep.batch(inp):
                pass
        with monkeypatch.context() as m:
            m.setattr(pe.residency, "_MAX_GATHER_BYTES", size // 4 - 1)
            with pytest.raises(ValueError, match="chunk"), prep.batch(tokens):
                pass
        assert not tables.reads
        with prep.batch(tokens) as batch:
            with pytest.raises(RuntimeError, match="active"), prep.batch(tokens):
                pass
            raise ValueError("decoder stopped")
    assert model._engram_preparer is None and prep.stats()["closed"]
    assert batch.tokens is None
    with pe.install_prepared_engram(model) as prep:
        layers[0].bind_tables(_Tables(BLOCKS))
        with pytest.raises(RuntimeError, match="binding changed"), prep.batch(tokens):
            pass


@pytest.mark.parametrize("dtype,width", [(mx.float32, 5), (mx.bfloat16, 1)])
def test_invalid_provider_rows_are_not_cast_or_reshaped(dtype, width):
    tables = SimpleNamespace(
        gather=lambda b, ids: mx.ones((*ids.shape, width), dtype=dtype)
    )
    model, layers = _layers(tables)
    with (
        pe.install_prepared_engram(model) as prep,
        pytest.raises(ValueError, match="original BF16"),
        prep.batch(mx.array(_TOKENS)) as batch,
    ):
        batch.take(layers[0])


@pytest.mark.parametrize("public_dtypes", [False, True], ids=["fp32", "bf16"])
def test_every_residual_block_cache_and_logit_matches_baseline(
    monkeypatch, public_dtypes
):
    # Extend the existing tiny, deterministically weighted decoder fixture to
    # four Engrams. Capture every block and residual, not only last-token argmax.
    initial = graph_fixture._args()
    args = replace(
        initial,
        num_hidden_layers=4,
        layer_types=initial.layer_types * 2,
        mlp_layer_types=initial.mlp_layer_types * 2,
        engram_layer_ids=[1, 2, 3, 4],
        engram_layer_head_vocab_sizes={
            str(i): [[11, 13], [17, 19]] for i in range(1, 5)
        },
    )
    monkeypatch.setattr(graph_fixture, "_args", lambda: args)
    model, _ = graph_fixture._graph(public_dtypes)
    tables = _Tables(range(4), 60, 4)
    model.set_engram_tables(tables)
    keys = [k for k, _ in tree_flatten(model.parameters())]
    captured = []

    def capture(cls):
        call = cls.__call__

        def wrapped(*a, **kw):
            out = call(*a, **kw)
            captured.append(out)
            return out

        return wrapped

    for cls in [EngramLayer, DecoderLayer]:
        monkeypatch.setattr(cls, "__call__", capture(cls))

    def replay():
        outputs = []
        caches = [model.make_cache(), model.make_cache()]
        cases = [
            (None, [[47, 3, 8, 3, 22], [9, 19, 0, 12, 4]], 0),
            (0, [[1, 47, 3]], 0),
            (1, [[20, 8]], 0),
            (0, [[21]], 1),
            (1, [[3, 1, 8]], 2),
            (0, [[9, 7]], 0),
        ]
        for req, chunk, keep in cases:
            cache = None if req is None else caches[req]
            logits = model(mx.array(chunk), cache=cache, logits_to_keep=keep)
            assert len(captured) == 8
            captured.append(logits)
            if cache is not None:
                captured.append(cache.history)
                assert cache.batch_size == 1 and all(
                    kv.offset == cache.offset for kv in cache.attention
                )
                for kv in cache.attention:
                    captured.extend(kv.state)
            mx.eval(*captured)
            for x in captured:
                arr = np.array(
                    x.view(mx.uint16)
                    if isinstance(x, mx.array) and x.dtype == mx.bfloat16
                    else x
                )
                outputs.append(arr)
            assert np.isfinite(np.array(logits)).all()
            captured.clear()
        return outputs

    expected = replay()
    expected_rows, tables.reads = tables.reads, []
    with pe.install_prepared_engram(model) as prep:
        assert keys == [k for k, _ in tree_flatten(model.parameters())]
        actual = replay()
        assert prep.stats()["input_host_reads"] == 6
    assert len(expected) == len(actual)
    for ref, got in zip(expected, actual):
        assert ref.shape == got.shape and ref.dtype == got.dtype
        assert ref.tobytes() == got.tobytes()
    assert len(expected_rows) == len(tables.reads) == 24
    for (b, ref), (block, got) in zip(expected_rows, tables.reads):
        assert b == block
        np.testing.assert_array_equal(ref, got)


def test_parallel_exact_preads_overlap_first_compute_and_keep_counters(
    artifact, monkeypatch
):
    # Four tiny, independently pinned BF16 safetensors. Real diskutil,
    # F_NOCACHE, fd identity, mlock and mincore; patch only syscall observation.
    artifact.verify()
    tokens = mx.array(_TOKENS)
    with artifact.open() as p:
        model, layers = _layers(p)
        expected = {
            layer.block: layer.hash_ngrams(_TOKENS) + layer._offsets for layer in layers
        }
        first = p.stats()
        by_fd = {t.fd: b for b, t in p._files.items()}
        reads = {b: [] for b in BLOCKS}
        entered = {b: Event() for b in BLOCKS[1:]}
        release = Event()
        pread = os.pread
        consumer = get_ident()

        def spy(fd, n, offset):
            b = by_fd.get(fd)
            if b is not None:
                assert b != BLOCKS[0] and get_ident() != consumer
                reads[b].append((offset, n, get_ident()))
                entered[b].set()
                assert release.wait(10), "late reads could not overlap first compute"
            return pread(fd, n, offset)

        monkeypatch.setattr(os, "pread", spy)
        with pe.install_prepared_engram(model) as prep, prep.batch(tokens) as batch:
            try:
                assert all(
                    ev.wait(10) for ev in entered.values()
                )  # all three live together
                selected = {BLOCKS[0]: batch.take(layers[0])}
                mx.eval(
                    selected[BLOCKS[0]].astype(mx.float32).sum()
                )  # can finish before late FDs
            finally:
                release.set()
            for layer in layers[1:]:
                selected[layer.block] = batch.take(layer)
        stats = p.stats()
        assert stats["first_locked_bytes"] == first["first_locked_bytes"] > 0
        assert stats["first_pages"] == stats["first_resident_pages"] > 0
        assert len({reads[b][0][-1] for b in BLOCKS[1:]}) == 3
        for b, ids in expected.items():
            record = stats["blocks"][str(b)]
            unique = np.unique(ids)
            size = 0 if b == BLOCKS[0] else len(unique)
            assert record["requested_rows"] == ids.size
            assert record["unique_rows"] == len(unique) < ids.size
            assert (record["pread_calls"], record["pread_bytes"]) == (size, size * 192)
            assert [(off, n) for off, n, _ in reads[b]] == (
                []
                if b == BLOCKS[0]
                else [(record["source_offset"] + int(row) * 192, 192) for row in unique]
            )
    for b, val in selected.items():
        assert val.dtype == mx.bfloat16
        np.testing.assert_array_equal(
            np.array(val.view(mx.uint16)), artifact.arrays[b][expected[b]]
        )
    assert stats["blocks"]["1"]["mode"] == "mlock" and p.stats()["closed"]


@pytest.mark.parametrize("fault", ["short", "replace"])
def test_failed_late_read_drains_workers_and_closes_verified_provider(
    artifact, monkeypatch, fault
):
    artifact.verify()
    with artifact.open() as p:
        model, layers = _layers(p)
        fds = [t.fd for t in p._files.values()]
        bad = p._files[31].fd
        if fault == "short":
            pread = os.pread
            monkeypatch.setattr(
                os, "pread", lambda fd, n, off: b"" if fd == bad else pread(fd, n, off)
            )
        else:
            path = artifact.paths[31]
            new = path.with_suffix(".replacement")
            new.write_bytes(path.read_bytes())
            os.replace(new, path)
        tokens = mx.array(_TOKENS)
        with pe.install_prepared_engram(model) as prep:
            with (
                pytest.raises(
                    (OSError, ValueError),
                    match="short positional read|changed or replaced",
                ),
                prep.batch(tokens) as batch,
            ):
                batch.take(layers[0])  # never consume the bad block
            assert p.stats()["closed"]
            for fd in fds:
                with pytest.raises(OSError):
                    os.fstat(fd)
            with (
                pytest.raises(RuntimeError, match="provider is closed"),
                prep.batch(tokens),
            ):
                pass


def test_verified_rows_prefix_dma_and_all_logits_bit_exact(artifact, monkeypatch):
    # Keep the tiny fixture's signed BF16 mantissas, but put exponents in a
    # finite range for numerical tests. Repin these TEST files before verify.
    for b, path in artifact.paths.items():
        spec = artifact.doc["external_tables"][str(b)]
        payload = ((artifact.arrays[b] & 0x807F) | 0x3F00).tobytes()
        data = path.read_bytes()[: spec["source_offset"]] + payload
        path.write_bytes(data)
        spec["payload_sha256"] = hashlib.sha256(payload).hexdigest()
        spec["file_sha256"] = hashlib.sha256(data).hexdigest()
        monkeypatch.setitem(pe.residency._PINS, b, (len(data), spec["file_sha256"]))
    artifact.manifest.write_text(json.dumps(artifact.doc))
    artifact.verify()
    # Five tiny decoder blocks: the first computes before any Engram. Map the
    # next four to the independently verified hot/late fixture FDs, without
    # constructing the public decoder shape or using any public weights.
    base = graph_fixture._args()
    args = replace(
        base,
        num_hidden_layers=5,
        layer_types=(base.layer_types * 3)[:5],
        mlp_layer_types=(base.mlp_layer_types * 3)[:5],
        engram_n_embed_per_ngram=192,
        engram_layer_ids=[2, 3, 4, 5],
        engram_layer_head_vocab_sizes={
            str(i): [[13, 17], [19, 256 + b - 49]]
            for i, b in enumerate(BLOCKS, start=2)
        },
    )
    monkeypatch.setattr(graph_fixture, "_args", lambda: args)
    model, _ = graph_fixture._graph(True)
    for b, decoder in zip(BLOCKS, model.layers[1:]):
        decoder.engram_layer.block = b
        decoder.engram_layer.layer_id = b + 1
    inputs = [mx.array([[47, 9, 3], [12, 8, 1]]), mx.array([[5], [22]])]
    with artifact.open() as provider:
        model.set_engram_tables(provider)
        baseline_cache = model.make_cache()
        expected = [np.array(model(x, cache=baseline_cache)) for x in inputs]
        assert all(np.isfinite(value).all() for value in expected)
        counters = provider.stats()["blocks"]
        late = provider._files[31].fd
        reading, allow_disk = Event(), Event()
        pread, async_eval = os.pread, mx.async_eval
        submitted = []

        def delayed(fd, n, off):
            if fd == late:
                reading.set()
                assert allow_disk.wait(10), (
                    "decoder prefix was not submitted during late IO"
                )
            return pread(fd, n, off)

        def submit(streams):
            try:
                assert reading.wait(10)
                async_eval(streams)
                mx.eval(
                    streams
                )  # TEST ONLY: actual prefix can finish with the late read held
                submitted.append(streams.shape)
            finally:
                allow_disk.set()

        monkeypatch.setattr(os, "pread", delayed)
        monkeypatch.setattr(mx, "async_eval", submit)
        cache = model.make_cache()
        with pe.install_prepared_engram(model) as prep:
            for x, ref in zip(inputs, expected):
                reading.clear()
                allow_disk.clear()
                out = np.array(model(x, cache=cache))
                assert out.dtype == ref.dtype and out.tobytes() == ref.tobytes()
            assert submitted == [
                (*x.shape, args.hc_mult, args.hidden_size) for x in inputs
            ]
            assert prep.stats()["input_host_reads"] == len(inputs)
        np.testing.assert_array_equal(
            np.array(cache.history), np.array(baseline_cache.history)
        )
        for block, after in provider.stats()["blocks"].items():
            for key in ("requested_rows", "unique_rows", "pread_calls", "pread_bytes"):
                assert after[key] == 2 * counters[block][key]


def test_concurrent_provider_close_waits_for_reads(artifact, monkeypatch):
    artifact.verify()
    with artifact.open() as p:
        started, reading, release, closed = (Event() for _ in range(4))
        block_fd = p._files[21].fd
        pread = os.pread

        def delay(fd, n, offset):
            if fd == block_fd:
                reading.set()
                assert release.wait(10)
            return pread(fd, n, offset)

        def close():
            started.set()
            p.close()
            closed.set()

        monkeypatch.setattr(os, "pread", delay)
        model, layers = _layers(p)
        with (
            pe.install_prepared_engram(model) as prep,
            prep.batch(mx.array(_TOKENS)) as batch,
        ):
            assert reading.wait(10)
            closer = Thread(target=close)
            closer.start()
            try:
                assert started.wait(10) and not closed.is_set()
                p._files[21].unchanged()  # still a live verified fd
            finally:
                release.set()
                closer.join(10)
            assert closed.is_set()
            # Original compact copies may be consumed after the provider closed.
            for layer in layers:
                assert batch.take(layer).dtype == mx.bfloat16
