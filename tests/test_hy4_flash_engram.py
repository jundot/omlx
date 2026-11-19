# SPDX-License-Identifier: Apache-2.0
"""Tiny synthetic Engram oracles and Darwin residency tests. No model assets."""

import hashlib
import json
import mmap
import os
import struct
import sys
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from omlx.patches.hy_v4_flash import residency
from omlx.patches.hy_v4_flash.engram import EngramLayer, _fp8_round_trip

_TOKENS = np.array(
    [[[7, 8, 8], [0, 7, 8], [2, 0, 7]], [[8, 7, 2], [20, 7, 2], [5, 4, 3]]],
    dtype=np.int32,
)
_LOOKUP = [0, 1, 0, 2, 3, 2, 1, 4, 4]
# Within-head ids from scalar int64 XOR with HF's PCG64 draws (compressed size 5).
_IDS = {
    1: [
        [[5, 16, 17, 13], [10, 15, 3, 21], [0, 0, 18, 19]],
        [[5, 16, 2, 2], [5, 16, 2, 2], [8, 13, 2, 6]],
    ],
    11: [
        [[10, 13, 9, 20], [12, 2, 13, 6], [0, 0, 13, 0]],
        [[10, 13, 2, 3], [10, 13, 2, 3], [10, 3, 9, 1]],
    ],
}


def _args():
    return SimpleNamespace(
        hidden_size=8,
        hc_mult=4,
        engram_max_ngram_size=3,
        engram_n_head_per_ngram=2,
        engram_n_embed_per_ngram=10,
        engram_pad_id=8,
        engram_seed=13,
        engram_fp8_embedding=True,
        engram_lookup_table_size=9,
        engram_layer_head_vocab_sizes={
            "2": [[13, 17], [19, 23]],
            "12": [[13, 17], [19, 23]],
        },
    )


@pytest.mark.parametrize("block", [1, 11])
def test_compressed_hash_seed_padding_and_local_offsets(block):
    layer = EngramLayer(_args(), block)
    tokens = mx.array(_TOKENS)
    identity_ids = layer.hash_ngrams(tokens)
    layer.load_weights([("lookup_table", mx.array(_LOOKUP, mx.int32))], strict=False)
    np.testing.assert_array_equal(layer.hash_ngrams(tokens), _IDS[block])
    assert not np.array_equal(
        identity_ids, _IDS[block]
    )  # recomputed when the loaded map replaces the initial map
    captured = []

    def gather(b, ids):
        captured.append((b, ids))
        return mx.ones((*ids.shape, 5), dtype=mx.bfloat16)

    layer.bind_tables(SimpleNamespace(gather=gather))
    layer(mx.ones((2, 3, 4, 8)), tokens)
    assert len(captured) == 1 and captured[0][0] == block
    np.testing.assert_array_equal(
        captured[0][1], np.array(_IDS[block]) + [0, 13, 30, 49]
    )
    assert not any(
        "embedding" in k or k.startswith("_")
        for k, _ in tree_flatten(layer.parameters())
    )


def test_requires_bf16_provider_and_integer_ids():
    layer, streams, tokens = (
        EngramLayer(_args(), 1),
        mx.ones((2, 3, 4, 8)),
        mx.array(_TOKENS),
    )
    with pytest.raises(RuntimeError, match="bind"):
        layer(streams, tokens)
    layer.bind_tables(SimpleNamespace(gather=lambda b, ids: mx.ones((*ids.shape, 5))))
    with pytest.raises(ValueError, match="BF16"):
        layer(streams, tokens)
    with pytest.raises(ValueError, match="negative"):
        layer.hash_ngrams(-tokens)
    with pytest.raises(ValueError, match="integer"):
        layer.hash_ngrams(tokens.astype(mx.float32))


def _fp8_reference(x):
    # Enumerate all 127 positive finite E4M3FN codes, including subnormals.
    # Choose the nearest endpoint (ties to even), not an MLX cast.
    code = np.arange(127)
    values = np.where(
        code < 8, code / 512.0, (1 + (code % 8) / 8) * 2.0 ** (code // 8 - 7)
    ).astype("f")
    scale = np.maximum(
        np.max(np.abs(x), axis=-1, keepdims=True), np.float32(1e-12)
    ) / np.float32(448)
    normalized = np.minimum(np.abs(x / scale), 448)
    hi = np.searchsorted(values, normalized)
    lo = np.maximum(0, hi - 1)
    left, right = normalized - values[lo], values[hi] - normalized
    nearest = np.where((left < right) | ((left == right) & (lo % 2 == 0)), lo, hi)
    q = np.copysign(values[nearest], x) * scale
    return x + (q - x)


def test_per_vector_e4m3fn_edges_and_scales():
    x = np.array(
        [
            [448, 0, 1 / 1024, 3 / 1024, 432, 2.375, -17],
            [1.0, -0.4999, 0.3, 0.1, -0.2, 0.0, 0.92],
            [2e-14, -3e-15, 1e-16, 0, 1e-17, 2e-19, 1e-13],
            [0] * 7,
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(
        np.asarray(_fp8_round_trip(mx.array(x))), _fp8_reference(x)
    )


def _fixture_layer(dtype):
    layer, rng = EngramLayer(_args(), 1), np.random.default_rng(132)
    weights = {}
    for name, p in tree_flatten(layer.parameters()):
        if name == "lookup_table":
            weights[name] = mx.array(_LOOKUP, mx.int32)
        else:
            weights[name] = mx.array(rng.normal(size=p.shape).astype("f") * 0.5).astype(
                dtype
            )
    layer.load_weights(list(weights.items()))
    table = mx.array(rng.normal(size=(72, 5)).astype("f")).astype(mx.bfloat16)
    streams = mx.array(rng.normal(size=(2, 3, 4, 8)).astype("f")).astype(dtype)
    layer.bind_tables(
        SimpleNamespace(gather=lambda block, ids: table[mx.array(ids.astype(np.int32))])
    )
    return layer, weights, table, streams


def test_four_streams_match_independent_float_reference():
    layer, weights, table, streams = _fixture_layer(mx.float32)
    w = {k: np.asarray(v) for k, v in weights.items()}
    rows = np.array(_IDS[1]) + [0, 13, 30, 49]
    vectors = _fp8_reference(np.asarray(table.astype(mx.float32))[rows])

    def rms(x):
        return x / np.sqrt(
            np.mean(x * x, axis=-1, keepdims=True) + np.finfo(np.float32).eps
        )

    def proj(x, name):
        return x @ w[name + ".weight"].T + w[name + ".bias"]

    x = np.concatenate(
        [
            rms(vectors[:, :, h]) * w[f"lookup_per_head_norms.{h}.weight"]
            for h in range(4)
        ],
        axis=-1,
    )
    signal = rms(proj(x, "value_proj")) * w["unified_pre_gamma"]
    branches = []
    for h in range(4):
        k = rms(proj(x, f"key_projs.{h}")) * w[f"norm_key.{h}.weight"]
        q = rms(np.asarray(streams)[:, :, h]) * w[f"norm_query.{h}.weight"]
        dot = (k * q).sum(-1) / np.sqrt(np.float32(8))
        rooted = np.sign(dot) * np.sqrt(np.maximum(np.abs(dot), 1e-6))
        gate = 1 / (1 + np.exp(-rooted))
        branches.append(
            gate[..., None] * signal + signal * w[f"unified_per_hc_scales.{h}"]
        )
    expected = np.stack(branches, axis=2)
    expected[_TOKENS[..., 0] == 8] = 0
    out = layer(streams, mx.array(_TOKENS))
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-5, atol=2e-6)
    assert np.any(
        expected[1, 1] != 0
    )  # OOV remaps to PAD for hashing but isn't a masked input position.


def test_bf16_fp8_and_gates_against_torch():
    # Additional BF16 cross-backend check where the optional reference runtime is installed.
    torch = pytest.importorskip("torch")
    if getattr(torch.__spec__, "origin", None) is None:
        pytest.skip("only the oMLX torch stub is installed")
    from torch.nn import functional

    layer, weights, table, streams = _fixture_layer(mx.bfloat16)

    def tensor(x):
        return torch.tensor(np.asarray(x.astype(mx.float32)), dtype=torch.bfloat16)

    w = {k: tensor(v) for k, v in weights.items() if k != "lookup_table"}
    rows = np.array(_IDS[1]) + [0, 13, 30, 49]
    e = tensor(table)[torch.tensor(rows)]
    scale = e.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 448.0
    q = (e / scale).clamp(-448, 448).to(torch.float8_e4m3fn).to(e.dtype) * scale
    e = e + (q - e)
    got = _fp8_round_trip(table[mx.array(rows.astype(np.int32))])
    np.testing.assert_array_equal(np.asarray(got.astype(mx.float32)), e.float().numpy())

    def norm(x, key=None):
        return functional.rms_norm(
            x, (x.shape[-1],), w[key + ".weight"] if key else None
        )

    def linear(x, key):
        return functional.linear(x, w[key + ".weight"], w[key + ".bias"])

    x = torch.cat(
        [norm(e[:, :, h], f"lookup_per_head_norms.{h}") for h in range(4)], dim=-1
    )
    s = w["unified_pre_gamma"] * norm(linear(x, "value_proj"))
    gates = []
    for h in range(4):
        k = norm(linear(x, f"key_projs.{h}"), f"norm_key.{h}")
        query = norm(tensor(streams)[:, :, h], f"norm_query.{h}")
        g = (k * query).sum(-1) * (8**-0.5)
        gates.append((g.abs().clamp_min(1e-6).sqrt() * g.sign()).sigmoid())
    result = torch.stack(gates, dim=2).unsqueeze(-1) * s.unsqueeze(2)
    result += torch.stack(
        [w[f"unified_per_hc_scales.{h}"] * s for h in range(4)], dim=2
    )
    result[torch.tensor(_TOKENS[..., 0] == 8)] = 0
    out = layer(streams, mx.array(_TOKENS))
    assert out.dtype == mx.bfloat16
    np.testing.assert_array_equal(
        np.asarray(out.astype(mx.float32)), result.float().numpy()
    )


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    if sys.platform != "darwin":
        pytest.skip("internal Data SSD test requires macOS")
    first, late = tmp_path / "original", tmp_path / "ssd"
    first.mkdir()
    late.mkdir()
    arrays, paths, pins, tables = {}, {}, {}, {}
    for index, block in enumerate((1, 11, 21, 31)):
        shard = f"model-{4 - index:05d}-of-00148.safetensors"
        key = f"model.layers.{block}.engram_layer.multi_head_embedding.embedding.weight"
        # Raw bits, not a conversion through float16 (includes non-finite BF16 representations).
        bits = np.random.default_rng(block).integers(
            0, 65536, size=(256 + block, 96), dtype="<u2"
        )
        payload = bits.tobytes()
        header = json.dumps(
            {
                key: {
                    "dtype": "BF16",
                    "shape": list(bits.shape),
                    "data_offsets": [0, len(payload)],
                }
            },
            separators=(",", ":"),
        ).encode()
        header += b" " * (-len(header) % 8)
        prefix = struct.pack("<Q", len(header)) + header
        data = prefix + payload
        path = (first if index == 0 else late) / shard
        path.write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        pins[block] = (len(data), sha)
        arrays[block], paths[block] = bits, path
        tables[str(block)] = dict(
            block=block,
            source_key=key,
            source_shard=shard,
            source_offset=len(prefix),
            shape=list(bits.shape),
            dtype="BF16",
            row_stride=192,
            byte_length=len(payload),
            file_size=len(data),
            file_sha256=sha,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
        )
    # Change only the public length/digest pins for tiny inputs. No test-only
    # runtime flags: real diskutil, FD identity, positional I/O, mlock and mincore.
    monkeypatch.setattr(residency, "_PINS", pins)
    doc = dict(
        schema="hy4-flash-oqh/v1",
        text_only=True,
        source=dict(
            repo="tencent/Hy4-flash",
            revision="d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959",
        ),
        external_tables=tables,
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(doc))
    kwargs = dict(first_root=first, late_root=late, receipt=tmp_path / "receipt.json")
    return SimpleNamespace(
        doc=doc,
        manifest=manifest,
        kwargs=kwargs,
        arrays=arrays,
        paths=paths,
        verify=lambda: residency.prepare_table_receipt(manifest, **kwargs),
        open=lambda: residency.open_tables(manifest, **kwargs),
    )


def test_only_first_wired_late_reads_exact_rows_and_close(artifact, monkeypatch):
    a = artifact
    receipt = a.verify()
    assert (receipt.stat().st_mode & 0o777) == 0o600
    with pytest.raises(FileExistsError):
        a.verify()
    inodes = {p.stat().st_ino: b for b, p in a.paths.items()}
    pread, mmap_fn, reads, maps = os.pread, mmap.mmap, [], []

    def spy_read(fd, n, offset):
        b = inodes.get(os.fstat(fd).st_ino)
        if b is not None:
            reads.append((b, offset, n))
        return pread(fd, n, offset)

    def spy_map(fd, n, *args, **kw):
        m = mmap_fn(fd, n, *args, **kw)
        maps.append((inodes[os.fstat(fd).st_ino], m))
        return m

    monkeypatch.setattr(os, "pread", spy_read)
    with monkeypatch.context() as m:
        m.setattr(mmap, "mmap", spy_map)
        provider = a.open()
    with provider:
        assert len(maps) == 1 and maps[0][0] == 1
        for b, off, n in reads:
            assert off + n <= a.doc["external_tables"][str(b)]["source_offset"]
        initial = provider.stats()
        assert initial["first_pages"] == initial["first_resident_pages"] > 1
        size = a.paths[1].stat().st_size
        assert size <= initial["first_locked_bytes"] < size + mmap.PAGESIZE
        assert all(v["pread_bytes"] == 0 for v in initial["blocks"].values())
        ids = np.array([[13, 2, 13], [255, 2, 255]], dtype=np.int64)
        results = {}
        for b in a.paths:
            assert provider.shape_for_block(b) == a.arrays[b].shape
            reads.clear()
            os.lseek(provider._files[b].fd, 5, os.SEEK_SET)
            results[b] = provider.gather(b, ids)
            assert os.lseek(provider._files[b].fd, 0, os.SEEK_CUR) == 5
            assert results[b].dtype == mx.bfloat16
            off = a.doc["external_tables"][str(b)]["source_offset"]
            assert reads == (
                [] if b == 1 else [(b, off + row * 192, 192) for row in [2, 13, 255]]
            )
            v = provider.stats()["blocks"][str(b)]
            assert (
                v["requested_rows"],
                v["unique_rows"],
                v["pread_calls"],
                v["pread_bytes"],
            ) == (6, 3, 0 if b == 1 else 3, 0 if b == 1 else 576)
        fds = [t.fd for t in provider._files.values()]
    provider.close()  # idempotent
    assert maps[0][1].closed and provider.stats()["first_locked_bytes"] == 0
    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)
    with pytest.raises(RuntimeError, match="closed"):
        provider.gather(1, ids)
    for b, out in results.items():
        # Even a lazily evaluated result outlives the descriptor/mapping.
        np.testing.assert_array_equal(np.asarray(out.view(mx.uint16)), a.arrays[b][ids])


def test_first_table_pread_mode_never_maps_and_reads_exact_rows(artifact, monkeypatch):
    a = artifact
    a.verify()
    inodes = {p.stat().st_ino: b for b, p in a.paths.items()}
    pread, reads, maps = os.pread, [], []

    def spy_read(fd, n, offset):
        b = inodes.get(os.fstat(fd).st_ino)
        if b is not None:
            reads.append((b, offset, n))
        return pread(fd, n, offset)

    monkeypatch.setattr(os, "pread", spy_read)
    monkeypatch.setattr(mmap, "mmap", lambda *x, **k: maps.append(x))
    with pytest.raises(ValueError, match="unknown first Engram"):
        residency.open_tables(a.manifest, first_mode="wire", **a.kwargs)
    provider = residency.open_tables(a.manifest, first_mode="pread", **a.kwargs)
    ids = np.array([[13, 2, 13], [255, 2, 255]], dtype=np.int64)
    with provider:
        assert maps == []
        s = provider.stats()
        assert s["first_mode"] == "pread"
        assert (
            s["first_locked_bytes"]
            == s["first_pages"]
            == s["first_resident_pages"]
            == 0
        )
        assert all(v["mode"] == "pread" for v in s["blocks"].values())
        reads.clear()
        out = provider.gather(1, ids)
        off = a.doc["external_tables"]["1"]["source_offset"]
        assert reads == [(1, off + row * 192, 192) for row in [2, 13, 255]]
        np.testing.assert_array_equal(np.asarray(out.view(mx.uint16)), a.arrays[1][ids])
        v = provider.stats()["blocks"]["1"]
        assert (v["pread_calls"], v["pread_bytes"]) == (3, 576)
    assert maps == []


def test_bad_row_refusal_and_empty_gather(artifact, monkeypatch):
    artifact.verify()
    with artifact.open() as p:
        for block, ids, error in [
            (1, [0.25], TypeError),
            (11, [-1], IndexError),
            (1, [257], IndexError),
            (2, [0], ValueError),
        ]:
            with pytest.raises(error):
                p.gather(block, ids)
        assert p.gather(11, np.zeros((2, 0, 4), dtype=np.int64)).shape == (2, 0, 4, 96)
        monkeypatch.setattr(residency, "_MAX_GATHER_BYTES", 192)
        with pytest.raises(ValueError, match="chunk"):
            p.gather(31, [0, 0])
        assert all(v["pread_calls"] == 0 for v in p.stats()["blocks"].values())


@pytest.mark.parametrize(
    "field,bad",
    [
        ("source_shard", "../table.safetensors"),
        ("file_sha256", "0" * 64),
        ("source_offset", 8),
        ("row_stride", 384),
        ("dtype", "F16"),
    ],
)
def test_manifest_layout_is_not_a_path_or_pin_authority(artifact, field, bad):
    artifact.doc["external_tables"]["11"][field] = bad
    artifact.manifest.write_text(json.dumps(artifact.doc))
    with pytest.raises(ValueError):
        artifact.verify()
    assert not artifact.kwargs["receipt"].exists()


def test_verification_rejects_incorrect_payload(artifact):
    artifact.doc["external_tables"]["21"]["payload_sha256"] = "f" * 64
    artifact.manifest.write_text(json.dumps(artifact.doc))
    with pytest.raises(ValueError, match="SHA-256"):
        artifact.verify()
    assert not artifact.kwargs["receipt"].exists()


def test_late_tables_off_internal_ssd_only_warn(artifact, monkeypatch):
    monkeypatch.setattr(residency, "_internal_ssd_device", lambda: -1)
    with pytest.warns(UserWarning, match="not on the internal SSD"):
        artifact.verify()
    with pytest.warns(UserWarning, match="not on the internal SSD"), artifact.open():
        pass


def test_symlinked_table_gets_a_clear_error(artifact, tmp_path):
    real = artifact.paths[11]
    moved = tmp_path / "blob"
    real.rename(moved)
    real.symlink_to(moved)
    with pytest.raises(ValueError, match="symlink"):
        artifact.verify()


def test_changed_file_receipt_and_runtime_identity_guards(artifact):
    receipt = artifact.verify()
    receipt.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        artifact.open()
    receipt.chmod(0o600)
    with artifact.open() as p:
        path = artifact.paths[11]
        # Even a byte-identical replacement is not the independently verified inode.
        replacement = path.with_suffix(".replacement")
        replacement.write_bytes(path.read_bytes())
        os.replace(replacement, path)
        with pytest.raises(ValueError, match="changed or replaced"):
            p.gather(11, [0])
        assert p.stats()["closed"]
    with pytest.raises(ValueError, match="stale table receipt"):
        artifact.open()


def test_header_and_manifest_digest_guards(artifact):
    artifact.verify()
    original = artifact.manifest.read_bytes()
    artifact.manifest.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="receipt for this manifest"):
        artifact.open()
    artifact.manifest.write_bytes(original)
    p = artifact.paths[31]
    data = p.read_bytes().replace(b'"BF16"', b'"FP16"', 1)
    p.write_bytes(data)
    with pytest.raises(ValueError, match="wrong dedicated Safetensors header"):
        artifact.open()


@pytest.mark.parametrize("fail_at", ["mlock", "mincore"])
def test_kernel_refusal_unwinds_without_fallback(artifact, monkeypatch, fail_at):
    artifact.verify()
    vm_call, calls = residency._vm_call, []

    def inject(fn, *args):
        calls.append(fn.__name__)
        if fn.__name__ == fail_at:
            if fail_at == "mincore":
                return  # all residency bits zero
            raise OSError("injected wired-memory refusal")
        return vm_call(fn, *args)

    monkeypatch.setattr(residency, "_vm_call", inject)
    with pytest.raises((OSError, RuntimeError)):
        artifact.open()
    assert calls == (
        ["mlock"] if fail_at == "mlock" else ["mlock", "mincore", "munlock"]
    )


def test_short_demand_read_closes_provider(artifact, monkeypatch):
    artifact.verify()
    with artifact.open() as p:
        monkeypatch.setattr(os, "pread", lambda fd, n, off: b"")
        with pytest.raises(OSError, match="short positional read"):
            p.gather(11, [0])
        assert p.stats()["closed"]
