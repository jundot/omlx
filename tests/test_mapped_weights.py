# SPDX-License-Identifier: Apache-2.0
"""Mapped (zero-copy) weight loading, alignment check and realign.

(omlx/utils/mapped_weights.py)
"""

import hashlib
import json
import struct
import time

import mlx.core as mx
import numpy as np
import pytest

from omlx.utils import mapped_weights as mw


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    calls = []

    def fake_wired(limit):
        calls.append(limit)
        return 7 << 30 if limit == 0 else 0

    monkeypatch.setattr(mx, "set_wired_limit", fake_wired)
    yield calls
    for key in list(mw._enabled_dirs):
        mw.disable(key)
    mw._files.clear()


def _weights():
    # Byte sizes are multiples of 8, like real checkpoints: only the
    # unpadded header shifts the tensors.
    return {
        "a.weight": mx.arange(1000, dtype=mx.uint32).reshape(10, 100),
        "b.scales": (mx.arange(64, dtype=mx.float32) / 7).astype(mx.bfloat16),
        "c.bias": mx.linspace(-1, 1, 32).astype(mx.float16),
        "d.idx": mx.arange(-8, 8, dtype=mx.int8),
    }


def _write_misaligned(path, weights):
    """mx.save_safetensors, with a metadata value sized so the data is shifted."""
    for pad in range(8):
        mx.save_safetensors(
            str(path), weights, metadata={"format": "mlx", "p": "x" * pad}
        )
        data_start, _ = mw.read_header(path)
        if data_start % 8 == 3:
            return
    raise AssertionError("could not produce a misaligned header")


def _write_raw(path, tensors):
    """Hand-written safetensors: tensors laid out back to back, no padding."""
    header, blobs, pos = {}, [], 0
    for name, (dtype, arr) in tensors.items():
        raw = arr.tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(arr.shape),
            "data_offsets": [pos, pos + len(raw)],
        }
        blobs.append(raw)
        pos += len(raw)
    encoded = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


def _assert_same(loaded, expected):
    assert set(loaded) == set(expected)
    for k, v in expected.items():
        assert loaded[k].dtype == v.dtype, k
        # Compare on the GPU: a misaligned zero-copy wrap reads fine on the
        # CPU and only goes wrong inside a Metal kernel.
        assert mx.array_equal(loaded[k] * 1, v), k


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_check_alignment_flags_shifted_mlx_checkpoint(tmp_path):
    _write_misaligned(tmp_path / "model.safetensors", _weights())
    report = mw.check_alignment(tmp_path)
    assert report.files == 1 and report.tensors == 4
    assert not report.aligned
    assert report.misaligned_files == ["model.safetensors"]
    # The int8 tensor is fine at any offset.
    assert report.misaligned_tensors == 3
    assert report.can_realign_in_place and report.can_realign_copy


def test_misaligned_tensors_are_copied_not_mapped(tmp_path):
    weights = _weights()
    _write_misaligned(tmp_path / "model.safetensors", weights)
    mw.enable(tmp_path)
    loaded = mx.load(str(tmp_path / "model.safetensors"))
    _assert_same(loaded, weights)
    # Only the int8 tensor (16 bytes) could be served from the mapping.
    assert mw.mapped_bytes() == 16


def test_realign_in_place_makes_everything_mappable(tmp_path):
    weights = _weights()
    path = tmp_path / "model.safetensors"
    _write_misaligned(path, weights)
    (tmp_path / "config.json").write_text("{}")
    config_sha = _sha(tmp_path / "config.json")

    seen = []
    out = mw.realign_model(
        tmp_path, mode="in_place", progress=lambda d, t: seen.append((d, t))
    )
    assert out == tmp_path
    assert seen and seen[-1][0] == seen[-1][1]
    assert mw.check_alignment(tmp_path).aligned
    assert _sha(tmp_path / "config.json") == config_sha
    assert not list(tmp_path.glob("*" + mw._TMP_SUFFIX))

    # Stock loader still reads it, unchanged, metadata included.
    stock, meta = mx.load(str(path), return_metadata=True)
    _assert_same(stock, weights)
    assert meta["format"] == "mlx"

    mw.enable(tmp_path)
    mapped = mx.load(str(path))
    _assert_same(mapped, weights)
    assert mw.mapped_bytes() == sum(v.nbytes for v in weights.values())


def test_realign_copy_leaves_the_original_alone(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    weights = _weights()
    _write_misaligned(src / "model-00001-of-00002.safetensors", weights)
    mx.save_safetensors(
        str(src / "model-00002-of-00002.safetensors"),
        {"e": mx.ones((8,), dtype=mx.uint8)},
    )
    (src / "config.json").write_text('{"model_type": "x"}')
    (src / "assets").mkdir()
    (src / "assets" / "t.txt").write_text("hi")
    before = {p.name: _sha(p) for p in src.glob("*.safetensors")}

    out = mw.realign_model(src, mode="copy", dest=dst)
    assert out == dst
    assert {p.name: _sha(p) for p in src.glob("*.safetensors")} == before
    assert (dst / "config.json").read_text() == '{"model_type": "x"}'
    assert (dst / "assets" / "t.txt").read_text() == "hi"
    assert mw.check_alignment(dst).aligned
    _assert_same(mx.load(str(dst / "model-00001-of-00002.safetensors")), weights)


def test_realign_fixes_relative_misalignment_without_holes(tmp_path):
    # A 3-byte uint8 tensor ahead of a uint32 one: no header padding can fix
    # this, the tensors have to be reordered.
    path = tmp_path / "model.safetensors"
    _write_raw(
        path,
        {
            "m.mask": ("U8", np.array([1, 2, 3], dtype=np.uint8)),
            "m.w": ("U32", np.arange(16, dtype=np.uint32)),
            "m.h": ("F16", np.arange(5, dtype=np.float16)),
        },
    )
    assert not mw.check_alignment(tmp_path).aligned
    mw.realign_model(tmp_path, mode="in_place")
    data_start, header = mw.read_header(path)
    spans = sorted(v["data_offsets"] for k, v in header.items() if k != "__metadata__")
    assert spans[0][0] == 0
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:])), "holes in data"
    assert path.stat().st_size == data_start + spans[-1][1]
    assert mw.check_alignment(tmp_path).aligned
    mw.enable(tmp_path)
    loaded = mx.load(str(path))
    assert loaded["m.w"].tolist() == list(range(16))
    assert loaded["m.mask"].tolist() == [1, 2, 3]


def test_in_place_refuses_symlinked_hf_cache(tmp_path):
    blobs, snap = tmp_path / "blobs", tmp_path / "snap"
    blobs.mkdir()
    snap.mkdir()
    _write_misaligned(blobs / "abc.safetensors", _weights())
    (blobs / "abc.safetensors").rename(blobs / "abc")
    (snap / "model.safetensors").symlink_to(blobs / "abc")
    report = mw.check_alignment(snap)
    assert report.symlinked and not report.can_realign_in_place
    with pytest.raises(ValueError, match="symlink"):
        mw.realign_model(snap, mode="in_place")
    mw.realign_model(snap, mode="copy", dest=tmp_path / "copy")
    assert mw.check_alignment(tmp_path / "copy").aligned


def test_loads_outside_enabled_dirs_pass_through(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    mx.save_safetensors(str(other / "model.safetensors"), _weights())
    mw.enable(tmp_path / "enabled")
    mx.load(str(other / "model.safetensors"))
    assert mw.mapped_bytes() == 0


def test_wired_limit_lifted_while_mapped_and_restored(tmp_path, _reset):
    calls = _reset
    mw.enable(tmp_path / "a")
    mw.enable(tmp_path / "b")
    assert calls == [0]
    mw.disable(tmp_path / "a")
    assert calls == [0]
    mw.disable(tmp_path / "b")
    assert calls == [0, 7 << 30]


def test_mapping_released_with_the_model(tmp_path):
    path = tmp_path / "model.safetensors"
    _write_misaligned(path, _weights())
    mw.realign_model(tmp_path, mode="in_place")
    mw.enable(tmp_path)
    loaded = mx.load(str(path))
    assert mw.mapped_bytes() > 0
    del loaded
    mx.synchronize()
    assert mw.mapped_bytes() == 0


def test_cached_resident_sample(tmp_path, monkeypatch):
    path = tmp_path / "model.safetensors"
    _write_misaligned(path, _weights())
    mw.realign_model(tmp_path, mode="in_place")
    mw.enable(tmp_path)
    loaded = mx.load(str(path))
    mx.eval(*[v * 1 for v in loaded.values()])
    monkeypatch.setitem(mw._resident, "at", 0.0)
    # Nothing is assumed resident before the first sample lands.
    assert mw.cached_resident_mapped_bytes() == 0
    for _ in range(200):
        if mw._resident["at"]:
            break
        time.sleep(0.01)
    total = mw.mapped_bytes()
    assert 0 < mw.cached_resident_mapped_bytes() <= total
    assert 0 < mw.resident_mapped_bytes() <= total
