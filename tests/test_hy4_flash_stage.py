# SPDX-License-Identifier: Apache-2.0
"""Offline staging checks: tiny synthetic originals, no model weights or inference."""

import hashlib
import json
import os
import stat
from types import SimpleNamespace

import pytest

from omlx.patches.hy_v4_flash import residency, stage


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _name(i):
    return f"model-{i:05d}-of-00148.safetensors"


@pytest.fixture
def sample(tmp_path, monkeypatch):
    tmp = tmp_path.resolve()
    model, src, dst = tmp / "converted", tmp / "original", tmp / "ssd"
    model.mkdir()
    src.mkdir()
    tables, pins, data, rows = {}, {}, {}, {}
    for i, b in enumerate((1, 11, 21, 31), 1):
        rows[b] = i
        k = f"model.layers.{b}.engram_layer.multi_head_embedding.embedding.weight"
        n = _name(5 - i)
        payload = bytes([b, 63]) * (96 * i)
        h = json.dumps(
            {k: {"shape": [i, 96], "dtype": "BF16", "data_offsets": [0, len(payload)]}},
            separators=(",", ":"),
        ).encode()
        assert len(h) <= 144
        raw = (144).to_bytes(8, "little") + h.ljust(144) + payload
        pins[b] = len(raw), _sha(raw)
        tables[str(b)] = {
            "source_key": k,
            "source_shard": n,
            "block": b,
            "shape": [i, 96],
            "dtype": "BF16",
            "row_stride": 192,
            "source_offset": 152,
            "byte_length": len(payload),
            "file_size": len(raw),
            "file_sha256": _sha(raw),
            "payload_sha256": _sha(payload),
        }
        if b != 1:  # Even opening #4 is a test failure: that original isn't present.
            (src / n).write_bytes(raw)
            data[n] = raw
    doc = {
        "schema": "hy4-flash-oqh/v1",
        "text_only": True,
        "source": {"repo": "tencent/Hy4-flash", "revision": residency._REVISION},
        "external_tables": tables,
    }
    (model / "manifest.json").write_text(
        json.dumps(doc)
    )  # No core, config, tokenizer or index available.
    monkeypatch.setattr(residency, "_PINS", pins)
    monkeypatch.setattr(stage, "_ROWS", {b: rows[b] for b in (11, 21, 31)})
    monkeypatch.setattr(residency, "_internal_ssd_device", lambda: tmp.stat().st_dev)
    monkeypatch.setattr(stage, "_CHUNK", 192)
    capacity = SimpleNamespace(f_bavail=100 << 30, f_frsize=1)
    monkeypatch.setattr(stage.os, "fstatvfs", lambda fd: capacity)
    return SimpleNamespace(
        model=model,
        src=src,
        dst=dst,
        data=data,
        doc=doc,
        capacity=capacity,
        run=lambda **kw: stage.stage_tables(model, src, dst, **kw),
    )


def test_public_late_pin_geometry():
    assert stage._ROWS == {11: 164999008, 21: 165004086, 31: 165008942}
    assert sum(residency._PINS[b][0] for b in stage._ROWS) == 95042311368
    assert all(residency._PINS[b][0] == n * 192 + 152 for b, n in stage._ROWS.items())
    assert stage._RESERVE >= 48 << 30


def test_cli_only_late_streaming_and_resume(sample, monkeypatch, capsys):
    s = sample
    args = ["--model", str(s.model), "--source", str(s.src), "--ssd-dir", str(s.dst)]
    manifest_size = (s.model / "manifest.json").stat().st_size
    pread = os.pread

    def bounded(fd, size, offset):
        if os.fstat(fd).st_size != manifest_size:
            assert size <= 192
        return pread(fd, size, offset)

    monkeypatch.setattr(stage.os, "pread", bounded)
    assert stage.main([*args, "--check"]) == 0
    assert json.loads(capsys.readouterr().out)["would_copy_blocks"] == [11, 21, 31]
    assert not s.dst.exists()
    assert stage.main(args) == 0
    assert json.loads(capsys.readouterr().out)["copied_blocks"] == [11, 21, 31]
    assert {p.name for p in s.dst.iterdir()} == s.data.keys()
    before = {p.name: residency._stamp(p.stat()) for p in s.dst.iterdir()}
    assert stage.main(args) == 0
    assert json.loads(capsys.readouterr().out)["reused_blocks"] == [11, 21, 31]
    assert before == {p.name: residency._stamp(p.stat()) for p in s.dst.iterdir()}
    for n, raw in s.data.items():
        assert (s.dst / n).read_bytes() == (s.src / n).read_bytes() == raw
        assert stat.S_IMODE((s.dst / n).stat().st_mode) == 0o400


@pytest.mark.parametrize(
    "problem", ["path", "dtype", "rows", "sha", "revision", "size"]
)
def test_manifest_cannot_choose_files_or_pins(sample, problem):
    s = sample
    t = s.doc["external_tables"]["11"]
    if (
        problem == "rows"
    ):  # Internally consistent extent and total length, but NOT the public shape.
        t["source_offset"] += 192
        t["shape"][0] -= 1
        t["byte_length"] -= 192
    elif problem == "revision":
        s.doc["source"]["revision"] = "latest"
    elif problem == "size":
        t["file_size"] -= 1
    elif problem == "dtype":
        t["dtype"] = "F16"
    elif problem == "sha":
        t["file_sha256"] = "0" * 64
    else:
        t["source_shard"] = "../../outside"
    (s.model / "manifest.json").write_text(json.dumps(s.doc))
    with pytest.raises(ValueError):
        s.run()
    assert not s.dst.exists()


def test_missing_and_changed_source_resumable(sample):
    s = sample
    p = s.src / _name(2)
    p.unlink()
    with pytest.raises(FileNotFoundError):
        s.run()
    assert not s.dst.exists()
    raw = s.data[p.name]
    p.write_bytes(raw[:-1] + b"!")
    with pytest.raises(ValueError, match="SHA-256"):
        s.run()
    assert list(s.dst.iterdir()) == [s.dst / _name(3)]
    p.write_bytes(raw)
    assert s.run()["reused_blocks"] == [11]


def test_aggregate_space_and_internal_device(sample, monkeypatch):
    s = sample
    s.capacity.f_bavail = (
        sum(map(len, s.data.values())) + (48 << 30) + stage._MARGIN - 1
    )
    for check in (True, False):
        with pytest.raises(ValueError, match="insufficient"):
            s.run(check=check)
    assert not s.dst.exists()
    s.dst.mkdir()
    n = _name(3)
    (s.dst / n).write_bytes(s.data[n])
    assert s.run(check=True)["missing_bytes_before"] == sum(
        len(v) for k, v in s.data.items() if k != n
    )
    monkeypatch.setattr(
        residency, "_internal_ssd_device", lambda: s.model.stat().st_dev + 1
    )
    with pytest.raises(ValueError, match="internal Data SSD"):
        s.run()
    assert list(s.dst.iterdir()) == [s.dst / n]


def test_overlaps_links_and_wrong_existing_bytes_refused(sample):
    s = sample
    n = _name(1)
    for path in [s.model, s.model / "sub", s.src, s.src / "sub", s.src.parent]:
        with pytest.raises(ValueError):
            stage.stage_tables(s.model, s.src, path)
    s.dst.symlink_to(s.src, target_is_directory=True)
    with pytest.raises(ValueError):
        s.run()
    s.dst.unlink()
    s.dst.mkdir()
    p = s.dst / n
    p.symlink_to(s.src / n)
    with pytest.raises(ValueError, match="symlink"):
        s.run()
    assert p.readlink() == s.src / n
    p.unlink()
    os.link(s.src / n, p)
    with pytest.raises(ValueError, match="share an inode"):
        s.run()
    p.unlink()
    p.write_bytes(s.data[n][:-1] + b"!")
    with pytest.raises(ValueError, match="SHA-256"):
        s.run()
    assert list(s.dst.iterdir()) == [p] and p.read_bytes()[-1:] == b"!"


def test_no_replace_race_and_unowned_orphan(sample, monkeypatch):
    s = sample
    s.dst.mkdir()
    orphan = s.dst / ".hy4-stage-occupied.partial"
    orphan.write_bytes(b"mine")
    with monkeypatch.context() as m:
        m.setattr(stage.secrets, "token_hex", lambda _: "occupied")
        with pytest.raises(FileExistsError):
            s.run()
    link = os.link

    def race(a, b, **kw):
        (s.dst / b).write_bytes(b"racer")
        return link(a, b, **kw)

    monkeypatch.setattr(stage.os, "link", race)
    with pytest.raises(FileExistsError):
        s.run()
    assert set(s.dst.iterdir()) == {orphan, s.dst / _name(3)}
    assert (
        orphan.read_bytes() == b"mine" and (s.dst / _name(3)).read_bytes() == b"racer"
    )


def test_readback_failure_and_space_drop_clean_up(sample, monkeypatch):
    s = sample
    write = os.write

    def corrupt(fd, buf):
        return write(fd, buf if len(buf) == 152 else bytes(len(buf)))

    with monkeypatch.context() as m:
        m.setattr(stage.os, "write", corrupt)
        with pytest.raises(ValueError, match="SHA-256"):
            s.run()
    assert not list(s.dst.iterdir())

    def available(fd):
        if stat.S_ISREG(os.fstat(fd).st_mode) and os.fstat(fd).st_size:
            return SimpleNamespace(f_bavail=48 << 30, f_frsize=1)
        return s.capacity

    monkeypatch.setattr(stage.os, "fstatvfs", available)
    with pytest.raises(ValueError, match="insufficient"):
        s.run()
    assert not list(s.dst.iterdir())
