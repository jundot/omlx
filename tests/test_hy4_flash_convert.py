# SPDX-License-Identifier: Apache-2.0
"""Offline synthetic tests of the pinned Hy4 Flash conversion path (no downloads)."""

import hashlib
import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from omlx.oqh import format as fmt
from omlx.patches.hy_v4_flash import convert as c
from omlx.patches.hy_v4_flash import quant_io as q

GATE = "model.layers.1.mlp.experts.gate_up_proj"
DOWN = "model.layers.1.mlp.experts.down_proj"
TABLE = "model.layers.1" + c.TABLE_END


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git_blob(raw):
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


def disk_json(p):
    return json.loads(p.read_text())


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    mx = pytest.importorskip("mlx.core")
    src = tmp_path / "original"
    src.mkdir()
    header_dir = tmp_path / "headers"
    header_dir.mkdir()
    rng = np.random.default_rng(5613)

    def bf(shape):
        return mx.array(rng.standard_normal(shape).astype(np.float32)).astype(
            mx.bfloat16
        )

    banks = {GATE: bf((3, 512, 512)), DOWN: bf((3, 512, 256))}
    native = {
        "model.layers.1.mlp.gate.weight": bf((3, 64)),
        "model.layers.1.mlp.gate.e_score_correction_bias": mx.array(
            [-0.0, 1.125, -3.25], dtype=mx.float32
        ),
        "model.layers.1.mlp.shared_experts.down_proj.weight": bf((16, 64)),
        "model.layers.1.mlp.routed_expert_down_proj.weight": bf((16, 128)),
        "model.layers.1.self_attn.linear_gate.weight": bf((16, 64)),
        "model.layers.1.self_attn.kv_b_proj.weight": bf((16, 64)),
        "model.layers.1.engram_layer.vocab_map": mx.arange(19, dtype=mx.int32),
        "lm_head.weight": bf((64, 64)),
    }
    table = bf((26, 96))
    cfg = {
        "model_type": "hy_v4_flash",
        "dtype": "bfloat16",
        "vision_config": {"fixture": True},
        "text_config": {
            "architectures": ["HYV4FlashForCausalLM"],
            "model_type": "hy_v4_flash",
            "num_hidden_layers": 2,
            "mlp_layer_types": ["dense", "sparse"],
            "routed_expert_hidden_size": 512,
            "moe_intermediate_size": 256,
            "n_routed_experts": 3,
            "engram_layer_ids": [2],
            "engram_layer_head_vocab_sizes": {"2": [[5, 6], [7, 8]]},
            "engram_n_head_per_ngram": 2,
            "engram_n_embed_per_ngram": 192,
        },
    }
    parts = [
        {TABLE: table},
        {"vit.encoder.weight": bf((4, 2))},
        {GATE: banks[GATE], "lm_head.weight": native["lm_head.weight"]},
        {
            DOWN: banks[DOWN],
            **{k: v for k, v in native.items() if k != "lm_head.weight"},
            "model.dfly_layers.0.mlp.down_proj.weight": bf((8, 16)),
        },
    ]
    inventory, basis = {}, {}
    index = {"metadata": {"total_size": 0}, "weight_map": {}}
    for n, tensors in enumerate(parts, 1):
        name = f"model-{n:05}-of-00004.safetensors"
        path = src / name
        mx.save_safetensors(str(path), tensors, metadata={"format": "pt"})
        raw = path.read_bytes()  # tiny synthetic inputs only
        inventory[name] = {"size": len(raw), "sha256": sha(raw)}
        base = 8 + struct.unpack("<Q", raw[:8])[0]
        (header_dir / (name + ".header")).write_bytes(raw[:base])
        hdr = json.loads(raw[8:base])
        for k, a in tensors.items():
            row = hdr[k]
            basis[k] = [name, base + row["data_offsets"][0], row["dtype"], row["shape"]]
            index["weight_map"][k] = name
            index["metadata"]["total_size"] += a.nbytes
    assets = {
        "config.json": json.dumps(cfg).encode(),
        c.INDEX: json.dumps(index).encode(),
        "LICENSE": b"synthetic fixture",
        "tokenizer.json": b"{}",
    }
    for n, data in assets.items():
        (src / n).write_bytes(data)
    monkeypatch.setattr(c, "REPO", "test/fixture")
    monkeypatch.setattr(c, "REVISION", "synthetic")
    monkeypatch.setattr(c, "ASSET_BLOBS", {k: git_blob(v) for k, v in assets.items()})
    monkeypatch.setattr(c, "LAYOUT_SHA256", c.canonical_sha256(basis))
    monkeypatch.setattr(c, "LFS_INVENTORY_SHA256", c.canonical_sha256(inventory))
    return SimpleNamespace(
        src=src,
        out=tmp_path / "artifact",
        headers=header_dir,
        banks=banks,
        native=native,
        table=table,
        inventory=inventory,
        index=index,
    )


def raw_tensor(a):
    import mlx.core as mx

    return np.asarray(a.view(mx.uint16) if a.dtype == mx.bfloat16 else a).tobytes()


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def check_bank(k, original, values):
    """Check exact MLX whole-bank RTN, then independent unpack/WHT and gather."""
    import mlx.core as mx

    p, s, b, signs = [
        values[k + "." + n] for n in ("weight", "oqh_scales", "oqh_biases", "oqh_signs")
    ]
    e, o, width = original.shape
    assert p.dtype == mx.uint32 and p.shape == (e, o, width // 4)
    assert s.dtype == b.dtype == signs.dtype == mx.bfloat16
    assert s.shape == b.shape == (e, o, width // 128)
    assert signs.shape == (width // 256, 256)
    expected = fmt.make_signs(k, width)
    np.testing.assert_array_equal(
        np.asarray(signs.astype(mx.float32)).ravel(), expected
    )
    # Independent of converter chunk boundaries, preserving all three source axes.
    wr = mx.hadamard_transform(
        (original.astype(mx.float32) * mx.array(expected)).reshape(e, o, -1, 256)
    )
    gold = mx.quantize(
        wr.reshape(original.shape), group_size=128, bits=8, mode="affine"
    )
    for a, ref in zip((p, s, b), gold, strict=True):
        assert raw_tensor(a) == raw_tensor(ref.astype(a.dtype))

    h = np.ones((1, 1), np.float32)  # normalized Sylvester matrix, not an MLX transform
    for _ in range(8):
        h = np.block([[h, h], [h, -h]])
    h /= 16
    words = np.asarray(p)
    codes = (
        np.stack([(words >> (8 * lane)) & 255 for lane in range(4)], axis=-1)
        .reshape(e, o, width)
        .astype(np.float32)
    )
    rotated = codes * np.asarray(s.astype(mx.float32)).repeat(
        128, axis=-1
    ) + np.asarray(b.astype(mx.float32)).repeat(128, axis=-1)
    effective = (rotated.reshape(-1, 256) @ h).reshape(e, o, width) * expected
    orig = np.asarray(original.astype(mx.float32))
    assert rel(effective, orig) < 0.012
    x = mx.array(
        np.random.default_rng(872).standard_normal((2, width)).astype(np.float32)
    ).astype(mx.bfloat16)
    xf = np.asarray(x.astype(mx.float32))
    xr = ((xf * expected).reshape(-1, 256) @ h).reshape(2, 1, width)
    lhs, rhs = [0, 1, 0, 0], [2, 0, 1, 2]
    y = mx.gather_qmm(
        mx.array(xr).astype(mx.bfloat16),
        p,
        s,
        b,
        lhs_indices=mx.array(lhs),
        rhs_indices=mx.array(rhs),
        transpose=True,
        group_size=128,
        bits=8,
    )
    got = np.asarray(y.astype(mx.float32)).reshape(-1, o)
    dense = (orig[rhs] @ xf[lhs, :, None]).squeeze(-1)
    dequant = (effective[rhs] @ xf[lhs, :, None]).squeeze(-1)
    return rel(got, dense), rel(got, dequant)


def test_stream_roundtrip(snapshot, monkeypatch, record_property):
    import mlx.core as mx
    from safetensors import safe_open

    s = snapshot
    chunk = 31777  # 31 or 62 complete rows: neither divides the expert dimension
    calls = []
    pack = q.quantize_rows

    def bounded(raw, signs, *bits):
        assert len(raw) <= chunk and len(raw) % (2 * signs.size) == 0
        calls.append(len(raw))
        return pack(raw, signs, *bits)

    monkeypatch.setattr(q, "quantize_rows", bounded)
    before = {
        p.name: (p.stat().st_mtime_ns, sha(p.read_bytes())) for p in s.src.iterdir()
    }
    manifest = c.build(s.src, s.out, budget=chunk)
    assert len(calls) > 70
    assert before == {
        p.name: (p.stat().st_mtime_ns, sha(p.read_bytes())) for p in s.src.iterdir()
    }
    assert disk_json(s.out / "manifest.json") == manifest
    assert (
        manifest["schema"] == c.SCHEMA
        and manifest["text_only"]
        and not manifest["inference_qualified"]
    )
    assert manifest["quantization"]["calibrated"] is False and str(
        s.src
    ) not in json.dumps(manifest)
    assert manifest["source"]["shards"] == s.inventory
    assert {p.name for p in s.out.iterdir()} == (
        manifest["core_shards"].keys() | manifest["assets"].keys() | {"manifest.json"}
    )
    for n, r in {**manifest["assets"], **manifest["core_shards"]}.items():
        raw = (s.out / n).read_bytes()
        assert r == {"sha256": sha(raw), "size": len(raw)}
    cfg = disk_json(s.out / "config.json")
    assert cfg["architectures"] == ["HYV4FlashForCausalLM"] and not {
        "vision_config",
        "text_config",
        "quantization",
    }.intersection(cfg)
    assert fmt.oqh_paths(cfg) == dict.fromkeys(
        s.banks, {"bits": 8, "block": 256, "group_size": 128}
    )
    table = manifest["external_tables"]["1"]
    assert table["source_key"] == TABLE and table["row_stride"] == 192
    assert table["shape"] == [26, 96] and table["dtype"] == "BF16"
    src_file = (s.src / table["source_shard"]).read_bytes()
    assert table["file_size"] == len(src_file) and table["file_sha256"] == sha(src_file)
    payload = src_file[table["source_offset"] :]
    assert (
        len(payload) == table["byte_length"] and sha(payload) == table["payload_sha256"]
    )
    assert payload == raw_tensor(s.table)
    idx = disk_json(s.out / c.INDEX)
    expected = set(s.native) | {
        k + "." + n
        for k in s.banks
        for n in ("weight", "oqh_scales", "oqh_biases", "oqh_signs")
    }
    assert set(idx["weight_map"]) == expected
    values = {}
    for fname in manifest["core_shards"]:
        path = s.out / fname
        with safe_open(str(path), framework="np") as f:
            assert set(f.keys()) == {
                k for k, v in idx["weight_map"].items() if v == fname
            }
        values.update(mx.load(str(path)))  # tiny synthetic shards only
    assert (
        sum(a.nbytes for a in values.values())
        == idx["metadata"]["total_size"]
        == manifest["counts"]["core_payload_bytes"]
    )
    for k, a in s.native.items():
        assert values[k].dtype == a.dtype and raw_tensor(values[k]) == raw_tensor(a)
    for k, a in s.banks.items():
        err, err_packed = check_bank(k, a, values)
        record_property(k + "_rel_l2_dense", err)
        record_property(k + "_rel_l2_unpacked", err_packed)
        assert err < 0.018 and err_packed < 0.008


@pytest.mark.parametrize("bits", [4, 5, 6])
def test_lower_bit_build_matches_mlx_rtn(snapshot, bits):
    import mlx.core as mx

    s = snapshot
    manifest = c.build(s.src, s.out, bits=bits)
    assert manifest["quantization"]["bits"] == bits
    cfg = disk_json(s.out / "config.json")
    assert fmt.oqh_paths(cfg) == dict.fromkeys(
        s.banks, {"bits": bits, "block": 256, "group_size": 128}
    )
    values = {}
    for fname in manifest["core_shards"]:
        values.update(mx.load(str(s.out / fname)))
    for k, original in s.banks.items():
        e, o, width = original.shape
        p, sc, bi = (
            values[k + "." + n] for n in ("weight", "oqh_scales", "oqh_biases")
        )
        assert p.dtype == mx.uint32 and p.shape == (e, o, width * bits // 32)
        signs = mx.array(fmt.make_signs(k, width)).astype(mx.float32)
        wr = mx.hadamard_transform(
            (original.astype(mx.float32) * signs).reshape(e, o, -1, 256)
        )
        gold = mx.quantize(
            wr.reshape(original.shape), group_size=128, bits=bits, mode="affine"
        )
        for a, ref in zip((p, sc, bi), gold, strict=True):
            assert raw_tensor(a) == raw_tensor(ref.astype(a.dtype))
    with pytest.raises(ValueError, match="bits"):
        c.inspect_source(s.src, bits=3)


TRUNK = {
    "lm_head",
    "model.layers.1.mlp.routed_expert_down_proj",
    "model.layers.1.mlp.shared_experts.down_proj",
    "model.layers.1.self_attn.linear_gate",
}


def test_trunk_build_matches_nn_quantize(snapshot):
    import mlx.core as mx
    import mlx.nn as nn

    from omlx.patches.hy_v4_flash import artifact

    s = snapshot
    # 64-row chunks: every trunk linear needs several row-aligned chunks.
    manifest = c.build(s.src, s.out, budget=8 << 10, trunk_bits=8)
    record = manifest["quantization"]["trunk"]
    assert record["modules"] == sorted(TRUNK) and record["group_size"] == 64
    cfg = disk_json(s.out / "config.json")
    assert cfg["trunk_quantization"] == record
    assert fmt.oqh_paths(cfg) == dict.fromkeys(
        s.banks, {"bits": 8, "block": 256, "group_size": 128}
    )
    assert manifest["counts"]["trunk_linears"] == len(TRUNK)
    values = {}
    for fname in manifest["core_shards"]:
        values.update(mx.load(str(s.out / fname)))
    for name, original in s.native.items():
        module = name.removesuffix(".weight")
        if module in TRUNK:
            linear = nn.Linear(*original.shape[::-1], bias=False)
            linear.weight = original
            gold = nn.QuantizedLinear.from_linear(linear, 64, 8)
            for part in ("weight", "scales", "biases"):
                got, ref = values[f"{module}.{part}"], getattr(gold, part)
                assert got.dtype == ref.dtype and raw_tensor(got) == raw_tensor(ref)
        else:  # router, kv_b_proj, Engram map, F32 bias: byte-exact
            assert raw_tensor(values[name]) == raw_tensor(original)
            assert module + ".scales" not in values
    artifact._validate_trunk(cfg["trunk_quantization"], values)
    with pytest.raises(ValueError, match="trunk bits"):
        c.inspect_source(s.src, trunk_bits=4)


def test_trunk_validation_is_all_or_nothing():
    from omlx.patches.hy_v4_flash.artifact import _validate_trunk

    index = {f"{m}.{p}": 0 for m in TRUNK for p in ("weight", "scales", "biases")}
    index["model.layers.1.self_attn.kv_b_proj.weight"] = 0
    good = c.trunk_record(8, TRUNK)
    _validate_trunk(good, index)
    for bad in (
        dict(good, modules=sorted(TRUNK)[1:]),
        dict(good, modules=sorted(TRUNK) + ["model.layers.1.self_attn.kv_b_proj"]),
        dict(good, group_size=128),
        dict(good, bits=4),
    ):
        with pytest.raises(ValueError, match="trunk_quantization"):
            _validate_trunk(bad, index)
    with pytest.raises(ValueError, match="without trunk_quantization"):
        _validate_trunk(None, index)
    bf16 = {f"{m}.weight": 0 for m in TRUNK}
    _validate_trunk(None, bf16)
    with pytest.raises(ValueError, match="trunk_quantization"):
        _validate_trunk(good, bf16)


def test_detached_dry_run(snapshot, capsys):
    s = snapshot
    before = set(s.src.iterdir())
    assert (
        c.main(["--source", str(s.src), "--headers", str(s.headers), "--dry-run"]) == 0
    )
    audit = json.loads(capsys.readouterr().out)
    assert not audit["payloads_verified"] and audit["counts"]["routed_banks"] == 2
    assert audit["counts"]["omitted_drafter"] == audit["counts"]["omitted_vision"] == 1
    assert audit == c.inspect_source(s.src).audit()
    assert not s.out.exists() and set(s.src.iterdir()) == before
    with pytest.raises(SystemExit):
        c.main(
            [
                "--source",
                str(s.src),
                "--headers",
                str(s.headers),
                "--output",
                str(s.out),
            ]
        )


def test_42_layer_policy_and_public_sizes():
    # Synthetic metadata only: allocate no real bank/table, even lazily.
    text = {
        "mlp_layer_types": ["dense"] + ["sparse"] * 41,
        "num_hidden_layers": 42,
        "n_routed_experts": 512,
        "routed_expert_hidden_size": 1536,
        "moe_intermediate_size": 1536,
        "engram_layer_ids": [2, 12, 22, 32],
        "engram_layer_head_vocab_sizes": {
            str(i): [[2] * 8, [3] * 8] for i in [2, 12, 22, 32]
        },
        "engram_n_head_per_ngram": 8,
        "engram_n_embed_per_ngram": 768,
    }
    banks, tables = c.policy(text)
    assert len(banks) == 82 and {i for i, _ in tables.values()} == {1, 11, 21, 31}
    assert all(s.shape[-1] == 96 for _, s in tables.values())
    assert sum(s.nbytes for s in banks.values()) == 297158049792
    packed = sum(
        o.nbytes for k, s in banks.items() for o in q.bank_specs(k, s).values()
    )
    assert packed == 153222371328
    with pytest.raises(ValueError, match="unexpected bank/table"):
        c.disposition(
            "model.layers.0.mlp.experts.fake", q.Spec("F32", (1,)), banks, tables
        )


@pytest.mark.parametrize(
    "field", ["table", "omitted", "asset", "missing_shard", "wrong_layout"]
)
def test_integrity_failures_never_publish(snapshot, monkeypatch, field):
    s = snapshot
    if field == "wrong_layout":
        monkeypatch.setattr(c, "LAYOUT_SHA256", "f" * 64)
    elif field == "missing_shard":
        (s.src / s.index["weight_map"][GATE]).unlink()
    else:
        name = {
            "table": s.index["weight_map"][TABLE],
            "omitted": s.index["weight_map"]["vit.encoder.weight"],
            "asset": "tokenizer.json",
        }[field]
        path = s.src / name
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)  # same-size corruption, including a payload not copied
    with pytest.raises(ValueError):
        c.build(s.src, s.out)
    assert not s.out.exists() and not list(s.out.parent.glob(".hy4-convert-*"))


def test_no_overwrite_or_source_alias(snapshot):
    s = snapshot
    s.out.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        c.build(s.src, s.out)
    stage = s.out.parent / "other-dir"
    stage.mkdir()
    ino = s.out.stat().st_ino
    with pytest.raises(
        FileExistsError
    ):  # race: empty destination created just before publish
        c.publish(stage, s.out)
    assert stage.is_dir() and s.out.stat().st_ino == ino
    alias = s.out.parent / "alias"
    alias.symlink_to(s.src, target_is_directory=True)
    with pytest.raises(ValueError, match="outside source"):
        c.build(s.src, alias / "do-not-create")


@pytest.mark.parametrize(
    "field, bad",
    [("dtype", []), ("shape", [True]), ("shape", [1] * 1000), ("data_offsets", None)],
)
def test_malformed_header_fails_closed(tmp_path, field, bad):
    path = tmp_path / "malformed.header"
    entry = {"dtype": "I32", "shape": [1], "data_offsets": [0, 4], field: bad}
    raw = json.dumps({"x": entry}).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    with path.open("rb") as f, pytest.raises(ValueError):
        q.read_header(f, detached=True)


def test_reject_overlaps_and_writer_readback(tmp_path):
    path = tmp_path / "synthetic.safetensors"
    e = {"dtype": "I32", "shape": [1], "data_offsets": [0, 4]}
    raw = json.dumps({"a": e, "b": e}).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(8))
    with path.open("rb") as f, pytest.raises(ValueError, match="overlap"):
        q.read_header(f)
    with path.open("w+b") as f:
        w = q.Writer(f, {"a": q.Spec("I32", (2,))})
        w.append("a", bytes(4))
        with pytest.raises(ValueError, match="unfinished"):
            w.finish(1)
        w.append("a", bytes(4))
        f.seek(w.views["a"].start)
        f.write(b"!")
        with pytest.raises(ValueError, match="readback mismatch"):
            w.finish(1)
