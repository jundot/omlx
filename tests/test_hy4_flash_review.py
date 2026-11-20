# SPDX-License-Identifier: Apache-2.0
from contextlib import contextmanager
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches.hy_v4_flash import artifact, residency, runner
from omlx.patches.hy_v4_flash.convert import ALGORITHM


@pytest.mark.parametrize(
    "kind,field",
    [
        ("expert", "weight"),
        ("expert", "oqh_signs"),
        ("trunk", "scales"),
        ("trunk", "weight"),
    ],
)
def test_bad_last_shard_never_loads_weights(tmp_path, monkeypatch, kind, field):
    a = mx.zeros((2,), dtype=mx.bfloat16)
    bad = mx.zeros((2,), dtype=mx.int32)
    model = nn.Module()
    model.first = a
    model.bank = nn.Module()
    model.bank[field] = a
    cfg = (
        {"omlx_oqh": {"tensors": {"bank": {}}}}
        if kind == "expert"
        else {"trunk_quantization": {"modules": ["bank"]}}
    )
    monkeypatch.setattr(runner, "quantize_trunk", lambda *args: None)
    for fn, tensors in [("a", {"first": a}), ("z", {"bank." + field: bad})]:
        mx.save_safetensors(
            str(tmp_path / (fn + ".safetensors")), tensors, metadata={"format": "mlx"}
        )
    monkeypatch.setattr(
        mx, "load", lambda *args: pytest.fail("payload load before preflight")
    )
    with pytest.raises(ValueError, match="wrong checkpoint tensor"):
        runner.load_weights(
            model,
            SimpleNamespace(
                root=tmp_path,
                config=cfg,
                weight_map={"first": "a.safetensors", "bank." + field: "z.safetensors"},
            ),
        )
    assert model.first is a and model.bank[field] is a


def test_packed_head_converter_to_fp32_logits(tmp_path):
    import numpy as np

    from omlx.patches.hy_v4_flash.model import Model
    from omlx.patches.hy_v4_flash.quant_io import Spec, Writer, trunk_specs, write_trunk

    name = "lm_head.weight"
    spec = Spec("BF16", (8, 64))
    weights = (
        mx.sin(mx.arange(512, dtype=mx.float32)).reshape(8, 64).astype(mx.bfloat16)
    )
    hidden = (
        mx.cos(mx.arange(64, dtype=mx.float32)).reshape(1, 1, 64).astype(mx.bfloat16)
    )
    specs = trunk_specs(name, spec, 8)
    with (tmp_path / "head.safetensors").open("w+b") as f:
        writer = Writer(f, specs)
        write_trunk(
            name, spec, [np.array(weights.view(mx.uint16)).tobytes()], writer, 8
        )
        writer.finish(4096)
    m = nn.Module()
    m.lm_head = nn.Linear(64, 8, bias=False)
    trunk = dict(bits=8, group_size=64, mode="affine", modules=["lm_head"])
    runner.load_weights(
        m,
        SimpleNamespace(
            root=tmp_path,
            config={"trunk_quantization": trunk},
            weight_map=dict.fromkeys(specs, "head.safetensors"),
        ),
    )
    target = SimpleNamespace(
        lm_head=m.lm_head,
        model=lambda *a: hidden,
        args=SimpleNamespace(
            max_position_embeddings=16,
            engram_layer_ids=[],
            tie_word_embeddings=False,
            enable_lm_head_fp32=True,
        ),
    )
    logits = Model.__call__(target, mx.array([[1]]))
    q, s, b = mx.quantize(weights, group_size=64, bits=8, mode="affine")
    expected = mx.quantized_matmul(
        hidden.astype(mx.float32), q, s, b, group_size=64, bits=8
    )
    assert logits.dtype == mx.float32 and mx.array_equal(logits, expected).item()


def test_array_loader_cannot_omit_validated_leaf(tmp_path, monkeypatch):
    model = nn.Linear(4, 3, bias=False)
    before = model.weight
    mx.save_safetensors(
        str(tmp_path / "a.safetensors"),
        {"weight": mx.ones((3, 4), dtype=mx.bfloat16)},
        metadata={"format": "mlx"},
    )
    monkeypatch.setattr(mx, "load", lambda *_: {})
    with pytest.raises(ValueError, match="loaded tensors disagree"):
        runner.load_weights(
            model,
            SimpleNamespace(
                root=tmp_path, config={}, weight_map={"weight": "a.safetensors"}
            ),
        )
    assert model.weight is before


@pytest.mark.parametrize("bad", ["8", [["lm_head"]], [True]])
def test_bad_trunk_metadata(bad):
    meta = (
        bad
        if isinstance(bad, str)
        else dict(bits=8, group_size=64, mode="affine", modules=bad)
    )
    with pytest.raises(ValueError, match="trunk_quantization"):
        artifact._validate_trunk(meta, {"lm_head.weight": "s"})


def test_manifest_disagreement_and_legacy():
    cfg = {"bank": {"bits": 4}}
    artifact._validate_quantization({}, cfg, None)
    artifact._validate_quantization(
        {"quantization": dict(ALGORITHM, bits=4)}, cfg, None
    )
    with pytest.raises(ValueError, match="disagrees"):
        artifact._validate_quantization(
            {"quantization": dict(ALGORITHM, bits=8)}, cfg, None
        )


def test_cli_stale_receipt_precedes_loading_and_defaults_to_pread(
    tmp_path, monkeypatch, capsys
):
    cfg = dict(max_position_embeddings=4096, eos_token_id=1)
    monkeypatch.setattr(
        artifact.Artifact,
        "open",
        lambda *a, **kw: SimpleNamespace(root=tmp_path, config=cfg),
    )
    monkeypatch.setattr(runner, "render_user", lambda *a: (None, [0]))
    monkeypatch.setattr(
        runner, "load_weights", lambda *a: pytest.fail("loaded before checking receipt")
    )

    @contextmanager
    def stale(*a, first_mode, **kw):
        assert first_mode == "pread"
        raise ValueError("stale table receipt")
        yield

    monkeypatch.setattr(residency, "open_tables", stale)
    with pytest.raises(SystemExit) as exc:
        runner.main(
            [
                "generate",
                "--model",
                str(tmp_path),
                "--source",
                str(tmp_path),
                "--ssd-dir",
                str(tmp_path),
                "--receipt",
                str(tmp_path / "r"),
                "--prompt",
                "hi",
            ]
        )
    assert exc.value.code == 1 and "stale table receipt" in capsys.readouterr().err
