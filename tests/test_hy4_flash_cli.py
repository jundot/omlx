# SPDX-License-Identifier: Apache-2.0
"""Check the public export contract, strict loader, and standalone CLI dispatch."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches.hy_v4_flash import artifact as ar
from omlx.patches.hy_v4_flash.runner import greedy, load_weights


def record(path):
    data = path.read_bytes()
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


@pytest.fixture
def export(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    banks = {
        f"model.layers.1.mlp.experts.{s}": {"bits": 8, "block": 256, "group_size": 128}
        for s in ("gate_up_proj", "down_proj")
    }
    weight_map = {
        k + "." + s: "core.safetensors"
        for k in banks
        for s in ("weight", "oqh_scales", "oqh_biases", "oqh_signs")
    }
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "hy_v4_flash",
                "mlp_layer_types": ["dense", "sparse"],
                "omlx_oqh": {"version": 1, "tensors": banks},
            }
        )
    )
    (root / ar.INDEX).write_text(json.dumps({"weight_map": weight_map}))
    (root / "chat_template.jinja").write_text("test")
    (root / "tokenizer.json").write_text("{}")
    (root / "core.safetensors").write_bytes(b"x" * 17)
    doc = {
        "schema": ar.SCHEMA,
        "text_only": True,
        "source": {"repo": ar.SOURCE, "revision": ar.REVISION},
        "assets": {
            n: record(root / n)
            for n in ("config.json", ar.INDEX, "chat_template.jinja", "tokenizer.json")
        },
        "core_shards": {"core.safetensors": record(root / "core.safetensors")},
    }
    (root / "manifest.json").write_text(json.dumps(doc))
    return root


def test_integrity_and_optional_core_rehash(export):
    ok = ar.Artifact.open(export, verify_core=True)
    assert ok.summary()["rotated_expert_banks"] == 2
    (export / "core.safetensors").write_bytes(b"y" * 17)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        ar.Artifact.open(export, verify_core=True)
    (export / "chat_template.jinja").write_text("evil")
    with pytest.raises(ValueError, match="chat_template"):
        ar.Artifact.open(export)


def test_reject_indirect_and_escaping_paths(export, tmp_path):
    for bad in ("../src", "./core.safetensors", "/tmp/a", "a/b", ".."):
        with pytest.raises(ValueError):
            ar.member(export, bad)
    (tmp_path / "elsewhere").write_text("unrelated")
    (export / "symlink").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="indirect"):
        ar.member(export, "symlink")


def test_strict_physical_shape_loading(tmp_path):
    layer = nn.Linear(3, 4, bias=False)
    val = mx.arange(12).reshape(4, 3).astype(mx.float32)
    mx.save_safetensors(
        str(tmp_path / "s.safetensors"), {"weight": val}, metadata={"format": "mlx"}
    )
    art = SimpleNamespace(root=tmp_path, weight_map={"weight": "s.safetensors"})
    load_weights(layer, art)
    assert mx.array_equal(layer.weight, val).item()
    with pytest.raises(ValueError, match="shape"):
        load_weights(nn.Linear(4, 3, bias=False), art)
    art.weight_map["bogus"] = "s.safetensors"
    with pytest.raises(ValueError, match="model/checkpoint mismatch"):
        load_weights(layer, art)


def test_shards_preserve_previously_assigned_array_list_entries(tmp_path):
    class Four(nn.Module):
        def __init__(self):
            super().__init__()
            self.scales = [mx.zeros((1,)) for _ in range(4)]

    model = Four()
    index = {}
    for i in range(4):
        name = f"scales.{i}"
        index[name] = f"shard-{i}.safetensors"
        mx.save_safetensors(
            str(tmp_path / index[name]),
            {name: mx.array([float(i + 10)])},
            metadata={"format": "mlx"},
        )
    load_weights(model, SimpleNamespace(root=tmp_path, weight_map=index))
    assert mx.stack(model.scales).tolist() == [[10], [11], [12], [13]]


def test_greedy_does_not_redecode_prompt_or_eos():
    class Next:
        def __init__(self):
            self.calls = []

        def make_cache(self):
            return []

        def __call__(self, ids, *, cache, logits_to_keep):
            cache.extend(ids.tolist()[0])
            self.calls.append(tuple(cache))
            assert logits_to_keep == 1
            return mx.eye(8)[(cache[-1] + 1) % 8].reshape(1, 1, 8)

    mod = Next()
    out, stats = greedy(mod, [0, 1, 2], eos={4}, count=6, chunk=2)
    assert out == [3, 4] and mod.calls == [(0, 1), (0, 1, 2), (0, 1, 2, 3)]
    assert stats["decode_steps"] == 1 and stats["stopped_on_eos"]


def test_cli_route_available_without_any_model():
    repo = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, "-B", "-m", "omlx.cli", "hy4-flash", "--help"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0 and all(
        s in r.stdout for s in ("convert", "inspect", "verify-tables", "generate")
    )
