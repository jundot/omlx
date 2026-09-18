"""Unit tests for per-expert DeepSeek-V4 spill-stacking."""

import json

import mlx.core as mx

from omlx.patches.deepseek_v4 import spill as S


def _fake_weights(n_layers=2, n_experts=2):
    weights = {}
    for i in range(n_layers):
        for e in range(n_experts):
            for src, shp in (("w1", (8, 4)), ("w2", (4, 8)), ("w3", (8, 4))):
                weights[f"model.layers.{i}.ffn.experts.{e}.{src}.weight"] = mx.zeros(shp)
                weights[f"model.layers.{i}.ffn.experts.{e}.{src}.scales"] = mx.zeros((shp[0], 2))
                weights[f"model.layers.{i}.ffn.experts.{e}.{src}.biases"] = mx.zeros((shp[0], 2))
    return weights


def test_stack_layer_to_spill_roundtrip(tmp_path):
    weights = _fake_weights()
    out = S.stack_layer_to_spill(weights, layer_idx=0, n_experts=2, spill_dir=tmp_path)
    # per-expert keys popped, stacked keys returned
    assert not any(".ffn.experts." in k and k.startswith("model.layers.0.") for k in weights)
    assert len(out) == 9
    assert out["model.layers.0.ffn.switch_mlp.gate_proj.weight"].shape == (2, 8, 4)
    assert out["model.layers.0.ffn.switch_mlp.down_proj.scales"].shape == (2, 4, 2)


def test_spill_validity_roundtrip(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"x" * 64)
    spill_dir = tmp_path / "spill"
    spill_dir.mkdir()
    (spill_dir / "spill_layer_00.safetensors").write_bytes(b"y" * 64)
    S.write_manifest(spill_dir, model_dir, ["spill_layer_00.safetensors"], {"k": "spill_layer_00.safetensors"})
    assert S.spill_is_valid(model_dir) is None  # spill_dir_for(model) != spill_dir
    # validity is anchored at the conventional location
    conv = S.spill_dir_for(model_dir)
    conv.mkdir(parents=True)
    (conv / "spill_layer_00.safetensors").write_bytes(b"y" * 64)
    S.write_manifest(conv, model_dir, ["spill_layer_00.safetensors"], {"k": "spill_layer_00.safetensors"})
    assert S.spill_is_valid(model_dir) == conv
    # source change invalidates; the stale shard must not validate
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"x" * 65)
    assert S.spill_is_valid(model_dir) is None
    # empty file list never validates (would otherwise glob stale shards)
    S.write_manifest(conv, model_dir, [], {})
    assert S.spill_is_valid(model_dir) is None


# ---------------------------------------------------------------------------
# Spill mode (manifest ``mode`` block, v3) — a load whose MTP/DSpark mode
# differs from the writing load's must MISS, not validate and serve
# mismatched keys into strict load_weights.
# ---------------------------------------------------------------------------

_BACKBONE_K2F = {
    "model.layers.0.ffn.switch_mlp.gate_proj.weight": "spill_layer_00.safetensors",
}
_MTP_K2F_BLOCK = {
    **_BACKBONE_K2F,
    "mtp.0.block.ffn.switch_mlp.gate_proj.weight": "spill_mtp_00.safetensors",
}
_MTP_K2F_DSPARK = {
    **_BACKBONE_K2F,
    "mtp.0.ffn.switch_mlp.gate_proj.weight": "spill_mtp_00.safetensors",
    "mtp.1.ffn.switch_mlp.gate_proj.weight": "spill_mtp_01.safetensors",
}


def _model_and_spill(tmp_path):
    """Fresh model dir + its (empty) conventional spill dir."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"x" * 64)
    conv = S.spill_dir_for(model_dir)
    conv.mkdir(parents=True)
    return model_dir, conv


def _touch(conv, *names):
    for name in names:
        (conv / name).write_bytes(b"y" * 64)


def _patch_manifest(conv, *, drop=(), **fields):
    """Rewrite the on-disk manifest with fields replaced/dropped."""
    mp = conv / "manifest.json"
    m = json.loads(mp.read_text())
    for k in drop:
        m.pop(k, None)
    m.update(fields)
    mp.write_text(json.dumps(m))
    return m


def test_spill_manifest_records_mode(tmp_path):
    model_dir, conv = _model_and_spill(tmp_path)
    _touch(conv, "spill_layer_00.safetensors", "spill_mtp_00.safetensors")
    S.write_manifest(
        conv,
        model_dir,
        ["spill_layer_00.safetensors", "spill_mtp_00.safetensors"],
        _MTP_K2F_BLOCK,
        mtp_stages=1,
        block_part=".block",
    )
    m = S.read_manifest(conv)
    assert m["mode"] == {"mtp_stages": 1, "block_part": ".block"}
    # Zero stages normalize the spelling away — it serves nothing.
    S.write_manifest(conv, model_dir, ["spill_layer_00.safetensors"], _BACKBONE_K2F)
    assert S.read_manifest(conv)["mode"] == {"mtp_stages": 0, "block_part": None}


def test_spill_mode_mtp_on_manifest_mtp_off_load_is_miss(tmp_path):
    """MTP-on spill injected into an MTP-off load would leave strict
    load_weights holding mtp.* keys with no module — must miss."""
    model_dir, conv = _model_and_spill(tmp_path)
    files = ["spill_layer_00.safetensors", "spill_mtp_00.safetensors"]
    _touch(conv, *files)
    S.write_manifest(
        conv, model_dir, files, _MTP_K2F_BLOCK, mtp_stages=1, block_part=".block"
    )
    assert S.spill_is_valid(model_dir, mtp_stages=0, block_part=".block") is None
    # Callers without a mode expectation (the backing-store absorb) still
    # see the source/freshness check only.
    assert S.spill_is_valid(model_dir) == conv


def test_spill_mode_mtp_off_manifest_mtp_on_load_is_miss(tmp_path):
    """MTP-off spill under an MTP-on load serves no stage banks — the raw
    mtp experts get deleted on the hit path, so it must miss."""
    model_dir, conv = _model_and_spill(tmp_path)
    _touch(conv, "spill_layer_00.safetensors")
    S.write_manifest(conv, model_dir, ["spill_layer_00.safetensors"], _BACKBONE_K2F)
    assert S.spill_is_valid(model_dir, mtp_stages=1, block_part=".block") is None
    assert S.spill_is_valid(model_dir, mtp_stages=2, block_part="") is None
    # The matching expectation still hits.
    assert S.spill_is_valid(model_dir, mtp_stages=0) == conv


def test_spill_mode_block_part_mismatch_is_miss(tmp_path):
    """DSpark serves mtp.{i}.ffn.*; Lightning serves mtp.{i}.block.ffn.* —
    a spelling toggle must miss even at equal stage counts."""
    model_dir, conv = _model_and_spill(tmp_path)
    files = [
        "spill_layer_00.safetensors",
        "spill_mtp_00.safetensors",
        "spill_mtp_01.safetensors",
    ]
    _touch(conv, *files)
    S.write_manifest(conv, model_dir, files, _MTP_K2F_DSPARK, mtp_stages=2, block_part="")
    assert S.spill_is_valid(model_dir, mtp_stages=2, block_part=".block") is None
    assert S.spill_is_valid(model_dir, mtp_stages=2, block_part="") == conv


def test_spill_mode_same_mode_hits(tmp_path):
    model_dir, conv = _model_and_spill(tmp_path)
    files = ["spill_layer_00.safetensors", "spill_mtp_00.safetensors"]
    _touch(conv, *files)
    S.write_manifest(
        conv, model_dir, files, _MTP_K2F_BLOCK, mtp_stages=1, block_part=".block"
    )
    assert S.spill_is_valid(model_dir, mtp_stages=1, block_part=".block") == conv


def test_spill_mode_legacy_v2_manifest(tmp_path):
    """v2 manifests predate the mode block; the mode is derived from the
    key_to_file spelling so they validate only for a matching load."""
    model_dir, conv = _model_and_spill(tmp_path)
    files = ["spill_layer_00.safetensors", "spill_mtp_00.safetensors"]
    _touch(conv, *files)
    S.write_manifest(
        conv, model_dir, files, _MTP_K2F_BLOCK, mtp_stages=1, block_part=".block"
    )
    _patch_manifest(conv, drop=("mode",), version=2)
    # MTP-on v2 spill: hits the same-mode load, misses an MTP-off one.
    assert S.spill_is_valid(model_dir, mtp_stages=1, block_part=".block") == conv
    assert S.spill_is_valid(model_dir, mtp_stages=0) is None
    assert S.spill_is_valid(model_dir, mtp_stages=1, block_part="") is None

    # Backbone-only v2 spill (no mode fields, no mtp keys): validates
    # ONLY for a load expecting zero MTP stages.
    _patch_manifest(conv, files=["spill_layer_00.safetensors"], key_to_file=_BACKBONE_K2F)
    assert S.spill_is_valid(model_dir, mtp_stages=0, block_part=".block") == conv
    assert S.spill_is_valid(model_dir, mtp_stages=1, block_part=".block") is None

    # v1 stays rejected outright even without a mode expectation.
    _patch_manifest(conv, version=1)
    assert S.spill_is_valid(model_dir) is None


def test_manifest_mode_derivation():
    # Explicit v3 mode block wins.
    assert S.manifest_mode({"mode": {"mtp_stages": 2, "block_part": ""}}) == (2, "")
    assert S.manifest_mode({"mode": {"mtp_stages": 1, "block_part": ".block"}}) == (
        1,
        ".block",
    )
    # Zero stages normalize block_part away on read as on write.
    assert S.manifest_mode({"mode": {"mtp_stages": 0, "block_part": ".block"}}) == (
        0,
        None,
    )
    # Legacy manifests: derive from the key_to_file spelling.
    assert S.manifest_mode({"key_to_file": _MTP_K2F_BLOCK}) == (1, ".block")
    assert S.manifest_mode({"key_to_file": _MTP_K2F_DSPARK}) == (2, "")
    assert S.manifest_mode({"key_to_file": _BACKBONE_K2F}) == (0, None)
    # spill_mtp_* files with no recorded keys still count as stages
    # (block_part unknown -> never matches a spelling-expecting load).
    assert S.manifest_mode({"files": ["spill_mtp_00.safetensors"]}) == (1, None)
    assert S.manifest_mode({}) == (0, None)
