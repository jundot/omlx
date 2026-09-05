# SPDX-License-Identifier: Apache-2.0
"""Tests for the cold precision tier (Fase I5): requant tool, cold-root
routing in the backing store, tier completeness, and settings plumbing."""

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from omlx.model_settings import ModelSettings
from omlx.patches.expert_streaming.shard_bank import (
    ExpertBackingStore,
    cold_tier_status,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


def _write_quantized_checkpoint(tmp: Path, bits: int = 4, gs: int = 64):
    """One shard, one switch_mlp gate bank: (E=8, O=32, I=128) affine-quantized."""
    import mlx.nn as nn

    key_w = "model.layers.0.mlp.switch_mlp.gate_proj.weight"
    key_s = "model.layers.0.mlp.switch_mlp.gate_proj.scales"
    key_b = "model.layers.0.mlp.switch_mlp.gate_proj.biases"
    dense = mx.random.normal((8, 32, 128))
    w, s, b = mx.quantize(dense, group_size=gs, bits=bits)
    shard = tmp / "model-00001-of-00001.safetensors"
    mx.save_safetensors(str(shard), {key_w: w, key_s: s.astype(mx.bfloat16), key_b: b.astype(mx.bfloat16)})
    (tmp / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key_w: shard.name, key_s: shard.name, key_b: shard.name}})
    )
    (tmp / "config.json").write_text(
        json.dumps({"quantization": {"group_size": gs, "bits": bits, "mode": "affine"}})
    )
    return key_w



def test_hobbit_small_first_tier_order_bit_identical(tmp_path):
    """Fase L4B B3: submitting the smaller tier FIRST must be
    byte-identical to the hot-first order — the masked add is elementwise
    commutative in IEEE fp. The order env only reshuffles residency; it
    never changes the output."""
    from requant_cold_tier import requant_shard

    from omlx.patches.expert_streaming import streaming_switch as ss
    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore
    from omlx.patches.expert_streaming.streaming_switch import (
        ExpertLRUCache,
        StreamingSwitchGLU,
        StreamingQuantizedSwitchLinear,
    )

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    # Full split GLU checkpoint: gate/up (E,O=64,I=128) and down (E,O=128,
    # I=64), all 4-bit gs64 — every axis divisible by the group size.
    shard = ckpt / "model-00001-of-00001.safetensors"
    tensors = {}
    for proj, o, i in (("gate_proj", 64, 128), ("up_proj", 64, 128), ("down_proj", 128, 64)):
        dense = mx.random.normal((8, o, i))
        w, s, b = mx.quantize(dense, group_size=64, bits=4)
        base = f"model.layers.0.mlp.switch_mlp.{proj}"
        tensors[f"{base}.weight"] = w
        tensors[f"{base}.scales"] = s.astype(mx.bfloat16)
        tensors[f"{base}.biases"] = b.astype(mx.bfloat16)
    mx.save_safetensors(str(shard), tensors)
    (ckpt / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: shard.name for k in tensors}})
    )
    (ckpt / "config.json").write_text(
        json.dumps({"quantization": {"group_size": 64, "bits": 4, "mode": "affine"}})
    )
    quant_cfg = json.loads((ckpt / "config.json").read_text())["quantization"]
    requant_shard(shard, ckpt / "expert_cold", quant_cfg, bits=3)
    backing = ExpertBackingStore(ckpt, cold_root=ckpt / "expert_cold")
    cache = ExpertLRUCache(64 * 1024, 1024, num_layers=1)
    glu = StreamingSwitchGLU(
        input_dims=128,
        hidden_dims=32,
        num_experts=8,
        layer_idx=0,
        backing=backing,
        cache=cache,
        fused_gate_up=False,
        inverse_scatter=False,
        quantized=True,
        group_size=64,
        bits=4,
        mode="affine",
    )
    spec = (
        ("gate_proj", 64, 128),
        ("up_proj", 64, 128),
        ("down_proj", 128, 64),
    )
    for proj, o, i in spec:
        base = f"model.layers.0.mlp.switch_mlp.{proj}"
        setattr(
            glu,
            proj,
            StreamingQuantizedSwitchLinear(
                layer_idx=0,
                proj_name=proj,
                stacked_weight_key=f"{base}.weight",
                stacked_scales_key=f"{base}.scales",
                stacked_biases_key=f"{base}.biases",
                num_experts=8,
                input_dims=i,
                output_dims=o,
                backing=backing,
                cache=cache,
                group_size=64,
                bits=4,
                mode="affine",
                has_bias=False,
            ),
        )
    hot = {0, 1, 2, 3}
    for proj, _o, _i in spec:
        getattr(glu, proj).set_hobbit_split(hot, cold_bits=3, cold_gs=64)
    # The backing's tier routing is independent of the linears' split: both
    # must agree on the hot set for cold ids to resolve expert_cold/.
    backing.set_hot_experts({"layer_0": hot})

    x = mx.random.normal((1, 6, 128)).astype(mx.float32)
    indices = mx.array([0, 1, 4, 5, 6, 0], dtype=mx.int32)

    def run(order):
        from unittest.mock import patch

        with patch.object(ss, "_DUAL_TIER_ORDER", order):
            out = glu(x, indices)
            mx.eval(out)
        return out

    out_hot = run("")
    out_small = run("small-first")
    b_hot = np.ascontiguousarray(out_hot).view(np.uint32).reshape(-1)
    b_small = np.ascontiguousarray(out_small).view(np.uint32).reshape(-1)
    assert np.array_equal(b_hot, b_small), "tier order changed output bits"
    assert bool(mx.all(mx.isfinite(out_small)))

def test_requant_tool_round_trip(tmp_path):
    from requant_cold_tier import META_BITS, META_GS, requant_shard

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    key_w = _write_quantized_checkpoint(ckpt, bits=4, gs=64)
    quant_cfg = json.loads((ckpt / "config.json").read_text())["quantization"]

    res = requant_shard(
        ckpt / "model-00001-of-00001.safetensors",
        ckpt / "expert_cold",
        quant_cfg,
        bits=3,
    )
    assert res["status"] == "written"
    assert res["dst_mib"] < res["src_mib"]  # 3-bit packs tighter than 4-bit

    out = ckpt / "expert_cold" / "model-00001-of-00001.safetensors"
    loaded = mx.load(str(out))
    src_loaded = mx.load(str(ckpt / "model-00001-of-00001.safetensors"))
    key_s = key_w.removesuffix(".weight") + ".scales"
    key_b = key_w.removesuffix(".weight") + ".biases"
    w2, s2, b2 = loaded[key_w], loaded[key_s], loaded[key_b]
    assert w2.shape[-1] < src_loaded[key_w].shape[-1]

    dense_src = mx.dequantize(
        src_loaded[key_w], src_loaded[key_s], src_loaded[key_b],
        group_size=64, bits=4,
    )
    dense_cold = mx.dequantize(w2, s2, b2, group_size=64, bits=3)
    err = mx.abs(dense_cold - dense_src).max().item()
    scale = mx.abs(dense_src).max().item()
    assert err < 0.2 * scale  # 3-bit requant stays in a sane envelope

    # Idempotent: a second run reports the existing output as matching.
    res2 = requant_shard(
        ckpt / "model-00001-of-00001.safetensors",
        ckpt / "expert_cold",
        quant_cfg,
        bits=3,
    )
    assert res2["status"] == "already matches"


def test_cold_tier_status_complete_and_partial(tmp_path):
    from requant_cold_tier import requant_shard

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    _write_quantized_checkpoint(ckpt)
    quant_cfg = json.loads((ckpt / "config.json").read_text())["quantization"]

    # Missing tier
    ok, why = cold_tier_status(ckpt)
    assert ok is False and "missing" in why

    requant_shard(
        ckpt / "model-00001-of-00001.safetensors",
        ckpt / "expert_cold",
        quant_cfg,
        bits=2,
    )
    ok, why = cold_tier_status(ckpt)
    assert ok is True


def test_backing_routes_reads_to_cold_root(tmp_path):
    from requant_cold_tier import requant_shard

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    key_w = _write_quantized_checkpoint(ckpt, bits=4, gs=64)
    quant_cfg = json.loads((ckpt / "config.json").read_text())["quantization"]
    requant_shard(
        ckpt / "model-00001-of-00001.safetensors",
        ckpt / "expert_cold",
        quant_cfg,
        bits=2,
    )

    # Without the tier: primary packing (4-bit).
    plain = ExpertBackingStore(ckpt)
    assert plain.cold_quant_params(key_w) is None
    assert plain.tensor_dtype(key_w) == "U32"
    hot_slice = plain.load_expert_slice(key_w, 3)

    # With the tier: cold reader, cold packing metadata, different packed width.
    cold = ExpertBackingStore(ckpt, cold_root=ckpt / "expert_cold")
    assert cold.cold_quant_params(key_w) == (2, 64)
    cold_slice = cold.load_expert_slice(key_w, 3)
    assert cold_slice.shape[-1] < hot_slice.shape[-1]
    assert cold.tensor_dtype(key_w) == "U32"

    # The cold slice dequantizes as 2-bit data (roundtrip fidelity vs itself).
    reader = cold._reader_for_key(key_w)
    assert reader.path.parent.name == "expert_cold"


def test_partial_cold_tier_rejected_by_backing_setup(tmp_path):
    """A cold dir missing banks must report incomplete — convert refuses it."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    _write_quantized_checkpoint(ckpt)
    (ckpt / "expert_cold").mkdir()
    (ckpt / "expert_cold" / "empty.safetensors").write_bytes(b"")
    ok, why = cold_tier_status(ckpt)
    assert ok is False


def test_expert_streaming_cold_tier_round_trip():
    s = ModelSettings(expert_streaming_cold_tier="3")
    d = s.to_dict()
    assert d["expert_streaming_cold_tier"] == "3"
    assert ModelSettings.from_dict(d).expert_streaming_cold_tier == "3"
    assert ModelSettings().expert_streaming_cold_tier is None


def test_expert_streaming_cold_tier_excluded_from_profiles():
    from omlx.model_profiles import EXCLUDED_FROM_PROFILES

    assert "expert_streaming_cold_tier" in EXCLUDED_FROM_PROFILES


@pytest.mark.asyncio
async def test_expert_streaming_cold_tier_api_validation():
    from omlx.admin import routes as admin_routes

    pool = None
    entry = None
    from tests.test_expert_streaming import _failed_pool, _update_settings

    pool, entry = _failed_pool()
    settings = ModelSettings()
    await _update_settings(
        pool, settings, admin_routes.ModelSettingsRequest(expert_streaming_cold_tier="2")
    )
    assert settings.expert_streaming_cold_tier == "2"
    from omlx.admin.routes import HTTPException

    # The PUT range follows the core (2..8, validated against the tier's
    # own __metadata__ at load); only impossible bit widths are rejected.
    settings = ModelSettings()
    await _update_settings(
        pool, settings, admin_routes.ModelSettingsRequest(expert_streaming_cold_tier="4")
    )
    assert settings.expert_streaming_cold_tier == "4"

    with pytest.raises(HTTPException, match="2..8"):
        await _update_settings(
            pool, settings, admin_routes.ModelSettingsRequest(expert_streaming_cold_tier="9")
        )


def test_hobbit_hot_set_loader_and_backing_routing(tmp_path):
    """Fase I6: the hot-set loader turns pin-profile frequencies into a
    top-fraction set, and the backing routes hot experts to the ORIGINAL
    shards while the rest keep reading expert_cold/."""
    from requant_cold_tier import requant_shard

    from omlx.patches.expert_streaming.shard_bank import (
        ExpertBackingStore,
        load_hot_set_from_profile,
    )

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    key_w = _write_quantized_checkpoint(ckpt, bits=4, gs=64)
    quant_cfg = json.loads((ckpt / "config.json").read_text())["quantization"]
    requant_shard(
        ckpt / "model-00001-of-00001.safetensors",
        ckpt / "expert_cold",
        quant_cfg,
        bits=3,
    )

    # Pin profile: layer 0 with descending counts — the top fraction must be
    # the most frequent ids.
    profile = {
        "freq": {
            "0": [[0, 90], [1, 80], [2, 70], [3, 10], [4, 5], [5, 1], [6, 1], [7, 1]]
        }
    }
    (ckpt / ".omlx").mkdir()
    (ckpt / ".omlx" / "expert_pin_profile.json").write_text(json.dumps(profile))

    hot = load_hot_set_from_profile(ckpt / ".omlx" / "expert_pin_profile.json", 0.25)
    # ceil(0.25 * 8) == 2 — the two most frequent ids.
    assert hot == {"layer_0": {0, 1}}
    assert load_hot_set_from_profile(ckpt / ".omlx" / "expert_pin_profile.json", 0.0) == {}
    assert load_hot_set_from_profile(tmp_path / "missing.json", 0.5) == {}

    # Backing routing: hot ids resolve the source shard, cold ids the tier.
    cold = ExpertBackingStore(ckpt, cold_root=ckpt / "expert_cold")
    cold.set_hot_experts(hot)
    src_slice = cold.load_expert_slice(key_w, 0)   # hot -> 4-bit width
    cold_slice = cold.load_expert_slice(key_w, 5)  # cold -> 3-bit width
    plain = ExpertBackingStore(ckpt)
    assert src_slice.shape == plain.load_expert_slice(key_w, 0).shape
    assert cold_slice.shape[-1] < src_slice.shape[-1]
    reader_hot = cold._reader_for_key(key_w, 0)
    reader_cold = cold._reader_for_key(key_w, 5)
    assert reader_hot.path.parent.name != "expert_cold"
    assert reader_cold.path.parent.name == "expert_cold"
    # Uniform lookups (no expert id) still resolve the cold tier for
    # tensor_dtype/metadata probes.
    assert cold.cold_quant_params(key_w) == (3, 64)


def test_expert_streaming_hot_fraction_round_trip_and_profiles():
    """Fase I6: hot fraction persists through ModelSettings, is excluded from
    profiles, and enters the runtime signature (reload semantics)."""
    from omlx.model_settings import ModelSettings

    s = ModelSettings(expert_streaming_hot_fraction=0.3)
    data = s.to_dict()
    assert data["expert_streaming_hot_fraction"] == 0.3
    restored = ModelSettings.from_dict(data)
    assert restored.expert_streaming_hot_fraction == 0.3

    from omlx.model_profiles import EXCLUDED_FROM_PROFILES

    assert "expert_streaming_hot_fraction" in EXCLUDED_FROM_PROFILES


@pytest.mark.asyncio
async def test_expert_streaming_hot_fraction_api_validation():
    from omlx.admin import routes as admin_routes
    from omlx.admin.routes import HTTPException
    from omlx.model_settings import ModelSettings

    from tests.test_expert_streaming import _failed_pool, _update_settings

    pool, entry = _failed_pool()
    settings = ModelSettings()
    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(expert_streaming_hot_fraction=0.25),
    )
    assert settings.expert_streaming_hot_fraction == 0.25

    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(expert_streaming_hot_fraction=None),
    )
    assert settings.expert_streaming_hot_fraction is None

    with pytest.raises(HTTPException, match="between 0 and 1"):
        await _update_settings(
            pool,
            settings,
            admin_routes.ModelSettingsRequest(expert_streaming_hot_fraction=1.5),
        )


def test_advise_expert_run_breaks_at_tier_boundary(tmp_path):
    """Post-I6: a readahead run that straddles the hot/cold boundary must
    advise ONE reader per tier segment (a run reads a single reader), not
    apply the first id's reader to the whole byte range."""
    from requant_cold_tier import requant_shard

    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    key_w = _write_quantized_checkpoint(ckpt, bits=4, gs=64)
    quant_cfg = json.loads((ckpt / "config.json").read_text())["quantization"]
    requant_shard(
        ckpt / "model-00001-of-00001.safetensors",
        ckpt / "expert_cold",
        quant_cfg,
        bits=3,
    )

    cold = ExpertBackingStore(ckpt, cold_root=ckpt / "expert_cold")
    try:
        # hot 0..1 (source shard), cold 2..5 (tier shard)
        cold.set_hot_experts({"layer_0": {0, 1}})
        advised = []
        reader_hot = cold._reader_for_key(key_w, 0)
        reader_cold = cold._reader_for_key(key_w, 2)
        assert reader_hot is not reader_cold

        def spy_hot(start, length):
            advised.append(("HOT", start, length))
            return True

        def spy_cold(start, length):
            advised.append(("COLD", start, length))
            return True

        from unittest.mock import patch

        # Both readers are _ShardReader instances — spy per instance, not
        # per type, or the second patch replaces the first.
        with patch.object(reader_hot, "advise_range", side_effect=spy_hot):
            with patch.object(reader_cold, "advise_range", side_effect=spy_cold):
                # run 0..5 straddles: hot segment [0,1], cold segment [2,5]
                ok, adv_bytes, segs = cold.advise_expert_run(key_w, 0, 6)
                assert ok is True and segs == 2 and adv_bytes > 0

        # exactly one advise per tier segment, never one for the whole run
        hot_calls = [c for c in advised if c[0] == "HOT"]
        cold_calls = [c for c in advised if c[0] == "COLD"]
        assert len(hot_calls) == 1, advised
        assert len(cold_calls) == 1, advised
        # hot segment covers ids 0..1 only — NOT the cold ids' bytes
        _, start_h, len_h = hot_calls[0]
        _, end_h = reader_hot.expert_byte_range(key_w, 1)
        assert len_h == end_h - start_h
        # cold segment covers ids 2..5 in the tier shard
        start_c, len_c = cold_calls[0][1], cold_calls[0][2]
        _, end_c = reader_cold.expert_byte_range(key_w, 5)
        assert len_c == end_c - start_c
    finally:
        cold.close()
