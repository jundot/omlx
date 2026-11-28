# SPDX-License-Identifier: Apache-2.0
"""Tuning presets persist per (model, hardware) so Start reuses a tuned setup."""

from __future__ import annotations

from omlx.cluster.assistant.tuning_store import TuningPreset, TuningPresetStore


def _preset(model="sig-m", hardware="m5max+m1max", **kw) -> TuningPreset:
    values = dict(
        chunk=2048,
        mtp=True,
        mtp_depth=4,
        sampler={"sampler": "balanced", "temperature": 0.7},
        measured_at="2026-11-14T00:00:00+00:00",
    )
    values.update(kw)
    return TuningPreset(
        model_signature=model, hardware_key=hardware, **values
    )


def test_save_and_get_round_trips(tmp_path) -> None:
    store = TuningPresetStore(tmp_path)
    store.save(_preset())
    loaded = store.get("sig-m", "m5max+m1max")
    assert loaded is not None
    assert loaded.chunk == 2048
    assert loaded.mtp is True
    assert loaded.mtp_depth == 4
    assert loaded.sampler["sampler"] == "balanced"


def test_presets_are_keyed_by_model_and_hardware(tmp_path) -> None:
    store = TuningPresetStore(tmp_path)
    store.save(_preset(model="sig-a", hardware="hw-1", chunk=1024))
    store.save(_preset(model="sig-a", hardware="hw-2", chunk=4096))
    assert store.get("sig-a", "hw-1").chunk == 1024
    assert store.get("sig-a", "hw-2").chunk == 4096
    # Never reused across hardware.
    assert store.get("sig-b", "hw-1") is None
    assert store.get("sig-a", "hw-3") is None


def test_save_overwrites_the_same_key(tmp_path) -> None:
    store = TuningPresetStore(tmp_path)
    store.save(_preset(chunk=2048))
    store.save(_preset(chunk=4096))
    assert store.get("sig-m", "m5max+m1max").chunk == 4096
    assert len(store.all()) == 1


def test_corrupt_file_fails_closed_to_no_preset(tmp_path) -> None:
    store = TuningPresetStore(tmp_path)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("{ not json", encoding="utf-8")
    assert store.all() == {}
    assert store.get("sig-m", "m5max+m1max") is None
    # A save after corruption still works (repairs the file).
    store.save(_preset())
    assert store.get("sig-m", "m5max+m1max").chunk == 2048


def test_wrong_schema_is_ignored(tmp_path) -> None:
    import json

    store = TuningPresetStore(tmp_path)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(
        json.dumps({"schema_version": 99, "presets": {}}), encoding="utf-8"
    )
    assert store.all() == {}

def test_hardware_key_is_the_node_set_not_the_rank_order() -> None:
    from omlx.cluster.assistant.tuning import hardware_key

    # A later election may flip rank 0; the same Macs must find the same preset.
    assert hardware_key(["b", "a"]) == hardware_key(["a", "b"]) == "a,b"


def test_merge_keeps_the_fields_this_save_did_not_measure() -> None:
    from omlx.cluster.assistant.tuning_store import merge_preset

    existing = _preset(mtp_depth=3, sampler=None)
    # /tune measures the sampler only; the redeploy's depth must survive.
    merged = merge_preset(
        existing,
        model_signature="sig-m",
        hardware_key="m5max+m1max",
        sampler={"sampler": "deterministic"},
        measured_at="2026-11-14T01:00:00+00:00",
    )
    assert merged.mtp_depth == 3
    assert merged.chunk == 2048
    assert merged.mtp is True
    assert merged.sampler["sampler"] == "deterministic"
    assert merged.measured_at == "2026-11-14T01:00:00+00:00"


def test_merge_keeps_the_sampler_a_redeploy_did_not_measure() -> None:
    from omlx.cluster.assistant.tuning_store import merge_preset

    existing = _preset(sampler={"sampler": "balanced"})
    # /tune-redeploy measures mtp + depth only; the sampler must survive.
    merged = merge_preset(
        existing,
        model_signature="sig-m",
        hardware_key="m5max+m1max",
        mtp=True,
        mtp_depth=2,
    )
    assert merged.sampler["sampler"] == "balanced"
    assert merged.mtp_depth == 2


def test_merge_lets_a_measured_false_or_none_win() -> None:
    from omlx.cluster.assistant.tuning_store import merge_preset

    existing = _preset(mtp=True, mtp_depth=3)
    # MTP off is a measurement, not a missing value.
    off = merge_preset(
        existing,
        model_signature="sig-m",
        hardware_key="m5max+m1max",
        mtp=False,
        mtp_depth=None,
    )
    assert off.mtp is False
    # The adaptive config won: "no fixed depth" is the measured answer.
    assert off.mtp_depth is None


def test_merge_without_a_preset_makes_a_partial_one() -> None:
    from omlx.cluster.assistant.tuning_store import merge_preset

    merged = merge_preset(
        None,
        model_signature="sig-m",
        hardware_key="m5max+m1max",
        sampler={"sampler": "balanced"},
    )
    assert merged.chunk is None
    assert merged.mtp is None
    assert merged.mtp_depth is None
    assert merged.sampler["sampler"] == "balanced"
