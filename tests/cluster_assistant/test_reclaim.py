# SPDX-License-Identifier: Apache-2.0
"""The automatic free-disk selection obeys the frozen delete rule."""

from __future__ import annotations

from types import SimpleNamespace

from omlx.cluster.assistant.reclaim import (
    Deletable,
    deletable_models,
    holders,
    model_key,
    select_for_space,
)


def _record(inventories, planned="/m/Planned"):
    return SimpleNamespace(
        model=SimpleNamespace(model_path=planned),
        inventories=inventories,
    )


def test_model_key_is_the_folder_name_case_insensitively() -> None:
    assert model_key("/Users/a/Models/GLM") == model_key("/Users/a/models/glm")


def test_only_models_that_mac_listed_are_candidates() -> None:
    record = _record(
        {
            "n1": [
                {"model_id": "A", "model_path": "/m/A", "size_bytes": 10},
            ],
            "n2": [
                {"model_id": "A", "model_path": "/m/A", "size_bytes": 10},
                {"model_id": "B", "model_path": "/m/B", "size_bytes": 20},
            ],
        }
    )
    names = {c.model_id for c in deletable_models(record, "n1")}
    assert names == {"A"}
    assert "B" not in names  # n1 never listed B


def test_the_planned_checkpoint_is_never_a_candidate() -> None:
    record = _record(
        {
            "n1": [
                {"model_id": "Planned", "model_path": "/m/Planned", "size_bytes": 99},
                {"model_id": "Old", "model_path": "/m/Old", "size_bytes": 10},
            ],
            "n2": [
                {"model_id": "Planned", "model_path": "/m/Planned", "size_bytes": 99},
                {"model_id": "Old", "model_path": "/m/Old", "size_bytes": 10},
            ],
        }
    )
    assert {c.model_id for c in deletable_models(record, "n1")} == {"Old"}


def test_the_last_cluster_copy_is_never_a_candidate() -> None:
    record = _record(
        {
            "n1": [
                {"model_id": "Only", "model_path": "/m/Only", "size_bytes": 50},
                {"model_id": "Shared", "model_path": "/m/Shared", "size_bytes": 10},
            ],
            "n2": [
                {"model_id": "Shared", "model_path": "/m/Shared", "size_bytes": 10},
            ],
        }
    )
    assert holders(record)["only"] == {"n1"}
    assert {c.model_id for c in deletable_models(record, "n1")} == {"Shared"}


def test_candidates_are_largest_first() -> None:
    record = _record(
        {
            "n1": [
                {"model_id": "S", "model_path": "/m/S", "size_bytes": 5},
                {"model_id": "L", "model_path": "/m/L", "size_bytes": 50},
                {"model_id": "M", "model_path": "/m/M", "size_bytes": 20},
            ],
            "n2": [
                {"model_id": "S", "model_path": "/m/S", "size_bytes": 5},
                {"model_id": "L", "model_path": "/m/L", "size_bytes": 50},
                {"model_id": "M", "model_path": "/m/M", "size_bytes": 20},
            ],
        }
    )
    assert [c.model_id for c in deletable_models(record, "n1")] == ["L", "M", "S"]


def test_select_for_space_stops_once_the_shortfall_is_covered() -> None:
    candidates = [
        Deletable("n1", "L", "/m/L", 50),
        Deletable("n1", "M", "/m/M", 20),
        Deletable("n1", "S", "/m/S", 5),
    ]
    assert [c.model_id for c in select_for_space(candidates, 40)] == ["L"]
    assert [c.model_id for c in select_for_space(candidates, 55)] == ["L", "M"]
    assert sum(c.size_bytes for c in select_for_space(candidates, 999)) == 75


def test_a_case_only_path_difference_is_one_model() -> None:
    record = _record(
        {
            "n1": [{"model_id": "A", "model_path": "/Users/a/Models/A", "size_bytes": 10}],
            "n2": [{"model_id": "A", "model_path": "/Users/a/models/a", "size_bytes": 10}],
        },
        planned="/Users/a/Models/Other",
    )
    # "A" is on two Macs (case differs), so it is not the last copy.
    assert {c.model_id for c in deletable_models(record, "n1")} == {"A"}