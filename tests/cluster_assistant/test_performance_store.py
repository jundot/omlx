# SPDX-License-Identifier: Apache-2.0
"""Phase B1: the synthetic compute probe after apply, used by the next plan.

The probe is calibration, never a gate. Its profiles are model-independent and
stored on the origin; the next measurement reads them and the election plans
with real compute instead of ``unmeasured_compute``.
"""

from __future__ import annotations

import stat

from assistant_harness import (
    PERFORMANCE,
    assistant_api,
    take_measurement,
    take_plan,
)

from omlx.cluster.assistant.performance_store import AssistantPerformanceStore


def _probe_reply() -> dict:
    return {"profiles": PERFORMANCE, "backend": "ring", "world_size": 2}


def test_the_store_is_mode_0600_and_has_no_secrets(tmp_path) -> None:
    store = AssistantPerformanceStore(tmp_path / "store")

    store.save(PERFORMANCE, deployment_id="a" * 64, world_size=2, backend="ring")

    mode = stat.S_IMODE(store.path.stat().st_mode)
    assert mode == 0o600
    raw = store.path.read_text()
    assert "node_id" in raw
    for forbidden in ("ssh", "api_key", "model_path", "password", "@"):
        assert forbidden not in raw
    restored = AssistantPerformanceStore(tmp_path / "store").for_nodes(
        ["m5", "m1"]
    )
    assert set(restored) == {"m5", "m1"}


def test_the_store_rejects_a_malformed_file(tmp_path) -> None:
    path = tmp_path / "store" / "cluster" / "assistant-performance.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"schema_version": 99, "profiles": []}')

    store = AssistantPerformanceStore(tmp_path / "store")

    assert store.load_error
    assert store.for_nodes(["m5"]) == {}


def test_a_stored_profile_is_used_by_the_next_measurement(tmp_path, monkeypatch) -> None:
    """A profile the last apply stored plans the next measurement.

    The synthetic probe runs after apply; its profiles are model-independent,
    so the measurement reads them instead of reporting ``unmeasured_compute``.
    """

    from assistant_harness import PERFORMANCE

    AssistantPerformanceStore(tmp_path / "store").save(
        PERFORMANCE, deployment_id="a" * 64, world_size=2, backend="ring"
    )
    with assistant_api(tmp_path, monkeypatch) as (client, _store, _ops):
        measurement_id = take_measurement(client)
        plan = take_plan(client, measurement_id)

    assert plan["predicted"]["measured"] is True
    assert "unmeasured_compute" not in {
        warning["code"] for warning in plan["warnings"]
    }
