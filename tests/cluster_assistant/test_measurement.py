# SPDX-License-Identifier: Apache-2.0
"""Tests for the measurement layer that feeds the coordinator election.

The one rule these tests exist to pin down is the negative one: **a measurement
that did not happen must not become a number**. A zero, a nominal rate or a
synthesised p50 would all make the election answer a question nobody measured —
and make an unmeasured plan look fast, which is what the project design notes forbids.
"""

from __future__ import annotations

import json

import pytest

from omlx.cluster.assistant.calibration import (
    CompletionTimings,
    compute_profile_from_timings,
)
from omlx.cluster.assistant.election import elect
from omlx.cluster.assistant.measurement import (
    DEFAULT_ENDPOINT_PORT,
    WARNING_ADMISSION_ABOVE_CAPACITY,
    WARNING_ADMISSION_DERIVED_FROM_ROLE,
    WARNING_ADMISSION_SPREAD_INCONSISTENT,
    WARNING_ADMISSION_SPREAD_WITHOUT_P05,
    WARNING_INSTALLED_RAM_CAPACITY,
    WARNING_PERFORMANCE_NODE_MISMATCH,
    WARNING_UNMEASURED_COMPUTE,
    WARNING_UNREADABLE_CAPACITY,
    WARNING_UNVERIFIED_CAPACITY,
    MeasuredNode,
    measure_budget_payloads,
    measure_nodes,
    measured_endpoint_port,
    measured_node_from_budget,
    reachable_endpoint_host,
    with_calibration,
)
from omlx.cluster.link_bandwidth import LinkBandwidth
from omlx.cluster.performance import NodePerformanceProfile
from omlx.cluster.planner import synthetic_model_layout

GIB = 1024**3

MEASURED_AT = "2026-11-03T12:00:00+00:00"


def profile(
    node_id: str = "m5",
    *,
    decode_gib_per_s: float = 2.0,
    backend: str = "jaccl-ring",
) -> NodePerformanceProfile:
    """A performance-probe record, in the units oMLX's probe writes."""

    return NodePerformanceProfile(
        node_id=node_id,
        rank=0,
        decode_weight_bytes_per_second=decode_gib_per_s * GIB,
        prefill_weight_bytes_per_second=decode_gib_per_s * 2 * GIB,
        collective_latency_seconds=0.0003,
        collective_bandwidth_bytes_per_second=6.6e9,
        backend=backend,
        measured_at=MEASURED_AT,
        samples=5,
    )


def budget_payload(**overrides) -> dict:
    """One node entry as ``/admin/api/cluster/node-budgets`` reports it."""

    payload = {
        "node_id": "m5",
        "ssh": "127.0.0.1",
        "capacity_bytes": int(107.5 * GIB),
        "reserve_bytes": int(10.75 * GIB),
        "usable_bytes": int(96.75 * GIB),
        "role": "headless",
        "capacity_source": "admission_ceiling",
        "summary": "97 GiB for the cluster of 108 GiB this Mac can admit",
    }
    payload.update(overrides)
    return payload


def small_budget_payload(**overrides) -> dict:
    """The 64 GiB M1 shape: 51.2 GiB usable, measured over SSH."""

    payload = {
        "node_id": "m1",
        "ssh": "192.0.2.14",
        "capacity_bytes": int(60 * GIB),
        "reserve_bytes": int(6 * GIB),
        "usable_bytes": int(54 * GIB),
        "role": "headless",
        "capacity_source": "admission_ceiling",
        "summary": "54 GiB for the cluster",
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Payload → facts
# ---------------------------------------------------------------------------


def test_node_budget_payload_becomes_node_facts() -> None:
    """The measured ceiling and the role's reserve, not installed RAM."""

    payload = budget_payload(
        endpoint_host="192.0.2.11",
        disk_free_bytes=300 * GIB,
        performance=profile().to_dict(),
    )

    measured = measured_node_from_budget(payload)
    assert measured.capacity_source == "admission_ceiling"
    assert measured.admission_p05_bytes == int(96.75 * GIB)

    result = measure_nodes([measured], measured_at=MEASURED_AT)
    assert result.measured
    node = result.nodes[0]
    assert node.node_id == "m5"
    assert node.capacity_gib == 107.5
    assert node.admission_p05_gib == 96.75
    assert node.admission_bytes == int(96.75 * GIB)
    # One admission number existed, so there is no spread to report.
    assert node.admission_p50_gib is None
    assert node.admission_p95_gib is None
    assert node.endpoint_host == "192.0.2.11"
    assert node.disk_free_gib == 300
    assert result.measured_at == MEASURED_AT
    # Everything that matters was measured: nothing to warn about.
    assert result.warnings == ()
    # ...and the payload survives an API round trip.
    assert json.loads(json.dumps(result.to_dict()))["nodes"][0]["capacity_gib"] == 107.5


def test_missing_measurements_stay_missing() -> None:
    """No probe, no percentiles, no timestamp: the facts say so by omission."""

    result = measure_nodes([measured_node_from_budget(budget_payload())])
    facts = result.nodes[0]

    assert facts.compute is None  # not a zero rate, not a chip-name estimate
    assert facts.admission_p50_gib is None
    assert facts.admission_p95_gib is None
    assert result.measured_at == ""  # this process did not date the measurement
    assert [item.code for item in result.warnings] == [WARNING_UNMEASURED_COMPUTE]
    assert (
        "unknown" in result.warning_messages()[0]
        or "worst case" in (result.warning_messages()[0])
    )


def test_unmeasured_admission_is_the_role_ceiling_and_says_so() -> None:
    """A role derived ceiling is not a measurement, and is labelled as one."""

    node = MeasuredNode(node_id="studio", capacity_bytes=int(64 * GIB), role="headless")
    result = measure_nodes([node])

    facts = result.nodes[0]
    # 64 GiB less the headless role's 10% planning margin.
    assert facts.admission_p05_gib == pytest.approx(57.6)
    assert facts.admission_p50_gib is None
    assert facts.admission_p95_gib is None
    assert [item.code for item in result.warnings] == [
        WARNING_ADMISSION_DERIVED_FROM_ROLE,
        WARNING_UNMEASURED_COMPUTE,
    ]


def test_a_measured_spread_is_kept_exactly_as_measured() -> None:
    node = MeasuredNode(
        node_id="studio",
        capacity_bytes=int(64 * GIB),
        admission_p05_bytes=int(48 * GIB),
        admission_p50_bytes=int(52 * GIB),
        admission_p95_bytes=int(56 * GIB),
    )
    facts = measure_nodes([node]).nodes[0]
    assert (
        facts.admission_p05_gib,
        facts.admission_p50_gib,
        facts.admission_p95_gib,
    ) == (48, 52, 56)


def test_a_spread_without_a_floor_is_dropped() -> None:
    """p50/p95 cannot be the budget: the plan is built against the floor."""

    node = MeasuredNode(
        node_id="studio",
        capacity_bytes=int(64 * GIB),
        admission_p50_bytes=int(50 * GIB),
        admission_p95_bytes=int(55 * GIB),
    )
    result = measure_nodes([node])
    facts = result.nodes[0]

    assert facts.admission_p50_gib is None
    assert facts.admission_p95_gib is None
    assert facts.admission_p05_gib == pytest.approx(57.6)
    assert WARNING_ADMISSION_SPREAD_WITHOUT_P05 in {
        item.code for item in result.warnings
    }


def test_an_inconsistent_spread_is_dropped_rather_than_reversed() -> None:
    node = MeasuredNode(
        node_id="studio",
        capacity_bytes=int(64 * GIB),
        admission_p05_bytes=int(52 * GIB),
        admission_p50_bytes=int(48 * GIB),
    )
    result = measure_nodes([node])
    facts = result.nodes[0]

    assert facts.admission_p05_gib == 52
    assert facts.admission_p50_gib is None
    assert WARNING_ADMISSION_SPREAD_INCONSISTENT in {
        item.code for item in result.warnings
    }


def test_admission_above_capacity_is_clamped_with_a_warning() -> None:
    node = MeasuredNode(
        node_id="studio",
        capacity_bytes=int(128 * GIB),
        admission_p05_bytes=int(140 * GIB),
    )
    result = measure_nodes([node])

    assert result.nodes[0].admission_p05_gib == 128
    assert WARNING_ADMISSION_ABOVE_CAPACITY in {item.code for item in result.warnings}


# ---------------------------------------------------------------------------
# Capacities that must not be planned against
# ---------------------------------------------------------------------------


def test_installed_ram_capacity_is_refused_by_default() -> None:
    """node_role.py's incident: 128 GiB installed is not 128 GiB addressable."""

    payload = budget_payload(
        capacity_source="installed_ram",
        capacity_bytes=int(128 * GIB),
        usable_bytes=int(115.2 * GIB),
    )
    refused = measure_budget_payloads([payload])

    assert refused.nodes == ()
    assert refused.measured is False
    assert [item.code for item in refused.warnings] == [WARNING_INSTALLED_RAM_CAPACITY]
    assert "installed RAM" in refused.warning_messages()[0]

    # An explicit opt-in is possible, and the warning still names the risk.
    allowed = measure_budget_payloads([payload], allow_installed_ram_capacity=True)
    assert allowed.nodes[0].capacity_gib == 128
    assert allowed.warnings[0].code == WARNING_INSTALLED_RAM_CAPACITY


def test_an_unverified_capacity_source_is_not_planned_against() -> None:
    result = measure_budget_payloads([budget_payload(capacity_source="")])
    assert result.nodes == ()
    assert result.warnings[0].code == WARNING_UNVERIFIED_CAPACITY


def test_an_unmeasured_capacity_is_refused() -> None:
    payload = budget_payload(capacity_bytes=0, reserve_bytes=0, usable_bytes=0)
    result = measure_budget_payloads([payload])
    assert result.nodes == ()
    assert result.warnings[0].code == WARNING_UNREADABLE_CAPACITY


# ---------------------------------------------------------------------------
# Compute and links
# ---------------------------------------------------------------------------


def test_performance_profile_maps_one_to_one_into_the_compute_profile() -> None:
    measured = profile(decode_gib_per_s=2.0)
    node = MeasuredNode(
        node_id="m5",
        capacity_bytes=int(107.5 * GIB),
        admission_p05_bytes=int(96.75 * GIB),
        performance=measured,
    )
    compute = measure_nodes([node]).nodes[0].compute

    assert compute is not None
    assert compute.decode_weight_bytes_per_second == (
        measured.decode_weight_bytes_per_second
    )
    assert compute.prefill_weight_bytes_per_second == (
        measured.prefill_weight_bytes_per_second
    )
    assert compute.collective_latency_seconds == pytest.approx(0.0003)
    assert compute.collective_bandwidth_bytes_per_second == pytest.approx(6.6e9)
    assert compute.backend == "jaccl-ring"
    assert compute.measured_at == MEASURED_AT
    assert compute.samples == 5


def test_a_profile_measured_on_another_mac_is_not_attributed() -> None:
    node = MeasuredNode(
        node_id="m1",
        capacity_bytes=int(60 * GIB),
        admission_p05_bytes=int(54 * GIB),
        # The probe ran on the M5 and was pasted onto the M1 by mistake.
        performance=profile("m5"),
    )
    result = measure_nodes([node])

    assert result.nodes[0].compute is None
    assert WARNING_PERFORMANCE_NODE_MISMATCH in {item.code for item in result.warnings}


@pytest.mark.parametrize(
    ("kind", "transport"),
    [
        ("rdma", "jaccl"),
        ("thunderbolt", "jaccl-ring"),
        ("ethernet", "ring"),
        ("unknown", "ring"),
    ],
)
def test_link_bandwidth_becomes_link_hints(kind: str, transport: str) -> None:
    """The kind → backend rule stays where oMLX already keeps it."""

    link = LinkBandwidth(
        source_node_id="m5",
        peer_node_id="m1",
        bytes_per_second=6.6e9,
        source="measured",
        kind=kind,
    )
    node = MeasuredNode(
        node_id="m5",
        capacity_bytes=int(107.5 * GIB),
        admission_p05_bytes=int(96.75 * GIB),
        performance=profile(backend="ring"),
        link=link,
    )
    hints = measure_nodes([node]).nodes[0].link

    assert hints.transport == transport
    assert hints.bandwidth_gbps == pytest.approx(6.6e9 * 8 / 10**9)
    assert hints.latency_ms == pytest.approx(0.3)


def test_a_node_without_a_probe_gets_no_latency_hint() -> None:
    node = MeasuredNode(
        node_id="m1",
        capacity_bytes=int(60 * GIB),
        admission_p05_bytes=int(54 * GIB),
        transport_kinds=("thunderbolt",),
    )
    hints = measure_nodes([node]).nodes[0].link

    assert hints.transport == "jaccl-ring"
    assert hints.bandwidth_gbps is None
    assert hints.latency_ms is None


# ---------------------------------------------------------------------------
# Endpoint host and port
# ---------------------------------------------------------------------------


def test_endpoint_host_and_port_are_carried_from_the_budget_payload() -> None:
    """The plan must not have to guess where the elected Mac is reachable."""

    measured = measured_node_from_budget(
        budget_payload(endpoint_host="192.0.2.11", endpoint_port=9123)
    )
    facts = measure_nodes([measured], measured_at=MEASURED_AT).nodes[0]

    assert measured.endpoint_host == "192.0.2.11"
    assert measured.endpoint_port == 9123
    assert facts.endpoint_host == "192.0.2.11"


def test_an_unmeasured_port_stays_missing() -> None:
    """No port measurement is not 8099: only unreadable settings are."""

    measured = measured_node_from_budget(budget_payload())

    assert measured.endpoint_port is None
    assert measured.endpoint_host == ""


@pytest.mark.parametrize("value", [0, -1, 70000, "not-a-port", True, None])
def test_an_unusable_port_measurement_is_not_a_port(value: object) -> None:
    measured = measured_node_from_budget(budget_payload(endpoint_port=value))
    assert measured.endpoint_port is None


def test_the_port_is_the_macs_own_and_8099_only_when_settings_are_unreadable() -> None:
    from types import SimpleNamespace

    assert measured_endpoint_port(settings=SimpleNamespace(port=9123)) == 9123
    assert measured_endpoint_port(configured_port=9000) == 9000
    # Nothing readable: the documented cluster default, never 0.
    assert measured_endpoint_port() == DEFAULT_ENDPOINT_PORT
    assert measured_endpoint_port(settings=SimpleNamespace(port=0)) == (
        DEFAULT_ENDPOINT_PORT
    )
    assert measured_endpoint_port(settings=SimpleNamespace(port=70000)) == (
        DEFAULT_ENDPOINT_PORT
    )
    assert measured_endpoint_port(configured_port=0) == DEFAULT_ENDPOINT_PORT
    assert measured_endpoint_port() != 0


def test_loopback_is_not_an_endpoint_another_mac_can_dial() -> None:
    assert reachable_endpoint_host("127.0.0.1", "192.0.2.11") == "192.0.2.11"
    assert reachable_endpoint_host("localhost", "::1") == ""
    assert reachable_endpoint_host(None, "") == ""
    assert reachable_endpoint_host("10.0.0.5") == "10.0.0.5"


# ---------------------------------------------------------------------------
# The measurement layer and the election compose
# ---------------------------------------------------------------------------


def test_measured_facts_are_accepted_by_the_election() -> None:
    """Measured oMLX payloads elect the M5, exactly like hand-built facts do."""

    m5 = measured_node_from_budget(
        budget_payload(endpoint_host="192.0.2.11", performance=profile("m5").to_dict())
    )
    m1 = measured_node_from_budget(
        small_budget_payload(
            endpoint_host="192.0.2.14",
            performance=profile("m1", decode_gib_per_s=0.8, backend="ring").to_dict(),
        )
    )
    measured = measure_nodes([m1, m5], measured_at=MEASURED_AT)
    assert {node.node_id for node in measured.nodes} == {"m1", "m5"}

    # 40 GiB of layers plus 70 GiB of head/coordinator memory: the M1's 54 GiB
    # cannot hold the tail, the M5's 96.75 GiB can.
    result = elect(
        measured.nodes,
        synthetic_model_layout(total_weight_bytes=40 * GIB, layer_count=20),
        context_tokens=8192,
        rank0_extra_bytes=70 * GIB,
        origin_node="m1",
    )

    assert result.coordinator_node == "m5"
    assert result.rank_order == ("m5", "m1")
    assert [item.node_id for item in result.rejected] == ["m1"]
    assert result.endpoint is not None
    assert result.endpoint.host == "192.0.2.11"


# ---------------------------------------------------------------------------
# Real-model calibration (preferred over the synthetic probe)
# ---------------------------------------------------------------------------


def calibration_profile(
    *,
    decode_tok_s: float = 500.0,
    prefill_tok_s: float = 4000.0,
    prompt_tokens: int = 10,
):
    """A real ComputeProfile as ``calibrate_node`` produces it (fake timings)."""

    return compute_profile_from_timings(
        CompletionTimings(
            decode_tok_per_s=decode_tok_s,
            prefill_tok_per_s=prefill_tok_s,
            prompt_tokens=prompt_tokens,
        ),
        model_weight_bytes=1_000_000_000,
        measured_at=MEASURED_AT,
    )


def test_real_calibration_is_preferred_over_the_synthetic_probe() -> None:
    """Ranking rests on real completions when both measurements exist."""

    real = calibration_profile()
    node = MeasuredNode(
        node_id="m5",
        capacity_bytes=int(107.5 * GIB),
        admission_p05_bytes=int(96.75 * GIB),
        performance=profile("m5", decode_gib_per_s=2.0),
        calibration=real,
    )
    result = measure_nodes([node])
    compute = result.nodes[0].compute

    assert compute is not None
    # The real rates (500 / 400 GB/s), not the synthetic 2 GiB/s probe.
    assert compute.decode_weight_bytes_per_second == pytest.approx(500e9)
    assert compute.prefill_weight_bytes_per_second == pytest.approx(400e9)
    assert compute.measured_at == MEASURED_AT
    assert result.warnings == ()


def test_real_calibration_alone_measures_compute_without_a_probe() -> None:
    node = MeasuredNode(
        node_id="m5",
        capacity_bytes=int(107.5 * GIB),
        admission_p05_bytes=int(96.75 * GIB),
        calibration=calibration_profile(),
    )
    result = measure_nodes([node])

    assert result.nodes[0].compute is not None
    # A calibrated Mac is measured: no unmeasured-compute warning.
    assert WARNING_UNMEASURED_COMPUTE not in {item.code for item in result.warnings}


def test_calibration_does_not_change_the_admission_or_memory_facts() -> None:
    base = MeasuredNode(
        node_id="m5",
        capacity_bytes=int(107.5 * GIB),
        admission_p05_bytes=int(96.75 * GIB),
        admission_p50_bytes=int(98 * GIB),
        admission_p95_bytes=int(100 * GIB),
    )
    without = measure_nodes([base]).nodes[0]
    with_real = measure_nodes([with_calibration(base, calibration_profile())]).nodes[0]

    assert (with_real.capacity_gib, with_real.admission_p05_gib) == (
        without.capacity_gib,
        without.admission_p05_gib,
    )
    assert (with_real.admission_p50_gib, with_real.admission_p95_gib) == (
        without.admission_p50_gib,
        without.admission_p95_gib,
    )


def test_with_calibration_folds_a_profile_into_a_copy() -> None:
    base = MeasuredNode(
        node_id="m5",
        capacity_bytes=int(107.5 * GIB),
        performance=profile("m5"),
    )
    real = calibration_profile()
    folded = with_calibration(base, real)

    assert folded.calibration is real
    assert folded.node_id == base.node_id
    assert folded.performance is base.performance
    # The original is untouched (frozen), and None clears the calibration.
    assert base.calibration is None
    assert with_calibration(folded, None).calibration is None


def test_a_mismatched_probe_is_still_flagged_when_calibration_wins() -> None:
    node = MeasuredNode(
        node_id="m1",
        capacity_bytes=int(60 * GIB),
        admission_p05_bytes=int(54 * GIB),
        performance=profile("m5"),  # pasted from the wrong Mac
        calibration=calibration_profile(),
    )
    result = measure_nodes([node])

    assert result.nodes[0].compute is not None
    assert result.nodes[0].compute.decode_weight_bytes_per_second == pytest.approx(
        500e9
    )
    assert WARNING_PERFORMANCE_NODE_MISMATCH in {item.code for item in result.warnings}


def test_budget_payload_carries_a_real_calibration() -> None:
    real = calibration_profile()
    measured = measured_node_from_budget(
        budget_payload(calibration=real.to_dict())
    )
    assert measured.calibration == real
    assert measured.to_dict()["calibration"] == real.to_dict()

    compute = measure_nodes([measured]).nodes[0].compute
    assert compute is not None
    assert compute.decode_weight_bytes_per_second == pytest.approx(500e9)


@pytest.mark.parametrize(
    "payload", [{"decode_weight_bytes_per_second": "fast"}, {}, None, []]
)
def test_a_malformed_calibration_payload_is_not_a_measurement(payload) -> None:
    measured = measured_node_from_budget(budget_payload(calibration=payload))
    assert measured.calibration is None
    result = measure_nodes([measured])
    assert result.nodes[0].compute is None
    assert WARNING_UNMEASURED_COMPUTE in {item.code for item in result.warnings}


def test_the_synthetic_probe_still_works_without_calibration() -> None:
    """The fallback path stays intact for a Mac without a calibration."""

    node = MeasuredNode(
        node_id="m5",
        capacity_bytes=int(107.5 * GIB),
        admission_p05_bytes=int(96.75 * GIB),
        performance=profile("m5", decode_gib_per_s=2.0),
    )
    compute = measure_nodes([node]).nodes[0].compute
    assert compute is not None
    assert compute.decode_weight_bytes_per_second == pytest.approx(2.0 * GIB)
