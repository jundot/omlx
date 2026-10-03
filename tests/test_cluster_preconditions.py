# SPDX-License-Identifier: Apache-2.0
"""B4: the precondition-row composer, pure over already-collected evidence.

Table-driven: each single failing input flips exactly its own row's state
and the cluster-level ``ready`` conjunction, while every other row stays
passing. No SSH, no Macs — that is the whole point of the composer's
purity.
"""

from types import SimpleNamespace

import pytest

from omlx.cluster.preconditions import ROW_IDS, readiness_rows

GIB = 1024**3


def _good_evidence() -> dict:
    """Evidence for a healthy two-Mac cluster with a staged 17 GiB model."""

    return {
        "peers": [
            {
                "host": "aphoenix@mbp.local",
                "reachable": True,
                "detail": "",
            }
        ],
        "fabric_required": True,
        "fabric_ok": True,
        "shared_link": SimpleNamespace(ok=True, kind="rdma"),
        "fabric_detail": "reachable · 172.16.99.1 ⇄ 172.16.99.2 · TB5",
        "staging": {
            "ready": True,
            "nodes": [
                {"node_id": "studio", "ready": True, "missing_bytes": 0},
                {"node_id": "mbp", "ready": True, "missing_bytes": 0},
            ],
        },
        "budgets": [
            {
                "node_id": "studio",
                "role": "headless",
                "capacity_bytes": 76 * GIB,
                "reserve_bytes": 6 * GIB,
                "usable_bytes": 70 * GIB,
                "breakdown": {"binding": "role_reserve"},
            },
            {
                "node_id": "mbp",
                "role": "workstation",
                "capacity_bytes": 48 * GIB,
                "reserve_bytes": 16 * GIB,
                "usable_bytes": 32 * GIB,
                "breakdown": {"binding": "role_reserve"},
            },
        ],
        "required_bytes": 17 * GIB,
        "strategies": {
            "tensor": {"supported": True, "reason": ""},
            "pipeline": {"supported": True, "reason": ""},
        },
        "ages": {"ssh": 40.0, "fabric": 3.0, "staging": 5.0, "budget": 2.0, "strategy": 2.0},
    }


def test_all_good_evidence_is_ready_with_five_passing_rows():
    report = readiness_rows(**_good_evidence())

    assert report.ready is True
    assert tuple(row.id for row in report.rows) == ROW_IDS
    assert all(row.state == "pass" for row in report.rows)


FAILING_INPUTS = {
    "ssh": {
        "peers": [
            {
                "host": "aphoenix@mbp.local",
                "reachable": False,
                "detail": "connection refused",
            }
        ],
    },
    "fabric": {
        "fabric_ok": False,
        "shared_link": SimpleNamespace(ok=False, kind="none"),
        "fabric_detail": "no shared subnet between the Macs",
    },
    "staging": {
        "staging": {
            "ready": False,
            "nodes": [
                {"node_id": "studio", "ready": True, "missing_bytes": 0},
                {
                    "node_id": "mbp",
                    "ready": False,
                    "missing_bytes": 9 * GIB,
                    "missing_sidecar_bytes": 0,
                },
            ],
            "total_missing_bytes": 9 * GIB,
        },
    },
    "budget": {"required_bytes": 200 * GIB},
    "strategy": {
        "strategies": {
            "tensor": {
                "supported": False,
                "reason": "No split divides 2 Macs across the head groups.",
            },
            "pipeline": {
                "supported": False,
                "reason": "The architecture has no pipeline forward path.",
            },
        },
    },
}


@pytest.mark.parametrize("failing_id", sorted(FAILING_INPUTS))
def test_each_failing_input_flips_exactly_its_own_row(failing_id):
    report = readiness_rows(**{**_good_evidence(), **FAILING_INPUTS[failing_id]})

    assert report.ready is False
    states = {row.id: row.state for row in report.rows}
    assert states[failing_id] == "fail"
    for row_id, state in states.items():
        if row_id != failing_id:
            assert state == "pass", f"{row_id} should stay passing"


def test_evidence_strings_carry_their_ages():
    report = readiness_rows(**_good_evidence())

    by_id = {row.id: row for row in report.rows}
    assert by_id["ssh"].evidence_age_s == 40.0
    assert "40 s ago" in by_id["ssh"].evidence
    for row in report.rows:
        assert "s ago" in row.evidence
        assert row.evidence_age_s >= 0.0


def test_failing_budget_row_names_the_role_and_offers_the_role_editor():
    report = readiness_rows(**{**_good_evidence(), **FAILING_INPUTS["budget"]})

    budget = {row.id: row for row in report.rows}["budget"]
    assert budget.state == "fail"
    # B5 tie-in: the row names the role whose reserve binds, from the
    # ``binding``/``role`` fields, and the fix targets that node.
    assert "workstation" in budget.evidence
    assert "role_reserve" in budget.evidence
    assert budget.fix == {"kind": "role_editor", "node_id": "mbp"}


def test_unreachable_peer_row_carries_the_ssh_detail_and_reverify_fix():
    report = readiness_rows(**{**_good_evidence(), **FAILING_INPUTS["ssh"]})

    ssh = {row.id: row for row in report.rows}["ssh"]
    assert "connection refused" in ssh.evidence
    assert "aphoenix@mbp.local" in ssh.evidence
    assert ssh.fix == {"kind": "reverify"}


def test_ethernet_only_fabric_is_startable_but_warns():
    evidence = _good_evidence()
    evidence["shared_link"] = SimpleNamespace(ok=True, kind="ethernet")
    evidence["fabric_detail"] = "reachable · 10.0.1.4 ⇄ 10.0.1.7 · ethernet"
    report = readiness_rows(**evidence)

    fabric = {row.id: row for row in report.rows}["fabric"]
    assert fabric.state == "warn"
    assert "TCP ring" in fabric.evidence
    # A verified Ethernet ring satisfies the conjunction: Start works.
    assert report.ready is True


def test_no_model_selected_blocks_ready_without_claiming_breakage():
    evidence = _good_evidence()
    evidence["staging"] = None
    evidence["required_bytes"] = 0
    evidence["strategies"] = None
    report = readiness_rows(**evidence)

    assert report.ready is False
    states = {row.id: row.state for row in report.rows}
    assert states["staging"] == "warn"
    assert states["budget"] == "warn"
    assert states["strategy"] == "warn"
    assert "fail" not in states.values()


def test_single_mac_needs_no_fabric_and_no_peers():
    report = readiness_rows(
        **{
            **_good_evidence(),
            "peers": [],
            "fabric_required": False,
            "fabric_ok": None,
            "shared_link": None,
            "fabric_detail": "",
        }
    )

    by_id = {row.id: row for row in report.rows}
    assert by_id["ssh"].state == "pass"
    assert by_id["fabric"].state == "pass"
    assert "no fabric link needed" in by_id["fabric"].evidence
