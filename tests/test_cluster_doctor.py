# SPDX-License-Identifier: Apache-2.0
"""Tests for the Fabric Doctor's ordered checks and errno-mapped probe."""

import ipaddress
import subprocess
from types import SimpleNamespace

import pytest

from omlx.cluster.collective import CollectiveSmokeError
from omlx.cluster.doctor import (
    DOCTOR_CHECKS,
    ERRNO_DIAGNOSES,
    DoctorFinding,
    DoctorProbes,
    FabricProbeRefusedError,
    check_jaccl_probe,
    check_link_address_sanity,
    check_reachability,
    check_staleness_admin,
    check_subnet_collision,
    run_fabric_collective_probe,
    run_fabric_doctor,
)
from omlx.cluster.transport import (
    HostInterfaces,
    InterfaceAddress,
    LinkEndpoint,
    SharedLink,
)
from omlx.cluster.vpn import VPNProfile

HOSTS = ("127.0.0.1", "peer.local")


def _interfaces(
    host: str,
    addresses=(),
    rdma=("en3",),
    thunderbolt=("en3", "en4"),
) -> HostInterfaces:
    return HostInterfaces(
        host=host,
        addresses=tuple(
            InterfaceAddress(interface=iface, address=addr, prefix_length=prefix)
            for iface, addr, prefix in addresses
        ),
        rdma_interfaces=frozenset(rdma),
        thunderbolt_interfaces=frozenset(thunderbolt),
    )


def _healthy_interfaces() -> dict:
    return {
        HOSTS[0]: _interfaces(
            HOSTS[0],
            addresses=(
                ("en0", "192.168.1.10", 24),
                ("en3", "172.16.99.1", 24),
            ),
        ),
        HOSTS[1]: _interfaces(
            HOSTS[1],
            addresses=(
                ("en0", "192.168.1.11", 24),
                ("en3", "172.16.99.2", 24),
            ),
        ),
    }


def _healthy_link() -> SharedLink:
    return SharedLink(
        source=LinkEndpoint(host=HOSTS[0], interface="en3", address="172.16.99.1"),
        peer=LinkEndpoint(host=HOSTS[1], interface="en3", address="172.16.99.2"),
        kind="rdma",
    )


def _probes(**overrides) -> DoctorProbes:
    interfaces = overrides.pop("interfaces_map", _healthy_interfaces())
    defaults = dict(
        interfaces=lambda host: interfaces[host],
        vpn=lambda host, **_: VPNProfile(),
        hostile=lambda hosts, **_: (
            ipaddress.ip_network("192.168.1.0/24"),
            ipaddress.ip_network("172.16.99.0/24"),
        ),
        shared_link=lambda source, peer, **_: _healthy_link(),
        verify=lambda link: (True, "verified"),
        collective=lambda hosts, addresses, rdma_matrix, timeout=10.0: {
            "ok": True,
            "elapsed_seconds": 1.25,
            "bandwidth_gbps": None,
        },
        deployments=lambda: (),
        admin_port=lambda: (True, "the admin API is answering on port 9000"),
    )
    defaults.update(overrides)
    return DoctorProbes(**defaults)


def _by_id(report):
    return {finding.check_id: finding for finding in report.findings}


# --- full-run behavior ----------------------------------------------------


def test_all_green_run_verifies_fabric():
    report = run_fabric_doctor(HOSTS, probes=_probes())
    assert report.ok
    states = {f.check_id: f.state for f in report.findings}
    assert states == {
        "link_presence": "pass",
        "address_sanity": "pass",
        "subnet_collision": "pass",
        "route_pinning": "pass",
        "bound_connect": "pass",
        "jaccl_probe": "pass",
        "rdma_staleness": "pass",
        "admin_port": "pass",
    }
    assert report.verdict == (
        "Fabric verified — the two-rank collective handshake completed in 1.25s."
    )


def test_success_verdict_carries_bandwidth_when_measured():
    probes = _probes(
        collective=lambda *a, **k: {
            "ok": True,
            "elapsed_seconds": 0.8,
            "bandwidth_gbps": 68.2,
        }
    )
    report = run_fabric_doctor(HOSTS, probes=probes)
    assert report.verdict == "Fabric verified — 68 Gb/s measured across the link."


def test_stop_at_first_red_marks_later_checks_skipped():
    """A collision failure marks route, JACCL, and staleness checks skipped."""

    def hostile(hosts, **_):
        # A tunnel-routed /8 that swallows the fabric subnet: 10.0.0.0/8 is
        # routed but sits on no interface.
        return (ipaddress.ip_network("10.0.0.0/8"),)

    interfaces = {
        HOSTS[0]: _interfaces(HOSTS[0], addresses=(("en3", "10.0.1.1", 24),)),
        HOSTS[1]: _interfaces(HOSTS[1], addresses=(("en3", "10.0.1.2", 24),)),
    }
    calls = []
    probes = _probes(
        interfaces_map=interfaces,
        hostile=hostile,
        vpn=lambda host, **_: VPNProfile(
            present=True,
            client="warp",
            full_tunnel=True,
            utun_interfaces=("utun4",),
        ),
        collective=lambda *a, **k: calls.append("collective"),
        verify=lambda link: calls.append("verify"),
    )
    report = run_fabric_doctor(HOSTS, probes=probes)
    by_id = _by_id(report)
    assert by_id["subnet_collision"].state == "fail"
    for later in ("route_pinning", "bound_connect", "jaccl_probe",
                  "rdma_staleness", "admin_port"):
        assert by_id[later].state == "skipped"
        assert "subnet_collision failed first" in by_id[later].evidence
    assert calls == []  # later probes never ran
    assert report.verdict.startswith("Fabric Doctor stopped at subnet_collision:")
    assert not report.ok


def test_finding_order_matches_ladder_order():
    report = run_fabric_doctor(HOSTS, probes=_probes())
    expected = [
        finding_id
        for check in DOCTOR_CHECKS
        for finding_id in check.finding_ids
    ]
    assert [finding.check_id for finding in report.findings] == expected


def test_unreadable_host_fails_link_presence():
    def interfaces(host):
        raise RuntimeError("ssh timed out")

    probes = _probes()
    probes = DoctorProbes(
        interfaces=interfaces,
        vpn=probes.vpn,
        hostile=probes.hostile,
        shared_link=probes.shared_link,
        verify=probes.verify,
        collective=probes.collective,
        deployments=probes.deployments,
        admin_port=probes.admin_port,
    )
    report = run_fabric_doctor(HOSTS, probes=probes)
    by_id = _by_id(report)
    assert by_id["link_presence"].state == "fail"
    assert "could not read interface state" in by_id["link_presence"].evidence


def test_requires_exactly_two_hosts():
    with pytest.raises(ValueError):
        run_fabric_doctor(("only-one",), probes=_probes())


# --- check 1: link/address sanity ----------------------------------------


def test_check1_no_thunderbolt_interfaces_fails_presence():
    interfaces = {
        HOSTS[0]: _interfaces(HOSTS[0], rdma=(), thunderbolt=()),
        HOSTS[1]: _healthy_interfaces()[HOSTS[1]],
    }
    findings = check_link_address_sanity(HOSTS, interfaces)
    assert findings[0].check_id == "link_presence"
    assert findings[0].state == "fail"
    assert HOSTS[0] in findings[0].evidence
    assert findings[1].check_id == "address_sanity"
    assert findings[1].state == "skipped"


def test_check1_self_assigned_address_fails_with_readdress_fix():
    interfaces = {
        HOSTS[0]: _interfaces(
            HOSTS[0], addresses=(("en3", "169.254.12.7", 16),)
        ),
        HOSTS[1]: _healthy_interfaces()[HOSTS[1]],
    }
    # parse_interface_addresses filters 169.254 in production reads; the
    # check still guards against a probe that reports one.
    findings = check_link_address_sanity(HOSTS, interfaces)
    sanity = findings[1]
    assert sanity.check_id == "address_sanity"
    assert sanity.state == "fail"
    assert "169.254.12.7" in sanity.evidence
    assert "self-assigned" in sanity.diagnosis
    assert sanity.fix_action == {"kind": "readdress", "hosts": list(HOSTS)}


def test_check1_renumbered_interface_named_in_diagnosis():
    interfaces = {
        # RDMA device is en3 but the fabric address sits on en4 — the
        # renumbering shape from the en6→en4 incident.
        HOSTS[0]: _interfaces(
            HOSTS[0],
            addresses=(("en4", "172.16.99.1", 24),),
            rdma=("en3",),
            thunderbolt=("en3", "en4"),
        ),
        HOSTS[1]: _healthy_interfaces()[HOSTS[1]],
    }
    findings = check_link_address_sanity(HOSTS, interfaces)
    sanity = findings[1]
    assert sanity.state == "fail"
    assert "renumbered" in sanity.diagnosis
    assert sanity.fix_action["kind"] == "readdress"


def test_check1_passes_on_healthy_addresses():
    findings = check_link_address_sanity(HOSTS, _healthy_interfaces())
    assert [f.state for f in findings] == ["pass", "pass"]
    assert "172.16.99.1/24" in findings[1].evidence


# --- check 2: subnet collision -------------------------------------------


def test_check2_tunnel_routed_prefix_names_the_client():
    interfaces = {
        HOSTS[0]: _interfaces(HOSTS[0], addresses=(("en3", "10.0.1.1", 24),)),
        HOSTS[1]: _interfaces(HOSTS[1], addresses=(("en3", "10.0.1.2", 24),)),
    }
    findings = check_subnet_collision(
        HOSTS,
        interfaces,
        hostile=(ipaddress.ip_network("10.0.0.0/8"),),
        vpn_profiles={
            HOSTS[0]: VPNProfile(
                present=True,
                client="warp",
                full_tunnel=True,
                utun_interfaces=("utun4",),
            ),
            HOSTS[1]: VPNProfile(),
        },
    )
    finding = findings[0]
    assert finding.state == "fail"
    assert finding.evidence == "WARP routes 10.0.0.0/8 through utun4"
    assert finding.fix_action == {"kind": "move_subnet", "hosts": list(HOSTS)}
    assert "Move link addresses" in finding.remedy


def test_check2_lan_overlap_reported_without_vpn_language():
    interfaces = {
        HOSTS[0]: _interfaces(
            HOSTS[0],
            addresses=(
                ("en0", "192.168.1.10", 16),
                ("en3", "192.168.7.1", 24),
            ),
        ),
        HOSTS[1]: _interfaces(HOSTS[1], addresses=(("en3", "192.168.7.2", 24),)),
    }
    findings = check_subnet_collision(
        HOSTS,
        interfaces,
        hostile=(
            ipaddress.ip_network("192.168.0.0/16"),
            ipaddress.ip_network("192.168.7.0/24"),
        ),
        vpn_profiles={host: VPNProfile() for host in HOSTS},
    )
    finding = findings[0]
    assert finding.state == "fail"
    assert "in use on a LAN interface" in finding.evidence


def test_check2_passes_when_fabric_range_is_clean():
    findings = check_subnet_collision(
        HOSTS,
        _healthy_interfaces(),
        hostile=(
            ipaddress.ip_network("192.168.1.0/24"),
            ipaddress.ip_network("172.16.99.0/24"),  # the link's own range
        ),
        vpn_profiles={host: VPNProfile() for host in HOSTS},
    )
    assert findings[0].state == "pass"
    assert "172.16.99.0/24" in findings[0].evidence


def test_check2_no_fabric_addresses_is_a_pass_with_honest_evidence():
    interfaces = {
        HOSTS[0]: _interfaces(HOSTS[0], addresses=(("en0", "192.168.1.10", 24),)),
        HOSTS[1]: _interfaces(HOSTS[1], addresses=()),
    }
    findings = check_subnet_collision(
        HOSTS, interfaces, hostile=(), vpn_profiles={}
    )
    assert findings[0].state == "pass"
    assert "no fabric addresses" in findings[0].evidence


# --- check 3: routes + bound connect -------------------------------------


def test_check3_route_failure_splits_from_bound_connect():
    detail = (
        "127.0.0.1 cannot use en3 to reach 172.16.99.2: route uses utun4."
    )
    findings = check_reachability(_healthy_link(), (False, detail))
    route, bound = findings
    assert route.check_id == "route_pinning"
    assert route.state == "fail"
    assert route.evidence == detail
    assert route.fix_action == {"kind": "readdress"}
    assert bound.check_id == "bound_connect"
    assert bound.state == "skipped"


def test_check3_firewall_refusal_fails_bound_connect_only():
    detail = (
        "peer.local cannot accept connections on the Thunderbolt link (a "
        "firewall or VPN on that Mac applies to all interfaces): 127.0.0.1 "
        "reached 172.16.99.2 by ping but the bound TCP connection was refused."
    )
    findings = check_reachability(_healthy_link(), (False, detail))
    route, bound = findings
    assert route.state == "pass"
    assert bound.state == "fail"
    assert "firewall or VPN" in bound.diagnosis
    assert bound.fix_action == {"kind": "move_subnet"}


def test_check3_dead_peer_fails_bound_connect_without_firewall_blame():
    detail = (
        "127.0.0.1 routes 172.16.99.2 over en3, but peer.local did not "
        "answer on that address (no TCP, no ping)."
    )
    findings = check_reachability(_healthy_link(), (False, detail))
    assert findings[0].state == "pass"
    assert findings[1].state == "fail"
    assert "did not answer" in findings[1].diagnosis


def test_check3_missing_link_fails_route_with_link_reason():
    link = SharedLink(reason="peer.local has no routable IPv4 address")
    findings = check_reachability(link, None)
    assert findings[0].state == "fail"
    assert "no routable IPv4 address" in findings[0].evidence
    assert findings[1].state == "skipped"


def test_check3_pass_carries_both_directions_evidence():
    findings = check_reachability(_healthy_link(), (True, "ok"))
    assert [f.state for f in findings] == ["pass", "pass"]
    assert "172.16.99.1 ⇄ 172.16.99.2" in findings[0].evidence


# --- check 4: JACCL probe + errno table ----------------------------------


def _deployment(*ssh_hosts, deployment_id="dep-1", model="qwen3.8-27b"):
    return SimpleNamespace(
        deployment_id=deployment_id,
        model=model,
        hosts=tuple(SimpleNamespace(ssh=ssh) for ssh in ssh_hosts),
    )


def test_check4_refuses_beside_active_deployment_and_never_probes():
    calls = []
    finding, result = check_jaccl_probe(
        HOSTS,
        _healthy_link(),
        _healthy_interfaces(),
        collective=lambda *a, **k: calls.append("probe"),
        deployments=(_deployment("127.0.0.1", "peer.local"),),
    )
    assert calls == []  # the refusal blocks the probe call outright
    assert finding.state == "skipped"
    assert "dep-1 (qwen3.8-27b)" in finding.evidence
    assert "perturb" in finding.evidence
    assert result is None


def test_check4_refusal_matches_local_alias_hosts():
    # rank 0 is always 127.0.0.1; "localhost" in the Doctor's host list must
    # still match it.
    finding, _ = check_jaccl_probe(
        ("localhost", "other.local"),
        _healthy_link(),
        {},
        collective=lambda *a, **k: (_ for _ in ()).throw(AssertionError),
        deployments=(_deployment("127.0.0.1", "peer.local"),),
    )
    assert finding.state == "skipped"


def test_check4_unrelated_deployment_does_not_block():
    finding, result = check_jaccl_probe(
        ("a.local", "b.local"),
        _healthy_link(),
        {},
        collective=lambda *a, **k: {"ok": True, "elapsed_seconds": 0.5},
        deployments=(_deployment("c.local", "d.local"),),
    )
    assert finding.state == "pass"
    assert result == {"ok": True, "elapsed_seconds": 0.5}


def test_check4_errno_60_maps_to_firewall_diagnosis():
    def probe(*args, **kwargs):
        raise CollectiveSmokeError(
            "fabric probe launcher exited with code 1: [jaccl] connect "
            "failed (error: 60)"
        )

    finding, _ = check_jaccl_probe(
        HOSTS, _healthy_link(), {}, collective=probe, deployments=()
    )
    assert finding.state == "fail"
    assert "ETIMEDOUT" in finding.evidence
    assert "firewall or VPN" in finding.diagnosis
    assert finding.fix_action == {"kind": "move_subnet", "hosts": list(HOSTS)}


def test_check4_errno_61_is_launch_order_not_network():
    def probe(*args, **kwargs):
        raise CollectiveSmokeError("rank 1 gave up (error: 61)")

    finding, _ = check_jaccl_probe(
        HOSTS, _healthy_link(), {}, collective=probe, deployments=()
    )
    assert finding.state == "fail"
    assert "ECONNREFUSED" in finding.evidence
    assert "launch-order problem" in finding.diagnosis
    assert finding.fix_action is None


def test_check4_unknown_errno_keeps_raw_detail():
    def probe(*args, **kwargs):
        raise CollectiveSmokeError("collective handshake died (error: 54)")

    finding, _ = check_jaccl_probe(
        HOSTS, _healthy_link(), {}, collective=probe, deployments=()
    )
    assert finding.state == "fail"
    assert "error 54" in finding.evidence
    assert "collective handshake died (error: 54)" in finding.evidence
    assert "does not recognize" in finding.diagnosis


def test_check4_missing_runtime_degrades_to_skipped():
    def probe(*args, **kwargs):
        raise CollectiveSmokeError(
            "oMLX worker runtime is not installed on peer.local"
        )

    finding, _ = check_jaccl_probe(
        HOSTS, _healthy_link(), {}, collective=probe, deployments=()
    )
    assert finding.state == "skipped"
    assert "worker runtime missing" in finding.evidence


def test_check4_builds_rdma_matrix_from_link_interfaces():
    captured = {}

    def probe(hosts, addresses, rdma_matrix, timeout=10.0):
        captured.update(
            hosts=hosts, addresses=addresses, rdma_matrix=rdma_matrix
        )
        return {"ok": True, "elapsed_seconds": 0.4}

    link = SharedLink(
        source=LinkEndpoint(host=HOSTS[0], interface="en2", address="172.16.99.1"),
        peer=LinkEndpoint(host=HOSTS[1], interface="en3", address="172.16.99.2"),
        kind="rdma",
    )
    finding, _ = check_jaccl_probe(
        HOSTS, link, {}, collective=probe, deployments=()
    )
    assert finding.state == "pass"
    assert captured["addresses"] == ("172.16.99.1", "172.16.99.2")
    assert captured["rdma_matrix"] == (
        (None, "rdma_en2"),
        ("rdma_en3", None),
    )


# --- check 5: staleness + admin port -------------------------------------


def test_check5_devices_without_addresses_is_the_reboot_finding():
    interfaces = {
        HOSTS[0]: _interfaces(HOSTS[0], addresses=(("en0", "192.168.1.10", 24),)),
        HOSTS[1]: _healthy_interfaces()[HOSTS[1]],
    }
    findings = check_staleness_admin(HOSTS, interfaces, (True, "ok"))
    stale = findings[0]
    assert stale.check_id == "rdma_staleness"
    assert stale.state == "fail"
    assert "no fabric addresses" in stale.evidence
    assert "after a reboot" in stale.diagnosis
    assert stale.fix_action == {"kind": "readdress", "hosts": list(HOSTS)}


def test_check5_admin_port_failure_reported():
    findings = check_staleness_admin(
        HOSTS,
        _healthy_interfaces(),
        (False, "nothing is answering on admin port 9000"),
    )
    assert findings[0].state == "pass"
    admin = findings[1]
    assert admin.check_id == "admin_port"
    assert admin.state == "fail"
    assert "9000" in admin.evidence


def test_check5_all_green():
    findings = check_staleness_admin(
        HOSTS, _healthy_interfaces(), (True, "answering on port 9000")
    )
    assert [f.state for f in findings] == ["pass", "pass"]


# --- run_fabric_collective_probe ------------------------------------------


_MATRIX = ((None, "rdma_en3"), ("rdma_en2", None))


def _fake_runner(records=(0, 1), returncode=0, stderr=""):
    def runner(argv, *, timeout):
        stdout = "\n".join(
            '{"type": "collective_result", "rank": %d, "size": 2, "sum": 3}'
            % rank
            for rank in records
        )
        return subprocess.CompletedProcess(
            args=list(argv), returncode=returncode, stdout=stdout, stderr=stderr
        )

    return runner


def test_probe_refuses_while_deployment_is_registered():
    """The mandatory safety test: a registered deployment blocks the launch."""

    launched = []

    def runner(argv, *, timeout):
        launched.append(argv)
        raise AssertionError("the probe must not launch beside a deployment")

    with pytest.raises(FabricProbeRefusedError) as excinfo:
        run_fabric_collective_probe(
            HOSTS,
            ("172.16.99.1", "172.16.99.2"),
            _MATRIX,
            runner=runner,
            deployments=(_deployment("127.0.0.1", "peer.local"),),
        )
    assert launched == []
    assert "dep-1" in str(excinfo.value)
    assert "Stop the deployment" in str(excinfo.value)


def test_probe_consults_the_registry_when_no_deployments_passed(monkeypatch):
    from omlx.cluster import doctor as doctor_module

    monkeypatch.setattr(
        doctor_module,
        "_registered_deployments",
        lambda: (_deployment("127.0.0.1", "peer.local"),),
    )
    with pytest.raises(FabricProbeRefusedError):
        run_fabric_collective_probe(
            HOSTS,
            ("172.16.99.1", "172.16.99.2"),
            _MATRIX,
            runner=_fake_runner(),
        )


def test_probe_success_reports_hosts_and_elapsed():
    result = run_fabric_collective_probe(
        HOSTS,
        ("172.16.99.1", "172.16.99.2"),
        _MATRIX,
        runner=_fake_runner(),
        deployments=(),
    )
    assert result["ok"] is True
    assert result["backend"] == "jaccl"
    assert result["hosts"] == list(HOSTS)
    assert result["addresses"] == ["172.16.99.1", "172.16.99.2"]
    assert result["elapsed_seconds"] >= 0


def test_probe_writes_deployment_shaped_hostfile():
    captured = {}

    def runner(argv, *, timeout):
        hostfile = argv[argv.index("--hostfile") + 1]
        import json
        from pathlib import Path

        captured.update(json.loads(Path(hostfile).read_text()))
        return _fake_runner()(argv, timeout=timeout)

    run_fabric_collective_probe(
        HOSTS,
        ("172.16.99.1", "172.16.99.2"),
        _MATRIX,
        runner=runner,
        deployments=(),
    )
    assert captured["backend"] == "jaccl"
    assert captured["hosts"][0] == {
        "ssh": "127.0.0.1",
        "ips": ["172.16.99.1"],
        "rdma": [None, "rdma_en3"],
    }
    assert captured["hosts"][1]["rdma"] == ["rdma_en2", None]


def test_probe_launcher_failure_carries_stderr():
    with pytest.raises(CollectiveSmokeError) as excinfo:
        run_fabric_collective_probe(
            HOSTS,
            ("172.16.99.1", "172.16.99.2"),
            _MATRIX,
            runner=_fake_runner(returncode=1, stderr="jaccl error: 60"),
            deployments=(),
        )
    assert "error: 60" in str(excinfo.value)


def test_probe_rejects_bad_matrix_shapes():
    bad_matrices = [
        ((None, "rdma_en3"),),  # one row
        (("rdma_en3", None), ("rdma_en2", None)),  # non-null diagonal
        ((None, None), ("rdma_en2", None)),  # missing peer path
    ]
    for matrix in bad_matrices:
        with pytest.raises(ValueError):
            run_fabric_collective_probe(
                HOSTS,
                ("172.16.99.1", "172.16.99.2"),
                matrix,
                runner=_fake_runner(),
                deployments=(),
            )


def test_probe_rejects_invalid_addresses():
    with pytest.raises(ValueError):
        run_fabric_collective_probe(
            HOSTS,
            ("172.16.99.1", "not-an-ip"),
            _MATRIX,
            runner=_fake_runner(),
            deployments=(),
        )


# --- errno table shape ----------------------------------------------------


def test_errno_table_contents():
    assert ERRNO_DIAGNOSES[60][0] == "ETIMEDOUT"
    assert "firewall or VPN" in ERRNO_DIAGNOSES[60][1]
    assert ERRNO_DIAGNOSES[60][2]  # has a remedy
    assert ERRNO_DIAGNOSES[61][0] == "ECONNREFUSED"
    assert "retrying is correct" in ERRNO_DIAGNOSES[61][1]
    assert ERRNO_DIAGNOSES[61][2] is None


def test_finding_serialization_round_trip():
    finding = DoctorFinding(
        check_id="jaccl_probe",
        state="fail",
        evidence="e",
        diagnosis="d",
        remedy="r",
        fix_action={"kind": "move_subnet"},
    )
    assert finding.to_dict() == {
        "check_id": "jaccl_probe",
        "state": "fail",
        "evidence": "e",
        "diagnosis": "d",
        "remedy": "r",
        "fix_action": {"kind": "move_subnet"},
    }
