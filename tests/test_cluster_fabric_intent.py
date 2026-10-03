# SPDX-License-Identifier: Apache-2.0
"""C2 — durable addressing: recorded fabric intent + service-owned assignment.

The intent store is the memory that makes a good address assignment survive
oMLX's own re-runs; ``configure_link``'s three-tier selection is what makes it
authoritative without ever overruling the collision check or the empirical
``assess_link`` verdict.
"""

from __future__ import annotations

import ipaddress
import json

import pytest

from omlx.cluster import fabric_intent as fabric_intent_module
from omlx.cluster.fabric_intent import (
    FabricIntentStore,
    configure_fabric_intent,
    get_fabric_intent,
)
from omlx.cluster.transport import (
    HostInterfaces,
    LinkSetupError,
    LinkStatus,
    _authorized_ifconfig,
    _authorized_networksetup,
    configure_link,
    parse_hardware_ports,
    parse_network_services,
)
from omlx.cluster.vpn import VPNProfile

HOSTS = ("127.0.0.1", "Studio.local")


def _reset_store():
    fabric_intent_module._configured_intent = None


# --------------------------------------------------------------------------
# FabricIntentStore
# --------------------------------------------------------------------------


def test_intent_round_trips_through_the_store_with_provenance(tmp_path):
    try:
        store = configure_fabric_intent(tmp_path)
        recorded = store.record(
            subnet="172.16.99.0/24",
            hosts=HOSTS,
            chosen_by="auto",
            reason="vpn_exclusion",
            addressing="ifconfig",
        )
        assert recorded.recorded_at > 0

        # A fresh store over the same base path reads the same record back.
        reread = FabricIntentStore(tmp_path).current()
        assert reread is not None
        assert reread.subnet == "172.16.99.0/24"
        assert reread.hosts == HOSTS
        assert reread.chosen_by == "auto"
        assert reread.reason == "vpn_exclusion"
        assert reread.addressing == "ifconfig"
        assert reread.recorded_at == recorded.recorded_at
        assert get_fabric_intent() is store
    finally:
        _reset_store()


def test_a_corrupt_record_fails_closed_and_recovers_on_the_next_write(tmp_path):
    path = tmp_path / "cluster" / "fabric-intent.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")

    store = FabricIntentStore(tmp_path)
    assert store.current() is None
    assert store.load_error is not None

    store.record(
        subnet="10.90.99.0/24",
        hosts=HOSTS,
        chosen_by="doctor",
        reason="collision_free_default",
        addressing="networksetup",
    )
    assert store.load_error is None
    assert FabricIntentStore(tmp_path).current().subnet == "10.90.99.0/24"


def test_an_unsupported_schema_or_bad_field_fails_closed(tmp_path):
    path = tmp_path / "cluster" / "fabric-intent.json"
    path.parent.mkdir(parents=True)
    for payload in (
        {"schema_version": 99, "intent": None},
        {
            "schema_version": 1,
            "intent": {
                "subnet": "not-a-subnet",
                "hosts": list(HOSTS),
                "chosen_by": "auto",
                "reason": "x",
                "recorded_at": 1.0,
                "addressing": "ifconfig",
            },
        },
        {
            "schema_version": 1,
            "intent": {
                "subnet": "172.16.99.0/24",
                "hosts": list(HOSTS),
                "chosen_by": "gremlin",
                "reason": "x",
                "recorded_at": 1.0,
                "addressing": "ifconfig",
            },
        },
    ):
        path.write_text(json.dumps(payload), encoding="utf-8")
        store = FabricIntentStore(tmp_path)
        assert store.current() is None
        assert store.load_error is not None


def test_recording_invalid_provenance_raises_and_writes_nothing(tmp_path):
    store = FabricIntentStore(tmp_path)
    with pytest.raises(ValueError, match="addressing"):
        store.record(
            subnet="172.16.99.0/24",
            hosts=HOSTS,
            chosen_by="auto",
            reason="x",
            addressing="sticky-tape",
        )
    assert store.current() is None
    assert not store.path.exists()


# --------------------------------------------------------------------------
# Service mapping parsers. Fixtures are the verbatim structure read off the
# two-Mac rig (coordinator Mac Studio + peer mini) on 2026-08-19: EXO renamed
# the per-port Thunderbolt services, and no "Thunderbolt Bridge" exists —
# -listallhardwareports never names a service, only the hardware port.
# --------------------------------------------------------------------------

HARDWARE_PORTS = """\
Hardware Port: Ethernet
Device: en0
Ethernet Address: d0:11:e5:18:1a:ba

Hardware Port: Wi-Fi
Device: en1
Ethernet Address: d0:11:e5:1d:b9:b8

Hardware Port: Thunderbolt 1
Device: en2
Ethernet Address: 36:b8:a3:15:10:80

Hardware Port: Thunderbolt 2
Device: en3
Ethernet Address: 36:b8:a3:15:10:84

Hardware Port: Thunderbolt 3
Device: en4
Ethernet Address: 36:b8:a3:15:10:88

VLAN Configurations
===================
"""

SERVICE_ORDER = """\
An asterisk (*) denotes that a network service is disabled.
(1) Ethernet
(Hardware Port: Ethernet, Device: en0)

(2) Wi-Fi
(Hardware Port: Wi-Fi, Device: en1)

(3) EXO Thunderbolt 1
(Hardware Port: Thunderbolt 1, Device: en2)

(4) EXO Thunderbolt 2
(Hardware Port: Thunderbolt 2, Device: en3)

(*) EXO Thunderbolt 3
(Hardware Port: Thunderbolt 3, Device: en4)
"""


def test_hardware_ports_map_devices_to_ports_never_services():
    ports = parse_hardware_ports(HARDWARE_PORTS)

    assert ports["en3"] == "Thunderbolt 2"
    assert ports["en0"] == "Ethernet"
    assert "EXO Thunderbolt 2" not in ports.values()


def test_service_order_maps_ports_to_the_renamed_services():
    services = parse_network_services(SERVICE_ORDER)

    assert services["Thunderbolt 2"] == "EXO Thunderbolt 2"
    assert services["Wi-Fi"] == "Wi-Fi"


def test_a_disabled_service_is_not_offered_for_addressing():
    services = parse_network_services(SERVICE_ORDER)

    assert "Thunderbolt 3" not in services


def test_a_bridge_member_without_a_service_maps_to_nothing():
    # The stock-macOS shape: TB ports are members of "Thunderbolt Bridge" and
    # own no service of their own — the ifconfig fallback case.
    services = parse_network_services(
        "An asterisk (*) denotes that a network service is disabled.\n"
        "(1) Wi-Fi\n(Hardware Port: Wi-Fi, Device: en0)\n\n"
        "(2) Thunderbolt Bridge\n(Hardware Port: Thunderbolt Bridge, Device: bridge0)\n"
    )
    ports = parse_hardware_ports(HARDWARE_PORTS)

    assert services.get(ports["en3"]) is None


# --------------------------------------------------------------------------
# _authorized_networksetup / netmask parameterization
# --------------------------------------------------------------------------


def _capture_authorized(monkeypatch):
    commands = []
    monkeypatch.setattr(
        "omlx.cluster.transport._run_authorized",
        lambda host, shell_command: commands.append((host, shell_command)),
    )
    return commands


def test_setmanual_is_one_fixed_validated_command(monkeypatch):
    commands = _capture_authorized(monkeypatch)

    _authorized_networksetup(
        "Studio.local", "EXO Thunderbolt 2", "172.16.99.2", "255.255.255.0"
    )

    assert commands == [
        (
            "Studio.local",
            "/usr/sbin/networksetup -setmanual 'EXO Thunderbolt 2' "
            "172.16.99.2 255.255.255.0",
        )
    ]


def test_creating_a_missing_service_is_one_combined_admin_prompt(monkeypatch):
    commands = _capture_authorized(monkeypatch)

    _authorized_networksetup(
        "Studio.local",
        "oMLX Thunderbolt 2",
        "172.16.99.2",
        "255.255.255.0",
        create_hardware_port="Thunderbolt 2",
    )

    assert len(commands) == 1
    command = commands[0][1]
    assert "-createnetworkservice 'oMLX Thunderbolt 2' 'Thunderbolt 2'" in command
    assert " && " in command
    assert command.endswith("172.16.99.2 255.255.255.0")


@pytest.mark.parametrize(
    "service,address,mask",
    [
        ("EXO; rm -rf /", "172.16.99.2", "255.255.255.0"),
        ("", "172.16.99.2", "255.255.255.0"),
        ("EXO Thunderbolt 2", "not-an-ip", "255.255.255.0"),
        ("EXO Thunderbolt 2", "172.16.99.2", "255.0.255.0"),
        ("EXO Thunderbolt 2", "172.16.99.2", "franken-mask"),
    ],
)
def test_networksetup_refuses_anything_that_is_not_a_clean_assignment(
    monkeypatch, service, address, mask
):
    commands = _capture_authorized(monkeypatch)

    with pytest.raises(LinkSetupError, match="Refusing"):
        _authorized_networksetup("Studio.local", service, address, mask)

    assert commands == []


def test_ifconfig_netmask_follows_the_chosen_prefix(monkeypatch):
    commands = _capture_authorized(monkeypatch)

    _authorized_ifconfig("Studio.local", "en3", "172.16.99.2", prefix_length=29)

    assert commands == [
        (
            "Studio.local",
            "/sbin/ifconfig en3 inet 172.16.99.2 netmask 255.255.255.248 up",
        )
    ]


# --------------------------------------------------------------------------
# configure_link's three-tier selection
# --------------------------------------------------------------------------


def _stub_link_setup(monkeypatch, *, current_ips, hostile=(), final_ready=True):
    """Everything configure_link touches except the decision under test."""

    states = iter(
        [
            LinkStatus(
                state="rdma_needs_setup",
                title="setup",
                detail="setup",
                backend="jaccl",
                ready=False,
                setup_available=True,
            ),
            LinkStatus(
                state="rdma_ready" if final_ready else "rdma_needs_setup",
                title="after",
                detail="after",
                backend="jaccl",
                ready=final_ready,
                setup_available=True,
            ),
        ]
    )
    monkeypatch.setattr(
        "omlx.cluster.transport.assess_link", lambda hosts: next(states)
    )
    monkeypatch.setattr(
        "omlx.cluster.transport._active_rdma_port", lambda host: "en3"
    )
    monkeypatch.setattr(
        "omlx.cluster.transport._interface_ip",
        lambda host, interface: current_ips.get(host),
    )
    monkeypatch.setattr(
        "omlx.cluster.transport.probe_host_interfaces",
        lambda host: HostInterfaces(host=host),
    )
    monkeypatch.setattr(
        "omlx.cluster.vpn.detect_vpn", lambda host, **_kwargs: VPNProfile()
    )
    monkeypatch.setattr(
        "omlx.cluster.vpn.hostile_networks",
        lambda hosts, **_kwargs: tuple(
            ipaddress.ip_network(net) for net in hostile
        ),
    )
    applied = []
    monkeypatch.setattr(
        "omlx.cluster.transport._authorized_ifconfig",
        lambda host, interface, address, **kw: applied.append(
            ("ifconfig", host, address, kw.get("prefix_length"))
        ),
    )
    return applied


def test_tier1_keeps_an_existing_valid_subnet_over_a_recorded_intent(
    monkeypatch, tmp_path
):
    try:
        store = configure_fabric_intent(tmp_path)
        store.record(
            subnet="172.16.105.0/24",
            hosts=HOSTS,
            chosen_by="user",
            reason="collision_free_default",
            addressing="ifconfig",
        )
        applied = _stub_link_setup(
            monkeypatch,
            current_ips={"Studio.local": "10.0.1.2"},
            hostile=("10.0.1.0/24",),  # the fabric's own subnet, not a collision
        )

        assert configure_link(HOSTS).ready is True
        assert applied == [("ifconfig", "127.0.0.1", "10.0.1.1", 24)]
        # Respecting the on-box choice does not rewrite the stored intent.
        assert store.current().subnet == "172.16.105.0/24"
    finally:
        _reset_store()


def test_tier2_reapplies_a_recorded_intent_verbatim_without_rewriting_it(
    monkeypatch, tmp_path
):
    try:
        store = configure_fabric_intent(tmp_path)
        original = store.record(
            subnet="172.16.105.0/24",
            hosts=HOSTS,
            chosen_by="doctor",
            reason="collision_free_default",
            addressing="ifconfig",
        )
        applied = _stub_link_setup(monkeypatch, current_ips={})

        assert configure_link(HOSTS).ready is True
        assert applied == [
            ("ifconfig", "127.0.0.1", "172.16.105.1", 24),
            ("ifconfig", "Studio.local", "172.16.105.2", 24),
        ]
        assert store.current() == original
    finally:
        _reset_store()


def test_an_intent_for_a_different_pair_never_steers_this_one(
    monkeypatch, tmp_path
):
    try:
        store = configure_fabric_intent(tmp_path)
        store.record(
            subnet="172.16.105.0/24",
            hosts=("mini.local", "studio.local"),
            chosen_by="auto",
            reason="collision_free_default",
            addressing="ifconfig",
        )
        applied = _stub_link_setup(monkeypatch, current_ips={})

        configure_link(HOSTS)

        # Tier 3 chose fresh (the first static candidate), not the record.
        assert applied[0][2] == "172.16.99.1"
    finally:
        _reset_store()


def test_a_colliding_intent_falls_through_to_tier3_and_is_rewritten(
    monkeypatch, tmp_path
):
    try:
        store = configure_fabric_intent(tmp_path)
        store.record(
            subnet="172.16.99.0/24",
            hosts=HOSTS,
            chosen_by="auto",
            reason="collision_free_default",
            addressing="ifconfig",
        )
        # A utun now claims the recorded range: the WARP-incident shape.
        applied = _stub_link_setup(
            monkeypatch, current_ips={}, hostile=("172.16.99.0/24",)
        )

        assert configure_link(HOSTS).ready is True
        assert applied == [
            ("ifconfig", "127.0.0.1", "172.16.100.1", 24),
            ("ifconfig", "Studio.local", "172.16.100.2", 24),
        ]
        rewritten = store.current()
        assert rewritten.subnet == "172.16.100.0/24"
        assert rewritten.chosen_by == "auto"
        assert rewritten.addressing == "ifconfig"
    finally:
        _reset_store()


def test_a_colliding_existing_subnet_is_readdressed_not_kept(monkeypatch, tmp_path):
    try:
        configure_fabric_intent(tmp_path)
        # Studio carries 10.0.1.2 while a utun routes all of 10/8 — tier 1
        # must reject the reuse and both endpoints move to the fresh range.
        applied = _stub_link_setup(
            monkeypatch,
            current_ips={"Studio.local": "10.0.1.2"},
            hostile=("10.0.0.0/8",),
        )

        configure_link(HOSTS)

        assert applied == [
            ("ifconfig", "127.0.0.1", "172.16.99.1", 24),
            ("ifconfig", "Studio.local", "172.16.99.2", 24),
        ]
    finally:
        _reset_store()


def test_intent_is_written_only_after_assess_link_reports_ready(
    monkeypatch, tmp_path
):
    try:
        store = configure_fabric_intent(tmp_path)
        _stub_link_setup(monkeypatch, current_ips={}, final_ready=False)

        with pytest.raises(LinkSetupError, match="did.*not become routable"):
            configure_link(HOSTS)

        assert store.current() is None
    finally:
        _reset_store()


def test_an_unconfigured_store_never_blocks_link_setup(monkeypatch):
    _reset_store()
    applied = _stub_link_setup(monkeypatch, current_ips={})

    assert configure_link(HOSTS).ready is True
    assert applied[0][2] == "172.16.99.1"


def test_service_addressing_is_used_and_recorded_when_asked_and_mapped(
    monkeypatch, tmp_path
):
    try:
        store = configure_fabric_intent(tmp_path)
        applied = _stub_link_setup(monkeypatch, current_ips={})
        monkeypatch.setattr(
            "omlx.cluster.transport._interface_service",
            lambda host, interface: "EXO Thunderbolt 2",
        )
        service_calls = []
        monkeypatch.setattr(
            "omlx.cluster.transport._authorized_networksetup",
            lambda host, service, address, mask, **_kw: service_calls.append(
                (host, service, address, mask)
            ),
        )

        configure_link(HOSTS, prefer_service_addressing=True)

        assert applied == []  # never fell back to ifconfig
        assert service_calls == [
            ("127.0.0.1", "EXO Thunderbolt 2", "172.16.99.1", "255.255.255.0"),
            ("Studio.local", "EXO Thunderbolt 2", "172.16.99.2", "255.255.255.0"),
        ]
        assert store.current().addressing == "networksetup"
    finally:
        _reset_store()


def test_no_service_mapping_falls_back_to_ifconfig_and_says_so(
    monkeypatch, tmp_path
):
    try:
        store = configure_fabric_intent(tmp_path)
        applied = _stub_link_setup(monkeypatch, current_ips={})
        monkeypatch.setattr(
            "omlx.cluster.transport._interface_service",
            lambda host, interface: None,
        )

        configure_link(HOSTS, prefer_service_addressing=True)

        assert [entry[0] for entry in applied] == ["ifconfig", "ifconfig"]
        # The watchdog must know this link drifts on reboot.
        assert store.current().addressing == "ifconfig"
    finally:
        _reset_store()


# --- C5: detect_drift — live addressing versus the recorded intent ----------


def _intent(addressing: str = "networksetup") -> fabric_intent_module.FabricIntent:
    return fabric_intent_module.FabricIntent(
        subnet="172.16.99.0/24",
        hosts=HOSTS,
        chosen_by="auto",
        reason="collision_free_default",
        recorded_at=1_700_000_000.0,
        addressing=addressing,
    )


def _live(*pairs: tuple[str, str]) -> tuple[HostInterfaces, ...]:
    from omlx.cluster.transport import InterfaceAddress

    return (
        HostInterfaces(
            host="studio.local",
            addresses=tuple(
                InterfaceAddress(interface=iface, address=addr, prefix_length=24)
                for iface, addr in pairs
            ),
            rdma_interfaces=frozenset({"en1"}),
            thunderbolt_interfaces=frozenset({"en1"}),
        ),
    )


def test_detect_drift_is_silent_while_live_matches_intent():
    finding = fabric_intent_module.detect_drift(
        _intent(), _live(("en1", "172.16.99.2"))
    )
    assert finding is None


def test_detect_drift_ignores_non_fabric_interfaces_when_matching():
    # A Wi-Fi address inside the subnet must not satisfy the fabric intent.
    finding = fabric_intent_module.detect_drift(
        _intent(), _live(("en0", "172.16.99.2"))
    )
    assert finding is not None
    assert finding.kind == "address_lost"


def test_detect_drift_reports_a_lost_address_after_reboot():
    finding = fabric_intent_module.detect_drift(_intent("ifconfig"), _live())
    assert finding is not None
    assert finding.kind == "address_lost"
    assert finding.live == ""
    assert finding.expected == "172.16.99.0/24"
    # An ifconfig-recorded link may not be silently re-addressed: the WARN
    # invites a consented Doctor re-address instead.
    assert finding.auto_restore is False
    assert "ifconfig" in finding.incident
    assert "Fabric Doctor" in finding.incident


def test_detect_drift_lost_networksetup_address_is_auto_restorable():
    finding = fabric_intent_module.detect_drift(_intent("networksetup"), _live())
    assert finding is not None
    assert finding.kind == "address_lost"
    # The service configuration persists; re-asserting it needs no new
    # consent, so the caller restores silently — no incident copy.
    assert finding.auto_restore is True
    assert finding.incident == ""


def test_detect_drift_reports_a_changed_address():
    finding = fabric_intent_module.detect_drift(
        _intent("ifconfig"), _live(("en1", "169.254.7.7"))
    )
    assert finding is not None
    assert finding.kind == "address_changed"
    assert finding.live == "en1 169.254.7.7"
    assert finding.expected == "172.16.99.0/24"
    assert finding.auto_restore is False
    assert "Fabric Doctor" in finding.incident


def test_detect_drift_reports_a_new_collision_with_the_design_copy():
    vpn_range = ipaddress.ip_network("172.16.0.0/12")

    finding = fabric_intent_module.detect_drift(
        _intent("networksetup"),
        _live(("en1", "172.16.99.2")),
        collides=lambda candidate: candidate.overlaps(vpn_range),
    )

    assert finding is not None
    assert finding.kind == "intent_collides"
    assert finding.expected == "172.16.99.0/24"
    assert "en1 172.16.99.2" in finding.live
    # A collided intent must never be silently re-applied, even for a
    # networksetup-recorded link.
    assert finding.auto_restore is False
    assert finding.incident == (
        "The link's saved addresses now collide with a new VPN range — "
        "Fabric Doctor needs to pick new ones."
    )


def test_detect_drift_collision_outranks_a_lost_address():
    finding = fabric_intent_module.detect_drift(
        _intent("networksetup"),
        _live(),
        collides=lambda candidate: True,
    )
    assert finding is not None
    assert finding.kind == "intent_collides"
    assert finding.auto_restore is False


def test_detect_drift_healthy_link_does_not_collide_with_itself():
    """The caller excludes the intent's own subnet, configure_link-style."""

    intent_net = ipaddress.ip_network("172.16.99.0/24")
    # hostile_networks always contains the link's own interface subnet; the
    # routes caller filters it before building the collision check.
    hostile = (intent_net, ipaddress.ip_network("192.168.1.0/24"))
    collision_set = tuple(
        net for net in hostile if not net.subnet_of(intent_net)
    )

    finding = fabric_intent_module.detect_drift(
        _intent(),
        _live(("en1", "172.16.99.2")),
        collides=lambda candidate: any(
            candidate.overlaps(net) for net in collision_set
        ),
    )
    assert finding is None
