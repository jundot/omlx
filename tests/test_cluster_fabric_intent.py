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
    InterfaceAddress,
    LinkSetupError,
    LinkStatus,
    _authorized_ifconfig,
    _own_network,
    configure_link,
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
# netmask parameterization (see test_cluster_link_status.py for
# parse_thunderbolt_interfaces / hardware-port parsing, which stayed --
# the networksetup-service half that lived here was split into its own
# change: see _own_network below and #2875's review)
# --------------------------------------------------------------------------


def _capture_authorized(monkeypatch):
    commands = []
    monkeypatch.setattr(
        "omlx.cluster.transport._run_authorized",
        lambda host, shell_command: commands.append((host, shell_command)),
    )
    return commands


def test_ifconfig_netmask_follows_the_chosen_prefix(monkeypatch):
    commands = _capture_authorized(monkeypatch)

    _authorized_ifconfig("Studio.local", "en3", "172.16.99.2", prefix_length=29)

    assert commands == [
        (
            "Studio.local",
            "/sbin/ifconfig en3 inet 172.16.99.2 netmask 255.255.255.248 up",
        )
    ]


def test_own_network_prefers_the_real_probed_prefix_over_a_slash24_guess():
    interfaces = HostInterfaces(
        host="Studio.local",
        addresses=(
            InterfaceAddress(interface="en3", address="172.16.99.2", prefix_length=28),
        ),
    )
    assert _own_network("Studio.local", "172.16.99.2", interfaces) == (
        ipaddress.ip_network("172.16.99.0/28")
    )


def test_own_network_falls_back_to_a_slash24_guess_when_the_probe_has_no_match():
    # Probe succeeded but doesn't carry this exact address (transient state,
    # or a probe for a different host) -- still needs some candidate.
    interfaces = HostInterfaces(
        host="Studio.local",
        addresses=(
            InterfaceAddress(interface="en0", address="10.0.0.5", prefix_length=8),
        ),
    )
    assert _own_network("Studio.local", "172.16.99.2", interfaces) == (
        ipaddress.ip_network("172.16.99.0/24")
    )
    assert _own_network("Studio.local", "172.16.99.2", None) == (
        ipaddress.ip_network("172.16.99.0/24")
    )


# --------------------------------------------------------------------------
# configure_link's three-tier selection
# --------------------------------------------------------------------------


def _stub_link_setup(
    monkeypatch, *, current_ips, hostile=(), final_ready=True, interfaces=None
):
    """Everything configure_link touches except the decision under test.

    ``interfaces`` lets a test supply real ``HostInterfaces`` (with a real
    ``prefix_length``) for the collision-check own-network fix -- omitted,
    every host probes as bare/empty, matching every pre-existing test's
    assumed-/24 behavior unchanged.
    """

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
        lambda host: (interfaces or {}).get(host, HostInterfaces(host=host)),
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
        assert applied[0][2] == "10.90.99.1"
    finally:
        _reset_store()


def test_a_colliding_intent_falls_through_to_tier3_and_is_rewritten(
    monkeypatch, tmp_path
):
    try:
        store = configure_fabric_intent(tmp_path)
        store.record(
            subnet="10.90.99.0/24",
            hosts=HOSTS,
            chosen_by="auto",
            reason="collision_free_default",
            addressing="ifconfig",
        )
        # A utun now claims the recorded range: the WARP-incident shape.
        applied = _stub_link_setup(
            monkeypatch, current_ips={}, hostile=("10.90.99.0/24",)
        )

        assert configure_link(HOSTS).ready is True
        assert applied == [
            ("ifconfig", "127.0.0.1", "10.91.99.1", 24),
            ("ifconfig", "Studio.local", "10.91.99.2", 24),
        ]
        rewritten = store.current()
        assert rewritten.subnet == "10.91.99.0/24"
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


def test_tier1_keeps_a_non_slash24_fabric_instead_of_readdressing_it(
    monkeypatch, tmp_path
):
    """#2875 review: own_networks used to assume /24 unconditionally, so a
    fabric on any other real mask failed the exact-equality check against
    hostile_networks' own (correctly real-masked) report of that same
    address -- tier 1 mistook its own working link for a collision with
    itself and re-addressed it behind an admin prompt. Fixed by comparing
    at the real prefix (read from the same probe hostile_networks uses)
    and by overlap, not equality."""
    try:
        configure_fabric_intent(tmp_path)
        studio_interfaces = {
            "Studio.local": HostInterfaces(
                host="Studio.local",
                addresses=(
                    InterfaceAddress(
                        interface="en3", address="172.16.99.2", prefix_length=28
                    ),
                ),
            ),
        }
        applied = _stub_link_setup(
            monkeypatch,
            current_ips={"Studio.local": "172.16.99.2"},
            # hostile_networks' own real-masked report of this same address
            # -- exactly what a live scan sees for a non-/24 fabric.
            hostile=("172.16.99.0/28",),
            interfaces=studio_interfaces,
        )

        configure_link(HOSTS)

        # Tier 1 kept the existing 172.16.99.0/24 range Studio already sits
        # in (only the peer, which had no address yet, gets one) -- NOT a
        # tier-3 fresh pick, which the old equality bug forced here.
        assert applied == [("ifconfig", "127.0.0.1", "172.16.99.1", 24)]
    finally:
        _reset_store()


def test_own_network_exclusion_holds_when_a_tunnel_claims_the_exact_fabric_subnet(
    monkeypatch, tmp_path
):
    """A tunnel routing the exact numeric range the fabric's own address
    sits in is indistinguishable, from the network value alone, from that
    same address showing up in hostile_networks' scan as the fabric
    interface itself -- own-address exclusion must still treat it as self,
    not a fresh collision, or a fabric could never stick on any range a
    VPN also happens to route (a real, not contrived, overlap given how
    small the private-address space is)."""
    try:
        configure_fabric_intent(tmp_path)
        studio_interfaces = {
            "Studio.local": HostInterfaces(
                host="Studio.local",
                addresses=(
                    InterfaceAddress(
                        interface="en3", address="172.16.99.2", prefix_length=24
                    ),
                ),
            ),
        }
        applied = _stub_link_setup(
            monkeypatch,
            current_ips={"Studio.local": "172.16.99.2"},
            hostile=("172.16.99.0/24",),
            interfaces=studio_interfaces,
        )

        configure_link(HOSTS)

        # Tier 1 kept it (only the peer, address-less so far, gets one).
        assert applied == [("ifconfig", "127.0.0.1", "172.16.99.1", 24)]
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
    assert applied[0][2] == "10.90.99.1"


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
