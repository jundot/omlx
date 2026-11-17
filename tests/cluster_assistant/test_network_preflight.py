# SPDX-License-Identifier: Apache-2.0
"""Read-only network preflight: evidence in, commands out, nothing executed."""

from __future__ import annotations

from omlx.cluster.assistant.network_preflight import (
    WARNING_EXISTING_BRIDGE,
    WARNING_NO_THUNDERBOLT,
    WARNING_RING,
    WARNING_SELF_ASSIGNED,
    build_network_preflight,
)
from omlx.cluster.transport import HostInterfaces, InterfaceAddress

URL = "/admin/api/cluster/assistant/network-preflight"


def facts(
    host: str,
    *,
    thunderbolt: set[str] | None = None,
    self_assigned: set[str] | None = None,
) -> HostInterfaces:
    return HostInterfaces(
        host=host,
        addresses=(
            InterfaceAddress("bridge0", "10.10.0.1", 24),
        ),
        thunderbolt_interfaces=frozenset(
            {"en5", "en6"} if thunderbolt is None else thunderbolt
        ),
        self_assigned_interfaces=frozenset(
            set() if self_assigned is None else self_assigned
        ),
        media_speeds={"en5": 40000},
    )


def test_the_preflight_returns_exact_commands_and_never_executes() -> None:
    calls: list[str] = []

    def probe(host: str) -> HostInterfaces:
        calls.append(host)
        return facts(host, self_assigned={"en6"})

    result = build_network_preflight(("127.0.0.1",), probe=probe)

    assert calls == ["127.0.0.1"]
    assert result["executed"] is False
    assert result["confirmation_required"] is True
    assert result["hosts"][0]["thunderbolt_interfaces"] == ["en5", "en6"]
    assert "sudo ifconfig bridge0 create" in result["commands"]
    # More than one Thunderbolt port: the preflight never adds all of them.
    assert not any(
        item.startswith("sudo ifconfig bridge0 addm") and item.count("addm") > 1
        for item in result["commands"]
    )
    assert any("choose one port" in item for item in result["commands"])
    assert any(item.startswith(WARNING_SELF_ASSIGNED) for item in result["warnings"])


def test_an_existing_bridge_is_never_recreated_or_added_to_itself() -> None:
    result = build_network_preflight(
        ("127.0.0.1",),
        probe=lambda host: facts(host, thunderbolt={"bridge0", "en5", "en6"}),
    )

    assert result["hosts"][0]["bridges"] == ["bridge0"]
    assert result["hosts"][0]["thunderbolt_interfaces"] == ["en5", "en6"]
    assert any(item.startswith(WARNING_EXISTING_BRIDGE) for item in result["warnings"])
    joined = "\n".join(result["commands"])
    assert "bridge0 create" not in joined
    assert "addm bridge0" not in joined
    assert "add only the port" in joined


def test_a_mac_without_thunderbolt_gets_advice_not_a_fake_command() -> None:
    result = build_network_preflight(
        ("127.0.0.1",), probe=lambda host: facts(host, thunderbolt=set())
    )

    assert any(item.startswith(WARNING_NO_THUNDERBOLT) for item in result["warnings"])
    assert not any("addm" in item for item in result["commands"])
    assert result["executed"] is False


def test_more_than_two_macs_warns_about_the_ring_and_stp() -> None:
    hosts = ("a", "b", "c")

    result = build_network_preflight(
        hosts, probe=lambda host: facts(host)
    )

    assert any(item.startswith(WARNING_RING) for item in result["warnings"])
    assert result["executed"] is False


def test_a_failed_read_is_a_warning_not_a_500() -> None:
    def probe(host: str) -> HostInterfaces:
        raise RuntimeError("ifconfig unavailable")

    result = build_network_preflight(("127.0.0.1",), probe=probe)

    assert result["hosts"] == []
    assert result["warnings"]
    assert result["executed"] is False


def test_the_route_is_read_only_and_returns_the_plan(api) -> None:
    client, _, _ = api()

    response = client.get(
        URL,
    )

    # The real probe may fail in this headless test; the route still answers
    # with an app-only plan and executed=false. It must never 500.
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["confirmation_required"] is True
    assert body["executed"] is False
    assert isinstance(body["commands"], list)
