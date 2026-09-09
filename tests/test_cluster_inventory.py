# SPDX-License-Identifier: Apache-2.0
"""Cluster inventory file: load/save/add/list/remove."""

from __future__ import annotations

import pytest

from omlx.cluster.inventory import (
    ClusterInventory,
    InventoryHost,
    add_host,
    load_inventory,
    remove_host,
    save_inventory,
)


def test_inventory_host_defaults_to_current_os_user(monkeypatch):
    from omlx.cluster.ssh_identity import default_ssh_user

    expected = default_ssh_user()
    assert expected, "default_ssh_user() must resolve to a non-empty name"
    host = InventoryHost(name="somehost.local")
    assert host.user == expected
    assert host.port == 22


def test_add_host_uses_current_os_user_when_omitted(monkeypatch):
    from omlx.cluster.ssh_identity import default_ssh_user

    inventory = add_host(ClusterInventory(), "fresh.local")
    assert inventory.get("fresh.local").user == default_ssh_user()
    assert inventory.get("fresh.local").port == 22


def test_from_dict_fills_missing_user_with_current_os_user(monkeypatch):
    from omlx.cluster.ssh_identity import default_ssh_user

    loaded = ClusterInventory.from_dict(
        {"hosts": [{"name": "peer.local", "port": 22}]}
    )
    assert loaded.get("peer.local").user == default_ssh_user()


def test_inventory_round_trips_through_yaml(tmp_path):
    path = tmp_path / "inventory.yaml"
    hosts = [
        InventoryHost(
            name="node-a.local",
            user="operator",
            port=22,
            group="macs",
            provisioned=True,
            discovered_from="bonjour",
        ),
        InventoryHost(
            name="192.168.1.5",
            user="operator",
            port=22,
            group="macs",
            provisioned=False,
            discovered_from="sweep",
        ),
    ]
    save_inventory(ClusterInventory(hosts=hosts), path)

    loaded = load_inventory(path)
    assert [h.name for h in loaded.hosts] == [
        "node-a.local",
        "192.168.1.5",
    ]
    node_a = loaded.get("node-a.local")
    assert node_a.user == "operator"
    assert node_a.group == "macs"
    assert node_a.provisioned is True
    assert node_a.discovered_from == "bonjour"


def test_load_missing_inventory_returns_empty():
    inventory = load_inventory("/nonexistent/inventory.yaml")
    assert inventory.hosts == []


def test_add_host_updates_in_place_and_keeps_existing(tmp_path):
    path = tmp_path / "inventory.yaml"
    inventory = load_inventory(path)
    inventory = add_host(
        inventory,
        "first.local",
        user="operator",
        group="all",
        discovered_from="manual",
    )
    assert [h.name for h in inventory.hosts] == ["first.local"]

    inventory = add_host(
        inventory,
        "first.local",
        user="root",
        group="all",
        discovered_from="manual",
    )
    assert len(inventory.hosts) == 1
    assert inventory.get("first.local").user == "root"

    inventory = add_host(
        inventory,
        "second.local",
        user="operator",
        group="all",
        discovered_from="manual",
    )
    assert len(inventory.hosts) == 2


def test_remove_host_returns_false_for_unknown(tmp_path):
    inventory = load_inventory(tmp_path / "inventory.yaml")
    assert remove_host(inventory, "missing.local") is False


def test_remove_host_drops_existing_entry(tmp_path):
    path = tmp_path / "inventory.yaml"
    inventory = add_host(
        load_inventory(path),
        "dropme.local",
        user="operator",
        group="all",
        discovered_from="manual",
    )
    assert remove_host(inventory, "dropme.local") is True
    assert inventory.get("dropme.local") is None


def test_inventory_hosts_in_group_filters_by_group(tmp_path):
    inventory = load_inventory(tmp_path / "inventory.yaml")
    inventory = add_host(
        inventory,
        "a.local", user="operator", group="macs", discovered_from="manual"
    )
    inventory = add_host(
        inventory,
        "b.local", user="operator", group="linux", discovered_from="manual"
    )
    assert [h.name for h in inventory.hosts_in_group("macs")] == ["a.local"]
    assert [h.name for h in inventory.hosts_in_group("linux")] == ["b.local"]


def test_save_inventory_writes_private_permissions(tmp_path):
    path = tmp_path / "inventory.yaml"
    save_inventory(ClusterInventory(hosts=[]), path)
    assert path.exists()
    assert (path.stat().st_mode & 0o077) == 0


def test_add_host_rejects_empty_or_invalid_name(tmp_path):
    inventory = load_inventory(tmp_path / "inventory.yaml")
    with pytest.raises(ValueError):
        add_host(inventory, "", user="operator", group="all", discovered_from="manual")
    with pytest.raises(ValueError):
        add_host(
            inventory,
            "-oProxyCommand=evil",
            user="operator",
            group="all",
            discovered_from="manual",
        )


def test_add_host_can_mark_host_as_provisioned(tmp_path):
    inventory = load_inventory(tmp_path / "inventory.yaml")
    inventory = add_host(
        inventory,
        "node.local",
        user="operator",
        group="all",
        discovered_from="manual",
        provisioned=True,
    )
    assert inventory.get("node.local").provisioned is True

    inventory = add_host(
        inventory,
        "node.local",
        user="operator",
        group="all",
        discovered_from="manual",
        provisioned=False,
    )
    assert inventory.get("node.local").provisioned is False
