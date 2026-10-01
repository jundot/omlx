# SPDX-License-Identifier: Apache-2.0
"""Cluster host inventory backed by a private YAML file."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .deployment import validate_ssh_target
from .ssh_identity import _SSH_PORT, default_ssh_user

_INVENTORY_VERSION = 1
_DEFAULT_INVENTORY = Path.home() / ".omlx" / "cluster" / "inventory.yaml"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass
class InventoryHost:
    """A single cluster host and how to reach it over SSH."""

    name: str
    user: str = field(default_factory=default_ssh_user)
    port: int = _SSH_PORT
    group: str = "all"
    provisioned: bool = False
    discovered_from: str = "manual"


@dataclass
class ClusterInventory:
    """Ordered host list persisted as ``~/.omlx/cluster/inventory.yaml``."""

    hosts: list[InventoryHost] = field(default_factory=list)

    def get(self, name: str) -> InventoryHost | None:
        for host in self.hosts:
            if host.name == name:
                return host
        return None

    def hosts_in_group(self, group: str) -> list[InventoryHost]:
        return [host for host in self.hosts if host.group == group]

    def to_dict(self) -> dict:
        return {
            "version": _INVENTORY_VERSION,
            "hosts": [asdict(host) for host in self.hosts],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> ClusterInventory:
        raw_hosts = payload.get("hosts", [])
        hosts = []
        for raw in raw_hosts:
            hosts.append(
                InventoryHost(
                    name=str(raw.get("name", "")),
                    user=str(raw.get("user", default_ssh_user())),
                    port=int(raw.get("port", _SSH_PORT)),
                    group=str(raw.get("group", "all")),
                    provisioned=bool(raw.get("provisioned", False)),
                    discovered_from=str(raw.get("discovered_from", "manual")),
                )
            )
        return cls(hosts=hosts)


def load_inventory(path: str | Path | None = None) -> ClusterInventory:
    """Load the inventory file; a missing or malformed file yields an empty list."""

    target = Path(path) if path is not None else _DEFAULT_INVENTORY
    if not target.exists():
        return ClusterInventory()
    try:
        payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return ClusterInventory()
    if not isinstance(payload, dict):
        return ClusterInventory()
    return ClusterInventory.from_dict(payload)


def save_inventory(inventory: ClusterInventory, path: str | Path | None = None) -> Path:
    """Write the inventory with owner-only permissions."""

    target = Path(path) if path is not None else _DEFAULT_INVENTORY
    target.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(target.parent, 0o700)
    payload = yaml.safe_dump(
        inventory.to_dict(),
        default_flow_style=False,
        sort_keys=False,
    )
    target.write_text(payload, encoding="utf-8")
    os.chmod(target, 0o600)
    return target


def add_host(
    inventory: ClusterInventory,
    name: str,
    *,
    user: str | None = None,
    port: int = _SSH_PORT,
    group: str = "all",
    discovered_from: str = "manual",
    provisioned: bool = False,
) -> ClusterInventory:
    """Add or update a host in the inventory, returning a new inventory."""

    name = validate_ssh_target(name)
    if _NAME_RE.fullmatch(name) is None:
        raise ValueError(f"invalid inventory host name: {name!r}")
    if not (1 <= int(port) <= 65535):
        raise ValueError("inventory host port must be between 1 and 65535")
    resolved_user = user or default_ssh_user()

    existing = inventory.get(name)
    if existing is not None:
        hosts = [
            InventoryHost(
                name=host.name,
                user=resolved_user if host.name == name else host.user,
                port=int(port) if host.name == name else host.port,
                group=group if host.name == name else host.group,
                provisioned=(
                    provisioned if host.name == name else host.provisioned
                ),
                discovered_from=(
                    discovered_from if host.name == name else host.discovered_from
                ),
            )
            for host in inventory.hosts
        ]
        return ClusterInventory(hosts=hosts)

    return ClusterInventory(
        hosts=[
            *inventory.hosts,
            InventoryHost(
                name=name,
                user=resolved_user,
                port=int(port),
                group=group,
                provisioned=provisioned,
                discovered_from=discovered_from,
            ),
        ]
    )


def remove_host(inventory: ClusterInventory, name: str) -> bool:
    """Remove a host; returns False when the name is unknown."""

    if inventory.get(name) is None:
        return False
    inventory.hosts = [host for host in inventory.hosts if host.name != name]
    return True
