# SPDX-License-Identifier: Apache-2.0
"""The two read-only lookups the assistant shares, and the layout resolver.

``plan``, the free-space delete and the measurement all have to answer "which
Macs did we measure, where can each one be reached, and what shape is this
checkpoint". Those answers come from oMLX's own registry and discovery, and
they are injected here so tests never touch a real registry or the network and
so the rule is visible in one place.

The apply bridge — re-elect from the stored measurement, verify an approved
``plan_signature`` and build the coordinator's deployment request — belongs to
the run backend and is not part of this module yet. Nothing here stages,
launches or activates anything, and nothing here can name the app origin as a
fallback coordinator: there is no such branch.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..planner import ModelLayout, inspect_safetensors_layout, synthetic_model_layout
from .state import ModelIdentity


@dataclass(frozen=True)
class ProvisioningSeams:
    """The two read-only oMLX lookups this module performs.

    Both are injected so tests never touch a real registry or the network, and
    so the resolution rule is visible in one place.
    """

    #: node_id -> that Mac's paired registry record, if any.
    device_record: Callable[[str], Mapping[str, Any] | None]
    #: Addresses this Mac owns; used only to prove "the Mac is me".
    local_addresses: Callable[[], Sequence[str]]


def default_provisioning_seams() -> ProvisioningSeams:
    """The real registry lookup, imported lazily so this module stays cheap."""

    from ..discovery import local_addr_dicts
    from ..registry import get_device_registry

    def device_record(node_id: str) -> Mapping[str, Any] | None:
        try:
            registry = get_device_registry()
            # Only a *paired* Mac is a real target. Discovery also lists
            # unpaired Macs, and their record carries whatever addresses the
            # network happened to announce — including IPv6 link-local ones.
            row = registry.get(node_id)
            if row is None or not registry.is_paired(node_id):
                return None
            return row
        except (RuntimeError, ValueError):
            return None

    def local_addresses() -> Sequence[str]:
        try:
            # ``local_addr_dicts`` reports ``{"ip": …, "if_type": …}``.
            return [
                str(item.get("ip") or item.get("address") or "")
                for item in local_addr_dicts()
            ]
        except Exception:  # noqa: BLE001 - address discovery is best effort
            return ()

    return ProvisioningSeams(
        device_record=device_record,
        local_addresses=local_addresses,
    )


def model_layout_for(model: ModelIdentity) -> ModelLayout:
    """The planner layout for the checkpoint a plan is about.

    One implementation, used by ``measure`` and ``plan`` (and later by apply and
    start): a real checkpoint is inspected on this Mac, a pre-download size plus
    layer count is planned as a synthetic layout. Reading headers is not a probe
    and not a network call.
    """

    if model.model_path:
        return inspect_safetensors_layout(model.model_path)
    return synthetic_model_layout(
        total_weight_bytes=model.model_size_bytes or 0,
        layer_count=model.layer_count,
    )


__all__ = [
    "ProvisioningSeams",
    "default_provisioning_seams",
    "model_layout_for",
]