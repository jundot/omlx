# SPDX-License-Identifier: Apache-2.0
"""Automatically free disk before a model is copied to a Mac that is short.

Decision 2026-11-14 (user): the delete rule is enforced automatically when a
model must be loaded/distributed, instead of requiring the user to click the
delete affordance. The rule itself is unchanged and is the only thing this
module may select from:

* only a model that Mac itself listed in the measurement;
* never the checkpoint the cluster is about to run;
* never the last complete copy of a checkpoint across the measured Macs
  (losing the last copy is data loss);
* largest first, so the fewest deletions free the most space.

The blocking delete (over the same admin seam) and the retry live in the route;
this module is pure so the selection is testable without a Mac.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def model_key(path: str) -> str:
    """The checkpoint's stable name: the folder name, lower-cased.

    oMLX keys per-model settings by the folder name, so a case-only path
    difference (``Models`` vs ``models`` on a case-insensitive Mac) must not
    make two paths look like two models.
    """

    text = str(path or "").strip()
    return Path(text).name.lower() if text else ""


@dataclass(frozen=True)
class Deletable:
    node_id: str
    model_id: str
    model_path: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "model_id": self.model_id,
            "model_path": self.model_path,
            "size_bytes": self.size_bytes,
        }


def _planned_path(record: Any) -> str:
    model = getattr(record, "model", None)
    if isinstance(model, Mapping):
        return str(model.get("model_path") or "")
    return str(getattr(model, "model_path", "") or "")


def _inventories(record: Any) -> Mapping[str, Sequence[Any]]:
    inventories = getattr(record, "inventories", None)
    return inventories if isinstance(inventories, Mapping) else {}


def holders(record: Any) -> dict[str, set[str]]:
    """model key -> the measured Macs that listed a copy of it."""

    out: dict[str, set[str]] = {}
    for node_id, entries in _inventories(record).items():
        for item in entries or ():
            if not isinstance(item, Mapping):
                continue
            key = model_key(str(item.get("model_path") or ""))
            if key:
                out.setdefault(key, set()).add(str(node_id))
    return out


def deletable_models(
    record: Any,
    node_id: str,
    *,
    planned_model_path: str | None = None,
) -> list[Deletable]:
    """The models this Mac may free, largest first, per the frozen rule."""

    planned = model_key(
        planned_model_path if planned_model_path is not None else _planned_path(record)
    )
    hold = holders(record)
    deletable: list[Deletable] = []
    for item in _inventories(record).get(node_id, ()) or ():
        if not isinstance(item, Mapping):
            continue
        path = str(item.get("model_path") or "")
        key = model_key(path)
        model_id = str(item.get("model_id") or key)
        if not key or not model_id:
            continue
        if key == planned:
            continue  # never the checkpoint the cluster is about to run
        if hold.get(key, set()) <= {str(node_id)}:
            continue  # never the last complete copy on the measured Macs
        deletable.append(
            Deletable(
                node_id=str(node_id),
                model_id=model_id,
                model_path=path,
                size_bytes=int(item.get("size_bytes") or 0),
            )
        )
    deletable.sort(key=lambda candidate: candidate.size_bytes, reverse=True)
    return deletable


def select_for_space(
    candidates: Sequence[Deletable], needed_bytes: int
) -> list[Deletable]:
    """Largest first until the shortfall is covered (or candidates run out)."""

    selected: list[Deletable] = []
    freed = 0
    for candidate in candidates:
        selected.append(candidate)
        freed += candidate.size_bytes
        if freed >= needed_bytes:
            break
    return selected
