# SPDX-License-Identifier: Apache-2.0
"""Validation and normalization for Soft-REAP expert pin manifests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SoftReapManifest:
    """Original expert IDs that remain resident for every routed layer."""

    layers: dict[int, tuple[int, ...]]
    source: Path | None = None

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    @property
    def pinned_count_range(self) -> tuple[int, int]:
        counts = [len(experts) for experts in self.layers.values()]
        return (min(counts), max(counts)) if counts else (0, 0)

    def experts_for_layer(self, layer: int) -> tuple[int, ...]:
        try:
            return self.layers[layer]
        except KeyError as exc:
            raise ValueError(
                f"Soft-REAP manifest has no entry for layer {layer}"
            ) from exc

    @classmethod
    def empty(
        cls,
        num_layers: int | None = None,
        *,
        layer_ids: list[int] | tuple[int, ...] | None = None,
    ) -> SoftReapManifest:
        """Build a manifest-free residency map for cache-only streaming."""

        if layer_ids is None:
            if num_layers is None:
                raise ValueError("Empty manifest requires layer IDs or a layer count")
            layer_ids = tuple(range(num_layers))
        return cls(layers={int(layer): () for layer in layer_ids})


def _unwrap_layers(data: Any) -> Any:
    if not isinstance(data, Mapping):
        return data
    for key in ("layers", "pinned_experts", "kept_experts"):
        value = data.get(key)
        if value is not None:
            return value
    return data


def _normalize_layers(
    data: Any,
    *,
    num_layers: int | None,
    layer_ids: list[int] | tuple[int, ...] | None,
    num_experts: int | None,
) -> dict[int, tuple[int, ...]]:
    data = _unwrap_layers(data)
    if isinstance(data, list):
        if not data or not all(isinstance(value, int) for value in data):
            raise ValueError("Soft-REAP expert list must contain integer expert IDs")
        if num_layers is None and layer_ids is None:
            raise ValueError("A shared expert list requires the model layer count")
        targets = layer_ids if layer_ids is not None else range(int(num_layers))
        data = {str(layer): data for layer in targets}
    if not isinstance(data, Mapping) or not data:
        raise ValueError("Soft-REAP manifest must contain a non-empty layer mapping")

    normalized: dict[int, tuple[int, ...]] = {}
    for raw_layer, raw_experts in data.items():
        try:
            layer = int(raw_layer)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid Soft-REAP layer ID: {raw_layer!r}") from exc
        if layer < 0:
            raise ValueError(f"Soft-REAP layer ID must be non-negative: {layer}")
        if not isinstance(raw_experts, list) or not raw_experts:
            raise ValueError(f"Layer {layer} must contain a non-empty expert list")
        if not all(
            isinstance(expert, int) and not isinstance(expert, bool)
            for expert in raw_experts
        ):
            raise ValueError(f"Layer {layer} contains a non-integer expert ID")
        experts = tuple(sorted(set(raw_experts)))
        if len(experts) != len(raw_experts):
            raise ValueError(f"Layer {layer} contains duplicate expert IDs")
        if experts[0] < 0:
            raise ValueError(f"Layer {layer} contains a negative expert ID")
        if num_experts is not None and experts[-1] >= num_experts:
            raise ValueError(
                f"Layer {layer} expert {experts[-1]} exceeds model range 0-{num_experts - 1}"
            )
        normalized[layer] = experts

    if layer_ids is not None or num_layers is not None:
        expected = (
            {int(layer) for layer in layer_ids}
            if layer_ids is not None
            else set(range(int(num_layers)))
        )
        present = set(normalized)
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing layers {missing}")
            if extra:
                details.append(f"unexpected layers {extra}")
            raise ValueError("Soft-REAP manifest layer mismatch: " + ", ".join(details))
    return dict(sorted(normalized.items()))


def load_soft_reap_manifest(
    path: str | Path,
    *,
    num_layers: int | None = None,
    layer_ids: list[int] | tuple[int, ...] | None = None,
    num_experts: int | None = None,
) -> SoftReapManifest:
    """Load official REAP maps and the wrapped Soft-REAP manifest form."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError(f"Soft-REAP manifest does not exist: {manifest_path}")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read Soft-REAP manifest: {exc}") from exc
    layers = _normalize_layers(
        data,
        num_layers=num_layers,
        layer_ids=layer_ids,
        num_experts=num_experts,
    )
    return SoftReapManifest(layers=layers, source=manifest_path)


def validate_soft_reap_manifest_data(
    data: Any,
    *,
    num_layers: int | None = None,
    layer_ids: list[int] | tuple[int, ...] | None = None,
    num_experts: int | None = None,
) -> SoftReapManifest:
    """Validate uploaded JSON before it is persisted by the admin API."""

    return SoftReapManifest(
        layers=_normalize_layers(
            data,
            num_layers=num_layers,
            layer_ids=layer_ids,
            num_experts=num_experts,
        )
    )
