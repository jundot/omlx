# SPDX-License-Identifier: Apache-2.0
"""Persist the fine-tuned knobs per (model, node config) so Start can just use them.

Product decision 2026-11-14: tuning a large model is expensive (each config is a
model setting, so a real sweep redeploys per config; mtplx does the same). Tune
once per model + hardware pair, SAVE the winning chunk / MTP / MTP-depth /
sampler, and a later Start applies the saved preset instead of re-tuning.

The preset is keyed by ``tuning.preset_key(model_signature, hardware_key)`` — the
same content-signature + hardware-pair key the tuner already computes — so a
preset from one model on one Mac set is never reused for another. The file lives
at ``<base_path>/cluster/assistant-tuning-presets.json`` (schema 1, atomic,
newest keeps; a corrupt file fails closed to "no preset", never to a guess).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tuning import preset_key

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TuningPreset:
    """The winning knobs for one model on one hardware pair."""

    model_signature: str
    hardware_key: str
    chunk: int | None = None
    mtp: bool | None = None
    mtp_depth: int | None = None
    sampler: Mapping[str, Any] | None = None
    measured_at: str = ""

    @property
    def key(self) -> str:
        return preset_key(self.model_signature, self.hardware_key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "model_signature": self.model_signature,
            "hardware_key": self.hardware_key,
            "chunk": self.chunk,
            "mtp": self.mtp,
            "mtp_depth": self.mtp_depth,
            "sampler": dict(self.sampler) if self.sampler is not None else None,
            "measured_at": self.measured_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TuningPreset:
        sampler = payload.get("sampler")
        return cls(
            model_signature=str(payload.get("model_signature") or ""),
            hardware_key=str(payload.get("hardware_key") or ""),
            chunk=_optional_int(payload.get("chunk")),
            mtp=_optional_bool(payload.get("mtp")),
            mtp_depth=_optional_int(payload.get("mtp_depth")),
            sampler=dict(sampler) if isinstance(sampler, Mapping) else None,
            measured_at=str(payload.get("measured_at") or ""),
        )


class _Keep:
    """Sentinel: "this save did not measure the field, keep the stored value"."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "KEEP"


#: Pass as a field value to ``merge_preset`` to leave the stored value untouched.
KEEP = _Keep()


def merge_preset(
    existing: TuningPreset | None,
    *,
    model_signature: str,
    hardware_key: str,
    chunk: int | None | _Keep = KEEP,
    mtp: bool | None | _Keep = KEEP,
    mtp_depth: int | None | _Keep = KEEP,
    sampler: Mapping[str, Any] | None | _Keep = KEEP,
    measured_at: str = "",
) -> TuningPreset:
    """One combined preset: the newly measured knobs win, the rest are kept.

    Tuning is split across two routes and neither measures all of it: ``/tune``
    measures the sampler (and MTP on/off), ``/tune-redeploy`` measures the MTP
    switch and its adaptive depth. A save therefore merges instead of
    overwriting, so running ``/tune`` after a redeploy sweep must never erase the
    tuned ``mtp_depth`` (and a later sweep must not erase the sampler).

    ``KEEP`` (the default) means "not measured by this save". A real value —
    including ``False`` and ``None`` — is a measurement and wins; the redeploy
    sweep passes ``mtp_depth=None`` when the adaptive config won, because "no
    fixed depth" is then the measured answer. A ``None`` sampler (the tuner
    found no winner) also keeps the stored sampler.
    """

    def keep(value: Any, current: Any) -> Any:
        return current if value is KEEP else value

    base = existing
    return TuningPreset(
        model_signature=model_signature,
        hardware_key=hardware_key,
        chunk=keep(chunk, base.chunk if base is not None else None),
        mtp=keep(mtp, base.mtp if base is not None else None),
        mtp_depth=keep(mtp_depth, base.mtp_depth if base is not None else None),
        sampler=(
            dict(sampler)
            if sampler is not KEEP and isinstance(sampler, Mapping)
            else (
                dict(base.sampler)
                if base is not None and base.sampler is not None
                else None
            )
        ),
        measured_at=measured_at or (base.measured_at if base is not None else ""),
    )


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


class TuningPresetStore:
    """Atomic, newest-keeps JSON store of tuning presets, keyed by model+hardware."""

    def __init__(self, base_path: str | Path) -> None:
        self._path = Path(base_path) / "cluster" / "assistant-tuning-presets.json"
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> dict[str, TuningPreset]:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(raw, Mapping) or raw.get("schema_version") != SCHEMA_VERSION:
            return {}
        presets = raw.get("presets")
        if not isinstance(presets, Mapping):
            return {}
        result: dict[str, TuningPreset] = {}
        for key, payload in presets.items():
            if isinstance(key, str) and isinstance(payload, Mapping):
                try:
                    result[key] = TuningPreset.from_dict(payload)
                except (TypeError, ValueError):
                    continue
        return result

    def _write(self, presets: Mapping[str, TuningPreset]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "schema_version": SCHEMA_VERSION,
            "presets": {key: preset.to_dict() for key, preset in presets.items()},
        }
        fd, temporary = tempfile.mkstemp(
            dir=str(self._path.parent), prefix=".tuning-presets-", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(document, handle, indent=2, sort_keys=True)
            os.replace(temporary, self._path)
        finally:
            with suppress(OSError):
                os.unlink(temporary)

    def save(self, preset: TuningPreset) -> None:
        with self._lock:
            presets = self._load()
            presets[preset.key] = preset
            self._write(presets)

    def get(self, model_signature: str, hardware_key: str) -> TuningPreset | None:
        with self._lock:
            return self._load().get(preset_key(model_signature, hardware_key))

    def all(self) -> dict[str, TuningPreset]:
        with self._lock:
            return dict(self._load())
